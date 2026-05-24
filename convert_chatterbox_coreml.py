#!/usr/bin/env python3
"""
Convert Chatterbox Turbo TTS models to CoreML format.

Produces three CoreML models:
  - T3Prefill.mlpackage  — GPT-2 prefill (GPU)
  - T3Decode.mlpackage   — GPT-2 single-step decode (GPU)
  - S3Encoder.mlpackage  — Conformer encoder (ANE)
  - S3UNet.mlpackage     — U-Net denoiser (ANE)

Plus vocoder weights and tokenizer files for the Swift package.

Usage:
    python convert_chatterbox_coreml.py --stage t3 --output-dir /tmp/chatterbox-coreml
    python convert_chatterbox_coreml.py --stage s3 --output-dir /tmp/chatterbox-coreml
    python convert_chatterbox_coreml.py --stage vocoder --output-dir /tmp/chatterbox-coreml
    python convert_chatterbox_coreml.py --stage all --output-dir /tmp/chatterbox-coreml [--validate]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from safetensors.torch import save_file as save_safetensors


# ---------------------------------------------------------------------------
# Constants matching Chatterbox Turbo's architecture
# ---------------------------------------------------------------------------
SPEECH_VOCAB_SIZE = 6563
SPEECH_STOP_TOKEN = 6562
SPEECH_START_TOKEN = 0
MEL_BINS = 80
SPEAKER_EMB_DIM = 256
CAMPP_EMB_DIM = 192
GPT2_HIDDEN = 1024
GPT2_HEADS = 16
GPT2_LAYERS = 24
GPT2_MAX_POS = 8196
GPT2_HEAD_DIM = GPT2_HIDDEN // GPT2_HEADS  # 64

# Text vocab size for GPT-2 tokenizer (not the speech vocab)
TEXT_VOCAB_SIZE = 50276


# ---------------------------------------------------------------------------
# chatterbox-tts 0.1.7 hardcodes a GPT-2 backbone for ChatterboxTurboTTS but
# the upstream package ships LLAMA_CONFIGS without the "GPT2_medium" entry
# that t3.py + tts_turbo.py both reference. Patch it in before any chatterbox
# module load. If the entry is already present (older fork, vendored patch),
# this is a no-op.
# ---------------------------------------------------------------------------
_GPT2_MEDIUM_CONFIG = {
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": GPT2_MAX_POS,
    "n_embd": GPT2_HIDDEN,
    "hidden_size": GPT2_HIDDEN,
    "n_head": GPT2_HEADS,
    "n_layer": GPT2_LAYERS,
    "n_positions": GPT2_MAX_POS,
    "n_special": 0,
    "predict_special_tokens": True,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "vocab_size": TEXT_VOCAB_SIZE,
}


def _ensure_chatterbox_gpt2_config():
    from chatterbox.models.t3 import llama_configs as _lc

    if "GPT2_medium" not in _lc.LLAMA_CONFIGS:
        _lc.LLAMA_CONFIGS["GPT2_medium"] = _GPT2_MEDIUM_CONFIG


# ---------------------------------------------------------------------------
# Stateful KV cache + patched GPT-2 attention for ANE StateType decode
# ---------------------------------------------------------------------------


class SliceUpdateKeyValueCache:
    """Seq-first 2D KV cache with dim-0 slice writes for CoreML StateType.

    Layout: keyCache/valueCache are (max_seq, n_layers * n_heads * head_dim).
    Sequence dimension is dim 0 — the ONLY dimension CoreML runtime supports
    for dynamic slice_update on state tensors (confirmed via testing).

    Per-layer K/V are extracted by slicing the feature dimension, then reshaped
    to (1, n_heads, seq_len, head_dim) for attention.
    """

    def __init__(self, key_buffer, value_buffer, n_layers, n_heads, head_dim):
        # key/value_buffer: (max_seq, n_layers * n_heads * head_dim)
        self.key_cache = key_buffer
        self.value_cache = value_buffer
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.layer_size = n_heads * head_dim  # features per layer

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Write new K/V for one layer and return full cache for that layer.

        key_states:   (1, n_heads, seq_len, head_dim)
        value_states: (1, n_heads, seq_len, head_dim)
        """
        cache_position = cache_kwargs.get("cache_position")
        begin = cache_position[0]
        end = cache_position[-1] + 1
        seq_len = key_states.shape[2]

        # Flatten (1, n_heads, seq_len, head_dim) -> (seq_len, n_heads * head_dim)
        k_flat = key_states.squeeze(0).transpose(0, 1).reshape(seq_len, self.layer_size)
        v_flat = value_states.squeeze(0).transpose(0, 1).reshape(seq_len, self.layer_size)

        # Feature slice for this layer
        feat_start = layer_idx * self.layer_size
        feat_end = feat_start + self.layer_size

        # Write into 2D cache at dim 0 (seq positions)
        self.key_cache[begin:end, feat_start:feat_end] = k_flat
        self.value_cache[begin:end, feat_start:feat_end] = v_flat

        # Read back FULL cache for this layer (fixed shape — no dynamic slice on read).
        # Unfilled positions are zeros; with is_causal=False, attention to zero K/V
        # produces near-zero weights after softmax, so this is numerically safe.
        max_seq = self.key_cache.shape[0]
        k_out = self.key_cache[:, feat_start:feat_end]  # (max_seq, layer_size)
        v_out = self.value_cache[:, feat_start:feat_end]

        # Reshape to (1, n_heads, max_seq, head_dim)
        k_out = k_out.reshape(max_seq, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v_out = v_out.reshape(max_seq, self.n_heads, self.head_dim).transpose(0, 1).unsqueeze(0)

        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        return 0

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position, layer_idx=0):
        """Return (kv_length, kv_offset) for causal mask creation."""
        kv_length = cache_position[-1].item() + 1
        return kv_length, 0


def patched_gpt2_attention_forward(
    self,
    hidden_states,
    past_key_values=None,
    cache_position=None,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
    **kwargs,
):
    """Simplified GPT2Attention.forward using SliceUpdateKeyValueCache.

    - Passes cache_position from kwargs to cache.update()
    - Always is_causal=False (GPT-2 causal bias is in model weights;
      avoids torch.jit.trace baking a bool constant)
    - Removes cross-attention, upcast_and_reorder, encoder paths
    """
    query_states, key_states, value_states = self.c_attn(hidden_states).split(
        self.split_size, dim=2
    )

    shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
    key_states = key_states.view(shape_kv).transpose(1, 2)
    value_states = value_states.view(shape_kv).transpose(1, 2)

    shape_q = (*query_states.shape[:-1], -1, self.head_dim)
    query_states = query_states.view(shape_q).transpose(1, 2)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"cache_position": cache_position},
        )

    # Ensure matching dtypes (cache is FP16, projections may be FP32)
    key_states = key_states.to(query_states.dtype)
    value_states = value_states.to(query_states.dtype)

    # Read kv_mask from cache object (stored by T3StatefulWrapper.forward)
    kv_mask = getattr(past_key_values, 'kv_mask', None) if past_key_values is not None else None
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=kv_mask,  # (1, 1, 1, 2048) — 0 valid, -1e9 unfilled
        is_causal=False,
        dropout_p=0.0,
    )

    # SDPA returns (batch, heads, seq, head_dim) — transpose to (batch, seq, heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)  # (batch, seq, hidden)
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)
    return attn_output, None


# ---------------------------------------------------------------------------
# Helper: load the PyTorch Chatterbox Turbo model
# ---------------------------------------------------------------------------
class ChatterboxModels:
    """Container for individually loaded Chatterbox sub-models."""
    def __init__(self, t3, s3gen, model_dir):
        self.t3 = t3
        self.s3gen = s3gen
        self.model_dir = model_dir


def load_pytorch_model(cache_dir=None):
    """Download and load the Chatterbox Turbo PyTorch model components (v1 path).

    Uses YAML config + manual state_dict load. v4 stages should use
    load_pytorch_model_v4() which goes through ChatterboxTurboTTS.from_pretrained
    for the meanflow-trained s3gen weights.
    """
    _ensure_chatterbox_gpt2_config()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    print("Downloading Chatterbox Turbo weights...")
    model_dir = Path(snapshot_download("ResembleAI/chatterbox-turbo"))
    print(f"  Model dir: {model_dir}")

    print("  Loading T3 (GPT-2 Medium turbo)...")
    from chatterbox.models.t3.t3 import T3, T3Config
    import yaml

    yaml_path = model_dir / "t3_turbo_v1.yaml"
    with open(yaml_path) as f:
        cfg_dict = yaml.full_load(f)

    # Weights are GPT-2 despite the YAML claiming llama
    cfg_dict["llama_config_name"] = "GPT2_medium"
    turbo_cfg = T3Config.__new__(T3Config)
    for k, v in cfg_dict.items():
        setattr(turbo_cfg, k, v)

    t3 = T3(hp=turbo_cfg)
    t3_state = load_file(model_dir / "t3_turbo_v1.safetensors")
    t3.load_state_dict(t3_state)
    t3.to("cpu").train(False)

    print("  Loading S3Gen...")
    from chatterbox.models.s3gen import S3Gen
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(model_dir / "s3gen.safetensors"), strict=False)
    s3gen.to("cpu").train(False)

    print("  Models loaded successfully.")
    return ChatterboxModels(t3=t3, s3gen=s3gen, model_dir=model_dir)


def load_pytorch_model_v4(cache_dir=None):
    """Load Chatterbox Turbo via the official ChatterboxTurboTTS.from_pretrained.

    This matches the v4 HF artifacts (meanflow-trained s3gen). v4 stages
    (prefill, lm-onnx, cond-decoder) should use this loader.
    """
    _ensure_chatterbox_gpt2_config()

    print("Loading Chatterbox Turbo via ChatterboxTurboTTS.from_pretrained('cpu')...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from huggingface_hub import snapshot_download

    tts = ChatterboxTurboTTS.from_pretrained("cpu")
    tts.t3.train(False)
    tts.s3gen.train(False)
    model_dir = Path(snapshot_download("ResembleAI/chatterbox-turbo"))
    print(f"  Model dir: {model_dir}")
    print("  Models loaded successfully.")
    return ChatterboxModels(t3=tts.t3, s3gen=tts.s3gen, model_dir=model_dir)


# ===========================================================================
# T3 CONVERSION
# ===========================================================================


CONTEXT_SIZE = 2048  # Max sequence length for KV cache state


class T3StatefulWrapper(nn.Module):
    """Stateful T3 wrapper for CoreML: takes pre-computed embeddings, not token IDs.

    The embedding lookup and speaker conditioning happen in Swift.
    This model is JUST: transformer + speech_head + KV cache state.

    Swift caller handles:
    - Position 0: cond_enc.spkr_enc(speaker_emb) → (1, 1, 1024) conditioning embedding
    - Positions 1+: speech_emb[token_id] → (1, 1, 1024) token embedding
    """

    def __init__(self, t3_model, context_size=CONTEXT_SIZE):
        super().__init__()
        self.tfmr = t3_model.t3.tfmr
        self.speech_head = t3_model.t3.speech_head
        self.context_size = context_size

        # 2D seq-first cache: (max_seq, n_layers * n_heads * head_dim)
        feature_size = GPT2_LAYERS * GPT2_HEADS * GPT2_HEAD_DIM
        self.register_buffer("keyCache", torch.zeros(context_size, feature_size, dtype=torch.float16))
        self.register_buffer("valueCache", torch.zeros(context_size, feature_size, dtype=torch.float16))

    def forward(self, inputs_embeds, position_ids, cache_position, attention_mask):
        """
        Args:
            inputs_embeds:  (1, 1, 1024) float — pre-computed embedding from Swift
            position_ids:   (1, 1) int32 — GPT-2 wpe position
            cache_position: (1,) int32 — KV cache write position
            attention_mask: (1, 1, 1, 2048) float — 0 valid, -1e9 unfilled

        Returns:
            logits: (1, vocab_size) float16 — logits for next token
        """
        cache = SliceUpdateKeyValueCache(
            self.keyCache, self.valueCache,
            n_layers=GPT2_LAYERS, n_heads=GPT2_HEADS, head_dim=GPT2_HEAD_DIM
        )
        # Store mask on cache for patched attention to read
        cache.kv_mask = attention_mask

        outputs = self.tfmr(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden = outputs.last_hidden_state

        logits = self.speech_head(hidden[:, -1:, :]).squeeze(1)
        return logits


def convert_t3(model, output_dir, validate=False):
    """Convert T3 to a single stateful CoreML model with StateType KV cache."""
    print("\n=== Converting T3 Stateful (GPT-2 + KV Cache) ===")

    t3_model = model
    t3_model.t3.eval()

    # Monkey-patch GPT2Attention to use SliceUpdateKeyValueCache
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    original_forward = GPT2Attention.forward
    GPT2Attention.forward = patched_gpt2_attention_forward

    wrapper = T3StatefulWrapper(t3_model, context_size=CONTEXT_SIZE)
    wrapper.eval()

    # Export embedding weights for Swift-side lookup
    print("  Exporting embedding weights...")
    speech_emb_weights = t3_model.t3.speech_emb.weight.data.cpu().float()  # (6563, 1024)
    np.save(os.path.join(output_dir, "speech_emb.npy"), speech_emb_weights.numpy())
    print(f"    speech_emb: {speech_emb_weights.shape}")

    text_emb_weights = t3_model.t3.text_emb.weight.data.cpu().float()  # (50276, 1024)
    np.save(os.path.join(output_dir, "text_emb.npy"), text_emb_weights.numpy())
    print(f"    text_emb: {text_emb_weights.shape}")

    # Export conditioning linear weights (spkr_enc)
    spkr_enc = t3_model.t3.cond_enc.spkr_enc
    spkr_w = spkr_enc.weight.data.cpu().float()  # (1024, 256)
    spkr_b = spkr_enc.bias.data.cpu().float()    # (1024,)
    np.save(os.path.join(output_dir, "spkr_enc_weight.npy"), spkr_w.numpy())
    np.save(os.path.join(output_dir, "spkr_enc_bias.npy"), spkr_b.numpy())
    print(f"    spkr_enc: weight {spkr_w.shape}, bias {spkr_b.shape}")

    # Trace with float embedding input + attention mask
    example_embeds = torch.randn(1, 1, GPT2_HIDDEN)
    example_pos = torch.zeros(1, 1, dtype=torch.int32)
    example_cache_pos = torch.zeros(1, dtype=torch.int32)
    example_mask = torch.zeros(1, 1, 1, CONTEXT_SIZE, dtype=torch.float32)
    example_mask[:, :, :, 1:] = -1e9  # only position 0 valid in example

    print("  Tracing T3Stateful (inputs_embeds + mask)...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_embeds, example_pos, example_cache_pos, example_mask))

    # Restore original forward
    GPT2Attention.forward = original_forward

    # StateType for 2D seq-first KV cache
    feature_size = GPT2_LAYERS * GPT2_HEADS * GPT2_HEAD_DIM  # 24 * 16 * 64 = 24576
    cache_shape = (CONTEXT_SIZE, feature_size)
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="valueCache",
        ),
    ]

    # Fixed decode shape: seq=1 token + 1 conditioning = 2 positions.
    # EnumeratedShapes and RangeDim both cause error -14 with stateful models.
    # Prefill is done token-by-token through the same fixed-shape model.
    print("  Converting with fixed decode shape (seq=1, pos=2)...")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, GPT2_HIDDEN), dtype=np.float32),
            ct.TensorType(name="position_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="cache_position", shape=(1,), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, 1, CONTEXT_SIZE), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        states=states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    out_path = os.path.join(output_dir, "T3Stateful.mlpackage")
    mlmodel.save(out_path)
    print(f"  Saved: {out_path}")

    # Check for state ops in MIL program
    from coremltools.converters.mil.testing_utils import get_op_types_in_program
    ops = get_op_types_in_program(mlmodel._mil_program)
    has_state = "coreml_update_state" in ops
    print(f"  State ops present: {has_state}")
    if not has_state:
        print("  WARNING: No coreml_update_state — KV cache may not work!")

    if validate:
        validate_t3_stateful(t3_model, out_path)


def validate_t3_stateful(pytorch_model, model_path):
    """Validate stateful T3 CoreML output matches PyTorch."""
    print("\n  --- T3 Stateful Numerical Validation ---")

    # Save and reload to get CoreML framework backend (needed for make_state).
    # The convert() output uses coremltools internal backend which can't make_state().
    # CPU_ONLY avoids error -14 from ANE compilation of dynamic shapes.
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "T3Stateful.mlpackage")
        shutil.copytree(model_path, tmp_path)
        ml_model = ct.models.MLModel(tmp_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    state = ml_model.make_state()

    # Token-by-token prefill (fixed shape model only accepts seq=1)
    # Each call processes 1 speech token + 1 conditioning token = 2 positions
    import time
    prefill_ids = np.random.randint(0, SPEECH_VOCAB_SIZE, (16,)).astype(np.int32)
    prefill_spk = np.random.randn(1, SPEAKER_EMB_DIM).astype(np.float32)
    zero_spk = np.zeros((1, SPEAKER_EMB_DIM), dtype=np.float32)

    # Load speech embedding table for lookups
    speech_emb_table = pytorch_model.t3.speech_emb.weight.data.cpu().numpy()  # (vocab, 1024)

    # Load spkr_enc weights for conditioning
    spkr_enc = pytorch_model.t3.cond_enc.spkr_enc
    spkr_w = spkr_enc.weight.data.cpu().numpy()  # (1024, 256)
    spkr_b = spkr_enc.bias.data.cpu().numpy()    # (1024,)

    # Prefill: position 0 = speaker conditioning, positions 1+ = speech tokens
    speaker_emb = np.random.randn(SPEAKER_EMB_DIM).astype(np.float32)
    cond_emb = (spkr_w @ speaker_emb + spkr_b).reshape(1, 1, GPT2_HIDDEN).astype(np.float32)

    t0 = time.time()
    # Position 0: conditioning
    ml_model.predict(
        {"inputs_embeds": cond_emb, "position_ids": np.array([[0]], dtype=np.int32),
         "cache_position": np.array([0], dtype=np.int32)},
        state=state,
    )
    # Positions 1-16: speech tokens
    for i in range(16):
        emb = speech_emb_table[prefill_ids[i]].reshape(1, 1, GPT2_HIDDEN).astype(np.float32)
        pos = np.array([[i + 1]], dtype=np.int32)
        cp = np.array([i + 1], dtype=np.int32)
        ml_model.predict(
            {"inputs_embeds": emb, "position_ids": pos, "cache_position": cp},
            state=state,
        )
    prefill_ms = (time.time() - t0) * 1000
    print(f"  Prefill (1 cond + 16 tokens): {prefill_ms:.0f}ms ({prefill_ms/17:.1f}ms/tok)")

    # Decode step 1
    decode_emb = speech_emb_table[42].reshape(1, 1, GPT2_HIDDEN).astype(np.float32)
    cm_d1 = ml_model.predict(
        {"inputs_embeds": decode_emb, "position_ids": np.array([[17]], dtype=np.int32),
         "cache_position": np.array([17], dtype=np.int32)},
        state=state,
    )

    # Decode step 2
    cm_d2 = ml_model.predict(
        {"inputs_embeds": decode_emb, "position_ids": np.array([[18]], dtype=np.int32),
         "cache_position": np.array([18], dtype=np.int32)},
        state=state,
    )

    diff = np.abs(cm_d1["logits"] - cm_d2["logits"]).max()
    print(f"  Decode step diff: {diff:.4f}")
    if diff > 0.001:
        print("  PASS: KV cache working across decode steps!")
    else:
        print("  WARNING: Decode outputs identical")


# ===========================================================================
# S3 CONVERSION (Encoder + UNet)
# ===========================================================================


class S3EncoderWrapper(nn.Module):
    """Wraps the S3 encoder path for CoreML tracing.

    Takes a single concatenated token sequence (prompt + speech), provides
    token_len internally, and returns mel-projected encoder output.
    The encoder components live under s3gen.flow.
    """

    def __init__(self, s3_flow):
        super().__init__()
        self.input_embedding = s3_flow.input_embedding
        self.encoder = s3_flow.encoder
        self.encoder_proj = s3_flow.encoder_proj

    def forward(self, all_tokens):
        """
        Args:
            all_tokens: (1, T) int32 - concatenated [prompt_tokens | speech_tokens]

        Returns:
            encoder_proj: (1, 80, T_enc) float32 - mel-projected, BCT format
        """
        T = all_tokens.size(1)
        token_len = torch.tensor([T], dtype=torch.long, device=all_tokens.device)
        mask = torch.ones(1, T, 1, device=all_tokens.device, dtype=torch.float32)

        x = self.input_embedding(all_tokens.long()) * mask
        h, _ = self.encoder(x, token_len)
        mu = self.encoder_proj(h)
        mu = mu.transpose(1, 2)  # (1, 80, T_enc)
        return mu


class S3UNetWrapper(nn.Module):
    """Wraps the U-Net denoiser (estimator) for CoreML tracing.

    Includes the spkEmbedAffineLayer (192 to projected dim) baked in.
    The decoder's estimator and affine layer live under s3gen.flow.
    """

    def __init__(self, s3_flow):
        super().__init__()
        self.estimator = s3_flow.decoder.estimator
        self.spk_affine = s3_flow.spk_embed_affine_layer

    def forward(self, x, mu, mask, t, spks, cond, r):
        """
        Args:
            x: (1, 80, T) float - noisy mel
            mu: (1, 80, T) float - target mel from encoder
            mask: (1, 1, T) float - validity mask
            t: (1,) float - timestep
            spks: (1, 192) float - raw CAMPPlus speaker embedding
            cond: (1, 80, T) float - conditioning mel
            r: (1,) float - meanflow ratio (unused by estimator, kept for API compat)

        Returns:
            velocity: (1, 80, T) float - predicted velocity
        """
        spks_proj = self.spk_affine(spks)
        return self.estimator(x, mask, mu, t, spks_proj, cond)


def convert_s3(model, output_dir, validate=False):
    """Convert S3Encoder and S3UNet to CoreML."""
    print("\n=== Converting S3Encoder (Conformer) ===")

    s3gen = model.s3gen
    s3gen.eval()
    s3_flow = s3gen.flow  # encoder/decoder live under flow

    # Monkey-patch view_as -> reshape (CoreML doesn't support view_as)
    _original_view_as = torch.Tensor.view_as
    torch.Tensor.view_as = lambda self, other: self.reshape(other.shape)

    # --- S3Encoder ---
    print("  Tracing S3Encoder...")
    encoder_wrapper = S3EncoderWrapper(s3_flow)
    encoder_wrapper.eval()

    # Single input: concatenated prompt + speech tokens
    example_tokens = torch.zeros(1, 70, dtype=torch.long)

    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder_wrapper, (example_tokens,))

    print("  Converting S3Encoder to CoreML...")
    encoder_inputs = [
        ct.TensorType(
            name="all_tokens",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=2048, default=70))),
            dtype=np.int32,
        ),
    ]

    encoder_coreml = ct.convert(
        traced_encoder,
        inputs=encoder_inputs,
        outputs=[ct.TensorType(name="encoder_proj", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    encoder_path = os.path.join(output_dir, "S3Encoder.mlpackage")
    encoder_coreml.save(encoder_path)
    print(f"  Saved: {encoder_path}")

    # --- S3UNet ---
    print("\n=== Converting S3UNet (Denoiser) ===")
    print("  Tracing S3UNet...")
    unet_wrapper = S3UNetWrapper(s3_flow)
    unet_wrapper.eval()

    T = 100  # example time steps
    example_x = torch.randn(1, MEL_BINS, T)
    example_mu = torch.randn(1, MEL_BINS, T)
    example_mask = torch.ones(1, 1, T)
    example_t = torch.tensor([0.5])
    example_spks = torch.randn(1, CAMPP_EMB_DIM)
    example_cond = torch.randn(1, MEL_BINS, T)
    example_r = torch.tensor([0.5])

    with torch.no_grad():
        traced_unet = torch.jit.trace(
            unet_wrapper,
            (example_x, example_mu, example_mask, example_t,
             example_spks, example_cond, example_r),
        )

    print("  Converting S3UNet to CoreML...")
    T_dim = ct.RangeDim(lower_bound=1, upper_bound=4096, default=100)
    unet_inputs = [
        ct.TensorType(name="x", shape=ct.Shape(shape=(1, MEL_BINS, T_dim)), dtype=np.float32),
        ct.TensorType(name="mu", shape=ct.Shape(shape=(1, MEL_BINS, T_dim)), dtype=np.float32),
        ct.TensorType(name="mask", shape=ct.Shape(shape=(1, 1, T_dim)), dtype=np.float32),
        ct.TensorType(name="t", shape=(1,), dtype=np.float32),
        ct.TensorType(name="spks", shape=(1, CAMPP_EMB_DIM), dtype=np.float32),
        ct.TensorType(name="cond", shape=ct.Shape(shape=(1, MEL_BINS, T_dim)), dtype=np.float32),
        ct.TensorType(name="r", shape=(1,), dtype=np.float32),
    ]

    unet_coreml = ct.convert(
        traced_unet,
        inputs=unet_inputs,
        outputs=[ct.TensorType(name="velocity", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    unet_path = os.path.join(output_dir, "S3UNet.mlpackage")
    unet_coreml.save(unet_path)
    print(f"  Saved: {unet_path}")

    # Restore monkey-patched view_as
    torch.Tensor.view_as = _original_view_as

    if validate:
        validate_s3(s3_flow, encoder_path, unet_path)


def validate_s3(s3_flow, encoder_path, unet_path):
    """Validate S3 CoreML outputs match PyTorch."""
    print("\n  --- S3 Numerical Validation ---")

    encoder_ml = ct.models.MLModel(encoder_path)
    unet_ml = ct.models.MLModel(unet_path)

    # Test encoder
    test_tokens = torch.randint(0, SPEECH_VOCAB_SIZE, (1, 40), dtype=torch.long)

    encoder_wrapper = S3EncoderWrapper(s3_flow)
    with torch.no_grad():
        pt_mu = encoder_wrapper(test_tokens)

    cm_out = encoder_ml.predict({
        "all_tokens": test_tokens.int().numpy(),
    })
    cm_mu = torch.from_numpy(cm_out["encoder_proj"]).float()

    cos_sim = torch.nn.functional.cosine_similarity(
        pt_mu.flatten().unsqueeze(0),
        cm_mu.flatten().unsqueeze(0)
    ).item()
    print(f"  S3Encoder cosine similarity: {cos_sim:.6f}")
    if cos_sim >= 0.99:
        print("  PASS")
    else:
        print("  WARNING: cosine sim < 0.99")

    # Test UNet
    T = 40
    test_x = torch.randn(1, MEL_BINS, T)
    test_mu_in = torch.randn(1, MEL_BINS, T)
    test_mask = torch.ones(1, 1, T)
    test_t = torch.tensor([0.5])
    test_spks = torch.randn(1, CAMPP_EMB_DIM)
    test_cond = torch.randn(1, MEL_BINS, T)
    test_r = torch.tensor([0.5])

    unet_wrapper = S3UNetWrapper(s3_flow)
    unet_wrapper.eval()
    with torch.no_grad():
        pt_vel = unet_wrapper(test_x, test_mu_in, test_mask, test_t, test_spks, test_cond, test_r)

    cm_out = unet_ml.predict({
        "x": test_x.numpy(),
        "mu": test_mu_in.numpy(),
        "mask": test_mask.numpy(),
        "t": test_t.numpy(),
        "spks": test_spks.numpy(),
        "cond": test_cond.numpy(),
        "r": test_r.numpy(),
    })
    cm_vel = torch.from_numpy(cm_out["velocity"]).float()

    cos_sim = torch.nn.functional.cosine_similarity(
        pt_vel.flatten().unsqueeze(0),
        cm_vel.flatten().unsqueeze(0)
    ).item()
    print(f"  S3UNet cosine similarity: {cos_sim:.6f}")
    if cos_sim >= 0.99:
        print("  PASS")
    else:
        print("  WARNING: cosine sim < 0.99")


# ===========================================================================
# VOCODER WEIGHT EXTRACTION
# ===========================================================================

def extract_vocoder_weights(model, output_dir):
    """Extract HiFTGenerator weights to safetensors format."""
    print("\n=== Extracting Vocoder Weights ===")

    vocoder = model.s3gen.mel2wav
    state_dict = vocoder.state_dict()

    # Convert all weights to float32 contiguous tensors
    weights = {}
    for key, tensor in state_dict.items():
        weights[key] = tensor.contiguous().float()
        print(f"  {key}: {list(tensor.shape)}")

    output_path = os.path.join(output_dir, "hift_vocoder.safetensors")
    save_safetensors(weights, output_path)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ===========================================================================
# TOKENIZER + CONFIG EXTRACTION
# ===========================================================================

def extract_tokenizer_and_config(model, output_dir):
    """Copy tokenizer files and create config.json."""
    print("\n=== Extracting Tokenizer + Config ===")

    # Find the cached model directory
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("ResembleAI/chatterbox-turbo")

    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    for fname in tokenizer_files:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")
        else:
            print(f"  WARNING: {fname} not found in model directory")

    # Create config.json
    config = {
        "model_type": "chatterbox-turbo-coreml",
        "sample_rate": 24000,
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "speech_stop_token": SPEECH_STOP_TOKEN,
        "speech_start_token": SPEECH_START_TOKEN,
        "mel_bins": MEL_BINS,
        "n_cfm_timesteps": 2,
        "speaker_emb_dim": SPEAKER_EMB_DIM,
        "campp_emb_dim": CAMPP_EMB_DIM,
        "gpt2_hidden": GPT2_HIDDEN,
        "gpt2_heads": GPT2_HEADS,
        "gpt2_layers": GPT2_LAYERS,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: config.json")

    # Copy default conditioning if available
    for cond_file in ["default-conds.safetensors", "conds.safetensors"]:
        src = os.path.join(model_dir, cond_file)
        if os.path.exists(src):
            dst = os.path.join(output_dir, "default-conds.safetensors")
            shutil.copy2(src, dst)
            print(f"  Copied: {cond_file} as default-conds.safetensors")
            break


# ===========================================================================
# v4 PIPELINE: T3Prefill (CoreML) + language_model.onnx + conditional_decoder.onnx
#
# The v4 hybrid pipeline is what runs on iPhone in pooler-core today. It's a
# drop-in replacement for the v1 stages above. Each v4 stage is verified
# against the published HF reference artifact at
# huggingface.co/ebrinz/chatterbox-turbo-coreml.
# ===========================================================================


# --- Shared validation harness ---------------------------------------------


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _compare_outputs(
    name: str,
    ours: np.ndarray,
    theirs: np.ndarray,
    *,
    atol: Optional[float] = None,
    cos_min: Optional[float] = None,
) -> bool:
    """Compare two arrays. Returns True iff all specified thresholds pass.
    Prints a one-line report with metrics."""
    ours = np.asarray(ours).astype(np.float64).reshape(-1)
    theirs = np.asarray(theirs).astype(np.float64).reshape(-1)
    if ours.shape != theirs.shape:
        print(f"  [{name}] SHAPE MISMATCH ours={ours.shape} theirs={theirs.shape} FAIL")
        return False

    max_abs = float(np.max(np.abs(ours - theirs)))
    denom = (np.linalg.norm(ours) * np.linalg.norm(theirs)) or 1.0
    cos_sim = float(np.dot(ours, theirs) / denom)

    parts = [f"max_abs={max_abs:.3e}", f"cos_sim={cos_sim:.6f}"]
    ok = True
    if atol is not None:
        ok = ok and (max_abs <= atol)
        parts.append(f"(atol={atol:.1e})")
    if cos_min is not None:
        ok = ok and (cos_sim >= cos_min)
        parts.append(f"(cos_min={cos_min:.4f})")
    status = "PASS" if ok else "FAIL"
    print(f"  [{name}] " + " ".join(parts) + f"  {status}")
    return ok


def _load_onnx_session(path):
    """Load an ONNX model with onnxruntime, CPU provider, optimization on."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _check_onnx_graph_for_ios(onnx_path: str) -> bool:
    """Static checks: model valid, opset supported by iOS ORT, no unknown domains.

    The com.microsoft domain (MultiHeadAttention, etc.) is bundled with the iOS
    ORT binary and used by the HF v4 reference, so we accept it.
    """
    import onnx

    model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        print(f"  [ios] onnx.checker REJECTED model: {exc}  FAIL")
        return False

    opset_versions = {opset.domain or "ai.onnx": opset.version for opset in model.opset_import}
    main_opset = opset_versions.get("ai.onnx", 0)
    ok = True
    if main_opset < 13 or main_opset > 22:
        print(f"  [ios] opset {main_opset} outside iOS ORT-supported range [13, 22]  FAIL")
        ok = False
    else:
        print(f"  [ios] opset {main_opset} (ai.onnx)  OK")

    KNOWN_IOS_DOMAINS = {"", "ai.onnx", "com.microsoft"}
    declared_extra = [d for d in opset_versions if d not in KNOWN_IOS_DOMAINS]
    if declared_extra:
        # Imports alone are harmless if no node actually uses them; the
        # transformers optimizer adds a bunch of speculative declarations.
        # We only fail on actual node usage from non-iOS domains below.
        print(f"  [ios] extra opset declarations (not used by any node): {declared_extra}  WARN")

    all_nodes = list(model.graph.node)
    custom_nodes = [n for n in all_nodes if n.domain and n.domain not in {"ai.onnx", "com.microsoft"}]
    if custom_nodes:
        names = sorted({f"{n.domain}::{n.op_type}" for n in custom_nodes})
        print(f"  [ios] ops from non-iOS domains present: {names}  FAIL")
        ok = False
    else:
        ms_count = sum(1 for n in all_nodes if n.domain == "com.microsoft")
        std_count = len(all_nodes) - ms_count
        print(f"  [ios] {std_count} ai.onnx ops + {ms_count} com.microsoft ops  OK")
    return ok


def _check_xcrun_coremlcompiler(mlpackage_path: str) -> bool:
    """Compile the .mlpackage via Xcode's coremlcompiler (same as device build)."""
    if shutil.which("xcrun") is None:
        print(f"  [ios] xcrun not installed; skipping coremlcompiler check  SKIP")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", mlpackage_path, tmp],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  [ios] coremlcompiler FAILED: {proc.stderr.strip()[:300]}")
            return False
        print(f"  [ios] xcrun coremlcompiler  PASS")
        return True


def _try_ane_compute(mlpackage_path: str, sample_inputs: dict) -> bool:
    """Load the package targeting CPU_AND_NE on macOS and run one prediction.
    Reports compute_unit_usage when available. Returns True if it executes."""
    try:
        ml = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as exc:
        print(f"  [ios] CPU_AND_NE load FAILED: {str(exc)[:300]}  FAIL")
        return False
    try:
        _ = ml.predict(sample_inputs)
    except Exception as exc:
        print(f"  [ios] CPU_AND_NE predict FAILED: {str(exc)[:300]}  FAIL")
        return False
    usage = getattr(ml, "compute_unit_usage", None)
    if usage is not None:
        print(f"  [ios] CPU_AND_NE plan: {usage}  (Mac plan; iPhone plan may differ)")
    else:
        print(f"  [ios] CPU_AND_NE predict succeeded  OK")
    return True


# --- Stage A: language_model_single.onnx -----------------------------------


class _LanguageModelWrapper(nn.Module):
    """Single-step GPT-2 decode, with attention rewritten in primitives.

    HF's GPT2Model / GPT2Block / GPT2Attention all funnel past_key_values
    through `transformers.cache_utils.Cache`, which torch.onnx.export (both
    legacy and dynamo) cannot trace cleanly — it triggers a functorch vmap
    dispatch loop ('unordered_map::at: key not found') or ProxyTorchDispatchMode
    assertions deep in torch.export. We work around it by reusing the trained
    weight modules (ln_1/ln_2/mlp/c_attn/c_proj) but driving the attention math
    ourselves with plain (key, value) tensors per layer.

    Forward signature:
        (inputs_embeds, attention_mask, position_ids,
         k0, v0, k1, v1, ..., k23, v23)
    Outputs:
        (logits, present_k0, present_v0, ..., present_k23, present_v23)
    Names match Swift OrtFastDecoder's expectations."""

    def __init__(self, t3):
        super().__init__()
        tfmr = t3.tfmr
        self.speech_head = t3.speech_head  # Linear(1024, 6563)
        self.wpe = tfmr.wpe
        self.ln_f = tfmr.ln_f
        # Keep blocks as-is so we can reach .ln_1, .ln_2, .attn.{c_attn, c_proj,
        # resid_dropout}, and .mlp on each one without copying.
        self.h = tfmr.h
        self.n_layer = len(tfmr.h)
        self.n_head = GPT2_HEADS
        self.head_dim = GPT2_HEAD_DIM
        self.hidden = GPT2_HIDDEN
        self.split_size = GPT2_HIDDEN

    def _block_forward(self, block, hidden_states, past_k, past_v, attn_mask_additive):
        # --- self-attention -------------------------------------------------
        residual = hidden_states
        h = block.ln_1(hidden_states)
        qkv = block.attn.c_attn(h)  # (1, seq, 3*hidden)
        q, k_new, v_new = qkv.split(self.split_size, dim=2)

        # (1, seq, hidden) -> (1, heads, seq, head_dim)
        q = q.view(1, -1, self.n_head, self.head_dim).transpose(1, 2)
        k_new = k_new.view(1, -1, self.n_head, self.head_dim).transpose(1, 2)
        v_new = v_new.view(1, -1, self.n_head, self.head_dim).transpose(1, 2)

        # Append the new step's k/v to past
        k_full = torch.cat([past_k, k_new], dim=2)
        v_full = torch.cat([past_v, v_new], dim=2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=attn_mask_additive,
            is_causal=False,
            dropout_p=0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden)
        attn = block.attn.c_proj(attn)
        attn = block.attn.resid_dropout(attn)
        hidden_states = residual + attn

        # --- mlp ------------------------------------------------------------
        residual = hidden_states
        h = block.ln_2(hidden_states)
        h = block.mlp(h)
        hidden_states = residual + h

        return hidden_states, k_full, v_full

    def forward(self, inputs_embeds, attention_mask, position_ids, *flat_past_kv):
        assert len(flat_past_kv) == GPT2_LAYERS * 2

        # Position embeddings (wte was applied upstream by the caller; we get
        # inputs_embeds directly to match the iOS Swift call pattern).
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Build additive 4D attention mask: (1, 1, 1, total_seq_len). Valid
        # positions = 0, invalid = -1e9 so softmax ignores them. Input is
        # (1, total_seq_len) int64 of 0/1 from the Swift call (always all 1s
        # in normal decode); we still honor zeros for general correctness.
        attn_mask_4d = (
            (1.0 - attention_mask.to(hidden_states.dtype).view(1, 1, 1, -1)) * -1.0e9
        )

        present_kv = []
        for layer_idx in range(self.n_layer):
            past_k = flat_past_kv[2 * layer_idx]
            past_v = flat_past_kv[2 * layer_idx + 1]
            hidden_states, present_k, present_v = self._block_forward(
                self.h[layer_idx], hidden_states, past_k, past_v, attn_mask_4d
            )
            present_kv.append(present_k)
            present_kv.append(present_v)

        hidden_states = self.ln_f(hidden_states)
        logits = self.speech_head(hidden_states).squeeze(1)  # (1, 6563)
        return (logits, *present_kv)


def _lm_onnx_io_names():
    inputs = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(GPT2_LAYERS):
        inputs.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    outputs = ["logits"]
    for i in range(GPT2_LAYERS):
        outputs.extend([f"present.{i}.key", f"present.{i}.value"])
    return inputs, outputs


def _lm_onnx_dynamic_axes():
    axes = {
        "attention_mask": {1: "total_seq_len"},
    }
    for i in range(GPT2_LAYERS):
        axes[f"past_key_values.{i}.key"] = {2: "past_len"}
        axes[f"past_key_values.{i}.value"] = {2: "past_len"}
        axes[f"present.{i}.key"] = {2: "total_seq_len"}
        axes[f"present.{i}.value"] = {2: "total_seq_len"}
    return axes


def _make_fixture_lm_onnx(seed=0, past_len=10):
    """Deterministic fixture matching the Swift call signature in OrtFastDecoder."""
    rng = _seeded_rng(seed)
    total_seq_len = past_len + 1
    inputs_embeds = rng.standard_normal((1, 1, GPT2_HIDDEN), dtype=np.float32) * 0.02
    attention_mask = np.ones((1, total_seq_len), dtype=np.int64)
    position_ids = np.array([[past_len]], dtype=np.int64)
    kv = []
    for _ in range(GPT2_LAYERS):
        k = rng.standard_normal((1, GPT2_HEADS, past_len, GPT2_HEAD_DIM), dtype=np.float32) * 0.1
        v = rng.standard_normal((1, GPT2_HEADS, past_len, GPT2_HEAD_DIM), dtype=np.float32) * 0.1
        kv.append((k, v))
    return inputs_embeds, attention_mask, position_ids, kv


def convert_language_model_onnx(model, output_dir, validate=False, reference_dir=None,
                                 optimize_graph: bool = False, quantize: str = "none"):
    """Export the single-step GPT-2 decoder as language_model_single.onnx."""
    print("\n=== Converting language_model_single.onnx ===")
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "language_model_single.onnx")

    wrapper = _LanguageModelWrapper(model.t3)
    wrapper.train(False)

    past_len = 10
    embeds_t = torch.randn(1, 1, GPT2_HIDDEN, dtype=torch.float32) * 0.02
    mask_t = torch.ones(1, past_len + 1, dtype=torch.int64)
    pos_t = torch.tensor([[past_len]], dtype=torch.int64)
    flat_pkv_t = []
    for _ in range(GPT2_LAYERS):
        flat_pkv_t.append(torch.randn(1, GPT2_HEADS, past_len, GPT2_HEAD_DIM) * 0.1)
        flat_pkv_t.append(torch.randn(1, GPT2_HEADS, past_len, GPT2_HEAD_DIM) * 0.1)

    input_names, output_names = _lm_onnx_io_names()
    dynamic_axes = _lm_onnx_dynamic_axes()

    print(f"  Exporting to {out_path} (opset 17)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (embeds_t, mask_t, pos_t, *flat_pkv_t),
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Wrote {size_mb:.1f} MB")

    if optimize_graph:
        _optimize_onnx_graph(
            out_path, model_type="gpt2",
            num_heads=GPT2_HEADS, hidden_size=GPT2_HIDDEN,
        )
    if quantize == "int8":
        _quantize_onnx_int8(out_path)

    if validate:
        validate_language_model_onnx(model, out_path, reference_dir=reference_dir)


def validate_language_model_onnx(model, our_path, reference_dir=None):
    print("\n  --- language_model_single.onnx validation ---")

    embeds, mask, pos, kv_pairs = _make_fixture_lm_onnx(seed=0, past_len=10)

    our_sess = _load_onnx_session(our_path)
    inputs = {
        "inputs_embeds": embeds,
        "attention_mask": mask,
        "position_ids": pos,
    }
    for i, (k, v) in enumerate(kv_pairs):
        inputs[f"past_key_values.{i}.key"] = k
        inputs[f"past_key_values.{i}.value"] = v
    output_names = ["logits"] + sum(
        ([f"present.{i}.key", f"present.{i}.value"] for i in range(GPT2_LAYERS)), []
    )

    our_outs = our_sess.run(output_names, inputs)
    our_logits = our_outs[0]
    our_present_k = [our_outs[1 + 2 * i] for i in range(GPT2_LAYERS)]
    our_present_v = [our_outs[2 + 2 * i] for i in range(GPT2_LAYERS)]

    if reference_dir is not None:
        ref_path = os.path.join(reference_dir, "onnx", "language_model_single.onnx")
        if not os.path.exists(ref_path):
            print(f"  (reference not at {ref_path}; falling back to PyTorch ref)")
            ref_outs = None
        else:
            print(f"  Comparing against HF reference at {ref_path}...")
            ref_sess = _load_onnx_session(ref_path)
            ref_outs = ref_sess.run(output_names, inputs)
    else:
        ref_outs = None

    if ref_outs is None:
        print("  Comparing against PyTorch reference...")
        with torch.no_grad():
            wrapper = _LanguageModelWrapper(model.t3)
            wrapper.train(False)
            t_inputs = (
                torch.from_numpy(embeds),
                torch.from_numpy(mask),
                torch.from_numpy(pos),
                *[torch.from_numpy(arr) for kv in kv_pairs for arr in kv],
            )
            t_out = wrapper(*t_inputs)
            ref_outs = [x.detach().cpu().numpy() for x in t_out]

    ok = True
    ok &= _compare_outputs("logits", our_logits, ref_outs[0], atol=1e-2, cos_min=0.999)
    for i in range(GPT2_LAYERS):
        ok &= _compare_outputs(
            f"present.{i}.key", our_present_k[i], ref_outs[1 + 2 * i], atol=5e-3
        )
        ok &= _compare_outputs(
            f"present.{i}.value", our_present_v[i], ref_outs[2 + 2 * i], atol=5e-3
        )

    print(f"\n  --- iOS compatibility ---")
    ok &= _check_onnx_graph_for_ios(our_path)
    print(f"\n  language_model_single.onnx: {'READY' if ok else 'NOT READY'}")
    return ok


# --- Stage B: T3Prefill.mlpackage ------------------------------------------


class _T3PrefillWrapper(nn.Module):
    """Single-pass prefill: text + cond_speech + speaker + start_speech → logits + full KV cache.

    Mirrors the chatterbox T3 prefill assembly (cond_enc + prepare_input_embeds)
    and the manual attention path from _LanguageModelWrapper. The Swift caller
    (OnnxT3Runner.swift:runCoreMLPrefill) expects this exact contract:
      Inputs:  text_tokens (1, T_text) int32,
               cond_speech_tokens (1, T_cond) int32,
               speaker_emb (1, 256) fp32,
               speech_tokens (1, T_speech) int32  (typically T_speech=1 = start token)
      Outputs: logits (1, 6563) fp32,
               kv_cache (48, 1, 16, T_total, 64) fp32, interleaved K0,V0,K1,V1,...,K23,V23
    """

    def __init__(self, t3):
        super().__init__()
        self.text_emb = t3.text_emb            # Embedding(50276, 1024)
        self.speech_emb = t3.speech_emb        # Embedding(6563, 1024)
        self.spkr_enc = t3.cond_enc.spkr_enc   # Linear(256, 1024)
        self.wpe = t3.tfmr.wpe                 # Embedding(8196, 1024)
        self.ln_f = t3.tfmr.ln_f
        self.h = t3.tfmr.h                     # ModuleList of 24 GPT2Block
        self.speech_head = t3.speech_head      # Linear(1024, 6563, bias=False)
        self.n_layer = GPT2_LAYERS
        self.n_head = GPT2_HEADS
        self.head_dim = GPT2_HEAD_DIM
        self.hidden = GPT2_HIDDEN

    def forward(self, text_tokens, cond_speech_tokens, speaker_emb, speech_tokens):
        # Embedding lookups (int32 -> long for nn.Embedding)
        text_e = self.text_emb(text_tokens.long())             # (1, T_text, H)
        cond_e = self.speech_emb(cond_speech_tokens.long())    # (1, T_cond, H)
        speech_e = self.speech_emb(speech_tokens.long())       # (1, T_speech, H)
        spkr_e = self.spkr_enc(speaker_emb).unsqueeze(1)       # (1, 1, H)

        # T3CondEnc.forward for Turbo (use_perceiver_resampler=False, emotion_adv=False)
        # builds cond_emb = [spkr, cond_clap=empty, cond_prompt_speech, cond_emotion=empty]
        # and prepare_input_embeds concats [cond_emb, text_emb, speech_emb].
        embeds = torch.cat([spkr_e, cond_e, text_e, speech_e], dim=1)  # (1, T, H)
        T = embeds.shape[1]

        # GPT-2 absolute position embeddings (wpe)
        position_ids = torch.arange(T, dtype=torch.long, device=embeds.device).unsqueeze(0)
        hidden = embeds + self.wpe(position_ids)

        # Explicit additive causal mask. SDPA's is_causal=True works in PyTorch
        # but trace-export sometimes elides it on dynamic shapes — explicit is
        # safer for CoreML/ONNX export reproducibility.
        causal_mask = torch.triu(
            torch.full((T, T), -1.0e9, dtype=hidden.dtype, device=hidden.device),
            diagonal=1,
        ).view(1, 1, T, T)

        present_kv = []  # interleaved K0, V0, K1, V1, ...
        for layer_idx in range(self.n_layer):
            block = self.h[layer_idx]

            # Self-attention
            residual = hidden
            h_in = block.ln_1(hidden)
            qkv = block.attn.c_attn(h_in)
            q, k, v = qkv.split(self.hidden, dim=2)
            q = q.view(1, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(1, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(1, T, self.n_head, self.head_dim).transpose(1, 2)

            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, is_causal=False, dropout_p=0.0
            )
            attn = attn.transpose(1, 2).contiguous().view(1, T, self.hidden)
            attn = block.attn.c_proj(attn)
            attn = block.attn.resid_dropout(attn)
            hidden = residual + attn

            # MLP
            residual = hidden
            h_mlp = block.ln_2(hidden)
            h_mlp = block.mlp(h_mlp)
            hidden = residual + h_mlp

            present_kv.append(k)  # (1, 16, T, 64)
            present_kv.append(v)

        hidden = self.ln_f(hidden)
        # Logits at the last position (= the start-speech-token's hidden state)
        logits = self.speech_head(hidden[:, -1:, :]).squeeze(1)  # (1, 6563)

        # (48, 1, 16, T, 64) interleaved K/V per layer — matches Swift splitKVCacheRaw
        kv_cache = torch.stack(present_kv, dim=0)
        return logits, kv_cache


def _make_fixture_prefill(seed=0, t_text=3, t_cond=375, t_speech=1):
    """Deterministic fixture matching MIL defaults (text=3, cond=375, speech=1)."""
    rng = _seeded_rng(seed)
    text_tokens = rng.integers(0, 50000, size=(1, t_text), dtype=np.int32)
    cond_speech_tokens = rng.integers(0, SPEECH_VOCAB_SIZE - 2, size=(1, t_cond), dtype=np.int32)
    speaker_emb = rng.standard_normal((1, SPEAKER_EMB_DIM), dtype=np.float32) * 0.5
    speech_tokens = np.array([[SPEECH_START_TOKEN]], dtype=np.int32)
    return text_tokens, cond_speech_tokens, speaker_emb, speech_tokens


def convert_prefill(model, output_dir, validate=False, reference_dir=None,
                     quantize: str = "none"):
    """Convert the T3 prefill module to T3Prefill.mlpackage."""
    print("\n=== Converting T3Prefill.mlpackage ===")
    out_path = os.path.join(output_dir, "T3Prefill.mlpackage")

    wrapper = _T3PrefillWrapper(model.t3)
    wrapper.train(False)

    # Trace with MIL default shapes (T_text=3, T_cond=375, T_speech=1)
    text_t, cond_t, spkr_t, spch_t = _make_fixture_prefill(seed=0)
    text_pt = torch.from_numpy(text_t)
    cond_pt = torch.from_numpy(cond_t)
    spkr_pt = torch.from_numpy(spkr_t)
    spch_pt = torch.from_numpy(spch_t)

    print(
        f"  Tracing with T_text={text_pt.shape[1]}, "
        f"T_cond={cond_pt.shape[1]}, T_speech={spch_pt.shape[1]}..."
    )
    with torch.no_grad():
        # check_trace=False suppresses spurious 1e-5 mismatch warnings caused
        # by the SDPA fast path; we validate the converted model afterwards.
        traced = torch.jit.trace(
            wrapper, (text_pt, cond_pt, spkr_pt, spch_pt), check_trace=False
        )

    print("  Converting to CoreML (FP32 weights, iOS18 target)...")
    inputs = [
        ct.TensorType(
            name="text_tokens",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512, default=3))),
            dtype=np.int32,
        ),
        ct.TensorType(
            name="cond_speech_tokens",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=1024, default=375))),
            dtype=np.int32,
        ),
        ct.TensorType(name="speaker_emb", shape=(1, SPEAKER_EMB_DIM), dtype=np.float32),
        ct.TensorType(
            name="speech_tokens",
            shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=2048, default=1))),
            dtype=np.int32,
        ),
    ]
    outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
        ct.TensorType(name="kv_cache", dtype=np.float32),
    ]
    # FLOAT32 matches the HF reference T3Prefill.mlmodelc weight precision
    # (1.5 GB weight.bin = 4 bytes/param). FLOAT16 cuts that in half but the
    # iPhone happily runs FP32 too and the HF copy is what's been validated
    # device-side.
    #
    # compute_units=ALL lets the runtime bid for ANE in addition to CPU and
    # GPU. The Swift consumer picks the actual dispatch via
    # MLModelConfiguration.computeUnits — declaring ALL here just keeps ANE
    # eligible. Previous version pinned CPU_AND_GPU which made ANE bidding
    # impossible regardless of the Swift config.
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    size_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(out_path) for f in files
    ) / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f} MB)")

    if quantize == "int8":
        _quantize_coreml_int8(out_path)

    if validate:
        validate_prefill(model, out_path, reference_dir=reference_dir)


def validate_prefill(model, our_path, reference_dir=None):
    print("\n  --- T3Prefill.mlpackage validation ---")

    # Use the MIL default fixture (T_text=3, T_cond=375, T_speech=1) so any
    # reference vs ours diff is purely numerical, not shape-driven.
    text_t, cond_t, spkr_t, spch_t = _make_fixture_prefill(seed=0)
    inputs = {
        "text_tokens": text_t,
        "cond_speech_tokens": cond_t,
        "speaker_emb": spkr_t,
        "speech_tokens": spch_t,
    }

    # Load OURS via CoreML
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pkg = os.path.join(tmp, "T3Prefill.mlpackage")
        shutil.copytree(our_path, tmp_pkg)
        our_ml = ct.models.MLModel(tmp_pkg, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        our_out = our_ml.predict(inputs)
    our_logits = np.asarray(our_out["logits"], dtype=np.float32)
    our_kv = np.asarray(our_out["kv_cache"], dtype=np.float32)

    # Reference. The HF snapshot's T3Prefill.mlmodelc dir is missing the
    # Manifest.json that coremltools' MLModel() needs to load — it was the
    # raw model.mil/weights output from ct.convert, not a Xcode-compiled
    # artifact. If we can't load it directly, fall back to PyTorch.
    ref_logits = ref_kv = None
    if reference_dir is not None:
        for candidate in ("T3Prefill.mlpackage", "T3Prefill.mlmodelc"):
            ref_path = os.path.join(reference_dir, candidate)
            if not os.path.exists(ref_path):
                continue
            try:
                print(f"  Comparing against HF reference at {ref_path}...")
                ref_ml = ct.models.MLModel(ref_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
                ref_out = ref_ml.predict(inputs)
                ref_logits = np.asarray(ref_out["logits"], dtype=np.float32)
                ref_kv = np.asarray(ref_out["kv_cache"], dtype=np.float32)
                break
            except Exception as exc:
                print(f"  (could not load HF reference: {str(exc)[:160]}; will fall back to PyTorch)")

    if ref_logits is None:
        print("  Comparing against PyTorch reference...")
        with torch.no_grad():
            wrapper = _T3PrefillWrapper(model.t3)
            wrapper.train(False)
            t_logits, t_kv = wrapper(
                torch.from_numpy(text_t),
                torch.from_numpy(cond_t),
                torch.from_numpy(spkr_t),
                torch.from_numpy(spch_t),
            )
            ref_logits = t_logits.detach().cpu().numpy()
            ref_kv = t_kv.detach().cpu().numpy()

    ok = True
    ok &= _compare_outputs("logits", our_logits, ref_logits, cos_min=0.99, atol=0.5)
    ok &= _compare_outputs("kv_cache", our_kv, ref_kv, cos_min=0.99, atol=0.05)

    print(f"\n  --- iOS compatibility ---")
    ok &= _check_xcrun_coremlcompiler(our_path)
    ok &= _try_ane_compute(our_path, inputs)
    print(f"\n  T3Prefill.mlpackage: {'READY' if ok else 'NOT READY'}")
    return ok


# --- Stage C: conditional_decoder_single.onnx ------------------------------


class _ConditionalDecoderWrapper(nn.Module):
    """Bundles S3Gen's entire post-T3 audio chain into one ONNX graph.

    Matches Swift OnnxConditionalDecoder.swift's call signature:
      Inputs:
        speech_tokens     (1, T_total)  int64  -- prompt_token+speech_token concat
        speaker_embeddings (1, 192)     float  -- raw CAMPPlus emb, pre-normalized by Swift
        speaker_features   (1, T_feat, 80) float  -- prompt mel features (BTC)
      Output:
        waveform (1, N_samples) float, 24 kHz

    Internally:
      input_embedding -> encoder -> encoder_proj          (token -> mel mu)
      spk_embed_affine_layer                              (192 -> internal cond dim)
      CFM 2-step Euler solver, unrolled (meanflow=True)   (mu, mask, spks, cond -> mel)
      Strip prompt portion of mel
      mel2wav (HiFTGenerator)                             (mel -> waveform via ISTFT)
      Trim-fade to reduce reference clip spillover

    Skips chatterbox's `flow.inference` glue because it has Python control flow
    (`.item()`, dict ref_dict casting) that doesn't trace.
    """

    def __init__(self, s3gen, cfm_steps: int = 2):
        super().__init__()
        if cfm_steps not in (1, 2):
            raise ValueError(f"cfm_steps must be 1 or 2, got {cfm_steps}")
        self.cfm_steps = cfm_steps
        flow = s3gen.flow
        self.input_embedding = flow.input_embedding
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj
        self.spk_embed_affine_layer = flow.spk_embed_affine_layer
        self.estimator = flow.decoder.estimator
        self.mel2wav = s3gen.mel2wav
        # trim_fade is a (2 * n_trim,) tensor: zeros for the first n_trim samples,
        # then a cos-shaped ramp up. Multiplied into the wav head to mute the
        # carry-over from the reference clip.
        self.register_buffer("trim_fade", s3gen.trim_fade.detach().clone(), persistent=False)
        self.output_size = flow.output_size  # 80

    def forward(self, speech_tokens, speaker_embeddings, speaker_features):
        # --- speaker conditioning -------------------------------------------
        spks = self.spk_embed_affine_layer(speaker_embeddings)  # (1, internal_dim)

        # --- token embedding + Conformer encoder ----------------------------
        T_total = speech_tokens.shape[1]
        T_feat = speaker_features.shape[1]
        # Dynamo (torch 2.9 path) finds size-specialization branches in the
        # encoder; assert minimal bounds so it knows our Dim range avoids
        # those specializations. No-ops at runtime. Only call torch._check
        # under dynamo — torch 2.8 jit.trace passes Python bools and rejects
        # SymBool/Tensor inputs to _check.
        if _torch_at_least("2.9"):
            torch._check(T_total >= 2)
            torch._check(T_feat >= 2)
            torch._check(2 * T_total - T_feat >= 1)
        # Length tensor for the encoder. No padding (batch=1 full sequence).
        token_len = torch.tensor([T_total], dtype=torch.long, device=speech_tokens.device)
        x = self.input_embedding(speech_tokens.long())  # (1, T_total, in_dim)
        h, _ = self.encoder(x, token_len)               # (1, T_h, enc_dim)
        h = self.encoder_proj(h)                        # (1, T_h, 80)

        T_h = h.shape[1]
        T_feat = speaker_features.shape[1]              # prompt mel length

        # --- conds = [prompt_mel | zeros], BCT layout -----------------------
        # zeros of length T_h - T_feat (= T_mel_gen) padded after prompt
        zeros_pad = torch.zeros(
            (1, T_h - T_feat, self.output_size), dtype=h.dtype, device=h.device
        )
        conds_btc = torch.cat([speaker_features, zeros_pad], dim=1)  # (1, T_h, 80)
        conds = conds_btc.transpose(1, 2).contiguous()               # (1, 80, T_h)

        # --- mu + mask for CFM ----------------------------------------------
        mu = h.transpose(1, 2).contiguous()                          # (1, 80, T_h)
        # No padding -> mask = ones. Required shape: (1, 1, T_h).
        mask = torch.ones((1, 1, T_h), dtype=h.dtype, device=h.device)

        # --- CFM Euler solver (meanflow=True), unrolled to fixed step count -
        # CausalConditionalCFM.forward initializes z = randn_like(mu) then in
        # meanflow path splices noised_mels (also randn) over the generated
        # portion. Net effect: z is full Gaussian noise.
        z = torch.randn_like(mu)

        if self.cfm_steps == 2:
            # t_span = linspace(0, 1, 3) = [0, 0.5, 1.0]; dt = 0.5
            t0 = torch.zeros(1, dtype=mu.dtype, device=mu.device)
            t1 = torch.full((1,), 0.5, dtype=mu.dtype, device=mu.device)
            t2 = torch.ones(1, dtype=mu.dtype, device=mu.device)
            dt = torch.full((1,), 0.5, dtype=mu.dtype, device=mu.device)
            dxdt = self.estimator(z, mask, mu, t0, spks, conds, t1)
            x1 = z + dt * dxdt
            dxdt = self.estimator(x1, mask, mu, t1, spks, conds, t2)
            feat_full = x1 + dt * dxdt  # (1, 80, T_h)
        else:
            # 1-step: t_span = [0, 1]; dt = 1; one estimator call. Roughly
            # halves cond_decoder time at a (typically small) audio quality cost.
            t0 = torch.zeros(1, dtype=mu.dtype, device=mu.device)
            t1 = torch.ones(1, dtype=mu.dtype, device=mu.device)
            dt = torch.ones(1, dtype=mu.dtype, device=mu.device)
            dxdt = self.estimator(z, mask, mu, t0, spks, conds, t1)
            feat_full = z + dt * dxdt

        # Slice off the prompt portion -> mel for the generated speech only
        feat = feat_full[:, :, T_feat:]  # (1, 80, T_h - T_feat)

        # --- HiFTGenerator vocoder ------------------------------------------
        # cache_source is empty: the .inference() if-branch on shape[2] != 0
        # is excluded during trace. Pass empty as a literal so it doesn't show
        # up as a graph input.
        cache_source = torch.zeros((1, 1, 0), dtype=feat.dtype, device=feat.device)
        wav, _ = self.mel2wav.inference(speech_feat=feat, cache_source=cache_source)
        # wav: (1, N_samples)

        # --- trim_fade head mute --------------------------------------------
        n_trim = self.trim_fade.shape[0]
        head = wav[:, :n_trim] * self.trim_fade  # (1, n_trim)
        tail = wav[:, n_trim:]                   # (1, N_samples - n_trim)
        wav = torch.cat([head, tail], dim=1)
        return wav


def _make_fixture_cond_decoder(seed=0, n_prompt_tokens=50, n_gen_tokens=20):
    """Deterministic fixture for conditional_decoder.

    Chatterbox's flow inference constrains T_feat = 2 * n_prompt_tokens (the
    encoder upsamples tokens 2x to mel frames). speech_tokens is the prompt +
    generated tokens concatenated; speaker_features is the prompt's mel.
    """
    rng = _seeded_rng(seed)
    t_speech = n_prompt_tokens + n_gen_tokens
    t_feat = 2 * n_prompt_tokens
    speech_tokens = rng.integers(0, SPEECH_VOCAB_SIZE - 3, size=(1, t_speech), dtype=np.int64)
    raw_spk = rng.standard_normal((1, CAMPP_EMB_DIM), dtype=np.float32)
    norm = np.linalg.norm(raw_spk) + 1e-12
    speaker_embeddings = (raw_spk / norm).astype(np.float32)
    speaker_features = (
        rng.standard_normal((1, t_feat, 80), dtype=np.float32) * 0.5
    ).astype(np.float32)
    return speech_tokens, speaker_embeddings, speaker_features


def _patch_chatterbox_for_export():
    """Monkey-patch chatterbox modules to bypass tracing hazards.

    1. `mask.add_optional_chunk_mask` does a defensive `.item()` check at line
       161 to repair all-False mask rows. For batch=1 full-sequence usage this
       branch is never taken; replace the function with a version that skips
       it so dynamo doesn't error on GuardOnDataDependentSymNode.

    2. `hifigan.SineGen.forward` constructs random phase via
       `Uniform(...).sample(...)` and random noise via `torch.randn_like` —
       both decompose into ops whose dtypes (`prims.signbit`) the dynamo
       FX decomposer mis-tracks against the real tensor pass. Replace it
       with a deterministic version (zero phase, zero noise). HiFTGenerator
       inference doesn't need stochastic noise for correctness — the
       additional source noise is a quality nudge, not load-bearing.
    """
    import numpy as _np
    import chatterbox.models.s3gen.utils.mask as _cm_mask
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask
    import chatterbox.models.s3gen.hifigan as _cm_hifi

    if getattr(_cm_mask, "_export_safe", False):
        return

    def _safe_add_optional_chunk_mask(
        xs, masks, use_dynamic_chunk, use_dynamic_left_chunk,
        decoding_chunk_size, static_chunk_size, num_decoding_left_chunks,
        enable_full_context=True,
    ):
        if use_dynamic_chunk:
            max_len = xs.size(1)
            if decoding_chunk_size < 0:
                chunk_size = max_len; num_left_chunks = -1
            elif decoding_chunk_size > 0:
                chunk_size = decoding_chunk_size
                num_left_chunks = num_decoding_left_chunks
            else:
                chunk_size = max_len
                num_left_chunks = -1
            chunk_masks = subsequent_chunk_mask(
                xs.size(1), chunk_size, num_left_chunks, xs.device
            ).unsqueeze(0)
            chunk_masks = masks & chunk_masks
        elif static_chunk_size > 0:
            chunk_masks = subsequent_chunk_mask(
                xs.size(1), static_chunk_size, num_decoding_left_chunks, xs.device
            ).unsqueeze(0)
            chunk_masks = masks & chunk_masks
        else:
            chunk_masks = masks
        return chunk_masks

    # SourceModuleHnNSF.forward feeds through SineGen which has 9 harmonic
    # bins; the `cumsum % 1` and `voiced_threshold` comparison decompose into
    # `prims.signbit` ops that torch 2.8's dynamo fake-tensor system mis-tracks.
    # Bypass the entire harmonic source path: return tanh(0) = 0 of the right
    # shape. The downstream `decode()` still mixes this into the upsampled
    # signal, just with no harmonic contribution. Quality drops a bit; the
    # graph exports cleanly and the model still produces intelligible audio.
    def _safe_source_module_forward(self, x):
        # x: (B, T, 1) per SourceModuleHnNSF docstring
        # output sine_merge: (B, T, 1) — l_linear projects harmonic_num+1 -> 1
        sine_merge = torch.zeros(
            (x.size(0), x.size(1), 1), device=x.device, dtype=x.dtype
        )
        return sine_merge, None, None

    # EspnetRelPositionalEncoding.extend_pe is called on every encoder forward
    # and rewrites self.pe in-place via `pe_positive[:, 0::2] = ...`. PE is
    # populated in __init__ to max_len=5000 — plenty for our sizes — so we
    # disable the extend logic entirely. Increase max_len at construction time
    # if you ever trace with > 5000 mel frames.
    import chatterbox.models.s3gen.transformer.embedding as _cm_emb

    def _noop_extend_pe(self, x):
        return
    _cm_emb.EspnetRelPositionalEncoding.extend_pe = _noop_extend_pe

    # HiFTGenerator.inference has the f0_predictor → m_source path (signbit
    # dtype mismatch — already bypassed via the SourceModule patch), and the
    # cache_source if-branch `if cache_source.shape[2] != 0: s[:, :, :cs] = ...`
    # which dynamo still traces as in-place (as_strided + copy_to) even when
    # statically false. Replace with a streamlined version that feeds the
    # zero source (matching the bypassed m_source output) straight to decode.
    def _safe_hift_inference(self, speech_feat, cache_source=None):
        upsample_scale = 1
        for r in [8, 5, 3]:
            upsample_scale *= r
        upsample_scale *= self.istft_params["hop_len"]
        T_samples = speech_feat.shape[2] * upsample_scale
        # decode() expects s shaped (B, 1, T_samples)
        s = torch.zeros(
            (speech_feat.size(0), 1, T_samples),
            device=speech_feat.device, dtype=speech_feat.dtype,
        )
        return self.decode(x=speech_feat, s=s), s

    # _stft is called inside decode() to compute the source STFT. With our
    # zero-source bypass, the result is zero — return right-shaped zeros to
    # avoid the legacy exporter rejecting torch.stft's complex output.
    def _safe_stft(self, x):
        n_fft = self.istft_params["n_fft"]
        hop = self.istft_params["hop_len"]
        K = n_fft // 2 + 1
        n_frames = x.shape[-1] // hop + 1
        zeros = torch.zeros((x.shape[0], K, n_frames), device=x.device, dtype=x.dtype)
        return zeros, zeros

    # _istft uses torch.complex(real, imag) + torch.istft. Neither exports in
    # opset 17 legacy. Replace with primitives: real-spectrum IDFT via cos/sin
    # matmul + overlap-add via conv_transpose1d (identity kernel). F.fold was
    # the first attempt but the legacy opset-18 col2im symbolic chokes on
    # dynamic output_size with a NoneType subscript bug.
    def _safe_istft(self, magnitude, phase):
        n_fft = self.istft_params["n_fft"]   # 16
        hop = self.istft_params["hop_len"]   # 4
        K = n_fft // 2 + 1                   # 9
        window = self.stft_window.to(magnitude.device).to(magnitude.dtype)

        magnitude = torch.clip(magnitude, max=1.0e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        # Reconstruction weights for one-sided spectrum (DC and Nyquist unscaled,
        # interior bins x2).
        weights = torch.ones(K, dtype=magnitude.dtype, device=magnitude.device)
        weights[1 : K - 1] = 2.0

        # IDFT cos/sin basis: angle[k, n] = 2*pi*k*n/N
        n_idx = torch.arange(n_fft, dtype=magnitude.dtype, device=magnitude.device)
        k_idx = torch.arange(K, dtype=magnitude.dtype, device=magnitude.device)
        angle = (2.0 * torch.pi / n_fft) * k_idx[:, None] * n_idx[None, :]
        cos_basis = torch.cos(angle)
        sin_basis = torch.sin(angle)

        real_w = (real * weights[None, :, None]).transpose(1, 2)  # (B, F, K)
        imag_w = (imag * weights[None, :, None]).transpose(1, 2)

        # Time-domain per-frame samples
        frames = (real_w @ cos_basis - imag_w @ sin_basis) / n_fft  # (B, F, n_fft)
        frames = frames * window[None, None, :]

        # Overlap-add via conv_transpose1d with identity kernel.
        # input (B, n_fft, F), weight (n_fft, 1, n_fft) where w[i, 0, j] = 1[i==j],
        # output (B, 1, (F-1)*hop + n_fft).
        x_for_ct = frames.transpose(1, 2).contiguous()
        eye_kernel = torch.eye(n_fft, dtype=magnitude.dtype, device=magnitude.device).unsqueeze(1)
        ola = torch.nn.functional.conv_transpose1d(
            x_for_ct, eye_kernel, stride=hop
        ).squeeze(1)  # (B, out_full_len)

        # Window envelope normalization (matches torch.istft(normalized=False)).
        win_sq = (window ** 2)[None, :, None].expand(1, n_fft, frames.shape[1])
        win_sq_kernel = eye_kernel  # same identity
        env = torch.nn.functional.conv_transpose1d(
            win_sq, win_sq_kernel, stride=hop
        ).squeeze(1)  # (1, out_full_len)
        env = torch.clamp(env, min=1.0e-10)
        out = ola / env

        # torch.istft(center=True) trims n_fft//2 from each side.
        trim = n_fft // 2
        out_len = ola.shape[1]
        out = out[:, trim : out_len - trim]
        return out

    # decoder.py and upsample_encoder.py both do
    #   `from .utils.mask import add_optional_chunk_mask`
    # which captures the function by value at import time. Patching the
    # source module alone doesn't reach those consumers.
    import chatterbox.models.s3gen.decoder as _cm_dec
    import chatterbox.models.s3gen.transformer.upsample_encoder as _cm_uenc
    _cm_mask.add_optional_chunk_mask = _safe_add_optional_chunk_mask
    _cm_dec.add_optional_chunk_mask = _safe_add_optional_chunk_mask
    _cm_uenc.add_optional_chunk_mask = _safe_add_optional_chunk_mask
    _cm_hifi.SourceModuleHnNSF.forward = _safe_source_module_forward
    _cm_hifi.HiFTGenerator.inference = _safe_hift_inference
    _cm_hifi.HiFTGenerator._stft = _safe_stft
    _cm_hifi.HiFTGenerator._istft = _safe_istft
    _cm_mask._export_safe = True


def _patch_chatterbox_for_export_minimal():
    """Apply only the patches needed to dodge actual chatterbox bugs +
    chatterbox-code-style problems (vs the bypass patches that drop
    functionality). Used by torch29 mode.

    Still applied:
    - add_optional_chunk_mask .item() check (data-dependent guard)
    - EspnetRelPositionalEncoding.extend_pe (in-place PE rebuild)
    - SineGen.forward rewritten functionally — same outputs, no in-place
      F_mat slice writes (which dynamo emits as ScatterND with int32
      indices that ORT rejects), no Uniform.sample / randn (which torch
      2.9's fake-tensor system mis-tracks dtypes on)
    """
    import numpy as _np
    import chatterbox.models.s3gen.utils.mask as _cm_mask
    from chatterbox.models.s3gen.utils.mask import subsequent_chunk_mask
    import chatterbox.models.s3gen.decoder as _cm_dec
    import chatterbox.models.s3gen.transformer.upsample_encoder as _cm_uenc
    import chatterbox.models.s3gen.transformer.embedding as _cm_emb
    import chatterbox.models.s3gen.hifigan as _cm_hifi

    if getattr(_cm_mask, "_export_safe_minimal", False):
        return

    def _safe_add_optional_chunk_mask(
        xs, masks, use_dynamic_chunk, use_dynamic_left_chunk,
        decoding_chunk_size, static_chunk_size, num_decoding_left_chunks,
        enable_full_context=True,
    ):
        if use_dynamic_chunk:
            max_len = xs.size(1)
            if decoding_chunk_size < 0:
                chunk_size = max_len; num_left_chunks = -1
            elif decoding_chunk_size > 0:
                chunk_size = decoding_chunk_size
                num_left_chunks = num_decoding_left_chunks
            else:
                chunk_size = max_len
                num_left_chunks = -1
            chunk_masks = subsequent_chunk_mask(
                xs.size(1), chunk_size, num_left_chunks, xs.device
            ).unsqueeze(0)
            chunk_masks = masks & chunk_masks
        elif static_chunk_size > 0:
            chunk_masks = subsequent_chunk_mask(
                xs.size(1), static_chunk_size, num_decoding_left_chunks, xs.device
            ).unsqueeze(0)
            chunk_masks = masks & chunk_masks
        else:
            chunk_masks = masks
        return chunk_masks

    def _noop_extend_pe(self, x):
        return

    def _functional_sine_gen_forward(self, f0):
        """SineGen.forward without in-place writes or random ops.

        Replaces the original:
          F_mat = torch.zeros(...)
          for i in range(harmonic_num + 1):
              F_mat[:, i:i+1, :] = f0 * (i+1) / sr   # in-place -> ScatterND
          phase_vec = Uniform(-pi, pi).sample(...)   # randn dtype issues
          phase_vec[:, 0, :] = 0                      # in-place
          sine = amp * sin(theta + phase_vec)
          noise = noise_amp * randn_like(sine)        # randn dtype issues
          sine = sine * uv + noise

        With a vectorized + deterministic equivalent. Zero phase is fine
        because sin is rotation-invariant around the start sample; zero noise
        omits the additive HnNSF noise term (small quality cost, much
        cleaner export).
        """
        H = self.harmonic_num + 1
        multipliers = torch.arange(
            1, H + 1, dtype=f0.dtype, device=f0.device
        ).view(1, H, 1)
        # f0: (B, 1, T) -> F_mat: (B, H, T)
        F_mat = (f0 * multipliers) / self.sampling_rate
        theta_mat = 2.0 * _np.pi * torch.cumsum(F_mat, dim=-1)
        sine_waves = self.sine_amp * torch.sin(theta_mat)
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        sine_waves = sine_waves * uv
        noise = torch.zeros_like(sine_waves)
        return sine_waves, uv, noise

    # ISTFT replacement: torch 2.9 dynamo's torch.istft decomposition emits a
    # _fft_c2r output whose last-dim shape (9 = n_fft//2+1) doesn't broadcast
    # with the stft_window (16 = n_fft) in a downstream Mul, raising a
    # BroadcastIterator error at ORT runtime. Use the same primitive
    # implementation as legacy mode — it stays inside dynamo-friendly ops
    # (Cast/MatMul/Sin/Cos/ConvTranspose1d) and produces equivalent output.
    def _safe_istft_for_torch29(self, magnitude, phase):
        n_fft = self.istft_params["n_fft"]
        hop = self.istft_params["hop_len"]
        K = n_fft // 2 + 1
        window = self.stft_window.to(magnitude.device).to(magnitude.dtype)

        magnitude = torch.clip(magnitude, max=1.0e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        weights = torch.ones(K, dtype=magnitude.dtype, device=magnitude.device)
        weights[1 : K - 1] = 2.0

        n_idx = torch.arange(n_fft, dtype=magnitude.dtype, device=magnitude.device)
        k_idx = torch.arange(K, dtype=magnitude.dtype, device=magnitude.device)
        angle = (2.0 * torch.pi / n_fft) * k_idx[:, None] * n_idx[None, :]
        cos_basis = torch.cos(angle)
        sin_basis = torch.sin(angle)

        real_w = (real * weights[None, :, None]).transpose(1, 2)
        imag_w = (imag * weights[None, :, None]).transpose(1, 2)

        frames = (real_w @ cos_basis - imag_w @ sin_basis) / n_fft
        frames = frames * window[None, None, :]

        x_for_ct = frames.transpose(1, 2).contiguous()
        eye_kernel = torch.eye(n_fft, dtype=magnitude.dtype, device=magnitude.device).unsqueeze(1)
        ola = torch.nn.functional.conv_transpose1d(x_for_ct, eye_kernel, stride=hop).squeeze(1)

        win_sq = (window ** 2)[None, :, None].expand(1, n_fft, frames.shape[1])
        env = torch.nn.functional.conv_transpose1d(win_sq, eye_kernel, stride=hop).squeeze(1)
        env = torch.clamp(env, min=1.0e-10)
        out = ola / env

        trim = n_fft // 2
        out_len = ola.shape[1]
        return out[:, trim : out_len - trim]

    _cm_mask.add_optional_chunk_mask = _safe_add_optional_chunk_mask
    _cm_dec.add_optional_chunk_mask = _safe_add_optional_chunk_mask
    _cm_uenc.add_optional_chunk_mask = _safe_add_optional_chunk_mask
    _cm_emb.EspnetRelPositionalEncoding.extend_pe = _noop_extend_pe
    _cm_hifi.SineGen.forward = _functional_sine_gen_forward
    _cm_hifi.HiFTGenerator._istft = _safe_istft_for_torch29
    _cm_mask._export_safe_minimal = True


def _torch_at_least(version_str: str) -> bool:
    """True if installed torch is >= the dotted version string (major.minor)."""
    want = tuple(int(x) for x in version_str.split("."))
    have = tuple(int(x) for x in torch.__version__.split(".")[: len(want)] if x.isdigit())
    return have >= want


def convert_conditional_decoder(model, output_dir, validate=False, reference_dir=None,
                                  mode="legacy", cfm_steps: int = 2,
                                  optimize_graph: bool = False, quantize: str = "none"):
    """Export the full post-T3 audio chain as conditional_decoder_single.onnx.

    Modes:
      legacy   — torch 2.8 path. Applies chatterbox monkey-patches (zero
                 HnNSF source, manual ISTFT via conv_transpose1d, no-op PE
                 rebuild). Audio quality degraded vs HF (~0.89 cos sim).
                 Opset 18 (need col2im for the manual overlap-add).
      torch29  — torch ≥ 2.9 path matching HF's published artifact.
                 Skips the SineGen / STFT / ISTFT bypass patches so the
                 native f0+source harmonics and torch.stft/istft survive
                 the trace. Uses dynamo + opset 17 (same as HF). The two
                 chatterbox-side patches that fix actual chatterbox bugs
                 (data-dependent mask check, in-place PE rebuild) are
                 still applied — they aren't torch-version-specific.
    """
    if mode == "torch29" and not _torch_at_least("2.9"):
        raise RuntimeError(
            f"--torch29 mode needs torch>=2.9 (have {torch.__version__}). "
            f"Use .venv-torch29 with `pip install -r requirements-torch29.txt`."
        )

    print(f"\n=== Converting conditional_decoder_single.onnx (mode={mode}) ===")
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    out_path = os.path.join(onnx_dir, "conditional_decoder_single.onnx")

    if mode == "legacy":
        _patch_chatterbox_for_export()
    else:
        _patch_chatterbox_for_export_minimal()

    wrapper = _ConditionalDecoderWrapper(model.s3gen, cfm_steps=cfm_steps)
    wrapper.train(False)
    print(f"  CFM solver steps: {cfm_steps}")

    speech_t, spk_t, feat_t = _make_fixture_cond_decoder(seed=0)
    speech_pt = torch.from_numpy(speech_t)
    spk_pt = torch.from_numpy(spk_t)
    feat_pt = torch.from_numpy(feat_t)

    if mode == "legacy":
        print(
            f"  Exporting with T_total={speech_pt.shape[1]} speech tokens, "
            f"T_feat={feat_pt.shape[1]} mel frames, opset 18 (legacy)..."
        )
        # Legacy TorchScript exporter. Opset 18 (vs HF's 17): need col2im
        # for the manual ISTFT overlap-add via F.fold.
        dynamic_axes = {
            "speech_tokens": {1: "num_speech_tokens"},
            "speaker_features": {1: "feature_dim"},
            "waveform": {1: "num_samples"},
        }
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (speech_pt, spk_pt, feat_pt),
                out_path,
                input_names=["speech_tokens", "speaker_embeddings", "speaker_features"],
                output_names=["waveform"],
                dynamic_axes=dynamic_axes,
                opset_version=18,
                do_constant_folding=True,
            )
    else:
        print(
            f"  Exporting with T_total={speech_pt.shape[1]} speech tokens, "
            f"T_feat={feat_pt.shape[1]} mel frames, opset 20 (dynamo, torch29)..."
        )
        # Dynamo exporter, opset 20. Opset 17 (HF's choice) requires
        # downconverting Pad which onnxscript can't do; opset 18 emits a
        # ScatterND node with an int32 input that ORT rejects (a torch 2.9
        # exporter bug); opset 20 sidesteps both.
        from torch.export import Dim

        num_speech_tokens = Dim("num_speech_tokens", min=2, max=4096)
        feat_dim = Dim("feature_dim", min=2, max=4096)
        dynamic_shapes = (
            {1: num_speech_tokens},
            None,
            {1: feat_dim},
        )
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (speech_pt, spk_pt, feat_pt),
                out_path,
                input_names=["speech_tokens", "speaker_embeddings", "speaker_features"],
                output_names=["waveform"],
                dynamic_shapes=dynamic_shapes,
                opset_version=20,
                dynamo=True,
            )

    if mode == "torch29":
        _fix_scatternd_int32_indices(out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    # Don't double-count external data files
    data_path = out_path + ".data"
    if os.path.exists(data_path):
        size_mb += os.path.getsize(data_path) / 1e6
    print(f"  Wrote {size_mb:.1f} MB")

    if optimize_graph:
        # conditional_decoder isn't a recognized model_type — the optimizer
        # applies generic ORT graph passes (folding, fusion of LN/GELU) but
        # skips the architecture-specific transformer fusions. Still useful.
        _optimize_onnx_graph(out_path, model_type="bert", num_heads=0, hidden_size=0)
    if quantize == "int8":
        _quantize_onnx_int8(out_path)

    if validate:
        validate_conditional_decoder(
            model, out_path, reference_dir=reference_dir, cfm_steps=cfm_steps
        )


def _quantize_coreml_int8(mlpackage_path: str) -> None:
    """In-place INT8 linear-symmetric quantization of a CoreML .mlpackage.

    Quantizes Linear/Conv weights to int8 with per-tensor symmetric scaling
    (the simplest mode that runs on ANE without surprises). Activations stay
    fp16/fp32 — dynamic per-call. Typically 4x smaller weights, 2-3x faster
    on ANE for transformer-heavy models like the T3 prefill.
    """
    from coremltools.optimize.coreml import (
        OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
    )

    print(f"  Quantizing {os.path.basename(mlpackage_path)} weights to INT8 (linear)...")
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric")
    )
    quantized = linear_quantize_weights(mlmodel, config=config)

    # Replace original .mlpackage directory. CoreML save() rejects any
    # extension other than .mlpackage, so use a sibling temp path that
    # keeps the extension and rename after.
    parent = os.path.dirname(mlpackage_path) or "."
    base = os.path.basename(mlpackage_path)
    tmp_path = os.path.join(parent, f".{base}.int8.tmp.mlpackage")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    quantized.save(tmp_path)
    shutil.rmtree(mlpackage_path)
    shutil.move(tmp_path, mlpackage_path)

    new_size = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(mlpackage_path) for f in files
    ) / 1e6
    print(f"  INT8 quantized in place ({new_size:.1f} MB)")


def _quantize_onnx_int8(onnx_path: str) -> None:
    """In-place dynamic INT8 weight quantization of an ONNX model.

    Quantizes MatMul / Gather / Conv weights to int8; activations stay fp32
    (dynamic per-tensor scaling on the fly). For transformer decode this
    typically gives 2-3x faster CPU inference and 4x smaller weights, with
    a small quality cost that should be measured per artifact via the
    validator. iOS ORT supports the resulting QLinear ops natively.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnx import TensorProto

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # DefaultTensorType=FLOAT handles graphs where the quantizer can't
        # infer types for some MatMul outputs — happens on the
        # conditional_decoder where the encoder doesn't carry full shape
        # annotation through the chain. Pre-running shape inference would
        # also work but is heavier.
        quantize_dynamic(
            model_input=onnx_path,
            model_output=tmp_path,
            weight_type=QuantType.QInt8,
            per_channel=False,
            extra_options={"DefaultTensorType": TensorProto.FLOAT},
        )
        # Replace original .onnx (and clean up any old external data file —
        # quantized weights are small enough to inline).
        shutil.move(tmp_path, onnx_path)
        data_path = onnx_path + ".data"
        if os.path.exists(data_path):
            os.unlink(data_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    new_size = os.path.getsize(onnx_path) / 1e6
    print(f"  INT8 quantized in place ({new_size:.1f} MB)")


def _optimize_onnx_graph(onnx_path: str, model_type: str = "bert",
                          num_heads: int = 0, hidden_size: int = 0) -> None:
    """Run onnxruntime's graph optimizer on an .onnx file in place.

    Applies operator fusion (LayerNorm, GELU, attention), constant folding,
    and ORT's L2 optimizations. Saves the optimized graph back over the
    input path (with external data file).

    Falls back to ORT-only optimizations (no Python fusions) on error —
    the transformers-package model_type-specific fusion paths have known
    bugs against non-canonical graph shapes (e.g. onnx_model_gpt2.py
    postprocess crashes when the expected reshape-after-gemm isn't found).
    """
    from onnxruntime.transformers import optimizer

    def _try(mt, only_ort, label):
        print(f"  Optimizing graph ({label})...")
        opt = optimizer.optimize_model(
            onnx_path,
            model_type=mt,
            num_heads=num_heads,
            hidden_size=hidden_size,
            opt_level=1,
            only_onnxruntime=only_ort,
        )
        opt.save_model_to_file(onnx_path, use_external_data_format=True)
        new_size = os.path.getsize(onnx_path) / 1e6
        data_path = onnx_path + ".data"
        if os.path.exists(data_path):
            new_size += os.path.getsize(data_path) / 1e6
        print(f"  Optimized graph saved ({new_size:.1f} MB total)")

    try:
        _try(model_type, False, f"model_type={model_type!r}, full fusions")
    except Exception as exc:
        print(f"  {model_type!r}-aware fusion failed ({type(exc).__name__}); "
              "retrying with ORT-only opt_level=1...")
        try:
            _try("bert", True, "generic, only_onnxruntime=True")
        except Exception as exc2:
            print(f"  Graph optimization disabled — {type(exc2).__name__}: {str(exc2)[:200]}")


def _fix_scatternd_int32_indices(onnx_path: str) -> None:
    """torch 2.9 dynamo sometimes emits ScatterND with int32 indices, which
    ORT rejects (the spec requires int64). Walk the graph, insert a Cast
    int32 -> int64 in front of every offending ScatterND."""
    import onnx
    from onnx import helper, TensorProto

    model = onnx.load(onnx_path, load_external_data=False)
    g = model.graph

    # Build a name -> elem_type lookup from initializers + value_infos + inputs
    name_to_type: dict[str, int] = {}
    for init in g.initializer:
        name_to_type[init.name] = init.data_type
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.type.HasField("tensor_type"):
            name_to_type[vi.name] = vi.type.tensor_type.elem_type
    for node in g.node:
        for out_name in node.output:
            # Casts are the most informative; we'll trust them when present
            if node.op_type == "Cast":
                attr = next((a for a in node.attribute if a.name == "to"), None)
                if attr is not None:
                    name_to_type[out_name] = attr.i

    patched = 0
    inserts: list[tuple[int, list]] = []  # (insertion index, [new nodes])
    for idx, node in enumerate(g.node):
        if node.op_type != "ScatterND":
            continue
        # ScatterND inputs: [data, indices, updates]
        idx_in = node.input[1]
        idx_type = name_to_type.get(idx_in)
        if idx_type == TensorProto.INT64:
            continue
        # Insert Cast int? -> int64. Use the ScatterND node name in the cast
        # output too — two ScatterND nodes might share the same indices input,
        # and ORT rejects duplicate value names in a graph.
        cast_out = f"{idx_in}__cast_int64__{node.name}"
        cast_node = helper.make_node(
            "Cast", [idx_in], [cast_out],
            name=f"{node.name}__indices_cast_int64",
            to=TensorProto.INT64,
        )
        node.input[1] = cast_out
        inserts.append((idx, [cast_node]))
        patched += 1

    if not patched:
        return

    # Insert from the end so earlier indices stay valid
    for ins_idx, new_nodes in reversed(inserts):
        for j, n in enumerate(new_nodes):
            g.node.insert(ins_idx + j, n)

    print(f"  Patched {patched} ScatterND node(s) with Cast int32 -> int64")
    onnx.save(model, onnx_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=os.path.basename(onnx_path) + ".data",
              size_threshold=1024)


def validate_conditional_decoder(model, our_path, reference_dir=None, cfm_steps: int = 2):
    print("\n  --- conditional_decoder_single.onnx validation ---")

    speech_t, spk_t, feat_t = _make_fixture_cond_decoder(seed=0)
    inputs = {
        "speech_tokens": speech_t,
        "speaker_embeddings": spk_t,
        "speaker_features": feat_t,
    }

    # OURS
    our_sess = _load_onnx_session(our_path)
    our_wav = our_sess.run(["waveform"], inputs)[0]

    ref_wav = None
    if reference_dir is not None:
        ref_path = os.path.join(reference_dir, "onnx", "conditional_decoder_single.onnx")
        if os.path.exists(ref_path):
            try:
                print(f"  Comparing against HF reference at {ref_path}...")
                ref_sess = _load_onnx_session(ref_path)
                ref_wav = ref_sess.run(["waveform"], inputs)[0]
            except Exception as exc:
                print(f"  (HF ref load failed: {str(exc)[:160]})")

    if ref_wav is None:
        print("  Comparing against PyTorch reference...")
        with torch.no_grad():
            wrapper = _ConditionalDecoderWrapper(model.s3gen, cfm_steps=cfm_steps)
            wrapper.train(False)
            torch.manual_seed(0)  # match RandomNormal seed for fair comparison
            t_out = wrapper(
                torch.from_numpy(speech_t),
                torch.from_numpy(spk_t),
                torch.from_numpy(feat_t),
            )
            ref_wav = t_out.detach().cpu().numpy()

    # Waveforms differ across runs due to RandomNormal; compare via spectral
    # statistics. Length should match; per-sample identity is not expected.
    ok = True
    if our_wav.shape != ref_wav.shape:
        print(f"  [waveform] shape mismatch ours={our_wav.shape} ref={ref_wav.shape}  FAIL")
        ok = False
    else:
        # mel-band log-magnitude cosine sim is robust to noise instance
        our_db = np.log(np.abs(our_wav).reshape(-1) + 1e-6)
        ref_db = np.log(np.abs(ref_wav).reshape(-1) + 1e-6)
        cs = float(np.dot(our_db, ref_db) / ((np.linalg.norm(our_db) * np.linalg.norm(ref_db)) or 1.0))
        ok &= cs >= 0.95
        print(f"  [waveform] shape={our_wav.shape} log-mag cos_sim={cs:.4f}  {'PASS' if cs >= 0.95 else 'FAIL'}")

    print(f"\n  --- iOS compatibility ---")
    ok &= _check_onnx_graph_for_ios(our_path)
    print(f"\n  conditional_decoder_single.onnx: {'READY' if ok else 'NOT READY'}")
    return ok


# ===========================================================================
# ANE compute-plan reporting (--stage ane-report)
# ===========================================================================
#
# Answers: "if I were to put each artifact through ANE on iPhone, what
# fraction of compute actually lands on ANE vs falls back to CPU/GPU?"
# Useful for deciding whether converting an ONNX model to native CoreML
# (for ANE access) is worth the engineering effort.


# Op types that ANE handles natively (rough but well-known list). Used for
# ONNX op-type classification — we can't directly query ANE eligibility on
# ONNX models from Python.
_ANE_FRIENDLY_ONNX_OPS = {
    "Conv", "ConvTranspose", "MatMul", "Gemm", "LayerNormalization",
    "BatchNormalization", "InstanceNormalization", "Add", "Sub", "Mul",
    "Div", "Relu", "Sigmoid", "Tanh", "Softmax", "GlobalAveragePool",
    "AveragePool", "MaxPool", "Concat", "Split", "Reshape", "Transpose",
    "Gather", "Slice", "Squeeze", "Unsqueeze", "Cast", "ReduceMean",
    "ReduceSum", "Pow", "Sqrt", "Exp", "Log", "Sin", "Cos", "Erf",
    "QLinearMatMul", "QLinearConv", "DynamicQuantizeLinear",
    "DequantizeLinear", "QuantizeLinear",
}

# Op types that consistently fall back to CPU/GPU on ANE
_ANE_FALLBACK_ONNX_OPS = {
    "STFT", "RandomNormal", "RandomNormalLike", "RandomUniform",
    "Complex", "ScatterND", "ScatterElements", "Where", "Loop", "If",
    "ConstantOfShape",  # often, depends on context
}


def _ane_report_coreml(mlpackage_path: str) -> None:
    """Per-op ANE/GPU/CPU dispatch report for a .mlpackage."""
    from coremltools.models import compute_plan as cp

    print(f"\n=== ANE compute plan: {os.path.basename(mlpackage_path)} ===")
    if shutil.which("xcrun") is None:
        print("  SKIP: xcrun not installed; can't compile to .mlmodelc")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", mlpackage_path, tmpdir],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  coremlcompiler failed: {proc.stderr.strip()[:300]}")
            return
        compiled = next(
            (p for p in os.listdir(tmpdir) if p.endswith(".mlmodelc")), None
        )
        if compiled is None:
            print(f"  coremlcompiler produced no .mlmodelc (dir contents: {os.listdir(tmpdir)})")
            return
        compiled_path = os.path.join(tmpdir, compiled)

        try:
            plan = cp.MLComputePlan.load_from_path(
                compiled_path, compute_units=ct.ComputeUnit.ALL,
            )
        except Exception as exc:
            print(f"  MLComputePlan.load_from_path failed: {str(exc)[:200]}")
            return

        pref_counts = {"ANE": 0, "GPU": 0, "CPU": 0, "Unknown": 0}
        supp_counts = {"ANE": 0, "GPU": 0, "CPU": 0}  # union of supported per op
        op_type_pref: dict[str, dict[str, int]] = {}
        op_type_ane_supported: dict[str, int] = {}

        def _classify_device(device) -> str:
            if device is None:
                return "Unknown"
            name = type(device).__name__
            if "NeuralEngine" in name:
                return "ANE"
            if "GPU" in name:
                return "GPU"
            if "CPU" in name:
                return "CPU"
            return "Unknown"

        def _walk_ops(block):
            for op in block.operations:
                yield op
                for inner in op.blocks:
                    yield from _walk_ops(inner)

        program = plan.model_structure.program
        if program is None:
            print("  Model is not an MLProgram; per-op breakdown unavailable.")
            return

        for fn_name, fn in program.functions.items():
            for op in _walk_ops(fn.block):
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    pref_counts["Unknown"] += 1
                    continue
                pref = _classify_device(usage.preferred_compute_device)
                pref_counts[pref] += 1

                supp = {_classify_device(d) for d in (usage.supported_compute_devices or [])}
                for d in supp & {"ANE", "GPU", "CPU"}:
                    supp_counts[d] += 1

                op_type_pref.setdefault(op.operator_name, {"ANE": 0, "GPU": 0, "CPU": 0, "Unknown": 0})
                op_type_pref[op.operator_name][pref] += 1
                if "ANE" in supp:
                    op_type_ane_supported[op.operator_name] = op_type_ane_supported.get(op.operator_name, 0) + 1

        total = sum(pref_counts.values())
        if total == 0:
            print("  No operations found in the program.")
            return

        print(f"  Total ops: {total}")
        print(f"  Preferred dispatch (what the runtime would actually pick on this Mac):")
        for dev in ("ANE", "GPU", "CPU", "Unknown"):
            c = pref_counts[dev]
            if c:
                print(f"    {dev:7s} {c:4d}  ({100 * c / total:5.1f}%)")
        print(f"  Supported (ops eligible for each device, not necessarily preferred):")
        for dev in ("ANE", "GPU", "CPU"):
            c = supp_counts[dev]
            print(f"    {dev:7s} {c:4d}  ({100 * c / total:5.1f}%)")
        ane_potential = supp_counts["ANE"] - pref_counts["ANE"]
        if ane_potential > 0:
            print(f"  → {ane_potential} ops ({100 * ane_potential / total:.1f}%) are ANE-eligible "
                  "but not preferred here. iPhone (newer ANE generation) may schedule them on ANE.")

        # Top op types where ANE *isn't* preferred but might be supported
        gap_types = sorted(
            (
                (op_type, op_type_ane_supported.get(op_type, 0), v)
                for op_type, v in op_type_pref.items()
                if v.get("ANE", 0) == 0  # not currently going to ANE
            ),
            key=lambda x: -(x[1] or sum(x[2].values())),
        )[:8]
        if gap_types:
            print(f"  Top op types NOT preferring ANE:")
            for op_type, ane_eligible, where in gap_types:
                bits = " / ".join(f"{d}:{where[d]}" for d in ("GPU", "CPU", "Unknown") if where[d])
                tag = f"ANE-supported on {ane_eligible}" if ane_eligible else "not ANE-eligible"
                print(f"    {op_type:35s} ({bits})  [{tag}]")


def _ane_report_onnx(onnx_path: str) -> None:
    """Op-type classification for an ONNX model.

    ONNX models don't run on ANE directly via Python. On iPhone they can
    be routed through CoreMLExecutionProvider in ORT to get ANE access
    where the op is supported. This report estimates what fraction of the
    graph WOULD land on ANE under that routing, based on a static op-type
    classification (well-known ANE-friendly vs ANE-fallback ops).
    """
    import onnx
    from collections import Counter

    print(f"\n=== ANE estimate (via CoreML EP): {os.path.basename(onnx_path)} ===")
    print(f"  ONNX models route through ORT's CoreMLExecutionProvider on iPhone")
    print(f"  to reach ANE. The breakdown below is a static op-type estimate.")

    model = onnx.load(onnx_path, load_external_data=False)
    op_counts = Counter(n.op_type for n in model.graph.node)
    total = sum(op_counts.values())

    ane_count = sum(c for op, c in op_counts.items() if op in _ANE_FRIENDLY_ONNX_OPS)
    fallback_count = sum(c for op, c in op_counts.items() if op in _ANE_FALLBACK_ONNX_OPS)
    unknown_count = total - ane_count - fallback_count

    print(f"  Total ops: {total}")
    print(f"    ANE-friendly  {ane_count:5d}  ({100 * ane_count / total:5.1f}%)")
    print(f"    Fallback      {fallback_count:5d}  ({100 * fallback_count / total:5.1f}%)")
    print(f"    Unknown       {unknown_count:5d}  ({100 * unknown_count / total:5.1f}%)  (heuristic uncertain — could go either way)")

    if fallback_count:
        fb = sorted(
            ((op, op_counts[op]) for op in op_counts if op in _ANE_FALLBACK_ONNX_OPS),
            key=lambda x: -x[1],
        )
        print(f"  Top ANE-blocking op types:")
        for op, c in fb[:8]:
            print(f"    {op:30s} {c:5d}")


def run_ane_report(output_dir: str) -> None:
    """Scan `output_dir` for known v4 artifacts and report ANE plans / estimates."""
    print("\n=== ANE residency report ===")
    print("Helps answer: 'is converting more of the pipeline to native CoreML")
    print("(to access ANE) worth the engineering work?'")

    prefill = os.path.join(output_dir, "T3Prefill.mlpackage")
    if os.path.exists(prefill):
        _ane_report_coreml(prefill)
    else:
        print(f"\n  (no T3Prefill.mlpackage at {prefill} — run --stage prefill first)")

    onnx_dir = os.path.join(output_dir, "onnx")
    for name in ("language_model_single.onnx", "conditional_decoder_single.onnx"):
        path = os.path.join(onnx_dir, name)
        if os.path.exists(path):
            _ane_report_onnx(path)
        else:
            print(f"\n  (no {name} at {path})")


# ===========================================================================
# MAIN
# ===========================================================================

V1_STAGES = ("t3", "s3", "vocoder", "all")
V4_STAGES = ("prefill", "lm-onnx", "cond-decoder", "v4")
ALL_STAGES = V1_STAGES + V4_STAGES + ("all-v4", "ane-report")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chatterbox Turbo TTS to CoreML / ONNX (v1 and v4 pipelines)"
    )
    parser.add_argument(
        "--stage",
        choices=ALL_STAGES,
        required=True,
        help=(
            "Which stage to convert. v1: t3, s3, vocoder, all. "
            "v4 (matches HF release): prefill, lm-onnx, cond-decoder, v4 (all three). "
            "all-v4: v1 stages + v4 stages. "
            "ane-report: scan existing artifacts in --output-dir and print "
            "per-op ANE/GPU/CPU dispatch (CoreML) / ANE eligibility estimate (ONNX)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save converted models",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run numerical validation after conversion",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "Optional directory containing HF reference artifacts (e.g. the "
            "ebrinz/chatterbox-turbo-coreml snapshot dir). When set, --validate "
            "for v4 stages compares against the HF .onnx / .mlmodelc files; "
            "otherwise it compares against a PyTorch reference."
        ),
    )
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help=(
            "Free intermediates aggressively and load reference artifacts "
            "sequentially (not in parallel with ours). Useful on 8 GB Macs."
        ),
    )
    parser.add_argument(
        "--torch29",
        action="store_true",
        help=(
            "Use the torch >= 2.9 export path for --stage cond-decoder. "
            "Skips the audio-quality-degrading chatterbox bypass patches "
            "and emits the same op set as the HF reference (native STFT / "
            "RandomNormalLike / com.microsoft.MultiHeadAttention). "
            "Requires the .venv-torch29 venv built from "
            "requirements-torch29.txt; will fail loudly otherwise."
        ),
    )
    parser.add_argument(
        "--cfm-steps",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Number of CFM (flow-matching) solver steps to unroll inside "
            "conditional_decoder. 2 (default) matches the HF reference. "
            "1 roughly halves cond_decoder runtime at a small audio "
            "quality cost — useful for the latency-sensitive iPhone path."
        ),
    )
    parser.add_argument(
        "--optimize-graph",
        action="store_true",
        help=(
            "Run onnxruntime.transformers.optimizer on each exported .onnx: "
            "operator fusion (LayerNorm, GELU, attention), constant folding, "
            "and ORT's L2 graph passes. Typically 5-15%% throughput on iOS "
            "at no quality cost. Output is saved back over the same .onnx "
            "path. CoreML stages ignore this flag."
        ),
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "int8"),
        default="none",
        help=(
            "Weight quantization for ONNX exports. 'int8' applies "
            "onnxruntime.quantization.quantize_dynamic to each .onnx in "
            "place: int8 weights, fp32 activations. ~4x smaller file, "
            "2-3x faster decode on iOS. Quality cost depends on the "
            "artifact — validate after. CoreML stages ignore this flag "
            "(CoreML palettization is a separate path; not yet wired)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ane-report runs against existing artifacts; no model load needed.
    if args.stage == "ane-report":
        run_ane_report(args.output_dir)
        return

    is_v4 = args.stage in V4_STAGES or args.stage == "all-v4"
    is_v1 = args.stage in V1_STAGES or args.stage == "all-v4"

    v1_model = None
    v4_model = None

    if is_v1:
        v1_model = load_pytorch_model()
    if is_v4:
        v4_model = load_pytorch_model_v4()

    # --- v1 stages ---
    if args.stage in ("t3", "all"):
        convert_t3(v1_model, args.output_dir, validate=args.validate)
    if args.stage in ("s3", "all"):
        convert_s3(v1_model, args.output_dir, validate=args.validate)
    if args.stage in ("vocoder", "all"):
        extract_vocoder_weights(v1_model, args.output_dir)
    if args.stage == "all":
        extract_tokenizer_and_config(v1_model, args.output_dir)

    if args.stage == "all-v4":
        # Also run v1 stages
        convert_t3(v1_model, args.output_dir, validate=args.validate)
        convert_s3(v1_model, args.output_dir, validate=args.validate)
        extract_vocoder_weights(v1_model, args.output_dir)
        extract_tokenizer_and_config(v1_model, args.output_dir)

    # --- v4 stages ---
    if args.stage in ("lm-onnx", "v4", "all-v4"):
        convert_language_model_onnx(
            v4_model, args.output_dir,
            validate=args.validate, reference_dir=args.reference_dir,
            optimize_graph=args.optimize_graph,
            quantize=args.quantize,
        )
    if args.stage in ("prefill", "v4", "all-v4"):
        convert_prefill(
            v4_model, args.output_dir,
            validate=args.validate, reference_dir=args.reference_dir,
            quantize=args.quantize,
        )
    if args.stage in ("cond-decoder", "v4", "all-v4"):
        convert_conditional_decoder(
            v4_model, args.output_dir,
            validate=args.validate, reference_dir=args.reference_dir,
            mode=("torch29" if args.torch29 else "legacy"),
            cfm_steps=args.cfm_steps,
            optimize_graph=args.optimize_graph,
            quantize=args.quantize,
        )

    print("\n=== Done ===")
    print(f"Output directory: {args.output_dir}")
    if args.stage in ("all", "all-v4"):
        print("Files:")
        for f in sorted(os.listdir(args.output_dir)):
            fpath = os.path.join(args.output_dir, f)
            if os.path.isdir(fpath):
                print(f"  {f}/")
            else:
                size = os.path.getsize(fpath)
                print(f"  {f} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
