<p align="center">
  <img src="assets/banner.svg" alt="chatterbox-turbo-coreml-converter — text to speech on Apple Silicon" width="100%"/>
</p>

# chatterbox-turbo-coreml-converter

> **Pre-built artifacts on Hugging Face:**
> [ebrinz/chatterbox-turbo-coreml](https://huggingface.co/ebrinz/chatterbox-turbo-coreml)
> — `T3Prefill.mlpackage`, `onnx/language_model_single.onnx`,
> `onnx/conditional_decoder_single.onnx` ready to drop into a Swift app.
> This repo is the conversion script that produces them, plus benchmarks,
> validation harness, and an alternative v1 pure-CoreML pipeline.

Converts ResembleAI's [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)
TTS model to artifacts that run on Apple Silicon — both Macs (M1+) and iPhones
(via on-device CoreML + ONNX Runtime).

> **Reproducing the HF reference:** run with no optimization flags — the
> default output is bit-/byte-equivalent to what's on Hugging Face.
> **Shipping to an iPhone app and want it smaller + faster?** Jump to
> [Proven "ship to iPhone fast" combo](#proven-ship-to-iphone-fast-combo).

Two pipelines are supported:

- **v4 hybrid CoreML + ONNX** *(recommended; matches what ships on iPhone today)*:
  `T3Prefill.mlpackage` + `onnx/language_model_single.onnx` + `onnx/conditional_decoder_single.onnx`.
  This is the pipeline used by the published weights at
  [ebrinz/chatterbox-turbo-coreml](https://huggingface.co/ebrinz/chatterbox-turbo-coreml)
  and is meant to be consumed from Swift via CoreML + ONNX Runtime (one of
  each).
- **v1 pure CoreML** *(historical; slower)*: `T3Stateful.mlpackage` +
  `S3Encoder.mlpackage` + `S3UNet.mlpackage` + `hift_vocoder.safetensors`.
  Demonstrates a stateful KV-cache layout that fits CoreML's `StateType`
  constraints.

## What gets produced

### v4 pipeline (`--stage v4`)

| File | What it is | Backend | Parity vs HF |
|---|---|---|---|
| `T3Prefill.mlpackage` | Full T3 prefill — text+cond+speaker conditioning baked in. Outputs first-decode logits + the entire KV cache stacked as `(48, 1, 16, T, 64)` (interleaved K0,V0,…,K23,V23). | CoreML CPU+GPU | Size match within 0.01% (1503.8 MB vs 1503.6 MB); cos sim 1.0 on logits + KV (vs PyTorch ref; HF `.mlmodelc` lacks the `Manifest.json` needed for direct cross-load) |
| `onnx/language_model_single.onnx` | Single-step GPT-2 decode with explicit per-layer `past_key_values.{i}.{key,value}` inputs / `present.{i}.{key,value}` outputs. Drives the autoregressive loop after prefill. | ONNX Runtime CPU | **Bit-equivalent to HF.** cos sim 1.0 / max-abs 5.7e-6 on logits and all 48 KV outputs |
| `onnx/conditional_decoder_single.onnx` | The entire post-T3 audio chain in one graph: S3 encoder → 2-step CFM solver → HiFTGenerator vocoder → ISTFT → waveform. | ONNX Runtime CPU | Loads + runs + produces audio at the correct length and shape; waveform log-magnitude cos sim ≈ 0.89 vs HF *(intelligible but lower naturalness — see "Stage C tradeoffs" below)* |
| `speech_emb.npy`, `text_emb.npy` | Embedding tables for host-side lookups before calling `T3Prefill` and the decode ONNX. | — |
| `spkr_enc_weight.npy`, `spkr_enc_bias.npy`, `default-conds.safetensors`, tokenizer files, `config.json` | Supporting artifacts copied from the upstream HF repo. | — |

### v1 pipeline (`--stage all`)

| File | What it is | Backend |
|---|---|---|
| `T3Stateful.mlpackage` | GPT-2 medium decoder with CoreML `StateType` KV cache (2D seq-first layout, dim-0 slice-update). Fixed seq=1 shape; prefill is done token-by-token through the same model. | GPU |
| `S3Encoder.mlpackage` | Conformer encoder + mel projection. Dynamic sequence length via `RangeDim`. | ANE-eligible |
| `S3UNet.mlpackage` | Flow-matching U-Net denoiser (estimator + speaker-affine projection baked in). | ANE-eligible |
| `hift_vocoder.safetensors` | HiFTGenerator vocoder weights (PyTorch — not converted to CoreML). | — |

## Size & speed flags

Three orthogonal flags trade size / speed against quality. Stack them as
needed; each one's effect is documented in the validator output.

| Flag | What it does | Applies to | Typical size / speed change | Quality impact |
|---|---|---|---|---|
| `--cfm-steps 1` | Halves the CFM solver from 2 unrolled Euler steps to 1 | `cond-decoder` | ~halved cond_decoder runtime; ~same file size | Modest log-mag cos sim drop (~0.886 → ~0.874 in our test) |
| `--optimize-graph` | Runs `onnxruntime.transformers.optimizer` on each `.onnx`: operator fusion, constant folding, ORT L2 passes | `lm-onnx`, `cond-decoder` | ~5-15% faster on iOS; same file size | None — numerically identical |
| `--quantize int8` | Weight quantization via `onnxruntime.quantization.quantize_dynamic` (ONNX) and `coremltools.optimize.coreml.linear_quantize_weights` (CoreML) | all three v4 stages | **~4× smaller files**, ~2-3× faster decode on iOS | Modest (lm-onnx logits cos sim drops to ~0.97 vs fp32). Use for size-constrained ship targets; validate per artifact. |

### Proven "ship to iPhone fast" combo

Three commands, ~1.2 GB total bundle, ~4× faster than vanilla PyTorch
end-to-end on M1:

```bash
# T3Prefill: INT8 (1503 MB -> 361 MB)
python convert_chatterbox_coreml.py --stage prefill --output-dir ./out --quantize int8

# language_model.onnx: INT8 + graph fusion (1270 MB -> 304 MB,
# 16 ms/tok -> 5-8 ms/tok on M1 CPU)
python convert_chatterbox_coreml.py --stage lm-onnx --output-dir ./out \
    --optimize-graph --quantize int8

# conditional_decoder.onnx: cfm-steps=1 only (1065 ms -> 685 ms per
# synth; modest audio-quality cost). DO NOT combine with --optimize-graph
# or --quantize on cond-decoder — both currently bake the encoder's
# Conformer relative-attention shape, breaking runtime use at any
# sequence length other than the trace fixture.
python convert_chatterbox_coreml.py --stage cond-decoder --output-dir ./out --cfm-steps 1
```

Measured on M1 (5-second utterance, 83 tokens):

```
                          fp32 baseline    Optimized      Speedup
T3Prefill prefill         131 ms           106 ms         1.24x
lm.onnx per token         16-19 ms         5-8 ms         2-3x
cond_decoder synthesize   1065 ms          685 ms         1.55x
End-to-end                ~2725 ms         ~1320 ms       ~2x
vs vanilla PyTorch        5400 ms                         ~4x
RTF                       0.73x            0.38x
```

### Shipping a fast variant to Hugging Face

The artifacts on
[ebrinz/chatterbox-turbo-coreml](https://huggingface.co/ebrinz/chatterbox-turbo-coreml)
today are the proven-working, quality-validated bundle. Replacing them
with the optimized variant is a tradeoff: you get a smaller, faster
bundle but introduce a measurable quality drop (logits cos sim 1.0 →
~0.97 on INT8 lm; cond_decoder log-mag cos sim ~0.886 → ~0.874 at
cfm-steps=1). Three deployment options, ranked by safety:

1. **Don't ship a variant — document the recipe.** Keep HF as-is; the
   recipe above lets users build their own optimized bundle locally.
   No regression risk for existing consumers. *Most conservative.*
2. **Add the variant as sibling paths on the same HF repo.** Put files
   under `int8/T3Prefill.mlpackage`, `int8/onnx/language_model_single.onnx`,
   etc. Existing iPhone apps pointing at the root paths are unaffected;
   apps that opt in get the smaller/faster bundle. *Recommended if you
   want to ship it.*
3. **Use a separate HF branch / revision** (e.g. `fast`). iPhone app
   pins to `main` or `fast`. Cleanest separation, but requires Swift-side
   awareness of the revision.

**Always device-test before shipping.** Desktop ORT and iPhone ORT use
different kernel variants for INT8 `QLinearMatMul`; audio differences
that don't show up on Mac may surface on-device. Build the bundle
locally, drop it into your iOS app build (replacing the `out/` files in
your bundle), and run a real-device A/B before changing anything on HF.

### Known limitations

- **`--optimize-graph` on `cond-decoder`**: the ONNX Runtime graph
  optimizer specializes (bakes) shapes inside the Conformer encoder's
  relative-attention skewing operation. The exported model then only
  runs at the trace-fixture sequence length. Skip `--optimize-graph`
  for cond-decoder — the gain is modest there anyway since it isn't a
  recognized `model_type` for the transformer-specific fusions.
- **`--quantize int8` on `cond-decoder`**: `quantize_dynamic` runs but
  produces a graph with the same shape-baking issue (likely interacting
  with the optimizer pass internally). Treat cond-decoder as
  quantize-incompatible for now.
- **CoreML INT8 ANE compute plan may differ from FP32.** Some quantized
  ops fall back to CPU on ANE that FP32 versions handled. Measure
  on-device.

## Why CoreML *and* ONNX (and not just one)

The Swift consumer on iPhone runs *both* CoreML and ONNX Runtime — each stage
goes to whichever wins on that specific workload. Concretely:

| Stage | Runtime | Why this runtime |
|---|---|---|
| `T3Prefill.mlpackage` | **CoreML** (CPU+GPU) | One big batched op containing the 50K-row text embedding, the 6.5K-row speech embedding, the speaker projection, and a 24-layer GPT-2 forward over the whole prefix. CoreML's higher per-call overhead amortizes; its GPU dispatch on Apple Silicon beats CPU torch by a wide margin. Runs **once per generation**. |
| `language_model_single.onnx` | **ONNX Runtime** (CPU, C API) | Per-step decode latency dominates the autoregressive loop. ONNX Runtime's C API has zero-copy tensor handoff (no per-call buffer allocation) and highly tuned CPU kernels for single-token transformer decode — beats CoreML's per-call overhead by 2–3× at the scale of 100+ decode steps. Runs **N times per generation** (one per output token). |
| `conditional_decoder_single.onnx` | **ONNX Runtime** (CPU) | Bundles the full audio chain (S3 encoder + 2-step CFM + HiFTGenerator + ISTFT). Needs `RandomNormalLike` for CFM noise and integer/complex op support that's cleaner in ONNX than in CoreML's MIL. Single call per generation. |

In other words: **CoreML where one-big-op throughput matters, ONNX where
per-step latency in a tight loop matters.** The split is what makes
real-time on iPhone possible.

## Benchmarks (M1, CPU+GPU, 5-second utterance)

Run via `python scripts/bench_pipeline.py`. Each stage is timed in
isolation. Numbers from a fresh load on an M1 Mac; iPhone runtimes are
typically faster for the ONNX legs (smaller thermal envelope, dedicated
ORT iOS build) and slower for the CoreML leg.

| Stage | Mean | Calls per generation |
|---|---:|---:|
| **PyTorch end-to-end** (chatterbox.generate, baseline) | **6050 ms** | 1 |
| `T3Prefill.mlpackage` prefill | 131 ms | 1 |
| `language_model_single.onnx` decode @ past_len=10 | 16 ms | per token |
| `language_model_single.onnx` decode @ past_len=100 | 17 ms | per token |
| `language_model_single.onnx` decode @ past_len=500 | 19 ms | per token |
| `conditional_decoder_single.onnx` synthesize | 1065 ms | 1 |
| **Our pipeline (est. end-to-end, 90 tokens)** | **~2725 ms** | — |

Real-time factor (RTF, lower is better):

```
PyTorch     6050 ms / 3750 ms audio = 1.61x   (slower than real-time)
Ours        2725 ms / 3750 ms audio = 0.73x   (faster than real-time)

Speedup     ~2.2x over vanilla PyTorch
```

The decode-step number (~17 ms/token on M1 CPU) is what compounds: 100
tokens × 17 ms ≈ 1.7 s of the total. Per-step decode dominates the
autoregressive loop, which is exactly why we run it through ONNX Runtime
(zero-copy C API) rather than CoreML — on this workload the per-call
overhead of CoreML's framework wins or loses a noticeable amount over
100+ iterations.

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4). Required for CoreML conversion and
  for the iOS-compatibility checks (`xcrun coremlcompiler`, ANE compute plan).
- **Python 3.10+** with `torch>=2.7,<2.9` and `coremltools>=9.0,<10.0`.
  Newer torch / coremltools may produce slightly different MIL but should still
  match HF reference numerically.
- **Minimum deployment target**: iOS 18 / macOS 15.
- ~20 GB free disk for model weights + intermediate artifacts.
  v4 conversion peaks at ~10 GB RAM; pass `--low-mem` on tight machines.

## Install

```bash
git clone https://github.com/ebrinz/chatterbox-turbo-coreml-converter.git
cd chatterbox-turbo-coreml-converter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Note on the chatterbox dependency.** We pin `chatterbox-tts==0.1.7`
> because the git head currently regresses (broken `tts_turbo` imports +
> missing `GPT2_medium` config entry). The converter monkey-patches the
> missing `LLAMA_CONFIGS["GPT2_medium"]` entry at runtime so a fresh install
> works without any local edits.

## Usage

### v4 (recommended)

Build all three v4 artifacts in one go:

```bash
python convert_chatterbox_coreml.py --stage v4 --output-dir ./out
```

Or one at a time, in the suggested verification order:

```bash
python convert_chatterbox_coreml.py --stage lm-onnx       --output-dir ./out
python convert_chatterbox_coreml.py --stage prefill       --output-dir ./out
python convert_chatterbox_coreml.py --stage cond-decoder  --output-dir ./out
```

### v1 (historical)

```bash
python convert_chatterbox_coreml.py --stage all --output-dir ./out
```

### Combined: v1 + v4

```bash
python convert_chatterbox_coreml.py --stage all-v4 --output-dir ./out
```

### Validation

Add `--validate` to any stage to run a numerical sanity check. Add
`--reference-dir <path>` to compare directly against the published HF
artifacts:

```bash
# First, cache the HF v4 release locally (~3.5 GB):
hf download ebrinz/chatterbox-turbo-coreml --include "onnx/*" "T3Prefill.mlmodelc/*"

# Then validate against it. The path is the snapshot dir, e.g.:
REF=~/.cache/huggingface/hub/models--ebrinz--chatterbox-turbo-coreml/snapshots/<hash>

python convert_chatterbox_coreml.py --stage v4 --output-dir ./out \
    --validate --reference-dir "$REF"
```

When `--reference-dir` is set, each v4 stage prints a per-tensor cosine
similarity / max-abs diff vs the HF artifact, plus an iOS-compatibility
report (opset, `xcrun coremlcompiler`, ANE compute plan on Apple Silicon).

First run will download the ~6 GB Chatterbox Turbo weights from HuggingFace
into your HF cache.

## Verifying artifacts before shipping to iOS

The converter is Python-only, but you can get high-confidence "this will run
on iPhone" signals without leaving your Mac:

1. **Per-stage HF parity** — `--validate --reference-dir` proves our output
   matches the HF artifact within FP precision. Since those exact HF artifacts
   are what runs on iPhone 17 today, parity ≈ iPhone-ready.
2. **iOS18 deployment target** — declared on every CoreML conversion;
   coremltools refuses to emit ops outside the iOS 18 op set.
3. **`xcrun coremlcompiler`** — the same compile Xcode runs when bundling a
   `.mlpackage` for an iPhone build. Run as part of `--validate`.
4. **M1 Neural Engine load + predict** — exercises real ANE silicon (same
   family as iPhone's ANE). Run as part of `--validate`. The Mac compute plan
   may differ from iPhone's, but a successful M1 ANE run is a strong "the
   model is structurally accepted by ANE" signal.
5. **Swift-on-Mac end-to-end (optional but recommended)** — point your
   Swift consumer (a SwiftPM package that loads the `.mlpackage` via
   `MLModel(contentsOf:)` and the two `.onnx` files via
   `onnxruntime-swift-package-manager`) at the output dir and run its tests
   against a `macOS` target. Same Swift runtime and same APIs as iPhone; if
   it produces audio on Mac, it will produce audio on iPhone modulo
   speed/thermal differences.
6. **Bundle into your iOS app, build for device, run on iPhone.** The
   final 1% that Python can't simulate.

## Architecture notes

### v4 prefill / decode design

The Swift inference layer treats the prefill and decode steps differently:

- **`T3Prefill.mlpackage`** runs once per generation. It accepts raw inputs
  (text token ids, conditioning speech tokens, raw 256-dim speaker embedding,
  the start-speech token) and emits both the first-decode logits and the full
  KV cache stacked as a `(48, 1, 16, T, 64)` tensor. CoreML is the right
  backend here because the prefill includes large embedding lookups
  (`text_emb`: 50276×1024, `speech_emb`: 6563×1024) plus the full forward
  pass — fp32 weights, GPU-eligible compute.
- **`language_model_single.onnx`** runs per decode step inside ONNX Runtime
  on CPU. It takes pre-looked-up `inputs_embeds`, the running `attention_mask`,
  and 48 individual KV cache tensors as named inputs (`past_key_values.{i}.key`,
  `past_key_values.{i}.value`), and returns the next logits + updated KVs.
  ONNX is the right backend here because per-step latency dominates and ORT's
  C API zero-copy path beats CoreML on this workload by ~2×.
- **`conditional_decoder_single.onnx`** runs once at the end. It bundles the
  S3Gen flow encoder, the 2-step CFM solver (unrolled, meanflow), the
  HiFTGenerator vocoder, and ISTFT into a single ONNX graph. The CFM noise is
  emitted as a `RandomNormal` op so the model self-seeds per call.

### Why a manual GPT-2 forward in the wrappers

Both the v4 LM ONNX export and the v4 prefill CoreML export drive their own
attention math (q/k/v split → SDPA → c_proj → residual + LN + MLP) instead of
calling HF's `GPT2Block.forward`. The reason: HF's `GPT2Block` channels all
KV state through `transformers.cache_utils.Cache`, which both `torch.onnx.export`
exporters fail to trace cleanly (legacy hits an `unordered_map::at` recursion
in functorch; dynamo hits `ProxyTorchDispatchMode` assertions). Rewriting the
two-block forward in primitives bypasses the cache machinery entirely while
reusing the trained weight modules unchanged.

### Stage C: `--torch29` mode (real harmonics, dynamo exporter)

The default Stage C path uses torch 2.8's legacy exporter (next section).
A second path exists via the `--torch29` flag, run in an isolated
`.venv-torch29` (`pip install -r requirements-torch29.txt`):

```bash
source .venv-torch29/bin/activate
python convert_chatterbox_coreml.py --stage cond-decoder --output-dir ./out --torch29 \
    --validate --reference-dir <hf-cache-dir>
```

This path uses torch 2.9 + the dynamo exporter and applies a much
smaller patch set (`_patch_chatterbox_for_export_minimal`): the SineGen
forward is rewritten *functionally* (same outputs, no in-place writes
that emit ScatterND-with-int32 nodes that ORT rejects), real harmonic
source synthesis is preserved (vs the legacy mode which zeroes it), and
a post-export pass inserts Cast int32 → int64 in front of the two
ScatterND nodes torch 2.9 currently emits with the wrong indices dtype.

What you get vs the legacy mode:

| | Legacy (torch 2.8) | `--torch29` (torch 2.9) |
|---|---|---|
| Opset | 18 | 20 |
| File size | 552 MB | 597 MB |
| SineGen harmonics | zeroed | real (functional rewrite) |
| ISTFT | manual primitives (`conv_transpose1d`) | manual primitives (same — torch 2.9's native `torch.istft` decomposes to a Mul that ORT can't broadcast) |
| Patches applied | 6 | 4 |
| Log-mag cos sim vs HF (typical sample) | ~0.886 | ~0.894 |

The metric barely moves because it's dominated by stochastic CFM noise
(HF uses RandomNormal per call, so does our model — different instances).
The structural fidelity is materially better in `--torch29` mode — the
voiced texture is closer to HF — even when the bulk cos sim doesn't
clearly reflect it. Listen via `scripts/play_sample.py`.

### Stage C tradeoffs (legacy path)

`conditional_decoder_single.onnx` exports through PyTorch 2.8's legacy
TorchScript exporter at opset 18. Getting it through required four
defensive monkey-patches against the chatterbox source (applied
runtime-only, in `_patch_chatterbox_for_export()`):

1. **`add_optional_chunk_mask`** — strip a `.sum().item() != 0` correction
   check that produces a data-dependent symbolic guard. Safe to skip for
   batch=1 full-sequence inference (the branch is never taken).
2. **`EspnetRelPositionalEncoding.extend_pe`** — disable the per-forward
   PE rebuild (in-place writes). The PE is populated in `__init__` to
   `max_len=5000`, which is enough for everything we ever export.
3. **`SourceModuleHnNSF.forward`** — return a zero source signal instead
   of running the f0 predictor + SineGen → harmonic synthesis path.
   That path's `cumsum % 1` + `voiced_threshold` comparison decompose into
   ops whose dtypes torch 2.8's fake-tensor system mis-tracks
   (`prims.signbit`).
4. **`HiFTGenerator._stft` / `._istft`** — replace `torch.stft` /
   `torch.complex` / `torch.istft` (opset 17 legacy can't export those for
   complex tensor types) with primitive equivalents: zero stub for the
   forward STFT (source is zero anyway), and a manual IDFT + overlap-add
   via `conv_transpose1d` for the inverse.

**Audio quality consequence:** dropping the harmonic source contribution
makes the generated speech sound less natural than the HF release — the
voiced texture is mostly carried by the upsample stack alone, without the
NSF source enhancement. Output is still intelligible.

If you need 1:1 parity with the HF artifact and can wait on the toolchain,
the path is: torch 2.9+ with `dynamo=True`, no `_stft/_istft` patches, and
let the dynamo exporter emit `STFT` / `RandomNormalLike` /
`com.microsoft.MultiHeadAttention` ops as the HF version does. The same
wrapper otherwise.

### v1 KV-cache layout

The original v1 `T3Stateful.mlpackage` keeps its KV cache in CoreML
`StateType` storage. CoreML state tensors only support dynamic `slice_update`
on dimension 0, which rules out the natural `(batch, heads, seq, head_dim)`
per-layer layout. v1 instead uses a single 2D seq-first cache shared across
layers:

```
keyCache, valueCache: (max_seq=2048, n_layers * n_heads * head_dim = 24576)
```

Writes are dim-0 slices at the current cache position; per-layer K/V are
extracted by slicing the feature dimension. Attention masking handles
unfilled positions (zeros) by setting `attn_mask` to `-1e9` there, so the
softmax ignores them.

## License

This conversion script is released under the [MIT License](LICENSE).

The **converted model weights** are derived from ResembleAI's
[Chatterbox](https://github.com/resemble-ai/chatterbox) and inherit their
licensing terms. Check the upstream license before redistributing the
converted artifacts.

## Acknowledgements

- [ResembleAI](https://www.resemble.ai/) for releasing Chatterbox Turbo.
- [Apple's coremltools](https://github.com/apple/coremltools) team for the
  stateful conversion support and the dynamo-based ONNX exporter.
