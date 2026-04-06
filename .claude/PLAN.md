# Gemma 4 → CoreML Porting Plan

**Goal**: Port Google DeepMind's Gemma 4 model family to Apple CoreML for on-device inference on Apple Silicon.

---

## Model Family Overview

| Model | Effective Params | Total Params | Context | On-Device Target |
|---|---|---|---|---|
| E2B | 2.3B (5.1B w/ embeddings) | ~5B | 128k | M-series Mac, iPhone 16 Pro |
| E4B | 4.5B (8B w/ embeddings) | ~8B | 128k | M-series Mac only |
| 26B A4B | 4B active / 26B total (MoE) | 26B | 256k | Mac Studio / Mac Pro |
| 31B dense | 31B | 31B | 256k | Not practical on-device |

**Starting point: E2B text-only**, then add vision encoder.

---

## Key Architecture Challenges

### 1. Per-Layer Embeddings (PLE)
A second embedding table alongside the main residual stream. Produces small dedicated vectors injected into every decoder layer via a lightweight residual block after attention/FFN.
- No standard CoreML op maps to this
- Options: fold statically into weights at export (batch=1 decode), or implement as `MLCustomLayer`

### 2. Alternating Local / Global Attention
- Odd layers: sliding-window attention (512-token window for small models, 1024 for large)
- Even layers: full-context global attention
- Dual RoPE: standard RoPE for sliding layers, proportional RoPE for global layers
- Strategy: implement as masked `sdpa` with a static causal + window mask baked in at trace time

### 3. Shared KV Cache
- Last `num_kv_shared_layers` layers reuse K/V tensors from the last non-shared layer of the same attention type
- Maps cleanly to CoreML's new `ct.StateType` (iOS 18 / macOS 15+)
- At export: hard-wire shared layers to read from the anchor layer's state slot

### 4. Vision Encoder
- Learned 2D positions + multidimensional RoPE
- Variable aspect ratio support with configurable token budgets (70, 140, 280, 560, 1120 tokens)
- Strategy: fix to a single token budget (280) for v1 CoreML port; export as a separate model or second function in the `.mlpackage`

### 5. MoE Routing (26B A4B only)
- Expert dispatch is not a CoreML primitive
- Requires custom op — defer to Phase 2 or later

### 6. Audio Encoder (E2B/E4B only)
- USM-style Conformer architecture
- Complex, low ROI for v1 — defer

---

## Implementation Phases

### Phase 0 — Environment + Feasibility (1–2 days)
- [ ] Install `coremltools >= 8.x`, `transformers`, `torch >= 2.4`
- [ ] Load `google/gemma-4-e2b-it` from HuggingFace, run a forward pass
- [ ] Profile which ops `torch.export` traces cleanly vs. what breaks (PLE, shared KV, attention masking)
- [ ] Decision gate: identify custom ops needing `ct.PassPipeline` patches

### Phase 1 — Text-only E2B (1–2 weeks)
- [ ] Trace decoder with `torch.export.export()`, catching graph breaks
- [ ] Handle PLE: fold into layer weights at export time (static) or `MLCustomLayer`
- [ ] Implement sliding-window attention as masked `sdpa` with static masks
- [ ] Map KV cache to CoreML `StateType` for efficient decode loop
- [ ] Hard-wire shared KV layers at export
- [ ] Convert: `coremltools.convert(model, compute_precision=ct.precision.FLOAT16)`
- [ ] Profile on ANE via Xcode Instruments

### Phase 2 — Quantization (3–5 days)
- [ ] Apply `ct.optimize.coreml.palettize_weights` (INT4 lookup)
- [ ] Target: E2B fits in ~3 GB RAM at INT4
- [ ] Evaluate quality degradation vs. memory savings

### Phase 3 — Vision Encoder (1 week)
- [ ] Export SigLIP-style vision encoder separately
- [ ] Fix token budget to 280 for v1 (avoids dynamic shape complexity)
- [ ] Bundle as second function in `.mlpackage` using CoreML multi-function support
- [ ] Wire image preprocessing (resize, normalize, patch extraction)

### Phase 4 — Swift Integration (3–5 days)
- [ ] Xcode auto-generates Swift interface from `.mlpackage`
- [ ] Integrate tokenizer via `swift-transformers` SPM package
- [ ] Build: tokenizer → vision encoder (if image) → decoder loop → detokenizer
- [ ] Demo app or Swift Package target

---

## Hard Problems

| Challenge | Difficulty | Notes |
|---|---|---|
| PLE custom op | Medium | May fold statically if batch=1 decode |
| Shared KV cache | Medium | CoreML `StateType` is new but documented |
| 128k context window | Hard | Full KV at 128k is GBs; need chunked prefill strategy |
| MoE routing (26B A4B) | Hard | Expert dispatch not a CoreML primitive |
| Audio encoder | Deferred | USM Conformer is complex, skip for v1 |
| Variable aspect ratio | Medium | Fix token budget for CoreML; relax later with dynamic shapes |

---

## Tools

```
coremltools >= 8.0    # conversion + quantization
torch >= 2.4          # torch.export
transformers          # model loading from HuggingFace
Xcode 16+             # ANE profiling, Swift interface generation
swift-transformers    # tokenizer in Swift (SPM)
```

---

## References

- [Gemma 4 HuggingFace Blog](https://huggingface.co/blog/gemma4)
- [Apple CoreML Developer Page](https://developer.apple.com/machine-learning/core-ml/)
- [coremltools docs](https://coremltools.readme.io/)
- [CoreML Stateful Models (WWDC 2024)](https://developer.apple.com/videos/play/wwdc2024/10161/)
- [swift-transformers](https://github.com/huggingface/swift-transformers)
