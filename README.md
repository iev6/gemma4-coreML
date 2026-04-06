# gemma4-coreML

Porting Google DeepMind's [Gemma 4](https://huggingface.co/blog/gemma4) model family to Apple CoreML for on-device inference on Apple Silicon.

## Status

Work in progress — starting with text-only E2B (2.3B effective parameters).

## Plan

See [`.claude/PLAN.md`](.claude/PLAN.md) for the full porting plan, architecture challenges, and implementation phases.

## Models

| Model | Effective Params | Target Hardware |
|---|---|---|
| E2B | 2.3B | M-series Mac, iPhone 16 Pro |
| E4B | 4.5B | M-series Mac |
| 26B A4B (MoE) | 4B active | Mac Studio / Mac Pro |

## Key Challenges

- Per-Layer Embeddings (PLE) — novel architecture requiring custom ops
- Alternating local/global attention with dual RoPE
- Shared KV Cache — maps to CoreML's `StateType` API
- Vision encoder with variable aspect ratio support

## License

Apache 2.0 (matching Gemma 4's license)
