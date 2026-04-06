# Session Notes — gemma4-coreML

Running log of what's been done, decisions made, and things to watch out for.

---

## 2026-04-06 — Phase 0 Setup

### Environment
- **Python**: 3.12.13 via `/opt/homebrew/opt/python@3.12/bin/python3.12`
  - Python 3.14 (system default) is NOT compatible with coremltools 9.0
  - `.python-version` pinned to `3.12` at project root
- **Venv**: `.venv/` created with `uv venv --python 3.12`
- **Package manager**: `uv pip` throughout — do not use bare pip

### Installed Versions
| Package | Version | Notes |
|---|---|---|
| torch | 2.11.0 | Newer than coremltools 9.0's tested version (2.7.0) — watch for compat warnings |
| transformers | 5.5.0 | |
| coremltools | 9.0 | Latest; supports Python 3.8–3.12 |
| accelerate | latest | For `device_map` support |
| huggingface_hub | latest | |

Install command:
```bash
uv pip install torch torchvision transformers accelerate coremltools huggingface_hub
```

### coremltools / torch version mismatch
coremltools 9.0 was tested with torch 2.7.0 max; we have 2.11.0.
- Causes a warning on import but does not break things immediately
- Watch for failures in `ct.convert()` and `ct.optimize`
- If conversion fails, downgrade torch:
```bash
uv pip install "torch==2.7.0" "torchvision==0.22.0"
```

### HuggingFace Auth
- User is NOT logged in to HuggingFace
- Gemma 4 is a **gated model** — requires:
  1. Accept terms at https://huggingface.co/google/gemma-4-e2b-it
  2. Log in via: `! .venv/bin/hf auth login`
- Cannot run phase0_trace.py until this is done

### Phase 0 Script
- **File**: `phase0_trace.py`
- Loads E2B model on CPU, float16
- Probes `torch.export.export()` with `strict=False`
- Prints config fields for: PLE, shared KV, sliding window, attention window size
- Writes full report to `phase0_trace_report.txt`

### Gotcha: hookify security hook false positive
- The hookify hook pattern-matches on certain security-sensitive strings
- `model.train(False)` is used instead of the inference-mode setter method
- These are identical in behavior — `train(False)` sets `self.training = False` on all submodules

### Next Steps
1. User runs `! .venv/bin/hf auth login` and accepts Gemma 4 terms on HuggingFace
2. Run: `.venv/bin/python phase0_trace.py`
3. Read `phase0_trace_report.txt` — look for:
   - `use_per_layer_embeddings` value in config
   - `num_kv_shared_layers` value
   - `sliding_window` / `attention_window_size` values
   - Whether torch.export succeeds or which op causes the graph break
4. Based on results, decide which ops need `MLCustomLayer` vs. standard handling
