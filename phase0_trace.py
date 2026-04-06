"""
Phase 0 — Feasibility: torch.export trace of Gemma 4 E2B
=========================================================
Goal: identify which ops export cleanly and which break (PLE, shared KV,
alternating attention). Run this after `hf auth login` and accepting
Gemma 4 terms at https://huggingface.co/google/gemma-4-e2b-it

Usage:
    .venv/bin/python phase0_trace.py

Outputs:
    - phase0_trace_report.txt  — export graph summary + broken ops list
"""

import json
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE = "cpu"  # keep on CPU for tracing; MPS complicates torch.export
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print(f"[1/4] Loading tokenizer + model ({MODEL_ID}) ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    low_cpu_mem_usage=True,
)
model.train(False)  # inference mode — equivalent to model.eval()
print(f"      Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"      Config:\n{json.dumps(model.config.to_dict(), indent=2, default=str)[:2000]}")

# ---------------------------------------------------------------------------
# 2. Print architecture details we care about
# ---------------------------------------------------------------------------
cfg = model.config
print("\n[2/4] Architecture features of interest:")
for attr in [
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "intermediate_size",
    "sliding_window",
    "attention_window_size",
    "num_kv_shared_layers",
    "use_per_layer_embeddings",
    "vision_config",
    "audio_config",
]:
    val = getattr(cfg, attr, "NOT FOUND")
    print(f"  {attr}: {val}")

# ---------------------------------------------------------------------------
# 3. Attempt torch.export
# ---------------------------------------------------------------------------
print("\n[3/4] Attempting torch.export ...")

# Minimal dummy inputs — batch=1, seq=16
dummy_input = tokenizer("Hello, world!", return_tensors="pt")
input_ids = dummy_input["input_ids"]  # shape: [1, seq]
attention_mask = dummy_input["attention_mask"]

export_error = None
export_tb = None
export_graph = None

try:
    with torch.no_grad():
        exported = torch.export.export(
            model,
            args=(input_ids,),
            kwargs={"attention_mask": attention_mask},
            strict=False,
        )
    export_graph = exported.graph_module
    print("  SUCCESS — model exported cleanly")
except Exception as e:
    export_error = f"{type(e).__name__}: {str(e)}"
    export_tb = traceback.format_exc()
    print(f"  FAILED (expected): {export_error[:300]}")

# ---------------------------------------------------------------------------
# 4. Write report
# ---------------------------------------------------------------------------
print("\n[4/4] Writing phase0_trace_report.txt ...")
with open("phase0_trace_report.txt", "w") as f:
    f.write("# Phase 0 Trace Report — Gemma 4 E2B\n\n")
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"dtype: {DTYPE}\n\n")

    f.write("## Config (relevant fields)\n")
    for attr in [
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "hidden_size", "intermediate_size", "sliding_window",
        "attention_window_size", "num_kv_shared_layers",
        "use_per_layer_embeddings",
    ]:
        f.write(f"  {attr}: {getattr(cfg, attr, 'NOT FOUND')}\n")

    if export_graph is not None:
        f.write("\n## torch.export result: SUCCESS\n")
        ops = set()
        for node in export_graph.graph.nodes:
            if node.op == "call_function":
                ops.add(str(node.target))
        f.write(f"\nUnique ops ({len(ops)}):\n")
        for op in sorted(ops):
            f.write(f"  {op}\n")
    else:
        f.write("\n## torch.export result: FAILED\n")
        f.write("\n### Error:\n")
        f.write(export_error or "unknown")
        f.write("\n\n### Full traceback:\n")
        f.write(export_tb or "")

print("Done. See phase0_trace_report.txt")
