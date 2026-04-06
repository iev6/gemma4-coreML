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
import sys
import time
import traceback
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE = "cpu"  # keep on CPU for tracing; MPS complicates torch.export
DTYPE = torch.float16


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def step(n, total, label):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] ── Step {n}/{total}: {label} ──", flush=True)


# ---------------------------------------------------------------------------
# 1. Load tokenizer
# ---------------------------------------------------------------------------
step(1, 5, "Load tokenizer")
log(f"Fetching tokenizer for {MODEL_ID} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
log(f"Tokenizer ready ({time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# 2. Load model  (this is the slow download step)
# ---------------------------------------------------------------------------
step(2, 5, "Download + load model weights")
log("Starting model download — this is the slow step (~10 GB at float16)")
log("You'll see shard progress from transformers below:")
print(flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    low_cpu_mem_usage=True,
)
model.train(False)  # inference mode
elapsed = time.time() - t0
param_b = sum(p.numel() for p in model.parameters()) / 1e9
log(f"Model loaded in {elapsed:.1f}s  |  {param_b:.2f}B parameters")

# ---------------------------------------------------------------------------
# 3. Print architecture details
# ---------------------------------------------------------------------------
step(3, 5, "Inspect architecture config")
cfg = model.config
attrs = [
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
]
for attr in attrs:
    val = getattr(cfg, attr, "NOT FOUND")
    log(f"  {attr}: {val}")

# ---------------------------------------------------------------------------
# 4. Attempt torch.export
# ---------------------------------------------------------------------------
step(4, 5, "torch.export trace")
log("Building dummy inputs (batch=1, seq=4) ...")
dummy_input = tokenizer("Hello!", return_tensors="pt")
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]
log(f"  input_ids shape: {input_ids.shape}")

export_error = None
export_tb = None
export_graph = None

log("Running torch.export.export(strict=False) ...")
t0 = time.time()
try:
    with torch.no_grad():
        exported = torch.export.export(
            model,
            args=(input_ids,),
            kwargs={"attention_mask": attention_mask},
            strict=False,
        )
    export_graph = exported.graph_module
    log(f"SUCCESS — export completed in {time.time()-t0:.1f}s")
except Exception as e:
    export_error = f"{type(e).__name__}: {str(e)}"
    export_tb = traceback.format_exc()
    log(f"FAILED in {time.time()-t0:.1f}s: {export_error[:200]}")

# ---------------------------------------------------------------------------
# 5. Write report
# ---------------------------------------------------------------------------
step(5, 5, "Write report")
report_path = "phase0_trace_report.txt"
with open(report_path, "w") as f:
    f.write("# Phase 0 Trace Report — Gemma 4 E2B\n\n")
    f.write(f"Model: {MODEL_ID}\n")
    f.write(f"dtype: {DTYPE}\n")
    f.write(f"Parameters: {param_b:.2f}B\n\n")

    f.write("## Config (relevant fields)\n")
    for attr in attrs[:-2]:  # skip vision/audio configs (too verbose)
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

log(f"Report written to {report_path}")
log("Phase 0 complete.")
