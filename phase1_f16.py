"""
Phase 1 — FLOAT16 CoreML conversion (standalone)
=================================================
Runs the f16 pass only. The f32 mlpackage is assumed to already exist from
phase1_convert.py. Keeping this separate avoids the OOM segfault caused by
holding a f32 model (20+ GB) in memory while reloading f16.

Usage:
    .venv/bin/python phase1_f16.py

Output:
    gemma4-e2b-text-f16.mlpackage
"""

import gc
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-e2b-it"
DEVICE   = "cpu"

report_lines = []

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    report_lines.append(line)

def step(n, total, label):
    line = f"\n── Step {n}/{total}: {label} ──"
    print(line, flush=True)
    report_lines.append(line)

def save_report(path="phase1_f16_report.txt"):
    with open(path, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Report saved to {path}")


class GemmaTextDecoder(torch.nn.Module):
    """Wrapper that bakes in boolean kwargs so they never appear as graph placeholders."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return out[0]  # logits: [batch, seq, vocab_size]


def patch_for_coreml(gm: torch.fx.GraphModule) -> dict:
    """
    Replace all ATen ops unsupported by coremltools 9.0 EXIR frontend.
    Verified individually — only these 4 patterns fail:
      __or__   → logical_or   (bool mask combination)
      __and__  → logical_and  (bool mask combination)
      new_ones → full         (no EXIR handler)
      alias    → identity     (no EXIR handler; alias is a no-op)
    """
    graph = gm.graph
    counts: dict = {}

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        if node.target == torch.ops.aten.__or__.Tensor:
            node.target = torch.ops.aten.logical_or.default
            counts["__or__→logical_or"] = counts.get("__or__→logical_or", 0) + 1

        elif node.target == torch.ops.aten.__and__.Tensor:
            node.target = torch.ops.aten.logical_and.default
            counts["__and__→logical_and"] = counts.get("__and__→logical_and", 0) + 1

        elif node.target == torch.ops.aten.new_ones.default:
            _, size = node.args[0], node.args[1]
            dtype = node.meta["val"].dtype if "val" in node.meta else torch.float16
            with graph.inserting_before(node):
                full_node = graph.call_function(
                    torch.ops.aten.full.default,
                    args=(size, 1),
                    kwargs={"dtype": dtype},
                )
                full_node.meta = dict(node.meta)
            node.replace_all_uses_with(full_node)
            graph.erase_node(node)
            counts["new_ones→full"] = counts.get("new_ones→full", 0) + 1

        elif node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            counts["alias→identity"] = counts.get("alias→identity", 0) + 1

    graph.lint()
    gm.recompile()
    return counts


# ---------------------------------------------------------------------------
# 1. Load in float16 from scratch — no f32 model in memory
# ---------------------------------------------------------------------------
step(1, 4, "Load model in float16")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model_raw = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map=DEVICE,
    low_cpu_mem_usage=True,
)
model_raw.train(False)
model = GemmaTextDecoder(model_raw)
param_b = sum(p.numel() for p in model.parameters()) / 1e9
log(f"Loaded in {time.time()-t0:.1f}s  —  {param_b:.2f}B params  —  dtype=float16")

# ---------------------------------------------------------------------------
# 2. Export + patch
# ---------------------------------------------------------------------------
step(2, 4, "torch.export + patch")
dummy   = tokenizer("Hello world", return_tensors="pt")
ids     = dummy["input_ids"]
mask    = dummy["attention_mask"]

t0 = time.time()
exported = torch.export.export(model, args=(ids, mask), strict=False)
exported = exported.run_decompositions({})
patch_counts = patch_for_coreml(exported.graph_module)
log(f"Export + patch done in {time.time()-t0:.1f}s  —  patches: {patch_counts}")

# Free raw model immediately — not needed for conversion
del model_raw, model
gc.collect()
log("Model freed from memory before conversion")

# ---------------------------------------------------------------------------
# 3. ct.convert() — FLOAT16
# ---------------------------------------------------------------------------
step(3, 4, "ct.convert() — FLOAT16")
log("Starting CoreML conversion ...")
t0 = time.time()
try:
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="input_ids",      shape=ids.shape,  dtype=np.int32),
            ct.TensorType(name="attention_mask",  shape=mask.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    log(f"ct.convert FLOAT16 SUCCESS in {time.time()-t0:.1f}s")
except Exception as e:
    log(f"ct.convert FLOAT16 FAILED in {time.time()-t0:.1f}s")
    log(f"  {type(e).__name__}: {str(e)[:400]}")
    log(traceback.format_exc())
    save_report()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Save + verify
# ---------------------------------------------------------------------------
step(4, 4, "Save + predict() verify")
pkg_path = "gemma4-e2b-text-f16.mlpackage"
mlmodel.save(pkg_path)
log(f"Saved: {pkg_path}")

log("Running predict() ...")
try:
    preds = mlmodel.predict({
        "input_ids":      ids.numpy().astype(np.int32),
        "attention_mask": mask.numpy().astype(np.int32),
    })
    log(f"predict() SUCCESS")
    for k, v in preds.items():
        log(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        log(f"  {k} sample (first 5 @ last token): {v[0, -1, :5].tolist()}")
except Exception as e:
    log(f"predict() FAILED: {e}")

save_report()
log("Phase 1 f16 complete.")
