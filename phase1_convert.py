"""
Phase 1 — CoreML Conversion: Gemma 4 E2B text decoder
======================================================
Steps:
  1. Load model + tokenizer (uses local HF cache)
  2. torch.export with dynamic seq_len dimension
  3. ct.convert() — FLOAT32 first pass to isolate conversion bugs
  4. Save gemma4-e2b-text-f32.mlpackage
  5. Sanity-check predict() — verify output shape + logit values
  6. FLOAT16 conversion pass — gemma4-e2b-text-f16.mlpackage

Usage:
    .venv/bin/python phase1_convert.py

Outputs:
    gemma4-e2b-text-f32.mlpackage
    gemma4-e2b-text-f16.mlpackage   (if f32 succeeds)
    phase1_convert_report.txt
"""

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
DTYPE    = torch.float32   # export in f32 for first pass; f16 for second


class GemmaTextDecoder(torch.nn.Module):
    """
    Thin wrapper that hard-codes boolean kwargs so they never appear as
    graph-level placeholders (which coremltools cannot ingest).
    - use_cache=False  : no DynamicCache in output
    - return_dict=False: output is a plain tuple, not a dataclass
    forward returns logits tensor directly: [batch, seq, vocab_size]
    """
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

def save_report(path="phase1_convert_report.txt"):
    with open(path, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Report saved to {path}")


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
step(1, 6, "Load model + tokenizer")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
log("Tokenizer loaded")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    low_cpu_mem_usage=True,
)
model.train(False)
model = GemmaTextDecoder(model)  # wrap to bake in boolean kwargs
log(f"Model loaded in {time.time()-t0:.1f}s  — {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")


# ---------------------------------------------------------------------------
# 2. torch.export with dynamic seq_len
# ---------------------------------------------------------------------------
step(2, 6, "torch.export with dynamic seq_len")

dummy   = tokenizer("Hello world", return_tensors="pt")
ids     = dummy["input_ids"].to(DEVICE)       # [1, seq]
mask    = dummy["attention_mask"].to(DEVICE)  # [1, seq]
log(f"Dummy input shape: {ids.shape}")

# Fixed shapes for Phase 1 — dynamic seq_len deferred to Phase 2
# (torch.export dynamic_shapes API requires careful kwargs structuring; fixed is fine for validation)
export_error = None
exported     = None
t0 = time.time()
try:
    exported = torch.export.export(
        model,
        args=(ids, mask),   # wrapper takes (input_ids, attention_mask) — no bool kwargs
        strict=False,
    )
    log(f"torch.export SUCCESS in {time.time()-t0:.1f}s")
except Exception as e:
    export_error = str(e)
    log(f"torch.export FAILED: {type(e).__name__}: {str(e)[:300]}")
    save_report()
    sys.exit(1)

# Lower TRAINING dialect → ATEN dialect (required by coremltools)
log("Running .run_decompositions({}) to lower to ATEN dialect ...")
exported = exported.run_decompositions({})
log("Dialect lowered to ATEN")


# ---------------------------------------------------------------------------
# 3. Inspect exported outputs
# ---------------------------------------------------------------------------
step(3, 6, "Inspect export graph outputs")
output_nodes = [n for n in exported.graph_module.graph.nodes if n.op == "output"]
log(f"Output nodes: {len(output_nodes)}")
for n in output_nodes:
    log(f"  args: {n.args}")

# Run one forward pass to check output shapes
with torch.no_grad():
    out = model(ids, mask)   # wrapper returns logits tensor directly
log(f"Forward pass output shape: {out.shape}, dtype: {out.dtype}")
log(f"  logits sample (first 5 vocab @ last token): {out[0, -1, :5].tolist()}")


# ---------------------------------------------------------------------------
# 4. ct.convert() — FLOAT32 first pass
# ---------------------------------------------------------------------------
step(4, 6, "ct.convert() — FLOAT32")
log("Starting CoreML conversion (this may take several minutes) ...")

mlmodel_f32 = None
convert_error_f32 = None
t0 = time.time()
try:
    mlmodel_f32 = ct.convert(
        exported,
        inputs=[
            ct.TensorType(name="input_ids",      shape=ids.shape,  dtype=np.int32),
            ct.TensorType(name="attention_mask",  shape=mask.shape, dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,   # CPU_ONLY for first pass — rules out GPU/ANE issues
        minimum_deployment_target=ct.target.macOS15,
    )
    elapsed = time.time() - t0
    log(f"ct.convert FLOAT32 SUCCESS in {elapsed:.1f}s")
except Exception as e:
    convert_error_f32 = f"{type(e).__name__}: {str(e)}"
    log(f"ct.convert FLOAT32 FAILED in {time.time()-t0:.1f}s")
    log(f"  {convert_error_f32[:400]}")
    log(f"  Full traceback:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 5. Save + sanity check (FLOAT32)
# ---------------------------------------------------------------------------
step(5, 6, "Save + predict() sanity check")

if mlmodel_f32:
    pkg_path = "gemma4-e2b-text-f32.mlpackage"
    mlmodel_f32.save(pkg_path)
    log(f"Saved: {pkg_path}")

    # Predict with same dummy input
    log("Running predict() ...")
    t0 = time.time()
    try:
        preds = mlmodel_f32.predict({
            "input_ids":     ids.numpy().astype(np.int32),
            "attention_mask": mask.numpy().astype(np.int32),
        })
        log(f"predict() SUCCESS in {time.time()-t0:.1f}s")
        log(f"  output keys: {list(preds.keys())}")
        for k, v in preds.items():
            log(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            log(f"  {k} sample (first 5): {v.flatten()[:5].tolist()}")
    except Exception as e:
        log(f"predict() FAILED: {e}")

    # ---------------------------------------------------------------------------
    # 6. ct.convert() — FLOAT16
    # ---------------------------------------------------------------------------
    step(6, 6, "ct.convert() — FLOAT16 (for ANE)")
    log("Reloading model in float16 for second conversion pass ...")
    del model
    model_f16 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model_f16.train(False)
    model_f16 = GemmaTextDecoder(model_f16)

    exported_f16 = torch.export.export(
        model_f16,
        args=(ids, mask),
        strict=False,
    )
    exported_f16 = exported_f16.run_decompositions({})
    log("Re-export + dialect lowering for f16 done")

    t0 = time.time()
    try:
        mlmodel_f16 = ct.convert(
            exported_f16,
            inputs=[
                ct.TensorType(name="input_ids",     shape=ids.shape,  dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=mask.shape, dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="logits", dtype=np.float16),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
        )
        log(f"ct.convert FLOAT16 SUCCESS in {time.time()-t0:.1f}s")
        pkg_f16 = "gemma4-e2b-text-f16.mlpackage"
        mlmodel_f16.save(pkg_f16)
        log(f"Saved: {pkg_f16}")
    except Exception as e:
        log(f"ct.convert FLOAT16 FAILED: {type(e).__name__}: {str(e)[:300]}")
        log(traceback.format_exc())
else:
    step(6, 6, "SKIPPED — f32 conversion failed")
    log("Fix f32 conversion first before attempting f16.")

save_report()
log("Phase 1 complete.")
