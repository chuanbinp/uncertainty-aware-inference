"""
ncu_fp16.py
─────────────────────────────────────────────────────────────────────────────
Minimal Mistral-7B FP16 inference script for ncu / nsys profiling.

Uses a single model() forward pass (not generate()) in the profiled region.
This fires exactly one set of GEMM kernels per transformer layer — 32 sets
total for Mistral-7B — making it much easier for ncu to target without
overshooting into non-GEMM kernels.

model.generate() loops internally (one forward pass per token), firing
thousands of small kernels before the first GEMM in the profiled region.
Single forward pass avoids all of that.

Setup:
    Create a .env file in the same directory:
        HF_TOKEN="hf_xxxxxxxxxxxx"

Usage:
    # Standalone test (no profiler):
    python ncu_fp16.py

    - profile
    ncu -o results/ncu/mistral_fp16_v3 \
    --metrics gpu__time_duration.sum,dram__bytes.sum,sm__inst_executed_pipe_tensor.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-skip 30 \
    --launch-count 20 \
    --kernel-name "regex:ampere_fp16_s16816gemm|fmha_cutlassF" \
    --force-overwrite \
    python ncu_fp16.py

    - convert results to csv
    ncu --import results/ncu/mistral_fp16_v3.ncu-rep \
        --csv --print-units base 2>/dev/null \
        > results/ncu/mistral_fp16_v3_metrics.csv

─────────────────────────────────────────────────────────────────────────────
"""

import os
import time

import torch
import torch.cuda.nvtx as nvtx
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── HuggingFace auth ──────────────────────────────────────────────────────────
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("[auth] HF_TOKEN not set — assuming model is cached locally")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
TARGET_SEQ_LEN = 128    # pad/truncate to this many tokens for consistent shapes
PROMPT         = "The key difference between quantization and pruning is"

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"[load] Loading {MODEL_ID} in FP16...")
t0 = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # add this line

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"[load] Done in {time.perf_counter() - t0:.1f}s  |  "
      f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Pad to TARGET_SEQ_LEN for consistent GEMM shapes across configs
inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
    padding="max_length",
    max_length=TARGET_SEQ_LEN,
    truncation=True,
).to("cuda:0")
input_ids      = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print(f"[input] Sequence length: {input_ids.shape[1]} tokens  |  "
      f"Batch size: {input_ids.shape[0]}")

# ── Warmup — not profiled ─────────────────────────────────────────────────────
# Use model() directly (not generate()) to match the profiled region.
# Warmup triggers cuBLAS autotuning so the profiled run sees stable,
# production-representative kernel choices.
print("[warmup] Running 3 warmup forward passes...")
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
print("[warmup] Done")

# ── Profiled region ───────────────────────────────────────────────────────────
# Single forward pass — fires exactly 32 GEMM sets (one per Mistral layer):
#   Q/K/V projections  → 3 GEMMs per layer
#   Output projection  → 1 GEMM per layer
#   MLP gate/up/down   → 3 GEMMs per layer
#   Total: 7 GEMMs × 32 layers = 224 GEMM kernel launches
#
# ncu with --launch-skip 0 --launch-count N will capture the first N of these.
# After warmup, the first kernels fired should be the attention GEMMs.
print("[profile] Running single forward pass (profiled)...")
t0 = time.perf_counter()

nvtx.range_push("profiling_region")

with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

torch.cuda.synchronize()
nvtx.range_pop()

elapsed = time.perf_counter() - t0
print(f"[profile] Forward pass in {elapsed*1000:.1f}ms")
print(f"[profile] Output logits shape: {outputs.logits.shape}")
print("done")