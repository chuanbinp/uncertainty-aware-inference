"""
ncu_gptq_int4.py
─────────────────────────────────────────────────────────────────────────────
Mistral-7B GPTQ INT4 inference script for ncu / nsys profiling.

Uses gptqmodel 7.x GPTQModel.load() API.
Same structure as ncu_fp16.py for direct comparability:
  - Same prompt and sequence length (TARGET_SEQ_LEN = 128)
  - Same warmup pattern (3 forward passes)
  - Same NVTX range name ("profiling_region")

HF model: TheBloke/Mistral-7B-Instruct-v0.2-GPTQ (revision: main = INT4)

Setup:
    Create a .env file in the same directory:
        HF_TOKEN="hf_xxxxxxxxxxxx"

Usage:
    # Standalone test:
    python ncu_gptq_int4.py

   # Scan for kernel names after extended warmup (use small skip):
    ncu -o /tmp/gptq_int4_scan --set roofline \
        --launch-skip 500 --launch-count 10 --force-overwrite \
        python ncu_gptq_int4.py
 
    # Full metrics profile:
    ncu -o results/ncu/mistral_gptq_int4 \
        --metrics gpu__time_duration.sum,dram__bytes.sum,\
sm__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
        --launch-skip 500 --launch-count 30 \
        --kernel-name-base demangled \
        --kernel-name "regex:marlin::Marlin|ampere_bf16_s16816gemm|fmha_cutlassF_bf16" \
        --force-overwrite \
        python ncu_gptq_int4.py
─────────────────────────────────────────────────────────────────────────────
"""
 
import os
import time
 
import torch
import torch.cuda.nvtx as nvtx
from dotenv import load_dotenv
from huggingface_hub import login
 
# ── HuggingFace auth ──────────────────────────────────────────────────────────
load_dotenv()
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("[auth] HF_TOKEN not set — assuming model is cached locally")
 
from gptqmodel import GPTQModel
from transformers import AutoTokenizer
 
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
REVISION       = "main"   # main = INT4 default quantization
TARGET_SEQ_LEN = 128
PROMPT         = "The key difference between quantization and pruning is"
 
# ── Load ──────────────────────────────────────────────────────────────────────
print(f"[load] Loading {MODEL_ID} (GPTQ INT4, revision={REVISION})...")
t0 = time.perf_counter()
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
tokenizer.pad_token = tokenizer.eos_token
 
model = GPTQModel.load(
    MODEL_ID,
    revision=REVISION,
    device="cuda:0",
    trust_remote_code=True,
)
model.eval()
 
print(f"[load] Done in {time.perf_counter() - t0:.1f}s  |  "
      f"GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
 
# ── Tokenize ──────────────────────────────────────────────────────────────────
inputs = tokenizer(
    PROMPT,
    return_tensors="pt",
    padding="max_length",
    max_length=TARGET_SEQ_LEN,
    truncation=True,
).to("cuda:0")
input_ids      = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
 
print(f"[input] Sequence length: {input_ids.shape[1]}  |  Batch: {input_ids.shape[0]}")
 
# ── Extended warmup — forces Marlin repack to complete ────────────────────────
# The Marlin backend repacks INT4 weights on the FIRST forward pass,
# firing ~5000 CUDA kernels (sort, repack, permute). This must complete
# before the profiled region so ncu sees only clean inference kernels.
#
# Pass 1: triggers Marlin repack (slow, ~5000 extra kernels)
# Pass 2-5: pure inference (fast, only Marlin GEMM + attention kernels)
#
# After this warmup, --launch-skip 500 is enough to clear load-time
# kernels and land directly on the Marlin inference GEMM.
print("[warmup] Running 5 warmup passes (pass 1 triggers Marlin repack)...")
with torch.no_grad():
    for i in range(5):
        t_pass = time.perf_counter()
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        print(f"[warmup] Pass {i+1}/5 — {(time.perf_counter()-t_pass)*1000:.1f}ms "
              f"{'(repack)' if i == 0 else '(inference)'}")
print("[warmup] Done — Marlin repack complete, subsequent passes are pure inference")
 
# ── Profiled region ───────────────────────────────────────────────────────────
# ncu will see only Marlin GEMM + BF16 attention kernels here.
# Use --launch-skip 500 --kernel-name "regex:marlin::Marlin|ampere_bf16"
# to target just the compute-relevant kernels.
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
print(f"[profile] Forward pass in {elapsed * 1000:.1f}ms")
print(f"[profile] Output logits shape: {outputs.logits.shape}")
print("done")
 