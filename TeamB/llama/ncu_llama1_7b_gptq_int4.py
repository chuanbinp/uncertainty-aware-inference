"""
ncu_llama2_7b_gptq_int4.py — Llama-2 7B GPTQ INT4 for ncu profiling (prefill, seq=128)

Usage:
    ncu -o results/ncu/llama2_7b_gptq_int4 \
        --metrics gpu__time_duration.sum,dram__bytes.sum,sm__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
        --launch-skip 500 --launch-count 30 \
        --kernel-name "regex:marlin::Marlin|ampere_bf16_s16816gemm|fmha_cutlass" \
        --force-overwrite \
        python ncu_llama2_7b_gptq_int4.py

    ncu --import results/ncu/llama2_7b_gptq_int4.ncu-rep \
        --csv --print-units base 2>/dev/null \
        > results/ncu/llama2_7b_gptq_int4_metrics.csv
"""
import os, time
import torch
import torch.cuda.nvtx as nvtx
from auto_gptq import AutoGPTQForCausalLM as GPTQModel
from transformers import AutoTokenizer

MODEL_ID       = "TheBloke/LLaMa-7B-GPTQ"
REVISION = "gptq-4bit-128g-actorder_True"
TARGET_SEQ_LEN = 128
PROMPT         = "The key difference between quantization and pruning is"

hf_token = os.environ.get("HF_TOKEN")
print(f"[load] Loading {MODEL_ID} (GPTQ INT4)...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token
model = GPTQModel.from_quantized(MODEL_ID, revision=REVISION, device="cuda:0", use_auth_token=True)
model.eval()
print(f"[load] Done in {time.perf_counter()-t0:.1f}s  |  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

inputs = tokenizer(PROMPT, return_tensors="pt", padding="max_length",
                   max_length=TARGET_SEQ_LEN, truncation=True).to("cuda:0")
input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

print("[warmup] Running 5 warmup passes (pass 1 triggers Marlin repack)...")
with torch.no_grad():
    for _ in range(5):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
print("[warmup] Done")

print("[profile] Running single forward pass (profiled)...")
t0 = time.perf_counter()
nvtx.range_push("profiling_region")
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
nvtx.range_pop()
print(f"[profile] Done in {(time.perf_counter()-t0)*1000:.1f}ms")
print("done")
