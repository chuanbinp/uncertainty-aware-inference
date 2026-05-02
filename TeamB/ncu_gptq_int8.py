"""
ncu_gptq_int8.py — Mistral-7B GPTQ INT8 for ncu profiling (prefill, seq=128)

Usage:
    ncu -o results/ncu/mistral_gptq_int8 \
        --metrics gpu__time_duration.sum,dram__bytes.sum,sm__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
        --launch-skip 500 --launch-count 30 \
        --kernel-name "regex:marlin::Marlin|ampere_bf16_s16816gemm|fmha_cutlass" \
        --force-overwrite \
        python ncu_gptq_int8.py

    ncu --import results/ncu/mistral_gptq_int8.ncu-rep \
        --csv --print-units base 2>/dev/null \
        > results/ncu/mistral_gptq_int8_metrics.csv
"""
import os, time
import torch
import torch.cuda.nvtx as nvtx

from huggingface_hub import login


hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from auto_gptq import AutoGPTQForCausalLM as GPTQModel
from transformers import AutoTokenizer

MODEL_ID       = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
REVISION       = "gptq-8bit-128g-actorder_True"   # INT8 revision
TARGET_SEQ_LEN = 128
PROMPT         = "The key difference between quantization and pruning is"

print(f"[load] Loading {MODEL_ID} (GPTQ INT8, revision={REVISION})...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
tokenizer.pad_token = tokenizer.eos_token

model = GPTQModel.from_quantized(
    MODEL_ID,
    revision=REVISION,
    device="cuda:0",
    trust_remote_code=True,
)
model.eval()
print(f"[load] Done in {time.perf_counter()-t0:.1f}s  |  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

inputs = tokenizer(PROMPT, return_tensors="pt", padding="max_length",
                   max_length=TARGET_SEQ_LEN, truncation=True).to("cuda:0")
input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

print("[warmup] Running 5 warmup passes (pass 1 triggers Marlin repack)...")
with torch.no_grad():
    for i in range(5):
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
