"""
ncu_llama2_13b_awq_int4.py — Llama-2 7B AWQ INT4 for ncu profiling (prefill, seq=128)
"""
import os, time
import torch
import torch.cuda.nvtx as nvtx
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

MODEL_ID       = "TheBloke/Llama-2-13B-AWQ"
TARGET_SEQ_LEN = 128
PROMPT         = "The key difference between quantization and pruning is"

hf_token = os.environ.get("HF_TOKEN")
print(f"[load] Loading {MODEL_ID} (AWQ INT4)...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoAWQForCausalLM.from_quantized(MODEL_ID, fuse_layers=False)
model.eval()
print(f"[load] Done in {time.perf_counter()-t0:.1f}s  |  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

inputs = tokenizer(PROMPT, return_tensors="pt", padding="max_length",
                   max_length=TARGET_SEQ_LEN, truncation=True).to("cuda:0")
input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

print("[warmup] Running 3 warmup passes...")
with torch.no_grad():
    for _ in range(3):
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
