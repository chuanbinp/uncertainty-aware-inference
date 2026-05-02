"""
ncu_llama2_7b_nf4.py — Llama-2 7B NF4 (bitsandbytes) for ncu profiling (prefill, seq=128)
"""
import os, time
import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID       = "huggyllama/llama-7b"
TARGET_SEQ_LEN = 128
PROMPT         = "The key difference between quantization and pruning is"

hf_token = os.environ.get("HF_TOKEN")
print(f"[load] Loading {MODEL_ID} (NF4)...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg,
    device_map="cuda:0",
)
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
