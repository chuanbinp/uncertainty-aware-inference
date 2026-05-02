"""ncu_llama2_13b_gptq_int4.py — Llama-2 13B GPTQ INT4 for ncu profiling"""
import os, time
import torch
import torch.cuda.nvtx as nvtx
from auto_gptq import AutoGPTQForCausalLM as GPTQModel
from transformers import AutoTokenizer

MODEL_ID       = "TheBloke/Llama-2-13B-GPTQ"
REVISION       = "gptq-4bit-128g-actorder_True"
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

print("[warmup] 5 passes (pass 1 = Marlin repack)...")
with torch.no_grad():
    for _ in range(5):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()

nvtx.range_push("profiling_region")
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
nvtx.range_pop()
print("done")
