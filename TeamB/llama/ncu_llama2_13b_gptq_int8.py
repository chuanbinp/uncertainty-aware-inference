import os, time
import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from the directly downloaded file
LOCAL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-GPTQ/blobs/model_int8.safetensors"
)
MODEL_ID   = "/tmp/llama2_13b_int8"
REVISION   = "gptq-8bit-128g-actorder_True"
TARGET_SEQ_LEN = 128
PROMPT     = "The key difference between quantization and pruning is"

hf_token = os.environ.get("HF_TOKEN")
print("[load] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

print("[load] Loading model from local safetensors...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, revision=REVISION,
    device_map="cuda:0", token=hf_token,
)
model.eval()
print(f"[load] Done  |  GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

inputs = tokenizer(PROMPT, return_tensors="pt", padding="max_length",
                   max_length=TARGET_SEQ_LEN, truncation=True).to("cuda:0")
input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

print("[warmup] 3 passes...")
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
print("[warmup] Done")

nvtx.range_push("profiling_region")
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
torch.cuda.synchronize()
nvtx.range_pop()
print("done")
