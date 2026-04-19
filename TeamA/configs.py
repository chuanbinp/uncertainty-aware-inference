# TeamA/configs.py

MODEL_REGISTRY = {
    "llama1-7b-fp16": {
        "hf_id": "huggyllama/llama-7b",
        "quant_type": "fp16",
        "bits": 16,
        "description": "FP16 Baseline",
    },
    "llama1-7b-nf4": {
        "hf_id": "huggyllama/llama-7b",
        "quant_type": "nf4",
        "bits": 4,
        "description": "NF4 (bitsandbytes 4-bit)",
    },
    "llama1-7b-awq-int4": {
        "hf_id": "TheBloke/LLaMA-7b-AWQ",
        "quant_type": "awq",
        "bits": 4,
        "description": "AWQ INT4",
    },
    "llama1-7b-gptq-int4": {
        "hf_id": "TheBloke/LLaMa-7B-GPTQ",
        "quant_type": "gptq",
        "bits": 4,
        "gptq_revision": "gptq-4bit-128g-actorder_True",
        "description": "GPTQ INT4",
    },
    "llama1-7b-gptq-int8": {
        "hf_id": "TheBloke/LLaMa-7B-GPTQ",
        "quant_type": "gptq",
        "bits": 8,
        "gptq_revision": "gptq-8bit-128g-actorder_True",
        "description": "GPTQ INT8",
    },
}
