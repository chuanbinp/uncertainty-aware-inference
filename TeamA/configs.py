# TeamA/configs.py

MODEL_REGISTRY = {
    "llama2-7b-fp16": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "quant_type": "fp16",
        "bits": 16,
        "description": "FP16 Baseline",
    },
    "llama2-7b-nf4": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "quant_type": "nf4",
        "bits": 4,
        "description": "NF4 (bitsandbytes 4-bit)",
    },
    "llama2-7b-awq-int4": {
        "hf_id": "TheBloke/Llama-2-7b-AWQ",
        "quant_type": "awq",
        "bits": 4,
        "description": "AWQ INT4",
    },
    "llama2-7b-gptq-int4": {
        "hf_id": "TheBloke/Llama-2-7B-GPTQ",
        "quant_type": "gptq",
        "bits": 4,
        "gptq_revision": "gptq-4bit-128g-actorder_True",  # ← add revision
        "description": "GPTQ INT4",
    },
    "llama2-7b-gptq-int8": {
        "hf_id": "PF-8-bit/Llama-2-7b-hf-gptq-8bit",
        "quant_type": "gptq",
        "bits": 8,
        "description": "GPTQ INT8",
    },
}