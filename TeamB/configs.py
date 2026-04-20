"""
Quantization configurations for Mistral-7B PTQ sweep.

Each config defines the loading method, precision, and HuggingFace model ID.
Update model IDs here if using different quantized checkpoints.
"""

MODEL_REGISTRY = {
    "mistral-7b-fp16": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "quant_type": "fp16",
        "bits": 16,
        "description": "FP16 Baseline",
    },
     "mistral-7b-gptq-int8": {
        "hf_id": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "quant_type": "gptq",
        "bits": 8,
        "gptq_revision": "gptq-8bit-128g-actorder_True",
        "description": "GPTQ INT8",
    },

    "mistral-7b-gptq-int4": {
        "hf_id": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "quant_type": "gptq",
        "bits": 4,
        "gptq_revision": "main",
        "description": "GPTQ INT4",
        },
   
    "mistral-7b-awq-int4": {
        "hf_id": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "quant_type": "awq",
        "bits": 4,
        "description": "AWQ INT4",
    },
     "mistral-7b-nf4": {
        "hf_id":"mistralai/Mistral-7B-Instruct-v0.2", #base model, not quantized
        "quant_type": "nf4",
        "bits": 4,
        "description": "NF4 (bitsandbytes 4-bit)",
    }

}
