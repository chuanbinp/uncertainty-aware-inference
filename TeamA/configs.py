# TeamA/configs.py

# KD_REGISTRY pairs each INT4 student with its FP16 teacher.
# Temperature is NOT stored here — it is a sweep parameter passed via CLI.
# AWQ/GPTQ entries kept as config references; training is NF4-only (see kd_train.py).
KD_REGISTRY = {
    # Llama-1 7B
    "llama1-7b-awq-int4-kd": {
        "teacher":              "llama1-7b-fp16",
        "teacher_hf_id":        "huggyllama/llama-7b",
        "student":              "llama1-7b-awq-int4",
        "student_hf_id":        "TheBloke/LLaMA-7b-AWQ",
        "student_quant_type":   "awq",
        "student_bits":         4,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "llama1-7b-gptq-int4-kd": {
        "teacher":              "llama1-7b-fp16",
        "teacher_hf_id":        "huggyllama/llama-7b",
        "student":              "llama1-7b-gptq-int4",
        "student_hf_id":        "TheBloke/LLaMa-7B-GPTQ",
        "student_quant_type":   "gptq",
        "student_bits":         4,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "llama1-7b-nf4-kd": {
        "teacher":              "llama1-7b-fp16",
        "teacher_hf_id":        "huggyllama/llama-7b",
        "student":              "llama1-7b-nf4",
        "student_hf_id":        "huggyllama/llama-7b",
        "student_quant_type":   "nf4",
        "student_bits":         4,
        "alpha":                0.7,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },

    # Mistral 7B
    "mistral-7b-awq-int4-kd": {
        "teacher":              "mistral-7b-fp16",
        "teacher_hf_id":        "mistralai/Mistral-7B-Instruct-v0.2",
        "student":              "mistral-7b-awq-int4",
        "student_hf_id":        "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "student_quant_type":   "awq",
        "student_bits":         4,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "mistral-7b-gptq-int4-kd": {
        "teacher":              "mistral-7b-fp16",
        "teacher_hf_id":        "mistralai/Mistral-7B-Instruct-v0.2",
        "student":              "mistral-7b-gptq-int4",
        "student_hf_id":        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "student_quant_type":   "gptq",
        "student_bits":         4,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "mistral-7b-nf4-kd": {
        "teacher":              "mistral-7b-fp16",
        "teacher_hf_id":        "mistralai/Mistral-7B-Instruct-v0.2",
        "student":              "mistral-7b-nf4",
        "student_hf_id":        "mistralai/Mistral-7B-Instruct-v0.2",
        "student_quant_type":   "nf4",
        "student_bits":         4,
        "alpha":                0.7,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },

    # Llama-2 13B
    "llama2-13b-awq-int4-kd": {
        "teacher":              "llama2-13b-fp16",
        "teacher_hf_id":        "meta-llama/Llama-2-13b-hf",
        "student":              "llama2-13b-awq-int4",
        "student_hf_id":        "TheBloke/Llama-2-13B-AWQ",
        "student_quant_type":   "awq",
        "student_bits":         4,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "llama2-13b-gptq-int4-kd": {
        "teacher":              "llama2-13b-fp16",
        "teacher_hf_id":        "meta-llama/Llama-2-13b-hf",
        "student":              "llama2-13b-gptq-int4",
        "student_hf_id":        "TheBloke/Llama-2-13B-GPTQ",
        "student_quant_type":   "gptq",
        "student_bits":         4,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "llama2-13b-nf4-kd": {
        "teacher":              "llama2-13b-fp16",
        "teacher_hf_id":        "meta-llama/Llama-2-13b-hf",
        "student":              "llama2-13b-nf4",
        "student_hf_id":        "meta-llama/Llama-2-13b-hf",
        "student_quant_type":   "nf4",
        "student_bits":         4,
        "alpha":                0.7,
        "lora_r":               16,
        "lora_alpha":           32,
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}

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