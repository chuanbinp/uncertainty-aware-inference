import gc
import os

import torch
from transformers import (
    AutoAWQForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
)
from awq import AutoAWQForCausalLM

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
        "description": "GPTQ INT4",
    },
    "llama2-7b-gptq-int8": {
        "hf_id": "PF-8-bit/Llama-2-7b-hf-gptq-8bit",
        "quant_type": "gptq",
        "bits": 8,
        "description": "GPTQ INT8",
    },
}


def _load_fp16_model(hf_id: str, hf_token: str):
    """Load model at full FP16 precision with automatic device mapping."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    return model, tokenizer


def _load_nf4_model(hf_id: str, hf_token: str):
    """Load model quantized to 4-bit NormalFloat (NF4) via bitsandbytes."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=hf_token,
    )
    return model, tokenizer


def _load_awq_model(hf_id: str, hf_token: str):
    """Load a pre-quantized AWQ INT4 model via AutoAWQ directly."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    model = AutoAWQForCausalLM.from_quantized(
        hf_id,
        fuse_layers=False,
        device_map="auto",
        safetensors=True,
    )
    return model, tokenizer


def _load_gptq_model(hf_id: str, bits: int, hf_token: str):
    """Load a pre-quantized GPTQ model via HuggingFace + GPTQConfig."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map={"": 0},
        trust_remote_code=True,
        quantization_config=GPTQConfig(bits=bits, disable_exllama=True),
        token=hf_token,
    )
    return model, tokenizer


def load_model(config_key: str, hf_token: str | None = None):
    """
    Load a model and tokenizer by registry key.
 
    Args:
        config_key: One of the keys in MODEL_REGISTRY (e.g. "llama2-7b-nf4").
        hf_token:   HuggingFace access token. Required for gated models
                    (Llama-2). Set via export HF_TOKEN=your_token_here.

    Returns:
        (model, tokenizer) tuple ready for inference.
 
    Raises:
        ValueError: If config_key is not in MODEL_REGISTRY.
    """
    if config_key not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown config key '{config_key}'. Valid options: {valid}"
        )

    hf_token = os.environ.get("HF_TOKEN")
    entry = MODEL_REGISTRY[config_key]
    hf_id = entry["hf_id"]
    quant_type = entry["quant_type"]
    bits = entry["bits"]
  
    if quant_type == "fp16":
        model, tokenizer = _load_fp16_model(hf_id, hf_token)
    elif quant_type == "nf4":
        model, tokenizer = _load_nf4_model(hf_id, hf_token)
    elif quant_type == "awq":
        model, tokenizer = _load_awq_model(hf_id, hf_token)
    elif quant_type == "gptq":
        model, tokenizer = _load_gptq_model(hf_id, bits, hf_token)
    else:
        raise ValueError(f"Unhandled quant_type '{quant_type}'")
 
    model.eval()
    print(f"Loaded successfully. Device: {next(model.parameters()).device}")
    return model, tokenizer


def free_model(model):
    """
    Release a model from GPU memory.
 
    Call this between sequential model loads to avoid OOM errors,
    especially critical when sweeping all 5 configs on a single GPU.
    """
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")