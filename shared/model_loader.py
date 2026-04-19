import gc
import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
)
from awq import AutoAWQForCausalLM
from gptqmodel import GPTQModel


def _load_tokenizer(hf_id: str, hf_token: str | None, trust_remote_code: bool = False):
    """Load tokenizer with pad token fallback."""
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, token=hf_token,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_fp16(hf_id, hf_token, **kwargs):
    tokenizer = _load_tokenizer(hf_id, hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16,
        device_map="auto", token=hf_token,
    )
    return model, tokenizer


def _load_nf4(hf_id, hf_token, **kwargs):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = _load_tokenizer(hf_id, hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, quantization_config=bnb_config,
        device_map={"": 0}, token=hf_token,
    )
    return model, tokenizer


def _load_awq(hf_id, hf_token, **kwargs):
    tokenizer = _load_tokenizer(hf_id, hf_token)
    model = AutoAWQForCausalLM.from_quantized(
        hf_id, fuse_layers=False,
        device_map="auto", safetensors=True,
        token=hf_token,
    )
    return model, tokenizer


def _load_gptq(hf_id, hf_token, bits, revision=None, **kwargs):
    tokenizer = _load_tokenizer(hf_id, hf_token, trust_remote_code=True)
    model = GPTQModel.from_quantized(
        hf_id,
        revision=revision,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    return model, tokenizer


_LOADERS = {
    "fp16": _load_fp16,
    "nf4":  _load_nf4,
    "awq":  _load_awq,
    "gptq": _load_gptq,
}


def load_model(config_key: str, registry: dict, hf_token: str | None = None):
    """Load model and tokenizer by registry key.   # ← registry passed in from outside
    
    Args:
        config_key: key in the provided registry dict
        registry:   MODEL_REGISTRY from the calling team's configs.py
        hf_token:   HuggingFace token, falls back to HF_TOKEN env var
    """
    if config_key not in registry:
        valid = ", ".join(registry.keys())
        raise ValueError(
            f"Unknown config '{config_key}'. Valid: {valid}"
        )

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    entry = registry[config_key]
    quant_type = entry["quant_type"]

    if quant_type not in _LOADERS:
        raise ValueError(f"Unhandled quant_type '{quant_type}'")

    model, tokenizer = _LOADERS[quant_type](
        entry["hf_id"], hf_token,
        bits=entry.get("bits"),
        revision=entry.get("gptq_revision"),
    )

    model.eval()
    print(f"Loaded: {config_key} on {next(model.parameters()).device}")
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")