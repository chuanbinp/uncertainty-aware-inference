"""
run_profiler.py — PyTorch Profiler + Roofline analysis for Mistral-7B PTQ sweep.

Loads each quantized model using the same model_loader.py approach from the
team codebase (gptqmodel for GPTQ, autoawq for AWQ, bitsandbytes for NF4).
Captures kernel breakdown, memory, tok/s, and chrome traces.

Usage:
    python run_profiler.py --config mistral-7b-fp16  --output-dir ./profiler_results
    python run_profiler.py --config mistral-7b-awq-int4 --output-dir ./profiler_results

Called from teamb_vllm_profiler.ipynb.
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Profiler settings
# ─────────────────────────────────────────────────────────────────────────────
PROFILE_PROMPT  = "Explain the implications of quantum computing for cryptography."
N_OUTPUT_TOKENS = 50
WARMUP_STEPS    = 3
PROFILE_STEPS   = 5

# L4 hardware specs for roofline
L4_TFLOPS_FP16  = 121.6   # TFLOPS FP16
L4_BANDWIDTH_TB = 0.300   # TB/s memory bandwidth

# ─────────────────────────────────────────────────────────────────────────────
# Model loader — mirrors model_loader.py but standalone for the profiler
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "mistral-7b-fp16": {
        "hf_id":         "mistralai/Mistral-7B-Instruct-v0.2",
        "quant_type":    "fp16",
        "bits":          16,
    },
    "mistral-7b-gptq-int8": {
        "hf_id":         "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "quant_type":    "gptq",
        "bits":          8,
        "gptq_revision": "gptq-8bit-128g-actorder_True",
    },
    "mistral-7b-gptq-int4": {
        "hf_id":         "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "quant_type":    "gptq",
        "bits":          4,
        "gptq_revision": "gptq-4bit-128g-actorder_True",
    },
    "mistral-7b-awq-int4": {
        "hf_id":         "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "quant_type":    "awq",
        "bits":          4,
    },
    "mistral-7b-nf4": {
        "hf_id":         "mistralai/Mistral-7B-Instruct-v0.2",
        "quant_type":    "nf4",
        "bits":          4,
    },
}


def load_model_for_profiling(config_key: str, hf_token: str | None):
    """
    Load model + tokenizer for the given config.
    Uses gptqmodel (matches the team codebase), autoawq with fuse_layers=False
    (avoids awq_ext CUDA kernel requirement), bitsandbytes for NF4.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg       = MODEL_REGISTRY[config_key]
    hf_id     = cfg["hf_id"]
    quant     = cfg["quant_type"]
    tok_kwargs = {"token": hf_token} if hf_token else {}

    logger.info(f"Loading {config_key} ({quant}) from {hf_id}")
    t0 = time.time()

    def _get_tokenizer(trust=False):
        tok = AutoTokenizer.from_pretrained(
            hf_id, use_fast=True, trust_remote_code=trust, **tok_kwargs
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    if quant == "fp16":
        tokenizer = _get_tokenizer()
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.float16,
            device_map="auto", **tok_kwargs,
        )

    elif quant == "gptq":
        from gptqmodel import GPTQModel
        tokenizer = _get_tokenizer(trust=True)
        model = GPTQModel.from_quantized(
            hf_id,
            revision     = cfg.get("gptq_revision"),
            device_map   = "auto",
            trust_remote_code = True,
            **tok_kwargs,
        )

    elif quant == "awq":
        from awq import AutoAWQForCausalLM
        tokenizer = _get_tokenizer()
        model = AutoAWQForCausalLM.from_quantized(
            hf_id,
            fuse_layers  = False,   # avoids awq_ext CUDA kernel requirement
            device_map   = "auto",
            safetensors  = True,
            **tok_kwargs,
        )

    elif quant == "nf4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.float16,
            bnb_4bit_use_double_quant = True,
        )
        tokenizer = _get_tokenizer()
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_cfg,
            device_map={"": 0}, **tok_kwargs,
        )
    else:
        raise ValueError(f"Unknown quant type: {quant}")

    model.eval()
    load_time = time.time() - t0
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"Loaded in {load_time:.1f}s | GPU mem: {gpu_mem:.2f} GB")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Profiler helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_model_param_gb(model: nn.Module) -> float:
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e9


def estimate_kv_cache_gb(model, seq_len: int, batch_size: int = 1, dtype_bytes: int = 2) -> float:
    config   = model.config
    n_layers = getattr(config, "num_hidden_layers", 32)
    n_heads  = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
    hidden   = getattr(config, "hidden_size", 4096)
    head_dim = hidden // getattr(config, "num_attention_heads", 32)
    return 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes / 1e9


def roofline_analysis(config_key: str, model_param_gb: float, tok_per_sec: float) -> dict:
    """Analytical roofline — arithmetic intensity for prefill and decode."""
    bits = MODEL_REGISTRY[config_key]["bits"]
    dtype_bytes = bits / 8

    # Parameters in elements (use param_gb and dtype_bytes to recover count)
    n_params = model_param_gb * 1e9 / dtype_bytes

    # Approximate FLOPs per token: 2 × n_params (matmul dominant)
    flops_per_token = 2 * n_params

    # Memory bytes read per token (weights) for decode (one token at a time)
    mem_bytes_decode  = n_params * dtype_bytes
    ai_decode         = flops_per_token / mem_bytes_decode    # FLOPs/Byte

    # Prefill approximation (seq_len = 256 tokens)
    seq_len_prefill   = 256
    mem_bytes_prefill = n_params * dtype_bytes
    ai_prefill        = (flops_per_token * seq_len_prefill) / mem_bytes_prefill

    # Ridge point: TFLOPS / bandwidth
    peak_flops  = L4_TFLOPS_FP16 * 1e12
    bandwidth   = L4_BANDWIDTH_TB * 1e12
    ridge_point = peak_flops / bandwidth

    return {
        "ai_decode":    round(ai_decode, 2),
        "ai_prefill":   round(ai_prefill, 2),
        "ridge_point":  round(ridge_point, 2),
        "bound_decode":  "memory" if ai_decode  < ridge_point else "compute",
        "bound_prefill": "memory" if ai_prefill < ridge_point else "compute",
        "l4_peak_tflops":     L4_TFLOPS_FP16,
        "l4_bandwidth_tbps":  L4_BANDWIDTH_TB,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main profiling function
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def profile_model(config_key: str, model, tokenizer, output_dir: Path) -> dict:
    """Run PyTorch profiler and return results dict."""

    # Determine device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda:0")

    inputs    = tokenizer(PROFILE_PROMPT, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    param_gb = get_model_param_gb(model)
    kv_gb    = estimate_kv_cache_gb(model, input_ids.shape[1] + N_OUTPUT_TOKENS)

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info(f"Warmup ({WARMUP_STEPS} steps)…")
    for _ in range(WARMUP_STEPS):
        model.generate(
            input_ids=input_ids, max_new_tokens=N_OUTPUT_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # ── Timed profiling ───────────────────────────────────────────────────────
    logger.info(f"Profiling ({PROFILE_STEPS} steps)…")
    total_time_ms = 0.0

    with profile(
        activities    = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes = True,
        profile_memory= True,
        with_flops    = True,
    ) as prof:
        for step in range(PROFILE_STEPS):
            with record_function(f"inference_step_{step}"):
                t0 = time.perf_counter()
                model.generate(
                    input_ids=input_ids, max_new_tokens=N_OUTPUT_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                total_time_ms += (t1 - t0) * 1000

    avg_time_ms = total_time_ms / PROFILE_STEPS
    tok_per_sec  = N_OUTPUT_TOKENS / (avg_time_ms / 1000)
    peak_mem_gb  = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # ── Chrome trace ──────────────────────────────────────────────────────────
    trace_path = output_dir / f"{config_key}_chrome.json"
    try:
        prof.export_chrome_trace(str(trace_path))
        logger.info(f"Chrome trace → {trace_path}")
    except Exception as e:
        logger.warning(f"Chrome trace export failed (non-fatal): {e}")

    # ── Kernel breakdown ──────────────────────────────────────────────────────
    key_avgs     = prof.key_averages()
    cuda_events  = [e for e in key_avgs if e.cuda_time_total > 0]
    total_cuda_us = sum(e.cuda_time_total for e in cuda_events) or 1

    top_kernels = sorted(cuda_events, key=lambda e: e.cuda_time_total, reverse=True)[:10]
    kernel_list = [
        {
            "name":         e.key,
            "cuda_time_ms": round(e.cuda_time_total / 1e3, 3),
            "pct":          round(e.cuda_time_total / total_cuda_us * 100, 2),
            "cpu_time_ms":  round(e.cpu_time_total / 1e3, 3),
            "calls":        e.count,
        }
        for e in top_kernels
    ]

    total_flops = float(sum(getattr(e, "flops", 0) or 0 for e in key_avgs))

    # ── Roofline ──────────────────────────────────────────────────────────────
    roofline = roofline_analysis(config_key, param_gb, tok_per_sec)

    result = {
        "config_key":    config_key,
        "timing": {
            "avg_inference_ms": round(avg_time_ms, 2),
            "tok_per_sec":      round(tok_per_sec, 2),
        },
        "memory": {
            "peak_gpu_gb":  round(peak_mem_gb, 3),
            "param_gb":     round(param_gb, 3),
            "kv_cache_est_gb": round(kv_gb, 3),
        },
        "compute": {
            "total_cuda_ms": round(total_cuda_us / 1e3, 2),
            "total_flops":   total_flops,
        },
        "top_kernels": kernel_list,
        "roofline":    roofline,
        "profiler_settings": {
            "warmup_steps":    WARMUP_STEPS,
            "profile_steps":   PROFILE_STEPS,
            "n_output_tokens": N_OUTPUT_TOKENS,
        },
    }

    logger.info(f"  {config_key}: {tok_per_sec:.1f} tok/s | {peak_mem_gb:.2f} GB | top kernel: {kernel_list[0]['name'] if kernel_list else 'n/a'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="PyTorch Profiler for Mistral-7B PTQ sweep")
    parser.add_argument("--config",     required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="model config key")
    parser.add_argument("--output-dir", default="./profiler_results",
                        help="directory to save JSON and chrome trace")
    parser.add_argument("--hf-token",   default=None,
                        help="HuggingFace token (falls back to HF_TOKEN env var)")
    parser.add_argument("--wandb-run-id",  default=None,
                        help="W&B run ID to log into (optional)")
    parser.add_argument("--wandb-project", default="UAI_Project",
                        help="W&B project name")
    args = parser.parse_args()

    hf_token   = args.hf_token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_for_profiling(args.config, hf_token)

    result = profile_model(args.config, model, tokenizer, output_dir)

    # Save JSON
    out_path = output_dir / f"{args.config}_profile.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved → {out_path}")

    # Optional W&B logging
    if args.wandb_run_id:
        try:
            import wandb
            run = wandb.init(
                project = args.wandb_project,
                entity  = "Uncertainty_Aware_Inference_Lab",
                id      = args.wandb_run_id,
                resume  = "allow",
                name    = f"teamB_{args.config}_profiler",
                config  = {
                    "model":       "mistral-7b",
                    "team":        "team-b",
                    "config_key":  args.config,
                    "experiment":  "pytorch_profiler",
                },
            )
            run.log({
                "profiler/tok_per_sec":      result["timing"]["tok_per_sec"],
                "profiler/avg_inference_ms": result["timing"]["avg_inference_ms"],
                "profiler/peak_gpu_gb":      result["memory"]["peak_gpu_gb"],
                "profiler/total_flops":      result["compute"]["total_flops"],
                "roofline/ai_decode":        result["roofline"]["ai_decode"],
                "roofline/ai_prefill":       result["roofline"]["ai_prefill"],
            })
            # Log top kernel table
            kernel_table = wandb.Table(
                columns=["name", "cuda_time_ms", "pct", "calls"],
                data=[[k["name"], k["cuda_time_ms"], k["pct"], k["calls"]]
                      for k in result["top_kernels"]],
            )
            run.log({f"profiler/top_kernels_{args.config}": kernel_table})
            run.finish()
            logger.info("W&B logging complete")
        except Exception as e:
            logger.warning(f"W&B logging failed (non-fatal): {e}")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Print summary for notebook capture
    print(f"\n=== PROFILER RESULT: {args.config} ===")
    print(f"  Avg inference   : {result['timing']['avg_inference_ms']:.1f} ms")
    print(f"  Tok/s           : {result['timing']['tok_per_sec']:.1f}")
    print(f"  Peak GPU mem    : {result['memory']['peak_gpu_gb']:.3f} GB")
    print(f"  Param size      : {result['memory']['param_gb']:.3f} GB")
    print(f"  AI (decode)     : {result['roofline']['ai_decode']:.2f} FLOPs/Byte")
    print(f"  AI (prefill)    : {result['roofline']['ai_prefill']:.2f} FLOPs/Byte")
    print(f"  Bound (decode)  : {result['roofline']['bound_decode']}")
    if result["top_kernels"]:
        k = result["top_kernels"][0]
        print(f"  Top kernel      : {k['name']} ({k['pct']:.1f}%)")
    print(f"  Saved to        : {out_path}")


if __name__ == "__main__":
    main()
