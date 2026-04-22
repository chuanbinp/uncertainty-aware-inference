"""
run_vllm.py — vLLM throughput benchmarking for Mistral-7B PTQ sweep.

Runs throughput measurement for a given config key. NF4 falls back to
HuggingFace batched generation (vLLM does not support bitsandbytes NF4).

Usage:
    python run_vllm.py --config mistral-7b-fp16 --output-dir ./vllm_results
    python run_vllm.py --config mistral-7b-nf4  --output-dir ./vllm_results

Called from teamb_vllm_profiler.ipynb.
"""

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import torch


def get_gpu_mem_gb() -> float:
    """
    Read actual GPU memory usage directly from the driver via nvidia-smi.
    This is reliable regardless of which memory allocator is used (PyTorch,
    vLLM's CUDA pool, bitsandbytes, etc.).
    Returns 0.0 if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        # Sum across all GPUs (handles multi-GPU setups)
        mib_values = [float(x.strip()) for x in out.stdout.strip().splitlines() if x.strip()]
        return sum(mib_values) / 1024.0  # MiB → GB
    except Exception:
        # Fallback to PyTorch allocator (less accurate for vLLM, but better than 0)
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config map: config_key → vLLM load params
# Key rule: gptq_int4 uses quantization=None so vLLM auto-selects the Marlin
# kernel (2.8× faster than explicit gptq=True). gptq_int8 must use "gptq"
# because there is no Marlin INT8 kernel.
# ─────────────────────────────────────────────────────────────────────────────
VLLM_CONFIGS = {
    "mistral-7b-fp16": {
        "model":        "mistralai/Mistral-7B-Instruct-v0.2",
        "quantization": None,
        "revision":     None,
        "dtype":        "float16",
    },
    "mistral-7b-gptq-int4": {
        "model":        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "quantization": None,           # ← None forces Marlin kernel auto-detect
        "revision":     "main",
        "dtype":        "float16",
    },
    "mistral-7b-gptq-int8": {
        "model":        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        "quantization": "gptq",         # ← explicit gptq (no INT8 Marlin)
        "revision":     "gptq-8bit-128g-actorder_True",
        "dtype":        "float16",
    },
    "mistral-7b-awq-int4": {
        "model":        "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        "quantization": "awq",
        "revision":     None,
        "dtype":        "float16",
    },
    # NF4 is handled via HF fallback below — no vLLM entry
}

# Prompts used for throughput measurement
BENCHMARK_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of climate change?",
    "Describe the process of protein synthesis in cells.",
    "What is the significance of the Turing test in AI?",
    "How does the immune system fight bacterial infections?",
    "Explain the difference between supervised and unsupervised learning.",
    "What are the ethical implications of genetic engineering?",
    "Describe how quantum computers differ from classical computers.",
    "What is the historical significance of the Silk Road?",
    "How does photosynthesis work in plants?",
]

N_OUTPUT_TOKENS  = 128    # tokens to generate per prompt
WARMUP_ITERS     = 2      # discarded warmup runs
BENCHMARK_ITERS  = 3      # averaged benchmark runs


def run_vllm_benchmark(config_key: str, hf_token: str | None, output_dir: Path) -> dict:
    """Run vLLM throughput benchmark for a vLLM-supported config."""
    from vllm import LLM, SamplingParams

    cfg = VLLM_CONFIGS[config_key]
    logger.info(f"Loading {config_key} via vLLM — model={cfg['model']}, quant={cfg['quantization']}")

    load_kwargs = dict(
        model       = cfg["model"],
        dtype       = cfg["dtype"],
        gpu_memory_utilization = 0.90,
        trust_remote_code      = True,
        max_model_len          = 2048,
    )
    if cfg["quantization"] is not None:
        load_kwargs["quantization"] = cfg["quantization"]
    if cfg["revision"] is not None:
        load_kwargs["revision"] = cfg["revision"]
    # vLLM does not accept tokenizer_kwargs in LLM(). The correct approach is
    # to set HF_TOKEN in the environment before calling LLM(), which the
    # notebook already does via: os.environ["HF_TOKEN"] = HF_TOKEN
    # No additional kwarg needed here.

    llm = LLM(**load_kwargs)

    sampling_params = SamplingParams(
        temperature = 0.0,
        max_tokens  = N_OUTPUT_TOKENS,
    )

    # Detect which kernel is active (Marlin vs gptq vs cuBLAS)
    kernel_label = "cuBLAS"
    if cfg["quantization"] == "awq":
        kernel_label = "fused_GEMV (AWQ)"
    elif cfg["quantization"] == "gptq":
        kernel_label = "gptq (exllama)"
    elif config_key == "mistral-7b-gptq-int4":
        kernel_label = "gptq_marlin"

    # Warmup
    logger.info(f"Warmup ({WARMUP_ITERS} iters)…")
    for _ in range(WARMUP_ITERS):
        llm.generate(BENCHMARK_PROMPTS, sampling_params)

    # Benchmark
    logger.info(f"Benchmarking ({BENCHMARK_ITERS} iters)…")
    iter_tok_per_sec = []
    for i in range(BENCHMARK_ITERS):
        t0 = time.perf_counter()
        outputs = llm.generate(BENCHMARK_PROMPTS, sampling_params)
        t1 = time.perf_counter()
        n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        iter_tok_per_sec.append(n_tokens / (t1 - t0))
        logger.info(f"  iter {i+1}: {iter_tok_per_sec[-1]:.1f} tok/s")

    avg_tok_per_sec = sum(iter_tok_per_sec) / len(iter_tok_per_sec)
    # Use nvidia-smi directly — vLLM allocates its own CUDA memory pool that
    # bypasses torch.cuda.max_memory_allocated(), which would return 0.
    peak_mem_gb = get_gpu_mem_gb()

    result = {
        "config_key":    config_key,
        "model":         cfg["model"],
        "quantization":  cfg["quantization"],
        "revision":      cfg["revision"],
        "kernel":        kernel_label,
        "avg_tok_per_sec": round(avg_tok_per_sec, 2),
        "iter_tok_per_sec": [round(x, 2) for x in iter_tok_per_sec],
        "peak_gpu_mem_gb": round(peak_mem_gb, 3),
        "n_prompts":     len(BENCHMARK_PROMPTS),
        "n_output_tokens": N_OUTPUT_TOKENS,
    }

    logger.info(f"  → {avg_tok_per_sec:.1f} tok/s  |  {peak_mem_gb:.2f} GB peak  |  kernel={kernel_label}")
    return result


def run_hf_nf4_benchmark(hf_token: str | None, output_dir: Path) -> dict:
    """
    HuggingFace batched-generation fallback for NF4 (vLLM does not support
    bitsandbytes NF4 quantization).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("NF4: using HF batched generation fallback (vLLM unsupported)")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.float16,
        bnb_4bit_use_double_quant  = True,
    )

    tok_kwargs = {"token": hf_token} if hf_token else {}
    model_id   = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg,
        device_map={"": 0}, **tok_kwargs,
    )
    model.eval()

    peak_after_load = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    # Warmup
    logger.info(f"NF4 warmup ({WARMUP_ITERS} iters)…")
    inputs = tokenizer(
        BENCHMARK_PROMPTS, return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    ).to("cuda")
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model.generate(**inputs, max_new_tokens=N_OUTPUT_TOKENS, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    logger.info(f"NF4 benchmarking ({BENCHMARK_ITERS} iters)…")
    iter_tok_per_sec = []
    with torch.no_grad():
        for i in range(BENCHMARK_ITERS):
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=N_OUTPUT_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            t1 = time.perf_counter()
            n_tokens = (out[:, inputs["input_ids"].shape[1]:] != tokenizer.eos_token_id).sum().item()
            iter_tok_per_sec.append(n_tokens / (t1 - t0))
            logger.info(f"  iter {i+1}: {iter_tok_per_sec[-1]:.1f} tok/s")

    avg_tok_per_sec = sum(iter_tok_per_sec) / len(iter_tok_per_sec)
    # nvidia-smi for consistency with vLLM configs
    peak_mem_gb = get_gpu_mem_gb()

    result = {
        "config_key":       "mistral-7b-nf4",
        "model":            model_id,
        "quantization":     "nf4_bitsandbytes",
        "revision":         None,
        "kernel":           "bitsandbytes_HF_batched",
        "avg_tok_per_sec":  round(avg_tok_per_sec, 2),
        "iter_tok_per_sec": [round(x, 2) for x in iter_tok_per_sec],
        "peak_gpu_mem_gb":  round(peak_mem_gb, 3),
        "n_prompts":        len(BENCHMARK_PROMPTS),
        "n_output_tokens":  N_OUTPUT_TOKENS,
        "note":             "HF batched fallback — vLLM does not support bitsandbytes NF4",
    }

    logger.info(f"  NF4 → {avg_tok_per_sec:.1f} tok/s  |  {peak_mem_gb:.2f} GB peak")
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="vLLM throughput benchmark for Mistral-7B PTQ sweep")
    parser.add_argument("--config",     required=True,
                        choices=list(VLLM_CONFIGS.keys()) + ["mistral-7b-nf4"],
                        help="config key to benchmark")
    parser.add_argument("--output-dir", default="./vllm_results",
                        help="directory to save JSON results")
    parser.add_argument("--hf-token",   default=None,
                        help="HuggingFace token (falls back to HF_TOKEN env var)")
    parser.add_argument("--wandb-run-id", default=None,
                        help="W&B run ID to log into (optional)")
    parser.add_argument("--wandb-project", default="UAI_Project",
                        help="W&B project name")
    args = parser.parse_args()

    hf_token   = args.hf_token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config == "mistral-7b-nf4":
        result = run_hf_nf4_benchmark(hf_token, output_dir)
    else:
        result = run_vllm_benchmark(args.config, hf_token, output_dir)

    # Save JSON
    out_path = output_dir / f"{args.config}_vllm.json"
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
                name    = f"teamB_{args.config}_vllm",
                config  = {
                    "model":        "mistral-7b",
                    "team":         "team-b",
                    "config_key":   args.config,
                    "quant_method": result.get("quantization", "none"),
                    "kernel":       result.get("kernel"),
                    "experiment":   "vllm_throughput",
                },
            )
            run.log({
                "vllm/avg_tok_per_sec": result["avg_tok_per_sec"],
                "vllm/peak_gpu_mem_gb": result["peak_gpu_mem_gb"],
                "vllm/kernel":          result["kernel"],
            })
            run.finish()
            logger.info("W&B logging complete")
        except Exception as e:
            logger.warning(f"W&B logging failed (non-fatal): {e}")

    # Print summary for notebook capture
    print(f"\n=== vLLM RESULT: {args.config} ===")
    print(f"  Kernel          : {result['kernel']}")
    print(f"  Avg tok/s       : {result['avg_tok_per_sec']:.2f}")
    print(f"  Peak GPU mem    : {result['peak_gpu_mem_gb']:.3f} GB")
    print(f"  Saved to        : {out_path}")


if __name__ == "__main__":
    main()
