"""
profiling/pytorch_profiler.py

PyTorch Profiler harness for CUDA kernel analysis.

Captures:
- GPU kernel execution times per operation
- Memory allocation/deallocation events
- CUDA activity timelines
- FLOPs estimation per layer
- Chrome trace for Perfetto/TensorBoard visualization

Usage:
    python profiling/pytorch_profiler.py \
        --model-path mistralai/Mistral-7B-Instruct-v0.2 \
        --precision fp16 \
        --output-dir results/profiles
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ProfilingResult:
    """Container for profiling outputs from a single model/precision configuration."""

    def __init__(self, model_name: str, precision: str):
        self.model_name = model_name
        self.precision = precision

        # Timing
        self.total_inference_time_ms: float = 0.0
        self.prefill_time_ms: float = 0.0
        self.decode_time_ms: float = 0.0
        self.tokens_per_second: float = 0.0

        # Memory
        self.peak_gpu_memory_gb: float = 0.0
        self.model_memory_gb: float = 0.0
        self.kv_cache_memory_gb: float = 0.0

        # Kernel analysis
        self.top_kernels: list[dict] = []  # [{name, time_ms, pct}]
        self.total_cuda_time_ms: float = 0.0
        self.total_cpu_time_ms: float = 0.0

        # FLOPs
        self.total_flops: float = 0.0
        self.arithmetic_intensity: float = 0.0  # FLOPs / bytes

        # Compute vs memory bound
        self.is_compute_bound: bool = False
        self.roofline_efficiency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "precision": self.precision,
            "timing": {
                "total_inference_ms": self.total_inference_time_ms,
                "tokens_per_second": self.tokens_per_second,
                "prefill_ms": self.prefill_time_ms,
                "decode_ms": self.decode_time_ms,
            },
            "memory": {
                "peak_gpu_gb": self.peak_gpu_memory_gb,
                "model_gb": self.model_memory_gb,
                "kv_cache_gb": self.kv_cache_memory_gb,
            },
            "compute": {
                "total_cuda_ms": self.total_cuda_time_ms,
                "total_flops": self.total_flops,
                "arithmetic_intensity": self.arithmetic_intensity,
                "is_compute_bound": self.is_compute_bound,
                "roofline_efficiency": self.roofline_efficiency,
            },
            "top_kernels": self.top_kernels,
        }


def get_model_memory_gb(model: nn.Module) -> float:
    """Estimate model parameter memory in GB."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total_bytes / 1e9


def get_gpu_memory_stats() -> dict:
    """Return current and peak GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def estimate_kv_cache_size(
    model,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # fp16 = 2 bytes
) -> float:
    """
    Estimate KV cache memory in GB.
    KV cache = 2 (K+V) * n_layers * n_heads * head_dim * seq_len * batch * dtype_bytes
    """
    config = model.config
    n_layers = getattr(config, "num_hidden_layers", 32)
    n_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
    hidden_size = getattr(config, "hidden_size", 4096)
    head_dim = hidden_size // getattr(config, "num_attention_heads", 32)

    kv_bytes = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes
    return kv_bytes / 1e9


@torch.no_grad()
def profile_inference(
    model,
    tokenizer,
    prompt: str = "Explain quantum computing in one sentence.",
    n_tokens: int = 50,
    warmup_steps: int = 3,
    profile_steps: int = 5,
    output_dir: str = "results/profiles",
    model_name: str = "model",
    precision: str = "fp16",
    export_chrome_trace: bool = True,
) -> ProfilingResult:
    """
    Profile model inference using PyTorch Profiler.

    Runs warmup steps (not profiled), then profiles `profile_steps` runs.
    Records GPU kernels, memory, and FLOPs.

    Args:
        model: Loaded HuggingFace model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt for profiling
        n_tokens: Number of tokens to generate per run
        warmup_steps: Number of warmup inference steps
        profile_steps: Number of profiled inference steps
        output_dir: Directory for trace files
        model_name, precision: Metadata
        export_chrome_trace: Whether to export .json chrome trace

    Returns:
        ProfilingResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    result = ProfilingResult(model_name, precision)
    result.model_memory_gb = get_model_memory_gb(model)
    result.kv_cache_memory_gb = estimate_kv_cache_size(model, seq_len + n_tokens)

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info(f"Warmup ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        _ = model.generate(input_ids=input_ids, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize()

    # ── Reset memory stats ────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()

    # Profiled runs (no schedule - schedule caused empty traces)
    # Using a simple context manager so all generate() CUDA kernels are captured.
    logger.info(f"Profiling ({profile_steps} steps)...")

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for step in range(profile_steps):
            with record_function(f"inference_step_{step}"):
                t_start = time.perf_counter()
                _ = model.generate(input_ids=input_ids, max_new_tokens=n_tokens, do_sample=False)
                torch.cuda.synchronize()
                t_end = time.perf_counter()
                result.total_inference_time_ms += (t_end - t_start) * 1000

    # Export chrome trace after context exits (all steps captured)
    chrome_path = output_dir / f"{model_name}_{precision}_chrome.json"
    prof.export_chrome_trace(str(chrome_path))
    logger.info(f"Chrome trace saved to {chrome_path}")

    result.total_inference_time_ms /= profile_steps
    result.tokens_per_second = n_tokens / (result.total_inference_time_ms / 1000)

    # ── Memory stats ──────────────────────────────────────────────────────────
    mem_stats = get_gpu_memory_stats()
    result.peak_gpu_memory_gb = mem_stats.get("peak_allocated_gb", 0.0)

    # ── Kernel analysis ───────────────────────────────────────────────────────
    key_averages = prof.key_averages()

    cuda_events = [e for e in key_averages if e.cuda_time_total > 0]
    total_cuda_us = sum(e.cuda_time_total for e in cuda_events)
    result.total_cuda_time_ms = total_cuda_us / 1e3

    cpu_events = [e for e in key_averages if e.cpu_time_total > 0]
    result.total_cpu_time_ms = sum(e.cpu_time_total for e in cpu_events) / 1e3

    # Top 10 kernels by CUDA time
    top_kernels = sorted(cuda_events, key=lambda e: e.cuda_time_total, reverse=True)[:10]
    result.top_kernels = [
        {
            "name": e.key,
            "cuda_time_ms": e.cuda_time_total / 1e3,
            "pct": e.cuda_time_total / total_cuda_us * 100 if total_cuda_us > 0 else 0,
            "cpu_time_ms": e.cpu_time_total / 1e3,
            "calls": e.count,
        }
        for e in top_kernels
    ]

    # FLOPs
    total_flops = sum(getattr(e, "flops", 0) or 0 for e in key_averages)
    result.total_flops = float(total_flops)

    # Chrome trace export (already saved by profiler context)

    # ── Log results ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"Profiling Results: {model_name} @ {precision}")
    logger.info(f"{'='*50}")
    logger.info(f"  Inference time:  {result.total_inference_time_ms:.1f} ms")
    logger.info(f"  Tokens/sec:      {result.tokens_per_second:.1f}")
    logger.info(f"  Peak GPU mem:    {result.peak_gpu_memory_gb:.2f} GB")
    logger.info(f"  Model mem:       {result.model_memory_gb:.2f} GB")
    logger.info(f"  KV cache est:    {result.kv_cache_memory_gb:.2f} GB")
    logger.info(f"  Total CUDA time: {result.total_cuda_time_ms:.1f} ms")
    logger.info(f"  Total FLOPs:     {result.total_flops:.2e}")
    logger.info(f"\nTop 5 CUDA kernels:")
    for k in result.top_kernels[:5]:
        logger.info(f"  {k['name'][:50]:50s} {k['cuda_time_ms']:8.2f}ms  ({k['pct']:.1f}%)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = output_dir / f"{model_name}_{precision}_profile.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"\nProfile saved to {json_path}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Profile LLM inference with PyTorch Profiler")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="mistral-7b")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "gptq_int8", "gptq_int4", "awq_int4", "nf4"])
    parser.add_argument("--n-tokens", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/profiles")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load model (simplified — use calibration/evaluate.py load_model_and_tokenizer in practice)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    result = profile_inference(
        model=model,
        tokenizer=tokenizer,
        n_tokens=args.n_tokens,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        output_dir=args.output_dir,
        model_name=args.model_name,
        precision=args.precision,
    )
