"""
run_profiler.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixed PyTorch Profiler sweep for all Mistral-7B PTQ configs.

Reads model IDs, quant types, and revisions directly from
MODEL_REGISTRY in configs.py — no other constants required.

Key fixes vs. original pytorch_profiler.py:
  ✓ with_modules=False   (was True → caused torch.fx tracing errors on all HF models)
  ✓ with_stack=False     (default; True produces 500MB+ traces on A100)
  ✓ cuda.synchronize()   inside profiler context (CUDA events flushed before trace)
  ✓ export_chrome_trace  inside 'with profile()' block (avoids empty-trace bug)
  ✓ profile_memory=False (was True → caused kineto_results=None on some PyTorch builds)
  ✓ cuda.synchronize()   before export_chrome_trace (forces Kineto to finalize)
  ✓ device = cuda:0      (not next(model.parameters()).device — returns 'meta' for quant layers)
  ✓ kernel times         are per-step averages (key_averages() accumulates across all steps)
  ✓ --nvtx flag          emits NVTX markers for nsys/ncu capture-range targeting

Usage:
    # Profile one config by its MODEL_REGISTRY key:
    python run_profiler.py --config mistral-7b-gptq-int4

    # Profile all configs sequentially:
    python run_profiler.py --all

    # Force re-profile even if JSON exists:
    python run_profiler.py --all --force

    # With NVTX markers (required when launched by nsys or ncu):
    python run_profiler.py --all --nvtx

    # With W&B logging:
    python run_profiler.py --config mistral-7b-fp16 \
        --wandb-project UAI_Project \
        --wandb-entity Uncertainty_Aware_Inference_Lab \
        --wandb-api-key <key>

    # With a pre-initialised W&B run ID (used by the notebook):
    python run_profiler.py --config mistral-7b-fp16 \
        --wandb-project UAI_Project \
        --wandb-run-id <run_id>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys

# On GCP with a proper CUDA 12.x install, PyTorch finds its CUDA libraries
# without any LD_LIBRARY_PATH manipulation.  The Colab-specific CUPTI path
# hack that was here previously caused libtorch_cuda.so to fail to load on
# GCP L4 instances (RuntimeError: Cannot initialize CUDA without ATen_cuda
# library), so it has been removed.

# Keep Kineto in synchronous mode — the daemon IPC socket causes spurious
# "Bad file descriptor" warnings in GCP's /tmp environment.
os.environ["KINETO_USE_DAEMON"] = "0"
os.environ["KINETO_DAEMON_INIT_WAIT_USECS"] = "50000"
os.environ["CUDA_INJECTION64_PATH"] = "none"  # disable CUPTI injection when under nsys

import argparse
import contextlib
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

# ── NVTX annotations (optional — only active when --nvtx is passed) ─────────
# nvtx_utils falls back to no-ops when torch.cuda.nvtx is unavailable,
# so this import is always safe regardless of environment.
from nvtx_utils import (
    NVTX_AVAILABLE,
    probe_nvtx,
    warmup_range,
    profiling_region,
    profile_step_range,
    generate_range,
)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from configs import MODEL_REGISTRY

# ─────────────────────────────────────────────────────────────
# Constants derived from MODEL_REGISTRY
# ─────────────────────────────────────────────────────────────

ALL_CONFIGS          = list(MODEL_REGISTRY.keys())
DEFAULT_PROFILE_DIR  = Path("./profiler_results")   # GCP-friendly relative path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Kineto probe — must run before any model load
# ─────────────────────────────────────────────────────────────

def _probe_kineto() -> bool:
    """
    Verify Kineto CUDA is working by running a tiny profiler context.
    Must be called BEFORE any model is loaded.
    Returns True if real GPU kernel stats are available, False otherwise.
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA device found — profiling will use CPU-only mode.")
        return False

    # ── Safe CUDA initialisation check ───────────────────────────────────────
    # torch.cuda.is_available() only checks whether the driver is present; it
    # does not actually initialise the CUDA context. torch.zeros(..., device="cuda")
    # triggers _lazy_init() which does the real init and raises RuntimeError if
    # libtorch_cuda.so cannot be loaded.  Catch that here so the script exits
    # with a clear message rather than a raw C++ traceback.
    try:
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except RuntimeError as e:
        logger.error(
            f"CUDA initialisation failed: {e}\n"
            "This usually means PyTorch cannot load libtorch_cuda.so. Common causes:\n"
            "  1. A modified LD_LIBRARY_PATH is shadowing a required library —\n"
            "     unset LD_LIBRARY_PATH and retry.\n"
            "  2. The installed PyTorch is CPU-only — reinstall the GPU build:\n"
            "     pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  3. CUDA driver/runtime version mismatch — check: nvidia-smi && nvcc --version"
        )
        return False

    import glob
    cupti_libs = glob.glob("/usr/local/cuda*/**/libcupti.so*", recursive=True)
    if cupti_libs:
        logger.info(f"  CUPTI found: {cupti_libs[0]}")
    else:
        logger.warning(
            "  libcupti.so not found — GPU kernel timing may fall back to CPU-side times.\n"
            "  On GCP: sudo apt-get install -y cuda-cupti-12-4"
        )

    torch.cuda.reset_peak_memory_stats()

    import tempfile
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_flops=False,
            with_stack=False,
            with_modules=False,
        ) as _prof:
            _ = torch.mm(
                torch.randn(64, 64, device="cuda"),
                torch.randn(64, 64, device="cuda"),
            )
            torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        _prof.export_chrome_trace(tmp_path)
        os.unlink(tmp_path)

        avgs = _prof.key_averages()
        def _any_cuda(e):
            for attr in ("cuda_time_total", "self_cuda_time_total", "device_time_total"):
                v = getattr(e, attr, None)
                if v and v > 0:
                    return True
            return False
        has_cuda = any(_any_cuda(e) for e in avgs)
        if has_cuda:
            logger.info("Kineto CUDA backend: ACTIVE — real GPU kernel stats available")
        else:
            logger.warning(
                "Kineto CUDA backend: active but no cuda_time_total found. "
                "Kernel stats will reflect CPU-side scheduling times."
            )
        return has_cuda

    except AttributeError:
        logger.warning(
            "Kineto CUDA backend: NOT active (kineto_results=None). "
            "Timing and per-kernel stats are unaffected; chrome trace will be skipped."
        )
        return False

    except RuntimeError as e:
        logger.warning(f"Kineto probe raised RuntimeError: {e} — continuing without GPU kernel stats.")
        return False


# ─────────────────────────────────────────────────────────────
# Model loader — dispatches by quant_type from MODEL_REGISTRY
# ─────────────────────────────────────────────────────────────

def load_model_for_profiling(config_key: str):
    """
    Load model + tokenizer for a given MODEL_REGISTRY key.
    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg        = MODEL_REGISTRY[config_key]
    hf_id      = cfg["hf_id"]
    quant_type = cfg["quant_type"]
    bits       = cfg["bits"]

    logger.info(f"Loading [{config_key}]  hf_id={hf_id}  quant={quant_type}  bits={bits}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_type == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    elif quant_type == "gptq":
        try:
            from auto_gptq import AutoGPTQForCausalLM as GPTQModel
        except ImportError:
            from auto_gptq import AutoGPTQForCausalLM as GPTQModel
        revision = cfg.get("gptq_revision", "main")
        logger.info(f"  GPTQ revision: {revision}")
        model = AutoGPTQForCausalLM.from_quantized(
            hf_id,
            revision=revision,
            device="cuda:0",
            trust_remote_code=True,
        )

    elif quant_type == "awq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            hf_id,
            fuse_layers=False,
            trust_remote_code=True,
        )

    elif quant_type == "nf4":
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        raise ValueError(f"Unknown quant_type '{quant_type}' for config '{config_key}'")

    model.eval()

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  GPU memory after load: {mem_gb:.2f} GB")
        expected_gb = bits / 16 * 14.0
        if mem_gb > expected_gb * 1.5:
            logger.warning(
                f"  Expected ~{expected_gb:.1f} GB for {bits}-bit but got {mem_gb:.1f} GB — "
                f"model may not be correctly quantized."
            )

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Fixed profiler harness
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def profile_inference_fixed(
    model,
    tokenizer,
    config_key: str,
    prompt: str = "Explain transformer attention in one sentence.",
    n_tokens: int = 50,
    warmup_steps: int = 3,
    profile_steps: int = 5,
    output_dir: Path = DEFAULT_PROFILE_DIR,
    export_chrome_trace: bool = True,
    with_stack: bool = False,
    use_nvtx: bool = False,     # emit NVTX markers for nsys/ncu capture-range targeting
    wandb_run=None,
) -> dict:
    """
    Fixed PyTorch Profiler harness for HuggingFace LLMs.

    Returns a dict with timing / memory / compute / top_kernels.
    Also saves {output_dir}/{config_key}_profile.json.
    All kernel times are per-step averages (key_averages accumulates across all steps).

    When use_nvtx=True, emits NVTX markers so that nsys and ncu can filter
    collection to just the profiling steps (not warmup or model load):
        nsys profile --capture-range=nvtx --nvtx-capture="profiling_region" ...
        ncu --nvtx --nvtx-include="profiling_region" ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = MODEL_REGISTRY[config_key]

    # Use cuda:0 explicitly — next(model.parameters()).device can return 'meta'
    # for quantized layers (bitsandbytes, GPTQ) with device_map="auto".
    device = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
    inputs         = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # ── Warmup ────────────────────────────────────────────────────────────────
    # NVTX: warmup steps are marked YELLOW and sit OUTSIDE the capture range.
    # nsys / ncu will not record these steps.
    logger.info(f"[{config_key}] Warmup ({warmup_steps} steps)...")
    for _wi in range(warmup_steps):
        _wctx = warmup_range(_wi) if use_nvtx else contextlib.nullcontext()
        with _wctx:
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # ── Profiling ─────────────────────────────────────────────────────────────
    # NVTX: the outer "profiling_region" range is what nsys --capture-range=nvtx
    # and ncu --nvtx-include latch onto. Each step gets a child range, and each
    # generate() call gets an inner "generate/..." range.
    timing_ms_list = []
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    logger.info(f"[{config_key}] Profiling ({profile_steps} steps)...")

    _outer = profiling_region() if use_nvtx else contextlib.nullcontext()
    with _outer:
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_flops=True,
            with_stack=with_stack,
            with_modules=False,
        ) as prof:
            for step in range(profile_steps):
                _step = profile_step_range(step) if use_nvtx else contextlib.nullcontext()
                with _step:
                    with record_function(f"inference_step_{step}"):
                        t0 = time.perf_counter()
                        _gen = generate_range(step, n_tokens) if use_nvtx else contextlib.nullcontext()
                        with _gen:
                            model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=n_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        timing_ms_list.append((time.perf_counter() - t0) * 1000)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Export chrome trace — guarded against kineto_results=None
            if export_chrome_trace:
                trace_path = output_dir / f"{config_key}_chrome.json"
                try:
                    prof.export_chrome_trace(str(trace_path))
                    logger.info(f"  Chrome trace → {trace_path}")
                except AttributeError:
                    logger.warning(
                        "  Chrome trace skipped: kineto_results=None "
                        "(Kineto CUDA backend did not initialize on this build). "
                        "Timing and kernel stats are unaffected."
                    )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg_ms = float(np.mean(timing_ms_list))
    tps    = n_tokens / (avg_ms / 1000) if avg_ms > 0 else 0.0

    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    key_avgs = prof.key_averages()

    def _get_cuda_time(e):
        for attr in ("cuda_time_total", "self_cuda_time_total", "device_time_total"):
            v = getattr(e, attr, None)
            if v is not None:
                return v
        return 0.0

    sample_cuda_times = [_get_cuda_time(e) for e in key_avgs]
    has_cuda_times    = any(v > 0 for v in sample_cuda_times)

    _SPAN_PREFIXES = ("inference_step_",)

    def _is_real_kernel(e):
        return not any(e.key.startswith(p) for p in _SPAN_PREFIXES)

    if has_cuda_times:
        logger.info(f"  [{config_key}] Kineto CUDA active — using real GPU kernel times")
        cuda_events     = [e for e in key_avgs if _get_cuda_time(e) > 0]
        total_cuda_us   = sum(_get_cuda_time(e) for e in cuda_events)
        real_events     = [e for e in cuda_events if _is_real_kernel(e)]
        top_kernels_raw = sorted(real_events, key=_get_cuda_time, reverse=True)[:10]
    else:
        logger.warning(
            f"  [{config_key}] No GPU kernel times found — falling back to cpu_time_total. "
            f"Kernel stats will reflect CPU-side scheduling, not actual GPU execution."
        )
        cuda_events     = [e for e in key_avgs if e.cpu_time_total > 0]
        total_cuda_us   = sum(e.cpu_time_total for e in cuda_events)
        real_events     = [e for e in cuda_events if _is_real_kernel(e)]
        top_kernels_raw = sorted(real_events, key=lambda e: e.cpu_time_total, reverse=True)[:10]

    avg_cuda_us        = total_cuda_us / profile_steps
    total_flops_all    = sum(getattr(e, "flops", 0) or 0 for e in key_avgs)
    avg_flops_per_step = total_flops_all / profile_steps

    if total_flops_all == 0:
        logger.warning(
            f"  [{config_key}] total_flops=0 — PyTorch flops counter does not support "
            f"custom quant kernels (GPTQ/AWQ/NF4). arithmetic_intensity will be 0."
        )

    arith_intensity = avg_flops_per_step / (mem_gb * 1e9) if mem_gb > 0 else 0.0

    def _kern_time_us(e):
        return _get_cuda_time(e) if has_cuda_times else e.cpu_time_total

    result = {
        "config_key":   config_key,
        "hf_id":        cfg["hf_id"],
        "quant_type":   cfg["quant_type"],
        "bits":         cfg["bits"],
        "kineto_cuda":  has_cuda_times,
        "timing": {
            "total_inference_ms": avg_ms,
            "tokens_per_second":  tps,
            "all_step_ms":        timing_ms_list,
        },
        "memory": {
            "peak_gpu_gb": mem_gb,
        },
        "compute": {
            "avg_cuda_ms":          avg_cuda_us / 1e3,
            "total_cuda_ms":        total_cuda_us / 1e3,
            "avg_flops_per_step":   float(avg_flops_per_step),
            "total_flops":          float(total_flops_all),
            "arithmetic_intensity": float(arith_intensity),
        },
        "top_kernels": [
            {
                "name":         e.key,
                "cuda_time_ms": _kern_time_us(e) / profile_steps / 1e3,
                "pct":          _kern_time_us(e) / total_cuda_us * 100 if total_cuda_us else 0,
                "calls":        max(e.count // profile_steps, 1) if e.count > 0 else e.count,
            }
            for e in top_kernels_raw
        ],
    }

    logger.info(
        f"  [{config_key}] {avg_ms:.1f} ms avg | {tps:.1f} tok/s | "
        f"{mem_gb:.2f} GB peak | CUDA {result['compute']['avg_cuda_ms']:.1f} ms/step"
    )
    logger.info("  Top 3 kernels (per-step avg):")
    for k in result["top_kernels"][:3]:
        logger.info(f"    {k['name'][:50]:<50} {k['cuda_time_ms']:8.1f} ms  ({k['pct']:.1f}%)")

    json_path = output_dir / f"{config_key}_profile.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved → {json_path}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if wandb_run is not None:
        wandb_run.log({
            "profiler/avg_inference_ms":  result["timing"]["total_inference_ms"],
            "profiler/tokens_per_second": result["timing"]["tokens_per_second"],
            "profiler/peak_gpu_gb":       result["memory"]["peak_gpu_gb"],
            "profiler/avg_cuda_ms":       result["compute"]["avg_cuda_ms"],
            "profiler/total_flops":       result["compute"]["total_flops"],
            "profiler/arith_intensity":   result["compute"]["arithmetic_intensity"],
        })
        if result.get("top_kernels"):
            kernel_table = wandb.Table(
                columns=["name", "cuda_time_ms", "pct", "calls"],
                data=[
                    [k["name"], k["cuda_time_ms"], k["pct"], k["calls"]]
                    for k in result["top_kernels"]
                ],
            )
            wandb_run.log({"profiler/top_kernels": kernel_table})
        logger.info(f"  W&B metrics logged for [{config_key}]")

    return result


# ─────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────

def run_profiler_sweep(
    config_keys: list[str],
    force: bool = False,
    output_dir: Path = DEFAULT_PROFILE_DIR,
    n_tokens: int = 50,
    warmup_steps: int = 3,
    profile_steps: int = 5,
    with_stack: bool = False,
    use_nvtx: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_ids: dict = None,
) -> dict:
    """Profile each config. Skips existing JSONs unless force=True.

    W&B behaviour:
      - wandb_run_ids maps config_key → existing run ID (notebook pre-creates one per config).
      - If wandb_project/entity are given without run IDs, a new run is created per config.
      - If neither is given, W&B is disabled.
    """
    results       = {}
    output_dir    = Path(output_dir)
    wandb_run_ids = wandb_run_ids or {}

    # Probe Kineto BEFORE any model is loaded — this is the only reliable way
    # to ensure Kineto's CUDA activity collector attaches to the CUDA context.
    kineto_active = _probe_kineto()

    for config_key in config_keys:
        if config_key not in MODEL_REGISTRY:
            logger.warning(f"[{config_key}] Not found in MODEL_REGISTRY — skipping.")
            continue

        json_path = output_dir / f"{config_key}_profile.json"

        if json_path.exists() and not force:
            with open(json_path) as f:
                results[config_key] = json.load(f)
            logger.info(f"[{config_key}] Loaded cached profile.")
            continue

        if not torch.cuda.is_available():
            logger.warning(f"[{config_key}] No GPU — skipping.")
            continue

        cfg = MODEL_REGISTRY[config_key]
        logger.info(f"\n{'='*60}")
        logger.info(f" Profiling : {cfg['description']}")
        logger.info(f" Config key: {config_key}")
        logger.info(f" HF model  : {cfg['hf_id']}")
        logger.info(f"{'='*60}")

        # ── W&B run setup ─────────────────────────────────────────────────────
        wandb_run = None
        run_id    = wandb_run_ids.get(config_key)
        if _WANDB_AVAILABLE and (wandb_project or run_id):
            if run_id:
                wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    id=run_id,
                    resume="allow",
                    reinit=True,
                )
            else:
                wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f"profiler_{config_key}",
                    reinit=True,
                    config={
                        "config_key":    config_key,
                        "hf_id":         cfg["hf_id"],
                        "quant_type":    cfg["quant_type"],
                        "bits":          cfg["bits"],
                        "n_tokens":      n_tokens,
                        "warmup_steps":  warmup_steps,
                        "profile_steps": profile_steps,
                    },
                )
            logger.info(f"  W&B run: {wandb_run.name} ({wandb_run.id})")

        model, tokenizer = load_model_for_profiling(config_key)

        results[config_key] = profile_inference_fixed(
            model=model,
            tokenizer=tokenizer,
            config_key=config_key,
            n_tokens=n_tokens,
            warmup_steps=warmup_steps,
            profile_steps=profile_steps,
            with_stack=with_stack,
            use_nvtx=use_nvtx,
            output_dir=output_dir,
            wandb_run=wandb_run,
        )

        if wandb_run is not None:
            wandb_run.finish()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"  [{config_key}] GPU memory freed.\n")

    return results


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_profiler_summary(results: dict) -> None:
    print(f"\n{'='*75}")
    print(f" Profiler Summary — Mistral-7B PTQ sweep")
    print(f"{'='*75}")
    print(f"  {'Config':<30} {'Infer(ms)':>10} {'Tok/s':>8} {'Peak(GB)':>10} {'CUDA(ms)':>10}")
    print(f"  {'-'*70}")
    for key in ALL_CONFIGS:
        if key not in results:
            cfg_desc = MODEL_REGISTRY[key]["description"]
            print(f"  {cfg_desc:<30} {'MISSING':>10}")
            continue
        d        = results[key]
        ms       = d["timing"]["total_inference_ms"]
        tps      = d["timing"]["tokens_per_second"]
        mem      = d["memory"]["peak_gpu_gb"]
        cuda     = d["compute"]["avg_cuda_ms"]
        bits     = MODEL_REGISTRY[key]["bits"]
        exp      = bits / 16 * 14.0
        flag     = "" if mem < exp * 1.5 else "  <- STILL FP16 WEIGHTS?"
        desc     = MODEL_REGISTRY[key]["description"]
        print(f"  {desc:<30} {ms:>10.1f} {tps:>8.1f} {mem:>10.2f} {cuda:>10.1f}{flag}")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fixed profiler sweep for Mistral-7B PTQ configs")
    p.add_argument("--config", type=str, choices=ALL_CONFIGS,
                   help="Profile a single config by its MODEL_REGISTRY key")
    p.add_argument("--all",   action="store_true", help="Profile all configs in MODEL_REGISTRY")
    p.add_argument("--force", action="store_true", help="Re-profile even if JSON exists")
    p.add_argument("--n-tokens",       type=int, default=50)
    p.add_argument("--warmup-steps",   type=int, default=3)
    p.add_argument("--profile-steps",  type=int, default=5)
    p.add_argument("--output-dir",     type=str, default=str(DEFAULT_PROFILE_DIR))
    p.add_argument("--with-stack",     action="store_true", default=False,
                   help="Enable Python call-stack capture (produces very large traces; off by default)")
    p.add_argument("--nvtx",           action="store_true", default=False,
                   help="Emit NVTX range markers for Nsight Systems / Nsight Compute. "
                        "Required when run under: "
                        "  nsys profile --capture-range=nvtx "
                        "  ncu --nvtx-include=profiling_region "
                        "No-op for standalone profiler runs — safe to omit.")
    # Authentication
    p.add_argument("--hf-token",       type=str, default=None,
                   help="HuggingFace token (also read from HF_TOKEN env var)")
    # W&B
    p.add_argument("--wandb-project",  type=str, default=None)
    p.add_argument("--wandb-entity",   type=str, default=None)
    p.add_argument("--wandb-run-id",   type=str, default=None)
    p.add_argument("--wandb-api-key",  type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Auth ──────────────────────────────────────────────────────────────────
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            logger.info("HuggingFace login successful.")
        except ImportError:
            pass

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if _WANDB_AVAILABLE:
            wandb.login(key=args.wandb_api_key)
            logger.info("W&B login successful.")

    # ── NVTX probe (only when --nvtx is requested) ────────────────────────────
    if args.nvtx:
        status = probe_nvtx()
        if status["available"]:
            logger.info(f"NVTX: {status['message']}")
        else:
            logger.warning(f"NVTX: {status['message']} — disabling NVTX markers")
            args.nvtx = False   # degrade gracefully; profiling still runs

    # ── Target configs ────────────────────────────────────────────────────────
    if args.all:
        targets = ALL_CONFIGS
    elif args.config:
        targets = [args.config]
    else:
        print("Specify --config <key> or --all")
        print(f"Available keys: {ALL_CONFIGS}")
        exit(1)

    wandb_run_ids = {}
    if args.wandb_run_id and len(targets) == 1:
        wandb_run_ids = {targets[0]: args.wandb_run_id}
    elif args.wandb_run_id and len(targets) > 1:
        logger.warning(
            "--wandb-run-id ignored when --all is set. "
            "Use --wandb-project/--wandb-entity instead."
        )

    results = run_profiler_sweep(
        config_keys=targets,
        force=args.force,
        output_dir=Path(args.output_dir),
        n_tokens=args.n_tokens,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        with_stack=args.with_stack,
        use_nvtx=args.nvtx,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_ids=wandb_run_ids,
    )
    print_profiler_summary(results)
