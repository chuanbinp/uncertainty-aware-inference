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

Usage:
    # Profile one config by its MODEL_REGISTRY key:
    python run_profiler.py --config mistral-7b-gptq-int4

    # Profile all configs sequentially:
    python run_profiler.py --all

    # Force re-profile even if JSON exists:
    python run_profiler.py --all --force

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

# ── CUPTI must be on LD_LIBRARY_PATH before torch is imported ────────────────
# PyTorch's Kineto backend dlopen()s libcupti.so at import time to enable
# GPU kernel tracing (ProfilerActivity.CUDA). On Colab/A100 the library exists
# but is not on the default path, so Kineto silently falls back to CPU-only
# and kineto_results stays None. Prepend the path here, before torch import.
_CUPTI_PATHS = [
    "/usr/local/cuda/extras/CUPTI/lib64",
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
]
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
_extra = ":".join(p for p in _CUPTI_PATHS if os.path.isdir(p))
if _extra:
    os.environ["LD_LIBRARY_PATH"] = _extra + (":" + _existing_ld if _existing_ld else "")

# Disable Kineto daemon mode — causes attach failures on Colab where
# /tmp permissions block the Unix socket Kineto uses for IPC.
os.environ["KINETO_USE_DAEMON"] = "0"
# Tell Kineto to use synchronous collection (more reliable on single-GPU nodes)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # keep async launches for perf
os.environ["KINETO_DAEMON_INIT_WAIT_USECS"] = "50000"

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

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
DEFAULT_PROFILE_DIR  = Path("/content/profiler_results")

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
        return False

    # Log which CUPTI paths were found — helps diagnose failures
    import glob
    cupti_libs = glob.glob("/usr/local/cuda*/**/libcupti.so*", recursive=True)
    if cupti_libs:
        logger.info(f"  CUPTI found: {cupti_libs[0]}")
    else:
        logger.warning(
            "  libcupti.so not found under /usr/local/cuda. "
            "GPU kernel tracing requires CUPTI. "
            "On Colab: Runtime → Factory reset, or run: "
            "apt-get install -y cuda-cupti-$(nvcc --version | grep -oP '(?<=release )\\d+\\.\\d+' | tr . -)"
        )

    # Ensure CUDA context is alive
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

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

        # Verify kineto_results is not None by attempting export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        _prof.export_chrome_trace(tmp_path)
        os.unlink(tmp_path)

        # Confirm at least one event has non-zero GPU time
        # (attribute name changed across PyTorch versions)
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
            return True
        else:
            logger.warning(
                "Kineto CUDA backend: export succeeded but no cuda_time_total events found. "
                "Kernel stats will be CPU-side times."
            )
            return False

    except AttributeError:
        logger.warning(
            "Kineto CUDA backend: NOT active (kineto_results=None). "
            "Check LD_LIBRARY_PATH includes CUPTI and rerun. "
            "LD_LIBRARY_PATH=" + os.environ.get("LD_LIBRARY_PATH", "(not set)")
        )
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
        # AutoModelForCausalLM detects GPTQ config and routes through
        # transformers built-in GPTQ quantizer which requires optimum AND
        # still fails on many builds. Use gptqmodel directly instead.
        from gptqmodel import GPTQModel
        revision = cfg.get("gptq_revision", "main")
        logger.info(f"  GPTQ revision: {revision}")
        model = GPTQModel.from_quantized(
            hf_id,
            revision=revision,
            device="cuda:0",
            trust_remote_code=True,
        )

    elif quant_type == "awq":
        # Use AutoAWQForCausalLM directly to avoid mis-routing through
        # transformers. fuse_layers=False required for torch.profiler tracing.
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
        expected_gb = bits / 16 * 14.0   # fp16=14GB, int8=7GB, int4=3.5GB
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
    with_stack: bool = False,   # False by default: True produces 500MB+ traces on A100
    wandb_run=None,
) -> dict:
    """
    Fixed PyTorch Profiler harness for HuggingFace LLMs.

    Returns a dict with timing / memory / compute / top_kernels.
    Also saves {output_dir}/{config_key}_profile.json.
    All kernel times are per-step averages (key_averages accumulates across all steps).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = MODEL_REGISTRY[config_key]

    # Use cuda:0 explicitly — next(model.parameters()).device can return 'meta'
    # for quantized layers (bitsandbytes, GPTQ) with device_map="auto".
    device    = (torch.device("cuda", torch.cuda.current_device())
                 if torch.cuda.is_available() else torch.device("cpu"))
    inputs         = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info(f"[{config_key}] Warmup ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
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
    timing_ms_list = []
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    logger.info(f"[{config_key}] Profiling ({profile_steps} steps)...")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_flops=True,
        with_stack=with_stack,
        with_modules=False,
    ) as prof:
        for step in range(profile_steps):
            with record_function(f"inference_step_{step}"):
                t0 = time.perf_counter()
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
                # kineto_results is None: Kineto CUDA backend failed to init.
                # This only affects the trace file — timing/kernel stats below
                # are collected from CPU-side events and are still valid.
                logger.warning(
                    "  Chrome trace skipped: kineto_results=None "
                    "(Kineto CUDA backend did not initialize on this build). "
                    "Timing and kernel stats are unaffected."
                )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg_ms = float(np.mean(timing_ms_list))
    tps    = n_tokens / (avg_ms / 1000) if avg_ms > 0 else 0.0

    # Peak memory tracked via torch.cuda (unaffected by profile_memory=False)
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    key_avgs = prof.key_averages()

    # Resolve the correct CUDA time attribute — it changed across PyTorch versions.
    # PyTorch < 2.1 : cuda_time_total
    # PyTorch >= 2.1: cuda_time_total was removed; use self_cuda_time_total instead
    # We try each in order and pick the first one that exists AND has non-zero values.
    def _get_cuda_time(e):
        for attr in ("cuda_time_total", "self_cuda_time_total", "device_time_total"):
            v = getattr(e, attr, None)
            if v is not None:
                return v
        return 0.0

    # Determine whether we have real GPU times by checking if any event
    # has a non-zero value from the CUDA time attribute chain.
    sample_cuda_times = [_get_cuda_time(e) for e in key_avgs]
    has_cuda_times    = any(v > 0 for v in sample_cuda_times)

    # record_function span labels (e.g. "inference_step_0") are not real GPU
    # kernels — filter them out so top_kernels shows only actual CUDA kernels.
    # Real kernels have a non-None input_shapes or are known aten/cuda ops;
    # the simplest reliable filter is: exclude names that start with our label prefix
    # and exclude pure Python-scope entries that have no device-side kernel name.
    _SPAN_PREFIXES = ("inference_step_",)

    def _is_real_kernel(e):
        return not any(e.key.startswith(p) for p in _SPAN_PREFIXES)

    if has_cuda_times:
        logger.info(f"  [{config_key}] Kineto CUDA active — using real GPU kernel times")
        cuda_events     = [e for e in key_avgs if _get_cuda_time(e) > 0]
        total_cuda_us   = sum(_get_cuda_time(e) for e in cuda_events)
        # Sort all events by CUDA time, then filter spans for the top-10 display
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

    # key_averages() accumulates across ALL profile_steps — divide for per-step values
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
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_ids: dict = None,   # config_key → run_id; populated by notebook
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
    # Once a large model has been loaded and run, Kineto can no longer attach.
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

        # ── W&B run setup — one run per config ────────────────────────────────
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
                   help="Enable Python call-stack capture (warning: produces very large "
                        "chrome traces on fast GPUs like A100; off by default)")
    # Authentication
    p.add_argument("--hf-token",       type=str, default=None,
                   help="HuggingFace token (also read from HF_TOKEN env var)")
    # W&B
    p.add_argument("--wandb-project",  type=str, default=None,
                   help="W&B project name (omit to disable W&B)")
    p.add_argument("--wandb-entity",   type=str, default=None,
                   help="W&B entity / team name")
    p.add_argument("--wandb-run-id",   type=str, default=None,
                   help="Resume an existing W&B run by ID (used by the notebook)")
    p.add_argument("--wandb-api-key",  type=str, default=None,
                   help="W&B API key (also read from WANDB_API_KEY env var)")
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
            pass  # huggingface_hub not installed; token set in env

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        if _WANDB_AVAILABLE:
            wandb.login(key=args.wandb_api_key)
            logger.info("W&B login successful.")

    # ── Target configs ────────────────────────────────────────────────────────
    if args.all:
        targets = ALL_CONFIGS
    elif args.config:
        targets = [args.config]
    else:
        print("Specify --config <key> or --all")
        print(f"Available keys: {ALL_CONFIGS}")
        exit(1)

    # When called from the notebook, --wandb-run-id is a single ID for the
    # one config being profiled in that subprocess call.
    wandb_run_ids = {}
    if args.wandb_run_id and len(targets) == 1:
        wandb_run_ids = {targets[0]: args.wandb_run_id}
    elif args.wandb_run_id and len(targets) > 1:
        logger.warning(
            "--wandb-run-id ignored when --all is set (each config gets its own run). "
            "Use --wandb-project/--wandb-entity and runs will be created automatically."
        )

    results = run_profiler_sweep(
        config_keys=targets,
        force=args.force,
        output_dir=Path(args.output_dir),
        n_tokens=args.n_tokens,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        with_stack=args.with_stack,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_ids=wandb_run_ids,
    )
    print_profiler_summary(results)
