"""
run_profiler.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixed PyTorch Profiler sweep for all Mistral-7B PTQ configs.

Key fixes vs. original pytorch_profiler.py:
  ✓ with_modules=False   (was True → caused torch.fx tracing errors on all HF models)
  ✓ with_stack=True      (proper kernel-level call stack attribution)
  ✓ cuda.synchronize()   inside profiler context (CUDA events flushed before trace)
  ✓ export_chrome_trace  inside 'with profile()' block (avoids empty-trace bug)
  ✓ __main__ dispatches  correctly per --precision (was always FP16)
  ✓ profile_memory=False (was True → caused kineto_results=None on some PyTorch builds)
  ✓ cuda.synchronize()   before export_chrome_trace (forces Kineto to finalize)

Usage:
    # Profile one config:
    python run_profiler.py --precision gptq_int4

    # Profile all configs sequentially:
    python run_profiler.py --all

    # Force re-profile even if JSON exists:
    python run_profiler.py --all --force

    # With W&B logging:
    python run_profiler.py --precision fp16 \
        --wandb-project UAI_Project \
        --wandb-entity Uncertainty_Aware_Inference_Lab \
        --wandb-api-key <key>

    # With a pre-initialised W&B run ID (used by the notebook):
    python run_profiler.py --precision fp16 \
        --wandb-project UAI_Project \
        --wandb-run-id <run_id>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import logging
import os
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

from config import (
    MODEL_NAME, PROFILE_DIR, PROFILING_CONFIGS,
    PRECISION_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Model loader (dispatches by precision)
# ─────────────────────────────────────────────────────────────

def load_model_for_profiling(model_path: str, precision: str):
    """
    Load the correct model variant for profiling.
    Uses transformers native GPTQ loading to avoid auto_gptq import issues.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from config import PTQ_CONFIGS

    logger.info(f"Loading {precision}: {model_path}")

    if precision == "fp16":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    elif precision in ("gptq_int8", "gptq_int4"):
        revision = PTQ_CONFIGS[precision].get("revision", "main")
        logger.info(f"  GPTQ revision: {revision}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    elif precision == "awq_int4":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    elif precision == "nf4":
        from transformers import BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )

    else:
        raise ValueError(f"Unknown precision: {precision}")

    model.eval()

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  GPU memory after load: {mem_gb:.2f} GB")
        expected = {"fp16": 14, "gptq_int8": 7, "gptq_int4": 3.5, "awq_int4": 3.5, "nf4": 3.5}
        exp = expected.get(precision, 0)
        if exp and mem_gb > exp * 1.5:
            logger.warning(
                f"  Expected ~{exp} GB for {precision} but got {mem_gb:.1f} GB — "
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
    prompt: str = "Explain transformer attention in one sentence.",
    n_tokens: int = 50,
    warmup_steps: int = 3,
    profile_steps: int = 5,
    output_dir: Path = PROFILE_DIR,
    model_name: str = MODEL_NAME,
    precision: str = "fp16",
    export_chrome_trace: bool = True,
    with_stack: bool = False,   # ← False by default: with_stack=True produces 500MB+ traces
    wandb_run=None,             #   on A100 (fast GPU = dense trace). Pass --with-stack to enable.
) -> dict:
    """
    Fixed PyTorch Profiler harness for HuggingFace LLMs.

    Key fixes vs original profiling/pytorch_profiler.py:
      ✓ with_modules=False  — was True, caused torch.fx tracing errors on HF models
      ✓ with_stack=True     — proper kernel-level call stack attribution
      ✓ profile_memory=False — was True, caused kineto_results=None bug on some PyTorch builds
      ✓ cuda.synchronize()  before export_chrome_trace — forces Kineto to finalize
      ✓ chrome trace export inside 'with' block — avoids empty-trace bug

    Returns:
        dict with timing, memory, compute, top_kernels keys.
        Also saved to {output_dir}/{model_name}_{precision}_profile.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use cuda:0 explicitly — next(model.parameters()).device can return 'meta'
    # for quantized layers (bitsandbytes, GPTQ) with device_map="auto", which
    # would cause .to(device) on input_ids to crash.
    device = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # ── Warmup ────────────────────────────────────────────────────────────────
    logger.info(f"[{precision}] Warmup ({warmup_steps} steps)...")
    for _ in range(warmup_steps):
        model.generate(
            input_ids=input_ids,
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

    logger.info(f"[{precision}] Profiling ({profile_steps} steps)...")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,   # ← FIXED: was True → caused kineto_results=None bug
        with_flops=True,
        with_stack=with_stack,  # ← controlled by caller; default False (see docstring)
        with_modules=False,     # ← FIXED: was True → caused fx tracing failures
    ) as prof:
        for step in range(profile_steps):
            with record_function(f"inference_step_{step}"):
                t0 = time.perf_counter()
                model.generate(
                    input_ids=input_ids,
                    max_new_tokens=n_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()   # ← flush CUDA events inside context
                timing_ms_list.append((time.perf_counter() - t0) * 1000)

        # ← FIXED: final sync forces Kineto to finalize kineto_results before export
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ← FIXED: export inside 'with' block (avoids empty-trace bug)
        if export_chrome_trace:
            trace_path = output_dir / f"{model_name}_{precision}_chrome.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info(f"  Chrome trace → {trace_path}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    avg_ms = float(np.mean(timing_ms_list))
    tps    = n_tokens / (avg_ms / 1000) if avg_ms > 0 else 0.0

    # Peak memory still tracked via torch.cuda (unaffected by profile_memory=False)
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    key_avgs      = prof.key_averages()
    cuda_events   = [e for e in key_avgs if e.cuda_time_total > 0]
    # key_averages() accumulates across ALL profile_steps — divide to get per-step avg
    total_cuda_us      = sum(e.cuda_time_total for e in cuda_events)
    avg_cuda_us        = total_cuda_us / profile_steps
    total_flops_all    = sum(getattr(e, "flops", 0) or 0 for e in key_avgs)
    avg_flops_per_step = total_flops_all / profile_steps

    # Warn when flops counter returns zero — expected for quantized kernels
    # (GPTQ marlin, exllama, bitsandbytes) which are custom CUDA ops unknown to
    # the PyTorch flops estimator.
    if total_flops_all == 0:
        logger.warning(
            f"  [{precision}] total_flops=0 — PyTorch flops counter does not support "
            f"custom quant kernels (GPTQ/AWQ/NF4). arithmetic_intensity will be 0."
        )

    # Arithmetic intensity: per-step FLOPs / peak bytes allocated (approximate).
    # Note: true AI requires bytes *read* from memory (bandwidth), not bytes allocated.
    # This is a proxy — use for relative comparison across configs, not absolute values.
    arith_intensity = avg_flops_per_step / (mem_gb * 1e9) if mem_gb > 0 else 0.0

    top_kernels_raw = sorted(cuda_events, key=lambda e: e.cuda_time_total, reverse=True)[:10]

    result = {
        "model":     model_name,
        "precision": precision,
        "timing": {
            "total_inference_ms": avg_ms,
            "tokens_per_second":  tps,
            "all_step_ms":        timing_ms_list,
        },
        "memory": {
            "peak_gpu_gb": mem_gb,
        },
        "compute": {
            "avg_cuda_ms":          avg_cuda_us / 1e3,       # per-step average
            "total_cuda_ms":        total_cuda_us / 1e3,     # across all steps (kept for compat)
            "avg_flops_per_step":   float(avg_flops_per_step),
            "total_flops":          float(total_flops_all),  # kept for compat
            "arithmetic_intensity": float(arith_intensity),  # per-step approx
        },
        "top_kernels": [
            {
                "name":             e.key,
                # per-step average time for this kernel
                "cuda_time_ms":     e.cuda_time_total / profile_steps / 1e3,
                "pct":              e.cuda_time_total / total_cuda_us * 100 if total_cuda_us else 0,
                "calls":            e.count // profile_steps,  # avg calls per step
            }
            for e in top_kernels_raw
        ],
    }

    logger.info(
        f"  [{precision}] {avg_ms:.1f} ms avg | {tps:.1f} tok/s | {mem_gb:.2f} GB peak | "
        f"CUDA {result['compute']['avg_cuda_ms']:.1f} ms/step"
    )
    logger.info(f"  Top 3 kernels (per-step avg):")
    for k in result["top_kernels"][:3]:
        logger.info(f"    {k['name'][:50]:<50} {k['cuda_time_ms']:8.1f} ms  ({k['pct']:.1f}%)")

    json_path = output_dir / f"{model_name}_{precision}_profile.json"
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
        wandb_run.log({"profiler/chrome_trace": str(
            output_dir / f"{model_name}_{precision}_chrome.json"
        )})
        logger.info(f"  W&B metrics logged for [{precision}]")

    return result


# ─────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────

def run_profiler_sweep(
    precisions: list[str],
    force: bool = False,
    output_dir: Path = PROFILE_DIR,
    n_tokens: int = 50,
    warmup_steps: int = 3,
    profile_steps: int = 5,
    with_stack: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_ids: dict = None,   # precision → run_id; populated by notebook per-config
) -> dict:
    """Profile each config. Skips existing JSONs unless force=True.

    W&B behaviour:
      - wandb_run_ids is a dict mapping precision → existing run ID (used when the
        notebook pre-creates one run per precision and passes their IDs).
      - If wandb_project/entity are given without run IDs, a new run is created
        per precision automatically.
      - If none are given, W&B is disabled.
    """
    results    = {}
    output_dir = Path(output_dir)
    wandb_run_ids = wandb_run_ids or {}

    for precision in precisions:
        json_path  = output_dir / f"{MODEL_NAME}_{precision}_profile.json"
        model_path = PROFILING_CONFIGS.get(precision)

        if not model_path:
            logger.warning(f"[{precision}] No model path configured — skipping.")
            continue

        if json_path.exists() and not force:
            with open(json_path) as f:
                results[precision] = json.load(f)
            logger.info(f"[{precision}] Loaded cached profile.")
            continue

        if not torch.cuda.is_available():
            logger.warning(f"[{precision}] No GPU — skipping.")
            continue

        logger.info(f"\n{'='*55}")
        logger.info(f" Profiling: {PRECISION_LABELS.get(precision, precision)}")
        logger.info(f" Model: {model_path}")
        logger.info(f"{'='*55}")

        # ── W&B run setup — one run per precision ─────────────────────────────
        wandb_run = None
        run_id    = wandb_run_ids.get(precision)
        if _WANDB_AVAILABLE and (wandb_project or run_id):
            if run_id:
                # Resume the per-precision run the notebook already created
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
                    name=f"profiler_{MODEL_NAME}_{precision}",
                    reinit=True,
                    config={
                        "model":          MODEL_NAME,
                        "precision":      precision,
                        "n_tokens":       n_tokens,
                        "warmup_steps":   warmup_steps,
                        "profile_steps":  profile_steps,
                    },
                )
            logger.info(f"  W&B run: {wandb_run.name} ({wandb_run.id})")

        model, tokenizer = load_model_for_profiling(model_path, precision)

        results[precision] = profile_inference_fixed(
            model=model,
            tokenizer=tokenizer,
            n_tokens=n_tokens,
            warmup_steps=warmup_steps,
            profile_steps=profile_steps,
            with_stack=with_stack,
            output_dir=output_dir,
            model_name=MODEL_NAME,
            precision=precision,
            wandb_run=wandb_run,
        )

        if wandb_run is not None:
            wandb_run.finish()

        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"  [{precision}] GPU memory freed.\n")

    return results


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_profiler_summary(results: dict) -> None:
    from config import PRECISIONS
    print(f"\n{'='*65}")
    print(f" Profiler Summary — {MODEL_NAME}")
    print(f"{'='*65}")
    print(f"  {'Config':<15} {'Infer(ms)':>10} {'Tok/s':>8} {'Peak(GB)':>10} {'CUDA(ms)':>10}")
    print(f"  {'-'*57}")
    expected_mem = {"fp16": 14, "gptq_int8": 7, "gptq_int4": 3.5, "awq_int4": 3.5, "nf4": 3.5}
    for prec in PRECISIONS:
        if prec not in results:
            print(f"  {PRECISION_LABELS.get(prec, prec):<15} {'MISSING':>10}")
            continue
        d    = results[prec]
        ms   = d["timing"]["total_inference_ms"]
        tps  = d["timing"]["tokens_per_second"]
        mem  = d["memory"]["peak_gpu_gb"]
        cuda = d["compute"]["avg_cuda_ms"]
        exp  = expected_mem.get(prec, 0)
        flag = "" if mem < exp * 1.5 else "  ← STILL FP16 WEIGHTS?"
        print(f"  {PRECISION_LABELS.get(prec, prec):<15} {ms:>10.1f} {tps:>8.1f} {mem:>10.2f} {cuda:>10.1f}{flag}")
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fixed profiler sweep for Mistral-7B PTQ configs")
    p.add_argument("--precision", type=str,
                   choices=["fp16", "gptq_int8", "gptq_int4", "awq_int4", "nf4"],
                   help="Profile a single precision config")
    p.add_argument("--all",   action="store_true", help="Profile all configs")
    p.add_argument("--force", action="store_true", help="Re-profile even if JSON exists")
    p.add_argument("--n-tokens",       type=int, default=50)
    p.add_argument("--warmup-steps",   type=int, default=3)
    p.add_argument("--profile-steps",  type=int, default=5)
    p.add_argument("--output-dir",     type=str, default=str(PROFILE_DIR))
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

    # ── Target precisions ─────────────────────────────────────────────────────
    if args.all:
        from config import PRECISIONS
        targets = PRECISIONS
    elif args.precision:
        targets = [args.precision]
    else:
        print("Specify --precision <n> or --all")
        print("Example: python run_profiler.py --precision gptq_int4")
        exit(1)

    # When called from the notebook, --wandb-run-id is a single ID for the
    # one precision being profiled in that subprocess call.
    wandb_run_ids = {}
    if args.wandb_run_id and len(targets) == 1:
        wandb_run_ids = {targets[0]: args.wandb_run_id}
    elif args.wandb_run_id and len(targets) > 1:
        logger.warning(
            "--wandb-run-id ignored when --all is set (each precision gets its own run). "
            "Use --wandb-project/--wandb-entity instead and runs will be created automatically."
        )

    results = run_profiler_sweep(
        precisions=targets,
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
