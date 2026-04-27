"""
run_nsys.py
─────────────────────────────────────────────────────────────────────────────
Nsight Systems launcher for Mistral-7B PTQ sweep.

nsys wraps the Python process from outside and captures a system-wide
execution timeline: CUDA kernels, memory copies, CPU↔GPU synchronisation,
OS threads, and NVTX range markers.

This script:
  1. Verifies nsys is installed and accessible.
  2. Runs `nsys profile` for each config (one at a time — nsys requires
     exclusive GPU access for timeline capture).
  3. Exports a stats summary CSV from each .nsys-rep using `nsys stats`.
  4. Parses the GPU trace CSV into a structured JSON for downstream use.

Output per config (in --output-dir):
  {config_key}_nsys.nsys-rep       ← full timeline (open in Nsight Systems GUI)
  {config_key}_nsys_gputrace.csv   ← GPU kernel trace (from nsys stats)
  {config_key}_nsys_summary.json   ← parsed summary: top kernels, timing

Requirements:
  - nsys >= 2023.1 (ships with CUDA 12 on GCP / any bare-metal GPU instance)
  - GCP VM or bare-metal: nsys typically at /usr/local/cuda/bin/nsys
    or /opt/nvidia/nsight-systems/*/bin/nsys
  - NOT available on Colab free tier (no perf_event access)

Usage:
    # Profile one config:
    python run_nsys.py --config mistral-7b-gptq-int4

    # Profile all configs sequentially:
    python run_nsys.py --all

    # Skip existing .nsys-rep files:
    python run_nsys.py --all            # (default: skip if exists)
    python run_nsys.py --all --force    # re-profile even if exists

    # Specify nsys binary location:
    python run_nsys.py --config mistral-7b-fp16 --nsys-path /usr/local/cuda/bin/nsys

Typical GCP workflow:
    # 1. SSH into your GCP VM (A100 instance)
    # 2. cd uncertainty-aware-inference/TeamB
    # 3. python run_nsys.py --all --output-dir /content/nsys_results
    # 4. Download .nsys-rep files and open in Nsight Systems GUI
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from configs import MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_nsys")

ALL_CONFIGS          = list(MODEL_REGISTRY.keys())
DEFAULT_OUTPUT_DIR   = Path("./nsys_results")

# nsys collect trace types:
#   cuda  — CUDA API calls and kernel launches
#   nvtx  — NVTX range markers (from nvtx_utils.py in run_profiler.py)
#   osrt  — OS runtime (thread scheduling, system calls)
#   cudnn — cuDNN library calls
NSYS_TRACE_TYPES = "cuda,nvtx,osrt,cudnn"

# Only capture inside the NVTX range named "profiling_region" (set by
# nvtx_utils.profiling_region() in run_profiler.py --nvtx).
# This excludes model load and warmup from the captured timeline,
# keeping .nsys-rep files small and focused on the actual inference.
NSYS_CAPTURE_RANGE = "nvtx"
NSYS_NVTX_CAPTURE  = "profiling_region"


# ─────────────────────────────────────────────────────────────────────────────
# nsys binary detection
# ─────────────────────────────────────────────────────────────────────────────

NSYS_SEARCH_PATHS = [
    "nsys",                                        # on PATH
    "/usr/local/cuda/bin/nsys",
    "/usr/local/cuda-12/bin/nsys",
    "/usr/local/cuda-11/bin/nsys",
    "/opt/nvidia/nsight-systems/2023.4.1/bin/nsys",
    "/opt/nvidia/nsight-systems/2023.3.1/bin/nsys",
    "/opt/nvidia/nsight-systems/2024.1.1/bin/nsys",
]


def find_nsys(explicit_path: str = None) -> str:
    """
    Locate the nsys binary. Checks explicit_path first, then NSYS_SEARCH_PATHS.
    Returns the path if found, raises FileNotFoundError otherwise.
    """
    candidates = ([explicit_path] if explicit_path else []) + NSYS_SEARCH_PATHS
    for candidate in candidates:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if resolved:
            logger.info(f"nsys found: {resolved}")
            return resolved

    raise FileNotFoundError(
        "nsys (Nsight Systems CLI) not found. "
        "On GCP with CUDA 12: it is at /usr/local/cuda/bin/nsys. "
        "Install via: apt-get install -y nsight-systems-cli  "
        "or download from https://developer.nvidia.com/nsight-systems. "
        "Colab free tier does not support nsys profiling (requires perf_event access)."
    )


def get_nsys_version(nsys_bin: str) -> str:
    """Return nsys version string."""
    try:
        result = subprocess.run(
            [nsys_bin, "--version"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def detect_gpu_architecture() -> dict:
    """
    Detect the active GPU's architecture name and CUDA compute capability.

    Returns a dict:
        {"name": "NVIDIA A100-SXM4-80GB", "arch": "ampere",
         "cc": "8.0", "metrics_set": "ga10x-A100"}

    Architecture → nsys --gpu-metrics-set mapping:
        Ampere GA100 (A100)          → ga10x-A100
        Ampere GA10x (A10, A30, A40) → ga10x
        Ada Lovelace (L4, L40, 4090) → ad10x
        Hopper (H100)                → gh100
        Volta (V100)                 → gv100
        Turing (T4)                  → tu10x

    The --gpu-metrics-set flag enables hardware performance counter collection
    in the nsys timeline (SM occupancy, memory throughput, etc.).  It is
    architecture-specific because each GPU generation exposes different counters.
    If the wrong set is specified, nsys silently skips HW counter collection
    rather than erroring out.
    """
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        cc   = torch.cuda.get_device_capability(0)  # e.g. (8, 0) for A100
        cc_str = f"{cc[0]}.{cc[1]}"
    except Exception:
        return {"name": "unknown", "arch": "unknown", "cc": "unknown", "metrics_set": "ga10x-A100"}

    name_lower = name.lower()
    cc_major   = cc[0]

    # ── Architecture classification by compute capability + name ─────────────
    if cc_major == 9:
        # Hopper (H100, H200)
        arch, metrics_set = "hopper",        "gh100"
    elif cc_major == 8:
        if "a100" in name_lower:
            arch, metrics_set = "ampere_ga100", "ga10x-A100"   # GA100 has dedicated set
        else:
            # A10, A30, A40, A6000, A800, RTX 3090 (all GA10x)
            arch, metrics_set = "ampere_ga10x", "ga10x"
    elif cc_major == 7:
        if cc[1] == 0:
            arch, metrics_set = "volta",  "gv100"   # V100
        else:
            arch, metrics_set = "turing", "tu10x"   # T4, RTX 20xx
    elif cc_major == 8 and "l4" in name_lower:
        # L4 is Ada Lovelace (AD104), cc=(8,9) → caught by the ada check below
        arch, metrics_set = "ada", "ad10x"
    else:
        # Ada Lovelace: cc=(8,9) — L4, L40, RTX 40xx, RTX 6000 Ada
        # cc major=8 minor≥9 indicates Ada on most driver versions
        if cc_major == 8 and cc[1] >= 9:
            arch, metrics_set = "ada", "ad10x"
        else:
            # Fallback — assume Ampere A100 (our primary platform)
            arch, metrics_set = "ampere_ga100", "ga10x-A100"
            logger.warning(
                f"Unknown GPU architecture (cc={cc_str}, name='{name}'). "
                f"Defaulting to ga10x-A100 metrics set — verify if correct."
            )

    logger.info(
        f"GPU detected: {name}  |  compute capability {cc_str}  |  "
        f"arch={arch}  |  nsys metrics-set={metrics_set}"
    )
    return {"name": name, "arch": arch, "cc": cc_str, "metrics_set": metrics_set}


# Cache the detection result so it runs once per process
_GPU_INFO: dict = {}

def _detect_nsys_metrics_set() -> str:
    """Return the --gpu-metrics-set value for the active GPU. Cached after first call."""
    global _GPU_INFO
    if not _GPU_INFO:
        _GPU_INFO = detect_gpu_architecture()
    return _GPU_INFO["metrics_set"]


# ─────────────────────────────────────────────────────────────────────────────
# Single-config nsys profile run
# ─────────────────────────────────────────────────────────────────────────────

def run_nsys_profile(
    config_key: str,
    nsys_bin: str,
    output_dir: Path,
    repo_dir: Path,
    hf_token: str = None,
    profile_steps: int = 3,    # fewer steps than PyTorch Profiler — .nsys-rep grows quickly
    n_tokens: int = 50,
    force: bool = False,
) -> dict:
    """
    Run `nsys profile` for one MODEL_REGISTRY config.

    Calls run_profiler.py with --nvtx so NVTX markers are emitted,
    then nsys captures only the "profiling_region" NVTX range.

    Returns a dict with paths to output files and status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rep_file = output_dir / f"{config_key}_nsys.nsys-rep"

    if rep_file.exists() and not force:
        logger.info(f"[{config_key}] .nsys-rep exists — skipping (use --force to re-run)")
        return {"config_key": config_key, "status": "cached", "rep_file": str(rep_file)}

    cfg = MODEL_REGISTRY[config_key]
    logger.info(f"\n{'='*65}")
    logger.info(f" nsys profile: {cfg['description']}")
    logger.info(f" Config key  : {config_key}")
    logger.info(f"{'='*65}")

    # ── Build the nsys command ──────────────────────────────────────────────
    # --capture-range=nvtx + --nvtx-capture restrict collection to the
    # "profiling_region" NVTX range emitted by run_profiler.py --nvtx.
    # Without this, nsys would capture model load and warmup too, bloating
    # the .nsys-rep and contaminating kernel timing statistics.
    nsys_cmd = [
        nsys_bin, "profile",
        "--output",             str(output_dir / f"{config_key}_nsys"),
        "--trace",              NSYS_TRACE_TYPES,
        "--capture-range",      NSYS_CAPTURE_RANGE,
        "--nvtx-capture",       NSYS_NVTX_CAPTURE,
        "--force-overwrite",    "true",
        "--stats",              "true",         # print summary to stdout after capture
        "--gpu-metrics-set",    _detect_nsys_metrics_set(),  # auto-detected for the active GPU
        "--export",             "sqlite",       # also write a .sqlite for programmatic queries
    ]

    # ── Append the Python command that nsys will wrap ──────────────────────
    profiler_script = repo_dir / "run_profiler.py"
    python_cmd = [
        sys.executable,
        str(profiler_script),
        "--config",         config_key,
        "--nvtx",                              # emit NVTX range markers
        "--profile-steps",  str(profile_steps),
        "--n-tokens",       str(n_tokens),
        "--warmup-steps",   "2",               # fewer warmups — nsys capture is outside anyway
        "--output-dir",     str(output_dir),   # PyTorch Profiler JSON also goes here
    ]

    full_cmd = nsys_cmd + python_cmd

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    logger.info("CMD: " + " ".join(full_cmd))
    t0 = time.perf_counter()

    result = subprocess.run(full_cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        logger.error(f"[{config_key}] nsys FAILED (exit {result.returncode})")
        logger.error("STDERR:\n" + result.stderr[-2000:])
        return {
            "config_key": config_key,
            "status":     "failed",
            "returncode": result.returncode,
            "stderr":     result.stderr[-2000:],
        }

    logger.info(f"[{config_key}] nsys completed in {elapsed:.0f}s")
    logger.info(result.stdout[-1000:])

    return {
        "config_key":  config_key,
        "status":      "ok",
        "elapsed_s":   round(elapsed, 1),
        "rep_file":    str(rep_file),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parse nsys stats GPU-trace CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_and_parse_stats(
    config_key: str,
    nsys_bin: str,
    output_dir: Path,
) -> dict:
    """
    Run `nsys stats --report gputrace --format csv` on the .nsys-rep file
    to extract a GPU kernel summary CSV, then parse it into a structured dict.

    nsys stats report types used here:
      gputrace  — per-kernel execution times and counts
      (cudaapisum — CUDA API call summary, available but not parsed here)

    Returns a dict with keys: top_kernels, total_gpu_time_ms, config_key.
    """
    rep_file = output_dir / f"{config_key}_nsys.nsys-rep"
    csv_file = output_dir / f"{config_key}_nsys_gputrace.csv"

    if not rep_file.exists():
        logger.warning(f"[{config_key}] .nsys-rep not found — cannot export stats")
        return {}

    # ── Export GPU trace CSV ────────────────────────────────────────────────
    stats_cmd = [
        nsys_bin, "stats",
        "--report",   "gputrace",
        "--format",   "csv",
        "--output",   str(csv_file.with_suffix("")),   # nsys appends _gputrace.csv
        str(rep_file),
    ]
    logger.info(f"[{config_key}] Exporting GPU trace CSV...")
    result = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.warning(f"[{config_key}] nsys stats failed: {result.stderr[:500]}")
        return {}

    # nsys appends the report type to the output path
    actual_csv = output_dir / f"{config_key}_nsys_gputrace.csv"
    if not actual_csv.exists():
        # nsys may have written {base}_gputrace.csv
        candidates = list(output_dir.glob(f"{config_key}*gputrace*"))
        if candidates:
            actual_csv = candidates[0]
            logger.info(f"  Found GPU trace CSV at: {actual_csv}")
        else:
            logger.warning(f"[{config_key}] GPU trace CSV not found after nsys stats")
            return {}

    # ── Parse the CSV ───────────────────────────────────────────────────────
    # nsys gputrace CSV columns (nsys >= 2023.1):
    #   "Start (ns)", "Duration (ns)", "CorrId", "GrdX", "GrdY", "GrdZ",
    #   "BlkX", "BlkY", "BlkZ", "Reg/Trd", "StcSMem (MB)", "DymSMem (MB)",
    #   "Bytes (MB)", "Throughput (MBps)", "SrcMemType", "DstMemType", "Device",
    #   "Ctx", "GreenCtx", "Strm", "Name"
    import csv
    from collections import defaultdict

    kernel_times = defaultdict(lambda: {"total_ns": 0, "count": 0})
    total_gpu_ns = 0

    try:
        with open(actual_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name     = row.get("Name", "unknown").strip()
                try:
                    duration = float(row.get("Duration (ns)", 0) or 0)
                except ValueError:
                    continue

                kernel_times[name]["total_ns"] += duration
                kernel_times[name]["count"]    += 1
                total_gpu_ns                   += duration
    except Exception as e:
        logger.warning(f"[{config_key}] CSV parse error: {e}")
        return {}

    # Sort by total time and build top-10 list
    sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1]["total_ns"], reverse=True)
    top_kernels = [
        {
            "name":          name,
            "total_ms":      round(info["total_ns"] / 1e6, 3),
            "pct":           round(info["total_ns"] / total_gpu_ns * 100, 2) if total_gpu_ns else 0,
            "count":         info["count"],
            "avg_us":        round(info["total_ns"] / info["count"] / 1e3, 2),
        }
        for name, info in sorted_kernels[:10]
    ]

    parsed = {
        "config_key":         config_key,
        "total_gpu_time_ms":  round(total_gpu_ns / 1e6, 3),
        "top_kernels":        top_kernels,
        "source":             "nsys_gputrace",
        "csv_file":           str(actual_csv),
    }

    # Save parsed JSON alongside the CSV
    json_out = output_dir / f"{config_key}_nsys_summary.json"
    with open(json_out, "w") as f:
        json.dump(parsed, f, indent=2)
    logger.info(f"  [{ config_key}] nsys summary → {json_out}")

    logger.info(f"  Total GPU time in profiling region: {parsed['total_gpu_time_ms']:.1f} ms")
    logger.info(f"  Top 3 kernels (nsys):")
    for k in top_kernels[:3]:
        logger.info(f"    {k['name'][:55]:<55} {k['total_ms']:>8.1f} ms  ({k['pct']:.1f}%)")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_nsys_sweep(
    config_keys: list[str],
    nsys_bin: str,
    output_dir: Path,
    repo_dir: Path,
    hf_token: str = None,
    profile_steps: int = 3,
    n_tokens: int = 50,
    force: bool = False,
) -> dict:
    """Profile each config with nsys and parse the GPU trace stats."""
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_key in config_keys:
        profile_result = run_nsys_profile(
            config_key=config_key,
            nsys_bin=nsys_bin,
            output_dir=output_dir,
            repo_dir=repo_dir,
            hf_token=hf_token,
            profile_steps=profile_steps,
            n_tokens=n_tokens,
            force=force,
        )
        stats_result = export_and_parse_stats(
            config_key=config_key,
            nsys_bin=nsys_bin,
            output_dir=output_dir,
        )
        results[config_key] = {**profile_result, **stats_result}

    # ── Print comparison table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" nsys Profiling Summary — Mistral-7B PTQ sweep")
    print(f"{'='*70}")
    print(f"  {'Config':<30} {'Status':>8} {'GPU time (ms)':>14} {'Top kernel':<35}")
    print(f"  {'-'*68}")
    for key in config_keys:
        r = results.get(key, {})
        status    = r.get("status", "missing")
        gpu_ms    = r.get("total_gpu_time_ms", 0)
        top_kern  = (r.get("top_kernels") or [{}])[0].get("name", "N/A")[:34]
        desc      = MODEL_REGISTRY[key]["description"]
        print(f"  {desc:<30} {status:>8} {gpu_ms:>14.1f} {top_kern:<35}")
    print()

    # Save sweep summary
    sweep_json = output_dir / "nsys_sweep_summary.json"
    with open(sweep_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sweep summary saved → {sweep_json}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Nsight Systems profiling sweep for Mistral-7B PTQ configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       type=str, choices=ALL_CONFIGS,
                   help="Profile a single config")
    p.add_argument("--all",          action="store_true",
                   help="Profile all configs sequentially")
    p.add_argument("--force",        action="store_true",
                   help="Re-profile even if .nsys-rep exists")
    p.add_argument("--output-dir",   type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--repo-dir",     type=str, default=".",
                   help="Path to TeamB directory containing run_profiler.py")
    p.add_argument("--nsys-path",    type=str, default=None,
                   help="Explicit path to nsys binary (default: auto-detect)")
    p.add_argument("--profile-steps",type=int, default=3,
                   help="Number of profiled generate() steps (default: 3). "
                        "Fewer than PyTorch Profiler since .nsys-rep files are large.")
    p.add_argument("--n-tokens",     type=int, default=50)
    p.add_argument("--hf-token",     type=str, default=None)
    p.add_argument("--stats-only",   action="store_true",
                   help="Skip profiling; only (re-)parse existing .nsys-rep files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    repo_dir   = Path(args.repo_dir).resolve()

    # ── Locate nsys ──────────────────────────────────────────────────────────
    try:
        nsys_bin = find_nsys(args.nsys_path)
        logger.info(f"nsys version: {get_nsys_version(nsys_bin)}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # ── Detect GPU and confirm A100 (warn if different) ──────────────────────
    gpu_info = detect_gpu_architecture()
    if gpu_info["arch"] != "ampere_ga100":
        logger.warning(
            f"Expected A100 (ampere_ga100) but detected arch='{gpu_info['arch']}' "
            f"({gpu_info['name']}).  All profiling results and roofline values in this "
            f"project are calibrated against A100-80GB.  The nsys metrics-set will be "
            f"set to '{gpu_info['metrics_set']}' automatically, but throughput and "
            f"bandwidth figures will differ from the project baseline."
        )
    else:
        logger.info(
            f"Confirmed A100.  nsys --gpu-metrics-set={gpu_info['metrics_set']}"
        )

    # ── Verify run_profiler.py is present ────────────────────────────────────
    profiler_script = repo_dir / "run_profiler.py"
    if not profiler_script.exists():
        logger.error(f"run_profiler.py not found at {profiler_script}")
        sys.exit(1)

    # ── Target configs ───────────────────────────────────────────────────────
    if args.all:
        targets = ALL_CONFIGS
    elif args.config:
        targets = [args.config]
    else:
        print("Specify --config <key> or --all")
        sys.exit(1)

    if args.stats_only:
        # Just (re-)parse existing .nsys-rep files
        for config_key in targets:
            export_and_parse_stats(config_key, nsys_bin, output_dir)
    else:
        run_nsys_sweep(
            config_keys=targets,
            nsys_bin=nsys_bin,
            output_dir=output_dir,
            repo_dir=repo_dir,
            hf_token=hf_token,
            profile_steps=args.profile_steps,
            n_tokens=args.n_tokens,
            force=args.force,
        )
