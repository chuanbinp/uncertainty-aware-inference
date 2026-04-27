"""
run_ncu.py
─────────────────────────────────────────────────────────────────────────────
Nsight Compute (ncu) launcher for Mistral-7B PTQ sweep.

ncu replays each targeted kernel multiple times to collect hardware counters:
SM occupancy, memory throughput, compute throughput, achieved FLOPs, and
DRAM bytes.  These are the ground-truth inputs for the Roofline model and
for diagnosing whether a kernel is compute-bound or memory-bound.

This script:
  1. Reads existing profiler_results/{config}_profile.json to identify the
     top-CUDA-time kernels per config (targeting only those avoids the
     prohibitive cost of profiling every kernel with counter replay).
  2. Builds a per-config kernel target regex from those kernel names.
  3. Runs `ncu --metrics ... --kernel-name "regex:..."` as a subprocess.
  4. Parses the CSV output into per-kernel metric dicts.
  5. Saves {config_key}_ncu_metrics.json for nsight_roofline.py.

Why kernel-targeted ncu instead of profiling everything:
  ncu uses kernel replay: each targeted kernel is re-executed N times (once
  per metric section) with performance counters enabled.  On a 7B model with
  ~11,000 matmul calls per generate() this would take hours.  By targeting
  only the 2-3 kernels that account for >80% of CUDA time, ncu finishes in
  ~5-10 minutes per config.

Requirements:
  - ncu >= 2023.1 (ships with CUDA Toolkit 12.x)
  - Root or sudo with perf_event_paranoid ≤ 1:
      sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
  - GCP VM or bare-metal GPU instance (NOT Colab free tier)

Usage:
    # Requires sudo on most systems:
    sudo python run_ncu.py --config mistral-7b-gptq-int4
    sudo python run_ncu.py --all
    sudo python run_ncu.py --config mistral-7b-fp16 --ncu-path /usr/local/cuda/bin/ncu

    # Profile steps = 1 is sufficient for ncu (kernel replay gives stable counts)
    sudo python run_ncu.py --all --profile-steps 1
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from configs import MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ncu")

ALL_CONFIGS          = list(MODEL_REGISTRY.keys())
DEFAULT_OUTPUT_DIR   = Path("./ncu_results")
DEFAULT_PROF_DIR     = Path("./profiler_results")   # for loading existing top-kernel lists


# ─────────────────────────────────────────────────────────────────────────────
# Hardware metrics to collect
# ─────────────────────────────────────────────────────────────────────────────
# These metrics are sufficient to compute arithmetic intensity and place each
# kernel on the Roofline.  They are available on all Ampere GPUs (A100, A10, L4).
#
# Metric naming follows the CUDA Profiling Tools Interface (CUPTI) convention.
# Verified against ncu 2023.3 on CUDA 12.2.
#
# ┌─────────────────────────────────────────────────────────────┬──────────────────────────────┐
# │ Metric name                                                 │ What it measures             │
# ├─────────────────────────────────────────────────────────────┼──────────────────────────────┤
# │ gpu__time_duration.sum                                      │ kernel duration (ns)         │
# │ dram__bytes.sum                                             │ DRAM read + write bytes      │
# │ l1tex__t_bytes_lookup_miss.sum                              │ L1→L2 miss bytes             │
# │ lts__t_bytes_lookup_miss.sum                                │ L2→DRAM miss bytes           │
# │ smsp__sass_thread_inst_executed_op_ffma_pred_on.sum         │ FP32 FMA count (=2 FLOPs)   │
# │ smsp__sass_thread_inst_executed_op_fadd_pred_on.sum         │ FP32 ADD count               │
# │ smsp__sass_thread_inst_executed_op_fmul_pred_on.sum         │ FP32 MUL count               │
# │ sm__ops_path_tensor_src_fp16.sum                            │ FP16 Tensor Core ops        │
# │ sm__throughput.avg.pct_of_peak_sustained_elapsed            │ SM compute utilisation %    │
# │ gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed      │ DRAM bandwidth utilisation %│
# │ sm__warps_active.avg.pct_of_peak_sustained_active           │ warp occupancy %             │
# └─────────────────────────────────────────────────────────────┴──────────────────────────────┘
NCU_METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "l1tex__t_bytes_lookup_miss.sum",
    "lts__t_bytes_lookup_miss.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__ops_path_tensor_src_fp16.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
])

# Also request the Speed of Light section for a quick sanity-check view in
# the .ncu-rep file (this is what the Nsight Compute GUI Roofline chart uses).
NCU_SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-config kernel target patterns
# ─────────────────────────────────────────────────────────────────────────────
# These regex patterns target the kernels that account for >80% of CUDA time
# for each config (identified from profiler_summary.json).
#
# ncu uses these with --kernel-name-base=demangled --kernel-name="regex:..."
# to restrict counter collection to just those kernels rather than replaying
# every single kernel in the model (which would take hours).
#
# Patterns are intentionally broad to catch variants across batch sizes.
# The static defaults below are the best effort; if you have a
# profiler_summary.json from run_profiler.py, the dynamic version
# (build_kernel_targets_from_profile()) is preferred.
STATIC_KERNEL_TARGETS = {
    # ── Mistral-7B ────────────────────────────────────────────────────────────
    # FP16: cuBLAS dispatches ampere_fp16_s16816gemm_* tiles on A100 (GA100).
    # The full kernel name varies by problem size (e.g. _128x64_ldg8_f2f_...)
    # so we match the common prefix only.  Confirmed from profiler_summary.json.
    "mistral-7b-fp16":      "ampere_fp16_s16816gemm",

    # GPTQ: QuantLinearFunction is the Python-registered op; dequant_kernel is
    # the explicit dequantization kernel that precedes the cuBLAS GEMM call.
    # Both appear in the profiler_summary.json top-kernels for INT4 and INT8.
    "mistral-7b-gptq-int4": "QuantLinearFunction|dequant_kernel",
    "mistral-7b-gptq-int8": "QuantLinearFunction|dequant_kernel",

    # AWQ: awq_gemm_kernel is the fused W4A16 GEMV kernel from the AutoAWQ
    # CUDA extension.  WQLinearMMFunction is the Python dispatcher for it.
    "mistral-7b-awq-int4":  "awq_gemm_kernel|WQLinearMMFunction",

    # NF4: bitsandbytes uses kgemm_4bit_inference_naive<__half, 128, 16> on A100.
    # gemv_4bit is the registered op name; the template-specialised CUDA kernel
    # name is caught by the kgemm_4bit prefix.
    "mistral-7b-nf4":       "kgemm_4bit_inference_naive|gemv_4bit",

    # ── Llama-2 7B ───────────────────────────────────────────────────────────
    # Same underlying libraries as Mistral-7B — kernel names are identical
    # since both are 7B attention-transformer models on the same GPU.
    "llama2-7b-fp16":       "ampere_fp16_s16816gemm",
    "llama2-7b-gptq-int4":  "QuantLinearFunction|dequant_kernel",
    "llama2-7b-gptq-int8":  "QuantLinearFunction|dequant_kernel",
    "llama2-7b-awq-int4":   "awq_gemm_kernel|WQLinearMMFunction",
    "llama2-7b-nf4":        "kgemm_4bit_inference_naive|gemv_4bit",

    # ── Llama-2 13B ──────────────────────────────────────────────────────────
    # 13B has larger matrix tiles — cuBLAS may select a different gemm variant
    # (e.g. _256x64 instead of _128x64) but the ampere_fp16_s16816gemm prefix
    # covers all tile sizes so the regex still matches.
    "llama2-13b-fp16":      "ampere_fp16_s16816gemm",
    "llama2-13b-gptq-int4": "QuantLinearFunction|dequant_kernel",
    "llama2-13b-gptq-int8": "QuantLinearFunction|dequant_kernel",
    "llama2-13b-awq-int4":  "awq_gemm_kernel|WQLinearMMFunction",
    "llama2-13b-nf4":       "kgemm_4bit_inference_naive|gemv_4bit",
}


def build_kernel_targets_from_profile(config_key: str, prof_dir: Path, top_n: int = 2) -> str:
    """
    Build a kernel regex from an existing profile JSON.  Extracts the top_n
    kernels by CUDA time and joins their names with '|'.

    Falls back to STATIC_KERNEL_TARGETS if the JSON is missing.
    """
    json_path = prof_dir / f"{config_key}_profile.json"
    if not json_path.exists():
        logger.warning(
            f"[{config_key}] No profile JSON at {json_path} — using static kernel targets"
        )
        return STATIC_KERNEL_TARGETS.get(config_key, "")

    with open(json_path) as f:
        prof = json.load(f)

    top_kernels = prof.get("top_kernels", [])[:top_n]
    if not top_kernels:
        return STATIC_KERNEL_TARGETS.get(config_key, "")

    # Filter out PyTorch dispatcher wrappers (aten::*) — these are not real
    # CUDA kernels and ncu cannot target them by name.  Keep only GPU kernels
    # with non-generic names.
    aten_prefixes = ("aten::", "c10::", "at::", "inference_step_")
    gpu_kernels = [
        k["name"] for k in top_kernels
        if not any(k["name"].startswith(p) for p in aten_prefixes)
    ]

    if not gpu_kernels:
        # All top kernels are aten wrappers — fall back to static targets
        return STATIC_KERNEL_TARGETS.get(config_key, "")

    # Escape special regex chars in kernel names (template brackets, etc.)
    # then join with | for OR matching
    def safe_pattern(name: str) -> str:
        # Take the base function name before template args to avoid fragility
        base = name.split("<")[0].split("(")[0].strip()
        return re.escape(base)

    pattern = "|".join(safe_pattern(k) for k in gpu_kernels)
    logger.info(f"[{config_key}] Kernel target pattern (from profile): {pattern}")
    return pattern


# ─────────────────────────────────────────────────────────────────────────────
# ncu binary detection
# ─────────────────────────────────────────────────────────────────────────────

NCU_SEARCH_PATHS = [
    "ncu",
    "/usr/local/cuda/bin/ncu",
    "/usr/local/cuda-12/bin/ncu",
    "/usr/local/cuda-11/bin/ncu",
    "/opt/nvidia/nsight-compute/2023.3.0/ncu",
    "/opt/nvidia/nsight-compute/2024.1.0/ncu",
]


def find_ncu(explicit_path: str = None) -> str:
    candidates = ([explicit_path] if explicit_path else []) + NCU_SEARCH_PATHS
    for candidate in candidates:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if resolved:
            logger.info(f"ncu found: {resolved}")
            return resolved
    raise FileNotFoundError(
        "ncu (Nsight Compute CLI) not found. "
        "On GCP with CUDA 12: /usr/local/cuda/bin/ncu. "
        "Install via: apt-get install -y nsight-compute. "
        "Also requires: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'. "
        "NOT available on Colab free tier."
    )


def get_ncu_version(ncu_bin: str) -> str:
    try:
        result = subprocess.run([ncu_bin, "--version"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip().split("\n")[0]
    except Exception:
        return "unknown"


def check_perf_event_paranoid() -> bool:
    """
    Verify kernel.perf_event_paranoid ≤ 1 (required for ncu hardware counters).
    Returns True if OK, False if ncu will likely fail silently.
    """
    try:
        val = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip())
        if val > 1:
            logger.warning(
                f"  kernel.perf_event_paranoid = {val} (must be ≤ 1 for ncu hardware counters). "
                f"Fix: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'"
            )
            return False
        logger.info(f"  perf_event_paranoid = {val} ✓")
        return True
    except Exception:
        logger.warning("  Cannot read /proc/sys/kernel/perf_event_paranoid (non-Linux?)")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Single-config ncu run
# ─────────────────────────────────────────────────────────────────────────────

def run_ncu_profile(
    config_key: str,
    ncu_bin: str,
    output_dir: Path,
    repo_dir: Path,
    prof_dir: Path,
    hf_token: str = None,
    profile_steps: int = 1,
    n_tokens: int = 50,
    force: bool = False,
) -> Path:
    """
    Run ncu for one config, targeting its dominant kernels.

    Saves:
      {config_key}_ncu.ncu-rep  — Nsight Compute report (open in ncu GUI)
      {config_key}_ncu_raw.csv  — raw metrics CSV for programmatic parsing

    Returns the path to the CSV file (or None on failure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / f"{config_key}_ncu_raw.csv"
    rep_file = output_dir / f"{config_key}_ncu.ncu-rep"

    if csv_file.exists() and not force:
        logger.info(f"[{config_key}] ncu CSV exists — skipping (use --force to re-run)")
        return csv_file

    cfg = MODEL_REGISTRY[config_key]
    logger.info(f"\n{'='*65}")
    logger.info(f" ncu profile: {cfg['description']}")
    logger.info(f"{'='*65}")

    # ── Kernel target pattern ────────────────────────────────────────────────
    kernel_pattern = build_kernel_targets_from_profile(config_key, prof_dir)
    if not kernel_pattern:
        kernel_pattern = STATIC_KERNEL_TARGETS.get(config_key, "")
    if not kernel_pattern:
        logger.error(f"[{config_key}] No kernel target pattern — skipping")
        return None

    # ── Build ncu command ────────────────────────────────────────────────────
    # --replay-mode=kernel (default): each targeted kernel is replayed once
    # per metrics pass.  Safe for quantized models that have side effects.
    # --replay-mode=application replays the full Python process per pass —
    # more accurate but much slower; use only if kernel replay gives wrong counts.
    ncu_cmd = [
        ncu_bin,
        "--target-processes",    "all",
        "--replay-mode",         "kernel",
        "--kernel-name-base",    "demangled",
        "--kernel-name",         f"regex:{kernel_pattern}",
        "--metrics",             NCU_METRICS,
        # Also collect Speed of Light section (shown in GUI Roofline chart)
        "--section",             "SpeedOfLight",
        "--section",             "MemoryWorkloadAnalysis",
        # NVTX filter: only profile kernels that run inside the "profiling_region"
        # NVTX range (requires run_profiler.py --nvtx)
        "--nvtx",
        "--nvtx-include",        "profiling_region",
        # Export formats
        "--csv",                                    # write metrics CSV to stdout
        "--log-file",            "/dev/null",       # suppress per-kernel text to stderr
        "--export",              str(rep_file.with_suffix("")),  # also write .ncu-rep
        "--force-overwrite",
    ]

    # ── Python command that ncu wraps ────────────────────────────────────────
    profiler_script = repo_dir / "run_profiler.py"
    python_cmd = [
        sys.executable,
        str(profiler_script),
        "--config",         config_key,
        "--nvtx",                               # required for --nvtx-include
        "--profile-steps",  str(profile_steps), # 1 is enough — ncu uses kernel replay
        "--n-tokens",       str(n_tokens),
        "--warmup-steps",   "1",
        "--output-dir",     str(output_dir),
    ]

    full_cmd = ncu_cmd + python_cmd
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    logger.info("CMD: " + " ".join(full_cmd))
    logger.info("(This may take 5-15 minutes due to kernel replay — ncu re-runs each kernel multiple times)")

    t0 = time.perf_counter()
    result = subprocess.run(full_cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        logger.error(f"[{config_key}] ncu FAILED (exit {result.returncode})")
        logger.error("STDERR:\n" + result.stderr[-2000:])
        return None

    logger.info(f"[{config_key}] ncu completed in {elapsed:.0f}s")

    # ncu --csv writes to stdout
    if not result.stdout.strip():
        logger.warning(f"[{config_key}] ncu produced no CSV output — check kernel target pattern")
        return None

    csv_file.write_text(result.stdout, encoding="utf-8")
    logger.info(f"  Raw CSV → {csv_file}")
    return csv_file


# ─────────────────────────────────────────────────────────────────────────────
# Parse ncu CSV output
# ─────────────────────────────────────────────────────────────────────────────

def parse_ncu_csv(csv_file: Path, config_key: str) -> dict:
    """
    Parse ncu --csv output into a structured dict keyed by kernel name.

    ncu CSV format (ncu >= 2023.1 with --csv):
        "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
        "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

    Each row is one (kernel, section, metric) triple.  We group by kernel name
    and pivot the metrics into a flat dict per kernel.

    Returns:
        {
          "kernels": {
            "kernel_name": {
              "gpu__time_duration.sum": 1234567.0,   # nanoseconds
              "dram__bytes.sum": 89012345.0,
              ...
            }
          },
          "config_key": "...",
          "source": "ncu_csv",
        }
    """
    if not csv_file or not csv_file.exists():
        return {}

    kernels = defaultdict(dict)
    try:
        with open(csv_file, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel_name = row.get("Kernel Name", "").strip().strip('"')
                metric_name = row.get("Metric Name", "").strip().strip('"')
                metric_val  = row.get("Metric Value", "").strip().strip('"')

                if not kernel_name or not metric_name:
                    continue

                # Parse numeric value — ncu may use commas as thousands separators
                try:
                    val = float(metric_val.replace(",", ""))
                except ValueError:
                    val = metric_val  # keep as string for non-numeric metrics

                # Accumulate across replay passes (ncu averages are already
                # computed; .sum metrics are totals across the kernel invocations
                # that were captured)
                if metric_name in kernels[kernel_name]:
                    # Multiple rows for the same kernel+metric = multiple invocations
                    # We average them (ncu replay already handles most averaging)
                    existing = kernels[kernel_name][metric_name]
                    if isinstance(existing, float) and isinstance(val, float):
                        kernels[kernel_name][metric_name] = (existing + val) / 2
                else:
                    kernels[kernel_name][metric_name] = val

    except Exception as e:
        logger.error(f"ncu CSV parse error: {e}")
        return {}

    return {
        "config_key": config_key,
        "source":     "ncu_csv",
        "kernels":    dict(kernels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compute Roofline metrics per kernel from parsed ncu data
# ─────────────────────────────────────────────────────────────────────────────

# ── Hardware specs for Roofline bound classification ────────────────────────
# All experiments in this project run on A100-80GB.  Other entries exist so
# the script errors clearly rather than silently computing wrong ridge values
# if run on a different GPU.  Values sourced from NVIDIA datasheets and
# published microbenchmarks (L2/L1 bandwidths are approximate).
GPU_SPECS = {
    "A100-80GB": {
        "gpu_name":           "A100-80GB",
        "peak_fp16_tflops":   312.0,   # Tensor Core FP16 (NVIDIA A100 datasheet)
        "peak_fp32_tflops":   19.5,    # FP32 CUDA Core peak
        "dram_bw_tb_s":       2.0,     # HBM2e — 2,039 GB/s rounded to 2.0 TB/s
        "l2_bw_tb_s":         12.0,    # L2 read bandwidth (approx)
        "l1_bw_tb_s":         33.0,    # L1/SMEM bandwidth (approx)
        "ridge_dram":         156.0,   # 312 TFLOPS / 2 TB/s
        "ridge_l2":           26.0,    # 312 / 12
    },
    "A100-40GB": {
        "gpu_name":           "A100-40GB",
        "peak_fp16_tflops":   312.0,
        "peak_fp32_tflops":   19.5,
        "dram_bw_tb_s":       1.555,   # 1,555 GB/s HBM2
        "l2_bw_tb_s":         8.0,
        "l1_bw_tb_s":         33.0,
        "ridge_dram":         200.6,
        "ridge_l2":           39.0,
    },
    "L4": {
        "gpu_name":           "L4",
        "peak_fp16_tflops":   121.0,
        "peak_fp32_tflops":   30.3,
        "dram_bw_tb_s":       0.300,   # 300 GB/s GDDR6
        "l2_bw_tb_s":         4.0,
        "l1_bw_tb_s":         20.0,
        "ridge_dram":         403.3,
        "ridge_l2":           30.25,
    },
    "V100-SXM2": {
        "gpu_name":           "V100-SXM2",
        "peak_fp16_tflops":   125.0,
        "peak_fp32_tflops":   15.7,
        "dram_bw_tb_s":       0.9,
        "l2_bw_tb_s":         4.5,
        "l1_bw_tb_s":         15.0,
        "ridge_dram":         138.9,
        "ridge_l2":           27.8,
    },
}

DEFAULT_GPU = "A100-80GB"   # project GPU — used everywhere as the default
A100_SPECS  = GPU_SPECS["A100-80GB"]   # alias kept for backward compatibility


def compute_roofline_metrics(parsed: dict, gpu_specs: dict = A100_SPECS) -> dict:
    """
    For each kernel in the parsed ncu dict, compute:
      - flops:         total floating-point operations (FP16 tensor + FP32 CUDA cores)
      - dram_bytes:    DRAM traffic
      - duration_s:    kernel duration in seconds
      - ai_dram:       arithmetic intensity against DRAM (FLOPs/byte)
      - ai_l2:         arithmetic intensity against L2
      - achieved_tflops: FLOPs / duration
      - sm_pct:        SM utilisation % from Speed-of-Light
      - mem_pct:       DRAM utilisation %
      - bound:         "Compute-bound", "Memory-bound (DRAM)", "Memory-bound (L2)", "Unknown"
    """
    ridge_dram = (gpu_specs["peak_fp16_tflops"] * 1e12) / (gpu_specs["dram_bw_tb_s"] * 1e12)
    ridge_l2   = (gpu_specs["peak_fp16_tflops"] * 1e12) / (gpu_specs["l2_bw_tb_s"]  * 1e12)

    roofline = {}
    for kname, metrics in parsed.get("kernels", {}).items():
        def m(key, default=0.0):
            v = metrics.get(key, default)
            return float(v) if isinstance(v, (int, float)) else default

        # ── FLOPs ──────────────────────────────────────────────────────────
        # FP16 Tensor Core ops (each op = 1 multiply-accumulate = 2 FLOPs)
        fp16_tensor_ops = m("sm__ops_path_tensor_src_fp16.sum")
        fp16_flops      = fp16_tensor_ops * 2

        # FP32 CUDA core ops
        fp32_fma_flops = m("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum") * 2
        fp32_add_flops = m("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
        fp32_mul_flops = m("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
        fp32_flops     = fp32_fma_flops + fp32_add_flops + fp32_mul_flops

        total_flops = fp16_flops + fp32_flops

        # ── Memory bytes ───────────────────────────────────────────────────
        dram_bytes = m("dram__bytes.sum")
        l2_bytes   = m("lts__t_bytes_lookup_miss.sum")   # L2→DRAM misses ≈ DRAM traffic
        l1_bytes   = m("l1tex__t_bytes_lookup_miss.sum")  # L1→L2 misses

        # ── Duration ───────────────────────────────────────────────────────
        duration_ns = m("gpu__time_duration.sum")
        duration_s  = duration_ns / 1e9 if duration_ns > 0 else None

        # ── Derived metrics ────────────────────────────────────────────────
        ai_dram = total_flops / dram_bytes if dram_bytes > 0 else 0.0
        ai_l2   = total_flops / l2_bytes   if l2_bytes   > 0 else 0.0

        achieved_tflops = (total_flops / duration_s / 1e12) if duration_s else 0.0

        # Speed-of-Light percentages (already in the CSV from --section SpeedOfLight)
        sm_pct  = m("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        mem_pct = m("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
        occ_pct = m("sm__warps_active.avg.pct_of_peak_sustained_active")

        # ── Bound classification ───────────────────────────────────────────
        if total_flops == 0 and sm_pct == 0:
            bound = "Unknown (no FLOPs or SM% measured)"
        elif ai_dram > ridge_dram * 1.1:
            bound = "Compute-bound"
        elif ai_l2 > ridge_l2 * 1.1 and l2_bytes > 0:
            bound = "Memory-bound (L2)"
        else:
            bound = "Memory-bound (DRAM)"

        roofline[kname] = {
            "flops":           total_flops,
            "fp16_tensor_flops": fp16_flops,
            "fp32_flops":      fp32_flops,
            "dram_bytes":      dram_bytes,
            "l2_bytes":        l2_bytes,
            "duration_ns":     duration_ns,
            "ai_dram":         round(ai_dram, 3),
            "ai_l2":           round(ai_l2, 3),
            "achieved_tflops": round(achieved_tflops, 4),
            "sm_pct":          round(sm_pct, 2),
            "mem_pct":         round(mem_pct, 2),
            "occupancy_pct":   round(occ_pct, 2),
            "bound":           bound,
        }

    return {
        "config_key":  parsed.get("config_key", ""),
        "gpu_specs":   gpu_specs,
        "ridge_dram":  round(ridge_dram, 1),
        "ridge_l2":    round(ridge_l2, 1),
        "kernels":     roofline,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_ncu_sweep(
    config_keys: list[str],
    ncu_bin: str,
    output_dir: Path,
    repo_dir: Path,
    prof_dir: Path,
    hf_token: str = None,
    profile_steps: int = 1,
    n_tokens: int = 50,
    force: bool = False,
    gpu_specs: dict = None,
) -> dict:
    """Run ncu for each config, parse results, and save per-config metric JSONs."""
    if gpu_specs is None:
        gpu_specs = A100_SPECS
    output_dir.mkdir(parents=True, exist_ok=True)
    all_roofline = {}

    for config_key in config_keys:
        csv_file = run_ncu_profile(
            config_key=config_key,
            ncu_bin=ncu_bin,
            output_dir=output_dir,
            repo_dir=repo_dir,
            prof_dir=prof_dir,
            hf_token=hf_token,
            profile_steps=profile_steps,
            n_tokens=n_tokens,
            force=force,
        )

        if csv_file is None:
            logger.warning(f"[{config_key}] Skipping parse — no CSV produced")
            continue

        parsed   = parse_ncu_csv(csv_file, config_key)
        roofline = compute_roofline_metrics(parsed, gpu_specs=gpu_specs)

        # Save metrics JSON
        metrics_json = output_dir / f"{config_key}_ncu_metrics.json"
        with open(metrics_json, "w") as f:
            json.dump({**parsed, "roofline": roofline}, f, indent=2)
        logger.info(f"  [{config_key}] Metrics saved → {metrics_json}")

        # Print per-kernel roofline summary
        for kname, r in roofline["kernels"].items():
            logger.info(
                f"  {kname[:45]:<45}  "
                f"AI(DRAM)={r['ai_dram']:>8.1f}  "
                f"achieved={r['achieved_tflops']:>6.2f} TFLOPS  "
                f"SM={r['sm_pct']:>4.0f}%  "
                f"Mem={r['mem_pct']:>4.0f}%  "
                f"→ {r['bound']}"
            )

        all_roofline[config_key] = roofline

    # Save sweep summary
    sweep_json = output_dir / "ncu_sweep_summary.json"
    with open(sweep_json, "w") as f:
        json.dump(all_roofline, f, indent=2)
    logger.info(f"\nSweep summary → {sweep_json}")

    # ── Print comparison table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f" ncu Roofline Summary — Mistral-7B PTQ sweep")
    print(f"{'='*80}")
    print(f"  {'Config':<28} {'Kernel':<30} {'AI(DRAM)':>10} {'TFLOPS':>7} {'SM%':>5} {'Bound'}")
    print(f"  {'-'*78}")
    for key in config_keys:
        rl = all_roofline.get(key, {})
        for kname, r in rl.get("kernels", {}).items():
            desc = MODEL_REGISTRY[key]["description"]
            print(
                f"  {desc:<28} {kname[:30]:<30} "
                f"{r['ai_dram']:>10.1f} {r['achieved_tflops']:>7.2f} "
                f"{r['sm_pct']:>5.0f}  {r['bound']}"
            )
    print()

    return all_roofline


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Nsight Compute profiling sweep for Mistral-7B PTQ configs"
    )
    p.add_argument("--config",        type=str, choices=ALL_CONFIGS)
    p.add_argument("--all",           action="store_true")
    p.add_argument("--force",         action="store_true")
    p.add_argument("--output-dir",    type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--repo-dir",      type=str, default=".")
    p.add_argument("--prof-dir",      type=str, default=str(DEFAULT_PROF_DIR),
                   help="Directory with existing *_profile.json files from run_profiler.py")
    p.add_argument("--ncu-path",      type=str, default=None)
    p.add_argument("--gpu",           type=str, default=DEFAULT_GPU,
                   choices=list(GPU_SPECS.keys()),
                   help=f"GPU spec for Roofline ridge-point calculation (default: {DEFAULT_GPU}). "
                        "All project experiments ran on A100-80GB — only change this if you "
                        "intentionally run ncu on a different GPU.")
    p.add_argument("--profile-steps", type=int, default=1,
                   help="1 is sufficient for ncu (kernel replay gives stable counts)")
    p.add_argument("--n-tokens",      type=int, default=50)
    p.add_argument("--hf-token",      type=str, default=None)
    p.add_argument("--parse-only",    action="store_true",
                   help="Skip profiling; only (re-)parse existing CSV files")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    hf_token   = args.hf_token or os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    repo_dir   = Path(args.repo_dir).resolve()
    prof_dir   = Path(args.prof_dir)

    # ── Resolve GPU specs and confirm we're on A100 ───────────────────────────
    selected_specs = GPU_SPECS[args.gpu]
    logger.info(
        f"GPU spec selected: {args.gpu}  "
        f"(FP16 peak={selected_specs['peak_fp16_tflops']:.0f} TFLOPS, "
        f"DRAM BW={selected_specs['dram_bw_tb_s']:.1f} TB/s, "
        f"ridge={selected_specs['ridge_dram']:.0f} FLOPs/B)"
    )

    # Detect the actual GPU and warn if it doesn't match --gpu
    try:
        import torch
        actual_name = torch.cuda.get_device_name(0)
        gpu_key_lower = args.gpu.lower().replace("-","")
        actual_lower  = actual_name.lower().replace(" ","").replace("-","")
        if not any(k in actual_lower for k in gpu_key_lower.split("80gb")[0].split("40gb")[0]):
            logger.warning(
                f"--gpu={args.gpu} but detected GPU is '{actual_name}'. "
                f"Roofline bounds will use {args.gpu} specs — verify this is intentional."
            )
        else:
            logger.info(f"Confirmed: running on {actual_name} ✓")
    except Exception:
        pass

    # ── Locate ncu ────────────────────────────────────────────────────────────
    try:
        ncu_bin = find_ncu(args.ncu_path)
        logger.info(f"ncu version: {get_ncu_version(ncu_bin)}")
    except FileNotFoundError as e:
        logger.error(str(e)); sys.exit(1)

    # ── Check perf permissions ────────────────────────────────────────────────
    check_perf_event_paranoid()

    # ── Target configs ────────────────────────────────────────────────────────
    if args.all:       targets = ALL_CONFIGS
    elif args.config:  targets = [args.config]
    else:
        print("Specify --config <key> or --all"); sys.exit(1)

    if args.parse_only:
        # Parse existing CSV files only
        for config_key in targets:
            csv_file = output_dir / f"{config_key}_ncu_raw.csv"
            parsed   = parse_ncu_csv(csv_file, config_key)
            roofline = compute_roofline_metrics(parsed, gpu_specs=selected_specs)
            metrics_json = output_dir / f"{config_key}_ncu_metrics.json"
            with open(metrics_json, "w") as f:
                json.dump({**parsed, "roofline": roofline}, f, indent=2)
            logger.info(f"  Parsed → {metrics_json}")
    else:
        run_ncu_sweep(
            config_keys=targets,
            ncu_bin=ncu_bin,
            output_dir=output_dir,
            repo_dir=repo_dir,
            prof_dir=prof_dir,
            hf_token=hf_token,
            profile_steps=args.profile_steps,
            n_tokens=args.n_tokens,
            force=args.force,
            gpu_specs=selected_specs,
        )
