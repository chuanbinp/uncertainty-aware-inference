"""
plot_ncu_roofline.py
─────────────────────────────────────────────────────────────────────────────
Generates a roofline plot from ncu explicit metrics CSV output.

Works with raw counter metrics collected via:
    ncu -o results/ncu/mistral_fp16_v2 \
        --metrics gpu__time_duration.sum,dram__bytes.sum,\
sm__ops_path_tensor_src_fp16.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
        --launch-skip 30 --launch-count 20 \
        --kernel-name "regex:ampere_fp16_s16816gemm|fmha_cutlassF" \
        --force-overwrite \
        python ncu_fp16.py

    ncu --import results/ncu/mistral_fp16_v2.ncu-rep \
        --csv --print-units base 2>/dev/null \
        > results/ncu/mistral_fp16_v2_metrics.csv

Computes per-kernel:
    FLOPs  = fp16_tensor_ops×2 + ffma×2 + fadd + fmul
    AI     = FLOPs / dram_bytes          (x-axis, FLOPs/byte)
    TFLOPS = FLOPs / duration_s          (y-axis, TFLOPS)

Plots against L4 hardware roofline:
    DRAM bandwidth ceiling:  300 GB/s
    FP16 tensor core peak:   242 TFLOPS
    Ridge point:             242e12 / 300e9 ≈ 807 FLOPs/byte

Usage:
    # Single config:
    python plot_ncu_roofline.py \
        --csv results/ncu/mistral_fp16_v2_metrics.csv \
        --label "Mistral-7B FP16" \
        --out results/roofline_fp16.png

    # Multiple configs overlaid:
    python plot_ncu_roofline.py \
        --csv results/ncu/mistral_fp16_v2_metrics.csv \
              results/ncu/mistral_gptq_int4_metrics.csv \
        --label "FP16" "GPTQ INT4" \
        --out results/roofline_comparison.png
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── L4 Hardware specs ─────────────────────────────────────────────────────────
# Source: NVIDIA L4 datasheet
L4_SPECS = {
    "name":             "NVIDIA L4 (Ada Lovelace, sm_89)",
    "peak_fp16_tflops": 121.0,   # dense FP16/BF16 tensor core (no sparsity)
    "peak_fp32_tflops": 30.3,    # FP32 CUDA core
    "dram_bw_gb_s":     300.0,   # GDDR6
}

# Ridge point: AI where memory roof meets compute roof
RIDGE_POINT = (L4_SPECS["peak_fp16_tflops"] * 1e12) / (L4_SPECS["dram_bw_gb_s"] * 1e9)


# ── Metric name constants (CUPTI counter names) ───────────────────────────────
# These are the exact "Metric Name" strings in the ncu --csv output when
# using explicit --metrics (not --set roofline display names).

FLOP_METRICS = {
    # Ampere tensor core pipe instructions.
    # Each sm__inst_executed_pipe_tensor fires a warp-level MMA instruction.
    # On Ampere (sm_80/sm_86/sm_89) with FP16, the s16816 instruction shape is
    # 16×8×16 = 2048 FLOPs per warp instruction.
    "sm__inst_executed_pipe_tensor.sum":                         2048.0,
    # FP32 / FP16 CUDA core ops (non-tensor-core fallback, e.g. FlashAttention epilogue)
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum":       2.0,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum":       1.0,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum":       1.0,
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum":       2.0,
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum":       1.0,
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum":       1.0,
}

DRAM_BYTES_METRIC  = "dram__bytes.sum"
DURATION_METRIC    = "gpu__time_duration.sum"   # nanoseconds
SM_THROUGHPUT_PCT  = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_THROUGHPUT_PCT = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"

SUM_METRICS = {m for m in FLOP_METRICS} | {DRAM_BYTES_METRIC, DURATION_METRIC}


# ── Parse ncu long-format CSV ─────────────────────────────────────────────────

def parse_ncu_csv(csv_path: Path) -> dict:
    """
    Parse ncu --csv --print-units base output into per-kernel metric dicts.

    ncu CSV is long-format: one row = (kernel, section, metric, value).
    Multiple invocations of the same kernel are aggregated:
      - .sum metrics  → summed across invocations (total work)
      - other metrics → averaged across invocations

    Returns: { kernel_name -> { metric_name -> aggregated_value } }
    """
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel  = row.get("Kernel Name",  "").strip().strip('"')
            metric  = row.get("Metric Name",  "").strip().strip('"')
            val_str = row.get("Metric Value", "").strip().strip('"')
            unit    = row.get("Metric Unit",  "").strip().strip('"')

            if not kernel or not metric or not val_str:
                continue

            try:
                val = float(val_str.replace(",", ""))
            except ValueError:
                continue

            # Normalize units — ncu --print-units base should return SI base
            # but defensively handle common non-base units
            if unit in ("Kbyte", "KB"):    val *= 1e3
            elif unit in ("Mbyte", "MB"):  val *= 1e6
            elif unit in ("Gbyte", "GB"):  val *= 1e9
            elif unit in ("usecond", "us"): val *= 1e3   # → ns
            elif unit in ("msecond", "ms"): val *= 1e6   # → ns
            elif unit in ("second",  "s"):  val *= 1e9   # → ns

            raw[kernel][metric].append(val)

    # Aggregate
    kernels: dict[str, dict[str, float]] = {}
    for kname, metrics in raw.items():
        kernels[kname] = {}
        for mname, vals in metrics.items():
            if mname in SUM_METRICS or ".sum" in mname:
                kernels[kname][mname] = sum(vals)
            else:
                kernels[kname][mname] = sum(vals) / len(vals)

    return kernels


# ── Compute roofline metrics ──────────────────────────────────────────────────

def compute_roofline_points(kernels: dict) -> list[dict]:
    """
    For each kernel compute arithmetic intensity and achieved TFLOPS.
    Returns a list of point dicts sorted by arithmetic intensity.
    """
    points = []

    for kname, m in kernels.items():

        def g(key, default=0.0):
            return float(m.get(key, default))

        # ── FLOPs ──────────────────────────────────────────────────────────
        total_flops = sum(
            g(metric) * mult for metric, mult in FLOP_METRICS.items()
        )

        # ── Memory ─────────────────────────────────────────────────────────
        dram_bytes = g(DRAM_BYTES_METRIC)

        # ── Duration ───────────────────────────────────────────────────────
        duration_ns = g(DURATION_METRIC)
        duration_s  = duration_ns / 1e9 if duration_ns > 0 else None

        # Skip kernels missing essential data
        if dram_bytes == 0 or not duration_s:
            continue

        # ── Roofline coordinates ───────────────────────────────────────────
        ai_dram         = total_flops / dram_bytes
        achieved_tflops = (total_flops / duration_s / 1e12) if duration_s else 0.0

        # Speed-of-Light utilization
        sm_pct   = g(SM_THROUGHPUT_PCT)
        dram_pct = g(DRAM_THROUGHPUT_PCT)

        # Bound classification
        if ai_dram == 0:
            bound = "Unknown"
        elif ai_dram > RIDGE_POINT * 1.05:
            bound = "Compute-bound"
        else:
            bound = "Memory-bound"

        # Short display name
        short_name = kname.split("<")[0].split("(")[0].strip()
        if "s16816gemm" in short_name:
            parts = kname.split("_")
            tile  = next((p for p in parts if "x" in p and p[0].isdigit()), "")
            short_name = f"s16816gemm {tile}" if tile else "s16816gemm"
        elif "fmha_cutlass" in short_name:
            short_name = "FlashAttention"
        elif "gemmk1" in short_name:
            short_name = "gemmk1"
        elif len(short_name) > 40:
            short_name = short_name[:37] + "..."

        points.append({
            "kernel":          kname,
            "short_name":      short_name,
            "flops":           total_flops,
            "dram_bytes":      dram_bytes,
            "duration_us":     duration_ns / 1e3,
            "ai_dram":         ai_dram,
            "achieved_tflops": achieved_tflops,
            "sm_pct":          sm_pct,
            "dram_pct":        dram_pct,
            "bound":           bound,
        })

    return sorted(points, key=lambda x: x["ai_dram"])


# ── Plot ──────────────────────────────────────────────────────────────────────

CONFIG_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
MARKER_STYLES = ["o", "s", "^", "D", "v"]


def draw_roofline_chart(
    all_points: list[list[dict]],
    labels: list[str],
    out_path: Path,
    title: str = "Roofline Analysis — Mistral-7B on NVIDIA L4",
):
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # ── Hardware roofline ─────────────────────────────────────────────────────
    x_min, x_max = 0.01, 2000.0
    ai_range = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    peak_fp16  = L4_SPECS["peak_fp16_tflops"]
    dram_bw    = L4_SPECS["dram_bw_gb_s"] * 1e9   # bytes/s

    # Memory bandwidth roof: achievable TFLOPS = AI (FLOPs/byte) × BW (bytes/s) / 1e12
    memory_roof  = ai_range * dram_bw / 1e12
    compute_roof = np.full_like(ai_range, peak_fp16)
    roofline     = np.minimum(memory_roof, compute_roof)

    ax.plot(ai_range, roofline,
            color="#FFD700", linewidth=2.5, zorder=3,
            label=f"L4 FP16 Roofline  "
                  f"({peak_fp16:.0f} TFLOPS | {L4_SPECS['dram_bw_gb_s']:.0f} GB/s)")

    # Ridge point
    ax.axvline(x=RIDGE_POINT, color="#FFD700", linewidth=1.0,
               linestyle="--", alpha=0.4, zorder=2)
    ax.text(RIDGE_POINT * 1.05, peak_fp16 * 0.88,
            f"Ridge\n{RIDGE_POINT:.0f} FLOPs/B",
            color="#FFD700", fontsize=8, alpha=0.75, va="top")

    # FP32 reference line
    ax.axhline(y=L4_SPECS["peak_fp32_tflops"],
               color="#666", linewidth=1.0, linestyle=":", alpha=0.5, zorder=2)
    ax.text(x_max * 0.6, L4_SPECS["peak_fp32_tflops"] * 1.05,
            f"FP32 peak ({L4_SPECS['peak_fp32_tflops']:.0f} TFLOPS)",
            color="#666", fontsize=7.5, alpha=0.6)

    # ── Kernel points ─────────────────────────────────────────────────────────
    for i, (points, label) in enumerate(zip(all_points, labels)):
        color  = CONFIG_COLORS[i % len(CONFIG_COLORS)]
        marker = MARKER_STYLES[i % len(MARKER_STYLES)]

        if not points:
            print(f"  WARNING: no plottable points for '{label}'")
            continue

        xs = [p["ai_dram"]         for p in points]
        ys = [p["achieved_tflops"] for p in points]

        ax.scatter(xs, ys,
                   color=color, marker=marker, s=140,
                   zorder=5, alpha=0.9,
                   edgecolors="white", linewidths=0.6,
                   label=label)

        for p in points:
            ax.annotate(
                p["short_name"],
                xy=(p["ai_dram"], p["achieved_tflops"]),
                xytext=(7, 4), textcoords="offset points",
                fontsize=7, color=color, alpha=0.9, zorder=6,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.001, peak_fp16 * 2.5)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", color="white", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOPS)",        color="white", fontsize=12)
    ax.set_title(title,                                    color="white", fontsize=13, pad=14)

    ax.tick_params(colors="white", which="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(True, which="major", color="#2a2a2a", linewidth=0.7)
    ax.grid(True, which="minor", color="#1a1a1a", linewidth=0.3)

    ax.text(0.02,  0.001 * 3,  "← Memory-bound",   color="#888", fontsize=8, alpha=0.6)
    ax.text(RIDGE_POINT * 1.1, 0.001 * 3, "Compute-bound →", color="#888", fontsize=8, alpha=0.6)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.25,
              facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ Roofline plot saved → {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(points: list[dict], label: str):
    print(f"\n{'='*90}")
    print(f" Roofline Summary — {label}")
    print(f" L4: {L4_SPECS['peak_fp16_tflops']:.0f} TFLOPS FP16  |  "
          f"{L4_SPECS['dram_bw_gb_s']:.0f} GB/s DRAM  |  "
          f"Ridge = {RIDGE_POINT:.0f} FLOPs/byte")
    print(f"{'='*90}")
    print(f"  {'Kernel':<35} {'AI(F/B)':>9} {'TFLOPS':>8} "
          f"{'SM%':>6} {'Mem%':>6} {'Dur(us)':>9}  Bound")
    print(f"  {'-'*88}")
    for p in points:
        print(
            f"  {p['short_name']:<35} "
            f"{p['ai_dram']:>9.2f} "
            f"{p['achieved_tflops']:>8.3f} "
            f"{p['sm_pct']:>6.1f} "
            f"{p['dram_pct']:>6.1f} "
            f"{p['duration_us']:>9.1f}  "
            f"{p['bound']}"
        )
    if not points:
        print("  (no kernels with sufficient data)")
        print("  Hint: check that dram__bytes.sum and gpu__time_duration.sum")
        print("  are present in the CSV — run with --metrics explicitly.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate roofline plot from ncu explicit metrics CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv",   nargs="+", type=Path, required=True,
                   help="One or more ncu CSV files (one per config)")
    p.add_argument("--label", nargs="+", type=str,  required=True,
                   help="Label for each CSV file (same order as --csv)")
    p.add_argument("--out",   type=Path,
                   default=Path("results/roofline.png"),
                   help="Output PNG path (default: results/roofline.png)")
    p.add_argument("--title", type=str,
                   default="Roofline Analysis — Mistral-7B on NVIDIA L4")
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.csv) != len(args.label):
        print("Error: --csv and --label must have the same number of arguments")
        sys.exit(1)

    all_points = []
    for csv_path, label in zip(args.csv, args.label):
        if not csv_path.exists():
            print(f"Error: {csv_path} not found")
            sys.exit(1)

        print(f"\nParsing {csv_path} ({label})...")
        kernels = parse_ncu_csv(csv_path)
        print(f"  {len(kernels)} unique kernels found")

        points = compute_roofline_points(kernels)
        print(f"  {len(points)} kernels with sufficient data for roofline")

        print_summary(points, label)
        all_points.append(points)

    draw_roofline_chart(
        all_points=all_points,
        labels=args.label,
        out_path=args.out,
        title=args.title,
    )


if __name__ == "__main__":
    main()