"""
nsight_roofline.py
─────────────────────────────────────────────────────────────────────────────
Roofline analysis and plotting from Nsight Compute metrics.

Reads the per-config *_ncu_metrics.json files produced by run_ncu.py and
generates:

  1. Hardware-counter-derived Roofline plot (all configs on one figure)
     - Roofline curves for DRAM, L2, and L1 memory levels
     - Each kernel plotted as (AI_DRAM, achieved_TFLOPS) scatter point
     - Compute roof = A100 peak FP16 (312 TFLOPS)

  2. Comparison plot: ncu AI vs PyTorch Profiler AI
     Highlights where the PyTorch Profiler estimate differs from ground truth.
     (Notable for AWQ and NF4 where PyTorch reports 0 FLOPs for custom kernels)

  3. Speed-of-Light bar chart
     SM% and Memory% utilisation per kernel — the primary bottleneck diagnostic.

  4. Bound classification table printed to stdout and saved as CSV.

  5. Optionally overlays kernel data points from the existing
     profiler_results/profiler_summary.json (PyTorch Profiler estimates)
     for direct comparison.

Usage:
    # Basic — reads ncu_results/ and profiler_results/ from current directory:
    python nsight_roofline.py

    # Custom paths:
    python nsight_roofline.py \
        --ncu-dir       ./ncu_results \
        --prof-dir      ./profiler_results \
        --output-dir    ./roofline_figures \
        --gpu           A100-80GB

    # Only generate the main roofline plot (skip comparison):
    python nsight_roofline.py --no-comparison

    # Include cross-model data (Llama-2 7B from TeamA, 13B from TeamC):
    python nsight_roofline.py \
        --extra-ncu-dirs ./ncu_results_llama7b:./ncu_results_llama13b
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nsight_roofline")


# ─────────────────────────────────────────────────────────────────────────────
# GPU hardware specs
# ─────────────────────────────────────────────────────────────────────────────

GPU_SPECS = {
    "A100-80GB": {
        "peak_fp16_tflops": 312.0,
        "peak_fp32_tflops": 19.5,
        "dram_bw_tb_s":     2.0,
        "l2_bw_tb_s":       12.0,
        "l1_bw_tb_s":       33.0,
        "ridge_dram":       156.0,   # 312 / 2
        "ridge_l2":         26.0,    # 312 / 12
        "ridge_l1":         9.5,     # 312 / 33
    },
    "A100-40GB": {
        "peak_fp16_tflops": 312.0,
        "peak_fp32_tflops": 19.5,
        "dram_bw_tb_s":     1.555,
        "l2_bw_tb_s":       8.0,
        "l1_bw_tb_s":       33.0,
        "ridge_dram":       200.6,
        "ridge_l2":         39.0,
        "ridge_l1":         9.5,
    },
    "L4": {
        "peak_fp16_tflops": 121.0,
        "peak_fp32_tflops": 30.3,
        "dram_bw_tb_s":     0.300,
        "l2_bw_tb_s":       4.0,
        "l1_bw_tb_s":       20.0,
        "ridge_dram":       403.3,
        "ridge_l2":         30.25,
        "ridge_l1":         6.05,
    },
    "V100-SXM2": {
        "peak_fp16_tflops": 125.0,
        "peak_fp32_tflops": 15.7,
        "dram_bw_tb_s":     0.9,
        "l2_bw_tb_s":       4.5,
        "l1_bw_tb_s":       15.0,
        "ridge_dram":       138.9,
        "ridge_l2":         27.8,
        "ridge_l1":         8.3,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Visual design constants
# ─────────────────────────────────────────────────────────────────────────────

QUANT_COLORS = {
    "fp16":  "#2196F3",   # blue
    "gptq":  "#F44336",   # red
    "awq":   "#4CAF50",   # green
    "nf4":   "#FF9800",   # orange
}

FAMILY_MARKERS = {
    "mistral-7b":  "o",
    "llama2-7b":   "s",
    "llama2-13b":  "^",
    "unknown":     "D",
}

BITS_SIZES = {16: 200, 8: 140, 4: 90}

CONFIG_META = {
    # Mistral-7B
    "mistral-7b-fp16":      {"family": "mistral-7b", "quant_type": "fp16",  "bits": 16},
    "mistral-7b-gptq-int8": {"family": "mistral-7b", "quant_type": "gptq",  "bits": 8},
    "mistral-7b-gptq-int4": {"family": "mistral-7b", "quant_type": "gptq",  "bits": 4},
    "mistral-7b-awq-int4":  {"family": "mistral-7b", "quant_type": "awq",   "bits": 4},
    "mistral-7b-nf4":       {"family": "mistral-7b", "quant_type": "nf4",   "bits": 4},
    # Llama-2 7B
    "llama2-7b-fp16":       {"family": "llama2-7b",  "quant_type": "fp16",  "bits": 16},
    "llama2-7b-gptq-int8":  {"family": "llama2-7b",  "quant_type": "gptq",  "bits": 8},
    "llama2-7b-gptq-int4":  {"family": "llama2-7b",  "quant_type": "gptq",  "bits": 4},
    "llama2-7b-awq-int4":   {"family": "llama2-7b",  "quant_type": "awq",   "bits": 4},
    "llama2-7b-nf4":        {"family": "llama2-7b",  "quant_type": "nf4",   "bits": 4},
    # Llama-2 13B
    "llama2-13b-fp16":      {"family": "llama2-13b", "quant_type": "fp16",  "bits": 16},
    "llama2-13b-gptq-int8": {"family": "llama2-13b", "quant_type": "gptq",  "bits": 8},
    "llama2-13b-gptq-int4": {"family": "llama2-13b", "quant_type": "gptq",  "bits": 4},
    "llama2-13b-awq-int4":  {"family": "llama2-13b", "quant_type": "awq",   "bits": 4},
    "llama2-13b-nf4":       {"family": "llama2-13b", "quant_type": "nf4",   "bits": 4},
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ncu_metrics(ncu_dir: Path) -> dict:
    """
    Load all *_ncu_metrics.json files from ncu_dir.
    Returns {config_key: roofline_dict}.
    """
    data = {}
    for json_path in sorted(ncu_dir.glob("*_ncu_metrics.json")):
        config_key = json_path.stem.replace("_ncu_metrics", "")
        try:
            with open(json_path) as f:
                d = json.load(f)
            data[config_key] = d.get("roofline", d)
            logger.info(f"  Loaded ncu metrics: {config_key} ({len(data[config_key].get('kernels',{}))} kernels)")
        except Exception as e:
            logger.warning(f"  Failed to load {json_path}: {e}")
    return data


def load_pytorch_profiler_estimates(prof_dir: Path) -> dict:
    """
    Load PyTorch Profiler arithmetic intensity estimates from
    profiler_results/profiler_summary.json (or per-config JSONs).
    Returns {config_key: {"ai": float, "tok_per_sec": float, "description": str}}.
    """
    summary_path = prof_dir / "profiler_summary.json"
    data = {}

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for key, r in summary.items():
            data[key] = {
                "ai":          r["compute"]["arithmetic_intensity"],
                "tok_per_sec": r["timing"]["tokens_per_second"],
                "peak_gpu_gb": r["memory"]["peak_gpu_gb"],
                "description": r.get("description", key),
            }
        logger.info(f"  Loaded PyTorch Profiler summary: {list(data)}")
    else:
        # Try per-config files
        for json_path in sorted(prof_dir.glob("*_profile.json")):
            config_key = json_path.stem.replace("_profile", "")
            with open(json_path) as f:
                r = json.load(f)
            data[config_key] = {
                "ai":          r["compute"]["arithmetic_intensity"],
                "tok_per_sec": r["timing"]["tokens_per_second"],
                "peak_gpu_gb": r["memory"]["peak_gpu_gb"],
                "description": r.get("description", config_key),
            }

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Hardware-counter Roofline
# ─────────────────────────────────────────────────────────────────────────────

def plot_ncu_roofline(
    ncu_data: dict,
    gpu_name: str = "A100-80GB",
    output_dir: Path = Path("."),
    show_l2: bool = True,
    show_l1: bool = False,
) -> Path:
    """
    Plot the hardware-counter Roofline from ncu metrics.

    X-axis: Arithmetic Intensity (FLOPs / DRAM byte)
    Y-axis: Achieved throughput (TFLOPS)

    Three memory rooflines are drawn (DRAM, L2, optionally L1).
    Each kernel is a scatter point with marker=model family, color=quant method.
    """
    specs = GPU_SPECS.get(gpu_name, GPU_SPECS["A100-80GB"])
    ai_range = np.logspace(-1, 4, 1000)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xscale("log"); ax.set_yscale("log")

    # ── Roofline curves ──────────────────────────────────────────────────────
    # DRAM roof (primary)
    dram_roof = np.minimum(
        specs["peak_fp16_tflops"],
        specs["dram_bw_tb_s"] * ai_range   # TB/s × FLOPs/byte = TFLOPS
    )
    ax.plot(ai_range, dram_roof, "k-", lw=2.5, label=f"{gpu_name} FP16 peak ({specs['peak_fp16_tflops']:.0f} TFLOPS)")
    ax.axvline(specs["ridge_dram"], color="black", linestyle="--", lw=1.0, alpha=0.5)
    ax.text(specs["ridge_dram"] * 1.06, specs["peak_fp16_tflops"] * 0.65,
            f"DRAM ridge\n{specs['ridge_dram']:.0f} FLOPs/B",
            color="black", fontsize=8, alpha=0.7)

    # L2 roof
    if show_l2:
        l2_roof = np.minimum(specs["peak_fp16_tflops"], specs["l2_bw_tb_s"] * ai_range)
        ax.plot(ai_range, l2_roof, color="#607D8B", linestyle="-.", lw=1.8,
                label=f"L2 cache BW ({specs['l2_bw_tb_s']:.0f} TB/s)")
        ax.axvline(specs["ridge_l2"], color="#607D8B", linestyle=":", lw=1.0, alpha=0.5)
        ax.text(specs["ridge_l2"] * 1.06, specs["peak_fp16_tflops"] * 0.85,
                f"L2 ridge {specs['ridge_l2']:.0f}", color="#607D8B", fontsize=7.5, alpha=0.8)

    # L1/SMEM roof
    if show_l1:
        l1_roof = np.minimum(specs["peak_fp16_tflops"], specs["l1_bw_tb_s"] * ai_range)
        ax.plot(ai_range, l1_roof, color="#9E9E9E", linestyle=":", lw=1.5,
                label=f"L1/SMEM BW ({specs['l1_bw_tb_s']:.0f} TB/s)")

    # ── Kernel scatter points ────────────────────────────────────────────────
    plotted_families = set()
    plotted_quants   = set()

    for config_key, rl in ncu_data.items():
        meta = CONFIG_META.get(config_key, {"family": "unknown", "quant_type": "fp16", "bits": 16})
        family    = meta["family"]
        quant     = meta["quant_type"]
        bits      = meta["bits"]
        marker    = FAMILY_MARKERS.get(family, "D")
        color     = QUANT_COLORS.get(quant, "#9C27B0")
        size      = BITS_SIZES.get(bits, 80)

        for kname, r in rl.get("kernels", {}).items():
            ai           = r.get("ai_dram", 0)
            achieved_tf  = r.get("achieved_tflops", 0)
            if ai <= 0 or achieved_tf <= 0:
                continue

            ax.scatter(ai, achieved_tf, marker=marker, s=size, color=color,
                       edgecolors="black", linewidths=0.7, zorder=6, alpha=0.9)

            # Label: short kernel name + config
            short_name = kname.split("::")[-1].split("<")[0][:20]
            ax.annotate(
                f"{short_name}\n({config_key.replace('mistral-7b-','m-').replace('llama2-','l2-')})",
                xy=(ai, achieved_tf),
                xytext=(6, 3), textcoords="offset points",
                fontsize=6.5, alpha=0.8, zorder=7,
            )
            plotted_families.add(family)
            plotted_quants.add(quant)

    # ── Bound region shading ─────────────────────────────────────────────────
    ax.axvspan(0.1, specs["ridge_dram"], alpha=0.03, color="blue",  label="_mem_region")
    ax.axvspan(specs["ridge_dram"], 10000, alpha=0.03, color="green", label="_comp_region")
    ax.text(1.0, specs["peak_fp16_tflops"] * 0.1, "Memory-bound", color="blue",   fontsize=9, alpha=0.6)
    ax.text(specs["ridge_dram"] * 2, specs["peak_fp16_tflops"] * 0.1, "Compute-bound", color="green", fontsize=9, alpha=0.6)

    # ── Legend ───────────────────────────────────────────────────────────────
    fam_handles = [
        Line2D([0],[0], marker=FAMILY_MARKERS.get(f,"D"), color="w",
               markerfacecolor="gray", markeredgecolor="k", markersize=9, label=f)
        for f in sorted(plotted_families)
    ]
    quant_handles = [
        mpatches.Patch(color=QUANT_COLORS.get(q,"gray"), label=q)
        for q in sorted(plotted_quants)
    ]
    bits_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="k", markersize=np.sqrt(s) * 0.7, label=f"{b}-bit")
        for b, s in BITS_SIZES.items()
    ]

    leg1 = ax.legend(handles=fam_handles,   title="Model family",   loc="upper left",   fontsize=8, framealpha=0.85)
    leg2 = ax.legend(handles=quant_handles, title="Quant method",   loc="lower right",  fontsize=8, framealpha=0.85)
    leg3 = ax.legend(handles=bits_handles,  title="Bit-width",      loc="upper right",  fontsize=8, framealpha=0.85)
    ax.add_artist(leg1); ax.add_artist(leg2)

    roofline_handles = [
        Line2D([0],[0], color="black",   linestyle="-",  lw=2, label=f"DRAM roof ({specs['peak_fp16_tflops']:.0f} TFLOPS @ {specs['dram_bw_tb_s']:.0f} TB/s)"),
        Line2D([0],[0], color="#607D8B", linestyle="-.", lw=2, label=f"L2 roof ({specs['l2_bw_tb_s']:.0f} TB/s)"),
    ]
    ax.legend(handles=roofline_handles, loc="center right", fontsize=8, framealpha=0.85)
    ax.add_artist(leg3)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / DRAM byte)", fontsize=12)
    ax.set_ylabel("Achieved Throughput (TFLOPS)",              fontsize=12)
    ax.set_title(
        f"Hardware-Counter Roofline (Nsight Compute) — {gpu_name}\n"
        f"Points = dominant kernels per PTQ config",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.1, 5000)
    ax.set_ylim(0.001, specs["peak_fp16_tflops"] * 3)

    plt.tight_layout()
    output_path = output_dir / f"roofline_ncu_{gpu_name.replace('-','_')}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Roofline plot → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: ncu AI vs PyTorch Profiler AI comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_ai_comparison(
    ncu_data: dict,
    pt_data: dict,
    output_dir: Path = Path("."),
) -> Path:
    """
    Scatter plot comparing ncu hardware-counter AI against PyTorch Profiler
    estimated AI for each config.

    X-axis: PyTorch Profiler AI (FLOPs from operator metadata / peak VRAM bytes)
    Y-axis: ncu AI (actual FLOPs from hardware counters / actual DRAM bytes)

    Points on the diagonal y=x mean both methods agree.
    Points ABOVE the diagonal = PyTorch Profiler *underestimates* true AI
    (common for quantized kernels where PyTorch FLOPs counter returns 0).
    Points BELOW = PyTorch Profiler overestimates.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    max_val = 0.1
    annotations = []

    for config_key, rl in ncu_data.items():
        pt = pt_data.get(config_key, {})
        pt_ai = pt.get("ai", 0)

        meta   = CONFIG_META.get(config_key, {"family": "unknown", "quant_type": "fp16", "bits": 16})
        color  = QUANT_COLORS.get(meta["quant_type"], "gray")
        marker = FAMILY_MARKERS.get(meta["family"], "D")

        for kname, r in rl.get("kernels", {}).items():
            ncu_ai = r.get("ai_dram", 0)
            if ncu_ai <= 0:
                continue
            ax.scatter(pt_ai, ncu_ai, marker=marker, s=120, color=color,
                       edgecolors="black", linewidths=0.7, zorder=5, alpha=0.9)
            annotations.append((pt_ai, ncu_ai, config_key.replace("mistral-7b-","m-").replace("llama2-","l2-")))
            max_val = max(max_val, pt_ai, ncu_ai)

    # Diagonal y=x (perfect agreement)
    diag = np.linspace(0, max_val * 1.2, 100)
    ax.plot(diag, diag, "k--", lw=1.5, alpha=0.5, label="y = x (perfect agreement)")

    for pt_ai, ncu_ai, label in annotations:
        ax.annotate(label, xy=(pt_ai, ncu_ai), xytext=(5, 3),
                    textcoords="offset points", fontsize=8)

    # Legend
    quant_handles = [mpatches.Patch(color=c, label=q) for q, c in QUANT_COLORS.items()]
    ax.legend(handles=quant_handles, title="Quant method", fontsize=9)

    ax.set_xlabel("PyTorch Profiler AI (FLOPs from op metadata / peak VRAM bytes)", fontsize=10)
    ax.set_ylabel("ncu AI (hardware FLOPs / actual DRAM bytes)",                    fontsize=10)
    ax.set_title(
        "Arithmetic Intensity: PyTorch Profiler Estimate vs ncu Ground Truth\n"
        "Points above y=x → PyTorch underestimates AI (common for custom quant kernels)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path = output_dir / "ai_comparison_ncu_vs_pytorch.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"AI comparison plot → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Speed-of-Light bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_of_light(
    ncu_data: dict,
    output_dir: Path = Path("."),
) -> Path:
    """
    Grouped bar chart showing SM utilisation% and DRAM utilisation% for each
    config's dominant kernel.  This is the primary bottleneck diagnostic.

    Kernels close to 100% SM are compute-bound.
    Kernels close to 100% Memory are memory-bound.
    Kernels far from both are latency-bound (kernel launch overhead, etc.).
    """
    labels, sm_pcts, mem_pcts, colors = [], [], [], []

    for config_key, rl in ncu_data.items():
        meta  = CONFIG_META.get(config_key, {"quant_type": "fp16", "bits": 16})
        color = QUANT_COLORS.get(meta["quant_type"], "gray")

        # Use the kernel with the highest achieved TFLOPS as representative
        kernels = rl.get("kernels", {})
        if not kernels:
            continue
        rep_kernel = max(kernels.items(), key=lambda x: x[1].get("achieved_tflops", 0))
        kname, r   = rep_kernel

        short_config = config_key.replace("mistral-7b-","").replace("llama2-7b-","l7-").replace("llama2-13b-","l13-")
        short_kern   = kname.split("::")[-1].split("<")[0][:20]
        labels.append(f"{short_config}\n({short_kern})")
        sm_pcts.append(r.get("sm_pct", 0))
        mem_pcts.append(r.get("mem_pct", 0))
        colors.append(color)

    if not labels:
        logger.warning("No kernels to plot for Speed-of-Light chart")
        return None

    x  = np.arange(len(labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 6))

    bars_sm  = ax.bar(x - w/2, sm_pcts,  w, label="SM utilisation [%]",   color=[c + "BB" for c in colors],
                      edgecolor="black", linewidth=0.5)
    bars_mem = ax.bar(x + w/2, mem_pcts, w, label="DRAM utilisation [%]", color=colors,
                      edgecolor="black", linewidth=0.5, hatch="//")

    # 100% guide line — approaching this means the kernel is saturating that resource
    ax.axhline(100, color="red", linestyle="--", lw=1.2, alpha=0.7, label="100% peak")
    ax.axhline(75,  color="orange", linestyle=":", lw=1.0, alpha=0.5,  label="75% (high utilisation)")

    # Value labels on bars
    for bar in list(bars_sm) + list(bars_mem):
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Utilisation (% of peak)", fontsize=11)
    ax.set_title(
        "Speed-of-Light Analysis (Nsight Compute) — SM vs DRAM Utilisation\n"
        "Representative kernel per config (highest achieved TFLOPS)",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 120)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Quant-colour legend
    quant_handles = [mpatches.Patch(color=c, label=q) for q, c in QUANT_COLORS.items()]
    ax.legend(handles=quant_handles, title="Quant", loc="upper right", fontsize=8,
              bbox_to_anchor=(1.12, 1))

    plt.tight_layout()
    output_path = output_dir / "speed_of_light_ncu.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Speed-of-Light plot → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Bound classification table
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_bound_table(ncu_data: dict, output_dir: Path) -> None:
    """Print bound classification for all kernels and save as CSV."""
    import csv

    rows = []
    for config_key, rl in ncu_data.items():
        for kname, r in rl.get("kernels", {}).items():
            rows.append({
                "config_key":       config_key,
                "kernel":           kname[:60],
                "AI_DRAM":          round(r.get("ai_dram", 0),     2),
                "AI_L2":            round(r.get("ai_l2", 0),       2),
                "achieved_TFLOPS":  round(r.get("achieved_tflops", 0), 4),
                "SM_pct":           round(r.get("sm_pct", 0),      1),
                "DRAM_pct":         round(r.get("mem_pct", 0),     1),
                "occupancy_pct":    round(r.get("occupancy_pct", 0), 1),
                "bound":            r.get("bound", "Unknown"),
            })

    if not rows:
        logger.warning("No roofline data to tabulate")
        return

    # Print table
    print(f"\n{'='*100}")
    print(f" Nsight Compute — Kernel Bound Classification")
    print(f"{'='*100}")
    print(f"  {'Config':<28} {'Kernel':<35} {'AI(DRAM)':>9} {'AI(L2)':>7} "
          f"{'TFLOPS':>7} {'SM%':>5} {'Mem%':>6}  Bound")
    print(f"  {'-'*98}")
    for r in rows:
        bound_marker = "🟢" if "Compute" in r["bound"] else ("🔴" if "DRAM" in r["bound"] else "🟡")
        print(f"  {r['config_key']:<28} {r['kernel'][:35]:<35} "
              f"{r['AI_DRAM']:>9.1f} {r['AI_L2']:>7.1f} "
              f"{r['achieved_TFLOPS']:>7.3f} {r['SM_pct']:>5.0f} {r['DRAM_pct']:>6.0f}  "
              f"{bound_marker} {r['bound']}")
    print()

    # Save CSV
    csv_path = output_dir / "ncu_bound_classification.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Bound classification CSV → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Nsight Compute Roofline analysis")
    p.add_argument("--ncu-dir",       type=str, default="./ncu_results",
                   help="Directory with *_ncu_metrics.json files from run_ncu.py")
    p.add_argument("--prof-dir",      type=str, default="./profiler_results",
                   help="Directory with *_profile.json files from run_profiler.py (for comparison)")
    p.add_argument("--output-dir",    type=str, default="./roofline_figures")
    p.add_argument("--gpu",           type=str, default="A100-80GB",
                   choices=list(GPU_SPECS.keys()),
                   help="GPU spec for roofline curves")
    p.add_argument("--extra-ncu-dirs", type=str, default=None,
                   help="Colon-separated additional ncu_results dirs (e.g. for Llama-2 7B/13B)")
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip ncu vs PyTorch Profiler comparison plot")
    p.add_argument("--show-l1",       action="store_true",
                   help="Include L1/SMEM roofline curve (off by default — very optimistic)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ncu_dir    = Path(args.ncu_dir)
    prof_dir   = Path(args.prof_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ncu metrics ─────────────────────────────────────────────────────
    ncu_data = load_ncu_metrics(ncu_dir)
    if args.extra_ncu_dirs:
        for extra_dir in args.extra_ncu_dirs.split(":"):
            extra_data = load_ncu_metrics(Path(extra_dir))
            ncu_data.update(extra_data)
            logger.info(f"  Merged {len(extra_data)} configs from {extra_dir}")

    if not ncu_data:
        logger.error(
            f"No ncu metrics found in {ncu_dir}. "
            f"Run `sudo python run_ncu.py --all` first."
        )
        import sys; sys.exit(1)

    logger.info(f"\nLoaded ncu metrics for: {list(ncu_data)}")

    # ── Load PyTorch Profiler estimates (for comparison) ─────────────────────
    pt_data = {}
    if not args.no_comparison and prof_dir.exists():
        pt_data = load_pytorch_profiler_estimates(prof_dir)

    # ── Generate figures ──────────────────────────────────────────────────────
    roofline_path = plot_ncu_roofline(
        ncu_data=ncu_data,
        gpu_name=args.gpu,
        output_dir=output_dir,
        show_l1=args.show_l1,
    )

    sol_path = plot_speed_of_light(ncu_data, output_dir)

    if not args.no_comparison and pt_data:
        comparison_path = plot_ai_comparison(ncu_data, pt_data, output_dir)

    print_and_save_bound_table(ncu_data, output_dir)

    print(f"\n✓ All Roofline figures saved to {output_dir}/")
    print(f"  Main Roofline       : {roofline_path.name}")
    if sol_path:
        print(f"  Speed-of-Light bars : {sol_path.name}")
    if not args.no_comparison and pt_data:
        print(f"  AI comparison       : ai_comparison_ncu_vs_pytorch.png")
    print(f"  Bound classification: ncu_bound_classification.csv")
