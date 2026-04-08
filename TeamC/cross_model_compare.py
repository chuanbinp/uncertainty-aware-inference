"""
Cross-model and cross-config comparison of calibration metrics.

Loads JSON result summaries from all teams and produces comparison tables,
overlay reliability diagrams, and entropy distribution comparisons.

Usage:
    python TeamC/cross_model_compare.py
    python TeamC/cross_model_compare.py --dataset arc_challenge
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIRS = [
    os.path.join(_PROJECT_ROOT, "TeamA", "results"),
    os.path.join(_PROJECT_ROOT, "TeamB", "results"),
    os.path.join(_PROJECT_ROOT, "TeamC", "results"),
]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_all_summaries() -> pd.DataFrame:
    """Load all JSON summary files into a DataFrame."""
    rows = []
    for d in RESULTS_DIRS:
        for path in glob.glob(os.path.join(d, "*.json")):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_raw_tensors(model: str, quant_method: str, precision: str, dataset: str) -> dict | None:
    """Load .pt tensor file for a specific model+config+dataset combo."""
    pattern = f"{model}_{quant_method}_{precision}_{dataset}.pt"
    for d in RESULTS_DIRS:
        path = os.path.join(d, pattern)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
    return None


def get_unique_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Get unique (model, quant_method, precision) combinations."""
    return df[["model", "quant_method", "precision"]].drop_duplicates()


def config_label(row) -> str:
    """Short human-readable label for a config."""
    return f"{row['model']} {row['quant_method']} {row['precision']}"


def print_comparison_table(df: pd.DataFrame, dataset: str | None = None):
    """Print formatted comparison table of calibration metrics."""
    subset = df if dataset is None else df[df["dataset"] == dataset]
    if subset.empty:
        print("No results found.")
        return
    cols = ["model", "quant_method", "precision", "dataset", "accuracy",
            "ECE", "MCE", "Brier_Score", "Avg_Entropy", "tokens_per_second"]
    available = [c for c in cols if c in subset.columns]
    print(subset[available].to_string(index=False))


def plot_reliability_overlay(dataset: str, configs: pd.DataFrame):
    """Overlay reliability diagrams for multiple configs on same axes."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    for i, (_, row) in enumerate(configs.iterrows()):
        label = config_label(row)
        data = load_raw_tensors(row["model"], row["quant_method"], row["precision"], dataset)
        if data is None:
            print(f"  Warning: no data for {label} / {dataset}")
            continue

        confs = data["confidences"].numpy()
        accs = data["accuracies"].numpy()
        bins = np.linspace(0, 1, 16)
        bin_idx = np.digitize(confs, bins) - 1

        bin_accs = np.zeros(15)
        bin_confs = np.zeros(15)
        bin_counts = np.zeros(15)
        for b in range(15):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_accs[b] = accs[mask].mean()
                bin_confs[b] = confs[mask].mean()
                bin_counts[b] = mask.sum()

        nonzero = bin_counts > 0
        ax.plot(bin_confs[nonzero], bin_accs[nonzero], "o-",
                color=colors[i % len(colors)], label=label)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Comparison: {dataset}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"cross_config_reliability_{dataset}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_entropy_comparison(dataset: str, configs: pd.DataFrame):
    """Side-by-side entropy distributions (correct vs incorrect) per config."""
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, configs.iterrows()):
        label = config_label(row)
        data = load_raw_tensors(row["model"], row["quant_method"], row["precision"], dataset)
        if data is None:
            ax.set_title(f"{label}\n(no data)")
            continue
        e = data["entropies"].numpy()
        a = data["accuracies"].numpy()
        ax.hist(e[a == 1], bins=20, alpha=0.6, color="green", label="Correct", density=True)
        ax.hist(e[a == 0], bins=20, alpha=0.6, color="red", label="Incorrect", density=True)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Entropy")
        ax.legend(fontsize=7)

    fig.suptitle(f"Entropy Distributions: {dataset}")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"cross_config_entropy_{dataset}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cross-model calibration comparison")
    parser.add_argument("--dataset", default=None, help="Filter to one dataset")
    args = parser.parse_args()

    df = load_all_summaries()
    if df.empty:
        print("No results found. Run eval scripts first.")
        return

    print("\n=== Calibration Comparison ===\n")
    print_comparison_table(df, args.dataset)

    configs = get_unique_configs(df)
    datasets = [args.dataset] if args.dataset else df["dataset"].unique().tolist()

    for ds in datasets:
        print(f"\nPlotting: {ds}")
        plot_reliability_overlay(ds, configs)
        plot_entropy_comparison(ds, configs)

    print("\nDone.")


if __name__ == "__main__":
    main()
