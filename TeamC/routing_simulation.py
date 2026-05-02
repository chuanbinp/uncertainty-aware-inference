"""
Uncertainty-based routing simulation.

Simulates a two-tier serving strategy: if the quantized model's confidence
exceeds a threshold, serve from it (cheap); otherwise escalate to FP16 (expensive).
Sweeps the confidence threshold and computes effective accuracy, calibration, and cost.

Usage:
    python TeamC/routing_simulation.py
    python TeamC/routing_simulation.py --dataset arc_challenge --num_thresholds 200
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIRS = [
    os.path.join(_PROJECT_ROOT, "TeamA", "results"),
    os.path.join(_PROJECT_ROOT, "TeamB", "calibration_results"),
    os.path.join(_PROJECT_ROOT, "TeamC", "full_results"),
]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "routing")


# -- Data loading --------------------------------------------------------------

def load_all_summaries(results_dirs: list[str]) -> pd.DataFrame:
    """Load all JSON summary files."""
    rows = []
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            with open(path) as f:
                rows.append(json.load(f))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_tensors(model: str, quant_method: str, precision: str,
                 dataset: str, results_dirs: list[str]) -> dict | None:
    """Load .pt file for a specific config+dataset."""
    fname = f"{model}_{quant_method}_{precision}_{dataset}.pt"
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "**", fname), recursive=True):
            return torch.load(path, map_location="cpu", weights_only=True)
    return None


# -- ECE computation -----------------------------------------------------------

def compute_ece(confidences: np.ndarray, accuracies: np.ndarray,
                num_bins: int = 15) -> float:
    """Compute Expected Calibration Error (numpy version for routing sim)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.sum() / n
        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += abs(avg_conf - avg_acc) * prop_in_bin

    return ece


# -- Core routing simulation ---------------------------------------------------

def simulate_routing(
    quant_confidences: np.ndarray,
    quant_accuracies: np.ndarray,
    fp16_accuracies: np.ndarray,
    fp16_confidences: np.ndarray,
    quant_tps: float,
    fp16_tps: float,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Simulate routing for one quantized config vs one FP16 baseline.

    For each threshold t:
    - If quant_confidence[i] >= t: serve from quantized model
    - Else: escalate to FP16

    Returns DataFrame with columns:
        threshold, frac_quantized, effective_accuracy, effective_ece,
        effective_tps, cost_savings_pct, fp16_accuracy
    """
    n = len(quant_confidences)
    results = []

    for t in thresholds:
        use_quant = quant_confidences >= t
        n_quant = use_quant.sum()
        frac_quant = n_quant / n

        effective_acc = np.where(use_quant, quant_accuracies, fp16_accuracies)
        effective_accuracy = effective_acc.mean()

        effective_conf = np.where(use_quant, quant_confidences, fp16_confidences)
        effective_ece = compute_ece(effective_conf, effective_acc)

        if frac_quant == 1.0:
            eff_tps = quant_tps
        elif frac_quant == 0.0:
            eff_tps = fp16_tps
        else:
            avg_time = frac_quant / quant_tps + (1 - frac_quant) / fp16_tps
            eff_tps = 1.0 / avg_time if avg_time > 0 else 0.0

        fp16_cost = 1.0 / fp16_tps if fp16_tps > 0 else float("inf")
        mixed_cost = 1.0 / eff_tps if eff_tps > 0 else float("inf")

        results.append({
            "threshold": t,
            "frac_quantized": frac_quant,
            "frac_escalated": 1.0 - frac_quant,
            "effective_accuracy": effective_accuracy,
            "effective_ece": effective_ece,
            "effective_tps": eff_tps,
            "cost_savings_pct": (1.0 - mixed_cost / fp16_cost) * 100 if fp16_cost > 0 else 0.0,
            "fp16_accuracy": fp16_accuracies.mean(),
        })

    return pd.DataFrame(results)


# -- Run all config pairs ------------------------------------------------------

def find_fp16_baseline(summaries: pd.DataFrame, model: str,
                       dataset: str) -> dict | None:
    """Find the FP16 baseline row for a given model and dataset."""
    mask = ((summaries["model"] == model) &
            (summaries["quant_method"].isin(["fp16", "16bit"])) &
            (summaries["dataset"] == dataset))
    matches = summaries[mask]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def simulate_all_pairs(summaries: pd.DataFrame, results_dirs: list[str],
                       thresholds: np.ndarray,
                       dataset_filter: str | None = None) -> dict:
    """Run routing simulation for all quantized configs vs their FP16 baselines.

    Returns dict keyed by (model, quant_method, precision, dataset) -> DataFrame.
    """
    all_results = {}
    quantized = summaries[~summaries["quant_method"].isin(["fp16", "16bit"])]

    if dataset_filter:
        quantized = quantized[quantized["dataset"] == dataset_filter]

    for _, row in quantized.iterrows():
        model = row["model"]
        qmethod = row["quant_method"]
        precision = row["precision"]
        dataset = row["dataset"]

        fp16_info = find_fp16_baseline(summaries, model, dataset)
        if fp16_info is None:
            print(f"  Warning: no FP16 baseline for {model}/{dataset}, skipping")
            continue

        quant_tensors = load_tensors(model, qmethod, precision, dataset,
                                     results_dirs)
        fp16_tensors = load_tensors(model, fp16_info["quant_method"],
                                    fp16_info["precision"], dataset,
                                    results_dirs)

        if quant_tensors is None or fp16_tensors is None:
            print(f"  Warning: missing .pt for {model} {qmethod} {precision} "
                  f"/ {dataset}, skipping")
            continue

        n_quant = len(quant_tensors["confidences"])
        n_fp16 = len(fp16_tensors["confidences"])
        if n_quant != n_fp16:
            print(f"  Warning: size mismatch {n_quant} vs {n_fp16} for "
                  f"{model}/{dataset}, skipping")
            continue

        key = (model, qmethod, precision, dataset)
        result_df = simulate_routing(
            quant_confidences=quant_tensors["confidences"].numpy(),
            quant_accuracies=quant_tensors["accuracies"].numpy(),
            fp16_accuracies=fp16_tensors["accuracies"].numpy(),
            fp16_confidences=fp16_tensors["confidences"].numpy(),
            quant_tps=row["tokens_per_second"],
            fp16_tps=fp16_info["tokens_per_second"],
            thresholds=thresholds,
        )
        result_df["model"] = model
        result_df["quant_method"] = qmethod
        result_df["precision"] = precision
        result_df["dataset"] = dataset
        all_results[key] = result_df

    return all_results


# -- Optimal threshold finding -------------------------------------------------

def find_optimal_thresholds(all_results: dict) -> pd.DataFrame:
    """For each config, find thresholds that optimize different criteria."""
    rows = []
    for key, df in all_results.items():
        model, qmethod, precision, dataset = key
        fp16_acc = df["fp16_accuracy"].iloc[0]

        # Strategy 1: max accuracy
        best_acc_row = df.loc[df["effective_accuracy"].idxmax()]

        # Strategy 2: max savings while accuracy >= FP16
        above_fp16 = df[df["effective_accuracy"] >= fp16_acc - 0.001]
        if not above_fp16.empty:
            best_savings_row = above_fp16.loc[above_fp16["cost_savings_pct"].idxmax()]
        else:
            best_savings_row = best_acc_row

        # Strategy 3: balanced (maximize accuracy * normalized savings)
        df_copy = df.copy()
        df_copy["score"] = (df_copy["effective_accuracy"] *
                            (df_copy["cost_savings_pct"].clip(lower=0) / 100))
        balanced_row = df_copy.loc[df_copy["score"].idxmax()]

        rows.append({
            "model": model,
            "quant_method": qmethod,
            "precision": precision,
            "dataset": dataset,
            "fp16_accuracy": fp16_acc,
            "best_acc_threshold": best_acc_row["threshold"],
            "best_acc_accuracy": best_acc_row["effective_accuracy"],
            "best_acc_savings": best_acc_row["cost_savings_pct"],
            "best_acc_frac_quant": best_acc_row["frac_quantized"],
            "max_savings_threshold": best_savings_row["threshold"],
            "max_savings_accuracy": best_savings_row["effective_accuracy"],
            "max_savings_pct": best_savings_row["cost_savings_pct"],
            "max_savings_frac_quant": best_savings_row["frac_quantized"],
            "balanced_threshold": balanced_row["threshold"],
            "balanced_accuracy": balanced_row["effective_accuracy"],
            "balanced_savings": balanced_row["cost_savings_pct"],
        })

    return pd.DataFrame(rows)


# -- Visualization -------------------------------------------------------------

def plot_threshold_curves(result_df: pd.DataFrame, config_label: str,
                          output_dir: str) -> str:
    """Plot threshold vs accuracy, ECE, frac_quantized, and cost savings."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    t = result_df["threshold"]

    ax = axes[0, 0]
    ax.plot(t, result_df["effective_accuracy"], "b-", linewidth=2,
            label="Mixed system")
    ax.axhline(y=result_df["fp16_accuracy"].iloc[0], color="gray",
               linestyle="--", label="FP16 baseline")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Effective Accuracy")
    ax.set_title("Accuracy vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, result_df["effective_ece"], "r-", linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Effective ECE")
    ax.set_title("Calibration Error vs Threshold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, result_df["frac_quantized"], "g-", linewidth=2)
    ax.fill_between(t, 0, result_df["frac_quantized"], alpha=0.2, color="green")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Fraction Served by Quantized")
    ax.set_title("Routing Distribution")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, result_df["cost_savings_pct"], "m-", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Cost Savings vs FP16 (%)")
    ax.set_title("Projected Cost Savings")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Routing Simulation: {config_label}", fontsize=14)
    plt.tight_layout()

    safe_name = config_label.replace(" ", "_").replace("/", "_")
    path = os.path.join(output_dir, f"routing_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_routing_summary(all_results: dict, output_dir: str) -> str:
    """Overlay accuracy-vs-cost_savings curves for all configs."""
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10.colors

    for i, (key, df) in enumerate(all_results.items()):
        model, qmethod, precision, dataset = key
        label = f"{model} {qmethod} {precision} ({dataset})"
        color = colors[i % len(colors)]
        ax.plot(df["cost_savings_pct"], df["effective_accuracy"], "-",
                color=color, linewidth=1.5, label=label, alpha=0.8)

        max_savings = df["cost_savings_pct"].max()
        sweet = df[df["cost_savings_pct"] >= max_savings * 0.5]
        if not sweet.empty:
            best = sweet.loc[sweet["effective_accuracy"].idxmax()]
            ax.scatter(best["cost_savings_pct"], best["effective_accuracy"],
                       s=80, color=color, edgecolors="black", zorder=5)

    ax.set_xlabel("Cost Savings vs FP16 (%)")
    ax.set_ylabel("Effective Accuracy")
    ax.set_title("Routing: Accuracy-Cost Tradeoff Across Configs")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "routing_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Routing simulation: quant + FP16 fallback")
    parser.add_argument("--dataset", default=None,
                        help="Filter to one dataset")
    parser.add_argument("--num_thresholds", type=int, default=101,
                        help="Number of threshold steps between 0 and 1")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    results_dirs = [d for d in RESULTS_DIRS if os.path.isdir(d)]
    if not results_dirs:
        print("ERROR: No results directories found. Run evaluation scripts first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    thresholds = np.linspace(0.0, 1.0, args.num_thresholds)

    summaries = load_all_summaries(results_dirs)
    if summaries.empty:
        print("ERROR: No JSON result files found. Run evaluation scripts first.")
        sys.exit(1)

    print(f"Loaded {len(summaries)} result summaries")
    print(f"Models: {summaries['model'].unique()}")
    print(f"Quant methods: {summaries['quant_method'].unique()}")
    print(f"Datasets: {summaries['dataset'].unique()}")

    all_results = simulate_all_pairs(summaries, results_dirs, thresholds,
                                     args.dataset)

    if not all_results:
        print("ERROR: No valid config pairs found for simulation.")
        print("Need both quantized and FP16 results for the same model/dataset.")
        sys.exit(1)

    print(f"\nSimulated {len(all_results)} config pairs")

    for key, df in all_results.items():
        model, qmethod, precision, dataset = key
        label = f"{model} {qmethod} {precision} {dataset}"

        csv_name = f"routing_{model}_{qmethod}_{precision}_{dataset}.csv"
        df.to_csv(os.path.join(args.output_dir, csv_name), index=False)

        plot_path = plot_threshold_curves(df, label, args.output_dir)
        print(f"  Saved: {plot_path}")

    plot_routing_summary(all_results, args.output_dir)
    print(f"  Saved: {os.path.join(args.output_dir, 'routing_summary.png')}")

    optimal = find_optimal_thresholds(all_results)
    optimal_path = os.path.join(args.output_dir, "optimal_thresholds.csv")
    optimal.to_csv(optimal_path, index=False)
    print(f"\n=== Optimal Thresholds ===\n")
    print(optimal.to_string(index=False, float_format="%.4f"))
    print(f"\nAll routing results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
