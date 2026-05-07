# """
# Entropy and confidence analysis across quantization configurations.

# Quantifies how quantization affects entropy distributions, confidence
# calibration, and the separation between correct and incorrect predictions.
# Computes shifts relative to FP16 baseline.

# Usage:
#     python TeamC/entropy_analysis.py
#     python TeamC/entropy_analysis.py --include_all_teams
# """

# import argparse
# import glob
# import json
# import os
# import sys

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import torch

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# _PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
# RESULTS_DIRS_TEAM_C = [os.path.join(os.path.dirname(__file__), "results")]
# RESULTS_DIRS_ALL = [
#     os.path.join(_PROJECT_ROOT, "TeamA", "results"),
#     os.path.join(_PROJECT_ROOT, "TeamB", "results"),
#     os.path.join(_PROJECT_ROOT, "TeamC", "results"),
# ]
# ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "results", "analysis")


# # ── Data loading ──────────────────────────────────────────────────────

# def load_all_summaries(results_dirs: list[str]) -> pd.DataFrame:
#     rows = []
#     for d in results_dirs:
#         for path in glob.glob(os.path.join(d, "*.json")):
#             with open(path) as f:
#                 rows.append(json.load(f))
#     return pd.DataFrame(rows) if rows else pd.DataFrame()


# def load_tensors(model: str, quant_method: str, precision: str, dataset: str,
#                  results_dirs: list[str]) -> dict | None:
#     pattern = f"{model}_{quant_method}_{precision}_{dataset}.pt"
#     for d in results_dirs:
#         path = os.path.join(d, pattern)
#         if os.path.exists(path):
#             return torch.load(path, map_location="cpu", weights_only=True)
#     return None


# # ── Per-config analysis ───────────────────────────────────────────────

# def compute_entropy_stats(tensors: dict) -> dict:
#     """Compute entropy and confidence statistics for correct vs incorrect."""
#     confs = tensors["confidences"].numpy()
#     accs = tensors["accuracies"].numpy()
#     ents = tensors["entropies"].numpy()

#     correct = accs == 1
#     incorrect = accs == 0

#     stats = {
#         "n_total": len(accs),
#         "n_correct": int(correct.sum()),
#         "n_incorrect": int(incorrect.sum()),
#         "accuracy": float(accs.mean()),
#     }

#     if correct.sum() > 0:
#         stats["entropy_correct_mean"] = float(ents[correct].mean())
#         stats["entropy_correct_std"] = float(ents[correct].std())
#         stats["confidence_correct_mean"] = float(confs[correct].mean())
#     else:
#         stats["entropy_correct_mean"] = float("nan")
#         stats["entropy_correct_std"] = float("nan")
#         stats["confidence_correct_mean"] = float("nan")

#     if incorrect.sum() > 0:
#         stats["entropy_incorrect_mean"] = float(ents[incorrect].mean())
#         stats["entropy_incorrect_std"] = float(ents[incorrect].std())
#         stats["confidence_incorrect_mean"] = float(confs[incorrect].mean())
#     else:
#         stats["entropy_incorrect_mean"] = float("nan")
#         stats["entropy_incorrect_std"] = float("nan")
#         stats["confidence_incorrect_mean"] = float("nan")

#     # Gaps: larger = better separation between correct and incorrect
#     stats["entropy_gap"] = stats["entropy_incorrect_mean"] - stats["entropy_correct_mean"]
#     stats["confidence_gap"] = stats["confidence_correct_mean"] - stats["confidence_incorrect_mean"]

#     return stats


# def build_analysis_table(df: pd.DataFrame, results_dirs: list[str]) -> pd.DataFrame:
#     """Build full analysis table: one row per (model, quant_method, precision, dataset)."""
#     rows = []
#     for _, summary in df.iterrows():
#         tensors = load_tensors(
#             summary["model"], summary["quant_method"],
#             summary["precision"], summary["dataset"],
#             results_dirs,
#         )
#         if tensors is None:
#             continue

#         stats = compute_entropy_stats(tensors)
#         rows.append({
#             "model": summary["model"],
#             "quant_method": summary["quant_method"],
#             "precision": summary["precision"],
#             "dataset": summary["dataset"],
#             "ECE": summary.get("ECE", float("nan")),
#             "MCE": summary.get("MCE", float("nan")),
#             "tokens_per_second": summary.get("tokens_per_second", float("nan")),
#             **stats,
#         })

#     return pd.DataFrame(rows)


# def add_fp16_deltas(analysis: pd.DataFrame) -> pd.DataFrame:
#     """Add columns for shifts relative to FP16 baseline per (model, dataset)."""
#     analysis = analysis.copy()
#     analysis["delta_ECE"] = float("nan")
#     analysis["delta_entropy_gap"] = float("nan")
#     analysis["delta_confidence_gap"] = float("nan")

#     fp16 = analysis[analysis["quant_method"] == "fp16"]

#     for _, fp16_row in fp16.iterrows():
#         mask = (
#             (analysis["model"] == fp16_row["model"]) &
#             (analysis["dataset"] == fp16_row["dataset"])
#         )
#         analysis.loc[mask, "delta_ECE"] = analysis.loc[mask, "ECE"] - fp16_row["ECE"]
#         analysis.loc[mask, "delta_entropy_gap"] = (
#             analysis.loc[mask, "entropy_gap"] - fp16_row["entropy_gap"]
#         )
#         analysis.loc[mask, "delta_confidence_gap"] = (
#             analysis.loc[mask, "confidence_gap"] - fp16_row["confidence_gap"]
#         )

#     return analysis


# # ── Plotting ──────────────────────────────────────────────────────────

# def config_label(row) -> str:
#     return f"{row['quant_method']} {row['precision']}"


# def plot_entropy_gap_bars(analysis: pd.DataFrame, dataset: str):
#     """Bar chart: entropy gap per config for one dataset."""
#     subset = analysis[analysis["dataset"] == dataset].sort_values("entropy_gap", ascending=False)
#     if subset.empty:
#         return

#     labels = [config_label(r) for _, r in subset.iterrows()]
#     gaps = subset["entropy_gap"].values

#     fig, ax = plt.subplots(figsize=(8, 4))
#     colors = ["steelblue" if r["quant_method"] == "fp16" else "coral"
#               for _, r in subset.iterrows()]
#     ax.barh(labels, gaps, color=colors)
#     ax.set_xlabel("Entropy Gap (incorrect - correct)")
#     ax.set_title(f"Entropy Separation by Config: {dataset}")
#     ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
#     plt.tight_layout()

#     path = os.path.join(ANALYSIS_DIR, f"entropy_gap_{dataset}.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     print(f"  Saved: {path}")
#     plt.close()


# def plot_confidence_scatter(analysis: pd.DataFrame, dataset: str):
#     """Scatter: mean confidence correct vs incorrect, one point per config."""
#     subset = analysis[analysis["dataset"] == dataset]
#     if subset.empty:
#         return

#     fig, ax = plt.subplots(figsize=(6, 6))
#     for _, row in subset.iterrows():
#         label = config_label(row)
#         marker = "D" if row["quant_method"] == "fp16" else "o"
#         ax.scatter(row["confidence_incorrect_mean"], row["confidence_correct_mean"],
#                    s=80, marker=marker, zorder=3)
#         ax.annotate(label, (row["confidence_incorrect_mean"], row["confidence_correct_mean"]),
#                     fontsize=7, textcoords="offset points", xytext=(5, 5))

#     ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
#     ax.set_xlabel("Mean Confidence (Incorrect)")
#     ax.set_ylabel("Mean Confidence (Correct)")
#     ax.set_title(f"Confidence Separation: {dataset}")
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()

#     path = os.path.join(ANALYSIS_DIR, f"confidence_scatter_{dataset}.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     print(f"  Saved: {path}")
#     plt.close()


# def plot_ece_delta_bars(analysis: pd.DataFrame, dataset: str):
#     """Bar chart: ECE change vs FP16 baseline for one dataset."""
#     subset = analysis[
#         (analysis["dataset"] == dataset) & (analysis["quant_method"] != "fp16")
#     ].sort_values("delta_ECE", ascending=True)
#     if subset.empty:
#         return

#     labels = [config_label(r) for _, r in subset.iterrows()]
#     deltas = subset["delta_ECE"].values

#     fig, ax = plt.subplots(figsize=(8, 4))
#     colors = ["green" if d <= 0 else "red" for d in deltas]
#     ax.barh(labels, deltas, color=colors)
#     ax.set_xlabel("Delta ECE (vs FP16 baseline)")
#     ax.set_title(f"Calibration Degradation by Config: {dataset}")
#     ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
#     plt.tight_layout()

#     path = os.path.join(ANALYSIS_DIR, f"ece_delta_{dataset}.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     print(f"  Saved: {path}")
#     plt.close()


# def plot_ece_heatmap(analysis: pd.DataFrame):
#     """Heatmap: delta ECE across configs × datasets."""
#     quantized = analysis[analysis["quant_method"] != "fp16"].copy()
#     if quantized.empty:
#         return

#     quantized["config"] = quantized.apply(config_label, axis=1)
#     pivot = quantized.pivot_table(index="config", columns="dataset", values="delta_ECE")

#     if pivot.empty:
#         return

#     fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(4, len(pivot) * 0.8)))
#     im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")

#     ax.set_xticks(range(len(pivot.columns)))
#     ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
#     ax.set_yticks(range(len(pivot.index)))
#     ax.set_yticklabels(pivot.index, fontsize=9)

#     for i in range(len(pivot.index)):
#         for j in range(len(pivot.columns)):
#             val = pivot.values[i, j]
#             if not np.isnan(val):
#                 ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

#     plt.colorbar(im, ax=ax, label="Delta ECE")
#     ax.set_title("Calibration Degradation: Configs × Datasets")
#     plt.tight_layout()

#     path = os.path.join(ANALYSIS_DIR, "ece_delta_heatmap.png")
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     print(f"  Saved: {path}")
#     plt.close()


# # ── Main ──────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(description="Entropy analysis across quant configs")
#     parser.add_argument("--include_all_teams", action="store_true",
#                         help="Include Teams A & B results if available")
#     args = parser.parse_args()

#     results_dirs = RESULTS_DIRS_ALL if args.include_all_teams else RESULTS_DIRS_TEAM_C
#     os.makedirs(ANALYSIS_DIR, exist_ok=True)

#     df = load_all_summaries(results_dirs)
#     if df.empty:
#         print("No results found. Run eval scripts first.")
#         return

#     print("Building analysis table...")
#     analysis = build_analysis_table(df, results_dirs)
#     analysis = add_fp16_deltas(analysis)

#     # Print summary
#     print(f"\n=== Entropy Analysis ({len(analysis)} config-dataset pairs) ===\n")
#     display_cols = [
#         "model", "quant_method", "precision", "dataset", "accuracy", "ECE",
#         "entropy_gap", "confidence_gap", "delta_ECE",
#     ]
#     available = [c for c in display_cols if c in analysis.columns]
#     print(analysis[available].to_string(index=False, float_format="%.4f"))

#     # Save CSV
#     csv_path = os.path.join(ANALYSIS_DIR, "entropy_summary.csv")
#     analysis.to_csv(csv_path, index=False)
#     print(f"\nSaved summary: {csv_path}")

#     # Generate figures per dataset
#     datasets = analysis["dataset"].unique()
#     for ds in datasets:
#         print(f"\nPlotting: {ds}")
#         plot_entropy_gap_bars(analysis, ds)
#         plot_confidence_scatter(analysis, ds)
#         plot_ece_delta_bars(analysis, ds)

#     # Cross-dataset heatmap
#     print("\nPlotting: cross-dataset heatmap")
#     plot_ece_heatmap(analysis)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()
