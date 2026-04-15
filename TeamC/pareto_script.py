"""
Cross-model Pareto frontier analysis.

Loads JSON result summaries from all teams and identifies Pareto-dominant
configurations across throughput, accuracy, and calibration error.

Usage:
    python TeamC/pareto_script.py
    python TeamC/pareto_script.py --output_dir TeamC/results/pareto
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIRS = [
    os.path.join(_PROJECT_ROOT, "TeamA", "results"),
    os.path.join(_PROJECT_ROOT, "TeamB", "results"),
    os.path.join(_PROJECT_ROOT, "TeamC", "results"),
]


# -- Data loading -------------------------------------------------------------

def load_cross_team_results(results_dirs: list[str]) -> pd.DataFrame:
    """Load all JSON summary files from all teams into a single DataFrame."""
    rows = []
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "*.json")):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No JSON result files found in: {results_dirs}")
    return pd.DataFrame(rows)


def aggregate_for_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-dataset results to per-config means for Pareto analysis."""
    agg = df.groupby(["model", "quant_method", "precision"]).agg({
        "tokens_per_second": "mean",
        "accuracy": "mean",
        "ECE": "mean",
    }).reset_index()

    agg = agg.rename(columns={"model": "model_name", "ECE": "ece"})
    agg["config_label"] = (
        agg["model_name"] + " " + agg["quant_method"] + " " + agg["precision"]
    )
    return agg


# -- Pareto computation --------------------------------------------------------

def find_pareto_frontier(df: pd.DataFrame,
                         throughput_col: str = 'tokens_per_second',
                         accuracy_col: str = 'accuracy',
                         ece_col: str = 'ece') -> pd.DataFrame:
    """
    Identifies Pareto-dominant configurations.
    Optimization goals:
    - Maximize Inference Throughput (tokens_per_second)
    - Maximize Task Accuracy (accuracy)
    - Minimize Expected Calibration Error (ece)
    """
    is_pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            higher_or_eq_throughput = df[throughput_col].iloc[j] >= df[throughput_col].iloc[i]
            higher_or_eq_accuracy = df[accuracy_col].iloc[j] >= df[accuracy_col].iloc[i]
            lower_or_eq_ece = df[ece_col].iloc[j] <= df[ece_col].iloc[i]

            strictly_better = (
                df[throughput_col].iloc[j] > df[throughput_col].iloc[i] or
                df[accuracy_col].iloc[j] > df[accuracy_col].iloc[i] or
                df[ece_col].iloc[j] < df[ece_col].iloc[i]
            )

            if higher_or_eq_throughput and higher_or_eq_accuracy and lower_or_eq_ece and strictly_better:
                is_pareto[i] = False
                break

    df_out = df.copy()
    df_out['is_pareto'] = is_pareto
    return df_out


# -- 3D Plotly visualization ---------------------------------------------------

def plot_3d_pareto(df: pd.DataFrame,
                   throughput_col: str = 'tokens_per_second',
                   accuracy_col: str = 'accuracy',
                   ece_col: str = 'ece',
                   model_col: str = 'model_name',
                   config_col: str = 'quant_method'):
    """Generates an interactive 3D Plotly visualization of the Pareto frontier."""
    df_analyzed = find_pareto_frontier(df, throughput_col, accuracy_col, ece_col)

    df_pareto = df_analyzed[df_analyzed['is_pareto'] == True]
    df_suboptimal = df_analyzed[df_analyzed['is_pareto'] == False]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df_suboptimal[throughput_col],
        y=df_suboptimal[accuracy_col],
        z=df_suboptimal[ece_col],
        mode='markers',
        name='Sub-optimal Configs',
        text=df_suboptimal[model_col] + ' - ' + df_suboptimal[config_col],
        marker=dict(size=6, color='lightgray', opacity=0.6,
                    line=dict(width=1, color='gray')),
        hovertemplate=("<b>%{text}</b><br>Throughput: %{x:.1f} tokens/s<br>"
                       "Accuracy: %{y:.3f}<br>ECE: %{z:.3f}<extra></extra>")
    ))

    fig.add_trace(go.Scatter3d(
        x=df_pareto[throughput_col],
        y=df_pareto[accuracy_col],
        z=df_pareto[ece_col],
        mode='markers',
        name='Pareto Frontier',
        text=df_pareto[model_col] + ' - ' + df_pareto[config_col],
        marker=dict(size=10, color=df_pareto[accuracy_col],
                    colorscale='Viridis', opacity=0.9, symbol='diamond',
                    line=dict(width=2, color='DarkSlateGrey')),
        hovertemplate=("<b>%{text} (OPTIMAL)</b><br>Throughput: %{x:.1f} tokens/s<br>"
                       "Accuracy: %{y:.3f}<br>ECE: %{z:.3f}<extra></extra>")
    ))

    fig.update_layout(
        title="3D Pareto Frontier: Cost vs. Accuracy vs. Calibration",
        scene=dict(
            xaxis_title="Throughput (tokens/sec)",
            yaxis_title="Task Accuracy",
            zaxis_title="Calibration Error (ECE)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        legend=dict(x=0.8, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


# -- 2D Matplotlib projections ------------------------------------------------

def plot_2d_pareto_projections(df: pd.DataFrame, output_dir: str):
    """Generate 2D Pareto projections for static figures."""
    df_analyzed = find_pareto_frontier(df)
    pareto = df_analyzed[df_analyzed["is_pareto"]]
    non_pareto = df_analyzed[~df_analyzed["is_pareto"]]

    projections = [
        ("tokens_per_second", "accuracy", "Throughput (tokens/s)", "Accuracy"),
        ("tokens_per_second", "ece", "Throughput (tokens/s)", "ECE (lower is better)"),
        ("accuracy", "ece", "Accuracy", "ECE (lower is better)"),
    ]

    for x_col, y_col, x_label, y_label in projections:
        fig, ax = plt.subplots(figsize=(8, 6))
        if not non_pareto.empty:
            ax.scatter(non_pareto[x_col], non_pareto[y_col], c="lightgray",
                       s=60, edgecolors="gray", label="Sub-optimal", zorder=2)
        if not pareto.empty:
            ax.scatter(pareto[x_col], pareto[y_col], c="steelblue", s=100,
                       edgecolors="black", marker="D", label="Pareto-optimal",
                       zorder=3)
            for _, row in pareto.iterrows():
                ax.annotate(row.get("config_label", ""),
                            (row[x_col], row[y_col]),
                            fontsize=7, textcoords="offset points",
                            xytext=(5, 5))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Pareto Frontier: {x_label} vs {y_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"pareto_2d_{x_col}_vs_{y_col}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
        plt.close()


# -- Per-model Pareto analysis --------------------------------------------------

def plot_per_model_pareto(df: pd.DataFrame, output_dir: str):
    """Compute per-model Pareto frontiers and overlay to compare model families."""
    models = df["model_name"].unique()
    if len(models) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    projections = [
        ("tokens_per_second", "accuracy", "Throughput (tokens/s)", "Accuracy"),
        ("tokens_per_second", "ece", "Throughput (tokens/s)", "ECE"),
        ("accuracy", "ece", "Accuracy", "ECE"),
    ]
    colors = {"llama2_7b": "blue", "llama2_13b": "red", "mistral_7b": "green"}
    markers = {"llama2_7b": "o", "llama2_13b": "s", "mistral_7b": "^"}

    per_model_rows = []
    for model in models:
        model_df = df[df["model_name"] == model].copy()
        model_analyzed = find_pareto_frontier(model_df)
        for _, row in model_analyzed.iterrows():
            per_model_rows.append({**row.to_dict(), "model_family": model})

        color = colors.get(model, "gray")
        marker = markers.get(model, "o")
        pareto = model_analyzed[model_analyzed["is_pareto"]]
        non_pareto = model_analyzed[~model_analyzed["is_pareto"]]

        for ax, (x_col, y_col, x_label, y_label) in zip(axes, projections):
            if not non_pareto.empty:
                ax.scatter(non_pareto[x_col], non_pareto[y_col], c=color,
                           marker=marker, s=40, alpha=0.3)
            if not pareto.empty:
                ax.scatter(pareto[x_col], pareto[y_col], c=color,
                           marker=marker, s=100, edgecolors="black",
                           label=f"{model} (Pareto)", zorder=3)
                for _, r in pareto.iterrows():
                    ax.annotate(r["quant_method"] + " " + r["precision"],
                                (r[x_col], r[y_col]), fontsize=6,
                                textcoords="offset points", xytext=(4, 4))

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, projections):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Model Pareto Frontiers: Comparing Model Families", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_per_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: pareto_per_model_comparison.png")
    plt.close()

    # Save per-model Pareto CSV
    if per_model_rows:
        pd.DataFrame(per_model_rows).to_csv(
            os.path.join(output_dir, "pareto_per_model.csv"), index=False)
        print(f"  Saved: pareto_per_model.csv")


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-model Pareto frontier analysis")
    parser.add_argument("--output_dir",
                        default=os.path.join(os.path.dirname(__file__), "results", "pareto"))
    args = parser.parse_args()

    results_dirs = [d for d in RESULTS_DIRS if os.path.isdir(d)]

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df_raw = load_cross_team_results(results_dirs)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run evaluation scripts first (run_sweep.sh).")
        sys.exit(1)

    df_pareto = aggregate_for_pareto(df_raw)

    print(f"Loaded {len(df_raw)} result files, aggregated to {len(df_pareto)} configs")
    print(df_pareto.to_string(index=False))

    # Save aggregated data
    df_pareto.to_csv(os.path.join(args.output_dir, "pareto_configs.csv"), index=False)

    # 2D matplotlib plots (always available)
    plot_2d_pareto_projections(df_pareto, args.output_dir)

    # Per-model Pareto comparison (analyze frontier shape by model family)
    plot_per_model_pareto(df_pareto, args.output_dir)

    # 3D interactive plot (if plotly available)
    if HAS_PLOTLY:
        fig = plot_3d_pareto(df_pareto, model_col="model_name",
                             config_col="quant_method")
        html_path = os.path.join(args.output_dir, "pareto_frontier_interactive.html")
        fig.write_html(html_path)
        print(f"  Saved: pareto_frontier_interactive.html")
    else:
        print("Skipping 3D interactive plot (plotly not installed)")

    # Per-dataset Pareto CSVs
    for dataset in df_raw["dataset"].unique():
        df_ds = df_raw[df_raw["dataset"] == dataset].copy()
        df_ds = df_ds.rename(columns={"model": "model_name", "ECE": "ece"})
        df_ds_analyzed = find_pareto_frontier(df_ds)
        csv_path = os.path.join(args.output_dir, f"pareto_{dataset}.csv")
        df_ds_analyzed.to_csv(csv_path, index=False)
        print(f"  Saved: pareto_{dataset}.csv")

    print(f"\nDone. Results saved to: {args.output_dir}")
