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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    os.path.join(_PROJECT_ROOT, "TeamB", "calibration_results"),
    os.path.join(_PROJECT_ROOT, "TeamC", "updated_results"),
]


# -- Data loading -------------------------------------------------------------

def load_cross_team_results(results_dirs: list) -> pd.DataFrame:
    rows = []
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No JSON result files found in: {results_dirs}")
    return pd.DataFrame(rows)


def aggregate_for_pareto(df: pd.DataFrame) -> pd.DataFrame:
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


def json_dir_to_df(folder_path) -> pd.DataFrame:
    entries = []
    for file in Path(folder_path).glob("**/*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
            if all(k in data for k in ['dataset', 'tokens_per_second', 'accuracy', 'ECE']):
                entries.append(data)
        except Exception:
            pass
    return pd.DataFrame(entries)


# -- Pareto computation -------------------------------------------------------

def find_pareto_frontier(df: pd.DataFrame,
                         metrics: list = None,
                         goals: list = None) -> pd.DataFrame:
    """
    Identifies Pareto-dominant configurations.
    metrics/goals default to tokens_per_second↑, accuracy↑, ece↓.
    """
    if df.empty:
        return df
    if metrics is None:
        metrics = ['tokens_per_second', 'accuracy', 'ece']
        goals = ['max', 'max', 'min']

    df_reset = df.reset_index(drop=True)
    is_pareto = np.ones(len(df_reset), dtype=bool)

    for i in range(len(df_reset)):
        for j in range(len(df_reset)):
            if i == j:
                continue
            is_better_or_equal = []
            is_strictly_better = False
            for m, goal in zip(metrics, goals):
                val_i = df_reset[m].iloc[i]
                val_j = df_reset[m].iloc[j]
                if goal == 'max':
                    is_better_or_equal.append(val_j >= val_i)
                    if val_j > val_i:
                        is_strictly_better = True
                else:
                    is_better_or_equal.append(val_j <= val_i)
                    if val_j < val_i:
                        is_strictly_better = True
            if all(is_better_or_equal) and is_strictly_better:
                is_pareto[i] = False
                break

    df_out = df_reset.copy()
    df_out['is_pareto'] = is_pareto
    return df_out


# -- Plotly visualizations ----------------------------------------------------

def create_plot(df: pd.DataFrame, dataset_name: str, view_type: str):
    """
    view_type: '3d', 't_vs_a', 't_vs_c', 'a_vs_c'
    Returns (fig, df_analyzed).
    """
    configs = {
        '3d':     {'metrics': ['tokens_per_second', 'accuracy', 'ECE'], 'goals': ['max', 'max', 'min']},
        't_vs_a': {'metrics': ['tokens_per_second', 'accuracy'],         'goals': ['max', 'max']},
        't_vs_c': {'metrics': ['tokens_per_second', 'ECE'],              'goals': ['max', 'min']},
        'a_vs_c': {'metrics': ['accuracy', 'ECE'],                       'goals': ['max', 'min']},
    }
    cfg = configs[view_type]
    df_analyzed = find_pareto_frontier(df, cfg['metrics'], cfg['goals'])
    fig = go.Figure()

    if view_type == '3d':
        df_pareto = df_analyzed[df_analyzed['is_pareto']]
        df_sub = df_analyzed[~df_analyzed['is_pareto']]
        for name, d, color, size, symbol in [
            ('Sub-optimal', df_sub, 'lightgray', 6, 'circle'),
            ('Pareto Optimal', df_pareto,
             df_pareto['accuracy'] if not df_pareto.empty else 'green', 10, 'diamond'),
        ]:
            if d.empty:
                continue
            fig.add_trace(go.Scatter3d(
                x=d['tokens_per_second'], y=d['accuracy'], z=d['ECE'],
                mode='markers', name=name, text=d['quant_method'],
                marker=dict(size=size, color=color,
                            colorscale='Viridis' if name == 'Pareto Optimal' else None,
                            symbol=symbol, opacity=0.8, line=dict(width=1, color='black')),
                hovertemplate=(
                    "<b>%{text}</b><br>Throughput: %{x:.2f}<br>"
                    "Accuracy: %{y:.4f}<br>ECE: %{z:.4f}<extra></extra>"
                ),
            ))
        fig.update_layout(scene=dict(
            xaxis_title="Throughput", yaxis_title="Accuracy", zaxis_title="ECE"
        ))
    else:
        x_m, y_m = cfg['metrics']
        df_sorted = df_analyzed.sort_values(by=x_m)
        df_pareto = df_sorted[df_sorted['is_pareto']]
        if not df_pareto.empty:
            fig.add_trace(go.Scatter(
                x=df_pareto[x_m], y=df_pareto[y_m], mode='lines', name='Frontier',
                line=dict(color='rgba(39, 174, 96, 0.5)', width=2, shape='hv'),
            ))
        fig.add_trace(go.Scatter(
            x=df_analyzed[x_m], y=df_analyzed[y_m], mode='markers+text',
            text=df_analyzed['quant_method'], textposition="top center",
            marker=dict(size=12,
                        color=df_analyzed['is_pareto'].map({True: '#27ae60', False: '#bdc3c7'}),
                        line=dict(width=1, color='black')),
            hovertemplate=f"<b>%{{text}}</b><br>{x_m}: %{{x:.4f}}<br>{y_m}: %{{y:.4f}}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title=x_m.replace('_', ' ').title(),
            yaxis_title=y_m.replace('_', ' ').title(),
        )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), template="plotly_white", height=450)
    return fig, df_analyzed


def plot_3d_pareto(df: pd.DataFrame,
                   throughput_col: str = 'tokens_per_second',
                   accuracy_col: str = 'accuracy',
                   ece_col: str = 'ece',
                   model_col: str = 'model_name',
                   config_col: str = 'quant_method'):
    """Interactive 3D Plotly visualization of the Pareto frontier."""
    df_analyzed = find_pareto_frontier(df, [throughput_col, accuracy_col, ece_col],
                                       ['max', 'max', 'min'])
    df_pareto = df_analyzed[df_analyzed['is_pareto']]
    df_suboptimal = df_analyzed[~df_analyzed['is_pareto']]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df_suboptimal[throughput_col], y=df_suboptimal[accuracy_col], z=df_suboptimal[ece_col],
        mode='markers', name='Sub-optimal Configs',
        text=df_suboptimal[model_col] + ' - ' + df_suboptimal[config_col],
        marker=dict(size=6, color='lightgray', opacity=0.6, line=dict(width=1, color='gray')),
        hovertemplate=(
            "<b>%{text}</b><br>Throughput: %{x:.1f} tokens/s<br>"
            "Accuracy: %{y:.3f}<br>ECE: %{z:.3f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter3d(
        x=df_pareto[throughput_col], y=df_pareto[accuracy_col], z=df_pareto[ece_col],
        mode='markers', name='Pareto Frontier',
        text=df_pareto[model_col] + ' - ' + df_pareto[config_col],
        marker=dict(size=10, color=df_pareto[accuracy_col], colorscale='Viridis',
                    opacity=0.9, symbol='diamond', line=dict(width=2, color='DarkSlateGrey')),
        hovertemplate=(
            "<b>%{text} (OPTIMAL)</b><br>Throughput: %{x:.1f} tokens/s<br>"
            "Accuracy: %{y:.3f}<br>ECE: %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="3D Pareto Frontier: Cost vs. Accuracy vs. Calibration",
        scene=dict(
            xaxis_title="Throughput (tokens/sec)",
            yaxis_title="Task Accuracy",
            zaxis_title="Calibration Error (ECE)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        legend=dict(x=0.8, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def generate_combined_html(df_metrics, output_file="pareto_comparison.html"):
    if df_metrics.empty:
        return
    datasets = sorted(df_metrics['dataset'].unique())
    view_types = [
        ('3d',     '3D Pareto Frontier'),
        ('t_vs_a', 'Throughput vs. Accuracy'),
        ('t_vs_c', 'Throughput vs. Calibration'),
        ('a_vs_c', 'Accuracy vs. Calibration'),
    ]

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Quantization Analysis</title>
    <style>
        html, body { height: 100%; margin: 0; padding: 0; font-family: sans-serif; background-color: #f8f9fa; overflow: hidden; }
        .main-container { display: flex; flex-direction: column; height: 100vh; padding: 10px; box-sizing: border-box; }
        .parent-tabs { display: flex; background: #2c3e50; padding: 5px 10px 0; gap: 5px; }
        .parent-tabs button { background: #34495e; color: #bdc3c7; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px 5px 0 0; font-weight: bold; }
        .parent-tabs button.active { background: #ecf0f1; color: #2c3e50; }
        .parent-content { display: none; flex-grow: 1; flex-direction: column; background: #ecf0f1; border: 1px solid #ccc; overflow: hidden; }
        .child-tabs { display: flex; background: #ddd; padding: 5px 10px 0; gap: 2px; border-bottom: 1px solid #ccc; }
        .child-tabs button { background: #ccc; border: none; padding: 8px 15px; cursor: pointer; border-radius: 4px 4px 0 0; font-size: 13px; }
        .child-tabs button.active { background: #fff; font-weight: bold; }
        .child-content { display: none; flex-grow: 1; flex-direction: column; padding: 10px; background: #fff; overflow: hidden; }
        .plot-area { flex: 1; min-height: 0; border-bottom: 1px solid #eee; }
        .table-area { flex: 0 0 35%; overflow-y: auto; font-size: 12px; padding-top: 10px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }
        th { background: #f2f2f2; position: sticky; top: 0; }
        .optimal { background: #e8f5e9; font-weight: bold; }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="parent-tabs">
"""
    for i, (vt_id, vt_name) in enumerate(view_types):
        active = "active" if i == 0 else ""
        html += f'<button class="ptab-links {active}" onclick="openParent(event,\'{vt_id}\')">{vt_name}</button>'
    html += "</div>"

    for i, (vt_id, vt_name) in enumerate(view_types):
        display = "flex" if i == 0 else "none"
        html += f'<div id="{vt_id}" class="parent-content" style="display:{display}"><div class="child-tabs">'
        for j, ds in enumerate(datasets):
            active = "active" if j == 0 else ""
            html += (f'<button class="ctab-links-{vt_id} {active}" '
                     f'onclick="openChild(event,\'{vt_id}_{ds}\',\'{vt_id}\')">{ds}</button>')
        html += '</div>'
        for j, ds in enumerate(datasets):
            display_c = "flex" if j == 0 else "none"
            html += f'<div id="{vt_id}_{ds}" class="child-content" style="display:{display_c}">'
            ds_df = df_metrics[df_metrics['dataset'] == ds]
            fig, df_analyzed = create_plot(ds_df, ds, vt_id)
            html += (f'<div class="plot-area">'
                     f'{fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=f"p_{vt_id}_{ds}")}'
                     f'</div>')
            html += '<div class="table-area"><table><thead><tr>'
            cols = ['model', 'quant_method', 'tokens_per_second', 'accuracy', 'ECE', 'is_pareto']
            for c in cols:
                html += f'<th>{c.replace("_"," ").title()}</th>'
            html += '</tr></thead><tbody>'
            for _, row in df_analyzed.sort_values('accuracy', ascending=False).iterrows():
                row_cls = "optimal" if row['is_pareto'] else ""
                html += f'<tr class="{row_cls}">'
                for c in cols:
                    val = ("Yes" if c == 'is_pareto' and row[c]
                           else ("No" if c == 'is_pareto' else row[c]))
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    html += f'<td>{val}</td>'
                html += '</tr>'
            html += '</tbody></table></div></div>'
        html += '</div>'

    html += """    </div>
    <script>
        function openParent(evt, pName) {
            document.querySelectorAll('.parent-content').forEach(e => e.style.display='none');
            document.querySelectorAll('.ptab-links').forEach(e => e.classList.remove('active'));
            document.getElementById(pName).style.display='flex';
            evt.currentTarget.classList.add('active');
            window.dispatchEvent(new Event('resize'));
        }
        function openChild(evt, cName, pName) {
            var parent = document.getElementById(pName);
            parent.querySelectorAll('.child-content').forEach(e => e.style.display='none');
            parent.querySelectorAll('.ctab-links-'+pName).forEach(e => e.classList.remove('active'));
            document.getElementById(cName).style.display='flex';
            evt.currentTarget.classList.add('active');
            window.dispatchEvent(new Event('resize'));
        }
        window.onload = () => window.dispatchEvent(new Event('resize'));
    </script>
</body>
</html>
"""
    with open(output_file, "w") as f:
        f.write(html)
    print(f"Generated {output_file}")


# -- Matplotlib visualizations ------------------------------------------------

def plot_2d_pareto_projections(df: pd.DataFrame, output_dir: str):
    df_analyzed = find_pareto_frontier(df)
    pareto = df_analyzed[df_analyzed["is_pareto"]]
    non_pareto = df_analyzed[~df_analyzed["is_pareto"]]

    projections = [
        ("tokens_per_second", "accuracy", "Throughput (tokens/s)", "Accuracy"),
        ("tokens_per_second", "ece",      "Throughput (tokens/s)", "ECE (lower is better)"),
        ("accuracy",          "ece",      "Accuracy",              "ECE (lower is better)"),
    ]
    for x_col, y_col, x_label, y_label in projections:
        fig, ax = plt.subplots(figsize=(8, 6))
        if not non_pareto.empty:
            ax.scatter(non_pareto[x_col], non_pareto[y_col], c="lightgray",
                       s=60, edgecolors="gray", label="Sub-optimal", zorder=2)
        if not pareto.empty:
            ax.scatter(pareto[x_col], pareto[y_col], c="steelblue", s=100,
                       edgecolors="black", marker="D", label="Pareto-optimal", zorder=3)
            for _, row in pareto.iterrows():
                ax.annotate(row.get("config_label", ""), (row[x_col], row[y_col]),
                            fontsize=7, textcoords="offset points", xytext=(5, 5))
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


def plot_per_model_pareto(df: pd.DataFrame, output_dir: str):
    models = df["model_name"].unique()
    if len(models) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    projections = [
        ("tokens_per_second", "accuracy", "Throughput (tokens/s)", "Accuracy"),
        ("tokens_per_second", "ece",      "Throughput (tokens/s)", "ECE"),
        ("accuracy",          "ece",      "Accuracy",              "ECE"),
    ]
    colors  = {"llama2_7b": "blue", "llama2_13b": "red",  "mistral_7b": "green"}
    markers = {"llama2_7b": "o",    "llama2_13b": "s",    "mistral_7b": "^"}

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
                ax.scatter(non_pareto[x_col], non_pareto[y_col],
                           c=color, marker=marker, s=40, alpha=0.3)
            if not pareto.empty:
                ax.scatter(pareto[x_col], pareto[y_col], c=color, marker=marker,
                           s=100, edgecolors="black", label=f"{model} (Pareto)", zorder=3)
                for _, r in pareto.iterrows():
                    ax.annotate(r["quant_method"] + " " + r["precision"],
                                (r[x_col], r[y_col]),
                                fontsize=6, textcoords="offset points", xytext=(4, 4))

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, projections):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Model Pareto Frontiers: Comparing Model Families", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_per_model_comparison.png"), dpi=150, bbox_inches="tight")
    print("  Saved: pareto_per_model_comparison.png")
    plt.close()

    if per_model_rows:
        pd.DataFrame(per_model_rows).to_csv(
            os.path.join(output_dir, "pareto_per_model.csv"), index=False)
        print("  Saved: pareto_per_model.csv")


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

    df_pareto.to_csv(os.path.join(args.output_dir, "pareto_configs.csv"), index=False)
    plot_2d_pareto_projections(df_pareto, args.output_dir)
    plot_per_model_pareto(df_pareto, args.output_dir)

    if HAS_PLOTLY:
        generate_combined_html(df_raw, os.path.join(args.output_dir, "pareto_comparison.html"))
        html_path = os.path.join(args.output_dir, "pareto_frontier_interactive.html")
        fig = plot_3d_pareto(df_pareto, model_col="model_name", config_col="quant_method")
        fig.write_html(html_path)
        print(f"  Saved: pareto_frontier_interactive.html")
    else:
        print("Skipping interactive plots (plotly not installed)")

    for dataset in df_raw["dataset"].unique():
        df_ds = df_raw[df_raw["dataset"] == dataset].copy()
        df_ds = df_ds.rename(columns={"model": "model_name", "ECE": "ece"})
        df_ds_analyzed = find_pareto_frontier(df_ds)
        csv_path = os.path.join(args.output_dir, f"pareto_{dataset}.csv")
        df_ds_analyzed.to_csv(csv_path, index=False)
        print(f"  Saved: pareto_{dataset}.csv")

    print(f"\nDone. Results saved to: {args.output_dir}")
