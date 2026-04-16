import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

def find_pareto_frontier(df: pd.DataFrame, metrics: list, goals: list):
    """
    Identifies Pareto-dominant configurations based on multiple metrics.
    metrics: list of column names
    goals: list of 'max' or 'min' for each metric
    """
    if df.empty: return df
    df_reset = df.reset_index(drop=True)
    is_pareto = np.ones(len(df_reset), dtype=bool)
    
    for i in range(len(df_reset)):
        for j in range(len(df_reset)):
            if i == j: continue
            
            # Check if j dominates i
            is_better_or_equal = []
            is_strictly_better = False
            
            for m, goal in zip(metrics, goals):
                val_i = df_reset[m].iloc[i]
                val_j = df_reset[m].iloc[j]
                
                if goal == 'max':
                    is_better_or_equal.append(val_j >= val_i)
                    if val_j > val_i: is_strictly_better = True
                else:
                    is_better_or_equal.append(val_j <= val_i)
                    if val_j < val_i: is_strictly_better = True
            
            if all(is_better_or_equal) and is_strictly_better:
                is_pareto[i] = False
                break
                
    df_out = df_reset.copy()
    df_out['is_pareto'] = is_pareto
    return df_out

def create_plot(df: pd.DataFrame, dataset_name: str, view_type: str):
    """
    Generates the requested plot type for a dataset.
    view_type: '3d', 't_vs_a', 't_vs_c', 'a_vs_c'
    """
    configs = {
        '3d': {'metrics': ['tokens_per_second', 'accuracy', 'ECE'], 'goals': ['max', 'max', 'min']},
        't_vs_a': {'metrics': ['tokens_per_second', 'accuracy'], 'goals': ['max', 'max']},
        't_vs_c': {'metrics': ['tokens_per_second', 'ECE'], 'goals': ['max', 'min']},
        'a_vs_c': {'metrics': ['accuracy', 'ECE'], 'goals': ['max', 'min']}
    }
    
    cfg = configs[view_type]
    df_analyzed = find_pareto_frontier(df, cfg['metrics'], cfg['goals'])
    
    fig = go.Figure()
    
    if view_type == '3d':
        df_pareto = df_analyzed[df_analyzed['is_pareto']]
        df_sub = df_analyzed[~df_analyzed['is_pareto']]
        
        for name, d, color, size, symbol in [
            ('Sub-optimal', df_sub, 'lightgray', 6, 'circle'),
            ('Pareto Optimal', df_pareto, df_pareto['accuracy'] if not df_pareto.empty else 'green', 10, 'diamond')
        ]:
            if d.empty: continue
            fig.add_trace(go.Scatter3d(
                x=d['tokens_per_second'], y=d['accuracy'], z=d['ECE'],
                mode='markers', name=name,
                text=d['quant_method'],
                marker=dict(size=size, color=color, colorscale='Viridis' if name=='Pareto Optimal' else None, 
                            symbol=symbol, opacity=0.8, line=dict(width=1, color='black')),
                hovertemplate="<b>%{text}</b><br>Throughput: %{x:.2f}<br>Accuracy: %{y:.4f}<br>ECE: %{z:.4f}<extra></extra>"
            ))
        fig.update_layout(scene=dict(xaxis_title="Throughput", yaxis_title="Accuracy", zaxis_title="ECE"))
    
    else:
        x_m, y_m = cfg['metrics']
        df_sorted = df_analyzed.sort_values(by=x_m)
        df_pareto = df_sorted[df_sorted['is_pareto']]
        
        if not df_pareto.empty:
            fig.add_trace(go.Scatter(x=df_pareto[x_m], y=df_pareto[y_m], mode='lines', 
                                     name='Frontier', line=dict(color='rgba(39, 174, 96, 0.5)', width=2, shape='hv')))
        
        fig.add_trace(go.Scatter(
            x=df_analyzed[x_m], y=df_analyzed[y_m], mode='markers+text',
            text=df_analyzed['quant_method'], textposition="top center",
            marker=dict(size=12, color=df_analyzed['is_pareto'].map({True: '#27ae60', False: '#bdc3c7'}),
                        line=dict(width=1, color='black')),
            hovertemplate=f"<b>%{{text}}</b><br>{x_m}: %{{x:.4f}}<br>{y_m}: %{{y:.4f}}<extra></extra>"
        ))
        fig.update_layout(xaxis_title=x_m.replace('_', ' ').title(), yaxis_title=y_m.replace('_', ' ').title())

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), template="plotly_white", height=450)
    return fig, df_analyzed

def json_dir_to_df(folder_path):
    folder = Path(folder_path)
    files = list(folder.glob("**/*.json"))
    entries = []
    for file in files:
        try:
            with open(file) as f: data = json.load(f)
            if all(k in data for k in ['dataset', 'tokens_per_second', 'accuracy', 'ECE']): entries.append(data)
        except: pass
    return pd.DataFrame(entries)

def generate_combined_html(df_metrics, output_file="pareto_comparison.html"):
    if df_metrics.empty: return
    datasets = sorted(df_metrics['dataset'].unique())
    view_types = [
        ('3d', '3D Pareto Frontier'),
        ('t_vs_a', 'Throughput vs. Accuracy'),
        ('t_vs_c', 'Throughput vs. Calibration'),
        ('a_vs_c', 'Accuracy vs. Calibration')
    ]
    
    html_content = """
    <!DOCTYPE html>
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
        html_content += f'<button class="ptab-links {active}" onclick="openParent(event, \'{vt_id}\')">{vt_name}</button>'
    
    html_content += "</div>"
    
    for i, (vt_id, vt_name) in enumerate(view_types):
        display = "flex" if i == 0 else "none"
        html_content += f'<div id="{vt_id}" class="parent-content" style="display: {display}">'
        html_content += '<div class="child-tabs">'
        for j, ds in enumerate(datasets):
            active = "active" if j == 0 else ""
            html_content += f'<button class="ctab-links-{vt_id} {active}" onclick="openChild(event, \'{vt_id}_{ds}\', \'{vt_id}\')">{ds}</button>'
        html_content += '</div>'
        
        for j, ds in enumerate(datasets):
            display_c = "flex" if j == 0 else "none"
            html_content += f'<div id="{vt_id}_{ds}" class="child-content" style="display: {display_c}">'
            
            ds_df = df_metrics[df_metrics['dataset'] == ds]
            fig, df_analyzed = create_plot(ds_df, ds, vt_id)
            
            html_content += f'<div class="plot-area">{fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=f"p_{vt_id}_{ds}")}</div>'
            html_content += '<div class="table-area"><table><thead><tr>'
            cols = ['model', 'quant_method', 'tokens_per_second', 'accuracy', 'ECE', 'is_pareto']
            for c in cols: html_content += f'<th>{c.replace("_", " ").title()}</th>'
            html_content += '</tr></thead><tbody>'
            
            df_analyzed = df_analyzed.sort_values(by='accuracy', ascending=False)
            for _, row in df_analyzed.iterrows():
                row_cls = "optimal" if row['is_pareto'] else ""
                html_content += f'<tr class="{row_cls}">'
                for c in cols:
                    val = "Yes" if c == 'is_pareto' and row[c] else ("No" if c == 'is_pareto' else row[c])
                    if isinstance(val, float): val = f"{val:.4f}"
                    html_content += f'<td>{val}</td>'
                html_content += '</tr>'
            html_content += '</tbody></table></div></div>'
        html_content += '</div>'

    html_content += """
        </div>
        <script>
            function openParent(evt, pName) {
                var i, pcontent, plinks;
                pcontent = document.getElementsByClassName("parent-content");
                for (i = 0; i < pcontent.length; i++) pcontent[i].style.display = "none";
                plinks = document.getElementsByClassName("ptab-links");
                for (i = 0; i < plinks.length; i++) plinks[i].className = plinks[i].className.replace(" active", "");
                document.getElementById(pName).style.display = "flex";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }
            function openChild(evt, cName, pName) {
                var i, ccontent, clinks;
                // Scope to parent to avoid affecting other parent tabs
                var parent = document.getElementById(pName);
                ccontent = parent.getElementsByClassName("child-content");
                for (i = 0; i < ccontent.length; i++) ccontent[i].style.display = "none";
                clinks = parent.getElementsByClassName("ctab-links-" + pName);
                for (i = 0; i < clinks.length; i++) clinks[i].className = clinks[i].className.replace(" active", "");
                document.getElementById(cName).style.display = "flex";
                evt.currentTarget.className += " active";
                window.dispatchEvent(new Event('resize'));
            }
            window.onload = function() { window.dispatchEvent(new Event('resize')); };
        </script>
    </body>
    </html>
    """
    with open(output_file, "w") as f: f.write(html_content)
    print(f"Generated {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an HTML Pareto analysis from a folder of JSON results.")
    parser.add_argument("results_folder", nargs="?", default="./updated_results",
                        help="Path to the folder containing JSON result files (default: ./updated_results)")
    parser.add_argument("--output", default="pareto_comparison.html",
                        help="Output HTML file name or path")
    args = parser.parse_args()

    df = json_dir_to_df(args.results_folder)
    if not df.empty:
        generate_combined_html(df, args.output)
    else:
        print(f"No data found in '{args.results_folder}'.")
