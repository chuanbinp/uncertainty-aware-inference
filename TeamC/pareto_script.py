import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import glob
from pathlib import Path
import json



def find_pareto_frontier(df: pd.DataFrame, 
                         throughput_col: str = 'tokens_per_second', 
                         accuracy_col: str = 'accuracy', 
                         ece_col: str = 'ece',
                        ) -> pd.DataFrame:
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
            
            # Check if configuration 'j' dominates configuration 'i'
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

def plot_3d_pareto(df: pd.DataFrame, 
                   throughput_col: str = 'tokens_per_second', 
                   accuracy_col: str = 'accuracy', 
                   ece_col: str = 'ECE',
                   model_col: str = 'model',
                   config_col: str = 'quant_method',
                   dataset_col: str = "dataset"):
    """
    Generates an interactive 3D plot using Plotly to visualize the Pareto frontier.
    """
    # Calculate Pareto dominance
    df_analyzed = find_pareto_frontier(df, throughput_col, accuracy_col, ece_col)
    
    # Split the dataframe for different marker styling
    df_pareto = df_analyzed[df_analyzed['is_pareto'] == True]
    df_suboptimal = df_analyzed[df_analyzed['is_pareto'] == False]

    fig = go.Figure()

    # Plot sub-optimal configurations
    fig.add_trace(go.Scatter3d(
        x=df_suboptimal[throughput_col],
        y=df_suboptimal[accuracy_col],
        z=df_suboptimal[ece_col],
        mode='markers',
        name='Sub-optimal Configs',
        text=df_suboptimal[model_col] + ' - ' + df_suboptimal[config_col] + ' [' + df_suboptimal[dataset_col] + ']',
        marker=dict(
            size=6,
            color='lightgray',
            opacity=0.6,
            line=dict(width=1, color='gray')
        ),
        hovertemplate="<b>%{text}</b><br>Throughput: %{x:.1f} tokens/s<br>Accuracy: %{y:.3f}<br>ECE: %{z:.3f}<extra></extra>"
    ))

    # Plot Pareto-optimal configurations
    fig.add_trace(go.Scatter3d(
        x=df_pareto[throughput_col],
        y=df_pareto[accuracy_col],
        z=df_pareto[ece_col],
        mode='markers',
        name='Pareto Frontier',
        text=df_pareto[model_col] + ' - ' + df_pareto[config_col] + ' [' + df_pareto[dataset_col] + ']',
        marker=dict(
            size=10,
            color=df_pareto[accuracy_col],
            colorscale='Viridis',
            opacity=0.9,
            symbol='diamond',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hovertemplate="<b>%{text} (OPTIMAL)</b><br>Throughput: %{x:.1f} tokens/s<br>Accuracy: %{y:.3f}<br>ECE: %{z:.3f}<extra></extra>"
    ))

    # Format the layout
    fig.update_layout(
        title="3D Pareto Frontier: Cost vs. Accuracy vs. Calibration",
        scene=dict(
            xaxis_title="Throughput (tokens/sec)",
            yaxis_title="Task Accuracy",
            zaxis_title="Calibration Error (ECE)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2) # Viewing angle
            )
        ),
        legend=dict(x=0.8, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

def json_dir_to_df(folder_path, recursive=True):

    folder = Path(folder_path)
    pattern = "**/*.json" if recursive == True else "*.json"
    files = list(folder.glob(pattern))

    if len(files) == 0:

        raise FileNotFoundError(f"No valid JSON files found in {folder_path}. Double check")
    
    entries = []

    for file in files:

        try:

            with open(file) as f:    
                data = json.load(f)
            entries.append(data)
        
        except:
            print(f"Skipping {file} due to some issue")
    
    output_df = pd.DataFrame(entries)
    return output_df



if __name__ == "__main__":
    # Mock data
    folder_path = "./updated_results"
    df_metrics = json_dir_to_df(folder_path)
    
    # Generate and show the plot
    fig = plot_3d_pareto(df_metrics)
    fig.write_html("pareto_frontier_interactive.html")