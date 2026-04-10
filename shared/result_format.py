"""
Canonical format for saving and loading calibration results.
All teams import this to ensure consistent result schema.
"""

import json
import os
import torch


def save_results(
    model: str,
    precision: str,
    quant_method: str,
    dataset: str,
    confidences: torch.Tensor,
    accuracies: torch.Tensor,
    entropies: torch.Tensor,
    metrics: dict,
    tokens_per_second: float,
    output_dir: str,
) -> dict:
    """Save raw tensors as .pt and summary metrics as .json.

    Files: {model}_{quant_method}_{precision}_{dataset}.pt / .json
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{model}_{quant_method}_{precision}_{dataset}"

    torch.save({
        "confidences": confidences.cpu(),
        "accuracies": accuracies.cpu(),
        "entropies": entropies.cpu(),
    }, os.path.join(output_dir, f"{prefix}.pt"))

    summary = {
        "model": model,
        "precision": precision,
        "quant_method": quant_method,
        "dataset": dataset,
        "tokens_per_second": tokens_per_second,
        "accuracy": accuracies.float().mean().item(),
        "num_examples": len(confidences),
        **metrics,
    }
    with open(os.path.join(output_dir, f"{prefix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_results(path: str) -> dict:
    """Load a .pt result file. Returns dict with confidences, accuracies, entropies."""
    return torch.load(path, map_location="cpu", weights_only=True)
