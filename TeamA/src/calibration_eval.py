from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_calibration_metrics(confidences, accuracies, entropies, num_bins=15):
    """Compute calibration metrics from prediction confidences, correctness labels, and entropies."""
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)

    ece = torch.tensor(0.0, device=confidences.device)
    mce = torch.tensor(0.0, device=confidences.device)
    brier_score = torch.mean((confidences - accuracies.float()) ** 2).item()

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += calibration_error * prop_in_bin
            if calibration_error > mce:
                mce = calibration_error

    return {
        "ECE": ece.item(),
        "MCE": mce.item(),
        "Brier_Score": brier_score,
        "Avg_Entropy": entropies.mean().item()
    }


def plot_reliability_diagram(confidences, accuracies, save_path: str, title="Reliability Diagram", num_bins=15):
    """Create and save a reliability diagram comparing average confidence and accuracy across bins."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(accuracies[mask])
            bin_confs[i] = np.mean(confidences[mask])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.bar(bins[:-1], bin_accs, width=1/num_bins, align='edge', alpha=0.5, edgecolor='black', label='Accuracy')
    plt.plot(bin_confs[bin_confs > 0], bin_accs[bin_confs > 0], 'ro-', label='Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_entropy_distribution(entropies, accuracies, save_path: str, title="Entropy Distribution"):
    """Create and save entropy histograms for correct and incorrect predictions."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    e = entropies.cpu().numpy()
    a = accuracies.cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(e[a == 1], bins=20, alpha=0.6, color='green', label='Correct', density=True)
    plt.hist(e[a == 0], bins=20, alpha=0.6, color='red', label='Incorrect', density=True)
    plt.xlabel('Shannon Entropy (bits)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()