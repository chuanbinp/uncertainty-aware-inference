import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)

def compute_calibration_metrics(confidences, accuracies, entropies, num_bins=15):
    """Computes ECE, MCE, Brier Score, and Average Entropy."""
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

def plot_reliability_diagram(confidences, accuracies, title="Reliability Diagram", num_bins=15):
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
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_reliability_diagram.png"))
    plt.show()

def plot_entropy(entropies, accuracies, title="Entropy Distribution"):
    e = entropies.cpu().numpy()
    a = accuracies.cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(e[a == 1], bins=20, alpha=0.6, color='green', label='Correct', density=True)
    plt.hist(e[a == 0], bins=20, alpha=0.6, color='red', label='Incorrect', density=True)
    plt.xlabel('Shannon Entropy (bits)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_entropy_distribution.png"))
    plt.show()
