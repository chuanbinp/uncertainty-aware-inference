import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from shared.result_format import save_results
from shared.data_loader import DATASET_CONFIGS, load_eval_dataset

MAX_GEN_TOKENS = 32


def eval_multiple_choice(model, tokenizer, examples: list[dict]):
    """Evaluate via conditional log-likelihood scoring over answer choices.

    Returns (confidences, accuracies, entropies, tokens_per_second).
    """
    all_confidences, all_accuracies, all_entropies = [], [], []
    total_tokens = 0
    start_time = time.time()

    for i, ex in enumerate(examples):
        if not ex.get("choices"):
            continue
        
        choice_logprobs = []
        for choice in ex["choices"]:
            full_text = ex["question"] + " " + choice
            input_ids = tokenizer.encode(full_text, return_tensors="pt")
            q_ids = tokenizer.encode(ex["question"] + " ", return_tensors="pt")
            choice_start = q_ids.shape[1]

            if input_ids.shape[1] > 2048:
                overflow = input_ids.shape[1] - 2048
                input_ids = input_ids[:, -2048:]
                choice_start = max(0, choice_start - overflow)

            input_ids = input_ids.to(model.device)
            total_tokens += input_ids.shape[1]

            with torch.no_grad():
                logits = model(input_ids).logits.to(torch.float32)

            log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
            targets = input_ids[0, 1:]
            token_log_probs = log_probs[torch.arange(len(targets)), targets]
            choice_log_prob = token_log_probs[choice_start - 1:].sum()
            choice_logprobs.append(choice_log_prob)

        choice_logprobs = torch.stack(choice_logprobs)
        probs = F.softmax(choice_logprobs, dim=0)
        confidence = probs.max().item()
        prediction = probs.argmax().item()
        entropy = -(probs * probs.log().clamp(min=-100)).sum().item()
        correct = 1.0 if prediction == ex["gold_index"] else 0.0

        all_confidences.append(confidence)
        all_accuracies.append(correct)
        all_entropies.append(entropy)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(examples)}] acc={sum(all_accuracies)/len(all_accuracies):.3f}")

    elapsed = time.time() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0

    return (
        torch.tensor(all_confidences),
        torch.tensor(all_accuracies),
        torch.tensor(all_entropies),
        tps,
    )


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def check_answer(generated: str, gold_answers: list[str]) -> bool:
    gen_norm = normalize_answer(generated)
    return any(normalize_answer(gold) in gen_norm for gold in gold_answers)


def eval_generative_qa(model, tokenizer, examples: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Evaluate via greedy decode + first-token confidence.

    Returns (confidences, accuracies, entropies, tokens_per_second).
    """
    all_confidences, all_accuracies, all_entropies = [], [], []
    total_tokens = 0
    start_time = time.time()

    for i, ex in enumerate(examples):
        prompt = f"Q: {ex['question']}\nA:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > 2048 - MAX_GEN_TOKENS:
            input_ids = input_ids[:, -(2048 - MAX_GEN_TOKENS):]
        input_ids = input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_GEN_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        first_token_logits = outputs.scores[0][0].to(torch.float32)
        first_token_probs = F.softmax(first_token_logits, dim=0)
        confidence = first_token_probs.max().item()
        entropy = -(first_token_probs * 
                    first_token_probs.log().clamp(min=-100)).sum().item()

        generated_ids = outputs.sequences[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        correct = 1.0 if check_answer(generated_text, ex["gold_answers"]) else 0.0

        total_tokens += outputs.sequences.shape[1]
        all_confidences.append(confidence)
        all_accuracies.append(correct)
        all_entropies.append(entropy)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(examples)}] "
                  f"acc={sum(all_accuracies)/len(all_accuracies):.3f}")

    elapsed = time.time() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    return (
        torch.tensor(all_confidences),
        torch.tensor(all_accuracies),
        torch.tensor(all_entropies),
        tps,
    )

                        
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

                          
def plot_reliability_diagram(confidences: torch.Tensor, accuracies: torch.Tensor, save_path: str,
    title: str = "Reliability Diagram", num_bins: int = 15) -> None:
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
    plt.bar(bins[:-1], bin_accs, width=1/num_bins, align='edge',
            alpha=0.5, edgecolor='black', label='Accuracy')
    plt.plot(bin_confs[bin_confs > 0], bin_accs[bin_confs > 0],
             'ro-', label='Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
                          
                          
def plot_entropy_distribution(entropies: torch.Tensor, accuracies: torch.Tensor, save_path: str, 
    title: str = "Entropy Distribution",) -> None:
    """Create and save entropy histograms for correct and incorrect predictions."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    e = entropies.cpu().numpy()
    a = accuracies.cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(e[a == 1], bins=20, alpha=0.6, color='green',
             label='Correct', density=True)
    plt.hist(e[a == 0], bins=20, alpha=0.6, color='red',
             label='Incorrect', density=True)
    plt.xlabel('Shannon Entropy (bits)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def run_eval(
    model,
    tokenizer,
    datasets_to_run: list[str],
    split: str,
    max_samples: int | None,
    output_dir: str,
    model_tag: str,
    precision: str,
    quant_method: str,
    seed: int = 42,
    wandb=None,
) -> None:
    """Run full calibration eval on specified datasets and save results.

    Args:
        model:            loaded HF model, already on GPU
        tokenizer:        loaded HF tokenizer
        datasets_to_run:  list of keys from DATASET_CONFIGS
        split:            HuggingFace split name e.g. "validation"
        max_samples:      cap per dataset, None = use all
        output_dir:       where to write .pt and .json files
        model_tag:        e.g. "llama2_7b", used in filenames
        precision:        e.g. "fp16", "int4", "nf4"
        quant_method:     e.g. "fp16", "gptq", "awq", "bitsandbytes"
        seed:             shuffle seed passed to load_eval_dataset
        wandb:            wandb module or None to skip logging
    """
    for ds_name in datasets_to_run:
        print(f"\n{'='*50}\nEvaluating: {ds_name}\n{'='*50}")

        cfg = DATASET_CONFIGS[ds_name]
        examples = load_eval_dataset( 
            ds_name, split, max_samples, seed
        )
        print(f"Loaded {len(examples)} examples")

        if cfg["type"] == "multiple_choice":
            confidences, accuracies, entropies, tps = eval_multiple_choice(
                model, tokenizer, examples
            )
        else:
            confidences, accuracies, entropies, tps = eval_generative_qa(
                model, tokenizer, examples
            )

        if len(confidences) == 0:
            print(f"No valid predictions for {ds_name}, skipping.")
            continue

        metrics = compute_calibration_metrics(confidences, accuracies, entropies)

        save_results(
            model=model_tag,
            precision=precision,
            quant_method=quant_method,
            dataset=ds_name,
            confidences=confidences,
            accuracies=accuracies,
            entropies=entropies,
            metrics=metrics,
            tokens_per_second=tps,
            output_dir=output_dir,
        )

        print(f"Accuracy:    {accuracies.mean():.4f}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"Tokens/sec:  {tps:.1f}")

        # Plots
        prefix = f"{output_dir}/{model_tag}_{quant_method}_{precision}_{ds_name}"
        plot_reliability_diagram(
            confidences, accuracies,
            save_path=f"{prefix}_reliability.png",
            title=f"{model_tag} {quant_method.upper()} {precision} - {ds_name}",
        )
        plot_entropy_distribution(
            entropies, accuracies,
            save_path=f"{prefix}_entropy.png",
            title=f"Entropy: {model_tag} {quant_method.upper()} {precision} - {ds_name}",
        )

        if wandb:
            wandb.log({
                f"{ds_name}/accuracy":          accuracies.mean().item(),
                f"{ds_name}/ECE":               metrics["ECE"],
                f"{ds_name}/MCE":               metrics["MCE"],
                f"{ds_name}/Brier_Score":       metrics["Brier_Score"],
                f"{ds_name}/Avg_Entropy":       metrics["Avg_Entropy"],
                f"{ds_name}/tokens_per_second": tps,
            })