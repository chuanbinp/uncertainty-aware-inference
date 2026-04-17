import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from netcal.metrics import ECE, MCE

from shared.result_format import save_results
from shared.data_loader import DATASET_CONFIGS, load_eval_dataset

MAX_GEN_TOKENS = 32


def eval_multiple_choice(model, tokenizer, examples: list[dict]):
    """Evaluate via conditional log-likelihood scoring over answer choices.

    Adopted from Team C. Uses mean log-likelihood per token instead of sum
    to prevent bias toward shorter choices (length normalization fix).

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
            # Encode choice with a leading space to match how it appears in full_text.
            # Use add_special_tokens=False so BOS is not counted.
            choice_token_len = tokenizer.encode(
                " " + choice, add_special_tokens=False, return_tensors="pt"
            ).shape[1]

            full_text = ex["question"] + " " + choice
            input_ids = tokenizer.encode(full_text, return_tensors="pt")

            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, -2048:]

            input_ids = input_ids.to(next(model.parameters()).device)
            total_tokens += input_ids.shape[1]

            with torch.no_grad():
                logits = model(input_ids).logits.to(torch.float32)

            log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
            targets = input_ids[0, 1:]
            token_log_probs = log_probs[torch.arange(len(targets)), targets]
            # Slice the last choice_token_len tokens — always correct regardless of
            # how the tokenizer handles spaces at the question/choice boundary.
            choice_log_prob = token_log_probs[-choice_token_len:].mean()
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
    """Adopted from Team C, unchanged."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def check_answer(generated: str, gold_answers: list[str]) -> bool:
    """Adopted from Team C (eval_common.py), unchanged."""
    gen_norm = normalize_answer(generated)
    return any(normalize_answer(gold) in gen_norm for gold in gold_answers)


def eval_generative_qa(model, tokenizer, examples: list[dict]):
    """Evaluate via greedy decode.
    FIX APPLIED: Computes sequence-level geometric mean of confidence and average entropy
    rather than just relying on the first token.
    """
    all_confidences = []
    all_accuracies = []
    all_entropies = []
    total_tokens = 0
    start_time = time.time()

    for i, ex in enumerate(examples):
        prompt = f"Q: {ex['question']}\nA:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        if input_ids.shape[1] > 2048 - MAX_GEN_TOKENS:
            input_ids = input_ids[:, -(2048 - MAX_GEN_TOKENS):]

        input_ids = input_ids.to(next(model.parameters()).device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_GEN_TOKENS,
                max_length=None,        # suppress conflict with model's generation_config
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0, prompt_len:]
        num_gen_tokens = len(generated_ids)
        seq_log_prob = 0.0
        seq_entropy = 0.0

        # FIX: Loop through generated sequence to calculate true sequence-level metrics
        for step, token_id in enumerate(generated_ids):
            step_logits = outputs.scores[step][0]
            # Upcast step_logits to float32 to prevent FP16 overflow
            step_probs = F.softmax(step_logits.float(), dim=-1)
            step_log_probs = F.log_softmax(step_logits.float(), dim=-1)

            seq_log_prob += step_log_probs[token_id].item()
            seq_entropy += -(step_probs * step_log_probs.clamp(min=-100)).sum().item()

        # Geometric mean over the sequence
        confidence = torch.exp(torch.tensor(seq_log_prob / num_gen_tokens)).item() if num_gen_tokens > 0 else 0.0
        # Average token entropy across the sequence
        entropy = seq_entropy / num_gen_tokens if num_gen_tokens > 0 else 0.0

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        correct = 1.0 if check_answer(generated_text, ex["gold_answers"]) else 0.0

        total_tokens += outputs.sequences.shape[1]
        all_confidences.append(confidence)
        all_accuracies.append(correct)
        all_entropies.append(entropy)

        if (i + 1) % 100 == 0:
            print(f" [{i+1}/{len(examples)}] acc={sum(all_accuracies)/len(all_accuracies):.3f}")

    elapsed = time.time() - start_time
    tps = total_tokens / elapsed if elapsed > 0 else 0

    return (
        torch.tensor(all_confidences),
        torch.tensor(all_accuracies),
        torch.tensor(all_entropies),
        tps,
    )


def compute_calibration_metrics(confidences, accuracies, entropies, num_bins=15):
    """Compute calibration metrics using netcal for ECE and MCE."""
    conf_np = confidences.cpu().numpy()
    acc_np = accuracies.cpu().numpy().astype(float)

    ece = float(ECE(bins=num_bins).measure(conf_np, acc_np))
    mce = float(MCE(bins=num_bins).measure(conf_np, acc_np))
    brier_score = torch.mean((confidences - accuracies.float()) ** 2).item()

    return {
        "ECE": ece,
        "MCE": mce,
        "Brier_Score": brier_score,
        "Avg_Entropy": entropies.mean().item(),
    }

                          
def plot_reliability_diagram(confidences: torch.Tensor, accuracies: torch.Tensor, save_path: str,
    title: str = "Reliability Diagram", num_bins: int = 15) -> None:
    """Create and save a reliability diagram comparing average confidence and accuracy across bins."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    conf_np = confidences.cpu().numpy()
    acc_np = accuracies.cpu().numpy()
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(conf_np, bins) - 1
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accs[i] = np.mean(acc_np[mask])
            bin_confs[i] = np.mean(conf_np[mask])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    ax.bar(bins[:-1], bin_accs, width=1/num_bins, align='edge',
           alpha=0.5, edgecolor='black', label='Accuracy')
    ax.plot(bin_confs[bin_confs > 0], bin_accs[bin_confs > 0],
            'ro-', label='Confidence')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)
                          
                          
def plot_entropy_distribution(entropies: torch.Tensor, accuracies: torch.Tensor, save_path: str,
    title: str = "Entropy Distribution") -> None:
    """Create and save entropy histograms for correct and incorrect predictions.

    netcal does not provide this plot type; kept as matplotlib.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    e = entropies.cpu().numpy()
    a = accuracies.cpu().numpy()
    finite = np.isfinite(e)
    fig, ax = plt.subplots(figsize=(6, 4))
    correct_e = e[finite & (a == 1)]
    incorrect_e = e[finite & (a == 0)]
    if len(correct_e) > 0:
        ax.hist(correct_e, bins=20, alpha=0.6, color='green', label='Correct', density=True)
    if len(incorrect_e) > 0:
        ax.hist(incorrect_e, bins=20, alpha=0.6, color='red', label='Incorrect', density=True)
    ax.set_xlabel('Shannon Entropy (bits)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)


def run_eval(
    model,
    tokenizer,
    datasets_to_run: list[str],
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
        max_samples:      cap per dataset, None = use all
        output_dir:       where to write .pt and .json files
        model_tag:        e.g. "llama2_7b", used in filenames
        precision:        e.g. "fp16", "int4", "nf4"
        quant_method:     e.g. "fp16", "gptq", "awq", "bitsandbytes"
        seed:             shuffle seed passed to load_eval_dataset
        wandb:            wandb module or None to skip logging
    
    Adopted from Team C.
    """
    for ds_name in datasets_to_run:
        print(f"\n{'='*50}\nEvaluating: {ds_name}\n{'='*50}")

        cfg = DATASET_CONFIGS[ds_name]
        examples = load_eval_dataset( 
            ds_name, max_samples, seed
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
