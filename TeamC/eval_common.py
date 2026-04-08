"""
Shared evaluation logic for calibration experiments.

Both eval_baseline.py (FP16) and eval_quantized.py (PTQ) import from here.
"""

import re
import sys
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from TeamA.calibration_eval import compute_calibration_metrics, plot_reliability_diagram, plot_entropy
from shared.result_format import save_results

SEED = 42
MAX_GEN_TOKENS = 32

DATASET_CONFIGS = {
    "arc_challenge": {"type": "multiple_choice", "hf_name": "allenai/ai2_arc", "hf_config": "ARC-Challenge"},
    "hellaswag":     {"type": "multiple_choice", "hf_name": "Rowan/hellaswag",  "hf_config": None},
    "triviaqa":      {"type": "generative",      "hf_name": "trivia_qa",        "hf_config": "rc.nocontext"},
    "nq":            {"type": "generative",      "hf_name": "nq_open",          "hf_config": None},
}


# ── Dataset loading ────────────────────────────────────────────────────

def load_eval_dataset(name: str, split: str, max_samples: int | None = None) -> list[dict]:
    """Load and normalize dataset into uniform format.

    Multiple-choice: {"question": str, "choices": list[str], "gold_index": int}
    Generative QA:   {"question": str, "gold_answers": list[str]}
    """
    cfg = DATASET_CONFIGS[name]
    hf_args = [cfg["hf_name"]]
    if cfg["hf_config"]:
        hf_args.append(cfg["hf_config"])
    raw = load_dataset(*hf_args, split=split)

    examples = []
    for row in raw:
        if name == "arc_challenge":
            labels = row["choices"]["label"]
            gold_idx = labels.index(row["answerKey"])
            examples.append({
                "question": row["question"],
                "choices": row["choices"]["text"],
                "gold_index": gold_idx,
            })
        elif name == "hellaswag":
            examples.append({
                "question": row["ctx"],
                "choices": row["endings"],
                "gold_index": int(row["label"]),
            })
        elif name == "triviaqa":
            examples.append({
                "question": row["question"],
                "gold_answers": row["answer"]["aliases"],
            })
        elif name == "nq":
            examples.append({
                "question": row["question"],
                "gold_answers": row["answer"],
            })

        if max_samples and len(examples) >= max_samples:
            break

    return examples


# ── Multiple-choice evaluation ─────────────────────────────────────────

def eval_multiple_choice(model, tokenizer, examples: list[dict]):
    """Evaluate via conditional log-likelihood scoring over answer choices.

    Returns (confidences, accuracies, entropies, tokens_per_second).
    """
    all_confidences = []
    all_accuracies = []
    all_entropies = []
    total_tokens = 0
    start_time = time.time()

    for i, ex in enumerate(examples):
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
                logits = model(input_ids).logits

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


# ── Generative QA evaluation ──────────────────────────────────────────

def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def check_answer(generated: str, gold_answers: list[str]) -> bool:
    gen_norm = normalize_answer(generated)
    return any(normalize_answer(gold) in gen_norm for gold in gold_answers)


def eval_generative_qa(model, tokenizer, examples: list[dict]):
    """Evaluate via greedy decode + first-token confidence.

    Returns (confidences, accuracies, entropies, tokens_per_second).
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

        first_token_logits = outputs.scores[0][0]
        first_token_probs = F.softmax(first_token_logits, dim=0)
        confidence = first_token_probs.max().item()
        entropy = -(first_token_probs * first_token_probs.log().clamp(min=-100)).sum().item()

        generated_ids = outputs.sequences[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        correct = 1.0 if check_answer(generated_text, ex["gold_answers"]) else 0.0

        total_tokens += outputs.sequences.shape[1]
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


# ── Shared eval loop ──────────────────────────────────────────────────

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
    wandb=None,
):
    """Run calibration eval on specified datasets and save results.

    Args:
        model: loaded HF model
        tokenizer: loaded HF tokenizer
        datasets_to_run: list of dataset keys from DATASET_CONFIGS
        split: dataset split name
        max_samples: cap on examples per dataset (None = all)
        output_dir: where to save .pt/.json results
        model_tag: e.g. "llama2_13b" (used in filenames and JSON)
        precision: e.g. "fp16", "int4", "nf4"
        quant_method: e.g. "fp16", "gptq", "awq", "bitsandbytes"
        wandb: wandb module or None to skip logging
    """
    for ds_name in datasets_to_run:
        print(f"\n{'='*50}")
        print(f"Evaluating: {ds_name}")
        print(f"{'='*50}")

        cfg = DATASET_CONFIGS[ds_name]
        examples = load_eval_dataset(ds_name, split, max_samples)
        print(f"Loaded {len(examples)} examples")

        if cfg["type"] == "multiple_choice":
            confidences, accuracies, entropies, tps = eval_multiple_choice(model, tokenizer, examples)
        else:
            confidences, accuracies, entropies, tps = eval_generative_qa(model, tokenizer, examples)

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

        print(f"Accuracy: {accuracies.mean():.4f}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"Tokens/sec: {tps:.1f}")

        title = f"{model_tag} {quant_method.upper()} {precision} - {ds_name}"
        plot_reliability_diagram(confidences, accuracies, title=title)
        plot_entropy(entropies, accuracies, title=title)

        if wandb:
            wandb.log({
                f"{ds_name}/accuracy": accuracies.mean().item(),
                f"{ds_name}/ECE": metrics["ECE"],
                f"{ds_name}/MCE": metrics["MCE"],
                f"{ds_name}/Brier_Score": metrics["Brier_Score"],
                f"{ds_name}/Avg_Entropy": metrics["Avg_Entropy"],
                f"{ds_name}/tokens_per_second": tps,
            })
