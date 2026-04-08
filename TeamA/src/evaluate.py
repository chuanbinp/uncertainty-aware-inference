import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from calibration_eval import compute_calibration_metrics, plot_reliability_diagram, plot_entropy_distribution
import json
from pathlib import Path

def save_results(results: dict, output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def evaluate_and_calibrate(model, tokenizer, config_name: str, run_id: str, num_samples: int = 1000, seed: int = 42) -> dict:
    """Evaluate model calibration on a held-out set of samples, returning metrics and optionally saving plots."""
    datasets_to_run = {
        "Reasoning (ARC)": load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
            .shuffle(seed=seed)
            .select(range(num_samples)),
        
        "Factual (TriviaQA)": load_dataset("trivia_qa", "rc", split="validation")
            .shuffle(seed=seed)
            .select(range(num_samples)),
        
        "OOD (PubMedQA)": load_dataset("pubmed_qa", "pqa_labeled", split="train")
            .shuffle(seed=seed)
            .select(range(num_samples)),
    }

    results = {}
    model.eval()

    # ROBUST DEVICE DETECTION: Works for HF, AWQ, and GPTQ wrappers
    current_device = next(model.parameters()).device

    valid_answer_tokens = {
                    tokenizer(" yes",   add_special_tokens=False).input_ids[0],
                    tokenizer(" no",    add_special_tokens=False).input_ids[0],
                    tokenizer(" maybe", add_special_tokens=False).input_ids[0],
                }

    output_dir = f"TeamA/results/{config_name}/{run_id}"

    for task_name, dataset in datasets_to_run.items():
        print(f"\nEvaluating {task_name}...")
        confidences, accuracies, entropies = [], [], []

        for item in tqdm(dataset, desc=task_name):
            if "ARC" in task_name:
                if len(item['choices']['text']) != 4: continue
                prompt = f"Question: {item['question']}\nChoices:\nA. {item['choices']['text'][0]}\nB. {item['choices']['text'][1]}\nC. {item['choices']['text'][2]}\nD. {item['choices']['text'][3]}\nAnswer:"
                target_str = " " + (item['answerKey'] if item['answerKey'] in ['A','B','C','D'] else ['A','B','C','D'][int(item['answerKey'])-1])

            elif "TriviaQA" in task_name:
                prompt = f"Question: {item['question']}\nAnswer:"

            elif "PubMedQA" in task_name:
                context_text = " ".join(item['context']['contexts'])
                prompt = f"Context: {context_text}\nQuestion: {item['question']}\nAnswer (yes/no/maybe):"
                target_str = " " + item['final_decision']

            inputs = tokenizer(prompt, return_tensors="pt").to(current_device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Extract logits
            next_token_logits = outputs.logits[0, -1, :]

            # torch.isfinite catches nan, inf, and -inf.
            if not torch.isfinite(next_token_logits).all():
                continue

            # Cast logits to float32 before softmaxing to ensure numerical stability
            next_token_logits = next_token_logits.to(torch.float32)

            probabilities = F.softmax(next_token_logits, dim=-1)
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-9)).item()
            top1_prob, top1_idx = torch.max(probabilities, dim=-1)
            
            if "ARC" in task_name:
                target_token_id = tokenizer(target_str, add_special_tokens=False).input_ids[0]
                confidence = top1_prob.item()
                is_correct = int(top1_idx.item() == target_token_id)

            elif "TriviaQA" in task_name:
                valid_first_tokens = set()
                for alias in item['answer']['normalized_aliases']:
                    tokens = tokenizer(" " + alias, add_special_tokens=False).input_ids
                    if tokens:
                        valid_first_tokens.add(tokens[0])

                confidence = sum(probabilities[tid].item() for tid in valid_first_tokens)
                is_correct = int(top1_idx.item() in valid_first_tokens)

            elif "PubMedQA" in task_name:
                target_token_id = tokenizer(target_str, add_special_tokens=False).input_ids[0]
                confidence = sum(probabilities[tid].item() for tid in valid_answer_tokens)
                is_correct = int(top1_idx.item() == target_token_id)

            confidences.append(confidence)
            accuracies.append(int(is_correct))
            entropies.append(entropy)

        # Check if we successfully processed samples (prevents empty list errors)
        if len(confidences) == 0:
            print(f"Failed to generate valid predictions for {task_name}. Skipping metrics.")
            continue

        # Move tensors to the dynamically detected device
        conf_t = torch.tensor(confidences, device=current_device)
        acc_t = torch.tensor(accuracies, device=current_device)
        ent_t = torch.tensor(entropies, device=current_device)

        metrics = compute_calibration_metrics(conf_t, acc_t, ent_t)
        results[task_name] = metrics

        print(f"\n{task_name} -> ECE: {metrics['ECE']:.4f} | Brier: {metrics['Brier_Score']:.4f} | Avg Entropy: {metrics['Avg_Entropy']:.4f}")

        safe_name = config_name.replace(" ", "_")
        safe_task = task_name.split("(")[-1].rstrip(")")
        rel_path = f"{output_dir}/{safe_name}_{safe_task}_reliability.png"
        ent_path = f"{output_dir}/{safe_name}_{safe_task}_entropy.png"
        plot_reliability_diagram(conf_t, acc_t, title=f"{config_name} - {task_name}", save_path=rel_path)
        plot_entropy_distribution(ent_t, acc_t, title=f"Entropy: {config_name} - {task_name}", save_path=ent_path)

    save_results(results, f"TeamA/results/{config_name}/{run_id}/metrics.json")
    return results