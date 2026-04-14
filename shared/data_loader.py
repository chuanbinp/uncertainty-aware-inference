from datasets import load_dataset

DATASET_CONFIGS = {
    "hellaswag": {
        "type": "multiple_choice",
        "hf_name": "Rowan/hellaswag",
        "hf_config": None,
        "split": "validation",
    },
    "triviaqa": {
        "type": "generative",
        "hf_name": "trivia_qa",
        "hf_config": "rc.nocontext",
        "split": "validation",
    },
    "pubmedqa": {
        "type": "multiple_choice",
        "hf_name": "pubmed_qa",
        "hf_config": "pqa_labeled",
        "split": "train",
    },
}

def load_eval_dataset(name: str, max_samples: int | None = None, seed: int = 42) -> list[dict]:
    """Load and normalize dataset into uniform format.

    Multiple-choice: {"question": str, "choices": list[str], "gold_index": int}
    Generative QA:   {"question": str, "gold_answers": list[str]}

    Structure adopted from Team C.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Valid options: {list(DATASET_CONFIGS)}")
    
    cfg = DATASET_CONFIGS[name]
    hf_args = [cfg["hf_name"]]
    if cfg["hf_config"]:
        hf_args.append(cfg["hf_config"])
    
    split = cfg["split"]
    raw = load_dataset(*hf_args, split=split)
    raw = raw.shuffle(seed=seed)

    if max_samples:
        actual = min(max_samples, len(raw))
        if actual < max_samples:
            print(f"Warning: {name} only has {actual} samples, "
                  f"requested {max_samples}")
        raw = raw.select(range(actual))

    examples = []
    for row in raw:
        if name == "hellaswag":
            examples.append({
                "question": row["ctx"],
                "choices": row["endings"],
                "gold_index": int(row["label"]),
            })
        elif name == "triviaqa":
            aliases = row["answer"]["aliases"] or row["answer"]["normalized_aliases"]
            if not aliases:
                continue
            examples.append({
                "question": row["question"],
                "gold_answers": aliases,
            })
        elif name == "pubmedqa":
            examples.append({
                "question": row["question"],
                "choices": ["yes", "no", "maybe"],
                "gold_index": ["yes", "no", "maybe"].index(
                    row["final_decision"]
                ),
            })

    return examples
