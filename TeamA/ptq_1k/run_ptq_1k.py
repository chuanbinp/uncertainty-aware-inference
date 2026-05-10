# TeamA/ptq_1k/run_ptq_1k.py
#
# Re-evaluates NF4 PTQ baselines on 1,000 samples for all three models.
# Fixes the sample-size mismatch between full-split PTQ baselines and
# 1,000-sample KD evaluations (mentor feedback #2).
#
# Results go to:  TeamA/results/ptq_1k/{config}/
# Existing full-split results in TeamA/results/ are NOT touched.
#
# Usage:
#   python TeamA/ptq_1k/run_ptq_1k.py                            # all three
#   python TeamA/ptq_1k/run_ptq_1k.py --config llama1-7b-nf4
#   python TeamA/ptq_1k/run_ptq_1k.py --config mistral-7b-nf4
#   python TeamA/ptq_1k/run_ptq_1k.py --config llama2-13b-nf4
#
# Llama-2 13B is a gated model — set HF_TOKEN in your environment first:
#   export HF_TOKEN=hf_...

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gc
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from shared.eval_utils import run_eval


def load_nf4(hf_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

SAMPLES = 1000
DATASETS = ["hellaswag", "triviaqa", "pubmedqa"]
SEED = 42

# Self-contained registry — no changes to TeamA/configs.py needed.
# File output: {model_tag}_{quant_method}_{precision}_{dataset}.json
# e.g. llama1-7b_nf4_4bit_hellaswag.json  (matches TeamA naming convention)
REGISTRY = {
    "llama1-7b-nf4": {
        "hf_id":      "huggyllama/llama-7b",
        "quant_type": "nf4",
        "bits":       4,
        "model_tag":  "llama1-7b",
    },
    "mistral-7b-nf4": {
        "hf_id":      "mistralai/Mistral-7B-Instruct-v0.2",
        "quant_type": "nf4",
        "bits":       4,
        "model_tag":  "mistral-7b",
    },
    "llama2-13b-nf4": {
        "hf_id":      "meta-llama/Llama-2-13b-hf",
        "quant_type": "nf4",
        "bits":       4,
        "model_tag":  "llama2-13b",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    choices=list(REGISTRY.keys()),
    default=None,
    help="which model to evaluate (default: run all three sequentially)",
)
args = parser.parse_args()

configs_to_run = [args.config] if args.config else list(REGISTRY.keys())

for config_key in configs_to_run:
    entry = REGISTRY[config_key]
    output_dir = f"TeamA/ptq_1k/results/{config_key}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Config  : {config_key}")
    print(f"Model   : {entry['hf_id']}")
    print(f"Samples : {SAMPLES} per dataset")
    print(f"Output  : {output_dir}/")
    print(f"{'='*60}\n")

    run = wandb.init(
        entity="Uncertainty_Aware_Inference_Lab",
        project="UAI_Project",
        name=f"team-a_ptq-1k_{config_key}",
        config={
            "model":        entry["hf_id"],
            "team":         "team-a",
            "quant_method": entry["quant_type"],
            "precision":    str(entry["bits"]) + "bit",
            "samples":      SAMPLES,
            "dataset":      DATASETS,
            "seed":         SEED,
            "note":         "1k-sample PTQ baseline for matched KD comparison",
        },
    )

    model, tokenizer = load_nf4(entry["hf_id"])

    try:
        run_eval(
            model=model,
            tokenizer=tokenizer,
            datasets_to_run=DATASETS,
            max_samples=SAMPLES,
            output_dir=output_dir,
            model_tag=entry["model_tag"],
            quant_method=entry["quant_type"],
            precision=str(entry["bits"]) + "bit",
            seed=SEED,
            wandb=wandb,
        )
    finally:
        free_model(model)
        run.finish()

    print(f"\nDone. Results saved to {output_dir}/")
