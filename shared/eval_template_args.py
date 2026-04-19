# Evaluation Template (args version)
# =========================================================
# Same as eval_template.py but config is passed via command-line args
# instead of editing the script. Useful for running multiple configs
# in sequence or from a shell script.
#
# Usage:
#   python TeamX/run_eval.py --config your-model-key --samples 100 --datasets hellaswag triviaqa --seed 42

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

# TODO: replace TeamX with your team folder
from TeamX.configs import MODEL_REGISTRY

parser = argparse.ArgumentParser()
parser.add_argument("--config",   required=True, choices=list(MODEL_REGISTRY.keys()),
                    help="model config key from MODEL_REGISTRY")
parser.add_argument("--samples",  type=int, default=1000,
                    help="samples per dataset (default: 1000, use 10 for quick check)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

OUTPUT_DIR = f"TeamX/results/{args.config}"   # TODO: replace TeamX

run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    name=f"teamX_{args.config}",              # TODO: replace teamX
    config={
        "model":        "your-model-name",    # TODO: e.g. "mistral-7b"
        "team":         "team-x",             # TODO: replace
        "quant_method": MODEL_REGISTRY[args.config]["quant_type"],
        "precision":    str(MODEL_REGISTRY[args.config]["bits"]) + "bit",
        "dataset":      args.datasets,
        "seed":         args.seed,
    },
)

model, tokenizer = load_model(args.config, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=OUTPUT_DIR,
        model_tag="your-model-name",          # TODO: e.g. "mistral-7b"
        quant_method=MODEL_REGISTRY[args.config]["quant_type"],
        precision=str(MODEL_REGISTRY[args.config]["bits"]) + "bit",
        seed=args.seed,
        wandb=wandb,
    )
finally:
    free_model(model)
    run.finish()
