# Team A — Calibration Evaluation
#
# Usage:
#   python TeamA/run_eval_args.py --config llama1-7b-awq-int4
#   python TeamA/run_eval_args.py --config llama1-7b-fp16 --samples 100
#   python TeamA/run_eval_args.py --config llama1-7b-gptq-int4 --datasets hellaswag triviaqa
#   python TeamA/run_all.sh                                      # run all configs
#
# Configs (see TeamA/configs.py): (no token needed for all)
#   llama1-7b-fp16       FP16 baseline
#   llama1-7b-nf4        NF4 bitsandbytes
#   llama1-7b-awq-int4   AWQ INT4
#   llama1-7b-gptq-int4  GPTQ INT4
#   llama1-7b-gptq-int8  GPTQ INT8
#

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(MODEL_REGISTRY.keys()),
                   help="config to run")
parser.add_argument("--samples", type=int, default=None,
                    help="samples per dataset (default: full dataset, use 10 for quick check)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

config_key = args.config

print(f"\n{'-'*60}")
print(f"Running: {config_key}")
print(f"Datasets: {args.datasets}  |  Samples: {args.samples}")
print(f"{'-'*60}\n")

output_dir = f"TeamA/results/{config_key}"
os.makedirs(output_dir, exist_ok=True)

run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    name=f"team-a_{config_key}",
    config={
        "model":        "llama1-7b",
        "team":         "team-a",
        "quant_method": MODEL_REGISTRY[config_key]["quant_type"],
        "precision":    str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
        "dataset":      args.datasets,
        "seed":         args.seed,
    },
)

model, tokenizer = load_model(config_key, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=output_dir,
        model_tag="llama1-7b",
        quant_method=MODEL_REGISTRY[config_key]["quant_type"],
        precision=str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
        seed=args.seed,
        wandb=wandb,
    )
finally:
    free_model(model)
    run.finish()

print(f"\nResults saved to {output_dir}/")