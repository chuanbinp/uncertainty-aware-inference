# Team A — Calibration Evaluation
#
# Usage:
#   python TeamA/run_eval_args.py --config llama1-7b-awq-int4
#   python TeamA/run_eval_args.py --config llama1-7b-fp16 --samples 100
#   python TeamA/run_eval_args.py --config llama1-7b-gptq-int4 --datasets hellaswag triviaqa
#   python TeamA/run_eval_args.py --all                          # run all available configs
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

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--config", choices=list(MODEL_REGISTRY.keys()),
                   help="single config to run")
group.add_argument("--all", action="store_true",
                   help="run all configs in MODEL_REGISTRY")
parser.add_argument("--samples",  type=int, default=1000,
                    help="samples per dataset (default: 1000, use 10 for quick check)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

configs_to_run = list(MODEL_REGISTRY.keys()) if args.all else [args.config]

for config_key in configs_to_run:
    print(f"\n{'-'*60}")
    print(f"Running: {config_key}")
    print(f"Datasets: {args.datasets}  |  Samples: {args.samples}")
    print(f"{'-'*60}\n")

    output_dir = f"TeamA/results/{config_key}"
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model(config_key, MODEL_REGISTRY)

    try:
        run_eval(
            model=model,
            tokenizer=tokenizer,
            datasets_to_run=args.datasets,
            max_samples=args.samples,
            output_dir=output_dir,
            model_tag="llama1-7b",
            precision=MODEL_REGISTRY[config_key]["quant_type"],
            quant_method=str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
            seed=args.seed,
        )
    finally:
        free_model(model)

    print(f"\nResults saved to {output_dir}/")