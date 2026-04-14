# Evaluation Template (args version)
# =========================================================
# Same as eval_template.py but config is passed via command-line args
# instead of editing the script. Useful for running multiple configs
# in sequence or from a shell script.
#
# Usage:
#   python TeamX/run_eval.py --config llama2-7b-fp16 --samples 100 --datasets hellaswag triviaqa --seed 42

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

# TODO: replace TeamX with your team folder
from TeamX.configs import MODEL_REGISTRY

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(MODEL_REGISTRY.keys()),
                    help="model config key from MODEL_REGISTRY")
parser.add_argument("--samples", type=int, default=1000,
                    help="samples per dataset (default: 1000, use 10 for quick check)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

OUTPUT_DIR = f"TeamX/results/{args.config}"   # TODO: replace TeamX

model, tokenizer = load_model(args.config, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=OUTPUT_DIR,
        model_tag="your-model-name",        # TODO: e.g. "mistral-7b"
        precision=MODEL_REGISTRY[args.config]["quant_type"],
        quant_method=str(MODEL_REGISTRY[args.config]["bits"]) + "bit",
        seed=args.seed,
    )
finally:
    free_model(model)
