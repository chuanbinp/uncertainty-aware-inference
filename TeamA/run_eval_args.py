# Team A — Calibration Evaluation (args version)
# Run any config from the registry without editing the script.
#
# Usage:
#   python TeamA/run_eval_args.py --config llama2-7b-awq-int4
#   python TeamA/run_eval_args.py --config llama2-7b-fp16 --samples 100
#   python TeamA/run_eval_args.py --config llama2-7b-gptq-int4 --datasets hellaswag triviaqa
#
# Configs (see TeamA/configs.py):
#   llama2-7b-fp16       FP16 baseline     needs HF_TOKEN
#   llama2-7b-nf4        NF4 bitsandbytes  needs HF_TOKEN
#   llama2-7b-awq-int4   AWQ INT4          no token needed
#   llama2-7b-gptq-int4  GPTQ INT4         no token needed

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(MODEL_REGISTRY.keys()),
                    help="model config key from MODEL_REGISTRY")
parser.add_argument("--samples", type=int, default=1000,
                    help="samples per dataset (default: 1000, use 10 for quick check)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

OUTPUT_DIR = f"TeamA/results/{args.config}"

model, tokenizer = load_model(args.config, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=OUTPUT_DIR,
        model_tag="llama2-7b",
        precision=MODEL_REGISTRY[args.config]["quant_type"],
        quant_method=str(MODEL_REGISTRY[args.config]["bits"]) + "bit",
        seed=args.seed,
    )
finally:
    free_model(model)
