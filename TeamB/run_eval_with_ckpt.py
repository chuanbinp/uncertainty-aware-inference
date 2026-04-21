# Team B — Calibration Evaluation
# Runs one quantization config across hellaswag, triviaqa, pubmedqa.
#
# Usage:
#   python TeamB/run_eval.py --config mistral-7b-fp16
#   python TeamB/run_eval.py --config mistral-7b-awq-int4 --samples 100
#   python TeamB/run_eval.py --config mistral-7b-gptq-int4 --datasets hellaswag triviaqa
#
# On Colab, point --output-root at a Drive path so checkpoints survive disconnects:
#   python TeamB/run_eval.py --config mistral-7b-fp16 \
#       --output-root /content/drive/MyDrive/uai_results
#
import os
import sys
import hashlib
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from TeamB.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(MODEL_REGISTRY.keys()),
                    help="config to run")
parser.add_argument("--samples", type=int, default=None,
                    help="samples per dataset (default: full dataset)")
parser.add_argument("--datasets", nargs="+", default=["hellaswag", "triviaqa", "pubmedqa"],
                    help="datasets to evaluate on")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-root", default="./updated_results",
                    help="root dir for outputs. On Colab, use a Drive path so "
                         "checkpoints survive runtime disconnects.")
parser.add_argument("--save-every", type=int, default=50,
                    help="checkpoint frequency (examples). Lower = less lost "
                         "work on crash, more disk I/O.")
args = parser.parse_args()

config_key = args.config
print(f"\n{'-'*60}")
print(f"Running: {config_key}")
print(f"Datasets: {args.datasets}  |  Samples: {args.samples}")
print(f"Output:   {args.output_root}/{config_key}")
print(f"Save every: {args.save_every} examples")
print(f"{'-'*60}\n")

output_dir = f"{args.output_root}/{config_key}"
os.makedirs(output_dir, exist_ok=True)

# Deterministic wandb run id so a Colab crash + restart rejoins the SAME
# wandb run instead of creating a new one. Keyed on (config, seed, datasets,
# samples) 

run_id_seed = f"{config_key}_{args.seed}_{'-'.join(sorted(args.datasets))}_{args.samples}"
run_id = hashlib.md5(run_id_seed.encode()).hexdigest()[:12]

run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    id=run_id,
    resume="allow",          # resume if id exists, else create
    name=f"team-b_{config_key}",
    config={
        "model":        "mistral-7b",
        "team":         "team-b",
        "quant_method": MODEL_REGISTRY[config_key]["quant_type"],
        "precision":    str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
        "dataset":      args.datasets,
        "seed":         args.seed,
        "save_every":   args.save_every,
    },
)

if wandb.run.resumed:
    print(f"Resumed wandb run {run_id} (picking up from prior state)")
else:
    print(f"Started new wandb run {run_id}")

model, tokenizer = load_model(config_key, MODEL_REGISTRY)
tokenizer.pad_token = tokenizer.eos_token

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=output_dir,
        model_tag="mistral-7b",
        precision=MODEL_REGISTRY[config_key]["quant_type"],
        quant_method=str(MODEL_REGISTRY[config_key]["bits"]) + "bit",
        seed=args.seed,
        wandb=wandb,
        save_every=args.save_every,
    )
finally:
    free_model(model)
    run.finish()

print(f"\nResults saved to {output_dir}/")