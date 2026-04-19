# Evaluation Template
# ==========================================
# Copy this into team folder (e.g. TeamB/run_eval.py), fill in the TODOs,
# and run from there. See TeamA/run_eval.py for a working example.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

# TODO: import your team's MODEL_REGISTRY. See TeamA/configs.py for an example.
from TeamX.configs import MODEL_REGISTRY

CONFIG_KEY  = "your-model-config-key"   # TODO: must match a key in MODEL_REGISTRY
MAX_SAMPLES = 1000
SEED        = 42

OUTPUT_DIR  = f"TeamX/results/{CONFIG_KEY}"   # TODO: replace TeamX

run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    name=f"teamX_{CONFIG_KEY}",              # TODO: replace teamX
    config={
        "model":        "your-model-name",   # TODO: e.g. "mistral-7b"
        "team":         "team-x",            # TODO: replace
        "quant_method": MODEL_REGISTRY[CONFIG_KEY]["quant_type"],
        "precision":    str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        "dataset":      ["hellaswag", "triviaqa", "pubmedqa"],
        "seed":         SEED,
    },
)

model, tokenizer = load_model(CONFIG_KEY, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=["hellaswag", "triviaqa", "pubmedqa"],
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
        model_tag="your-model-name",         # TODO: e.g. "mistral-7b"
        quant_method=MODEL_REGISTRY[CONFIG_KEY]["quant_type"],
        precision=str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        seed=SEED,
        wandb=wandb,
    )
finally:
    free_model(model)
    run.finish()
