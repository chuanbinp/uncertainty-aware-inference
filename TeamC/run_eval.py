# Team C — Calibration Evaluation
# Runs one quantization config across hellaswag, triviaqa, pubmedqa.
#
# Configs (see TeamA/configs.py):
#   llama2-13b-fp16       FP16 baseline     needs HF_TOKEN
#   llama2-13b-nf4        NF4 bitsandbytes  needs HF_TOKEN
#   llama2-13b-awq-int4   AWQ INT4          no token needed
#   llama2-13b-gptq-int4  GPTQ INT4         no token needed
#   llama2-13b-gptq-int8  GPTQ INT8         no token needed
#
# Usage:
#   python TeamC/run_eval.py
#   HF_TOKEN=hf_... python TeamC/run_eval.py   # for gated models


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamC.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval
import wandb

CONFIG_KEY = "llama2-13b-nf4"
MAX_SAMPLES = None
SEED = 42

OUTPUT_DIR = f"./full_results/{CONFIG_KEY}"

run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    name=f"teamC_{CONFIG_KEY}",
    config={
        "model": "llama2-13b",
        "team": "team-C",
        "precision": MODEL_REGISTRY[CONFIG_KEY]["quant_type"] ,
        "quant_method": str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        "dataset": ["hellaswag, pubmedqa"],
        "seed": SEED
    }
)


model, tokenizer = load_model(CONFIG_KEY, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=["hellaswag", "pubmedqa"],
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
        model_tag="llama2-13b",
        precision=MODEL_REGISTRY[CONFIG_KEY]["quant_type"],
        quant_method=str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        seed=SEED,
        wandb=wandb
    )
finally:
    free_model(model)
    run.finish()
