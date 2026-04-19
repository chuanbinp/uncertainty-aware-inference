# Team A — Calibration Evaluation
# Runs one quantization config across hellaswag, triviaqa, pubmedqa.
#
# Configs (see TeamA/configs.py):
#   llama1-7b-fp16       FP16 baseline     no token needed
#   llama1-7b-nf4        NF4 bitsandbytes  no token needed
#   llama1-7b-awq-int4   AWQ INT4          no token needed
#   llama1-7b-gptq-int4  GPTQ INT4         no token needed
#   llama1-7b-gptq-int8  GPTQ INT8         no token needed
#
# Usage:
#   python TeamA/run_eval.py
#   HF_TOKEN=hf_... python TeamA/run_eval.py   # for gated models


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

CONFIG_KEY = "llama1-7b-awq-int4"
MAX_SAMPLES = 1000
SEED = 42

OUTPUT_DIR = f"TeamA/results/{CONFIG_KEY}"

model, tokenizer = load_model(CONFIG_KEY, MODEL_REGISTRY)

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=["hellaswag", "triviaqa", "pubmedqa"],
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
        model_tag="llama1-7b",
        precision=MODEL_REGISTRY[CONFIG_KEY]["quant_type"],
        quant_method=str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        seed=SEED,
    )
finally:
    free_model(model)
