# Team A — Calibration Evaluation
# Runs one quantization config across hellaswag, triviaqa, pubmedqa.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamA.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

CONFIG_KEY = "llama1-7b-awq-int4"
MAX_SAMPLES = None   # full dataset; set to e.g. 10 for a quick check
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
        quant_method=MODEL_REGISTRY[CONFIG_KEY]["quant_type"],
        precision=str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        seed=SEED,
    )
finally:
    free_model(model)
