# Team B — Calibration Evaluation
# Runs one quantization config across hellaswag, triviaqa, pubmedqa.



import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TeamB.configs import MODEL_REGISTRY
from shared.model_loader import load_model, free_model
from shared.eval_utils import run_eval

CONFIG_KEY = "mistral-7B-fp16"
MAX_SAMPLES = None
SEED = 42

OUTPUT_DIR = f"./updated_results/{CONFIG_KEY}"

model, tokenizer = load_model(CONFIG_KEY, MODEL_REGISTRY)
tokenizer.pad_token = tokenizer.eos_token # to avoid warnings about no pad token during evaluation

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=["hellaswag", "triviaqa", "pubmedqa"],
        max_samples=MAX_SAMPLES,
        output_dir=OUTPUT_DIR,
        model_tag="llama2-13b",
        precision=MODEL_REGISTRY[CONFIG_KEY]["quant_type"],
        quant_method=str(MODEL_REGISTRY[CONFIG_KEY]["bits"]) + "bit",
        seed=SEED,
    )
finally:
    free_model(model)

