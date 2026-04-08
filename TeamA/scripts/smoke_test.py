import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from evaluate import evaluate_and_calibrate

MODEL_ID = "facebook/opt-125m"
NUM_SAMPLES = 5
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}")

print("\nRunning smoke test with 5 samples per dataset...")
results = evaluate_and_calibrate(
    model=model,
    tokenizer=tokenizer,
    config_name="opt-125m-fp16",
    run_id=RUN_ID,
    num_samples=NUM_SAMPLES,
    seed=42,
)

print("\n--- Results ---")
for task, metrics in results.items():
    print(f"{task}: ECE={metrics['ECE']:.4f} Brier={metrics['Brier_Score']:.4f}")

print("\nSmoke test complete. Check results/opt-125m-fp16/ for output files.")