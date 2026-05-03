# TeamA/run_eval_kd.py
# Evaluate a KD-trained model (NF4 base + LoRA adapter).
#
# Single-temperature mode:
#   python TeamA/run_eval_kd.py --config llama1-7b-nf4-kd --temperature 4.0
#
# Dual-temperature mode:
#   python TeamA/run_eval_kd.py --config llama1-7b-nf4-kd --teacher-temp 4.0 --student-temp 2.0
#
# Requires adapter produced by kd_train.py at:
#   TeamA/results/kd/{config}/T{t}/adapter/          (single)
#   TeamA/results/kd/{config}/Tt{t}_Ts{s}/adapter/   (dual)

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from peft import PeftModel

from TeamA.configs import KD_REGISTRY
from shared.model_loader import free_model, load_model
from shared.eval_utils import run_eval

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(KD_REGISTRY.keys()))
parser.add_argument("--temperature", type=float, default=2.0,
                    help="symmetric temperature used during training (single mode)")
parser.add_argument("--teacher-temp", type=float, default=None,
                    help="teacher temperature used during training (dual mode)")
parser.add_argument("--student-temp", type=float, default=None,
                    help="student temperature used during training (dual mode)")
parser.add_argument("--samples", type=int,   default=None)
parser.add_argument("--datasets", nargs="+",  default=["hellaswag", "triviaqa", "pubmedqa"])
parser.add_argument("--seed", type=int,   default=42)
parser.add_argument("--no-wandb", action="store_true")
args = parser.parse_args()

if (args.teacher_temp is None) != (args.student_temp is None):
    parser.error("--teacher-temp and --student-temp must be set together")

# Run tag (must match kd_train.py)
if args.teacher_temp is not None:
    T_t, T_s = args.teacher_temp, args.student_temp
    run_tag = f"Tt{T_t:g}_Ts{T_s:g}"
    mode = "dual"
else:
    T_t = T_s = args.temperature
    run_tag = f"T{args.temperature:g}"
    mode = "single"

kd_cfg = KD_REGISTRY[args.config]
adapter_dir = f"TeamA/results/kd/{args.config}/{run_tag}/adapter"
output_dir = f"TeamA/results/kd/{args.config}/{run_tag}"

_student_registry = {
    kd_cfg["student"]: {
        "hf_id": kd_cfg["student_hf_id"],
        "quant_type": kd_cfg["student_quant_type"],
        "bits": kd_cfg["student_bits"],
        "gptq_revision": kd_cfg.get("student_gptq_revision"),
    }
}

if not os.path.exists(adapter_dir):
    raise FileNotFoundError(
        f"Adapter not found at {adapter_dir}. Run kd_train.py first."
    )

os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"Evaluating KD model: {args.config}  [{run_tag}]")
print(f"Adapter: {adapter_dir}")
print(f"{'='*60}\n")

# Start wandb
if not args.no_wandb:
    run = wandb.init(
        entity="Uncertainty_Aware_Inference_Lab",
        project="UAI_Project",
        name=f"team-a_kd-eval_{args.config}_{run_tag}",
        config={
            "model": kd_cfg["student_hf_id"],
            "team": "team-a",
            "quant_method": kd_cfg["student_quant_type"],
            "precision": str(kd_cfg["student_bits"]) + "bit",
            "kd": True,
            "kd_teacher": kd_cfg["teacher"],
            "mode": mode,
            "T_teacher": T_t,
            "T_student": T_s,
            "dataset": args.datasets,
            "seed": args.seed,
        },
    )

# Evaluate
model, tokenizer = load_model(kd_cfg["student"], _student_registry)
model = PeftModel.from_pretrained(model, adapter_dir)
model.eval()

try:
    run_eval(
        model=model,
        tokenizer=tokenizer,
        datasets_to_run=args.datasets,
        max_samples=args.samples,
        output_dir=output_dir,
        model_tag=f"{kd_cfg['student']}-kd-{run_tag}",
        quant_method=kd_cfg["student_quant_type"],
        precision=str(kd_cfg["student_bits"]) + "bit",
        seed=args.seed,
        wandb=None if args.no_wandb else wandb,
    )
finally:
    free_model(model)
    if not args.no_wandb:
        run.finish()

print(f"\nResults saved to {output_dir}/")