# TeamA/kd_train.py
# Knowledge Distillation: FP16 teacher → NF4 student (LoRA adapters)
# Asymmetric-temperature KD loss (Jin et al. NeurIPS 2022), C4 training data.
#
# Single-temperature mode:
#   python TeamA/kd_train.py --config llama1-7b-nf4-kd --temperature 4.0
#
# Dual-temperature mode:
#   python TeamA/kd_train.py --config llama1-7b-nf4-kd --teacher-temp 4.0 --student-temp 2.0
#
# Adapter saved to:
#   TeamA/results/kd/{config}/T{t}/adapter/ (single)
#   TeamA/results/kd/{config}/Tt{t}_Ts{s}/adapter/ (dual)
#
# NF4 only — AWQ/GPTQ do not support LoRA backpropagation.

import os
import sys
import argparse
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from TeamA.configs import KD_REGISTRY
from shared.model_loader import free_model, load_model

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, choices=list(KD_REGISTRY.keys()))
parser.add_argument("--temperature", type=float, default=2.0,
                    help="symmetric temperature (single mode)")
parser.add_argument("--teacher-temp", type=float, default=None,
                    help="teacher temperature (dual mode)")
parser.add_argument("--student-temp", type=float, default=None,
                    help="student temperature (dual mode)")
parser.add_argument("--samples", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--grad-accum", type=int, default=16)
parser.add_argument("--max-len", type=int, default=1024)
parser.add_argument("--alpha", type=float, default=None,
                    help="KL weight (overrides config); CE weight = 1-alpha")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Validate dual-mode args
if (args.teacher_temp is None) != (args.student_temp is None):
    parser.error("--teacher-temp and --student-temp must be set together")

kd_cfg     = KD_REGISTRY[args.config]
quant_type = kd_cfg["student_quant_type"]
alpha      = args.alpha if args.alpha is not None else kd_cfg.get("alpha", 0.7)

if quant_type in ("awq", "gptq"):
    nf4_config = args.config.replace("awq-int4", "nf4").replace("gptq-int4", "nf4")
    raise ValueError(
        f"{quant_type.upper()} does not support LoRA backpropagation "
        f"({'WQLinear layers' if quant_type == 'awq' else 'TritonV2QuantLinear kernel'} "
        f"incompatible with autograd). Use NF4 instead: {nf4_config}"
    )

# Temperature config
if args.teacher_temp is not None:
    T_t, T_s = args.teacher_temp, args.student_temp
    run_tag  = f"Tt{T_t:g}_Ts{T_s:g}"
    mode     = "dual"
else:
    T_t = T_s = args.temperature
    run_tag   = f"T{args.temperature:g}"
    mode      = "single"

# Paths
adapter_dir = f"TeamA/results/kd/{args.config}/{run_tag}/adapter"
os.makedirs(adapter_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"KD Training: {kd_cfg['student']} ← {kd_cfg['teacher']}")
print(f"Mode: {mode}  |  T_teacher: {T_t}  |  T_student: {T_s}")
print(f"Alpha: {alpha}  |  Samples: {args.samples}  |  Epochs: {args.epochs}")
print(f"Loss: {alpha:.2f} * KL + {1 - alpha:.2f} * CE")
print(f"Adapter → {adapter_dir}")
print(f"{'='*60}\n")

# Registries for shared model_loader

_teacher_registry = {
    kd_cfg["teacher"]: {
        "hf_id":      kd_cfg["teacher_hf_id"],
        "quant_type": "fp16",
        "bits":       16,
    }
}
_student_registry = {
    kd_cfg["student"]: {
        "hf_id":         kd_cfg["student_hf_id"],
        "quant_type":    kd_cfg["student_quant_type"],
        "bits":          kd_cfg["student_bits"],
        "gptq_revision": kd_cfg.get("student_gptq_revision"),
    }
}

# Load Data

class C4Stream(Dataset):
    """Stream C4 until n examples pass the minimum-length filter."""
    def __init__(self, tokenizer, n: int, max_len: int):
        self.examples = []
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for sample in ds:
            enc = tokenizer(
                sample["text"], return_tensors="pt",
                truncation=True, max_length=max_len,
            )
            if enc.input_ids.shape[1] < max_len // 4:
                continue
            self.examples.append({
                "input_ids":      enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
            })
            if len(self.examples) >= n:
                break
        print(f"Loaded {len(self.examples)} C4 examples")

    def __len__(self):        return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


def _collate(batch, pad_id: int):
    L   = max(b["input_ids"].shape[0] for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    am  = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        ids[i, :n] = b["input_ids"]
        am[i, :n]  = b["attention_mask"]
    return {"input_ids": ids, "attention_mask": am}

# Loss

def kd_loss(s_logits, t_logits, targets, am, T_t, T_s, alpha, pad_id):
    """Asymmetric-temperature KD loss (Jin et al. NeurIPS 2022).
    T_t == T_s reduces to Hinton 2015: alpha*T^2*KL + (1-alpha)*CE.
    """
    s   = s_logits[:, :-1, :]
    t   = t_logits[:, :-1, :]
    tgt = targets[:, 1:]
    mask = am[:, 1:].bool() & (tgt != pad_id)

    s_lp = F.log_softmax(s / T_s, dim=-1)
    t_p  = F.softmax(t / T_t, dim=-1)
    kl   = (t_p * (t_p.clamp_min(1e-12).log() - s_lp)).sum(dim=-1)
    kl   = kl[mask].mean() * (T_t * T_s)

    ce = F.cross_entropy(
        s.reshape(-1, s.size(-1)), tgt.reshape(-1), reduction="none",
    ).reshape(tgt.shape)
    ce = ce[mask].mean()

    return {
        "loss": alpha * kl + (1 - alpha) * ce,
        "kl":   kl.detach(),
        "ce":   ce.detach(),
    }

# Start wandb
run = wandb.init(
    entity="Uncertainty_Aware_Inference_Lab",
    project="UAI_Project",
    name=f"team-a_kd-train_{args.config}_{run_tag}",
    config={
        "model":        kd_cfg["student_hf_id"],
        "teacher":      kd_cfg["teacher_hf_id"],
        "team":         "team-a",
        "quant_method": kd_cfg["student_quant_type"],
        "precision":    str(kd_cfg["student_bits"]) + "bit",
        "mode":         mode,
        "T_teacher":    T_t,
        "T_student":    T_s,
        "alpha":        alpha,
        "samples":      args.samples,
        "epochs":       args.epochs,
        "lr":           args.lr,
        "grad_accum":   args.grad_accum,
        "lora_r":       kd_cfg["lora_r"],
        "lora_alpha":   kd_cfg["lora_alpha"],
        "seed":         args.seed,
    },
)

# Models
torch.manual_seed(args.seed)

print("Loading teacher (FP16)...")
teacher, tokenizer = load_model(kd_cfg["teacher"], _teacher_registry)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
teacher_device = next(teacher.parameters()).device

print(f"\nLoading student ({quant_type})...")
student, _ = load_model(kd_cfg["student"], _student_registry)
student = prepare_model_for_kbit_training(student, use_gradient_checkpointing=True)
student = get_peft_model(student, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=kd_cfg["lora_r"],
    lora_alpha=kd_cfg["lora_alpha"],
    target_modules=kd_cfg["lora_target_modules"],
    lora_dropout=0.05,
    bias="none",
))
student.train()
student.print_trainable_parameters()
student_device = next(student.parameters()).device

# Data + optimizer
ds = C4Stream(tokenizer, args.samples, args.max_len)
loader = DataLoader(
    ds, batch_size=1, shuffle=True,
    collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
)

total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=args.lr,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.03 * total_steps),
    num_training_steps=total_steps,
)

# Training loop
global_step = 0

for epoch in range(args.epochs):
    total_loss = total_kl = total_ce = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        ids = batch["input_ids"].to(student_device)
        am  = batch["attention_mask"].to(student_device)

        with torch.no_grad():
            t_logits = teacher(
                input_ids=ids.to(teacher_device),
                attention_mask=am.to(teacher_device),
            ).logits.float().to(student_device)

        s_logits = student(input_ids=ids, attention_mask=am).logits.float()

        losses = kd_loss(s_logits, t_logits, ids, am, T_t, T_s, alpha,
                         tokenizer.pad_token_id)

        (losses["loss"] / args.grad_accum).backward()
        total_loss += losses["loss"].item()
        total_kl   += losses["kl"].item()
        total_ce   += losses["ce"].item()

        if (i + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                avg_loss = total_loss / (i + 1)
                avg_kl   = total_kl   / (i + 1)
                avg_ce   = total_ce   / (i + 1)
                print(f"  Epoch {epoch+1} | Step {global_step} | "
                      f"Loss {avg_loss:.4f} | KL {avg_kl:.4f} | CE {avg_ce:.4f}")
                wandb.log({
                    "train/loss": avg_loss,
                    "train/kl":   avg_kl,
                    "train/ce":   avg_ce,
                    "train/lr":   scheduler.get_last_lr()[0],
                    "epoch":      epoch + 1,
                }, step=global_step)

    # Flush remaining gradients
    if len(loader) % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    epoch_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} done | Avg Loss: {epoch_loss:.4f}")
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch + 1})

# Save results

student.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"\nAdapter saved to {adapter_dir}/")

free_model(teacher)
free_model(student)
run.finish()