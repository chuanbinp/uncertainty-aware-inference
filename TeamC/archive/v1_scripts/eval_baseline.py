# """
# FP16 baseline calibration evaluation for Llama-2 13B.

# Usage:
#     python TeamC/eval_baseline.py --dataset arc_challenge --split validation
#     python TeamC/eval_baseline.py --dataset all --split validation
#     python TeamC/eval_baseline.py --dataset arc_challenge --no_wandb --max_samples 50
# """

# import argparse
# import os

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from eval_common import DATASET_CONFIGS, SEED, run_eval

# MODEL_ID = "meta-llama/Llama-2-13b-hf"


# def load_model(model_id: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
#     model.eval()
#     return model, tokenizer


# def main():
#     parser = argparse.ArgumentParser(description="FP16 baseline eval for Llama-2 13B")
#     parser.add_argument("--dataset", choices=[*DATASET_CONFIGS, "all"], required=True)
#     parser.add_argument("--split", default="validation")
#     parser.add_argument("--max_samples", type=int, default=None)
#     parser.add_argument("--no_wandb", action="store_true")
#     args = parser.parse_args()

#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)

#     datasets_to_run = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]

#     wandb = None
#     # if not args.no_wandb:
#     #     import wandb as _wandb
#     #     wandb = _wandb
#     #     wandb.init(
#     #         project="Uncertainty-Aware-Inference",
#     #         config={
#     #             "model": "llama2-13b",
#     #             "team": "team-c",
#     #             "quant_method": "fp16",
#     #             "precision": "fp16",
#     #             "dataset": args.dataset,
#     #             "split": args.split,
#     #             "seed": SEED,
#     #         },
#     #     )

#     print(f"Loading {MODEL_ID}...")
#     model, tokenizer = load_model(MODEL_ID)
#     print("Model loaded.")

#     output_dir = os.path.join(os.path.dirname(__file__), "results")

#     run_eval(
#         model=model,
#         tokenizer=tokenizer,
#         datasets_to_run=datasets_to_run,
#         split=args.split,
#         max_samples=args.max_samples,
#         output_dir=output_dir,
#         model_tag="llama2_13b",
#         precision="fp16",
#         quant_method="fp16",
#         wandb=wandb,
#     )

#     # if wandb:
#     #     wandb.finish()

#     print("\nDone. Results saved to", output_dir)


# if __name__ == "__main__":
#     main()
