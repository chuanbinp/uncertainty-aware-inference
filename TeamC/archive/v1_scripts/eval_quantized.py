# """
# Quantized model calibration evaluation for Llama-2 13B.

# Loads a quantized model (GPTQ, AWQ, or bitsandbytes NF4) and runs the same
# calibration evaluation pipeline as the FP16 baseline.

# Usage:
#     python TeamC/eval_quantized.py --config gptq_int4 --dataset all --split validation
#     python TeamC/eval_quantized.py --config bnb_nf4 --dataset arc_challenge --max_samples 50 --no_wandb
# """

# import argparse
# import os

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from eval_common import DATASET_CONFIGS, SEED, run_eval
# from configs import QUANT_CONFIGS


# def load_tokenizer(model_id: str):
#     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer


# def load_gptq_model(cfg: dict):
#     """Load a GPTQ-quantized model via transformers native GPTQConfig."""
#     # from transformers import GPTQConfig
#     from gptqmodel import GPTQModel

#     model = GPTQModel.load(
#         cfg["hf_model_id"],
#         revision=cfg.get("gptq_revision"),
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )
#     model.eval()
#     tokenizer = load_tokenizer(cfg["hf_model_id"])
#     return model, tokenizer


# def load_awq_model(cfg: dict):
#     """Load an AWQ-quantized model (quantization config embedded in model)."""

#     from awq import AutoAWQForCausalLM
 
#     model = AutoAWQForCausalLM.from_quantized(
#         cfg["hf_model_id"],
#         fuse_layers=False,
#         torch_dtype=torch.float16,
#     )

#     model = model.model
#     model.eval()
#     tokenizer = load_tokenizer(cfg["hf_model_id"])
#     return model, tokenizer


# def load_bnb_model(cfg: dict):
#     """Load a model with bitsandbytes NF4 quantization."""
#     from transformers import BitsAndBytesConfig

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         cfg["hf_model_id"],
#         quantization_config=bnb_config,
#         device_map="auto",
#     )
#     model.eval()
#     tokenizer = load_tokenizer(cfg["hf_model_id"])
#     return model, tokenizer


# LOADERS = {
#     "gptq": load_gptq_model,
#     "awq": load_awq_model,
#     "bitsandbytes": load_bnb_model,
# }


# def main():
#     parser = argparse.ArgumentParser(description="Quantized eval for Llama-2 13B")
#     parser.add_argument("--config", choices=list(QUANT_CONFIGS), required=True)
#     parser.add_argument("--dataset", choices=[*DATASET_CONFIGS, "all"], required=True)
#     parser.add_argument("--split", default="validation")
#     parser.add_argument("--max_samples", type=int, default=None)
#     parser.add_argument("--no_wandb", action="store_true")
#     args = parser.parse_args()

#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)

#     cfg = QUANT_CONFIGS[args.config]
#     datasets_to_run = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]

#     # wandb = None
#     # if not args.no_wandb:
#     #     import wandb as _wandb
#     #     wandb = _wandb
#     #     wandb.init(
#     #         project="Uncertainty-Aware-Inference",
#     #         config={
#     #             "model": "llama2-13b",
#     #             "team": "team-c",
#     #             "quant_method": cfg["quant_method"],
#     #             "precision": cfg["precision"],
#     #             "dataset": args.dataset,
#     #             "split": args.split,
#     #             "seed": SEED,
#     #         },
#     #     )

#     print(f"Loading {args.config}: {cfg['hf_model_id']}...")
#     loader = LOADERS[cfg["method"]]
#     model, tokenizer = loader(cfg)
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
#         precision=cfg["precision"],
#         quant_method=cfg["quant_method"],
#     )

#     # if wandb:
#     #     wandb.finish()

#     print("\nDone. Results saved to", output_dir)


# if __name__ == "__main__":
#     main()
