# Uncertainty-Aware Inference: How Quantization Affects LLM Confidence Calibration

This repository contains the evaluation framework, profiling tools, and analysis scripts for our HPML group project. We systematically study how post-training quantization (PTQ) affects LLM confidence calibration across multiple architectures and scales, and whether knowledge distillation can recover lost calibration quality.

## Project Deliverables
- **Calibration analysis framework:** An open-source, reproducible pipeline for evaluating how any PTQ method affects LLM uncertainty across model families (reusable beyond this project).
- **Cross-model precision–calibration tradeoff curves:** Pareto frontiers showing cost vs. accuracy vs. calibration for Llama-2 7B/13B and Mistral 7B, with Roofline model annotations.
- **Distillation recovery analysis:** Quantitative results on whether KD can restore calibration quality across architectures, with analysis of which calibration properties are recoverable.
- **Stretch (QAT vs. PTQ calibration comparison):** A comparison on Llama-2 7B, testing whether training-time quantization inherently preserves calibration better.
- **Routing simulation results:** Projected cost savings and quality impact of uncertainty-based model routing, with threshold sensitivity analysis.
- **Technical report (8–10 pages):** Structured for potential workshop submission (e.g., ICML Efficient Systems workshop, NeurIPS WANT workshop).
- **Final presentation:** Includes a live demo of the calibration evaluation pipeline.

## Team Structure and Models
The project is divided across three teams, each owning a specific model and a cross-cutting specialty:

- **Team A (Llama-2 7B):** Builds the shared calibration pipeline, leads Knowledge Distillation (KD) experiments, and owns the QAT stretch goal.
- **Team B (Mistral 7B):** Leads CUDA/Nsight profiling, Roofline analysis, and sets up vLLM serving infrastructure.
- **Team C (Llama-2 13B):** Leads cross-model Pareto analysis, routing simulations, and repository/report structures.

## Getting Started

### 1. Environment Setup
To create and activate the shared environment, run:
```bash
conda env create -f environment.yml
conda activate uncertainty_aware_env
```

After collecting JSON result files, generate the Pareto analysis HTML with:
```bash
python TeamC/pareto_script.py ./updated_results
```

### 2. Weights & Biases (W&B) Tracking
[https://wandb.ai/sm5916-columbia-university/Uncertainty-Aware-Inference](https://wandb.ai/sm5916-columbia-university/Uncertainty-Aware-Inference)   
Log experimental results to the shared W&B project. Ensure you are logged in:
```bash
wandb login
```
When running your sweeps, consider the following fields:
```bash
config = {
    "model": "llama2-7b | llama2-13b | mistral-7b",
    "team": "team-a | team-b | team-c",
    "quant_method": "fp16 | gptq | awq | bnb",
    "precision": "fp16 | int8 | int4 | nf4",
    "dataset": "triviaqa | arc | hellaswag | nq",
    "split": "validation | test",
    "seed": 42,
}
```

Also ensure you log the following metrics:

```bash
wandb.log({
    tokens_per_second,
    accuracy,
    ece
})
```
