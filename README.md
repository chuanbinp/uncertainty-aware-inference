# Uncertainty-Aware Inference: How Quantization Affects LLM Confidence Calibration

This repository contains the evaluation framework, profiling tools, and analysis scripts for our HPML group project. We systematically study how post-training quantization (PTQ) affects LLM confidence calibration across multiple architectures and scales, and whether knowledge distillation can recover lost calibration quality.

## Project Deliverables
- **Calibration analysis framework:** An open-source, reproducible pipeline for evaluating how any PTQ method affects LLM uncertainty across model families (reusable beyond this project).
- **Cross-model precision–calibration tradeoff curves:** Pareto frontiers showing cost vs. accuracy vs. calibration for Llama-2 7B/13B and Mistral 7B, with Roofline model annotations.
- **Distillation recovery analysis:** Quantitative results on whether KD can restore calibration quality across architectures, with analysis of which calibration properties are recoverable.
- **Stretch (QAT vs. PTQ calibration comparison):** A comparison on Llama-2 7B, testing whether training-time quantization inherently preserves calibration better.
- **Routing simulation results:** Projected cost savings and quality impact of uncertainty-based model routing, with threshold sensitivity analysis.

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

### 2. Weights & Biases (W&B) Tracking
[https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project)   
Log experimental results to the shared W&B project. Ensure you are logged in:
```bash
wandb login
```
