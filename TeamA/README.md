# HPML Final Project: Temperature-Scaled Knowledge Distillation for Calibrating Quantized Language Models

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team A
- **Members:**
  - Jiaming Liu (jl7250) — *Calibration evaluation pipeline; shared model loading and dataset modules; LLaMA-1 7B FP16, AWQ-INT4, NF4 baseline evaluation; dual-temperature KD training and evaluation*
  - Yanhao Bai (yb2630) — *Calibration evaluation groundings; ECE, Brier Score, reliability diagram implementation; LLaMA-1 7B GPTQ-INT8 and GPTQ-INT4 evaluation*
  - Haotian Lei (hl3945) — *Team A report integration and KD result analysis; single and dual-temperature KD training and evaluation; ECE/Brier/entropy failure mode analysis*
  - Leah Li (ql2481) — *Calibration evaluation across Llama-2 7B, Mistral 7B, and Llama-2 13B (FP16, LLM.int8, GPTQ-INT4, AWQ-INT4, NF4, NF4+KD); KD single-temperature sweep; headline results table; report writing and figures*

## Submission

- **GitHub repository:** [https://github.com/chuanbinp/uncertainty-aware-inference](https://github.com/chuanbinp/uncertainty-aware-inference)
- **Final report:** [`TeamA/TeamA_HPML_Final_Report.pdf`](TeamA/TeamA_HPML_Final_Report.pdf)
- **Final presentation:** [`TeamA/TeamA_HPML_Final_Presentation.pptx`](TeamA/TeamA_HPML_Final_Presentation.pdf)
- **Experiment-tracking dashboard:** [https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project)

The final report PDF and the presentation file are checked into the `TeamA/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

Post-training quantization reduces LLM inference cost significantly — a 13B model in FP16 requires 26 GB of GPU memory, while 4-bit quantization reduces this to under 8 GB. However, quantization may degrade model calibration, the match between predicted confidence and actual accuracy, in ways that standard accuracy benchmarks do not capture. This project systematically studies how quantization affects calibration across three model families and five quantization formats, and whether knowledge distillation can recover calibration quality. We target inference optimization, with memory bandwidth and output distribution fidelity as the primary bottlenecks.

---

## 2. Model/Application Description

- **Model architectures:** LLaMA-1 7B, Mistral 7B Instruct v0.2, Llama-2 13B
- **Framework:** PyTorch 2.5, Hugging Face Transformers, PEFT (LoRA), BitsAndBytes, AutoGPTQ, AutoAWQ
- **Datasets:** HellaSwag (validation, 10,042 samples, MIT License), TriviaQA (validation, 17,944 samples, Apache 2.0), PubMedQA pqa_labeled (train, 1,000 samples, MIT License); C4 English web text (ODC-BY License, 2,000 samples) for KD training
- **Custom components:** Shared calibration evaluation pipeline (ECE, MCE, Brier Score, entropy, reliability diagrams); asymmetric dual-temperature KD training loop with LoRA adapters on frozen NF4 weights
- **Hardware target:** 1× NVIDIA A100 SXM4-80 GB (RunPod, CUDA 12.4, PyTorch 2.5)

---

## 3. Final Results Summary

| Metric | Baseline (NF4) | Best KD Config | Δ |
|---|---|---|---|
| HellaSwag ECE — LLaMA-1 7B | 0.3021 | 0.2703 (T=1) | −0.032 |
| HellaSwag ECE — Mistral 7B | 0.2793 | 0.2824 (T=2 stable) | +0.003 |
| PubMedQA ECE — Llama-2 13B | 0.2400 | 0.2139 (T=12) | −0.026 |
| PubMedQA Brier — Llama-2 13B | 0.3097 | 0.2991 (T=8) | −0.009 |
| Peak GPU Memory — LLaMA-1 7B | 13.5 GB (FP16) | 4.0 GB (NF4) | 70% less |

**Hardware:** 1× NVIDIA A100 SXM4-80 GB, CUDA 12.4, PyTorch 2.5, RunPod

**Headline result:** Applying temperature-scaled knowledge distillation to NF4-quantized LLMs improves calibration most reliably on Llama-2 13B, reducing PubMedQA ECE by 0.026 and Brier Score by 0.009 without accuracy loss, while GPTQ-INT4 achieves 3.4× throughput over FP16 at 4× lower memory with near-identical calibration.

---

## 4. Repository Structure

```
.
├── environment.yml             # Conda environment specification
├── shared/                     # Shared evaluation pipeline (all teams)
│   ├── eval_utils.py           # ECE, Brier, entropy, reliability diagrams
│   ├── data_loader.py          # HellaSwag, TriviaQA, PubMedQA loaders
│   ├── model_loader.py         # FP16, NF4, AWQ, GPTQ loaders
│   └── result_format.py        # Canonical result schema
└── TeamA/
    ├── README.md               # This file
    ├── RESULTS_SUMMARY.md      # Full numerical results for all configs
    ├── configs.py              # Model and KD registry
    ├── kd_train.py             # KD training script
    ├── run_eval.py             # Evaluation (hardcoded config)
    ├── run_eval_args.py        # Evaluation (CLI config)
    ├── run_eval_kd.py          # KD model evaluation
    ├── run_kd.sh               # Full dual-temperature sweep (train + eval)
    ├── test_kd.sh              # Smoke test (10 samples)
    ├── run_all.sh              # Run all PTQ baseline configs sequentially
    ├── TeamA_HPML_Final_Report.pdf
    ├── TeamA_HPML_Final_Presentation.pptx
    ├── ptq_1k/results/         # 1k-sample PTQ baseline results (matched to KD comparison)
    └── results/                # Results — JSON metrics and plots (adapters excluded)
        ├── llama1-7b-{fp16,nf4,gptq-int4,gptq-int8,awq-int4}/
        ├── kd/                 # Single-temperature KD (T1–T16)
        └── kd_dual/            # Dual-temperature KD (Tt*_Ts*)
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/chuanbinp/uncertainty-aware-inference.git
cd uncertainty-aware-inference

# Create and activate the conda environment
conda env create -f environment.yml
conda activate uncertainty_aware_env
```

**System requirements:** Python 3.10+, CUDA 12.x, 16 GB+ GPU memory for 7B models, 24 GB+ for Llama-2 13B. See `environment.yml` for pinned package versions.

### B. Experiment Tracking Dashboard

Public experiment-tracking dashboard with calibration metrics, throughput, and baseline vs. KD comparisons:

> **Dashboard:** [https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project)
>
> *Platform:* Weights & Biases

Verify the link opens in an incognito browser. Runs are tagged by model, quantization method, and temperature for easy comparison.

### C. Dataset

Datasets are loaded automatically from Hugging Face Hub on first run. No manual download required. C4 for KD training is streamed and does not require local storage.

### D. PTQ Baseline Evaluation

```bash
# Full split evaluation (used for Table II in the report)
python TeamA/run_eval_args.py --config llama1-7b-nf4
```

Available configs: `llama1-7b-fp16`, `llama1-7b-nf4`, `llama1-7b-awq-int4`, `llama1-7b-gptq-int4`, `llama1-7b-gptq-int8`

1k-sample PTQ baseline results (matched to KD comparisons) are pre-computed and available under `TeamA/ptq_1k/results/`.

### E. KD Training

```bash
# Single temperature
python TeamA/kd_train.py --config llama1-7b-nf4-kd --temperature 2.0

# Dual temperature
python TeamA/kd_train.py --config llama1-7b-nf4-kd --teacher-temp 4.0 --student-temp 3.0
```

### F. KD Evaluation

```bash
# Single temperature
python TeamA/run_eval_kd.py --config llama1-7b-nf4-kd --temperature 2.0

# Dual temperature
python TeamA/run_eval_kd.py --config llama1-7b-nf4-kd --teacher-temp 4.0 --student-temp 3.0
```

### G. Quickstart: Reproduce the Headline Result

The following reproduces the Llama-2 13B PubMedQA calibration improvement (approximately 3 hours on a single A100):

```bash
# 1. Set up environment
conda env create -f environment.yml && conda activate uncertainty_aware_env
export HF_TOKEN=hf_...

# 2. Run NF4 baseline
python TeamA/run_eval_args.py --config llama2-13b-nf4 --datasets pubmedqa

# 3. Run KD training at T=12
python TeamA/kd_train.py --config llama2-13b-nf4-kd --temperature 12.0

# 4. Evaluate KD model
python TeamA/run_eval_kd.py --config llama2-13b-nf4-kd --temperature 12.0 --datasets pubmedqa
```

---

## 6. Results and Observations

- **PTQ does not reliably degrade calibration:** ECE differences between FP16 and 4-bit formats are within 0.01 across most model-dataset pairs, suggesting quantization-induced miscalibration is smaller than previously assumed.
- **KD calibration effects are model-dependent:** LLaMA-1 7B is stable across temperatures but ECE improvements come at an accuracy cost. Mistral 7B collapses sharply beyond T=2. Llama-2 13B shows consistent, genuine calibration improvement on PubMedQA.
- **ECE alone is unreliable:** At high temperatures, Mistral 7B achieves low ECE by producing near-uniform predictions, not by improving calibration. Brier Score and entropy are necessary complements.
- **Asymmetric dual-temperature KD partially recovers accuracy:** For LLaMA-1 on TriviaQA, accuracy recovers from 0.575 to 0.654 under Tt=2, Ts=1, though the ECE-accuracy trade-off is not fully eliminated.
- **GPTQ-INT4 is the best inference format:** 3.4× throughput over FP16 at 4× lower memory with near-FP16 calibration. NF4 is preferred when LoRA post-training is planned.

---

## 7. Notes

- **Model substitution:** The original plan included LLaMA-2 7B, but a reliable GPTQ-INT8 model was not available for that checkpoint. LLaMA-1 7B (huggyllama/llama-7b) was used as a substitute for the 7B PTQ baseline experiments.
- All results are saved to `TeamA/results/` with per-dataset JSON metrics and reliability diagram PNGs.
- Trained LoRA adapters stored locally only. Re-run `kd_train.py` to regenerate them.
- Model checkpoints are not committed. Base models are loaded from Hugging Face Hub on demand.

---

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks).*

- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude (Anthropic)

**Specific purpose:** AI tools were used in two capacities. For the final report, AI was used to polish prose and improve writing clarity on drafts written by the authors. For implementation, AI tools were used for clarification of library APIs and debugging assistance.

**Sections affected:** Final report prose (all sections); debugging during implementation of the shared evaluation pipeline and KD training loop.

**How we verified correctness:** All reported experimental results were produced and verified by the authors independently. Profiler trace interpretations, performance reasoning, experimental design, and analytical conclusions are entirely the authors' own work. AI-assisted prose was reviewed and approved by all team members before inclusion.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure appears as an appendix in the final report.

---

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{teamA2026hpml,
  title  = {Temperature-Scaled Knowledge Distillation for Calibrating Quantized Language Models},
  author = {Liu, Jiaming and Li, Leah and Lei, Haotian and Bai, Yanhao},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/chuanbinp/uncertainty-aware-inference}
}
```

### Contact

Open a GitHub Issue or email the team via Columbia University email.

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*