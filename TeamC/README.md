# HPML Final Project: Uncertainty-Aware Inference at 13B Scale: How Quantization Affects LLM Confidence Calibration

> **Course:** High Performance Machine Learning  
> **Semester:** Spring 2026  
> **Instructor:** Dr. Kaoutar El Maghraoui
---
## Team Information
- **Team Name:** Team 17 (Team C of this project)
- **Advisor / Mentor:** Dr. Ruchi Mahindru, IBM Research
- **Members:**
   - **Yechan Jeon** (yj2910) — routing simulation, scripting, report, slides
   - **Chuan Bin Phoe** (cp3451) — experiment scripts, pareto dashboard scripts, report, slides
   - **Samuel Lee** (sl5806) — experiments runs, scripting, report, slides
   - **Sanket Makkar** (sm5916) — experiment runs, report, slides
---
## Submission
- **GitHub repository:** [uncertainty-aware-inference](https://github.com/chuanbinp/uncertainty-aware-inference)
   - **Project subdirectory:** [`TeamC/`](https://github.com/chuanbinp/uncertainty-aware-inference/tree/master/TeamC)
- **Final report:** [deliverables/17_HPML_Final_Report.pdf](https://github.com/chuanbinp/uncertainty-aware-inference/tree/master/TeamC/deliverables/17_HPML_Final_Report.pdf)
- **Final presentation:** [deliverables/presentation.pdf](https://github.com/chuanbinp/uncertainty-aware-inference/tree/master/TeamC/deliverables/presentation.pdf)
- **Experiment-tracking dashboard:** [Weights & Biases dashboard](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project/) [Pareto dashboard](https://chuanbinp.github.io/uncertainty-aware-inference/TeamC/analysis_results/pareto/pareto_comparison.html)

The final report and presentation are included in the repository deliverables and were also submitted through CourseWorks.

---
## 1. Problem Statement
Large language model inference is expensive, and while post-training quantization (PTQ) can reduce latency, GPU memory use, and serving cost, standard evaluations usually focus on throughput and accuracy while overlooking whether the model’s confidence remains trustworthy. This project studies inference-time optimization for Llama-2 13B by evaluating how GPTQ, AWQ, and NF4 quantization affect confidence calibration across HellaSwag, TriviaQA, and PubMedQA, using metrics such as ECE, Brier score, and entropy. We then examine the tradeoff between efficiency, accuracy, and calibration quality, and test whether uncertainty-aware routing can make quantized inference practically useful without introducing silent miscalibration in high-stakes settings.

---
## 2. Model/Application Description
This project uses **Llama-2 13B** as the primary application model and evaluates how post-training quantization affects both inference efficiency and confidence calibration in deployment-oriented question-answering settings. It compares a full-precision baseline with several low-bit variants to study the tradeoff among throughput, accuracy, and calibration quality.

- **Model architecture:** Llama-2 13B causal language model with 40 layers and 13 billion parameters.
- **Framework / stack:** Hugging Face Transformers for inference, with quantized variants implemented using bitsandbytes, AutoGPTQ, and AutoAWQ.
- **Configurations evaluated:**
  - **FP16 baseline** — full-precision Llama-2 13B.
  - **GPTQ-INT8** — 8-bit GPTQ quantized configuration.
  - **GPTQ-INT4** — 4-bit GPTQ quantized configuration.
  - **AWQ-INT4** — 4-bit AWQ quantized configuration.
  - **NF4** — 4-bit NF4 configuration using bitsandbytes.
- **Datasets:**
  - **HellaSwag** — commonsense multiple-choice QA with 10,042 examples.
  - **TriviaQA** — factual QA with 17,944 examples.
  - **PubMedQA** — biomedical QA / out-of-distribution benchmark with 1,000 examples.
- **Custom modifications:** The project keeps the underlying model architecture unchanged and instead extends the evaluation pipeline to capture answer-position confidence, compute calibration metrics such as ECE, MCE, Brier score, and entropy, save per-example tensors, generate Pareto frontier analyses, and simulate uncertainty-aware routing.
- **Hardware target:** Single NVIDIA A100 40GB GPU on Google Colab, batch size 1, seed 42
---
## 3. Final Results Summary
### Table II. Llama-2 13B calibration sweep

| Config | HellaSwag tok/s | HellaSwag Acc | HellaSwag ECE | HellaSwag Brier | TriviaQA tok/s | TriviaQA Acc | TriviaQA ECE | TriviaQA Brier | PubMedQA tok/s | PubMedQA Acc | PubMedQA ECE | PubMedQA Brier |
| ------ | --------------- | ------------- | ------------- | --------------- | -------------- | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | --------------- |
| FP16 | 1903.394 | 0.773 | 0.331 | 0.269 | 39.290 | 0.767 | 0.164 | 0.211 | 527.541 | 0.470 | 0.229 | 0.307 |
| GPTQ-INT8 | 865.263 | 0.773 | 0.331 | 0.268 | 18.803 | 0.767 | 0.162 | 0.211 | 233.754 | 0.473 | 0.228 | 0.308 |
| GPTQ-INT4 | 2179.729 | 0.766 | 0.327 | 0.269 | 41.740 | 0.754 | 0.162 | 0.220 | 660.891 | 0.472 | 0.222 | 0.308 |
| AWQ-INT4 | 928.439 | 0.772 | 0.332 | 0.274 | 22.822 | 0.752 | 0.165 | 0.220 | 288.567 | 0.470 | 0.242 | 0.314 |
| NF4 | 734.664 | 0.769 | 0.329 | 0.270 | 19.907 | 0.757 | 0.166 | 0.216 | 201.293 | 0.467 | 0.240 | 0.390 |

### Table III. Best average accuracy gain (vs FP16) for each model architecture

| Architecture | Avg Acc Gain vs FP16 |
| ------------ | -------------------- |
| Llama-1 7B | +0.11 pp |
| Llama-2 13B | +0.13 pp |
| Mistral-7B | +0.07 pp |

### Table IV. Savings across Llama-1 7B

| Config | Dataset | Quant Frac | Acc (vs FP16) | Savings |
| ------ | ------- | ---------- | ------------- | ------- |
| GPTQ 4-bit | PubMedQA | 100% | +0.30 pp | 69.85% |
| GPTQ 8-bit | PubMedQA | 99.8% | 0.00 pp | 67.78% |
| GPTQ 8-bit | HellaSwag | 87.4% | +0.03 pp | 59.77% |
| GPTQ 8-bit | TriviaQA | 80.7% | +0.08 pp | 42.68% |
| GPTQ 4-bit | HellaSwag | 35.5% | +0.03 pp | 25.07% |

**Hardware:** 1× NVIDIA A100 40GB on Google Colab, single-GPU inference, batch size 1, seed 42.

**Headline result**   
*Within-model calibration (Llama-2 13B):* Calibration was largely stable under quantization on HellaSwag, and GPTQ-INT4 was the most calibration-preserving 4-bit option for Llama-2 13B, improving ECE over FP16 on all three datasets while AWQ-INT4 and NF4 degraded calibration on PubMedQA.

*Cross-model calibration:* Larger scale did not guarantee better calibration. On HellaSwag, Llama-2 13B had the highest ECE values in the study and no configuration was Pareto-optimal, while smaller Llama-1 7B GPTQ variants achieved lower ECE and occupied the Pareto frontier. Although some Llama-2 13B configurations reached dataset-specific frontiers on TriviaQA and PubMedQA, they were not cross-model dominant. GPTQ emerged as the most consistently Pareto-efficient quantization family across throughput, accuracy, and ECE tradeoffs.

*Uncertainty-based routing:* Confidence-based routing generally preserved FP16-level accuracy across model families, but the gains were small and practical savings were concentrated in specific Llama-1 7B GPTQ configurations

## 4. Repository Structure
```text
TeamC/
├── archive/                            # Archived outputs from previous runs or deprecated experiments
├── deliverables/                       
│   ├── 17_HPML_Final_Report.pdf        # Final written report
│   └── presentation.pdf                # Final presentation deck
├── analysis_results/                   
│   ├── pareto/                         # Pareto frontier plots, summaries, and dashboard assets
│   └── routing/                        # Routing simulation outputs, threshold sweeps, and tradeoff results
├── sweep_results/                      
│   ├── llama2-13b-awq-int4/            # AWQ INT4 evaluation results
│   ├── llama2-13b-gptq-int4/           # GPTQ INT4 evaluation results
│   ├── llama2-13b-gptq-int8/           # GPTQ INT8 evaluation results
│   ├── llama2-13b-nf4/                 # NF4 evaluation results
│   └── llama2-13b_16bit_fp16/          # FP16 baseline evaluation results
├── README.md                           # Project overview, setup, results summary, and usage
├── configs.py                          # Registry of Llama-2 13B quantization configs and HF model IDs
├── run_eval.py                         # Main evaluation script for one quantized config across QA datasets
├── llama2_13b.ipynb                    # Colab notebook for running evaluation sweeps using run_eval.py
├── pareto_script.py                    # Cross-model Pareto frontier analysis across throughput/accuracy/ECE
├── routing_simulation.py               # Uncertainty-aware routing simulator with threshold sweeps
├── LICENSE                             # Repository license
└── .gitignore                          # Git ignore rules for caches, outputs, and temporary files
```
---
## 5. Reproducibility Instructions
### A. Environment Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/chuanbinp/uncertainty-aware-inference.git
   cd uncertainty-aware-inference
   ```

2. **Set up conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate uncertainty_aware_env
   ```
**System requirements:** Python 3.10+, CUDA 12.x, ≥ 28 GB GPU memory for running Llama2-13B in FP16.
See `environment.yml` for pinned package versions.

### B. Experiment Tracking Dashboard
Public experiment-tracking dashboards for evaluation metrics, calibration analysis, and deployment tradeoff visualizations. We logged experiment runs to **Weights & Biases** and published a separate interactive **Pareto dashboard** for cross-model frontier analysis.

1. **Experiment Tracking Dashboard:** [Weights & Biases dashboard](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project)
2. **Pareto Dashboard:**  [Pareto dashboard](https://chuanbinp.github.io/uncertainty-aware-inference/TeamC/analysis_results/pareto/pareto_comparison.html)  
> *Platform used:* Weights & Biases, GitHub Pages

### C. Dataset
The project evaluates three QA benchmarks: HellaSwag, TriviaQA, and PubMedQA. The dataset is *not* committed to the repository.
  - **HellaSwag** — commonsense multiple-choice QA with 10,042 examples.
  - **TriviaQA** — factual QA with 17,944 examples.
  - **PubMedQA** — biomedical QA / out-of-distribution benchmark with 1,000 examples.

### D. Evaluation
Run calibration evaluation for a single quantization configuration across the three QA benchmarks: HellaSwag, TriviaQA, and PubMedQA. Supported configurations include FP16, GPTQ INT4/INT8, AWQ INT4, and NF4.

```bash
python TeamC/run_eval.py
HF_TOKEN=hf_... python TeamC/run_eval.py   # required for gated models such as FP16 / NF4
```

Supported configs:
- `llama2-13b-fp16`
- `llama2-13b-nf4`
- `llama2-13b-awq-int4`
- `llama2-13b-gptq-int4`
- `llama2-13b-gptq-int8`

### E. Pareto Analysis

Run cross-model Pareto frontier analysis to compare configurations across throughput, accuracy, and calibration error. This script loads JSON result summaries from TeamA, TeamB, and TeamC and identifies Pareto-dominant configurations.

```bash
python TeamC/pareto_script.py
python TeamC/pareto_script.py --output_dir TeamC/results/pareto
```

### F. Routing Simulation

Simulate uncertainty-based routing with a two-tier serving policy: serve high-confidence queries from a cheaper quantized model and escalate uncertain queries to FP16. The routing sweep evaluates threshold-dependent tradeoffs in effective accuracy, calibration, and cost.

```bash
python TeamC/routing_simulation.py
python TeamC/routing_simulation.py --dataset arc_challenge --num_thresholds 200
```

### G. Quickstart: Reproduce the Headline Result

The sequence below reproduces the main Team C workflow: run the Llama-2 13B quantization sweep, analyze Pareto tradeoffs, and simulate uncertainty-based routing. The reported setup uses a single NVIDIA A100 40GB GPU with batch size 1 and deterministic seed 42.

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate uncertainty_aware_env

# 2. Run one evaluation sweep
python TeamC/run_eval.py

# 3. Generate Pareto frontier analysis
python TeamC/pareto_script.py

# 4. Run routing simulation
python TeamC/routing_simulation.py
```
---

## 6. Results and Observations
- **GPTQ-INT4 emerged as the strongest 4-bit option for Llama-2 13B** because it was the most calibration-preserving low-bit configuration, improving ECE over the FP16 baseline on all three datasets while also delivering higher throughput. 
- **HellaSwag was effectively quantization-invariant at the 13B scale**, with calibration remaining stable across all tested configurations and ECE staying in a narrow band from 0.327 to 0.332. 
- **AWQ-INT4 and NF4 were less robust for calibration preservation on Llama-2 13B**, particularly on PubMedQA, where both methods increased ECE relative to FP16 despite similar accuracy. 
- **Bigger models were not automatically better calibrated**, as no Llama-2 13B configuration was cross-model Pareto-optimal
- **GPTQ was the most consistently Pareto-efficient quantization family** across throughput, accuracy, and ECE tradeoffs. 
- **Uncertainty-aware routing preserved FP16-level accuracy but produced limited practical gains**, with only small accuracy improvements across model families and meaningful cost savings concentrated in specific Llama-1 7B GPTQ configurations. 
- **Routing was beneficial only when the systems stack favored the quantized path**, reinforcing that deployment wins depend not just on calibration quality but also on whether quantized serving is actually faster than the FP16 baseline.
---

## 7. Notes
- Team C focuses on running calibration sweeps for *Llama-2 13B*, cross-model evaluations, pareto-analysis and uncertainty-aware routing strategies for quantized language models, while the broader report also uses Team A’s Llama-1 7B and Team B’s Mistral-7B results.
- All secrets (API keys, Wandb tokens) are loaded from environment variables. To run the experiments, make sure the following are available:
1. **Hugging Face account and access token**: Required for accessing gated Llama-2 checkpoints used in the FP16 and some quantized evaluation flows. 
   - Sign up at [huggingface.co](https://huggingface.co)
   - Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Set the token as an environment variable:
     ```bash
     export HF_TOKEN=hf_...
     ```
   - In Google Colab, the token can also be stored in Colab Secrets.

2. **Weights & Biases account**: Used for experiment tracking, logging evaluation metrics, and monitoring calibration results across runs.
   - Sign up at [wandb.ai](https://wandb.ai)
   - Login from the terminal:
     ```bash
     wandb login
     ```

3. **GPU access**: Experiments were run on a single **NVIDIA A100 40GB** GPU, including Google Colab-based workflows for Team C evaluations.

### AI Use Disclosure
*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**
- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below. 

**Tool(s) used:** ChatGPT

**Specific purpose:** AI-assisted tools were used in a limited capacity for proofreading, formatting assistance, and improving clarity of prose after the technical content had already been written by the authors. 

**Sections affected:** Written presentation materials, including report prose and related documentation text. 

**How we verified correctness:** All experimental design, implementation, profiling, evaluation methodology, analysis, interpretation of results, and conclusions were performed and verified by the authors. 

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above.  The same disclosure block appears as an appendix in the final report. 

### License
Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation
If you build on this work, please cite:
```bibtex
@misc{teamc2026hpml,
  title  = {Uncertainty-Aware Inference --- How Quantization Affects LLM Confidence Calibration},
  author = {Jeon, Yechan and Phoe, Chuan Bin and Lee, Samuel and Makkar, Sanket},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/chuanbinp/uncertainty-aware-inference/tree/master/TeamC}
}
```

### Blog post for extra credit
[Medium Article](https://medium.com/@sm5916/why-confidence-calibration-matters-in-quantized-llm-routing-4ff2246c109a?postPublishedType=initial)

### Contact
Email *[chuanbin.p@columbia.edu]*.

---
*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
