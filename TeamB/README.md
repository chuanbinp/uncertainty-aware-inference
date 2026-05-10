# HPML Final Project: Uncertainty-Aware Inference - How Quantization Affects LLM Confidence Calibration

> **Course:** High Performance Machine Learning (COMS 6998)
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team 29 (Team B)
- **Members:**
  - Anubha Vyasamudri (av3329) - *calibration pipeline, Pytorch profiling, report/ppt structure*
  - Rohit Ramesh (rr3713) - *CUDA/Nsight Compute profiling, Roofline analysis, report/ppt structure*
  - Sanjita Chandan Ballapur (sb5216) - *calibration pipeline, Pytorch profiling, report/ppt structure*
  - Tamanna Ananna Haque (ta2642) - *calibration evaluation, CUDA/Nsight Compute profiling, report/ppt structure*
  - Vishal Menon (vm2820) - *calibration evaluation, vLLM serving infrastructure, report/ppt structure*

## Submission

- **GitHub repository:** [https://github.com/chuanbinp/uncertainty-aware-inference](https://github.com/chuanbinp/uncertainty-aware-inference)
- **Final report:** [`deliverables/Team29_HPML_Final_Report.pdf`](deliverables/Team29_HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/Team29_Final_Presentation.pptx`](deliverables/Team29_Final_Presentation.pptx)
- **Experiment-tracking dashboard:** [https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project)

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

Post-training quantization (PTQ) reduces LLM inference cost substantially, as a 7B model at 4-bit precision uses approximately 70% less GPU memory, but standard accuracy benchmarks mask a critical failure mode: a quantized model can preserve top-1 accuracy while becoming severely miscalibrated, producing overconfident incorrect predictions and underconfident correct ones. We target **inference**, characterising the calibration–efficiency tradeoff across 3 model families × 5 PTQ configurations × 3 datasets on NVIDIA L4 and A100. The primary bottleneck for standard FP16 inference is DRAM bandwidth; fused quantized kernels (Marlin W4A16, awq\_gemm) shift the bottleneck to compute, which is the central hardware finding of this project.

---

## 2. Model/Application Description

- **Model architecture:** Mistral-7B-Instruct-v0.2 (Team B primary - sliding-window attention, grouped-query attention); LLaMA-1 7B (`huggyllama/llama-7b`, Team A); LLaMA-2 13B (`meta-llama/Llama-2-13b-hf`, Team C).
- **Framework:** PyTorch 2.6.0+cu124 / PyTorch 2.12.0.dev+cu128 · HuggingFace Transformers 4.51.3 · vLLM · AutoGPTQ 0.7.1 · AutoAWQ 0.2.9 · bitsandbytes 0.49.2.
- **Dataset:** TriviaQA (10,003 examples, Apache-2.0, [HuggingFace](https://huggingface.co/datasets/trivia_qa)); HellaSwag (17,215 examples, MIT, [HuggingFace](https://huggingface.co/datasets/hellaswag)); PubMedQA (1,000 examples, MIT, [HuggingFace](https://huggingface.co/datasets/pubmed_qa)).
- **Custom layers or modifications:** No custom CUDA kernels written. The Marlin W4A16 kernel (shipped with AutoGPTQ) and the fused awq\_gemm kernel (shipped with AutoAWQ) are used unmodified. Original contributions are the ncu instrumentation scripts (`TeamB/ncu_*.py`, `TeamB/llama_workspace/ncu_*.py`), the roofline plotting utility (`TeamB/plot_ncu_roofline.py`), and the calibration evaluation pipeline (`shared/eval_utils.py`).
- **Hardware target:** NVIDIA L4 24 GB GDDR6 (GCP `asia-south1-b`) for ncu kernel-level profiling; NVIDIA A100 80 GB SXM (Colab Pro) for calibration evaluation, PyTorch Profiler, and vLLM throughput benchmarking.

---

## 3. Final Results Summary

| Metric                        | Baseline            | Optimized                    | Δ (Improvement)  |
| ----------------------------- | ------------------- | ---------------------------- | ---------------- |
| HellaSwag Accuracy            | 82.9% (FP16)        | 82.6% (NF4)                  | −0.3 pp          |
| TriviaQA ECE ↓                | 0.143 (FP16)        | 0.152 (GPTQ INT4)            | +0.009           |
| PubMedQA ECE ↓                | 0.229 (FP16)        | 0.067 (AWQ INT4)†            | −0.162†          |
| Inference Throughput          | 942 tok/s (FP16)    | 1,677 tok/s (GPTQ INT4)      | **+1.78×**       |
| Peak GPU Memory               | 14.5 GB (FP16)      | 4.2 GB (AWQ INT4)            | **−71%**         |
| SM Utilization (prefill)      | 52% (FP16)          | 82% (GPTQ INT4 Marlin)       | +30 pp           |
| Arithmetic Intensity          | 243 FLOPs/B (FP16)  | 1,865 FLOPs/B (AWQ INT4)     | **+19×**         |

† AWQ INT4 ECE improvement on PubMedQA is a confidence redistribution artefact, not a genuine calibration gain as MCE simultaneously triples from 0.330 to 0.939.

**Hardware:** 1× NVIDIA A100 80 GB SXM, CUDA 12.8, PyTorch 2.12.0.dev, Ubuntu 22.04 (calibration + vLLM); 1× NVIDIA L4 24 GB GDDR6, CUDA 12.4, PyTorch 2.6.0, Ubuntu 22.04 (profiling).

**Headline result (one sentence):** *GPTQ INT4 with the Marlin kernel delivers 1.78× vLLM throughput and 71% memory reduction vs. FP16, shifts inference from memory-bound (SM=52%, AI=243 FLOPs/B) to compute-bound (SM=82%, AI=951 FLOPs/B), and incurs less than 1% accuracy degradation on in-distribution tasks, making it the Pareto-dominant PTQ configuration for Mistral-7B across efficiency, accuracy, and calibration.*

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt               # pip packages - A100/Colab calibration env
├── requirements_gcp.txt           # pip packages - GCP L4 ncu profiling env 
├── environment_uai.yml            # Full conda env export for exact L4 reproduction
├── setup_gcp_l4.sh                # One-shot GCP L4 environment setup script
├── .env.example                   # Template for HF_TOKEN and WANDB_API_KEY
├── .gitignore
├── deliverables/                  # Final report (PDF) and final presentation (PPT/PDF) - same files uploaded to CourseWorks
│   ├── Team29_HPML_Final_Report.pdf
│   └── Team29_Final_Presentation.pptx
├── shared/                        # Shared evaluation utilities (used by all three teams)
│   ├── model_loader.py            # Unified model loading (FP16 / GPTQ / AWQ / NF4)
│   ├── eval_utils.py              # ECE, MCE, Brier score, entropy computation
│   ├── eval_template.py           # Evaluation loop template
│   ├── data_loader.py             # Dataset loading (HellaSwag / TriviaQA / PubMedQA)
│   └── result_format.py           # JSON result schema
├── TeamB/                         # Team B: Mistral-7B + cross-model systems profiling toolkit
│   ├── configs.py                 # Model registry for all 3 model families
│   ├── run_eval.py                # Calibration sweep entry point
│   ├── run_vllm.py                # vLLM throughput benchmarking
│   ├── run_profiler.py            # PyTorch Profiler harness
│   ├── ncu_fp16.py                # Nsight Compute profiling scripts (Mistral-7B)
│   ├── ncu_gptq_int4.py
│   ├── ncu_gptq_int8.py
│   ├── ncu_awq_int4.py
│   ├── ncu_nf4.py
│   ├── plot_ncu_roofline.py       # Roofline plot generation from ncu CSV output
│   ├── nsight_roofline.py         # NSight roofline helper utilities
│   ├── nvtx_utils.py              # NVTX range push/pop helpers for ncu region marking
│   ├── llama_workspace/           # LLaMA-1 7B and LLaMA-2 13B ncu profiling scripts
│   │   ├── ncu_llama1_7b_*.py
│   │   ├── ncu_llama2_13b_*.py
│   │   ├── run_llama_ncu.sh       # Batch ncu sweep for all LLaMA configs on L4
│   │   └── ncu_results/           # Raw ncu CSV outputs (*_metrics.csv)
│   ├── notebooks/
│   │   ├── mistral_7b_calibration.ipynb   # Full PTQ calibration sweep
│   │   ├── teamb_vllm.ipynb               # vLLM throughput benchmark
│   │   ├── teamb_profiler.ipynb           # PyTorch Profiler sweep
│   │   ├── teamb_nsight.ipynb             # NSight Compute roofline analysis
│   │   ├── crossmodel_profiling.ipynb     # Cross-model roofline comparison
│   │   └── systems_analysis.ipynb         # Combined systems analysis and plots
│   ├── calibration_results/       # Per-config calibration JSONs and reliability diagrams
│   ├── nsight_profiler_results/
│   │   ├── ncu/                   # *_metrics.csv files (raw .ncu-rep excluded via .gitignore)
│   │   └── roofline_*.png
│   ├── pytorch_profiler_results/
│   │   ├── *_profile.json         # Per-config timing and roofline summary
│   │   ├── profiler_summary.json
│   │   └── roofline_mistral7b_A100-80GB.png
│   └── vllm_results/              # Per-config vLLM throughput JSONs
├── TeamA/                         # Team A: LLaMA-1 7B + shared calibration pipeline
│   ├── configs.py
│   ├── run_eval.py
│   └── run_eval_args.py
└── TeamC/                         # Team C: LLaMA-2 13B + Pareto analysis and routing
    ├── configs.py
    ├── run_eval.py
    ├── routing_simulation.py
    └── pareto_script.py
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# Clone
git clone https://github.com/chuanbinp/uncertainty-aware-inference.git
cd uncertainty-aware-inference

# (Recommended) calibration evaluation and vLLM - A100 / Colab
pip install -r requirements.txt

# ncu kernel profiling - GCP L4 VM
bash setup_gcp_l4.sh        # one-shot: installs uai conda env, fixes CUDA paths, sets perf_event_paranoid
conda activate uai
# Alternatively, restore the exact conda environment:
conda env create -f environment_uai.yml
conda activate uai
```

**System requirements:** Python 3.11, CUDA 12.4 (profiling) or CUDA 12.x (calibration), ≥40 GB GPU memory for FP16 inference (A100 recommended); ncu profiling additionally requires `perf_event_paranoid ≤ 0` and root/sudo access. See `requirements.txt` and `requirements_gcp.txt` for pinned package versions.

### B. Experiment Tracking Dashboard

Public experiment-tracking dashboard with calibration metrics, vLLM throughput, and profiler results across all quantization configurations:

> **🔗 Dashboard:** [https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project](https://wandb.ai/Uncertainty_Aware_Inference_Lab/UAI_Project)
>
> *Platform used:* Weights & Biases

Verify the link opens in an incognito browser. The dashboard includes per-config runs tagged by model, quantization method, and experiment type (`calibration`, `vllm_throughput`, `pytorch_profiler`), with a curated report summarising the calibration and roofline findings.

### C. Dataset

```bash
# Datasets are fetched automatically via HuggingFace Datasets on first eval run
python TeamB/run_eval.py --config mistral-7b-fp16 --datasets hellaswag triviaqa pubmedqa
```

No manual download is required. Datasets are fetched from HuggingFace Hub and cached locally (TriviaQA: Apache-2.0; HellaSwag: MIT; PubMedQA: MIT). Datasets are *not* committed to the repository.

### D. Training

This project evaluates post-training quantization only - no training phase. To reproduce the calibration evaluation across all five configurations:

```bash
export HF_TOKEN=hf_your_token_here

python TeamB/run_eval.py --config mistral-7b-fp16
python TeamB/run_eval.py --config mistral-7b-gptq-int4
python TeamB/run_eval.py --config mistral-7b-gptq-int8
python TeamB/run_eval.py --config mistral-7b-awq-int4
python TeamB/run_eval.py --config mistral-7b-nf4
```

### E. Evaluation

```bash
# Single config - results written to TeamB/calibration_results/{config}/
python TeamB/run_eval.py --config mistral-7b-gptq-int4 --datasets hellaswag triviaqa pubmedqa

# Full sweep via notebook (Colab A100)
# Open TeamB/notebooks/mistral_7b_calibration.ipynb and run all cells
```

Results per config: `{dataset}_results.json` containing ECE, MCE, Brier score, average entropy, and per-sample predictions.

### F. Profiling

To regenerate the PyTorch Profiler traces referenced in the report:

```bash
python TeamB/run_profiler.py --config mistral-7b-fp16     --output-dir TeamB/pytorch_profiler_results
python TeamB/run_profiler.py --config mistral-7b-gptq-int4 --output-dir TeamB/pytorch_profiler_results
# View Chrome trace at perfetto.dev
```

To regenerate ncu kernel-level hardware counter measurements (GCP L4 required):

```bash
conda activate uai && export HF_TOKEN=hf_your_token_here

# Mistral-7B ncu sweep - FP16 example
sudo rm -f /tmp/nsight-compute-lock
sudo -E /usr/local/cuda/bin/ncu \
    -o TeamB/nsight_profiler_results/ncu/mistral_fp16 \
    --metrics gpu__time_duration.sum,dram__bytes.sum,sm__inst_executed_pipe_tensor.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
    --launch-skip 30 --launch-count 20 \
    --kernel-name "regex:ampere_fp16_s16816gemm|fmha_cutlassF" \
    --force-overwrite \
    ~/miniconda3/envs/uai/bin/python TeamB/ncu_fp16.py

# LLaMA ncu sweep (all 9 configs - LLaMA-1 7B and LLaMA-2 13B)
cd TeamB/llama_workspace && bash run_llama_ncu.sh
```

### G. Quickstart: Reproduce the Headline Result

The following sequence reproduces the 1.78× throughput headline number end-to-end (≈5 minutes on a single A100):

```bash
# 1. Set up environment
pip install -r requirements.txt

# 2. No dataset download needed - fetched automatically on first eval run

# 3. Run vLLM throughput benchmark for FP16 and GPTQ INT4
export HF_TOKEN=hf_your_token_here
python TeamB/run_vllm.py --config mistral-7b-fp16      --output-dir ./results
python TeamB/run_vllm.py --config mistral-7b-gptq-int4  --output-dir ./results

# 4. Compare
python -c "
import json
fp16  = json.load(open('results/mistral-7b-fp16_vllm.json'))
gptq4 = json.load(open('results/mistral-7b-gptq-int4_vllm.json'))
ratio = gptq4['avg_tokens_per_second'] / fp16['avg_tokens_per_second']
print(f'FP16:      {fp16[\"avg_tokens_per_second\"]:.1f} tok/s')
print(f'GPTQ INT4: {gptq4[\"avg_tokens_per_second\"]:.1f} tok/s')
print(f'Speedup:   {ratio:.2f}x')
"
```

---

## 6. Results and Observations

All numerical results are committed under `TeamB/calibration_results/`, `TeamB/vllm_results/`, and `TeamB/pytorch_profiler_results/`.

- *Kernel > bit-width (vLLM throughput):* GPTQ INT4 (Marlin) achieves 1.78× vLLM throughput vs. FP16 and shifts prefill inference to compute-bound operation (SM=82%, AI=951 FLOPs/B). GPTQ INT8 with the exllama kernel regresses to 0.47× FP16 - the same bit-width with a different kernel produces the opposite outcome.
- *19× arithmetic intensity gap at identical bit-width:* NF4 (AI=388 FLOPs/B) vs. AWQ INT4 (AI=1,865 FLOPs/B) - both 4-bit, entirely different hardware regimes due to the two-step kDequantize+GEMM path vs. the fused awq\_gemm kernel.
- *ECE alone is insufficient for OOD evaluation:* On PubMedQA, AWQ INT4 ECE drops to 0.067 while MCE simultaneously triples to 0.939. Brier score and entropy distributions are required to identify confidence redistribution artefacts that ECE masks.
- *Knowledge distillation does not reliably recover calibration:* At temperature T≥2, MCE explodes across all datasets even as ECE appears to improve. At T≥4, accuracy collapses entirely.
- *What did not work:* NF4 achieves 81% SM utilization but produces only 0.09× vLLM throughput because the intermediate dequantization step introduces a full DRAM round-trip per layer that fused kernels avoid. Confidence-threshold routing for Mistral-7B produces negative cost savings at batch=1 because kernel overhead exceeds the memory saving at low batch sizes.

![Roofline - Mistral-7B on NVIDIA L4](TeamB/nsight_profiler_results/roofline_comparison.png)

---

## 7. Notes

- Large binary files (`.ncu-rep`, `.pt` tensors, Chrome profiler traces, `.tar.gz` archives) are excluded via `.gitignore`. Only `*_metrics.csv` and `*_profile.json` summaries are committed.
- HuggingFace model checkpoints are not committed to the repository. All models download automatically at runtime via `HF_TOKEN`.
- Two separate environments are used: `uai` conda env (CUDA 12.4, PyTorch 2.6.0, `auto-gptq==0.7.1`) for ncu profiling on the GCP L4 VM; the Colab notebook environment (CUDA 12.8, PyTorch 2.12.0.dev) for calibration evaluation and vLLM benchmarking. See `requirements_gcp.txt` and `environment_uai.yml` for the profiling env.
- All secrets (API keys, W&B tokens) are loaded from environment variables. See `.env.example`.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude (Anthropic)

**Specific purpose:** Debugging CUDA permission errors; fixing package conflicts that arose during execution due to circular dependencies; generating boilerplate code structure for profiling.

**Sections affected:** `TeamB/ncu_fp16.py` initial setup, `TeamB/colab_notebooks/teamb_profiler.ipynb` initial setup.

**How we verified correctness:** All reported experimental results (ECE, MCE, Brier score, tok/s, SM%, AI FLOPs/B) were produced by running scripts independently on the target hardware and confirmed against raw JSON outputs in `calibration_results/`, `vllm_results/`, `pytorch_profiler_results/`, and `nsight_profiler_results/`. All profiler-trace interpretations were confirmed against raw `.ncu-rep` files opened in the NVIDIA Nsight Compute GUI. No AI tool generated any numerical result or profiling interpretation.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

If you build on this work, please cite:

```bibtex
@misc{team29_2026_uai,
  title  = {Uncertainty-Aware Inference: How Post-Training Quantization Affects LLM Confidence Calibration},
  author = {Vyasamudri, Anubha and Ramesh, Rohit and Ballapur, Sanjita Chandan
            and Haque, Tamanna Ananna and Menon, Vishal},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/chuanbinp/uncertainty-aware-inference}
}
```

### Contact

Open a GitHub Issue or email the team via Columbia email (UNIs listed above).

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
