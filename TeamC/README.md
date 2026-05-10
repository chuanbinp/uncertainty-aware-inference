# Team C

Team C focuses on running calibration sweeps for Llama-2 13B, cross-model evaluations, pareto-analysis and uncertainty-aware routing strategies for quantized language models.

## Prerequisites

To run the experiments, ensure you have:

1. **HuggingFace Account and Token**: Required for accessing gated models (Llama-2).
   - Sign up at [huggingface.co](https://huggingface.co)
   - Generate a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Set environment variable: `export HF_TOKEN=hf_...` or add it to Colab Secrets

2. **Weights & Biases Account**: Used for experiment tracking.
   - Sign up at [wandb.ai](https://wandb.ai)
   - Login: `wandb login`

3. **GPU Access**: We used A100 40GB on Google Colab for our experiments

## Environment Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/chuanbinp/uncertainty-aware-inference.git
   cd uncertainty-aware-inference
   ```

2. **Set up conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate uncertainty_aware_env
   ```

## Data Setup

The evaluation scripts automatically download required datasets from HuggingFace:
- hellaswag
- triviaqa
- pubmedqa

No manual data preparation is required.

## Scripts

### `configs.py`
Defines quantization configurations for Llama-2 13B PTQ sweep. Contains a model registry with different quantization types (FP16, GPTQ INT4/INT8, NF4, AWQ INT4) and their HuggingFace model IDs.

### `run_eval.py`
Runs calibration evaluation for one quantization configuration across multiple datasets (hellaswag, triviaqa, pubmedqa). Uses Weights & Biases for logging results.

Supported configs:
- `llama2-13b-fp16`: FP16 baseline (requires HF_TOKEN)
- `llama2-13b-nf4`: NF4 bitsandbytes (requires HF_TOKEN)
- `llama2-13b-awq-int4`: AWQ INT4
- `llama2-13b-gptq-int4`: GPTQ INT4
- `llama2-13b-gptq-int8`: GPTQ INT8

Usage:
```bash
python TeamC/run_eval.py
HF_TOKEN=hf_... python TeamC/run_eval.py  # for gated models
```

### `llama2_13b.ipynb`
A Google Colab notebook for running calibration evaluations on different quantization configurations of Llama-2 13B. The notebook:
- Clones the repository
- Installs required dependencies
- Sets up authentication for HuggingFace and Weights & Biases
- Runs `run_eval.py` for different configs (FP16, AWQ INT4, NF4)
- Zips and downloads the results

### `pareto_script.py`
Performs cross-model Pareto frontier analysis. Loads JSON result summaries from all teams (TeamA, TeamB, TeamC) and identifies Pareto-dominant configurations across throughput, accuracy, and calibration error. Generates plots and analysis of the Pareto frontier.

Usage:
```bash
python TeamC/pareto_script.py
python TeamC/pareto_script.py --output_dir TeamC/results/pareto
```

### `routing_simulation.py`
Simulates uncertainty-based routing strategy. Implements a two-tier serving approach: if the quantized model's confidence exceeds a threshold, serve from it (cheap); otherwise escalate to FP16 (expensive). Sweeps confidence thresholds and computes effective accuracy, calibration, and cost.

Usage:
```bash
python TeamC/routing_simulation.py
python TeamC/routing_simulation.py --dataset arc_challenge --num_thresholds 200
```

## Results Locations

### `analysis_results/`
Contains results from analysis scripts. [Pareto Dashboard Results](https://chuanbinp.github.io/uncertainty-aware-inference/TeamC/analysis_results/pareto/pareto_comparison.html)

- `pareto/`: Output from Pareto frontier analysis, including plots and dominant configuration summaries.
- `routing/`: Results from routing simulation, including threshold sweeps and cost-accuracy tradeoffs.

### `sweep_results/`
Contains evaluation results for each quantization configuration sweep.

- `llama2-13b-awq-int4/`: AWQ INT4 evaluation results
- `llama2-13b-gptq-int4/`: GPTQ INT4 evaluation results
- `llama2-13b-gptq-int8/`: GPTQ INT8 evaluation results
- `llama2-13b-nf4/`: NF4 evaluation results
- `llama2-13b_16bit_fp16/`: FP16 baseline evaluation results

Each subdirectory contains JSON summaries and tensor files with calibration metrics, accuracy scores, and evaluation outputs.

### `archive/`
Archived results from previous runs or deprecated configurations.
