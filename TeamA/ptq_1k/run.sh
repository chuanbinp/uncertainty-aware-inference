#!/bin/bash
# TeamA/ptq_1k/run.sh
# Run all three NF4 PTQ baselines at 1,000 samples.
# Results → TeamA/results/ptq_1k/{config}/
#
# Usage:
#   bash TeamA/ptq_1k/run.sh                   # all three models
#   bash TeamA/ptq_1k/run.sh llama1-7b-nf4     # single model
#
# For Llama-2 13B (gated model), set your token first:
#   export HF_TOKEN=hf_...

set -e
cd "$(dirname "$0")/../.."   # run from project root

if [ -n "$1" ]; then
    echo "Running single config: $1"
    python TeamA/ptq_1k/run_ptq_1k.py --config "$1"
else
    echo "Running all three NF4 configs sequentially..."
    python TeamA/ptq_1k/run_ptq_1k.py --config llama1-7b-nf4
    python TeamA/ptq_1k/run_ptq_1k.py --config mistral-7b-nf4
    python TeamA/ptq_1k/run_ptq_1k.py --config llama2-13b-nf4
    echo ""
    echo "All done. Results in TeamA/ptq_1k/results/"
fi
