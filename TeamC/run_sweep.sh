#!/bin/bash
# Run all quantized configurations for Llama-2 13B PTQ sweep.
# FP16 baseline is assumed to already exist from Week 2.
# Run from project root: bash TeamC/run_sweep.sh
set -e

cd "$(dirname "$0")"

for config in gptq_int8 gptq_int4 awq_int4 bnb_nf4; do
    echo "=== Running: $config ==="
    python eval_quantized.py --config "$config" --dataset all --split validation
done

echo "=== Sweep complete. Running comparison ==="
python cross_model_compare.py
