#!/bin/bash
# Run all Team A configs sequentially, one process each.
set -e

SAMPLES=${1:-""}  # optional: pass sample count as first arg, e.g. ./run_all.sh 10

for config in llama1-7b-fp16 llama1-7b-nf4 llama1-7b-awq-int4 llama1-7b-gptq-int4 llama1-7b-gptq-int8; do
    echo ""
    echo "############################################################"
    echo "Starting: $config"
    echo "############################################################"
    if [ -n "$SAMPLES" ]; then
        python TeamA/run_eval_args.py --config "$config" --samples "$SAMPLES"
    else
        python TeamA/run_eval_args.py --config "$config"
    fi
done

echo ""
echo "All configs done. Commit and push TeamA/results/ to share results."
