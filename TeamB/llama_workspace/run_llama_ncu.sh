#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_llama_ncu.sh  —  ncu profiling sweep for Llama-2 7B and 13B
# Run from the TeamB directory after copying all ncu_llama2_*.py scripts there
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash run_llama_ncu.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

NCU=/usr/local/cuda/bin/ncu
PY_TAMANNA=~/miniconda3/envs/tamanna/bin/python
PY_UAI=~/miniconda3/envs/uai/bin/python
PY=$PY_UAI   # use tamanna env for gptqmodel
OUTDIR=$(pwd)/results/ncu
mkdir -p "$OUTDIR"

METRICS="gpu__time_duration.sum,dram__bytes.sum,sm__inst_executed_pipe_tensor.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"

run_ncu() {
    local script=$1 outname=$2 skip=$3 count=$4 regex=$5
    echo ""
    echo "=== Profiling $outname ==="
    sudo rm -f /tmp/nsight-compute-lock
    sudo -E GPTQMODEL_FORCE_KERNEL=torch "$NCU" \
        -o "$OUTDIR/$outname" \
        --metrics "$METRICS" \
        --launch-skip "$skip" --launch-count "$count" \
        --kernel-name "regex:$regex" \
        --force-overwrite \
        "$PY" "$script"

    echo "[extract] $outname → CSV"
    "$NCU" --import "$OUTDIR/${outname}.ncu-rep" \
        --csv --print-units base 2>/dev/null \
        > "$OUTDIR/${outname}_metrics.csv"
    echo "[done] $outname"
}

# ── Llama-2 7B ──────────────────────────────────────────────────────────────
#run_ncu ncu_llama1_7b_fp16.py       llama1_7b_fp16        30  20 "ampere_fp16_s16816gemm|fmha_cutlass"
#run_ncu ncu_llama1_7b_gptq_int4.py  llama1_7b_gptq_int4   50  20 "ampere_fp16_s16816gemm|ampere_bf16_s16816gemm|fmha_cutlass"
#run_ncu ncu_llama1_7b_gptq_int8.py  llama1_7b_gptq_int8   50  20 "ampere_fp16_s16816gemm|ampere_fp16_s1688gemm|fmha_cutlass"
#run_ncu ncu_llama1_7b_awq_int4.py   llama1_7b_awq_int4    50  20 "awq_gemm|fmha_cutlass"
#run_ncu ncu_llama1_7b_nf4.py        llama1_7b_nf4          0  20 "kDequantizeBlockwise|ampere_fp16_s16816gemm|fmha_cutlass"

# ── Llama-2 13B ─────────────────────────────────────────────────────────────
# Note: 13B needs ~26 GB for FP16 — exceeds L4's 24 GB VRAM
# FP16 will OOM; quantized versions (4-bit ~7 GB, 8-bit ~14 GB) will fit
echo ""
echo "NOTE: Llama-2 13B FP16 requires 26 GB — will OOM on L4 (24 GB)"
echo "Skipping 13B FP16, running quantized configs only"

# run_ncu ncu_llama2_13b_gptq_int4.py llama2_13b_gptq_int4  50  20 "ampere_fp16_s16816gemm|ampere_bf16_s16816gemm|fmha_cutlass"
run_ncu ncu_llama2_13b_gptq_int8.py llama2_13b_gptq_int8  50  20 "ampere_fp16_s16816gemm|ampere_fp16_s1688gemm|fmha_cutlass"
# run_ncu ncu_llama2_13b_awq_int4.py  llama2_13b_awq_int4   50  20 "awq_gemm|fmha_cutlass"
# run_ncu ncu_llama2_13b_nf4.py       llama2_13b_nf4         0  20 "kDequantizeBlockwise|ampere_fp16_s16816gemm|fmha_cutlass"

echo ""
echo "All done. CSVs in $OUTDIR/"
echo "Upload *_metrics.csv files to generate the combined roofline plot."
