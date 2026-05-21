#!/bin/bash
# Run 1D Burgers experiments.
# Usage: ./run_burgers_1d.sh [--backends vanilla,canpinn,compile,triton] [--runs 5] [--gpu 0] [--max-epochs 30000] [--threshold 1e-3]

BACKENDS="vanilla,canpinn,compile,triton"
RUNS=1
GPU=0
MAX_EPOCHS=30000
THRESHOLD=1e-3

while [[ $# -gt 0 ]]; do
    case $1 in
        --backends) BACKENDS="$2"; shift 2 ;;
        --runs)     RUNS="$2";     shift 2 ;;
        --gpu)      GPU="$2";      shift 2 ;;
        --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2";  shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(conda run -n PINN_WE which python 2>/dev/null || echo python)"

IFS=',' read -ra BS <<< "$BACKENDS"
for B in "${BS[@]}"; do
    echo "====== $B ======"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$DIR/mlp_${B}.py" \
        --runs $RUNS --max-epochs $MAX_EPOCHS --threshold $THRESHOLD --gpu $GPU
done

echo "====== Done ======"
