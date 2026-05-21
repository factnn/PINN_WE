#!/bin/bash
# Run 2D TGV experiments.
# Usage: ./run_tgv_2d.sh [--backends mlp_vanilla,mlp_canpinn,mlp_compile,mlp_triton,cnn_canpinn,cnn_compile,cnn_triton] [--runs 1] [--gpu 0]

BACKENDS="mlp_vanilla,mlp_canpinn,mlp_compile,mlp_triton,cnn_canpinn,cnn_compile,cnn_triton"
RUNS=1
GPU=0
MAX_EPOCHS=50000
THRESHOLD=1e-3

while [[ $# -gt 0 ]]; do
    case $1 in
        --backends)   BACKENDS="$2";   shift 2 ;;
        --runs)       RUNS="$2";       shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
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
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$DIR/${B}.py" \
        --runs $RUNS --max-epochs $MAX_EPOCHS --threshold $THRESHOLD --gpu $GPU
done

echo "====== Done ======"
