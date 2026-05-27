#!/bin/bash
# ============================================================
#  Track 2 Full Run: 8 cases × 7 backends × 300k epochs
#  4 GPUs, each GPU runs 2 cases sequentially
#  Reusable: just change CASES/GPUS/MAX_EPOCHS
# ============================================================
set -e
cd "$(dirname "$0")"  # cd to triton_accelerate/

MAX_EPOCHS=${1:-300000}
LOGDIR="/tmp/track2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# 8 cases assigned to 4 GPUs (2 per GPU)
GPU0_CASES=("burgers_1d_steady" "ldc_2d")
GPU1_CASES=("burgers_1d_unsteady" "ldc_3d")
GPU2_CASES=("tgv_2d" "transport_2d")
GPU3_CASES=("sod_1d" "tgv_3d")

run_gpu() {
    local gpu=$1
    shift
    local cases=("$@")

    for case in "${cases[@]}"; do
        local log="$LOGDIR/${case}.log"
        echo "[GPU $gpu] Starting $case (max_epochs=$MAX_EPOCHS) → $log"
        python run.py --case "$case" --backend all --track 2 \
            --gpu "$gpu" --max-epochs "$MAX_EPOCHS" > "$log" 2>&1
        echo "[GPU $gpu] Finished $case at $(date)"
    done
}

echo "============================================================"
echo "  Track 2 Full Run"
echo "  8 cases, 4 GPUs, $MAX_EPOCHS epochs each"
echo "  Logs: $LOGDIR/"
echo "  Started: $(date)"
echo "============================================================"
echo ""
echo "  GPU 0: ${GPU0_CASES[*]}"
echo "  GPU 1: ${GPU1_CASES[*]}"
echo "  GPU 2: ${GPU2_CASES[*]}"
echo "  GPU 3: ${GPU3_CASES[*]}"
echo ""

# Launch 4 GPU workers in parallel
run_gpu 0 "${GPU0_CASES[@]}" &
run_gpu 1 "${GPU1_CASES[@]}" &
run_gpu 2 "${GPU2_CASES[@]}" &
run_gpu 3 "${GPU3_CASES[@]}" &

# Wait for all GPUs to finish
wait

echo ""
echo "============================================================"
echo "  All 8 cases complete at $(date)"
echo "============================================================"

# Generate plots for cases with checkpoints
echo ""
echo "Generating plots..."
for case in burgers_1d_steady burgers_1d_unsteady ldc_2d ldc_3d tgv_2d tgv_3d transport_2d sod_1d; do
    if ls output/$case/model_*.pt 1>/dev/null 2>&1; then
        echo "  Plotting $case..."
        python -m plots.$case 2>/dev/null || echo "    (plot failed)"
    fi
done

# Print summary
echo ""
echo "============================================================"
echo "  Results Summary"
echo "============================================================"
for case in burgers_1d_steady burgers_1d_unsteady ldc_2d ldc_3d tgv_2d tgv_3d transport_2d sod_1d; do
    echo ""
    echo "=== $case ==="
    if [ -f "$LOGDIR/${case}.log" ]; then
        grep -E "T2S|Final_Loss|L2_Error|Total_Epochs|Avg_Step" "$LOGDIR/${case}.log" | head -21
    else
        echo "  (no log)"
    fi
done

echo ""
echo "Logs: $LOGDIR/"
echo "Plots: output/*/figures/"
echo "CSV: output/*/history_*.csv"
