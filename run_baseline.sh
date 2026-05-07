#!/usr/bin/env bash
# Train EfficientNetV2-B0 baseline on the real-only dataset preprocessed with
# resize + center crop (no segmentation). Then evaluate on the real test split
# with bias-aware (per-Fitzpatrick) metrics.
#
# Pair with run_baseline_upsampled.sh (same SEED, EPOCHS, hyperparams) for a
# clean A/B against the GAN-upsampled version.
#
# Usage:
#   ./run_baseline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate gan

# ---- Inputs ----
CSV_PATH="dataset/fitzpatrick17k_c.csv"
IMAGE_DIR="dataset/images_center_sam2_edge"

# ---- Training (matches run_baseline_upsampled.sh for clean A/B) ----
IMAGE_SIZE=64
BATCH_SIZE=32
EPOCHS=40
LR=1e-4
WEIGHT_DECAY=1e-4
SEED=42                  # MUST match run_baseline_upsampled.sh
UNFREEZE_EPOCH=3         # head-only warmup, then full fine-tune
FINE_TUNE_LR=1e-5
NUM_WORKERS=4

OUTPUT_DIR="outputs/baseline_efficientnet_center"

# ---- Device ----
DEVICE=cuda              # cuda | mps | cpu
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# ---- Step 1: train ----
python src/train_baseline_efficientnet.py \
    --csv_path "$CSV_PATH" \
    --image_dir "$IMAGE_DIR" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --class_weights \
    --freeze_backbone \
    --unfreeze_epoch "$UNFREEZE_EPOCH" \
    --fine_tune_lr "$FINE_TUNE_LR" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# ---- Step 2: bias-aware eval on the real test split ----
LATEST_RUN="$(ls -1dt "$OUTPUT_DIR"/*/ 2>/dev/null | head -n1)"
if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}checkpoint.pt" ]; then
    python src/evaluate.py \
        --checkpoint "${LATEST_RUN}checkpoint.pt" \
        --split test \
        --device "$DEVICE"

    python src/metrics_report.py \
        --json "logs/evaluation_metrics.json"
else
    echo "WARN: could not locate checkpoint under $OUTPUT_DIR for evaluation." >&2
fi
