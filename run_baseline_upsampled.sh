#!/usr/bin/env bash
# Train EfficientNetV2-B0 on the GAN-upsampled training set, then evaluate on
# the real-only test partition with bias-aware (per-Fitzpatrick) metrics.
#
# Pipeline:
#   1. Merge generated_metadata.csv into a new partitioned CSV
#      (synthetics added to partition=train only — val/test stay real-only).
#   2. Train baseline on the merged CSV.
#   3. Run src/evaluate.py on the resulting checkpoint.
#
# Pair this with run_baseline.sh (real-only, same SEED, same EPOCHS) to get
# a clean A/B comparison on the same real test set.
#
# Usage:
#   ./run_baseline_upsampled.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate gan

# ---- Inputs ----
REAL_CSV="dataset/fitzpatrick17k_c.csv"
SYNTH_CSV="generated_images/generated_metadata.csv"
IMAGE_DIR="dataset/images"
COMBINED_CSV="dataset/fitzpatrick17k_c_upsampled.csv"

# ---- Training ----
IMAGE_SIZE=64
BATCH_SIZE=32
EPOCHS=30
LR=1e-4
WEIGHT_DECAY=1e-4
DROPOUT=0.3
SEED=42                  # MUST match run_baseline.sh for a clean A/B
UNFREEZE_EPOCH=-1        # set e.g. 10 to fine-tune backbone after head warmup
FINE_TUNE_LR=1e-5
NUM_WORKERS=4

OUTPUT_DIR="outputs/baseline_efficientnet_upsampled"

# ---- Device ----
DEVICE=cuda              # cuda | mps | cpu
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# ---- Step 1: build the merged CSV (idempotent — re-encoded JPEGs are skipped) ----
if [ ! -f "$SYNTH_CSV" ]; then
    echo "ERROR: $SYNTH_CSV not found. Run ./run_generate.sh first." >&2
    exit 1
fi

python src/merge_synthetic.py \
    --real_csv "$REAL_CSV" \
    --synth_csv "$SYNTH_CSV" \
    --image_dir "$IMAGE_DIR" \
    --out_csv "$COMBINED_CSV"

# ---- Step 2: train ----
python src/train_baseline_efficientnet.py \
    --csv_path "$COMBINED_CSV" \
    --image_dir "$IMAGE_DIR" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --seed "$SEED" \
    --unfreeze_epoch "$UNFREEZE_EPOCH" \
    --fine_tune_lr "$FINE_TUNE_LR" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# ---- Step 3: bias-aware eval on the real-only test split ----
LATEST_RUN="$(ls -1dt "$OUTPUT_DIR"/*/ 2>/dev/null | head -n1)"
if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}checkpoint.pt" ]; then
    python src/evaluate.py \
        --checkpoint "${LATEST_RUN}checkpoint.pt" \
        --split test \
        --device "$DEVICE"
else
    echo "WARN: could not locate checkpoint under $OUTPUT_DIR for evaluation." >&2
fi
