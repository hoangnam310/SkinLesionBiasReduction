#!/usr/bin/env bash
# Train EfficientNetV2-B0 on the GAN-upsampled training set for ONE preprocessing
# variant (center or center_sam2_edge), then evaluate on the real-only test
# partition with bias-aware (per-Fitzpatrick) metrics.
#
# Pipeline (per variant):
#   1. Run ./run_generate.sh <variant> to produce synthetics into
#      generated_images/<variant>/ (filenames + metadata tagged with the GAN).
#   2. Merge generated_metadata.csv into a per-variant upsampled CSV
#      (synthetics added to partition=train only; val/test stay real-only).
#   3. Train baseline on the merged CSV against the variant's image_dir.
#   4. Run src/evaluate.py + src/metrics_report.py on the resulting checkpoint.
#
# Usage:
#   ./run_baseline_upsampled.sh center
#   ./run_baseline_upsampled.sh sam2          # alias for center_sam2_edge
#   ./run_baseline_upsampled.sh center_sam2_edge

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate gan

# ---- Variant dispatch ----
VARIANT="${1:-}"
if [ -z "$VARIANT" ]; then
    echo "ERROR: missing variant argument." >&2
    echo "Usage: $0 {center|sam2|center_sam2_edge}" >&2
    exit 2
fi

case "$VARIANT" in
    center)
        IMAGE_DIR="dataset/images_center"
        LABEL="center"
        ;;
    sam2|center_sam2_edge)
        IMAGE_DIR="dataset/images_center_sam2_edge"
        LABEL="center_sam2_edge"
        ;;
    *)
        echo "ERROR: unknown variant '$VARIANT'." >&2
        echo "Usage: $0 {center|sam2|center_sam2_edge}" >&2
        exit 2
        ;;
esac

# ---- Inputs / outputs ----
REAL_CSV="dataset/fitzpatrick17k_c.csv"
GEN_DIR="generated_images/${LABEL}"
SYNTH_CSV="${GEN_DIR}/generated_metadata.csv"
COMBINED_CSV="dataset/fitzpatrick17k_c_upsampled_${LABEL}.csv"
OUTPUT_DIR="outputs/baseline_efficientnet_upsampled_${LABEL}"

# ---- Training (matches run_baseline.sh for clean A/B) ----
IMAGE_SIZE=64
BATCH_SIZE=32
EPOCHS=40
LR=1e-4
WEIGHT_DECAY=1e-4
SEED=42
UNFREEZE_EPOCH=3         # head-only warmup, then full fine-tune
FINE_TUNE_LR=1e-5
NUM_WORKERS=4

# ---- Device ----
DEVICE=cuda              # cuda | mps | cpu
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# ---- Step 1: generate synthetics for THIS variant ----
if [ ! -x "./run_generate.sh" ]; then
    echo "ERROR: run_generate.sh not executable. Run: chmod +x run_generate.sh" >&2
    exit 1
fi

echo
echo "############################################################"
echo "# Variant: $LABEL"
echo "# 1/4 generating synthetics via ./run_generate.sh $LABEL"
echo "############################################################"
./run_generate.sh "$LABEL"

if [ ! -f "$SYNTH_CSV" ]; then
    echo "ERROR: ./run_generate.sh $LABEL did not produce $SYNTH_CSV." >&2
    exit 1
fi

# ---- Step 2: merge synthetics into the variant's image_dir + upsampled CSV ----
echo
echo "############################################################"
echo "# 2/4 merging synthetics -> $COMBINED_CSV (image_dir=$IMAGE_DIR)"
echo "############################################################"
python src/merge_synthetic.py \
    --real_csv "$REAL_CSV" \
    --synth_csv "$SYNTH_CSV" \
    --image_dir "$IMAGE_DIR" \
    --out_csv "$COMBINED_CSV" \
    --prefix "synth_${LABEL}"

# ---- Step 3: train ----
echo
echo "############################################################"
echo "# 3/4 training classifier -> $OUTPUT_DIR"
echo "############################################################"
python src/train_baseline_efficientnet.py \
    --csv_path "$COMBINED_CSV" \
    --image_dir "$IMAGE_DIR" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --seed "$SEED" \
    --freeze_backbone \
    --unfreeze_epoch "$UNFREEZE_EPOCH" \
    --fine_tune_lr "$FINE_TUNE_LR" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"
#--class_weights \

# ---- Step 4: bias-aware eval on the real test split ----
echo
echo "############################################################"
echo "# 4/4 evaluating on real test split"
echo "############################################################"
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
