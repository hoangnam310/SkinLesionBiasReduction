#!/usr/bin/env bash
# Preprocess dataset/images (17k raw) → two output dirs, filtered to the ~11k
# rows in dataset/fitzpatrick17k_c.csv.
#
# Outputs:
#   dataset/images_center            — resize + center crop only (no segmentation)
#   dataset/images_center_sam2_edge  — Step 1 center crop + SAM2 skin mask + edge ROI
#
# Usage:
#   ./run_preprocess.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate gan

# ---- Inputs ----
CSV_PATH="dataset/fitzpatrick17k_c.csv"
IMAGE_DIR="dataset/images"

# ---- Output sizing (matches existing images_*_224 dirs) ----
SEG_SIZE=256
OUT_SIZE=224
CROP_FRAC=0.6
MIN_SKIN=0.85
JPEG_QUALITY=95

# ---- SAM2 ----
SAM2_CKPT="checkpoints/sam2_hiera_tiny.pt"
SAM2_CFG="sam2_hiera_t.yaml"
DEVICE=cuda
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# Fetch SAM2 checkpoint if missing
# if [ ! -f "$SAM2_CKPT" ]; then
#     echo "SAM2 checkpoint not found, downloading to $SAM2_CKPT ..."
#     mkdir -p "$(dirname "$SAM2_CKPT")"
#     wget -O "$SAM2_CKPT" \
#         https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
# fi

# # ---- 1. images_center: resize + center crop only ----
# OUT_CENTER="dataset/images_center"
# echo
# echo "=== [1/2] $OUT_CENTER (no segmentation, center strategy) ==="
# python src/preprocess_segmentation.py \
#     --csv_path "$CSV_PATH" \
#     --image_dir "$IMAGE_DIR" \
#     --output_dir "$OUT_CENTER" \
#     --backend none \
#     --strategy center \
#     --seg_size "$SEG_SIZE" \
#     --out_size "$OUT_SIZE" \
#     --jpeg_quality "$JPEG_QUALITY"

# ---- 2. images_center_sam2_edge: center crop + SAM2 mask + edge ROI ----
OUT_SAM2_EDGE="dataset/images_center_sam2_edge"
echo
echo "=== [2/2] $OUT_SAM2_EDGE (sam2 backend, edge strategy) ==="
python src/preprocess_segmentation.py \
    --csv_path "$CSV_PATH" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUT_SAM2_EDGE" \
    --backend sam2 \
    --strategy edge \
    --seg_size "$SEG_SIZE" \
    --out_size "$OUT_SIZE" \
    --crop_frac "$CROP_FRAC" \
    --min_skin "$MIN_SKIN" \
    --sam2_checkpoint "$SAM2_CKPT" \
    --sam2_config "$SAM2_CFG" \
    --device "$DEVICE" \
    --jpeg_quality "$JPEG_QUALITY"

echo
echo "Done. Counts:"
echo "  $OUT_CENTER:           $(ls "$OUT_CENTER" 2>/dev/null | wc -l) jpgs"
echo "  $OUT_SAM2_EDGE: $(ls "$OUT_SAM2_EDGE" 2>/dev/null | wc -l) jpgs"
