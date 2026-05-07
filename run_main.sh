#!/usr/bin/env bash
# Train two conditional WGAN-GP cGANs back-to-back, one per preprocessing variant:
#   1. dataset/images_center            — resize + center crop only
#   2. dataset/images_center_sam2_edge  — center crop + SAM2 mask + edge ROI
#
# Architecture per run (matches the best prior run, FID ~150-200 on raw):
#   - Conditional WGAN-GP (Wasserstein loss + gradient penalty)
#   - Generator: Upsample + Conv ("upsample" arch)
#   - SAGAN-style self-attention at 32x32 in G and Critic (--use_attention)
#   - Critic LayerNorm (Gulrajani 2017)
#   - DiffAug: color, translation, cutout
#   - image_size=64 (matches downstream classifier; fast iteration)
#
# Each run writes to outputs/wgan_<timestamp>/. Edit IMAGE_DIRS to skip one.
#
# Usage:
#   ./run_main.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate gan

# ---- Data / output ----
DATASET_CSV="dataset/fitzpatrick17k_c.csv"
OUTPUT_DIR="outputs"

# Two preprocessing variants, trained sequentially.
IMAGE_DIRS=(
    "dataset/images_center"
    "dataset/images_center_sam2_edge"
)

# ---- Training ----
EPOCHS=500
BATCH_SIZE=128
LR_G=1e-4
LR_C=1e-4
BETA1=0.0
BETA2=0.9
N_CRITIC=5
LAMBDA_GP=10.0

# ---- Augmentation / architecture ----
DIFFAUG_POLICY="color,translation,cutout"
GEN_ARCH="upsample"          # "upsample" (Upsample+Conv) | "deconv" (64x64 only)
USE_ATTENTION=true           # SAGAN self-attention at 32x32 in G and Critic
CRITIC_NORM="layer"          # "instance" | "layer" (Gulrajani 2017)
LATENT_DIM=100
EMBEDDING_DIM=50
NGF=64
NDF=64
IMAGE_SIZE=64                # downsampled from 224 preprocessed input at load time

# ---- Logging / eval ----
CHECKPOINT_INTERVAL=10
SAMPLE_INTERVAL=5
FID_INTERVAL=20
FID_NUM_SAMPLES=1000
NUM_WORKERS=8

# ---- Device ----
DEVICE=cuda
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# ---- Resume (optional) ----
#RESUME_CKPT="outputs/wgan_center_20260507_005647/checkpoints/checkpoint_epoch_0130.pt"

if [ "$USE_ATTENTION" = "true" ]; then
    ATTENTION_FLAG="--use_attention"
else
    ATTENTION_FLAG="--no-use_attention"
fi

run_one() {
    local image_dir="$1"
    # Derive run name from the dir basename, stripping the leading "images_" prefix.
    # e.g. dataset/images_center_sam2_edge -> center_sam2_edge
    local run_name
    run_name="$(basename "$image_dir")"
    run_name="${run_name#images_}"

    echo
    echo "============================================================"
    echo "Training cGAN on $image_dir  (run_name=$run_name)"
    echo "============================================================"

    local cmd=(
        python src/train_wgan.py
        --csv_path "$DATASET_CSV"
        --image_dir "$image_dir"
        --run_name "$run_name"
        --epochs "$EPOCHS"
        --batch_size "$BATCH_SIZE"
        --lr_g "$LR_G"
        --lr_c "$LR_C"
        --beta1 "$BETA1"
        --beta2 "$BETA2"
        --n_critic "$N_CRITIC"
        --lambda_gp "$LAMBDA_GP"
        --diffaug_policy "$DIFFAUG_POLICY"
        --gen_arch "$GEN_ARCH"
        "$ATTENTION_FLAG"
        --critic_norm "$CRITIC_NORM"
        --latent_dim "$LATENT_DIM"
        --embedding_dim "$EMBEDDING_DIM"
        --ngf "$NGF"
        --ndf "$NDF"
        --image_size "$IMAGE_SIZE"
        --checkpoint_interval "$CHECKPOINT_INTERVAL"
        --sample_interval "$SAMPLE_INTERVAL"
        --fid_interval "$FID_INTERVAL"
        --fid_num_samples "$FID_NUM_SAMPLES"
        --num_workers "$NUM_WORKERS"
        --output_dir "$OUTPUT_DIR"
        --device "$DEVICE"
    )

    if [ -n "${RESUME_CKPT:-}" ]; then
        cmd+=(--resume "$RESUME_CKPT")
    fi

    "${cmd[@]}"
}

for d in "${IMAGE_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "WARN: $d does not exist yet — skipping. Run ./run_preprocess.sh first." >&2
        continue
    fi
    run_one "$d"
done
echo
echo "Done. Two cGAN runs complete. Latest run dirs:"
ls -1dt outputs/wgan_*/ 2>/dev/null | head -2
