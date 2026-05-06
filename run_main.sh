#!/usr/bin/env bash
# Run WGAN-GP (cgan.py Critic + train_wgan.py) with conda env 'gan'.
# Mirrors the colab_cgan_training.ipynb WGAN2 (attention) cell.
# Usage:
#   ./run_main.sh
# Edit the variables below to change a run; resume by uncommenting RESUME_CKPT.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate gan

# ---- Data / output ----
DATASET_CSV="dataset/fitzpatrick17k_cleaned.csv"
TRAIN_IMAGE_DIR="dataset/images"
OUTPUT_DIR="outputs"

# ---- Training ----
EPOCHS=200
BATCH_SIZE=64
LR_G=1e-4
LR_C=1e-4
BETA1=0.0
BETA2=0.9
N_CRITIC=5
LAMBDA_GP=10.0

# ---- Augmentation / architecture ----
DIFFAUG_POLICY="color,translation,cutout"
GEN_ARCH="upsample"          # "upsample" (Upsample+Conv) | "deconv" (ConvTranspose2d)
USE_ATTENTION=true           # SAGAN-style SelfAttention at 32x32 in G and Critic
CRITIC_NORM="layer"          # "instance" | "layer" (Gulrajani 2017's recommendation)
LATENT_DIM=100
EMBEDDING_DIM=50
NGF=64
NDF=64

# ---- Logging / eval ----
CHECKPOINT_INTERVAL=10
SAMPLE_INTERVAL=5
FID_INTERVAL=20
FID_NUM_SAMPLES=1000
NUM_WORKERS=4

# ---- Device ----
DEVICE=mps                   # cuda | mps | cpu

# ---- Resume (optional) ----
# RESUME_CKPT="outputs/wgan_20260131_230948/checkpoints/final_model.pt"

# train_wgan.py uses argparse.BooleanOptionalAction for --use_attention
if [ "$USE_ATTENTION" = "true" ]; then
    ATTENTION_FLAG="--use_attention"
else
    ATTENTION_FLAG="--no-use_attention"
fi

CMD=(
    python src/train_wgan.py
    --csv_path "$DATASET_CSV"
    --image_dir "$TRAIN_IMAGE_DIR"
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
    --checkpoint_interval "$CHECKPOINT_INTERVAL"
    --sample_interval "$SAMPLE_INTERVAL"
    --fid_interval "$FID_INTERVAL"
    --fid_num_samples "$FID_NUM_SAMPLES"
    --num_workers "$NUM_WORKERS"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE"
)

if [ -n "${RESUME_CKPT:-}" ]; then
    CMD+=(--resume "$RESUME_CKPT")
fi

"${CMD[@]}"
