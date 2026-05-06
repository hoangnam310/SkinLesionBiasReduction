#!/usr/bin/env bash
# Generate synthetic skin-lesion images from a trained cGAN/WGAN-GP checkpoint.
# Usage:
#   1. Set CHECKPOINT below to the .pt file you want to sample from.
#   2. ./run_generate.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate gan

# ---- Checkpoint (edit this) ----
CHECKPOINT="outputs/<RUN_TIMESTAMP>/checkpoints/final_model.pt"

# ---- Output ----
OUTPUT_DIR="generated_images"

# ---- What to generate ----
NUM_SAMPLES=500                       # per (skin_tone x lesion_type) cell
TARGET_SKIN_TONES=(5 6)               # Fitzpatrick scale 1-6
TARGET_LESION_TYPES=(benign malignant non-neoplastic)
BATCH_SIZE=64
SEED=42

# ---- Architecture (must match training; 'auto' reads from the checkpoint) ----
GEN_ARCH="auto"                       # auto | upsample | deconv
USE_ATTENTION="auto"                  # auto | true | false
LATENT_DIM=100
EMBEDDING_DIM=50
NGF=64

# ---- Device ----
DEVICE=cuda                           # cuda | mps | cpu
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    echo "Edit CHECKPOINT in $0 to point at a trained .pt file." >&2
    exit 1
fi

CMD=(
    python src/generate.py
    --checkpoint "$CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --num_samples "$NUM_SAMPLES"
    --target_skin_tones "${TARGET_SKIN_TONES[@]}"
    --target_lesion_types "${TARGET_LESION_TYPES[@]}"
    --batch_size "$BATCH_SIZE"
    --seed "$SEED"
    --gen_arch "$GEN_ARCH"
    --use_attention "$USE_ATTENTION"
    --latent_dim "$LATENT_DIM"
    --embedding_dim "$EMBEDDING_DIM"
    --ngf "$NGF"
    --device "$DEVICE"
    --save_grid
    --create_csv
)

"${CMD[@]}"
