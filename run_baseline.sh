#!/usr/bin/env bash
# Run EfficientNetV2-B0 baseline with conda env 'gan'
# Usage:
#   ./run_baseline.sh                    # Quick test (1 epoch, 2 steps, no pretrained)
#   ./run_baseline.sh --epochs 10        # Full run (override args)
#   For pretrained: ./run_baseline.sh --epochs 10  # remove --no_pretrained by editing below

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate gan

python src/train_baseline_efficientnet.py \
    --image_size 64 \
    --epochs 5 \
    --batch_size 32 \
    --device mps