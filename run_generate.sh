#!/usr/bin/env bash
# Generate synthetic skin-lesion images from a trained cGAN/WGAN-GP checkpoint.
#
# Per-cell counts: each (Fitzpatrick tone × lesion type) cell can be sized
# independently via the CELLS array below. Tilt the budget toward the
# under-represented + clinically important cells (FST 5/6 malignant, benign);
# keep non-neoplastic small since it is already the majority class.
#
# Usage:
#   1. Set CHECKPOINT below to the .pt file you want to sample from.
#   2. Edit CELLS to control per-cell sample counts.
#   3. ./run_generate.sh

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

# ---- Per-cell generation plan ----
# Format: "<fitzpatrick_scale> <lesion_type> <num_samples>"
# Tilt toward minority + clinically important cells.
CELLS=(
    "5 benign         300"
    "5 malignant      400"
    "5 non-neoplastic 200"
    "6 benign         400"
    "6 malignant      400"
    "6 non-neoplastic 200"
)

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

mkdir -p "$OUTPUT_DIR"

# Clear any stale combined metadata so a partial rerun cannot leave it
# inconsistent with the per-cell CSVs we are about to write.
rm -f "$OUTPUT_DIR/generated_metadata.csv"

TOTAL=0
for cell in "${CELLS[@]}"; do
    # collapse internal whitespace then split into ST / LT / N
    read -r ST LT N <<< "$(echo "$cell" | tr -s ' ')"

    echo
    echo "============================================================"
    echo "Generating cell: Fitzpatrick $ST, $LT, n=$N"
    echo "============================================================"

    python src/generate.py \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --num_samples "$N" \
        --target_skin_tones "$ST" \
        --target_lesion_types "$LT" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --gen_arch "$GEN_ARCH" \
        --use_attention "$USE_ATTENTION" \
        --latent_dim "$LATENT_DIM" \
        --embedding_dim "$EMBEDDING_DIM" \
        --ngf "$NGF" \
        --device "$DEVICE" \
        --save_grid \
        --create_csv

    # generate.py always writes generated_metadata.csv — rename it so the
    # next cell does not clobber it. We will concat them all at the end.
    mv "$OUTPUT_DIR/generated_metadata.csv" \
       "$OUTPUT_DIR/generated_metadata_st${ST}_${LT}.csv"

    TOTAL=$((TOTAL + N))
done

# ---- Concatenate per-cell metadata CSVs into a single generated_metadata.csv ----
python - <<PY
import glob
import pandas as pd
from pathlib import Path

out_dir = Path("$OUTPUT_DIR")
parts = sorted(out_dir.glob("generated_metadata_st*_*.csv"))
if not parts:
    raise SystemExit("No per-cell metadata CSVs found; nothing to concat.")

dfs = [pd.read_csv(p) for p in parts]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(out_dir / "generated_metadata.csv", index=False)

print()
print(f"Combined {len(parts)} per-cell CSVs -> {out_dir/'generated_metadata.csv'}")
print(f"Total synthetic rows: {len(combined)}")
print()
print("Per-cell counts (from combined CSV):")
print(combined.groupby(["fitzpatrick_scale", "three_partition_label"]).size().unstack(fill_value=0))
PY

echo
echo "Done. Total samples requested across cells: $TOTAL"
