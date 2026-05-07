#!/usr/bin/env bash
# Overnight pipeline:
#   1. Train two cGANs (run_main.sh -> trains both center + center_sam2_edge variants).
#   2. For each cGAN, find best-FID checkpoint via scripts/find_best_checkpoint.py.
#   3. Generate FST 5/6 synthetics from each best checkpoint into per-variant dirs.
#   4. Merge synthetics into per-variant upsampled CSVs (synthetic JPEGs land alongside
#      the variant's preprocessed reals so the classifier loads them through one image_dir).
#   5. Train EfficientNet on each upsampled variant.
#   6. Evaluate each on the real test split; write per-variant reports.
#
# stop-on-error: any failure aborts the pipeline (set -e).
#
# Usage:
#   ./run_pipeline_overnight.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate gan

# ---- Per-variant configuration ----
VARIANTS=(
    "center            dataset/images_center"
    "center_sam2_edge  dataset/images_center_sam2_edge"
)

# ---- Real test data ----
REAL_CSV="dataset/fitzpatrick17k_c.csv"

# ---- Generation plan (per cGAN) ----
GEN_BATCH_SIZE=64
GEN_SEED=42
# Format: "<fitzpatrick> <lesion_type> <count>"
CELLS=(
    "5 benign         500"
    "5 malignant      500"
    "5 non-neoplastic 200"
    "6 benign         500"
    "6 malignant      500"
    "6 non-neoplastic 200"
)

# ---- Classifier hyperparams (match run_baseline.sh exactly) ----
IMAGE_SIZE=64
BATCH_SIZE=32
EPOCHS=40
LR=1e-4
WEIGHT_DECAY=1e-4
SEED=42
UNFREEZE_EPOCH=3
FINE_TUNE_LR=1e-5
NUM_WORKERS=4

# ---- Device ----
DEVICE=cuda
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# ---- Step 1: train both cGANs ----
echo
echo "############################################################"
echo "# STEP 1/6: train both cGANs (run_main.sh)"
echo "############################################################"
./run_main.sh

# ---- Steps 2-6: per-variant generation + classifier training ----
for entry in "${VARIANTS[@]}"; do
    read -r LABEL IMAGE_DIR <<< "$entry"

    echo
    echo "############################################################"
    echo "# Variant: $LABEL  (image_dir=$IMAGE_DIR)"
    echo "############################################################"

    # Find the most recent wgan_<label>_<timestamp>/ for this variant
    RUN_DIR="$(ls -1dt outputs/wgan_${LABEL}_*/ 2>/dev/null | head -n1 | sed 's:/$::')"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: no wgan_${LABEL}_* run dir found in outputs/" >&2
        exit 1
    fi
    echo "Using cGAN run dir: $RUN_DIR"

    # ---- Step 2: best checkpoint ----
    CHECKPOINT="$(python scripts/find_best_checkpoint.py "$RUN_DIR")"
    echo "Best checkpoint: $CHECKPOINT"

    # ---- Step 3: generate synthetics for FST 5/6 cells ----
    GEN_DIR="generated_images/${LABEL}"
    rm -rf "$GEN_DIR"
    mkdir -p "$GEN_DIR"

    for cell in "${CELLS[@]}"; do
        read -r ST LT N <<< "$(echo "$cell" | tr -s ' ')"
        echo
        echo "Generating $LABEL: FST=$ST, lesion=$LT, n=$N"
        python src/generate.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$GEN_DIR" \
            --num_samples "$N" \
            --target_skin_tones "$ST" \
            --target_lesion_types "$LT" \
            --batch_size "$GEN_BATCH_SIZE" \
            --seed "$GEN_SEED" \
            --gen_arch auto \
            --use_attention auto \
            --device "$DEVICE" \
            --save_grid \
            --create_csv

        # rename per-cell metadata so the next cell does not clobber it
        mv "$GEN_DIR/generated_metadata.csv" \
           "$GEN_DIR/generated_metadata_st${ST}_${LT}.csv"
    done

    # Concat per-cell CSVs into one generated_metadata.csv
    python - <<PY
import pandas as pd
from pathlib import Path

out_dir = Path("$GEN_DIR")
parts = sorted(out_dir.glob("generated_metadata_st*_*.csv"))
combined = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
combined.to_csv(out_dir / "generated_metadata.csv", index=False)
print(f"Combined {len(parts)} CSVs -> {out_dir/'generated_metadata.csv'} ({len(combined)} rows)")
print(combined.groupby(["fitzpatrick_scale", "three_partition_label"]).size().unstack(fill_value=0))
PY

    # ---- Step 4: merge synthetics into the variant's image_dir + upsampled CSV ----
    UPSAMPLED_CSV="dataset/fitzpatrick17k_c_upsampled_${LABEL}.csv"
    echo
    echo "Merging synthetics into $IMAGE_DIR + $UPSAMPLED_CSV"
    python src/merge_synthetic.py \
        --real_csv "$REAL_CSV" \
        --synth_csv "$GEN_DIR/generated_metadata.csv" \
        --image_dir "$IMAGE_DIR" \
        --out_csv "$UPSAMPLED_CSV" \
        --prefix "synth_${LABEL}"

    # ---- Step 5: train EfficientNet on the upsampled variant ----
    OUTPUT_DIR="outputs/baseline_efficientnet_upsampled_${LABEL}"
    echo
    echo "Training classifier on $UPSAMPLED_CSV -> $OUTPUT_DIR"
    python src/train_baseline_efficientnet.py \
        --csv_path "$UPSAMPLED_CSV" \
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

    # ---- Step 6: evaluate on real test split ----
    LATEST_RUN="$(ls -1dt "$OUTPUT_DIR"/*/ 2>/dev/null | head -n1)"
    if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}checkpoint.pt" ]; then
        echo
        echo "Evaluating on real test split (${LATEST_RUN}checkpoint.pt)"
        python src/evaluate.py \
            --checkpoint "${LATEST_RUN}checkpoint.pt" \
            --split test \
            --device "$DEVICE"
        python src/metrics_report.py --json logs/evaluation_metrics.json
    else
        echo "ERROR: classifier checkpoint missing under $OUTPUT_DIR" >&2
        exit 1
    fi
done

echo
echo "############################################################"
echo "# Pipeline complete. Per-variant outputs:"
echo "############################################################"
for entry in "${VARIANTS[@]}"; do
    read -r LABEL _IMG <<< "$entry"
    echo "  $LABEL:"
    echo "    cGAN:           $(ls -1dt outputs/wgan_${LABEL}_*/ 2>/dev/null | head -n1)"
    echo "    classifier:     outputs/baseline_efficientnet_upsampled_${LABEL}/"
    echo "    upsampled CSV:  dataset/fitzpatrick17k_c_upsampled_${LABEL}.csv"
done
echo
echo "Reports written to logs/<timestamp>_report.md (one per evaluate run)."
