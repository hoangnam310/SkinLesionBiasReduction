"""Merge GAN-generated images into the partitioned training CSV.

Reads `generated_images/generated_metadata.csv` (produced by `generate.py`),
re-encodes each synthetic PNG as JPEG into `dataset/images/synth_<id>.jpg`
so the existing SkinLesionDataset loader (which builds paths as
`<image_dir>/<md5hash>.jpg`) can find them, and writes a new CSV that
mirrors `dataset/fitzpatrick17k_c.csv` with synthetic rows appended under
partition='train'.

Real val/test rows are passed through untouched. Synthetics are NEVER
added to val or test — that would measure GAN quality, not classifier
generalization.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge synthetic images into the partitioned training CSV."
    )
    parser.add_argument("--real_csv", type=str,
                        default="dataset/fitzpatrick17k_c.csv",
                        help="Partitioned real-data CSV (must have md5hash, "
                             "fitzpatrick_scale, three_partition_label, partition)")
    parser.add_argument("--synth_csv", type=str,
                        default="generated_images/generated_metadata.csv",
                        help="Metadata CSV emitted by src/generate.py")
    parser.add_argument("--image_dir", type=str, default="dataset/images",
                        help="Directory the trainer reads images from. Synthetic "
                             "JPEGs will be written here as synth_<id>.jpg.")
    parser.add_argument("--out_csv", type=str,
                        default="dataset/fitzpatrick17k_c_upsampled.csv",
                        help="Destination for the merged CSV")
    parser.add_argument("--prefix", type=str, default="synth",
                        help="Prefix for synthetic md5hash IDs (default: 'synth')")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode synthetics even if the .jpg already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real = pd.read_csv(args.real_csv)
    required = {"md5hash", "fitzpatrick_scale", "three_partition_label", "partition"}
    missing = required - set(real.columns)
    if missing:
        sys.exit(f"ERROR: real CSV {args.real_csv} is missing columns: {missing}")

    synth = pd.read_csv(args.synth_csv)
    print(f"Loaded {len(real)} real rows and {len(synth)} synthetic rows")

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Re-encode each synthetic PNG as JPEG under image_dir with a stable ID.
    new_rows = []
    skipped = 0
    for i, row in tqdm(synth.iterrows(), total=len(synth), desc="copy synth"):
        synth_id = f"{args.prefix}_{i:06d}"
        dst = image_dir / f"{synth_id}.jpg"

        if dst.exists() and not args.overwrite:
            skipped += 1
        else:
            src = Path(row["filepath"])
            if not src.exists():
                print(f"  WARN: missing source image {src}, skipping", file=sys.stderr)
                continue
            with Image.open(src) as img:
                img.convert("RGB").save(dst, format="JPEG", quality=95)

        new_rows.append({
            "md5hash": synth_id,
            "fitzpatrick_scale": int(row["fitzpatrick_scale"]),
            "three_partition_label": row["three_partition_label"],
            "partition": "train",
        })

    if skipped:
        print(f"Skipped re-encoding for {skipped} existing files (use --overwrite to force)")

    synth_df = pd.DataFrame(new_rows)

    # Preserve any extra columns in the real CSV (e.g. fst_consensus) by
    # filling them with NaN for synthetic rows.
    for col in real.columns:
        if col not in synth_df.columns:
            synth_df[col] = pd.NA
    synth_df = synth_df[real.columns]

    combined = pd.concat([real, synth_df], ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    print()
    print(f"Wrote {args.out_csv} ({len(combined)} rows = "
          f"{len(real)} real + {len(synth_df)} synthetic)")
    print()
    print("Per-partition counts:")
    print(combined["partition"].value_counts())
    print()
    print("Train-partition FST x lesion (real + synthetic combined):")
    train = combined[combined["partition"] == "train"]
    print(train.groupby(["fitzpatrick_scale", "three_partition_label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
