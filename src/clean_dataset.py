"""
Filter Fitzpatrick17k down to the Fitzpatrick17k-C "keep" set
(Abhishek et al., Zenodo 12739457) and apply the same filter to any
parallel image directory (e.g. SAM2-segmented images produced on Colab).

Typical local run (originals + SAM2-segmented set in one go):

    python src/clean_dataset.py \
        --source_csv dataset/fitzpatrick17k.csv \
        --output_csv dataset/fitzpatrick17k_c.csv \
        --image_dirs dataset/images dataset/images_sam2_color \
        --mode move

On Colab, point `--image_dirs` at the segmented directory there, e.g.

    python src/clean_dataset.py \
        --image_dirs /content/drive/MyDrive/.../images_sam2_color \
        --mode move

The keep CSV (`Fitzpatrick17k-C.csv`) is auto-downloaded from Zenodo
into `dataset/` if it isn't already there.
"""

import argparse
import os
import shutil
import sys
import urllib.request
from typing import Iterable, Set, Tuple

import pandas as pd

ZENODO_URL = "https://zenodo.org/records/12739457/files/Fitzpatrick17k-C.csv?download=1"
DEFAULT_KEEP_CSV = "../dataset/Fitzpatrick17k-C.csv"
DEFAULT_SOURCE_CSV = "../dataset/fitzpatrick17k_cleaned.csv"
DEFAULT_OUTPUT_CSV = "../dataset/fitzpatrick17k_c.csv"
REMOVED_SUBDIR = "_removed"


def ensure_keep_csv(keep_csv: str) -> str:
    if os.path.exists(keep_csv):
        return keep_csv
    os.makedirs(os.path.dirname(keep_csv) or ".", exist_ok=True)
    print(f"Downloading Fitzpatrick17k-C.csv → {keep_csv}")
    urllib.request.urlretrieve(ZENODO_URL, keep_csv)
    return keep_csv


def build_filtered_csv(keep_csv: str, source_csv: str, output_csv: str) -> Set[str]:
    keep_df = pd.read_csv(keep_csv)
    if "md5hash" not in keep_df.columns or "partition" not in keep_df.columns:
        raise ValueError(
            f"{keep_csv} is missing required columns 'md5hash'/'partition'."
        )
    keep_set: Set[str] = set(keep_df["md5hash"].astype(str))

    src_df = pd.read_csv(source_csv)
    if "md5hash" not in src_df.columns:
        raise ValueError(f"{source_csv} is missing 'md5hash' column.")

    # Merge in the canonical train/val/test partition + the consensus FST
    # (`fst`) so downstream code can pick whichever it wants.
    keep_cols = keep_df[["md5hash", "partition", "fst"]].rename(
        columns={"fst": "fst_consensus"}
    )
    merged = src_df.merge(keep_cols, on="md5hash", how="inner")

    n_src = len(src_df)
    n_keep = len(keep_set)
    n_out = len(merged)
    n_missing_in_src = n_keep - n_out
    print(
        f"Source CSV rows:     {n_src}\n"
        f"Keep-list rows:      {n_keep}\n"
        f"Filtered output:     {n_out}\n"
        f"In keep but missing\n  from source CSV:   {n_missing_in_src}"
    )
    if n_missing_in_src > 0:
        missing = sorted(keep_set - set(src_df["md5hash"].astype(str)))[:5]
        print(f"  examples: {missing}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    return keep_set


def filter_image_dir(
    image_dir: str,
    keep_set: Set[str],
    mode: str,
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png"),
) -> Tuple[int, int, int]:
    """
    Returns (kept, removed, missing_from_dir).
    `missing_from_dir` is the count of keep-set md5s with no matching file here
    (useful to know which images still need to be segmented on Colab).
    """
    if not os.path.isdir(image_dir):
        print(f"[skip] {image_dir} does not exist")
        return 0, 0, len(keep_set)

    ext_set = {e.lower() for e in extensions}
    removed_dir = os.path.join(image_dir, REMOVED_SUBDIR)

    present_md5: Set[str] = set()
    to_remove = []
    for name in os.listdir(image_dir):
        full = os.path.join(image_dir, name)
        if not os.path.isfile(full):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in ext_set:
            continue
        present_md5.add(stem)
        if stem not in keep_set:
            to_remove.append(full)

    kept = len(present_md5 & keep_set)
    missing = len(keep_set - present_md5)

    print(
        f"\n[{image_dir}]\n"
        f"  files scanned:        {len(present_md5)}\n"
        f"  kept (in Fitz17k-C):  {kept}\n"
        f"  to remove:            {len(to_remove)}\n"
        f"  keep-set md5s missing\n"
        f"    from this dir:      {missing}"
    )

    if mode == "dryrun" or not to_remove:
        return kept, len(to_remove), missing

    if mode == "move":
        os.makedirs(removed_dir, exist_ok=True)
        for path in to_remove:
            shutil.move(path, os.path.join(removed_dir, os.path.basename(path)))
        print(f"  moved {len(to_remove)} files → {removed_dir}")
    elif mode == "delete":
        for path in to_remove:
            os.remove(path)
        print(f"  deleted {len(to_remove)} files")
    else:
        raise ValueError(f"unknown mode: {mode}")

    return kept, len(to_remove), missing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep_csv", default=DEFAULT_KEEP_CSV,
                   help="Path to Fitzpatrick17k-C.csv (auto-downloaded if absent).")
    p.add_argument("--source_csv", default=DEFAULT_SOURCE_CSV,
                   help="Original Fitzpatrick17k metadata CSV. Set to '' to skip CSV step.")
    p.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV,
                   help="Where to write the filtered CSV (with partition column).")
    p.add_argument("--image_dirs", nargs="*", default=[],
                   help="One or more image directories (filenames must be <md5>.<ext>).")
    p.add_argument("--mode", choices=["move", "delete", "dryrun"], default="move",
                   help="What to do with non-keep images. 'move' relocates them to _removed/ inside each dir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    keep_csv = ensure_keep_csv(args.keep_csv)

    if args.source_csv:
        if not os.path.exists(args.source_csv):
            print(f"[error] source_csv not found: {args.source_csv}", file=sys.stderr)
            return 1
        keep_set = build_filtered_csv(keep_csv, args.source_csv, args.output_csv)
    else:
        keep_set = set(pd.read_csv(keep_csv)["md5hash"].astype(str))
        print(f"Loaded {len(keep_set)} md5 hashes from {keep_csv}")

    for image_dir in args.image_dirs:
        filter_image_dir(image_dir, keep_set, args.mode)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
