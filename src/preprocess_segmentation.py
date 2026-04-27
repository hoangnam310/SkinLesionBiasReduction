"""
Offline batch preprocessor: applies the segmentation-aware pipeline to every
image referenced by `fitzpatrick17k_cleaned.csv` and writes the results to
`--output_dir`. Run this once before training; afterwards, point the trainer's
`--image_dir` at `--output_dir` to skip preprocessing entirely.

Backends
    --backend cv2          # no SAM, runs anywhere (default)
    --backend sam2         # needs facebookresearch/sam2 + a checkpoint
    --backend sam3         # needs facebookresearch/sam3 (text-prompted, "skin lesion")
    --backend medsam3      # needs Joey-S-Liu/MedSAM3 (LoRA config + weights)

Examples
    # cv2-only, fast, runs on a laptop
    python src/preprocess_segmentation.py --backend cv2 --strategy color \\
        --output_dir dataset/images_cv2_color

    # SAM2 (download checkpoint from sam2 repo first)
    python src/preprocess_segmentation.py --backend sam2 \\
        --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt \\
        --output_dir dataset/images_sam2_color

    # MedSAM3 (clone repo so infer_sam.py is on PYTHONPATH)
    PYTHONPATH=external/MedSAM3 python src/preprocess_segmentation.py --backend medsam3 \\
        --medsam3_config external/MedSAM3/configs/full_lora_config.yaml \\
        --medsam3_weights external/MedSAM3/outputs/sam3_lora_full/best_lora_weights.pt \\
        --output_dir dataset/images_medsam3_color
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation import (  # noqa: E402
    load_medsam3_mask_generator,
    load_sam2_mask_generator,
    load_sam3_mask_generator,
    process_image,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch segmentation-aware preprocessing")
    p.add_argument("--csv_path", default="dataset/fitzpatrick17k_cleaned.csv")
    p.add_argument("--image_dir", default="dataset/images")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--backend", choices=["cv2", "sam2", "sam3", "medsam3"], default="cv2")
    p.add_argument("--strategy", choices=["edge", "color", "center"], default="color")
    p.add_argument("--seg_size", type=int, default=256,
                   help="Resolution at which SAM + ROI selection runs (downscaled to out_size after).")
    p.add_argument("--out_size", type=int, default=64,
                   help="Final image side length (matches the trainer's --image_size).")
    p.add_argument("--crop_frac", type=float, default=0.6)
    p.add_argument("--min_skin", type=float, default=0.85)

    # SAM2
    p.add_argument("--sam2_checkpoint", type=str)
    p.add_argument("--sam2_config", type=str, default="sam2_hiera_t.yaml")

    # SAM3 / MedSAM3
    p.add_argument("--prompt", type=str, default="skin lesion",
                   help="Text prompt for SAM3 / MedSAM3.")
    p.add_argument("--medsam3_config", type=str)
    p.add_argument("--medsam3_weights", type=str)
    p.add_argument("--medsam3_resolution", type=int, default=1008)
    p.add_argument("--detection_threshold", type=float, default=0.3)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, only process the first N rows (for testing).")
    p.add_argument("--no_resume", action="store_true",
                   help="Reprocess images that already exist in --output_dir.")
    p.add_argument("--jpeg_quality", type=int, default=95)
    return p.parse_args()


def build_mask_fn(args: argparse.Namespace):
    if args.backend == "cv2":
        return None
    if args.backend == "sam2":
        if not args.sam2_checkpoint:
            raise SystemExit("--sam2_checkpoint is required for backend=sam2")
        return load_sam2_mask_generator(
            checkpoint=args.sam2_checkpoint,
            config=args.sam2_config,
            device=args.device,
        )
    if args.backend == "sam3":
        return load_sam3_mask_generator(
            prompt=args.prompt,
            device=args.device,
            score_thresh=args.detection_threshold,
        )
    if args.backend == "medsam3":
        if not (args.medsam3_config and args.medsam3_weights):
            raise SystemExit("--medsam3_config and --medsam3_weights are required for backend=medsam3")
        return load_medsam3_mask_generator(
            config_path=args.medsam3_config,
            weights_path=args.medsam3_weights,
            prompt=args.prompt,
            resolution=args.medsam3_resolution,
            detection_threshold=args.detection_threshold,
            device=args.device,
        )
    raise SystemExit(f"Unknown backend: {args.backend}")


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    df = df[df["fitzpatrick_scale"] != -1].reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit)

    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend:      {args.backend}")
    print(f"Strategy:     {args.strategy}")
    print(f"seg_size:     {args.seg_size}")
    print(f"out_size:     {args.out_size}")
    print(f"Input dir:    {image_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Total rows:   {len(df)}")

    mask_fn = build_mask_fn(args)
    if mask_fn is None:
        print("Mask: cv2-only (all-ones fallback)")
    else:
        print(f"Mask: {args.backend} loaded")

    ok = skip = missing = fail = 0
    bar = tqdm(df.itertuples(index=False), total=len(df), desc="preprocess")
    for row in bar:
        md5 = row.md5hash
        src = image_dir / f"{md5}.jpg"
        dst = out_dir / f"{md5}.jpg"

        if not src.exists():
            missing += 1
            continue
        if dst.exists() and not args.no_resume:
            skip += 1
            continue

        try:
            img = np.array(Image.open(src).convert("RGB"))
            out = process_image(
                img,
                mask_fn=mask_fn,
                strategy=args.strategy,
                seg_size=args.seg_size,
                out_size=args.out_size,
                crop_frac=args.crop_frac,
                min_skin=args.min_skin,
            )
            Image.fromarray(out).save(dst, quality=args.jpeg_quality)
            ok += 1
        except Exception as exc:  # keep going; report at end
            fail += 1
            tqdm.write(f"[fail] {md5}: {exc}")

        bar.set_postfix(ok=ok, skip=skip, miss=missing, fail=fail)

    print(f"\nDone. processed={ok}, skipped={skip}, missing_src={missing}, failed={fail}")
    print(f"Output: {out_dir}")
    print("Train against the new dir, e.g.:")
    print(f"  python src/train_baseline_efficientnet.py --image_dir {out_dir} "
          f"--image_size {args.out_size}")


if __name__ == "__main__":
    main()
