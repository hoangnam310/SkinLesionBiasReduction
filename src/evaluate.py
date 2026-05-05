"""
Standalone evaluation for the EfficientNetV2-B0 baseline classifier.

Loads a trained checkpoint, runs inference, computes bias-aware metrics
(global Top-1 / Macro AUROC / Macro AUPRC, plus per-Fitzpatrick subgroup
breakdowns), and writes them to ``logs/evaluation_metrics.json``.

Usage:
    python src/evaluate.py --checkpoint outputs/baseline_efficientnet/<run>/checkpoint.pt
    python src/evaluate.py --checkpoint <path> --split val   # use stored val_indices instead
    python src/evaluate.py --checkpoint <path> --split all   # ignore stored indices
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_metrics import (
    collect_predictions,
    compute_bias_metrics,
    compute_classification_metrics,
)
from data_process import NUM_LESION_TYPES, SkinLesionDataset
from train_baseline_efficientnet import (
    EfficientNetV2Baseline,
    build_transforms,
    get_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline checkpoint with bias-aware metrics."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint.pt produced by train_baseline_efficientnet.py")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Override CSV path; defaults to the value stored in the checkpoint")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Override image dir; defaults to the value stored in the checkpoint")
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image size; defaults to the value stored in the checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test",
                        help="Filters by the CSV's 'partition' column when available "
                             "(falls back to *_indices stored in the checkpoint for older runs). "
                             "'all' ignores the partition and evaluates every sample.")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="Directory to write evaluation_metrics.json into")
    return parser.parse_args()


def _resolve(value, fallback):
    return value if value is not None else fallback


def build_loader(
    csv_path: str,
    image_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    partition: Optional[str] = None,
    indices: Optional[List[int]] = None,
) -> DataLoader:
    _, val_transform = build_transforms(image_size)
    dataset = SkinLesionDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        transform=val_transform,
        partition=partition,
    )
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def format_summary(bias: dict) -> str:
    """Render a console-friendly summary of the bias metrics dict."""
    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    lines = ["=" * 64, "Evaluation Summary", "=" * 64]
    g = bias["global"]
    lines.append(f"Samples: {g['n_samples']}")
    lines.append(f"Top-1 Accuracy: {fmt(g['top1_accuracy'])}")
    lines.append(f"Macro AUROC:    {fmt(g['macro_auroc'])}")
    lines.append(f"Macro AUPRC:    {fmt(g['macro_auprc'])}")
    lines.append("")
    lines.append("Per-Fitzpatrick breakdown")
    lines.append("-" * 64)
    for fst_key, sub in bias["per_fitzpatrick"].items():
        n = sub["n_samples"]
        if n == 0:
            lines.append(f"{fst_key}: no samples")
            continue
        lines.append(
            f"{fst_key} (n={n}): acc={fmt(sub['accuracy'])} "
            f"macroAUROC={fmt(sub['macro_auroc'])} macroAUPRC={fmt(sub['macro_auprc'])}"
        )
        for class_name, cm in sub["per_class"].items():
            lines.append(
                f"    {class_name:>15}: n={cm['n_samples']:5d} "
                f"acc={fmt(cm['accuracy'])} "
                f"AUROC={fmt(cm['auroc'])} AUPRC={fmt(cm['auprc'])}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    csv_path = _resolve(args.csv_path, ckpt_args.get("csv_path", "dataset/fitzpatrick17k_cleaned.csv"))
    image_dir = _resolve(args.image_dir, ckpt_args.get("image_dir", "dataset/images"))
    image_size = int(_resolve(args.image_size, ckpt_args.get("image_size", 64)))
    dropout = float(ckpt_args.get("dropout", 0.3))

    model = EfficientNetV2Baseline(
        num_classes=NUM_LESION_TYPES,
        pretrained=False,  # weights come from the checkpoint
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    partition: Optional[str] = None
    indices: Optional[List[int]] = None
    split_label = args.split

    csv_has_partition = "partition" in pd.read_csv(csv_path, nrows=0).columns

    if args.split in ("train", "val", "test"):
        if csv_has_partition:
            partition = args.split
        else:
            legacy_key = f"{args.split}_indices"
            indices = ckpt.get(legacy_key)
            if indices is None and args.split == "test":
                indices = ckpt.get("val_indices")
                if indices is not None:
                    print("Checkpoint has no test_indices (older run); falling back to val_indices.")
                    split_label = "val"
            if indices is None:
                print(f"CSV has no 'partition' column and no {legacy_key} in checkpoint; "
                      "evaluating on full dataset.")
                split_label = "all"

    loader = build_loader(
        csv_path=csv_path,
        image_dir=image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        partition=partition,
        indices=indices,
    )
    print(f"Evaluating on {len(loader.dataset)} samples ({len(loader)} batches)")

    y_true, y_pred, y_prob, skin_tones = collect_predictions(model, loader, device)
    bias = compute_bias_metrics(y_true, y_pred, y_prob, skin_tones)
    classification = compute_classification_metrics(y_true, y_pred)

    output = {
        "checkpoint": str(args.checkpoint),
        "split": split_label,
        "image_size": image_size,
        "csv_path": csv_path,
        "image_dir": image_dir,
        "timestamp": datetime.now().isoformat(),
        "bias": bias,
        "classification": classification,
    }

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "evaluation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(format_summary(bias))
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
