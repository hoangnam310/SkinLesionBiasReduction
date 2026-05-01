"""
Baseline classifier training with EfficientNetV2-B0.

This script trains a lesion-type classifier on Fitzpatrick17k using:
- Input image size fixed to 64x64 (as requested for current testing phase)
- EfficientNetV2-B0 backbone from timm
- 3-way lesion classification (benign, malignant, non-neoplastic)
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# Add src directory for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_process import LESION_TYPE_ENCODING, NUM_LESION_TYPES, SkinLesionDataset
from baseline_metrics import (
    collect_predictions,
    compute_bias_metrics,
    compute_classification_metrics,
)

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for EfficientNetV2-B0. Install with: pip install timm"
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Metrics:
    loss: float
    accuracy: float


class EfficientNetV2Baseline(nn.Module):
    """EfficientNetV2-B0 backbone + custom classification head."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-B0 baseline classifier"
    )
    parser.add_argument("--csv_path", type=str, default="dataset/fitzpatrick17k_c.csv",
                        help="CSV with a 'partition' column (train/val/test)")
    parser.add_argument("--image_dir", type=str, default="dataset/images")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="If > 0, cap train/val loop steps for quick smoke testing")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained weights (default: True)")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable pretrained weights (for offline/testing)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout for custom classification head")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze backbone and train classifier head first (default: True)")
    parser.add_argument("--no_freeze_backbone", action="store_true",
                        help="Train the entire model from the start")
    parser.add_argument("--unfreeze_epoch", type=int, default=-1,
                        help="Epoch index (0-based) to unfreeze backbone for fine-tuning; -1 disables")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5,
                        help="Learning rate after unfreezing backbone")
    parser.add_argument("--class_weights", action="store_true",
                        help="Use inverse-frequency class weights in CrossEntropyLoss")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/baseline_efficientnet",
        help="Directory for checkpoint.pt and metrics.json (timestamped subfolder)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write checkpoint or metrics files",
    )
    return parser.parse_args()


def get_device(device_str: str = None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, val_transform


def build_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, SkinLesionDataset]:
    train_transform, val_transform = build_transforms(args.image_size)

    train_dataset = SkinLesionDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=train_transform,
        partition="train",
    )
    val_dataset = SkinLesionDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=val_transform,
        partition="val",
    )
    test_dataset = SkinLesionDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        transform=val_transform,
        partition="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_dataset


def build_model(
    num_classes: int = NUM_LESION_TYPES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> EfficientNetV2Baseline:
    return EfficientNetV2Baseline(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


def set_backbone_trainable(model: EfficientNetV2Baseline, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def create_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def compute_inverse_class_weights(
    dataset: SkinLesionDataset,
    num_classes: int = NUM_LESION_TYPES,
) -> torch.Tensor:
    """Inverse-frequency class weights normalized to mean = 1."""
    counts = torch.zeros(num_classes, dtype=torch.float)
    for label_str in dataset.df["three_partition_label"]:
        counts[LESION_TYPE_ENCODING[label_str]] += 1
    return counts.sum() / (num_classes * counts.clamp(min=1))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer = None,
    max_steps: int = 0,
    desc: str = "",
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    total = min(max_steps, len(loader)) if max_steps > 0 else len(loader)
    pbar = tqdm(loader, total=total, desc=desc or ("train" if is_train else "val"), leave=False)
    for step, (images, _, lesion_targets) in enumerate(pbar):
        if max_steps > 0 and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        lesion_targets = lesion_targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, lesion_targets)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == lesion_targets).sum().item()
        running_total += lesion_targets.size(0)
        running_loss += loss.item() * lesion_targets.size(0)

        pbar.set_postfix(
            loss=f"{running_loss / max(1, running_total):.4f}",
            acc=f"{running_correct / max(1, running_total):.4f}",
        )
    pbar.close()

    avg_loss = running_loss / max(1, running_total)
    avg_acc = running_correct / max(1, running_total)
    return Metrics(loss=avg_loss, accuracy=avg_acc)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    train_loader, val_loader, test_loader, train_dataset = build_loaders(args)
    print(
        f"Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches), "
        f"Val: {len(val_loader.dataset)} ({len(val_loader)}), "
        f"Test: {len(test_loader.dataset)} ({len(test_loader)})"
    )

    pretrained = args.pretrained and not args.no_pretrained
    freeze_backbone = args.freeze_backbone and not args.no_freeze_backbone
    print(f"Pretrained: {pretrained}")
    print(f"Freeze backbone: {freeze_backbone}")
    model = build_model(
        num_classes=NUM_LESION_TYPES,
        pretrained=pretrained,
        dropout=args.dropout,
    ).to(device)

    if freeze_backbone:
        set_backbone_trainable(model, trainable=False)
        print("Training classifier head only (backbone frozen).")

    if args.class_weights:
        weights = compute_inverse_class_weights(train_dataset).to(device)
        print(f"Class weights (benign, malignant, non-neoplastic): "
              f"{[round(w, 4) for w in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(args.epochs):
        if freeze_backbone and args.unfreeze_epoch >= 0 and epoch == args.unfreeze_epoch:
            set_backbone_trainable(model, trainable=True)
            optimizer = create_optimizer(
                model,
                lr=args.fine_tune_lr,
                weight_decay=args.weight_decay,
            )
            print(
                f"Backbone unfrozen at epoch {epoch}; "
                f"continuing fine-tuning with lr={args.fine_tune_lr}"
            )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_steps=args.max_steps,
            desc=f"epoch {epoch + 1}/{args.epochs} train",
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            max_steps=args.max_steps,
            desc=f"epoch {epoch + 1}/{args.epochs} val",
        )

        is_best = val_metrics.loss < best_val_loss
        marker = "  ← best" if is_best else ""
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"train_loss={train_metrics.loss:.4f}, train_acc={train_metrics.accuracy:.4f}, "
            f"val_loss={val_metrics.loss:.4f}, val_acc={val_metrics.accuracy:.4f}"
            f"{marker}"
        )

        if is_best:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"Restored best weights from epoch {best_epoch + 1} "
            f"(val_loss={best_val_loss:.4f}) for final evaluation."
        )

    if not args.no_save:
        out = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        eval_batches = args.max_steps if args.max_steps > 0 else 0
        y_true, y_pred, y_prob, skin_tones = collect_predictions(
            model, test_loader, device, max_batches=eval_batches
        )
        classification_metrics = compute_classification_metrics(y_true, y_pred)
        bias_metrics = compute_bias_metrics(y_true, y_pred, y_prob, skin_tones)
        metrics = {
            "split": "test",
            "classification": classification_metrics,
            "bias": bias_metrics,
        }
        if eval_batches > 0:
            metrics["_note"] = (
                f"Metrics computed on first {eval_batches} test batch(es) only "
                "(because --max_steps > 0). For full-test metrics, train with --max_steps 0."
            )
        metrics_path = out / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        ckpt_path = out / "checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "metrics": metrics,
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
                "best_val_loss": best_val_loss if best_state is not None else None,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")
        print(f"Saved metrics: {metrics_path}")
        print("Baseline test run complete.")
    else:
        print("Baseline run complete (--no_save: no checkpoint written).")


if __name__ == "__main__":
    main()
