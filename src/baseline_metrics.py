"""
Evaluation metrics for the EfficientNet baseline (3-class lesion classification).
Used by training script and notebooks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

LESION_CLASS_NAMES = ["benign", "malignant", "non-neoplastic"]
NUM_FITZPATRICK = 6


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run model on loader and return (y_true, y_pred, y_prob, skin_tones).

    skin_tones uses the dataset's encoded values (0..5 for Fitzpatrick 1..6).
    y_prob is the softmax probability matrix with shape (N, num_classes).
    """
    model.eval()
    num_classes = len(LESION_CLASS_NAMES)
    ys: List[int] = []
    preds: List[int] = []
    probs_chunks: List[np.ndarray] = []
    tones: List[int] = []
    for step, batch in enumerate(loader):
        if max_batches > 0 and step >= max_batches:
            break
        images, skin_tones, lesion_targets = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        p = np.argmax(prob, axis=1).tolist()
        t = lesion_targets.numpy().tolist()
        st = skin_tones.numpy().tolist()
        ys.extend(t)
        preds.extend(p)
        probs_chunks.append(prob)
        tones.extend(st)
    if probs_chunks:
        y_prob = np.concatenate(probs_chunks, axis=0)
    else:
        y_prob = np.zeros((0, num_classes), dtype=np.float32)
    return (
        np.array(ys, dtype=np.int64),
        np.array(preds, dtype=np.int64),
        y_prob,
        np.array(tones, dtype=np.int64),
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute accuracy, balanced accuracy, macro F1, per-class precision/recall/F1,
    and confusion matrix. Uses sklearn when available.
    """
    if class_names is None:
        class_names = LESION_CLASS_NAMES

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1-score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "classification_report_str": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        ),
    }


def compute_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    skin_tones: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute global + per-Fitzpatrick subgroup metrics.

    Inputs use encoded label space:
        y_true / y_pred ∈ {0..C-1} for C lesion classes
        skin_tones ∈ {0..5} for Fitzpatrick 1..6
        y_prob shape (N, C)

    Subgroups are reported with their original Fitzpatrick scale (1..6).
    Per-class AUROC/AUPRC are one-vs-rest. Returns ``None`` for any metric
    that cannot be computed (e.g. a class missing from a subgroup).
    """
    if class_names is None:
        class_names = LESION_CLASS_NAMES
    num_classes = len(class_names)

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        roc_auc_score,
    )

    n_total = int(len(y_true))

    def _safe_macro_auroc(yt: np.ndarray, yp: np.ndarray) -> Optional[float]:
        if len(yt) == 0 or len(set(yt.tolist())) < 2:
            return None
        try:
            return float(roc_auc_score(yt, yp, multi_class="ovr", average="macro"))
        except ValueError:
            return None

    def _safe_macro_auprc(yt: np.ndarray, yp: np.ndarray) -> Optional[float]:
        if len(yt) == 0:
            return None
        try:
            onehot = np.zeros((len(yt), num_classes), dtype=np.int64)
            onehot[np.arange(len(yt)), yt] = 1
            return float(average_precision_score(onehot, yp, average="macro"))
        except ValueError:
            return None

    def _safe_binary_auroc(binary_true: np.ndarray, score: np.ndarray) -> Optional[float]:
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            return None
        try:
            return float(roc_auc_score(binary_true, score))
        except ValueError:
            return None

    def _safe_binary_auprc(binary_true: np.ndarray, score: np.ndarray) -> Optional[float]:
        if binary_true.sum() == 0:
            return None
        try:
            return float(average_precision_score(binary_true, score))
        except ValueError:
            return None

    # ---- Global ----
    global_metrics: Dict = {
        "n_samples": n_total,
        "top1_accuracy": float(accuracy_score(y_true, y_pred)) if n_total else None,
        "macro_auroc": _safe_macro_auroc(y_true, y_prob),
        "macro_auprc": _safe_macro_auprc(y_true, y_prob),
    }

    # ---- Per Fitzpatrick subgroup ----
    per_fitzpatrick: Dict[str, Dict] = {}
    for tone_enc in range(NUM_FITZPATRICK):
        fst_label = f"fitzpatrick_{tone_enc + 1}"
        mask = skin_tones == tone_enc
        n = int(mask.sum())
        if n == 0:
            per_fitzpatrick[fst_label] = {"n_samples": 0}
            continue

        sub_y_true = y_true[mask]
        sub_y_pred = y_pred[mask]
        sub_y_prob = y_prob[mask]

        per_class: Dict[str, Dict] = {}
        for c, name in enumerate(class_names):
            binary_true = (sub_y_true == c).astype(np.int64)
            n_class = int(binary_true.sum())
            n_pred = int((sub_y_pred == c).sum())
            n_correct = int(((sub_y_true == c) & (sub_y_pred == c)).sum())
            recall = float(n_correct / n_class) if n_class > 0 else None
            precision = float(n_correct / n_pred) if n_pred > 0 else None
            per_class[name] = {
                "n_samples": n_class,
                "accuracy": recall,  # per-class accuracy = recall = TPR
                "recall": recall,
                "precision": precision,
                "auroc": _safe_binary_auroc(binary_true, sub_y_prob[:, c]),
                "auprc": _safe_binary_auprc(binary_true, sub_y_prob[:, c]),
            }

        per_fitzpatrick[fst_label] = {
            "n_samples": n,
            "accuracy": float(accuracy_score(sub_y_true, sub_y_pred)),
            "macro_auroc": _safe_macro_auroc(sub_y_true, sub_y_prob),
            "macro_auprc": _safe_macro_auprc(sub_y_true, sub_y_prob),
            "per_class": per_class,
        }

    return {
        "global": global_metrics,
        "per_fitzpatrick": per_fitzpatrick,
    }
