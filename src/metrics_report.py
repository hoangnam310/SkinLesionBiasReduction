"""
Render a markdown report from an ``evaluation_metrics.json`` produced by
``src/evaluate.py``.

Usage:
    python src/metrics_report.py                                  # uses logs/evaluation_metrics.json
    python src/metrics_report.py --json logs/evaluation_metrics.json
    python src/metrics_report.py --json <path> --output report.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


CLASS_ORDER = ["benign", "malignant", "non-neoplastic"]
FST_KEYS = [f"fitzpatrick_{i}" for i in range(1, 7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evaluation_metrics.json into a markdown report."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="logs/evaluation_metrics.json",
        help="Path to evaluation_metrics.json (default: logs/evaluation_metrics.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path. Defaults to logs/<run_name>_report.md.",
    )
    return parser.parse_args()


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def run_name_from_checkpoint(checkpoint: str) -> str:
    """Pull the timestamped run dir out of a checkpoint path."""
    parts = Path(checkpoint).parts
    # …/baseline_efficientnet/<run>/checkpoint.pt → <run>
    if len(parts) >= 2 and parts[-1].endswith(".pt"):
        return parts[-2]
    return Path(checkpoint).stem or "evaluation"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def render_global(global_metrics: Dict) -> str:
    rows = [
        ["Samples", str(global_metrics.get("n_samples", "N/A"))],
        ["Top-1 Accuracy", fmt(global_metrics.get("top1_accuracy"))],
        ["Macro AUROC", fmt(global_metrics.get("macro_auroc"))],
        ["Macro AUPRC", fmt(global_metrics.get("macro_auprc"))],
    ]
    return md_table(["Metric", "Value"], rows)


def render_fst_summary(per_fst: Dict) -> str:
    headers = ["Fitzpatrick", "n", "Accuracy", "Macro AUROC", "Macro AUPRC"]
    rows = []
    for i, key in enumerate(FST_KEYS, start=1):
        sub = per_fst.get(key, {})
        n = sub.get("n_samples", 0)
        if n == 0:
            rows.append([str(i), "0", "—", "—", "—"])
            continue
        rows.append([
            str(i),
            str(n),
            fmt(sub.get("accuracy")),
            fmt(sub.get("macro_auroc")),
            fmt(sub.get("macro_auprc")),
        ])
    return md_table(headers, rows)


def render_per_class_metric(per_fst: Dict, metric_key: str) -> str:
    """Table with rows = FST, columns = lesion classes, cells = AUROC or AUPRC."""
    headers = ["Fitzpatrick"] + [f"{c} (n / {metric_key.upper()})" for c in CLASS_ORDER]
    rows = []
    for i, key in enumerate(FST_KEYS, start=1):
        sub = per_fst.get(key, {})
        if sub.get("n_samples", 0) == 0:
            rows.append([str(i)] + ["—"] * len(CLASS_ORDER))
            continue
        per_class = sub.get("per_class", {})
        row = [str(i)]
        for c in CLASS_ORDER:
            cm = per_class.get(c, {})
            n = cm.get("n_samples", 0)
            v = cm.get(metric_key)
            row.append(f"{n} / {fmt(v)}")
        rows.append(row)
    return md_table(headers, rows)


def render_classification(classification: Dict) -> str:
    overall = md_table(
        ["Metric", "Value"],
        [
            ["Accuracy", fmt(classification.get("accuracy"))],
            ["Balanced accuracy", fmt(classification.get("balanced_accuracy"))],
            ["Macro F1", fmt(classification.get("macro_f1"))],
            ["Weighted F1", fmt(classification.get("weighted_f1"))],
        ],
    )

    per_class = classification.get("per_class", {})
    per_class_rows = []
    for c in CLASS_ORDER:
        d = per_class.get(c, {})
        per_class_rows.append([
            c,
            fmt(d.get("precision")),
            fmt(d.get("recall")),
            fmt(d.get("f1-score")),
            str(d.get("support", 0)),
        ])
    per_class_table = md_table(
        ["Class", "Precision", "Recall", "F1", "Support"],
        per_class_rows,
    )

    cm = classification.get("confusion_matrix", [])
    if cm:
        headers = ["True \\ Pred"] + CLASS_ORDER
        rows = []
        for i, row in enumerate(cm):
            rows.append([CLASS_ORDER[i]] + [str(v) for v in row])
        cm_table = md_table(headers, rows)
    else:
        cm_table = "_No confusion matrix in JSON._"

    return f"{overall}\n\n### Per-class\n\n{per_class_table}\n\n### Confusion matrix\n\n{cm_table}"


def build_report(data: Dict) -> str:
    bias = data.get("bias", {})
    classification = data.get("classification", {})
    global_metrics = bias.get("global", {})
    per_fst = bias.get("per_fitzpatrick", {})

    parts = []
    parts.append("# Evaluation Report")
    parts.append("")

    meta_rows = [
        ["Checkpoint", f"`{data.get('checkpoint', 'N/A')}`"],
        ["Split", str(data.get("split", "N/A"))],
        ["Image size", str(data.get("image_size", "N/A"))],
        ["CSV path", f"`{data.get('csv_path', 'N/A')}`"],
        ["Image dir", f"`{data.get('image_dir', 'N/A')}`"],
        ["Generated at", str(data.get("timestamp", "N/A"))],
    ]
    parts.append(md_table(["Field", "Value"], meta_rows))
    parts.append("")

    parts.append("## Global metrics")
    parts.append("")
    parts.append(render_global(global_metrics))
    parts.append("")

    parts.append("## Per-Fitzpatrick subgroup")
    parts.append("")
    parts.append(render_fst_summary(per_fst))
    parts.append("")

    parts.append("### Per-class AUROC by Fitzpatrick (one-vs-rest)")
    parts.append("")
    parts.append(render_per_class_metric(per_fst, "auroc"))
    parts.append("")

    parts.append("### Per-class AUPRC by Fitzpatrick (one-vs-rest)")
    parts.append("")
    parts.append(render_per_class_metric(per_fst, "auprc"))
    parts.append("")

    if classification:
        parts.append("## Classification metrics (sklearn)")
        parts.append("")
        parts.append(render_classification(classification))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.output is None:
        run = run_name_from_checkpoint(data.get("checkpoint", ""))
        out_path = json_path.parent / f"{run}_report.md"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
