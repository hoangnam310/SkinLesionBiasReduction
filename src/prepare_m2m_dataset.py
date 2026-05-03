"""Prepare M2M (skin-diff) train/test splits and LoRA JSON from local Fitzpatrick17k images.

The vendored repo at third_party/skin-diff ships a curated 2,707-row subset
(`data_splits/Fitz_subset.csv`) that already has the `label` column M2M's
Textual Inversion script keys off of. Our own `dataset/fitzpatrick17k_cleaned.csv`
only carries `three_partition_label`, so we use the M2M-bundled subset directly
and just filter it down to images that actually exist on this machine.

Reproduces the `light_and_dark_flex_to_dark` split from
`third_party/skin-diff/notebooks/finetune.ipynb`:
  - train: all FST 1-2 rows + N rows per (FST in {5,6}, label) cell
  - test:  remaining FST 5-6 rows
"""

import argparse
import json
import os
import sys

import pandas as pd

# 7 conditions M2M trained on, in the exact spelling used by the bundled CSV's `label` column.
M2M_CONDITIONS = [
    "basal cell carcinoma",
    "folliculitis",
    "nematode infection",
    "neutrophilic dermatoses",
    "prurigo nodularis",
    "psoriasis",
    "squamous cell carcinoma",
]

# Maps from `label` -> the placeholder token used in TI/LoRA prompts.
# Same convention as the M2M notebook (first 3 chars + "-class").
def token_for(label: str) -> str:
    return label.replace(" ", "_")[:3] + "-class"

SKIN_TYPE_PROMPT = {
    1: "a very light-skinned",
    2: "a light-skinned",
    5: "a dark-skinned",
    6: "a very dark-skinned",
}


def filter_to_local_images(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    have = []
    for h in df["md5hash"]:
        if os.path.exists(os.path.join(image_dir, f"{h}.jpg")):
            have.append(h)
    kept = df[df["md5hash"].isin(have)].copy()
    return kept


def build_split(df: pd.DataFrame, n_dark_per_cell: int, seed: int):
    df = df[df["fitzpatrick_scale"].isin([1, 2, 5, 6])].copy()
    df = df[df["label"].isin(M2M_CONDITIONS)].copy()

    dark = df[df["fitzpatrick_scale"].isin([5, 6])]
    light = df[df["fitzpatrick_scale"].isin([1, 2])]

    # Sample n_dark_per_cell rows per disease label from FST 5-6.
    # Use min() so smoke tests with small subsets don't crash if a class is short.
    flex_dark_parts = []
    for cls, sub in dark.groupby("label"):
        n = min(n_dark_per_cell, len(sub))
        if n > 0:
            flex_dark_parts.append(sub.sample(n=n, random_state=seed, replace=False))
    flex_dark = pd.concat(flex_dark_parts, ignore_index=True) if flex_dark_parts else dark.iloc[0:0]

    # Train = all FST 1-2 + the sampled dark flex rows.
    train = pd.concat([light, flex_dark], ignore_index=True)
    # Test = the FST 5-6 rows we did NOT sample into the train flex set.
    test = dark[~dark["md5hash"].isin(flex_dark["md5hash"])].copy()
    return train, test


def build_lora_json(train_df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in train_df.iterrows():
        label = row["label"]
        fst = int(row["fitzpatrick_scale"])
        rows.append({
            "image_path": f"{row['md5hash']}.jpg",
            "label": label,
            "skin_type": fst,
            "prompt": (
                f"An image of {token_for(label)} on the skin of "
                f"{SKIN_TYPE_PROMPT[fst]} individual"
            ),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m2m_csv", default="third_party/skin-diff/data_splits/Fitz_subset.csv")
    p.add_argument("--image_dir", default="dataset/images")
    p.add_argument("--output_dir", default="third_party/skin-diff/data_splits")
    p.add_argument("--split_name", default="light_and_dark_flex_to_dark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_dark_per_cell", type=int, default=8,
                   help="Per-label FST 5-6 samples drawn into train (M2M default: 8).")
    args = p.parse_args()

    df = pd.read_csv(args.m2m_csv)
    n_total = len(df)

    df = filter_to_local_images(df, args.image_dir)
    n_local = len(df)
    print(f"[coverage] {n_local}/{n_total} rows from {args.m2m_csv} have images in {args.image_dir}")

    if n_local == 0:
        print("ERROR: no images found locally. Check --image_dir.", file=sys.stderr)
        sys.exit(1)

    train, test = build_split(df, args.n_dark_per_cell, args.seed)

    print("[train] FST counts:")
    print(train["fitzpatrick_scale"].value_counts().sort_index().to_string())
    print(f"[train] total: {len(train)}")
    print("[test] FST counts:")
    print(test["fitzpatrick_scale"].value_counts().sort_index().to_string())
    print(f"[test] total: {len(test)}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_csv = os.path.join(args.output_dir, f"train_{args.split_name}_seed={args.seed}.csv")
    test_csv = os.path.join(args.output_dir, f"test_{args.split_name}_seed={args.seed}.csv")
    json_path = os.path.join(args.output_dir, f"train_lora_{args.split_name}_seed={args.seed}.json")

    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)

    lora_rows = build_lora_json(train)
    with open(json_path, "w") as f:
        json.dump(lora_rows, f, indent=2)

    print(f"[write] {train_csv}")
    print(f"[write] {test_csv}")
    print(f"[write] {json_path} ({len(lora_rows)} entries)")


if __name__ == "__main__":
    main()
