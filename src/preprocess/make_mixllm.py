# -*- coding: utf-8 -*-
import argparse
import os
from typing import List

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".jsonl", ".json"]:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file extension: {ext}")


def pick_label_col(df: pd.DataFrame) -> str:
    for c in ["y_spam", "label", "y", "is_spam", "spam"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find label column in input.")


def choose_dedup_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "user_id",
        "user",
        "reviewerID",
        "product_id",
        "product",
        "asin",
        "reviewTime",
        "timestamp",
        "reviewText",
        "text",
        "reviewSummary",
        "star",
        "rating",
    ]
    cols = [c for c in candidates if c in df.columns]
    return cols if cols else df.columns.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--genuine_from", type=int, default=0)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--dedup_cols", nargs="*", default=None)
    args = ap.parse_args()

    if args.genuine_from < 0 or args.genuine_from >= len(args.inputs):
        raise ValueError("genuine_from index out of range.")

    dfs = [load_table(p) for p in args.inputs]
    base_df = dfs[args.genuine_from]
    label_col = args.label_col or pick_label_col(base_df)

    if label_col not in base_df.columns:
        raise ValueError(f"label_col {label_col} not in base dataset.")

    genuine = base_df[base_df[label_col] == 0].copy()

    spam_list = []
    for df in dfs:
        lc = args.label_col or pick_label_col(df)
        spam = df[df[lc] == 1].copy()
        if lc != label_col:
            spam = spam.rename(columns={lc: label_col})
        spam_list.append(spam)

    spam_all = pd.concat(spam_list, ignore_index=True)
    dedup_cols = args.dedup_cols or choose_dedup_cols(spam_all)
    spam_all = spam_all.drop_duplicates(subset=dedup_cols)

    out_df = pd.concat([genuine, spam_all], ignore_index=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] wrote mix dataset: {args.out}")
    print(f"[INFO] genuine={len(genuine):,} spam={len(spam_all):,} total={len(out_df):,}")


if __name__ == "__main__":
    main()
