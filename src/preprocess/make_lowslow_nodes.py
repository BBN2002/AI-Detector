# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_nodes", required=True)
    ap.add_argument("--out_nodes", required=True)
    ap.add_argument("--span_days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--min_days", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_nodes)
    if "ts_ms" not in df.columns or "month_id" not in df.columns:
        raise ValueError("nodes.parquet must contain ts_ms and month_id.")
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    if label_col not in df.columns:
        raise ValueError("nodes.parquet must contain y_spam or y column.")

    rng = np.random.default_rng(args.seed)
    span_ms = int(args.span_days * 24 * 3600 * 1000)
    min_ms = int(args.min_days * 24 * 3600 * 1000)

    spam_mask = df[label_col] == 1
    if "product_id" not in df.columns:
        raise ValueError("nodes.parquet must contain product_id for LowSlow.")

    df_out = df.copy()
    spam_df = df_out[spam_mask]

    def resample_group(group: pd.DataFrame) -> pd.DataFrame:
        base_ts = group["ts_ms"].min()
        if pd.isna(base_ts):
            return group
        offsets = rng.integers(min_ms, span_ms + 1, size=len(group))
        group.loc[:, "ts_ms"] = (base_ts + offsets).astype("int64")
        return group

    spam_df = spam_df.groupby("product_id", group_keys=False).apply(resample_group)
    df_out.loc[spam_mask, "ts_ms"] = spam_df["ts_ms"].values

    dt = pd.to_datetime(df_out["ts_ms"].astype("int64"), unit="ms", utc=True, errors="coerce")
    df_out["month_id"] = dt.dt.strftime("%Y-%m")

    os.makedirs(os.path.dirname(args.out_nodes), exist_ok=True)
    df_out.to_parquet(args.out_nodes, index=False)
    print(f"[OK] wrote LowSlow nodes: {args.out_nodes}")


if __name__ == "__main__":
    main()
