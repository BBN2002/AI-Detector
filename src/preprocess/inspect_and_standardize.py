# -*- coding: utf-8 -*-
import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil import parser as dtparser


CAND_TS_COLS = [
    "timestamp",
    "time",
    "ts",
    "datetime",
    "date",
    "created_at",
    "created",
    "reviewTime",
    "review_time",
    "unixReviewTime",
    "unix_review_time",
]

CAND_USER_COLS = ["user_id", "reviewerID", "reviewerId", "uid", "user", "account_id"]
CAND_PROD_COLS = ["product_id", "asin", "item_id", "pid", "product"]
CAND_RATING_COLS = ["rating", "overall", "stars", "score", "star"]
CAND_TEXT_COLS = ["text", "reviewText", "content", "review", "body"]
CAND_LABEL_COLS = ["label", "y", "y_spam", "is_spam", "spam", "fraud", "target"]


def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _is_probably_epoch(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna().astype("float64")
    if len(s) == 0:
        return False
    q50 = float(np.quantile(s, 0.5))
    return (1e8 <= q50 <= 2e10) or (1e11 <= q50 <= 2e13)


def _to_ts_ms(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and _is_probably_epoch(series):
        s = series.astype("float64")
        q50 = float(np.nanquantile(s, 0.5))
        if q50 > 1e11:
            ts_ms = s.round().astype("Int64")
        else:
            ts_ms = (s * 1000.0).round().astype("Int64")
        return ts_ms

    def parse_one(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer, float, np.floating)):
            return x
        try:
            dt = dtparser.parse(str(x), fuzzy=True)
            return int(dt.timestamp() * 1000)
        except Exception:
            return np.nan

    out = series.map(parse_one)
    out = pd.to_numeric(out, errors="coerce").astype("Int64")
    if _is_probably_epoch(out.astype("float64")):
        q50 = float(np.nanquantile(out.astype("float64"), 0.5))
        if q50 < 1e11:
            out = (out.astype("float64") * 1000.0).round().astype("Int64")
    return out


def _auto_pick_timestamp_col(df: pd.DataFrame) -> Tuple[Optional[str], List[str]]:
    cols = list(df.columns)
    ts_col = _pick_first_existing(cols, CAND_TS_COLS)
    candidates = []
    if ts_col is not None:
        candidates.append(ts_col)

    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["time", "date", "timestamp", "ts"]):
            if c not in candidates:
                candidates.append(c)

    best = None
    best_rate = -1.0
    for c in candidates:
        s = df[c]
        ts_ms = _to_ts_ms(s)
        rate = float(ts_ms.notna().mean())
        if rate > best_rate:
            best_rate = rate
            best = c

    if best is None or best_rate < 0.80:
        return None, candidates
    return best, candidates


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".jsonl", ".json"]:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file extension: {ext}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--user_col", default=None)
    ap.add_argument("--product_col", default=None)
    ap.add_argument("--rating_col", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--timestamp_col", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_table(args.in_path)
    print(f"[INFO] loaded: {args.in_path}")
    print(f"[INFO] rows={len(df):,} cols={len(df.columns)}")
    print("[INFO] columns:", list(df.columns))

    user_col = args.user_col or _pick_first_existing(df.columns.tolist(), CAND_USER_COLS)
    product_col = args.product_col or _pick_first_existing(
        df.columns.tolist(), CAND_PROD_COLS
    )
    rating_col = args.rating_col or _pick_first_existing(
        df.columns.tolist(), CAND_RATING_COLS
    )
    text_col = args.text_col or _pick_first_existing(df.columns.tolist(), CAND_TEXT_COLS)
    label_col = args.label_col or _pick_first_existing(
        df.columns.tolist(), CAND_LABEL_COLS
    )

    if args.timestamp_col:
        ts_col = args.timestamp_col
        candidates = []
    else:
        ts_col, candidates = _auto_pick_timestamp_col(df)

    print("[INFO] inferred cols:")
    print("  user_col   =", user_col)
    print("  product_col=", product_col)
    print("  rating_col =", rating_col)
    print("  text_col   =", text_col)
    print("  label_col  =", label_col)
    print("  ts_col     =", ts_col)
    if ts_col is None:
        print("[WARN] Could not confidently infer timestamp column.")
        print("[WARN] Timestamp candidates were:", candidates)
        print("[HINT] Re-run with: --timestamp_col <one_of_candidates>")
        raise SystemExit(2)

    need = [user_col, product_col, rating_col, text_col, label_col, ts_col]
    for c in need:
        if c is None or c not in df.columns:
            raise ValueError(f"Missing required column after inference: {c}")

    out = pd.DataFrame(
        {
            "user_id": df[user_col].astype(str),
            "product_id": df[product_col].astype(str),
            "rating": pd.to_numeric(df[rating_col], errors="coerce").astype("Int64"),
            "text": df[text_col].astype(str),
            "y_spam": pd.to_numeric(df[label_col], errors="coerce").astype("Int64"),
        }
    )

    out["ts_ms"] = _to_ts_ms(df[ts_col])
    bad_ts = out["ts_ms"].isna().mean()
    print(f"[INFO] ts_ms NA rate = {bad_ts:.3%}")
    if bad_ts > 0.05:
        print("[WARN] >5% timestamps could not be parsed; consider --timestamp_col.")

    dt = pd.to_datetime(out["ts_ms"].astype("float64"), unit="ms", errors="coerce", utc=True)
    out["month_id"] = dt.dt.strftime("%Y-%m")
    out["review_id"] = np.arange(len(out), dtype=np.int64)

    print("[INFO] unique users   :", out["user_id"].nunique())
    print("[INFO] unique products:", out["product_id"].nunique())
    print("[INFO] spam rate (y_spam==1):", float((out["y_spam"] == 1).mean()))

    out_path = os.path.join(args.out_dir, "nodes.parquet")
    out.to_parquet(out_path, index=False)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
