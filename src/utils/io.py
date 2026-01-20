import json
from datetime import datetime, timezone
from typing import Iterable, List, Tuple

import pandas as pd


def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_timestamp_series(ts_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Parse timestamp to pandas datetime (UTC) and return (dt, month_key).
    Supports:
    - int/float epoch seconds or milliseconds
    - ISO strings or common date formats
    """
    if pd.api.types.is_numeric_dtype(ts_series):
        ts_numeric = ts_series.astype("int64")
        is_ms = ts_numeric.abs() >= 1_000_000_000_000
        seconds = ts_numeric.copy()
        seconds[is_ms] = (ts_numeric[is_ms] // 1000)
        dt = pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ts_series, utc=True, errors="coerce")

    if dt.isna().any():
        bad = dt.isna().sum()
        raise ValueError(f"timestamp parse failed for {bad} rows")

    month_key = dt.dt.strftime("%Y-%m")
    return dt, month_key


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
