import argparse
import os
from typing import List, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.io import ensure_columns, parse_timestamp_series, read_jsonl


EDGE_SAME_USER = 0
EDGE_SAME_PROD_SAME_STAR = 1
EDGE_SAME_PROD_SAME_MONTH = 2


def _pairs_from_group(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if idx.size <= 1:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    comb = np.array(list(zip(*np.triu_indices(idx.size, k=1))), dtype=np.int64)
    if comb.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    src = idx[comb[:, 0]]
    dst = idx[comb[:, 1]]
    src_all = np.concatenate([src, dst])
    dst_all = np.concatenate([dst, src])
    return src_all, dst_all


def _build_edges(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_src: List[np.ndarray] = []
    all_dst: List[np.ndarray] = []
    all_type: List[np.ndarray] = []

    for _, group in tqdm(df.groupby("user_id"), desc="edges: same_user"):
        idx = group["review_id"].to_numpy(dtype=np.int64)
        src, dst = _pairs_from_group(idx)
        if src.size:
            all_src.append(src)
            all_dst.append(dst)
            all_type.append(np.full(src.shape, EDGE_SAME_USER, dtype=np.int64))

    for _, group in tqdm(df.groupby(["product_id", "rating"]), desc="edges: prod+star"):
        idx = group["review_id"].to_numpy(dtype=np.int64)
        src, dst = _pairs_from_group(idx)
        if src.size:
            all_src.append(src)
            all_dst.append(dst)
            all_type.append(np.full(src.shape, EDGE_SAME_PROD_SAME_STAR, dtype=np.int64))

    for _, group in tqdm(df.groupby(["product_id", "month_key"]), desc="edges: prod+month"):
        idx = group["review_id"].to_numpy(dtype=np.int64)
        src, dst = _pairs_from_group(idx)
        if src.size:
            all_src.append(src)
            all_dst.append(dst)
            all_type.append(np.full(src.shape, EDGE_SAME_PROD_SAME_MONTH, dtype=np.int64))

    if not all_src:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    edge_type = np.concatenate(all_type)
    return src, dst, edge_type


def build_graph(df: pd.DataFrame) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    src, dst, edge_type = _build_edges(df)
    g = dgl.graph((src, dst), num_nodes=df.shape[0])
    return g, torch.from_numpy(edge_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--format", choices=["jsonl", "csv", "parquet"], default="jsonl")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.format == "jsonl":
        records = read_jsonl(args.input_path)
        df = pd.DataFrame.from_records(records)
    elif args.format == "csv":
        df = pd.read_csv(args.input_path)
    else:
        df = pd.read_parquet(args.input_path)

    required = ["user_id", "product_id", "rating", "text", "y_spam"]
    ensure_columns(df, required)

    if "review_id" not in df.columns:
        df = df.copy()
        df["review_id"] = np.arange(df.shape[0], dtype=np.int64)

    df["rating"] = df["rating"].astype(int)
    if "month_id" in df.columns:
        df["month_key"] = df["month_id"].astype(str)
    elif "month_key" in df.columns:
        df["month_key"] = df["month_key"].astype(str)
    else:
        ts_col = "timestamp" if "timestamp" in df.columns else "ts_ms"
        if ts_col not in df.columns:
            raise ValueError("missing timestamp/ts_ms or month_id/month_key for month edge")
        dt, month_key = parse_timestamp_series(df[ts_col])
        df["timestamp_dt"] = dt
        df["month_key"] = month_key

    df = df.sort_values("review_id").reset_index(drop=True)
    df["review_id"] = np.arange(df.shape[0], dtype=np.int64)

    g, edge_type = build_graph(df)
    graph_path = os.path.join(args.out, "graph.bin")
    dgl.save_graphs(graph_path, [g], {"edge_type": edge_type})

    nodes_path = os.path.join(args.out, "nodes.parquet")
    df.to_parquet(nodes_path, index=False)

    meta = {
        "dataset": args.dataset,
        "num_nodes": int(df.shape[0]),
        "num_edges": int(g.num_edges()),
        "input_path": args.input_path,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
