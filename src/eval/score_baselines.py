# -*- coding: utf-8 -*-
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def precision_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int):
    if k <= 0:
        return 0.0, 0.0
    idx = np.argsort(-scores)[:k]
    tp = labels[idx].sum()
    precision = float(tp / k)
    total_pos = labels.sum()
    recall = float(tp / total_pos) if total_pos > 0 else 0.0
    return precision, recall


def load_vector(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--score", default=None, help="score npy (N or Nx1)")
    ap.add_argument("--p_ai", default=None, help="p_ai npy (N or Nx1)")
    ap.add_argument("--s_node", default=None)
    ap.add_argument("--s_edge", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk_ratio", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.nodes)
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    labels = df[label_col].to_numpy(dtype=np.int64)

    masks = torch.load(args.splits)
    test_idx = np.where(masks["mask_test"].numpy())[0]

    if args.score:
        scores = load_vector(args.score)
    elif args.p_ai:
        scores = load_vector(args.p_ai)
    else:
        if not args.s_node or not args.s_edge:
            raise ValueError("Need --score or --p_ai or --s_node/--s_edge")
        s_node = load_vector(args.s_node)
        s_edge = load_vector(args.s_edge)
        scores = args.alpha * s_node + (1.0 - args.alpha) * s_edge

    test_scores = scores[test_idx]
    test_labels = labels[test_idx]

    if args.topk_ratio is None:
        topk_ratio = float(labels.mean())
    else:
        topk_ratio = float(args.topk_ratio)
    k = int(max(1, int(round(topk_ratio * len(test_idx)))))

    test_auc = eval_auc(test_scores, test_labels)
    ap_score = float(average_precision_score(test_labels, test_scores))
    p_at, r_at = precision_recall_at_k(test_scores, test_labels, k)

    out = {
        "auc": test_auc,
        "ap": ap_score,
        "precision_at_k": p_at,
        "recall_at_k": r_at,
        "topk": k,
        "topk_ratio": topk_ratio,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()
