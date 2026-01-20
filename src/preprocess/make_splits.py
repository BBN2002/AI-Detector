import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from src.utils.io import ensure_columns


def stratified_split(
    y: np.ndarray, train_ratio: float, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(y.shape[0])
    train_idx = []
    val_idx = []
    test_idx = []

    for label in [0, 1]:
        idx = idx_all[y == label]
        rng.shuffle(idx)
        n = idx.size
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = max(n_train, 1) if n > 0 else 0
        n_val = max(n_val, 1) if n - n_train > 1 else max(n_val, 0)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--nodes", required=True, help="nodes.parquet path")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.09)
    parser.add_argument("--test_ratio", type=float, default=0.90)
    args = parser.parse_args()

    if not np.isclose(
        args.train_ratio + args.val_ratio + args.test_ratio, 1.0, atol=1e-6
    ):
        raise ValueError("train/val/test ratios must sum to 1")

    df = pd.read_parquet(args.nodes)
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    ensure_columns(df, ["review_id", label_col])

    y = df[label_col].to_numpy(dtype=np.int64)
    n = y.shape[0]
    os.makedirs(args.out, exist_ok=True)

    for seed in args.seeds:
        train_idx, val_idx, test_idx = stratified_split(
            y, args.train_ratio, args.val_ratio, seed
        )
        mask_train = torch.zeros(n, dtype=torch.bool)
        mask_val = torch.zeros(n, dtype=torch.bool)
        mask_test = torch.zeros(n, dtype=torch.bool)
        mask_train[train_idx] = True
        mask_val[val_idx] = True
        mask_test[test_idx] = True

        out_path = os.path.join(args.out, f"seed{seed}_masks.pt")
        torch.save(
            {
                "mask_train": mask_train,
                "mask_val": mask_val,
                "mask_test": mask_test,
            },
            out_path,
        )

        stats = {
            "seed": seed,
            "n_total": int(n),
            "n_train": int(mask_train.sum().item()),
            "n_val": int(mask_val.sum().item()),
            "n_test": int(mask_test.sum().item()),
            "spam_rate": float(y.mean()),
            "spam_train": int(y[mask_train.numpy()].sum()),
            "spam_val": int(y[mask_val.numpy()].sum()),
            "spam_test": int(y[mask_test.numpy()].sum()),
        }
        with open(
            os.path.join(args.out, f"seed{seed}_stats.json"),
            "w",
            encoding="utf-8",
        ) as f:
            import json

            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
