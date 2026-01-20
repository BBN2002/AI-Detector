# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import ttest_rel


def load_metric(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_metrics(root: str, ds: str, model: str, seeds: List[str]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {"auc": [], "ap": [], "precision_at_k": [], "recall_at_k": []}
    for seed in seeds:
        path = os.path.join(root, model, ds, f"seed{seed}", "metrics_test.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        m = load_metric(path)
        for k in list(values.keys()):
            values[k].append(float(m.get(k, 0.0)))
    return values


def mean_std(vals: List[float]):
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/root/autodl-fs/fraudsquad_project/runs")
    args = ap.parse_args()

    models = ["fs", "fs_hc3", "fs_grand", "fs_pp"]
    metrics = ["auc", "ap", "precision_at_k", "recall_at_k"]

    result = {}
    for ds in args.datasets:
        result[ds] = {}
        base = collect_metrics(args.root, ds, "fs", args.seeds)
        for model in models:
            cur = collect_metrics(args.root, ds, model, args.seeds)
            entry = {"mean": {}, "std": {}, "p_vs_fs": {}}
            for metric in metrics:
                m_mean, m_std = mean_std(cur[metric])
                entry["mean"][metric] = m_mean
                entry["std"][metric] = m_std
                if model == "fs":
                    entry["p_vs_fs"][metric] = None
                else:
                    stat, pval = ttest_rel(cur[metric], base[metric])
                    entry["p_vs_fs"][metric] = float(pval)
            result[ds][model] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()
