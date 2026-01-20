# -*- coding: utf-8 -*-
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import average_precision_score, roc_auc_score

FS_ROOT = "/root/autodl-fs/fraudsquad_project/data/FraudSquad-5389"
import sys

sys.path.append(os.path.join(FS_ROOT, "methods"))
from fraudsquad_src import GraphAttnModel
from fraudsquad_src.lpa import load_lpa_subtensor


def load_features(args) -> np.ndarray:
    feats = [np.load(args.x_text)]
    if args.p_ai:
        p = np.load(args.p_ai)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        feats.append(p)
    if args.h_det:
        feats.append(np.load(args.h_det))
    if args.s_node:
        s = np.load(args.s_node)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        feats.append(s)
    if args.s_edge:
        s = np.load(args.s_edge)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        feats.append(s)
    return np.concatenate(feats, axis=1)


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


def require_feature(ckpt_args, name: str, path: str):
    need = ckpt_args.get(name)
    if need and not path:
        raise ValueError(f"ckpt expects {name}, but target path not provided.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--x_text", required=True)
    ap.add_argument("--p_ai", default=None)
    ap.add_argument("--h_det", default=None)
    ap.add_argument("--s_node", default=None)
    ap.add_argument("--s_edge", default=None)
    ap.add_argument("--topk_ratio", type=float, default=0.03)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    require_feature(ckpt_args, "p_ai", args.p_ai)
    require_feature(ckpt_args, "h_det", args.h_det)
    require_feature(ckpt_args, "s_node", args.s_node)
    require_feature(ckpt_args, "s_edge", args.s_edge)

    g, _ = dgl.load_graphs(args.graph)
    g = g[0]
    df = pd.read_parquet(args.nodes)
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    labels_np = df[label_col].to_numpy(dtype=np.int64)
    masks = torch.load(args.splits)
    test_idx = np.where(masks["mask_test"].numpy())[0]

    features = load_features(args)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    g = g.to(device)
    labels = torch.from_numpy(labels_np).long().to(device)
    feat_tensor = torch.from_numpy(features).float().to(device)

    model = GraphAttnModel(
        in_feats=features.shape[1],
        hidden_dim=ckpt_args.get("hidden_dim", 100),
        n_layers=ckpt_args.get("n_layers", 2),
        n_classes=2,
        heads=[ckpt_args.get("num_heads", 3)] * ckpt_args.get("n_layers", 2),
        activation=nn.PReLU(),
        drop=[ckpt_args.get("dropout", 0.2), ckpt_args.get("dropout", 0.2)],
        gated=True,
        n2v_feat=False,
        ref_df=None,
        cat_features=[],
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    sampler = MultiLayerFullNeighborSampler(ckpt_args.get("n_layers", 2))
    test_loader = DataLoader(
        g,
        torch.from_numpy(test_idx).long().to(device),
        sampler,
        device=device,
        use_ddp=False,
        batch_size=ckpt_args.get("batch_size", 512),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    oof = torch.zeros((features.shape[0], 2), device=device)
    with torch.no_grad():
        for _, (input_nodes, seeds, blocks) in enumerate(test_loader):
            batch_inputs, _, batch_labels, lpa_labels = load_lpa_subtensor(
                feat_tensor, {}, labels, seeds, input_nodes, device
            )
            blocks = [block.to(device) for block in blocks]
            logits = model(blocks, batch_inputs, lpa_labels, None)
            oof[seeds] = logits

    test_scores = torch.softmax(oof[test_idx], dim=1)[:, 1].cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy()
    k = int(max(1, int(round(args.topk_ratio * len(test_idx)))))

    test_auc = eval_auc(test_scores, test_labels)
    ap = float(average_precision_score(test_labels, test_scores))
    p_at, r_at = precision_recall_at_k(test_scores, test_labels, k)

    out = {
        "auc": test_auc,
        "ap": ap,
        "precision_at_k": p_at,
        "recall_at_k": r_at,
        "topk": k,
        "topk_ratio": args.topk_ratio,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote: {args.out}")


if __name__ == "__main__":
    main()
