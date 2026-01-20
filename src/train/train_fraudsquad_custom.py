# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR

FS_ROOT = "/root/autodl-fs/fraudsquad_project/data/FraudSquad-5389"
sys.path.append(os.path.join(FS_ROOT, "methods"))
from fraudsquad_src import GraphAttnModel, early_stopper
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--x_text", required=True)
    ap.add_argument("--p_ai", default=None)
    ap.add_argument("--h_det", default=None)
    ap.add_argument("--s_node", default=None)
    ap.add_argument("--s_edge", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--early_stop", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    g, _ = dgl.load_graphs(args.graph)
    g = g[0]
    df = pd.read_parquet(args.nodes)
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    labels_np = df[label_col].to_numpy(dtype=np.int64)

    masks = torch.load(args.splits)
    train_idx = np.where(masks["mask_train"].numpy())[0]
    val_idx = np.where(masks["mask_val"].numpy())[0]
    test_idx = np.where(masks["mask_test"].numpy())[0]

    features = load_features(args)
    device = args.device
    g = g.to(device)
    labels = torch.from_numpy(labels_np).long().to(device)
    feat_tensor = torch.from_numpy(features).float().to(device)

    train_sampler = MultiLayerFullNeighborSampler(args.n_layers)
    val_sampler = MultiLayerFullNeighborSampler(args.n_layers)

    train_loader = DataLoader(
        g,
        torch.from_numpy(train_idx).long().to(device),
        train_sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        g,
        torch.from_numpy(val_idx).long().to(device),
        val_sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = GraphAttnModel(
        in_feats=features.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_classes=2,
        heads=[args.num_heads] * args.n_layers,
        activation=nn.PReLU(),
        drop=[args.dropout, args.dropout],
        gated=True,
        n2v_feat=False,
        ref_df=None,
        cat_features=[],
        device=device,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)
    stopper = early_stopper(patience=args.early_stop, verbose=True)

    best_auc = -1.0
    best_state = None
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for _, (input_nodes, seeds, blocks) in enumerate(train_loader):
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                feat_tensor, {}, labels, seeds, input_nodes, device
            )
            blocks = [block.to(device) for block in blocks]
            logits = model(blocks, batch_inputs, lpa_labels, None)
            mask = batch_labels == 2
            logits = logits[~mask]
            batch_labels = batch_labels[~mask]
            loss = loss_fn(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        oof = torch.zeros((features.shape[0], 2), device=device)
        with torch.no_grad():
            for _, (input_nodes, seeds, blocks) in enumerate(val_loader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                    feat_tensor, {}, labels, seeds, input_nodes, device
                )
                blocks = [block.to(device) for block in blocks]
                logits = model(blocks, batch_inputs, lpa_labels, None)
                oof[seeds] = logits

        val_scores = torch.softmax(oof[val_idx], dim=1)[:, 1].cpu().numpy()
        val_labels = labels[val_idx].cpu().numpy()
        val_auc = eval_auc(val_scores, val_labels)
        print(
            f"[Epoch {epoch}] loss={np.mean(train_losses):.6f} val_auc={val_auc:.6f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        stopper.earlystop(-val_auc, model=model)
        if stopper.is_earlystop:
            break

    if best_state is not None:
        torch.save({"model_state": best_state, "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))

    with open(os.path.join(args.out, "best_val_auc.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_auc:.6f}\n")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    sampler = MultiLayerFullNeighborSampler(args.n_layers)
    test_loader = DataLoader(
        g,
        torch.from_numpy(test_idx).long().to(device),
        sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    oof = torch.zeros((features.shape[0], 2), device=device)
    with torch.no_grad():
        for _, (input_nodes, seeds, blocks) in enumerate(test_loader):
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                feat_tensor, {}, labels, seeds, input_nodes, device
            )
            blocks = [block.to(device) for block in blocks]
            logits = model(blocks, batch_inputs, lpa_labels, None)
            oof[seeds] = logits
    test_scores = torch.softmax(oof[test_idx], dim=1)[:, 1].cpu().numpy()
    test_labels = labels[test_idx].cpu().numpy()
    test_auc = eval_auc(test_scores, test_labels)
    ap = float(average_precision_score(test_labels, test_scores))
    with open(os.path.join(args.out, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump({"auc": test_auc, "ap": ap}, f, indent=2)


if __name__ == "__main__":
    main()
