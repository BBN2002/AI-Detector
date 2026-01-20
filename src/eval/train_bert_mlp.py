# -*- coding: utf-8 -*-
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--x_text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--early_stop", type=int, default=5)
    ap.add_argument("--topk_ratio", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df = pd.read_parquet(args.nodes)
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    labels = df[label_col].to_numpy(dtype=np.int64)

    masks = torch.load(args.splits)
    train_idx = np.where(masks["mask_train"].numpy())[0]
    val_idx = np.where(masks["mask_val"].numpy())[0]
    test_idx = np.where(masks["mask_test"].numpy())[0]

    x = np.load(args.x_text).astype(np.float32)
    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx])
    x_val = torch.from_numpy(x[val_idx])
    y_val = torch.from_numpy(labels[val_idx])
    x_test = torch.from_numpy(x[test_idx])
    y_test = torch.from_numpy(labels[test_idx])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MLP(x.shape[1], args.hidden_dim, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
    )

    best_auc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        oof = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                oof.append(logits.detach().cpu())
        oof = torch.cat(oof, dim=0)
        val_scores = torch.softmax(oof, dim=1)[:, 1].numpy()
        val_auc = eval_auc(val_scores, y_val.numpy())
        print(f"[Epoch {epoch}] loss={np.mean(losses):.6f} val_auc={val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop:
                break

    os.makedirs(args.out, exist_ok=True)
    if best_state is not None:
        torch.save({"model_state": best_state}, os.path.join(args.out, "ckpt.pt"))

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )
    oof = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            oof.append(logits.detach().cpu())
    oof = torch.cat(oof, dim=0)
    test_scores = torch.softmax(oof, dim=1)[:, 1].numpy()
    test_labels = y_test.numpy()

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
    with open(os.path.join(args.out, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, "best_val_auc.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_auc:.6f}\n")


if __name__ == "__main__":
    main()
