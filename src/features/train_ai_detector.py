# -*- coding: utf-8 -*-
import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ReviewDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


class DetectorModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(hidden_size, proj_dim)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls).squeeze(-1)
        h_det = self.proj(cls)
        return logits, h_det


def make_loader(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    sampler=None,
    shuffle: bool = False,
):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=torch.tensor(labels, dtype=torch.float32),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
    )


def eval_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(
                batch.input_ids.to(device), batch.attention_mask.to(device)
            )
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            all_scores.append(scores)
            all_labels.append(batch.labels.numpy())
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--early_stop", type=int, default=2)
    ap.add_argument("--proj_dim", type=int, default=64)
    ap.add_argument("--balanced_sampling", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_parquet(args.nodes)
    label_col = "y_spam" if "y_spam" in df.columns else "y"
    if "text" not in df.columns:
        raise ValueError("nodes.parquet missing 'text' column")
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df[label_col].to_numpy()

    masks = torch.load(args.splits)
    mask_train = masks["mask_train"].numpy()
    mask_val = masks["mask_val"].numpy()

    train_texts = [t for t, m in zip(texts, mask_train) if m]
    train_labels = labels[mask_train]
    val_texts = [t for t, m in zip(texts, mask_val) if m]
    val_labels = labels[mask_val]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backbone = AutoModel.from_pretrained(args.model)
    model = DetectorModel(backbone, backbone.config.hidden_size, args.proj_dim)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = ReviewDataset(train_texts, train_labels)
    val_dataset = ReviewDataset(val_texts, val_labels)

    sampler = None
    if args.balanced_sampling and len(train_labels) > 0:
        class_counts = np.bincount(train_labels.astype(int), minlength=2)
        weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = weights[train_labels.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

    train_loader = make_loader(
        train_dataset,
        tokenizer,
        args.batch_size,
        args.max_length,
        sampler=sampler,
        shuffle=sampler is None,
    )
    val_loader = make_loader(
        val_dataset, tokenizer, args.batch_size, args.max_length, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            optimizer.zero_grad()
            logits, _ = model(
                batch.input_ids.to(device), batch.attention_mask.to(device)
            )
            loss = criterion(logits, batch.labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_auc = eval_auc(model, val_loader, device)
        print(f"[Epoch {epoch}] loss={np.mean(losses):.4f} val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "proj_dim": args.proj_dim,
                },
                os.path.join(args.out, "ckpt.pt"),
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.early_stop:
                break

    with open(os.path.join(args.out, "best_val_auc.txt"), "w", encoding="utf-8") as f:
        f.write(f"{best_auc:.6f}\n")


if __name__ == "__main__":
    main()
