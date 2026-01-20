# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class ReviewDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_p", required=True)
    ap.add_argument("--out_h", required=True)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = ap.parse_args()

    df = pd.read_parquet(args.nodes)
    if "text" not in df.columns:
        raise ValueError("nodes.parquet missing 'text' column")
    texts = df["text"].fillna("").astype(str).tolist()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt["model_name"]
    proj_dim = int(ckpt["proj_dim"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    model = DetectorModel(backbone, backbone.config.hidden_size, proj_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ReviewDataset(texts)

    def collate_fn(batch):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    out_dtype = np.float16 if args.dtype == "fp16" else np.float32
    p_ai = np.zeros((len(texts), 1), dtype=np.float32)
    h_det = np.zeros((len(texts), proj_dim), dtype=out_dtype)

    start = 0
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(loader, desc="infer ai detector"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits, h = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1, 1)
            h_np = h.detach().cpu().numpy().astype(out_dtype)
            end = start + probs.shape[0]
            p_ai[start:end] = probs
            h_det[start:end] = h_np
            start = end

    os.makedirs(os.path.dirname(args.out_p), exist_ok=True)
    np.save(args.out_p, p_ai)
    np.save(args.out_h, h_det)
    print(f"[OK] wrote: {args.out_p} {p_ai.shape}")
    print(f"[OK] wrote: {args.out_h} {h_det.shape}")


if __name__ == "__main__":
    main()
