# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="nodes.parquet")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    df = pd.read_parquet(args.nodes)
    if "text" not in df.columns:
        raise ValueError("nodes.parquet missing 'text' column")
    texts = df["text"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    out_dtype = np.float16 if args.dtype == "fp16" else np.float32
    embeddings = np.zeros((len(texts), model.config.hidden_size), dtype=out_dtype)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch), desc="bert cls"):
            batch_texts = texts[i : i + args.batch]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            cls = cls.detach().cpu().numpy().astype(out_dtype)
            embeddings[i : i + cls.shape[0]] = cls

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, embeddings)
    print(f"[OK] wrote: {args.out} shape={embeddings.shape} dtype={embeddings.dtype}")


if __name__ == "__main__":
    main()
