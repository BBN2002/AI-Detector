# -*- coding: utf-8 -*-
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import dgl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class SubgraphBatch:
    node_ids: torch.Tensor
    x0: torch.Tensor
    adj: torch.Tensor


class EgoDataset(Dataset):
    def __init__(self, g: dgl.DGLGraph, node_ids: np.ndarray):
        self.g = g
        self.node_ids = node_ids

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        return int(self.node_ids[idx])


class SimpleGraphTransformer(nn.Module):
    def __init__(self, dx: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dx, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.encoder(x, src_key_padding_mask=attn_mask)


class GrandDenoiser(nn.Module):
    def __init__(self, dx: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.backbone = SimpleGraphTransformer(dx, n_layers, n_heads, dropout)
        self.out = nn.Linear(dx, dx)

    def forward(self, x_noisy: torch.Tensor, pad_mask: torch.Tensor):
        h = self.backbone(x_noisy, pad_mask)
        return self.out(h)


def load_features(x_text, p_ai, h_det) -> np.ndarray:
    x = np.load(x_text)
    p = np.load(p_ai)
    h = np.load(h_det)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    return np.concatenate([x, p, h], axis=1)


def sample_subgraph(g: dgl.DGLGraph, center: int, hop: int, n_max: int) -> np.ndarray:
    nodes = {center}
    frontier = {center}
    for _ in range(hop):
        neighbors = set()
        for n in frontier:
            neighbors.update(g.successors(n).tolist())
            neighbors.update(g.predecessors(n).tolist())
        nodes.update(neighbors)
        frontier = neighbors
        if len(nodes) >= n_max:
            break
    nodes = list(nodes)
    if len(nodes) > n_max:
        nodes = nodes[:n_max]
    return np.array(nodes, dtype=np.int64)


def collate_fn(batch, g, x_proj, hop, n_max):
    xs = []
    masks = []
    for nid in batch:
        nodes = sample_subgraph(g, nid, hop, n_max)
        x = x_proj[nodes]
        pad = n_max - x.shape[0]
        if pad > 0:
            x = np.vstack([x, np.zeros((pad, x.shape[1]), dtype=x.dtype)])
            mask = np.array([False] * (n_max - pad) + [True] * pad)
        else:
            mask = np.array([False] * n_max)
        xs.append(x)
        masks.append(mask)
    x0 = torch.from_numpy(np.stack(xs)).float()
    pad_mask = torch.from_numpy(np.stack(masks))
    return x0, pad_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--x_text", required=True)
    ap.add_argument("--p_ai", required=True)
    ap.add_argument("--h_det", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dx", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    g, _ = dgl.load_graphs(args.graph)
    g = g[0]
    x = load_features(args.x_text, args.p_ai, args.h_det)
    proj = nn.Linear(x.shape[1], args.dx)
    x_proj = proj(torch.from_numpy(x).float()).detach().cpu().numpy()

    dataset = EgoDataset(g, np.arange(g.num_nodes()))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, g, x_proj, args.hop, args.n_max),
    )

    model = GrandDenoiser(args.dx, args.n_layers, args.n_heads, args.dropout)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    proj.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(proj.parameters()), lr=args.lr
    )
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x0, pad_mask in tqdm(loader, desc=f"grand epoch {epoch}"):
            x0 = x0.to(device)
            pad_mask = pad_mask.to(device)
            noise = torch.randn_like(x0)
            x_noisy = x0 + 0.1 * noise
            pred = model(x_noisy, pad_mask)
            loss = loss_fn(pred, x0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] loss={np.mean(losses):.6f}")

    torch.save(
        {"model_state": model.state_dict(), "proj_state": proj.state_dict()},
        os.path.join(args.out, "ckpt.pt"),
    )


if __name__ == "__main__":
    main()
