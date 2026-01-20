# -*- coding: utf-8 -*-
import argparse
import os

import dgl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


class NodeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + eps)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--x_text", required=True)
    ap.add_argument("--p_ai", required=True)
    ap.add_argument("--h_det", required=True)
    ap.add_argument("--out_node", required=True)
    ap.add_argument("--out_edge", required=True)
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--n_max", type=int, default=1024)
    ap.add_argument("--samples_per_node", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dx", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--normalize", action="store_true", help="median/MAD z-score")
    ap.add_argument("--edge_agg", choices=["diff", "mean"], default="diff")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    g, _ = dgl.load_graphs(args.graph)
    g = g[0]
    x = load_features(args.x_text, args.p_ai, args.h_det)
    proj = nn.Linear(x.shape[1], args.dx)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = GrandDenoiser(args.dx, args.n_layers, args.n_heads, args.dropout)
    model.load_state_dict(ckpt["model_state"])
    proj.load_state_dict(ckpt["proj_state"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    proj.to(device)
    model.eval()
    proj.eval()

    dataset = NodeDataset(g.num_nodes())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    s_node = np.zeros((g.num_nodes(),), dtype=np.float32)
    counts = np.zeros((g.num_nodes(),), dtype=np.int32)

    with torch.no_grad():
        for batch_ids in tqdm(loader, desc="grand infer"):
            batch_ids = batch_ids.numpy().tolist()
            for nid in batch_ids:
                for _ in range(args.samples_per_node):
                    nodes = sample_subgraph(g, nid, args.hop, args.n_max)
                    x_sub = torch.from_numpy(x[nodes]).float().to(device)
                    x_sub = proj(x_sub)
                    pad = args.n_max - x_sub.shape[0]
                    if pad > 0:
                        x_sub = torch.cat(
                            [x_sub, torch.zeros(pad, args.dx, device=device)], dim=0
                        )
                        pad_mask = torch.tensor(
                            [False] * (args.n_max - pad) + [True] * pad,
                            device=device,
                        )
                    else:
                        pad_mask = torch.zeros(args.n_max, dtype=torch.bool, device=device)
                    x_sub = x_sub.unsqueeze(0)
                    pad_mask = pad_mask.unsqueeze(0)
                    noise = torch.randn_like(x_sub)
                    x_noisy = x_sub + 0.1 * noise
                    pred = model(x_noisy, pad_mask)
                    err = ((pred - x_sub) ** 2).mean().item()
                    s_node[nid] += err
                    counts[nid] += 1

    counts = np.maximum(counts, 1)
    s_node = s_node / counts

    # edge aggregation from node errors
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()
    if args.edge_agg == "diff":
        edge_err = np.abs(s_node[src] - s_node[dst])
    else:
        edge_err = s_node[dst]
    s_edge = np.zeros_like(s_node)
    edge_cnt = np.zeros_like(counts)
    np.add.at(s_edge, src, edge_err)
    np.add.at(edge_cnt, src, 1)
    edge_cnt = np.maximum(edge_cnt, 1)
    s_edge = s_edge / edge_cnt

    if args.normalize:
        s_node = robust_zscore(s_node)
        s_edge = robust_zscore(s_edge)

    os.makedirs(os.path.dirname(args.out_node), exist_ok=True)
    np.save(args.out_node, s_node.reshape(-1, 1))
    np.save(args.out_edge, s_edge.reshape(-1, 1))
    print(f"[OK] wrote: {args.out_node}")
    print(f"[OK] wrote: {args.out_edge}")


if __name__ == "__main__":
    main()
