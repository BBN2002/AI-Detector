# train_grand_subgraph.py (更新版: 完整GRAND扩散训练)
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

from diffusion_model import DiffusionModel  # 从提供的diffusion_model.py导入
from transformer import GraphTransformer  # 从提供的transformer.py导入
from utils import ensure_two_channel_onehot_edges, estimate_edge_marginal, discrete_forward_sample, safe_compute_graph_features  # 从utils.py导入

@dataclass
class SubgraphBatch:
    node_ids: torch.Tensor
    x0: torch.Tensor
    adj: torch.Tensor
    node_mask: torch.Tensor = None  # 新增: 节点掩码

class EgoDataset(Dataset):
    def __init__(self, g: dgl.DGLGraph, node_ids: np.ndarray, max_n_nodes: int = 64):
        self.g = g
        self.node_ids = node_ids
        self.max_n_nodes = max_n_nodes

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        nid = int(self.node_ids[idx])
        # 采样k-hop子图 (k=2 for robustness)
        subg = dgl.khop_in_subgraph(self.g, nid, k=2)[0]
        # 填充到固定大小
        n_nodes = subg.num_nodes()
        if n_nodes > self.max_n_nodes:
            subg = dgl.node_subgraph(subg, torch.randperm(n_nodes)[:self.max_n_nodes])
        x0 = subg.ndata['feat'].float()  # 假设节点特征为'feat'
        adj = subg.adj().to_dense().float()  # 稠密邻接矩阵
        pad_mask = torch.ones(self.max_n_nodes, dtype=torch.float32)
        pad_mask[n_nodes:] = 0.0  # padding mask
        # 填充
        x_pad = torch.zeros((self.max_n_nodes, x0.shape[-1]))
        adj_pad = torch.zeros((self.max_n_nodes, self.max_n_nodes))
        x_pad[:n_nodes] = x0
        adj_pad[:n_nodes, :n_nodes] = adj
        return nid, x_pad, adj_pad, pad_mask

def main():
    parser = argparse.ArgumentParser(description="Train GRAND Diffusion Model")
    parser.add_argument('--graph', type=str, required=True, help='Path to DGL graph.bin')
    parser.add_argument('--out', type=str, required=True, help='Output dir for ckpt.pt')
    parser.add_argument('--dx', type=int, default=64, help='Node feature dim')
    parser.add_argument('--de', type=int, default=2, help='Edge feature dim (one-hot)')
    parser.add_argument('--n_layers', type=int, default=4, help='Transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Diffusion timesteps T')
    parser.add_argument('--lamda', type=float, default=0.5, help='Node-edge score weight')
    parser.add_argument('--discrete_edge_noise', action='store_true', help='Use discrete edge diffusion')
    parser.add_argument('--marginal_edge_noise', action='store_true', help='Use marginal edge prob')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载图
    g = dgl.load_graphs(args.graph)[0][0]
    node_ids = np.arange(g.num_nodes())
    dataset = EgoDataset(g, node_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 配置 (基于论文)
    cfg = {
        'diffusion_steps': args.diffusion_steps,
        'lamda': args.lamda,
        'discrete_edge_noise': args.discrete_edge_noise,
        'marginal_edge_noise': args.marginal_edge_noise
    }
    # 估计边先验 (structural priors: avg density)
    avg_edge_density = g.num_edges() / (g.num_nodes() * (g.num_nodes() - 1))  # 全局密度作为prior
    dataset_info = {'max_n_nodes': dataset.max_n_nodes}  # 模拟数据集信息

    model = DiffusionModel(cfg, dataset_info, avg_edge_density, device).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for nids, x0, adj, pad_mask in tqdm(loader, desc=f"Epoch {epoch}"):
            x0, adj, pad_mask = x0.to(device), adj.to(device), pad_mask.to(device)
            # 前向扩散 & 去噪训练 (完整多步)
            loss = model.compute_loss(x0, adj, pad_mask)  # 假设diffusion_model有compute_loss方法
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        avg_loss = np.mean(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(args.out, 'ckpt.pt'))
    print(f"[OK] Saved model to {args.out}/ckpt.pt")

if __name__ == "__main__":
    main()