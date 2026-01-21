# infer_grand_scores.py (更新版: 完整GRAND推断与鲁棒评分)
# -*- coding: utf-8 -*-
import argparse
import os

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model import DiffusionModel  # 导入更新模型
from utils import robust_zscore, plot_roc_pr_curve  # 导入MAD norm和可视化

def main():
    parser = argparse.ArgumentParser(description="Infer GRAND scores with robust scoring")
    parser.add_argument('--graph', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True, help='Path to ckpt.pt')
    parser.add_argument('--out_node', type=str, required=True)
    parser.add_argument('--out_edge', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--edge_agg', type=str, default='diff', choices=['diff', 'mean'])
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--topk_ratio', type=float, default=0.1, help='Top-k pooling ratio for sparse preservation')
    parser.add_argument('--var_threshold', type=float, default=1e-4, help='Degeneracy var threshold for fallback')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 加载图和模型
    g = dgl.load_graphs(args.graph)[0][0]
    model = DiffusionModel(...)  # 初始化同train，加载cfg/dataset_info
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device).eval()

    # 结构prior: clean edge stats (全局平均度/密度，无标签泄露)
    avg_degree = g.out_degrees().float().mean().item()
    avg_density = g.num_edges() / (g.num_nodes() * (g.num_nodes() - 1))

    # 推断: 多步逆扩散 + 鲁棒评分
    s_node = np.zeros(g.num_nodes())
    s_edge = np.zeros(g.num_nodes())
    counts = np.zeros(g.num_nodes())

    # 子图采样loader (同train)
    dataset = EgoDataset(g, np.arange(g.num_nodes()))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for nids, x_clean, adj_clean, pad_mask in tqdm(loader):
            x_clean, adj_clean, pad_mask = x_clean.to(device), adj_clean.to(device), pad_mask.to(device)
            # 多步逆扩散 (论文long reverse chains)
            x_noisy, e_noisy = model.forward_diffuse(x_clean, adj_clean, pad_mask)  # 假设方法
            x_gen, e_gen = model.reverse_denoise(x_noisy, e_noisy, pad_mask, time_steps=model.time_steps,
                                                 priors={'avg_degree': avg_degree, 'avg_density': avg_density})
            
            # 退化检测: 监控分布collapse
            var_x = x_gen.var(dim=-1).mean().item()
            if var_x < args.var_threshold:
                print(f"[WARN] Degeneracy detected (var={var_x:.6f}), fallback to recon error")
                node_err = ((x_gen - x_clean) ** 2).mean(dim=-1)
                edge_err = ((e_gen - adj_clean) ** 2).mean(dim=-1)
            else:
                # 鲁棒评分: Top-k pooling (preserve sparse deviations)
                node_err = ((x_gen - x_clean) ** 2).mean(dim=-1) * pad_mask
                k = int(args.topk_ratio * pad_mask.sum(dim=1).max().item())
                node_err_topk = torch.topk(node_err, k, dim=1)[0].mean(dim=1)
                
                if args.discrete_edge_noise:
                    edge_err = ((e_gen - ensure_two_channel_onehot_edges(adj_clean, pad_mask)) ** 2).mean(dim=-1)
                else:
                    edge_err = ((e_gen - adj_clean) ** 2).mean(dim=-1)
                edge_err_topk = torch.topk(edge_err.view(edge_err.size(0), -1), k, dim=1)[0].mean(dim=1)
                
                # 聚合 (lamda权重)
                score = (1 - model.lamda) * node_err_topk + model.lamda * edge_err_topk
            
            # 累加 (per-node)
            s_node[nids.numpy()] += node_err_topk.cpu().numpy()
            s_edge[nids.numpy()] += edge_err_topk.cpu().numpy()
            counts[nids.numpy()] += 1

    # 平均 + MAD norm (cross-graph comparability)
    counts = np.maximum(counts, 1)
    s_node /= counts
    s_edge /= counts
    if args.normalize:
        s_node = robust_zscore(s_node)  # MAD-based
        s_edge = robust_zscore(s_edge)

    # 保存
    np.save(args.out_node, s_node.reshape(-1, 1))
    np.save(args.out_edge, s_edge.reshape(-1, 1))
    print(f"[OK] Wrote {args.out_node} and {args.out_edge}")

if __name__ == "__main__":
    main()