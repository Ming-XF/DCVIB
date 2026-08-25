"""GNN 公共工具函数。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """图卷积层：out = adj_norm @ (x @ W)，adj_norm 为归一化邻接（稠密张量）。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        return adj_norm @ (x @ self.weight)


class GCNEncoder(nn.Module):
    """多层 GCN 编码器：逐层 Â·(H·W)，除末层外每层后接 ReLU + Dropout。

    各层维度由 hidden_dims 列表独立指定（如 (512, 256)：1433→512→256），
    输出 h 为 hidden_dims[-1] 维。
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            self.convs.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, adj_norm)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def build_gcn_encoder(
    input_dim: int = 1433,
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout: float = 0.2,
) -> GCNEncoder:
    """构造 GCN 特征提取器：层数与维度由 hidden_dims 列表决定（与 MLP 骨干约定一致）。"""
    return GCNEncoder(input_dim, hidden_dims, dropout)


def graph_readout(h: torch.Tensor, batch: torch.Tensor, pool: str = "mean"):
    """图级读出：把节点表示按所属图聚合为 (B, d) 的图表示。

    batch 为 (N,) 长整型节点→图索引（由 collate_zinc 的 repeat_interleave
    生成，必然覆盖 0..B-1）。mean 池化用 index_add_ 求和后除以各图节点数
    （torch.bincount 统计），全程 GPU 原地完成、不触发与 CPU 的同步。
    """
    if pool != "mean":
        raise ValueError(f"不支持的图读出方式：{pool}")
    counts = torch.bincount(batch).float()
    sums = torch.zeros(counts.size(0), h.size(1), device=h.device, dtype=h.dtype)
    sums.index_add_(0, batch, h)
    return sums / counts.unsqueeze(1)


def kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask=None):
    """逐节点对角高斯 KL(q||p)，按 mask 平均（mask 为 None 时全批平均）。

    与 model/mlp/utils.py 的 kl_divergence 同款闭式解和 clamp 约定，区别是
    支持按节点 mask 计算——转导式图训练中 KL 只在训练节点上计算，避免
    CEB 的标签条件先验用到验证/测试节点标签（标签泄漏）。
    """
    logvar = logvar.clamp(-10.0, 10.0)
    logvar_p = logvar_p.clamp(-10.0, 10.0)
    var, var_p = logvar.exp(), logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar + (var + (mu - mu_p).pow(2)) / var_p - 1.0)
    kl = kl.sum(dim=1)
    if mask is None:
        return kl.mean()
    return kl[mask].mean()
