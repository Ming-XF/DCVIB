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
    """两层 GCN 编码器：ReLU(Dropout(ÂXW1)) -> ÂHW2，输出 h（hidden_dim 维）。"""

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, adj_norm):
        h = F.relu(self.conv1(x, adj_norm))
        h = self.dropout(h)
        return self.conv2(h, adj_norm)


def build_gcn_encoder(
    input_dim: int = 1433,
    hidden_dim: int = 16,
    dropout: float = 0.2,
) -> GCNEncoder:
    """构造两层 GCN 特征提取器（Kipf & Welling 标准结构，hidden 默认 16）。"""
    return GCNEncoder(input_dim, hidden_dim, dropout)


def kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask=None):
    """逐节点对角高斯 KL(q||p)，按 mask 平均（mask 为 None 时全批平均）。

    与 model/mlp/utils.py 的 kl_divergence 同款闭式解和 clamp 约定，区别是
    支持按节点 mask 计算——转导式图训练中 KL 只在训练节点上计算，避免
    SCVIB/DCVIB 的标签条件先验用到验证/测试节点标签（标签泄漏）。
    """
    logvar = logvar.clamp(-10.0, 10.0)
    logvar_p = logvar_p.clamp(-10.0, 10.0)
    var, var_p = logvar.exp(), logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar + (var + (mu - mu_p).pow(2)) / var_p - 1.0)
    kl = kl.sum(dim=1)
    if mask is None:
        return kl.mean()
    return kl[mask].mean()
