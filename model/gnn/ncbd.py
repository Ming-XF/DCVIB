"""NCBD-NONL（Neural Collapse by Design，ICML 2026）GNN 模型。

与 MLP 版同构（见 model/mlp/ncbd.py），仅编码器换成 GCN。分类（cora 转导
式）：NONL 损失按 mask（训练时 train_mask）只在训练节点子集上计算，避免
对比负样本接触验证/测试节点标签（与 GNN 版 CEB 的 KL-mask 约定一致）；
logits 为全图节点的温度缩放余弦得分。batch 非 None 时（图级任务，当前
未使用）编码器后先做图读出。仅分类任务。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import nonl_loss
from .utils import build_gcn_encoder, graph_readout


class NCBD(nn.Module):
    """GCN 编码器 + 单位球原型分类器 + NONL 损失（mask 节点上计算）。"""

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        temperature: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.pooling = pooling
        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dims[-1]))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None, batch=None):
        """返回 (logits, nonl_loss)；NONL 在 mask 节点子集上计算。"""
        h = self.encoder(x, adj_norm)
        if batch is not None:
            h = graph_readout(h, batch, self.pooling)
        u = F.normalize(h, dim=1)
        w = F.normalize(self.prototypes, dim=1)
        logits = (u @ w.t()) / self.temperature
        if labels is None:
            return logits, None
        if mask is not None:
            u_m, y_m = u[mask], labels[mask]
        else:
            u_m, y_m = u, labels
        return logits, nonl_loss(u_m, w, y_m, self.temperature)
