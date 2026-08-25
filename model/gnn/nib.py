"""NIB（非线性信息瓶颈）GNN 模型定义。"""

import math

import torch
import torch.nn as nn

from ..mlp.utils import nib_mi_upper_bound
from .utils import build_gcn_encoder


class NIB(nn.Module):
    """非线性信息瓶颈（NIB）GNN 版，结构与 MLP 版一一对应（见 model/mlp/nib.py）。

    编码器换成 GCN；瓶颈表示 f(x) 注入同方差高斯噪声后，压缩项用成对距离
    非参数上界 Î 估计 I(X;M)，损失为 CE + β·Î。图转导设置下 Î 只在 mask
    （训练时为 train_mask）节点上计算，与 VIB 系 masked KL 的约定一致。
    σ² 为可学习参数（log 空间，初始为论文值 1.0）。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        noise_var_init: float = 1.0,
    ):
        super().__init__()
        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self.bottleneck_head = nn.Linear(hidden_dims[-1], z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        # σ² 可学习（log 空间，初始为论文值 1.0）
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(noise_var_init)))

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None):
        """返回 (logits, î)，î 在 mask 节点上计算（mask=None 时为全图）。"""
        h = self.encoder(x, adj_norm)
        m = self.bottleneck_head(h)
        noise_var = self.log_noise_var.clamp(-10.0, 10.0).exp()
        if stochastic:
            m = m + torch.randn_like(m).mul_(noise_var.sqrt())
        m_masked = m[mask] if mask is not None else m
        return self.classifier(m), nib_mi_upper_bound(m_masked, noise_var)
