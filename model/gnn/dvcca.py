"""DVCCA（监督 β-DVCCA）GNN 模型定义。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import kl_divergence, reparameterize
from .utils import build_gcn_encoder, kl_divergence_masked


class DVCCA(nn.Module):
    """监督 β-DVCCA 的 GNN 版（见 model/mlp/dvcca.py 的推导说明）。

    VIB 结构 + 节点特征重建解码器（类似 GAE 的重建项）；KL 按 mask 限制在
    训练节点上（与 VIB 系 masked KL 约定一致），重建项在全部节点上计算
    （转导设置下节点特征不含标签、无泄漏）。forward 返回
    (logits, kl, recon_loss)。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], input_dim),
        )

        # 置零初始化：训练开始时 sigma = 1，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None):
        """返回 (logits, kl, recon_loss)；先验为 N(0, I)。labels 参数仅为统一接口。"""
        h = self.encoder(x, adj_norm)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)
        recon = F.mse_loss(self.decoder(z), x)

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)
        kl = kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask)
        return logits, kl, recon
