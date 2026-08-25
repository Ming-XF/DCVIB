"""DVCCA（监督 β-DVCCA）CNN 模型定义。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import kl_divergence, reparameterize
from .utils import build_cnn_encoder


class DVCCA(nn.Module):
    """监督 β-DVCCA 的 CNN 版（见 model/mlp/dvcca.py 的推导说明）。

    VIB 结构 + 重建解码器；解码器把 z 映射回 784 维后 reshape 成原图形状
    计算 MSE。forward 返回 (logits, kl, recon_loss)。
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 256,
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        input_size: int = 28,
    ):
        super().__init__()
        self.encoder = build_cnn_encoder(
            input_channels, conv_channels, hidden_dim, dropout
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_channels * input_size * input_size),
        )
        self.input_channels = input_channels
        self.input_size = input_size

        # 置零初始化：训练开始时 sigma = 1，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl, recon_loss)；先验为 N(0, I)。labels 参数仅为统一接口。"""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)
        recon_flat = self.decoder(z)
        recon = F.mse_loss(
            recon_flat, x.view(x.size(0), -1)
        )

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl, recon
