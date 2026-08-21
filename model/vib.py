"""VIB（变分信息瓶颈）模型定义。"""

import torch
import torch.nn as nn

from .utils import build_hidden_layers, flatten, kl_divergence, reparameterize


class VIB(nn.Module):
    """在 MLP 中间表示上引入变分信息瓶颈（VIB）机制。

    编码器输出 h 后，由两个线性头得到后验 q(z|x) 的均值和对数方差，
    重参数化采样得到 z，分类器输出 logits；先验为固定的标准正态分布
    N(0, I)，训练损失中加入 KL(q(z|x) || N(0, I))。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)

        # 置零初始化：训练开始时 sigma = 1，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)，先验为 N(0, I)。labels 参数仅为统一接口。"""
        h = self.encoder(flatten(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
