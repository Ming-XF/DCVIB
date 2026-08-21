"""CVIB（条件变分信息瓶颈）模型定义。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_hidden_layers


class CVIB(nn.Module):
    """在 MLP 中间表示上引入条件变分信息瓶颈（CVIB）机制。

    编码器输出 h 后，由两个线性头得到后验 q(z|x) 的均值和对数方差，
    重参数化采样得到 z，分类器输出 logits；同时用一个无激活的线性层
    把 one-hot 标签映射为先验 r(z|y) 的均值和对数方差，训练损失中加入
    KL(q(z|x) || r(z|y))。
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
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        # 标签先验线性层：无激活，one-hot 标签 -> (mu_prior, logvar_prior)
        self.prior_net = nn.Linear(num_classes, 2 * z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)

        # 置零初始化：训练开始时 sigma = sigma_p = 1、mu_p = 0，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.zeros_(self.prior_net.weight)
        nn.init.zeros_(self.prior_net.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool):
        if stochastic:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    @staticmethod
    def kl_divergence(mu, logvar, mu_p, logvar_p):
        # clamp 防止方差过小导致 KL 数值发散
        logvar = logvar.clamp(-10.0, 10.0)
        logvar_p = logvar_p.clamp(-10.0, 10.0)
        var, var_p = logvar.exp(), logvar_p.exp()
        kl = 0.5 * (logvar_p - logvar + (var + (mu - mu_p).pow(2)) / var_p - 1.0)
        return kl.sum(dim=1).mean()

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        stochastic=True 时 z 由重参数化采样得到（训练）；
        stochastic=False 时 z = mu（确定性评估）。
        """
        if x.dim() == 4 or x.dim() == 3:
            x = x.flatten(1)
        h = self.encoder(x)

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)

        if labels is None:
            return logits, None

        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        mu_p, logvar_p = self.prior_net(y_onehot).chunk(2, dim=1)

        kl = self.kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
