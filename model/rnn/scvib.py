"""SCVIB（随机标签条件变分信息瓶颈）RNN 模型定义。"""

import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import kl_divergence, reparameterize
from .utils import build_rnn_encoder


class SCVIB(nn.Module):
    """在 RNN 文本表示上引入随机标签条件变分信息瓶颈（SCVIB）机制。

    编码器输出 h 后，由两个线性头得到后验 q(z|x) 的均值和对数方差，
    重参数化采样得到 z，分类器输出 logits；同时用一个无激活的线性层
    把 one-hot 标签映射为先验 r(z|y) 的均值和对数方差，训练损失中加入
    KL(q(z|x) || r(z|y))。
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = build_rnn_encoder(vocab_size, hidden_dims, dropout, pad_idx)
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
        # 标签先验线性层：无激活，one-hot 标签 -> (mu_prior, logvar_prior)
        self.prior_net = nn.Linear(num_classes, 2 * z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)

        # 置零初始化：训练开始时 sigma = sigma_p = 1、mu_p = 0，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.zeros_(self.prior_net.weight)
        nn.init.zeros_(self.prior_net.bias)

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        stochastic=True 时 z 由重参数化采样得到（训练）；
        stochastic=False 时 z = mu（确定性评估）。
        """
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)

        if labels is None:
            return logits, None

        y_feat = F.one_hot(labels, num_classes=self.num_classes).float()
        mu_p, logvar_p = self.prior_net(y_feat).chunk(2, dim=1)

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
