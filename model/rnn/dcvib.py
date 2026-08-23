"""DCVIB（确定性标签条件变分信息瓶颈）RNN 模型定义。"""

import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import kl_divergence
from .utils import build_rnn_encoder


class DCVIB(nn.Module):
    """在 RNN 旁路引入标签条件变分信息瓶颈（DCVIB）机制。

    主路是确定性分类路径：编码器输出 h 后直接由分类器得到 logits。
    旁路从 h 引出两个线性头，得到后验 q(z|x) 的均值和对数方差，
    z 不参与分类（不受分类损失约束）；同时用一个无激活的线性层把
    one-hot 标签映射为先验 r(z|y) 的均值和对数方差，训练损失中加入
    KL(q(z|x) || r(z|y))，KL 梯度回传到共享编码器 h 作为正则。
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

        # 主路：h 直接分类，无采样
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        # 旁路：h -> (mu, logvar)，参数化 z ~ N(mu, sigma^2)
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
        # 标签先验线性层：无激活，one-hot 标签 -> (mu_prior, logvar_prior)
        self.prior_net = nn.Linear(num_classes, 2 * z_dim)

        # 置零初始化：训练开始时 sigma = sigma_p = 1、mu_p = 0，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.zeros_(self.prior_net.weight)
        nn.init.zeros_(self.prior_net.bias)

    def forward(self, x, labels=None):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        logits 仅由确定性主路产生；kl 来自旁路 z 与标签先验的
        KL(q(z|x) || r(z|y))，只依赖 mu/logvar，无需采样。
        """
        h = self.encoder(x)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        y_feat = F.one_hot(labels, num_classes=self.num_classes).float()
        mu_p, logvar_p = self.prior_net(y_feat).chunk(2, dim=1)

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
