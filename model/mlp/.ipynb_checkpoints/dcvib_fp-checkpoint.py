"""DCVIB-FP（固定类条件先验）模型定义。"""

import math

import torch
import torch.nn as nn

from .utils import build_hidden_layers, flatten, kl_divergence


class DCVIB_FP(nn.Module):
    """带固定类条件先验（anchor prior）的 DCVIB 变体。

    主路与 DCVIB 相同：编码器输出 h 后直接由分类器得到 logits（确定性），
    z 不参与前向。旁路从 h 引出 mu/logvar 头得到后验 q(z|x)；先验不再是
    可学习的标签编码器，而是按类别索引的固定高斯分布表——K 个正交方向
    缩放 anchor_scale 作为各类均值，方差固定为 anchor_var——训练损失中
    加入 KL(q(z|x) || r(z|y))，梯度只通过共享编码器 h 回传作为正则。
    先验为 register_buffer、不参与梯度，因此 KL 无法靠移动先验来减小，
    只要 z 未对齐锚点，正则力就持续存在。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
        anchor_var: float = 1.0,
    ):
        super().__init__()
        assert num_classes <= z_dim, "固定锚点先验要求类别数不超过 z 维度"
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]

        # 主路：h 直接分类，无采样
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 旁路：h -> (mu, logvar)，参数化 z ~ N(mu, sigma^2)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

        # 固定类条件先验表：每类一个高斯分布，不参与梯度
        g = torch.randn(z_dim, num_classes)
        q, _ = torch.linalg.qr(g)  # q 为 (z_dim, num_classes) 正交列
        prior_mu = (q * anchor_scale).t().contiguous()  # (num_classes, z_dim)
        prior_logvar = torch.full((num_classes, z_dim), math.log(anchor_var))
        self.register_buffer("prior_mu", prior_mu)
        self.register_buffer("prior_logvar", prior_logvar)

        # 置零初始化：训练起点 sigma = 1，初始 KL ≈ 0.5 * anchor_scale^2
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        logits 仅由确定性主路产生；kl 来自旁路 z 与固定类条件先验的
        KL(q(z|x) || r(z|y))，只依赖 mu/logvar，无需采样。
        """
        h = self.encoder(flatten(x))
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        mu_p = self.prior_mu[labels]
        logvar_p = self.prior_logvar[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
