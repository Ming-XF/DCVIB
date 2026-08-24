"""FGIB（Fixed-Geometry Information Bottleneck，固定几何信息瓶颈）CNN 模型定义。"""

import torch.nn as nn

from ..mlp.utils import build_anchor_prior, kl_divergence
from .utils import build_cnn_encoder


class FGIB(nn.Module):
    """CNN 版 FGIB：DCVIB 的标签条件先验换成固定正交锚点先验。

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
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 256,
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
        anchor_var: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = build_cnn_encoder(
            input_channels, conv_channels, hidden_dim, dropout
        )

        # 主路：h 直接分类，无采样
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 旁路：h -> (mu, logvar)，参数化 z ~ N(mu, sigma^2)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

        # 固定类条件先验表：每类一个高斯分布，不参与梯度
        prior_mu, prior_logvar = build_anchor_prior(
            z_dim, num_classes, anchor_scale, anchor_var
        )
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
        h = self.encoder(x)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        mu_p = self.prior_mu[labels]
        logvar_p = self.prior_logvar[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
