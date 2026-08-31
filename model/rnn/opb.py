"""OPB（Orthogonal-Prior Bottleneck，正交先验信息瓶颈）RNN 模型定义。"""

import torch
import torch.nn as nn

from ..mlp.utils import kl_divergence, qr_anchor_table, reparameterize
from .utils import build_rnn_encoder


class OPB(nn.Module):
    """RNN 版 OPB：CEB 同构的随机主路瓶颈 + 每前向 QR 正交化的类别先验。

    与 MLP 版 OPB 相同的瓶颈与先验构造（全类别 one-hot → prior_net →
    均值矩阵 QR 正交帧 a·Q_p + 逐类可学习 logvar，单 KL），仅编码器换成
    LSTM 文本编码器。要求 num_classes <= z_dim，仅支持分类任务（stsb
    回归不在 OPB v1 范围）。
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
        anchor_scale: float = 4.0,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
    ):
        super().__init__()
        assert num_classes <= z_dim, "OPB 正交先验要求类别数不超过 z 维度"
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale

        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
        self.prior_net = nn.Linear(num_classes, 2 * z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)

        self.register_buffer("class_eye", torch.eye(num_classes))

        # 初始化约定同 MLP 版 OPB：logvar 头置零；先验编码器均值块恒等
        # 初始化（QR 需要满列秩，不能置零）、方差块置零
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        with torch.no_grad():
            self.prior_net.weight.zero_()
            self.prior_net.weight[:num_classes, :num_classes] = torch.eye(num_classes)
            self.prior_net.bias.zero_()

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。"""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)

        if labels is None:
            return logits, None

        prior_out = self.prior_net(self.class_eye)
        prior_mu_raw, prior_logvar_table = prior_out.chunk(2, dim=1)
        prior_mu = qr_anchor_table(prior_mu_raw.t(), self.anchor_scale)
        mu_p = prior_mu[labels]
        logvar_p = prior_logvar_table[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
