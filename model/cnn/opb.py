"""OPB（Orthogonal-Prior Bottleneck，正交先验信息瓶颈）CNN 模型定义。

分类方案见 paper/OPB.txt、回归方案（OPB-R）见 paper/OPB-R.txt；两类任务
共用同一模型类，先验构造按 continuous_y 分叉（与 MLP 版同构）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import kl_divergence, qr_anchor_table, reparameterize
from .utils import build_cnn_encoder


class OPB(nn.Module):
    """CNN 版 OPB：CEB 同构的随机主路瓶颈 + 正交（分类）/等距（回归）先验。

    与 MLP 版 OPB 相同的瓶颈与先验构造，仅编码器换成卷积特征提取器。
    分类：全类别 one-hot → prior_net → 均值矩阵 QR 正交帧 a·Q_p + 逐类
    可学习 logvar（要求 num_classes <= z_dim）；回归：等距轴先验
    mu_p(y) = rho·y_tilde·normalize(W) + y_tilde 经 nn.Linear 的可学习
    logvar，回归头复用 classifier（num_classes=1）。初始化的非零例外与
    MLP 版一致（prior_net 均值块恒等 / prior_direction = e_1，两者都不能
    置零——QR 反向与归一化都需要满秩/非零）。
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
        continuous_y: bool = False,
        input_size: int = 28,
    ):
        super().__init__()
        assert num_classes <= z_dim, "OPB 正交先验要求类别数不超过 z 维度"
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.continuous_y = continuous_y

        self.encoder = build_cnn_encoder(
            input_channels, conv_channels, hidden_dim, dropout, input_size=input_size
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        # 分类器（回归时 num_classes=1，作普通回归头）
        self.classifier = nn.Linear(z_dim, num_classes)

        if continuous_y:
            self.prior_direction = nn.Linear(1, z_dim, bias=False)
            self.prior_logvar_net = nn.Linear(1, z_dim)
        else:
            self.prior_net = nn.Linear(num_classes, 2 * z_dim)
            self.register_buffer("class_eye", torch.eye(num_classes))

        # 初始化约定同 MLP 版 OPB：logvar 头置零；分类先验编码器均值块恒等
        # 初始化（QR 需要满列秩，不能置零）、方差块置零；回归 prior_direction
        # 置为 e_1、logvar 网络置零
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        if continuous_y:
            with torch.no_grad():
                self.prior_direction.weight.zero_()
                self.prior_direction.weight[0, 0] = 1.0
                self.prior_logvar_net.weight.zero_()
                self.prior_logvar_net.bias.zero_()
        else:
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

        if self.continuous_y:
            y_feat = labels.float().unsqueeze(-1)
            u = F.normalize(self.prior_direction.weight.squeeze(-1), dim=0)
            mu_p = self.anchor_scale * y_feat * u
            logvar_p = self.prior_logvar_net(y_feat)
        else:
            prior_out = self.prior_net(self.class_eye)
            prior_mu_raw, prior_logvar_table = prior_out.chunk(2, dim=1)
            prior_mu = qr_anchor_table(prior_mu_raw.t(), self.anchor_scale)
            mu_p = prior_mu[labels]
            logvar_p = prior_logvar_table[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
