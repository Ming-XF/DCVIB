"""OPB（Orthogonal-Prior Bottleneck，正交先验信息瓶颈）RNN 模型定义。

分类方案见 paper/OPB.txt、回归方案（OPB-R）见 paper/OPB-R.txt；两类任务
共用同一模型类，先验构造按 continuous_y 分叉（与 MLP 版同构）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import kl_divergence, qr_anchor_table, reparameterize
from .utils import build_rnn_encoder


class OPB(nn.Module):
    """RNN 版 OPB：CEB 同构的随机主路瓶颈 + 正交（分类）/等距（回归）先验。

    与 MLP 版 OPB 相同的瓶颈与先验构造，仅编码器换成 LSTM 文本编码器。
    分类（imdb/agnews）：全类别 one-hot → prior_net → 均值矩阵 QR 正交帧
    a·Q_p + 逐类可学习 logvar（要求 num_classes <= z_dim）；回归（stsb，
    continuous_y=True）：等距轴先验 mu_p(y) = rho·y_tilde·normalize(W) +
    y_tilde 经 nn.Linear 的可学习 logvar，回归头复用 classifier
    （num_classes=1）。初始化的非零例外与 MLP 版一致（prior_net 均值块
    恒等 / prior_direction = e_1，两者都不能置零——QR 反向与归一化都需要
    满秩/非零）。

    消融（均与 MLP 版同构）：energy_classifier=True（仅分类）把分类器改为
    锚点能量分类器 logit_k = −‖z − a·Q_p[:,k]‖²/(2τ²)；tied_head=True
    （仅回归）把回归头改为 tied 投影头 y_hat_tilde = uᵀz/rho。两种消融下
    self.classifier 均保留为死参数。
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
        continuous_y: bool = False,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
        energy_classifier: bool = False,
        tied_head: bool = False,
    ):
        super().__init__()
        assert num_classes <= z_dim, "OPB 正交先验要求类别数不超过 z 维度"
        if energy_classifier and continuous_y:
            raise ValueError(
                "能量分类器是分类方案（paper/OPB.txt §12）的消融，回归"
                "（continuous_y）无类别锚点表，不支持 energy_classifier"
            )
        if tied_head and not continuous_y:
            raise ValueError(
                "tied 投影头是回归方案（paper/OPB-R.txt）的消融，分类"
                "（continuous_y=False）无等距轴，不支持 tied_head"
            )
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.continuous_y = continuous_y
        self.energy_classifier = energy_classifier
        self.tied_head = tied_head

        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
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

    def _prior_table(self):
        """分类分支的全类别先验表：QR 正交锚点表 (K, d) + 逐类 logvar 表 (K, d)。

        QR 只作用于均值（符号规范化），logvar 表直接按类别索引。能量分类器
        模式下评估时（labels=None）也需调用本方法构造锚点。
        """
        prior_out = self.prior_net(self.class_eye)  # (K, 2*z_dim)
        prior_mu_raw, prior_logvar_table = prior_out.chunk(2, dim=1)
        prior_mu = qr_anchor_table(prior_mu_raw.t(), self.anchor_scale)  # (K, z_dim)
        return prior_mu, prior_logvar_table

    def _energy_logits(self, z, prior_mu, prior_logvar_table):
        """锚点能量分类器：logit_k = −‖z − a·Q_p[:,k]‖² / (2·τ²)（paper/OPB.txt §12 消融）。

        τ² 取 prior_net 方差块的逐类可学习 exp(logvar)——与 KL 共用同一套锚点
        与方差、零新参数；分类器无法绕过正交几何。
        """
        tau2 = prior_logvar_table.exp()  # (K, d)
        diff2 = (z.unsqueeze(1) - prior_mu.unsqueeze(0)).pow(2)  # (B, K, d)
        return -(diff2 / (2.0 * tau2)).sum(dim=2)  # (B, K)

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        energy_classifier=True 时 logits 为锚点能量分类器（评估时也构造先验表）；
        tied_head=True（回归）时 logits 为 tied 投影头 y_hat_tilde = uᵀz/rho
        （评估时同样计算 u）。
        """
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)

        if self.continuous_y:
            u = F.normalize(self.prior_direction.weight.squeeze(-1), dim=0)
            if self.tied_head:
                logits = ((z @ u) / self.anchor_scale).unsqueeze(-1)
            else:
                logits = self.classifier(z)
            if labels is None:
                return logits, None
            y_feat = labels.float().unsqueeze(-1)
            mu_p = self.anchor_scale * y_feat * u
            logvar_p = self.prior_logvar_net(y_feat)
        elif self.energy_classifier:
            prior_mu, prior_logvar_table = self._prior_table()
            logits = self._energy_logits(z, prior_mu, prior_logvar_table)
            if labels is None:
                return logits, None
            mu_p = prior_mu[labels]
            logvar_p = prior_logvar_table[labels]
        else:
            logits = self.classifier(z)
            if labels is None:
                return logits, None
            prior_mu, prior_logvar_table = self._prior_table()
            mu_p = prior_mu[labels]
            logvar_p = prior_logvar_table[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
