"""OPB（Orthogonal-Prior Bottleneck，正交先验信息瓶颈）MLP 模型定义。

方案见 paper/OPB.txt：在 CEB 的条件高斯先验上增加显式的类别几何结构。
"""

import torch
import torch.nn as nn

from .utils import (
    build_hidden_layers,
    flatten,
    kl_divergence,
    qr_anchor_table,
    reparameterize,
)


class OPB(nn.Module):
    """在 MLP 中间表示上引入正交先验信息瓶颈（OPB）机制。

    架构与 CEB 同构：编码器输出 h 后，由两个线性头得到后验 q(z|x) 的
    均值和对数方差，重参数化采样得到 z，分类器输出 logits。区别在先验：
    每个前向把全部 K 个类别的 one-hot 矩阵送入先验编码器 prior_net 得到
    全类别均值矩阵 M_p（K×d）与逐类对数方差表，对 M_p 转置后做 QR 分解
    得正交帧 Q_p（符号规范化，R 对角取正），各类先验均值取 a·Q_p[:, k]
    （a 为锚点尺度）；先验方差使用 prior_net 输出的逐类可学习 logvar。
    损失为 CE + β·KL(q(z|x) || r(z|y))，仍只有一个 KL——Q_p 是先验的
    构造方式，不是第二个独立随机变量。

    无状态：不维护 EMA prototype，也不依赖 batch 中出现哪些类别；Q_p 是
    当前先验编码器参数的即时函数。要求 num_classes <= z_dim（K 个正交
    类别方向），仅支持分类任务（OPB v1 定义）。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
    ):
        super().__init__()
        assert num_classes <= z_dim, "OPB 正交先验要求类别数不超过 z 维度"
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale

        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        # 先验编码器：全类别 one-hot -> (mu_prior, logvar_prior)，无激活
        self.prior_net = nn.Linear(num_classes, 2 * z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)

        # 全类别 one-hot 矩阵（无状态，每前向整体送入先验编码器）
        self.register_buffer("class_eye", torch.eye(num_classes))

        # 后验 logvar 头置零初始化（sigma = 1，仓库约定）。
        # 先验编码器：均值块恒等初始化（起点 Q_p = [e_1..e_K] 满列秩、
        # QR 反向稳定，训练起点 KL ≈ 0.5 * anchor_scale^2），方差块置零
        # （起点 sigma_p = 1）。注意整个 prior_net 不能置零初始化：零矩阵
        # 秩亏，QR 反向会数值发散——这是对 CEB/FGIB 置零约定的有意例外。
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        with torch.no_grad():
            self.prior_net.weight.zero_()
            self.prior_net.weight[:num_classes, :num_classes] = torch.eye(num_classes)
            self.prior_net.bias.zero_()

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        stochastic=True 时 z 由重参数化采样得到（训练）；
        stochastic=False 时 z = mu（确定性评估）。
        """
        h = self.encoder(flatten(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)

        if labels is None:
            return logits, None

        # 全类别先验表：QR 只作用于均值，logvar 表直接按类别索引
        prior_out = self.prior_net(self.class_eye)  # (K, 2*z_dim)
        prior_mu_raw, prior_logvar_table = prior_out.chunk(2, dim=1)
        prior_mu = qr_anchor_table(prior_mu_raw.t(), self.anchor_scale)  # (K, z_dim)
        mu_p = prior_mu[labels]
        logvar_p = prior_logvar_table[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
