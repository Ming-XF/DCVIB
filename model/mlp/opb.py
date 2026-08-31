"""OPB（Orthogonal-Prior Bottleneck，正交先验信息瓶颈）MLP 模型定义。

分类方案见 paper/OPB.txt：在 CEB 的条件高斯先验上增加显式的类别几何结构；
回归方案（OPB-R）见 paper/OPB-R.txt：用一条等距先验轴表达连续标签的顺序与
数值距离。两类任务共用同一模型类，先验构造按 continuous_y 分叉。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    build_hidden_layers,
    flatten,
    kl_divergence,
    qr_anchor_table,
    reparameterize,
)


class OPB(nn.Module):
    """在 MLP 中间表示上引入正交/等距先验信息瓶颈（OPB / OPB-R）机制。

    架构与 CEB 同构：编码器输出 h 后，由两个线性头得到后验 q(z|x) 的
    均值和对数方差，重参数化采样得到 z，分类器/回归头输出结果。

    分类（continuous_y=False，paper/OPB.txt）：每个前向把全部 K 个类别的
    one-hot 矩阵送入先验编码器 prior_net 得到全类别均值矩阵 M_p（K×d）与
    逐类对数方差表，对 M_p 转置后做 QR 分解得正交帧 Q_p（符号规范化，
    R 对角取正），各类先验均值取 a·Q_p[:, k]（a 为锚点尺度）；先验方差
    使用 prior_net 输出的逐类可学习 logvar。要求 num_classes <= z_dim。

    回归（continuous_y=True，OPB-R，paper/OPB-R.txt）：先验均值为等距轴
    mu_p(y) = rho · y_tilde · u，其中 y_tilde 为训练管道已归一化的连续
    标签（MinMax 缩放，模型内不二次标准化），u = W/||W|| 为可训练方向
    W∈R^{d×1}（无激活、无偏置）每前向即时归一化的单位方向——任意两个
    标签的先验距离严格等于 rho·|Δy_tilde|；先验方差用 nn.Linear 从
    y_tilde 映射的可学习 logvar（不固定 tau²）。回归头复用现有
    classifier（num_classes=1）。

    两种模式损失均为 CE/MSE + β·KL(q(z|x) || r(z|y))，只有一个 KL——
    先验的构造方式不是第二个独立随机变量。无状态：无 EMA prototype、
    不依赖 batch 组成。初始化例外：分类 prior_net 均值块恒等初始化
    （QR 反向需满列秩，整个网络不能置零）、回归 prior_direction 置为
    e_1（非零、‖W‖=1 可归一化），方差网络与 logvar 头仍置零
    （起点 sigma = sigma_p = 1，KL ≈ 0.5·anchor_scale²）。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
        continuous_y: bool = False,
    ):
        super().__init__()
        assert num_classes <= z_dim, "OPB 正交先验要求类别数不超过 z 维度"
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.continuous_y = continuous_y

        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        # 分类器（回归时 num_classes=1，作普通回归头）
        self.classifier = nn.Linear(z_dim, num_classes)

        if continuous_y:
            # 回归：等距轴先验（W ∈ R^{d×1}，每前向归一化为单位方向）
            self.prior_direction = nn.Linear(1, z_dim, bias=False)
            # y_tilde -> 可学习 logvar（不固定 tau²）
            self.prior_logvar_net = nn.Linear(1, z_dim)
        else:
            # 分类：全类别 one-hot -> (mu_prior, logvar_prior)，无激活
            self.prior_net = nn.Linear(num_classes, 2 * z_dim)
            # 全类别 one-hot 矩阵（无状态，每前向整体送入先验编码器）
            self.register_buffer("class_eye", torch.eye(num_classes))

        # 后验 logvar 头置零初始化（sigma = 1，仓库约定）。
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        if continuous_y:
            # W 置为 e_1（非零、起点 u 为第一坐标轴）；logvar 网络置零
            # （起点 sigma_p = 1），训练起点 KL ≈ 0.5 * anchor_scale^2
            with torch.no_grad():
                self.prior_direction.weight.zero_()
                self.prior_direction.weight[0, 0] = 1.0
                self.prior_logvar_net.weight.zero_()
                self.prior_logvar_net.bias.zero_()
        else:
            # 先验编码器：均值块恒等初始化（起点 Q_p = [e_1..e_K] 满列秩、
            # QR 反向稳定，训练起点 KL ≈ 0.5 * anchor_scale^2），方差块置零
            # （起点 sigma_p = 1）。注意整个 prior_net 不能置零初始化：零矩阵
            # 秩亏，QR 反向会数值发散——这是对 CEB/FGIB 置零约定的有意例外。
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

        if self.continuous_y:
            # 回归：等距轴先验（复用训练管道已 MinMax 归一化的连续标签）
            y_feat = labels.float().unsqueeze(-1)  # (B, 1)
            u = F.normalize(self.prior_direction.weight.squeeze(-1), dim=0)  # (d,)
            mu_p = self.anchor_scale * y_feat * u  # (B, d)
            logvar_p = self.prior_logvar_net(y_feat)
        else:
            # 分类：全类别先验表：QR 只作用于均值，logvar 表直接按类别索引
            prior_out = self.prior_net(self.class_eye)  # (K, 2*z_dim)
            prior_mu_raw, prior_logvar_table = prior_out.chunk(2, dim=1)
            prior_mu = qr_anchor_table(prior_mu_raw.t(), self.anchor_scale)  # (K, z_dim)
            mu_p = prior_mu[labels]
            logvar_p = prior_logvar_table[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
