"""DCVIB（确定性标签条件变分信息瓶颈）GNN 模型定义。"""

import torch.nn as nn
import torch.nn.functional as F

from .utils import build_gcn_encoder, kl_divergence_masked


class DCVIB(nn.Module):
    """在 GCN 旁路引入标签条件变分信息瓶颈（DCVIB）机制。

    主路是确定性分类路径：编码器输出 h 后直接由分类器得到 logits。
    旁路从 h 引出两个线性头，得到后验 q(z|x) 的均值和对数方差，
    z 不参与分类（不受分类损失约束）；同时用一个无激活的线性层把
    one-hot 标签映射为先验 r(z|y) 的均值和对数方差，训练损失中加入
    KL(q(z|x) || r(z|y))，KL 梯度回传到共享编码器 h 作为正则。

    图转导设置：KL 按 mask（训练时为 train_mask）只在训练节点上计算，
    标签条件先验只接触训练节点的标签（避免标签泄漏）。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dim: int = 16,
        z_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = build_gcn_encoder(input_dim, hidden_dim, dropout)

        # 主路：h 直接分类，无采样
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 旁路：h -> (mu, logvar)，参数化 z ~ N(mu, sigma^2)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        # 标签先验线性层：无激活，one-hot 标签 -> (mu_prior, logvar_prior)
        self.prior_net = nn.Linear(num_classes, 2 * z_dim)

        # 置零初始化：训练开始时 sigma = sigma_p = 1、mu_p = 0，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)
        nn.init.zeros_(self.prior_net.weight)
        nn.init.zeros_(self.prior_net.bias)

    def forward(self, x, labels=None, adj_norm=None, mask=None):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        logits 仅由确定性主路产生；kl 来自旁路 z 与标签先验的
        KL(q(z|x) || r(z|y))，只依赖 mu/logvar，无需采样。
        """
        h = self.encoder(x, adj_norm)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        y_feat = F.one_hot(labels, num_classes=self.num_classes).float()
        mu_p, logvar_p = self.prior_net(y_feat).chunk(2, dim=1)

        kl = kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask)
        return logits, kl
