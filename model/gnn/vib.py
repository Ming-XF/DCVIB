"""VIB（变分信息瓶颈）GNN 模型定义。"""

import torch
import torch.nn as nn

from ..mlp.utils import reparameterize, CosineClassifier
from .utils import build_gcn_encoder, graph_readout, kl_divergence_masked


class VIB(nn.Module):
    """在 GCN 节点特征上引入变分信息瓶颈（VIB）机制。

    编码器输出 h 后，由两个线性头得到后验 q(z|x) 的均值和对数方差，
    重参数化采样得到 z，分类器输出 logits；先验为固定的标准正态分布
    N(0, I)，训练损失中加入 KL(q(z|x) || N(0, I))。

    图转导设置：KL 按 mask（训练时为 train_mask）只在训练节点上计算。
    batch 非 None 时（图级任务）编码器后先做 mean 图读出，z 与 logits
    均按图输出，KL 按图平均（mask=None）。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean",
        cosine_classifier: bool = False,
    ):
        super().__init__()
        self.pooling = pooling

        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
        # --cosine-classifier 时用固定温度 cosine 分类器（理论验证实验）
        if cosine_classifier:
            self.classifier = CosineClassifier(z_dim, num_classes)
        else:
            self.classifier = nn.Linear(z_dim, num_classes)

        # 置零初始化：训练开始时 sigma = 1，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None, batch=None):
        """返回 (logits, kl)，先验为 N(0, I)。labels 参数仅为统一接口。"""
        h = self.encoder(x, adj_norm)
        if batch is not None:
            h = graph_readout(h, batch, self.pooling)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)
        kl = kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask)
        return logits, kl
