"""FGIB（Fixed-Geometry Information Bottleneck，固定几何信息瓶颈）GNN 模型定义。"""

import torch.nn as nn

from ..mlp.utils import ContinuousAnchorPrior, build_anchor_prior
from .utils import build_gcn_encoder, graph_readout, kl_divergence_masked


class FGIB(nn.Module):
    """GNN 版 FGIB：确定性主路 + 固定锚点先验的旁路瓶颈。

    主路为确定性：编码器输出 h 后直接由分类器得到 logits，z 不参与前向。
    旁路从 h 引出 mu/logvar 头得到后验 q(z|x)；先验不是可学习的标签编码器，
    而是固定的高斯分布（分类：K 个正交方向缩放 anchor_scale 作为各类均值，
    按类别索引的 buffer 表；回归 continuous_y：固定随机傅里叶特征
    ContinuousAnchorPrior，锚点近似落在半径 anchor_scale 的球面上，条件在
    缩放后的连续 y 上）——训练损失中加入 KL(q(z|x) || r(z|y))，梯度只通过
    共享编码器 h 回传作为正则。先验为 register_buffer / 零参数模块、不参与
    梯度，因此 KL 无法靠移动先验来减小，只要 z 未对齐锚点，正则力就持续
    存在。batch 非 None 时（图级任务）编码器后先做 mean 图读出。

    图转导设置：KL 按 mask（训练时为 train_mask）只在训练节点上计算，
    与 GNN 版 CEB 的约定一致（避免评估时混入非训练节点）。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
        anchor_var: float = 1.0,
        continuous_y: bool = False,
        pooling: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.continuous_y = continuous_y
        self.pooling = pooling

        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)

        # 主路：h 直接分类，无采样
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        # 旁路：h -> (mu, logvar)，参数化 z ~ N(mu, sigma^2)
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)

        # 固定先验：分类为类条件锚点表，回归为固定 RFF 连续锚点，均不参与梯度
        if continuous_y:
            self.anchor_prior = ContinuousAnchorPrior(z_dim, anchor_scale, anchor_var)
        else:
            prior_mu, prior_logvar = build_anchor_prior(
                z_dim, num_classes, anchor_scale, anchor_var
            )
            self.register_buffer("prior_mu", prior_mu)
            self.register_buffer("prior_logvar", prior_logvar)

        # 置零初始化：训练起点 sigma = 1，初始 KL ≈ 0.5 * anchor_scale^2
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None, adj_norm=None, mask=None, batch=None):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        logits 仅由确定性主路产生；kl 来自旁路 z 与固定先验的
        KL(q(z|x) || r(z|y))，按 mask 只在指定节点上平均。
        """
        h = self.encoder(x, adj_norm)
        if batch is not None:
            h = graph_readout(h, batch, self.pooling)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if self.continuous_y:
            mu_p, logvar_p = self.anchor_prior(labels.float().unsqueeze(-1))
        else:
            mu_p = self.prior_mu[labels]
            logvar_p = self.prior_logvar[labels]

        kl = kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask)
        return logits, kl
