"""OPB（Orthogonal-Prior Bottleneck，正交先验信息瓶颈）GNN 模型定义。"""

import torch
import torch.nn as nn

from ..mlp.utils import qr_anchor_table, reparameterize
from .utils import build_gcn_encoder, graph_readout, kl_divergence_masked


class OPB(nn.Module):
    """GNN 版 OPB：CEB 同构的随机主路瓶颈 + 每前向 QR 正交化的类别先验。

    与 MLP 版 OPB 相同的瓶颈与先验构造（全类别 one-hot → prior_net →
    均值矩阵 QR 正交帧 a·Q_p + 逐类可学习 logvar，单 KL），仅编码器换成
    GCN。KL 按 mask（训练时为 train_mask）只在训练节点上计算，与 GNN 版
    CEB 的约定一致（标签条件先验只接触训练节点标签、避免泄漏）；先验帧
    构造只用全类别 one-hot 矩阵、不接触真实标签。batch 非 None 时（图级
    任务）编码器后先做 mean 图读出。要求 num_classes <= z_dim，仅支持
    分类任务（zinc 图级回归不在 OPB v1 范围）。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
        pooling: str = "mean",
    ):
        super().__init__()
        assert num_classes <= z_dim, "OPB 正交先验要求类别数不超过 z 维度"
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.pooling = pooling

        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
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

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None, batch=None):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        batch 非 None 时（图级任务）编码器后先做 mean 图读出；
        KL 按 mask 只在指定节点上平均。
        """
        h = self.encoder(x, adj_norm)
        if batch is not None:
            h = graph_readout(h, batch, self.pooling)
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

        kl = kl_divergence_masked(mu, logvar, mu_p, logvar_p, mask)
        return logits, kl
