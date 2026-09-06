"""AdaCap（Adaptive Contrastive Approach，ESANN 2026）GNN 模型。

与 MLP 版同构（见 model/mlp/adacap.py）：GCN 编码器（图级任务先 mean
图读出）→ BN → Tikhonov 闭式读出 + 置换对比损失。仅回归任务（zinc 图级
回归，batch 非 None 时按图读出、置换对比在图级标签 (B,1) 上）。
"""

import torch
import torch.nn as nn

from ..mlp.adacap import _AdaCapMixin
from .utils import build_gcn_encoder, graph_readout


class AdaCap(_AdaCapMixin, nn.Module):
    """GCN 编码器 + AdaCap（Tikhonov 闭式输出 + 置换对比损失）。"""

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pooling: str = "mean",
        lambda_init: float = 100.0,
        n_permuted: int = 10,
    ):
        super().__init__()
        self.pooling = pooling
        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self._setup_shared(hidden_dims[-1], lambda_init, n_permuted)

    def encode(self, x, adj_norm=None, batch=None):
        """BN 后的图级表示（fit_adacap_readout 用，eval 模式）。"""
        h = self.encoder(x, adj_norm)
        if batch is not None:
            h = graph_readout(h, batch, self.pooling)
        return self.bn(h)

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None, batch=None):
        """训练返回 (y_hat, ada_loss)；评估返回 (h @ beta, None)。"""
        h = self.encode(x, adj_norm, batch)
        if not self.training:
            return h @ self.beta, None
        return self._train_step(h, labels)
