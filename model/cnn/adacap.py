"""AdaCap（Adaptive Contrastive Approach，ESANN 2026）CNN 模型。

与 MLP 版同构（见 model/mlp/adacap.py）：卷积编码器 → BN → Tikhonov 闭式
读出 + 置换对比损失。仅回归任务（agedb 人脸年龄回归）。
"""

import torch
import torch.nn as nn

from ..mlp.adacap import _AdaCapMixin
from .utils import build_cnn_encoder


class AdaCap(_AdaCapMixin, nn.Module):
    """CNN 编码器 + AdaCap（Tikhonov 闭式输出 + 置换对比损失）。"""

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.2,
        input_size: int = 28,
        lambda_init: float = 100.0,
        n_permuted: int = 10,
    ):
        super().__init__()
        self.encoder = build_cnn_encoder(
            input_channels, conv_channels, hidden_dim, dropout, input_size=input_size
        )
        self._setup_shared(hidden_dim, lambda_init, n_permuted)

    def encode(self, x):
        """BN 后的隐藏表示（fit_adacap_readout 用，eval 模式）。"""
        return self.bn(self.encoder(x))

    def forward(self, x, labels=None, stochastic=True):
        """训练返回 (y_hat, ada_loss)；评估返回 (h @ beta, None)。"""
        h = self.bn(self.encoder(x))
        if not self.training:
            return h @ self.beta, None
        return self._train_step(h, labels)
