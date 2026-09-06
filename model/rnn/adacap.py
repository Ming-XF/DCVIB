"""AdaCap（Adaptive Contrastive Approach，ESANN 2026）RNN 模型。

与 MLP 版同构（见 model/mlp/adacap.py）：LSTM 文本编码器 → BN → Tikhonov
闭式读出 + 置换对比损失。仅回归任务（stsb 文本相似度回归）。
"""

import torch
import torch.nn as nn

from ..mlp.adacap import _AdaCapMixin
from .utils import build_rnn_encoder


class AdaCap(_AdaCapMixin, nn.Module):
    """LSTM 文本编码器 + AdaCap（Tikhonov 闭式输出 + 置换对比损失）。"""

    def __init__(
        self,
        vocab_size: int | None = None,
        num_classes: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
        lambda_init: float = 100.0,
        n_permuted: int = 10,
    ):
        super().__init__()
        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )
        self._setup_shared(hidden_dims[-1], lambda_init, n_permuted)

    def encode(self, x):
        """BN 后的隐藏表示（fit_adacap_readout 用，eval 模式）。"""
        return self.bn(self.encoder(x))

    def forward(self, x, labels=None, stochastic=True):
        """训练返回 (y_hat, ada_loss)；评估返回 (h @ beta, None)。"""
        h = self.bn(self.encoder(x))
        if not self.training:
            return h @ self.beta, None
        return self._train_step(h, labels)
