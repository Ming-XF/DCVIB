"""NCBD-NONL（Neural Collapse by Design，ICML 2026）RNN 模型。

与 MLP 版同构（见 model/mlp/ncbd.py），仅编码器换成 LSTM 文本编码器
（token 模式或 BERT 逐 token 特征模式，由 input_dim 决定）。仅分类任务
（imdb 二分类 / agnews 四分类）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import nonl_loss
from .utils import build_rnn_encoder


class NCBD(nn.Module):
    """LSTM 文本编码器 + 单位球原型分类器 + NONL 损失。"""

    def __init__(
        self,
        vocab_size: int | None = None,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dims[-1]))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, nonl_loss)；logits = uᵀŵ_c/τ。"""
        h = self.encoder(x)
        u = F.normalize(h, dim=1)
        w = F.normalize(self.prototypes, dim=1)
        logits = (u @ w.t()) / self.temperature
        if labels is None:
            return logits, None
        return logits, nonl_loss(u, w, labels, self.temperature)
