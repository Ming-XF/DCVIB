"""NCBD-NONL（Neural Collapse by Design，ICML 2026）CNN 模型。

与 MLP 版同构（见 model/mlp/ncbd.py）：卷积编码器输出 h 归一化为 u，与
逐行归一化的可学习类原型做温度缩放的余弦 logits，损失为 NONL 对比损失
（无 CE、无 KL）。仅分类任务（mnist / imagenet100 空间池化特征）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import nonl_loss
from .utils import build_cnn_encoder


class NCBD(nn.Module):
    """CNN 编码器 + 单位球原型分类器 + NONL 损失。"""

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        input_size: int = 28,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.encoder = build_cnn_encoder(
            input_channels, conv_channels, hidden_dim, dropout, input_size=input_size
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dim))
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
