"""NIB（非线性信息瓶颈）RNN 模型定义。"""

import math

import torch
import torch.nn as nn

from ..mlp.utils import nib_mi_upper_bound
from .utils import build_rnn_encoder


class NIB(nn.Module):
    """非线性信息瓶颈（NIB）RNN 版，结构与 MLP 版一一对应（见 model/mlp/nib.py）。

    编码器换成 LSTM 文本编码器（IMDb token id 或 AG News BERT 逐 token 特征）；
    瓶颈表示 f(x) 注入同方差高斯噪声后，压缩项用成对距离非参数上界 Î
    估计 I(X;M)，损失为 CE + β·Î。σ² 为可学习参数（log 空间，初始为论文值 1.0）。
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
        noise_var_init: float = 1.0,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
    ):
        super().__init__()
        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )
        self.bottleneck_head = nn.Linear(hidden_dims[-1], z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        # σ² 可学习（log 空间，初始为论文值 1.0）
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(noise_var_init)))

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, î)，î 为 I(X;M) 的非参数上界。labels 参数仅为统一接口。"""
        h = self.encoder(x)
        m = self.bottleneck_head(h)
        noise_var = self.log_noise_var.clamp(-10.0, 10.0).exp()
        if stochastic:
            m = m + torch.randn_like(m).mul_(noise_var.sqrt())
        return self.classifier(m), nib_mi_upper_bound(m, noise_var)
