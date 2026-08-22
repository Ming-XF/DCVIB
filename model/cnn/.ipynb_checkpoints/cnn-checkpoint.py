"""MNIST 卷积神经网络（CNN）模型定义。"""

import torch
import torch.nn as nn

from .utils import build_cnn_encoder


class CNN(nn.Module):
    """一个简单的卷积神经网络，用于 MNIST 手写数字分类。

    输入为 28x28 的灰度图，经过卷积特征提取器（默认 32 -> 64 通道），
    展平后映射到 hidden_dim（默认 256），最终输出 10 类 logits。
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = build_cnn_encoder(
            input_channels, conv_channels, hidden_dim, dropout
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)
