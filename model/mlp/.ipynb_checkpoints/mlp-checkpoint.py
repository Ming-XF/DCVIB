"""MNIST 多层感知机（MLP）模型定义。"""

import torch
import torch.nn as nn

from .utils import build_hidden_layers, flatten


class MLP(nn.Module):
    """一个简单的多层感知机，用于 MNIST 手写数字分类。

    输入为 28x28 的灰度图，展平后为 784 维向量，
    经过若干隐藏层（默认 512 -> 256），最终输出 10 类 logits。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = build_hidden_layers(input_dim, hidden_dims, dropout)
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(flatten(x))
