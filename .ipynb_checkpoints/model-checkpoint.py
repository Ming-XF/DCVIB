"""MNIST 多层感知机（MLP）模型定义。"""

import torch
import torch.nn as nn


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

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28) 或 (B, 784)
        if x.dim() == 4:
            x = x.flatten(1)
        elif x.dim() == 3:
            x = x.flatten(1)
        return self.net(x)
