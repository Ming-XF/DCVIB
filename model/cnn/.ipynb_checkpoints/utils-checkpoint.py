"""CNN 公共工具函数。"""

import torch.nn as nn


def build_cnn_encoder(
    input_channels: int = 1,
    conv_channels: tuple[int, ...] = (32, 64),
    hidden_dim: int = 256,
    dropout: float = 0.0,
    input_size: int = 28,
) -> nn.Sequential:
    """构造卷积特征提取器：Conv3x3+ReLU+MaxPool2x2 堆叠，展平后接 Linear 到 hidden_dim。

    空间尺寸每次池化减半，默认 (32, 64) 通道时 28x28 -> 64x7x7=3136 维。
    input_size <= 8 的小特征图（如 ImageNet-100 的 4×4 池化特征）不做下采样：
    两层 Conv3x3 把通道降到 64 后展平，避免 MaxPool 把仅有的空间单元压没。
    """
    layers = []
    if input_size <= 8:
        layers.append(nn.Conv2d(input_channels, 128, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        flat_dim = 64 * input_size * input_size
    else:
        in_ch = input_channels
        size = input_size
        for out_ch in conv_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            size //= 2
            in_ch = out_ch
        flat_dim = conv_channels[-1] * size * size
    layers.append(nn.Flatten())
    layers.append(nn.Linear(flat_dim, hidden_dim))
    layers.append(nn.ReLU(inplace=True))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
