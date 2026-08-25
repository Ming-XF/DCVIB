"""NIB（非线性信息瓶颈）模型定义。"""

import math

import torch
import torch.nn as nn

from .utils import build_hidden_layers, flatten, nib_mi_upper_bound


class NIB(nn.Module):
    """非线性信息瓶颈（NIB，Kolchinsky et al., Entropy 2019）。

    编码器输出经瓶颈头映射到 z_dim 维表示 f(x)，训练时注入同方差高斯噪声
    m = f(x) + ε（ε~N(0, σ²·I)）；压缩项不用任何先验，而是用成对距离的
    非参数上界 Î 估计 I(X;M)（论文 Eq. 10/16，见 utils.nib_mi_upper_bound）。
    forward 返回 (logits, Î)，损失为 CE + β·Î。

    噪声方差 σ² 按论文为可学习参数（"the noise parameter σ² was one of the
    trainable parameters in θ"，Section IVA）：log 空间参数化保证正值，
    初始 log σ² = 0 即 σ² = 1（论文初始值）。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        noise_var_init: float = 1.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        self.bottleneck_head = nn.Linear(hidden_dims[-1], z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        # σ² 可学习（log 空间，初始为论文值 1.0）
        self.log_noise_var = nn.Parameter(torch.tensor(math.log(noise_var_init)))

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, î)，î 为 I(X;M) 的非参数上界。labels 参数仅为统一接口。"""
        h = self.encoder(flatten(x))
        m = self.bottleneck_head(h)
        noise_var = self.log_noise_var.clamp(-10.0, 10.0).exp()
        if stochastic:
            m = m + torch.randn_like(m).mul_(noise_var.sqrt())
        return self.classifier(m), nib_mi_upper_bound(m, noise_var)
