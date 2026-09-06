"""NCBD-NONL（Neural Collapse by Design，Koromilas et al., ICML 2026）MLP 模型。

审稿人对比基线（几何分类方向）：特征与可学习类原型均归一化到单位超球面，
logits = uᵀŵ_c/τ（无偏置），损失完全以 NONL 对比损失替代 CE（无 KL、无
随机后验）。官方理论保证全局极小点达到 NC1–NC3 / simplex ETF；本实现为
同 backbone/划分/seed/训练预算的适配实现（Adam+100ep 而非原文 SGD+500ep）。

forward 返回 (logits, nonl_loss)：NONL 值放第二项（训练管道对 ncbd 直接
用该项作损失、忽略 beta），labels 为 None 时（评估无标签）返回
(logits, None)。仅分类任务。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_hidden_layers, flatten, nonl_loss


class NCBD(nn.Module):
    """单位球原型分类器 + NONL 损失（无 IB）：u = h/‖h‖、ŵ_c = w_c/‖w_c‖。"""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 10,
        dropout: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        # 可学习类原型 (K, d)：初始化同 nn.Linear 权重（kaiming uniform），
        # 前向时逐行归一化
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden_dims[-1]))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, nonl_loss)；logits = uᵀŵ_c/τ（评估用 softmax 类别得分）。"""
        h = self.encoder(flatten(x))
        u = F.normalize(h, dim=1)
        w = F.normalize(self.prototypes, dim=1)
        logits = (u @ w.t()) / self.temperature
        if labels is None:
            return logits, None
        return logits, nonl_loss(u, w, labels, self.temperature)
