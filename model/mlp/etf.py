"""ETF（固定等角紧框架分类器）基线模型定义。

审稿人补充基线：MLP 编码器 + **冻结**的 ETF（equiangular tight frame）
分类头，检验"固定几何"的收益究竟来自表示侧锚点正则还是分类头几何。
分类头权重 W ∈ R^{K×d} 按标准单形 ETF 构造：取 QR 正交帧 Q ∈ R^{d×K}，
W = √(K/(K-1))·(I_K − 1_K1_Kᵀ/K)·Qᵀ —— 各行范数为 1、互余弦 −1/(K−1)，
bias 关闭且整体不参与梯度（requires_grad_(False)）。无任何旁路正则项。
"""

import math

import torch
import torch.nn as nn

from .utils import build_hidden_layers, flatten


class ETF(nn.Module):
    """确定性 MLP + 冻结 ETF 分类头（仅 MNIST/MLP 基线）。"""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]
        assert z_dim == hidden_dim and num_classes <= hidden_dim, (
            "ETF 分类头要求 z_dim == hidden_dims[-1] 且 num_classes <= hidden_dim"
        )

        # 冻结 ETF 分类头：行范数 1、互余弦 -1/(K-1)，整体不参与梯度
        q, _ = torch.linalg.qr(torch.randn(hidden_dim, num_classes))
        s = math.sqrt(num_classes / (num_classes - 1))
        w = s * (
            torch.eye(num_classes)
            - torch.ones(num_classes, num_classes) / num_classes
        ) @ q.t()
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(w)
        self.classifier.weight.requires_grad_(False)

    def forward(self, x):
        """纯基线：只返回 logits（与 MLP 基线接口一致，run_model 兜底分支处理）。"""
        return self.classifier(self.encoder(flatten(x)))
