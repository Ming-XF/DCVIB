"""GCN（图卷积网络）基线模型定义。"""

import torch.nn as nn

from .utils import build_gcn_encoder


class GCN(nn.Module):
    """GCN 基线，用于 Cora 节点分类。

    输入为节点特征矩阵 (N, input_dim) 与归一化邻接 (N, N)，经过 GCN 编码器
    （层数与维度由 hidden_dims 列表决定，默认 (512, 256)，与 MLP 骨干约定一致）
    后由线性层输出 num_classes 类 logits。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x, adj_norm=None):
        h = self.encoder(x, adj_norm)
        return self.classifier(h)
