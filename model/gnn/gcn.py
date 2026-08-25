"""GCN（图卷积网络）基线模型定义。"""

import torch.nn as nn

from .utils import build_gcn_encoder, graph_readout


class GCN(nn.Module):
    """GCN 基线，用于 Cora 节点分类与 ZINC 图级回归。

    输入为节点特征矩阵 (N, input_dim) 与归一化邻接 (N, N)，经过 GCN 编码器
    （层数与维度由 hidden_dims 列表决定，默认 (512, 256)，与 MLP 骨干约定一致）
    后由线性层输出 num_classes 类 logits。batch 非 None 时（图级任务，
    ZINC 图批）在编码器后做 mean 图读出，logits 按图输出。
    """

    def __init__(
        self,
        input_dim: int = 1433,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling

        self.encoder = build_gcn_encoder(input_dim, hidden_dims, dropout)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x, adj_norm=None, batch=None):
        h = self.encoder(x, adj_norm)
        if batch is not None:
            h = graph_readout(h, batch, self.pooling)
        return self.classifier(h)
