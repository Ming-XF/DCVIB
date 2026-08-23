"""RNN（LSTM 文本分类）基线模型定义。"""

import torch.nn as nn

from .utils import build_rnn_encoder


class RNN(nn.Module):
    """RNN 基线，用于 IMDb 情感二分类。

    输入为 (B, L) 的 token id 张量（pad_idx 后填充），经多层 LSTM 编码器
    得到末层末时刻隐状态 h（hidden_dims[-1] 维），由线性层输出
    num_classes 类 logits。
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
    ):
        super().__init__()

        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim
        )
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)
