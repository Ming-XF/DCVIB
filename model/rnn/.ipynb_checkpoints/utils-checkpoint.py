"""RNN 公共工具函数。"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class RNNEncoder(nn.Module):
    """多层 LSTM 文本编码器：词嵌入 + 逐层 LSTM，输出末层末时刻隐状态 h。

    词嵌入维度取 hidden_dims 首元素；层数与各层隐层维度由 hidden_dims 列表
    决定（与 MLP/GNN 骨干约定一致，如 (512, 256)：嵌入 512 维 + 两层 LSTM
    512→256）。输入为 (B, L) 的 token id 张量（pad_idx 后填充）；长度由
    pad 掩码计算（lengths = (x != pad_idx).sum(1)），经 pack_padded_sequence
    跳过填充位置后逐层送入 LSTM，除末层外每层输出后接 Dropout；取末层
    h_n[-1]（形状 (B, hidden_dims[-1])）作句子表示，再经 Dropout 输出。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_dims[0], padding_idx=pad_idx)
        self.lstms = nn.ModuleList()
        in_dim = hidden_dims[0]
        for out_dim in hidden_dims:
            self.lstms.append(nn.LSTM(in_dim, out_dim, batch_first=True))
            in_dim = out_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # clamp 防全 pad 样本导致 pack 报错（数据层已用 [unk] 兜底，双保险）
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1)
        # pack_padded_sequence 要求 lengths 在 CPU 上
        packed = pack_padded_sequence(
            self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        for i, lstm in enumerate(self.lstms):
            packed, (h_n, _) = lstm(packed)
            if i < len(self.lstms) - 1:
                packed = PackedSequence(
                    self.dropout(packed.data), packed.batch_sizes,
                    packed.sorted_indices, packed.unsorted_indices,
                )
        return self.dropout(h_n[-1])


def build_rnn_encoder(
    vocab_size: int,
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout: float = 0.2,
    pad_idx: int = 0,
) -> RNNEncoder:
    """构造 RNN 特征提取器：Embedding（维度 = hidden_dims[0]）+ 多层 LSTM（逐层维度由 hidden_dims 决定）。"""
    return RNNEncoder(vocab_size, hidden_dims, dropout, pad_idx)
