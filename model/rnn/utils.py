"""RNN 公共工具函数。"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class RNNEncoder(nn.Module):
    """多层 LSTM 文本编码器，输出末层末时刻隐状态 h。

    支持两种输入：
    - token id 序列（input_dim=None）：词嵌入维度取 hidden_dims 首元素，
      输入为 (B, L) 的 token id 张量（pad_idx 后填充），长度由
      (x != pad_idx).sum(1) 计算；
    - 连续向量序列（input_dim=D，如 BERT 逐 token 特征）：输入为
      (B, L, D) 浮点张量（全零向量后填充），长度由零向量检测
      （首个全零向量之前的位置数）计算。

    层数与各层隐层维度由 hidden_dims 列表决定（与 MLP/GNN 骨干约定一致，
    如 (512, 256)：两层 LSTM 512→256）。经 pack_padded_sequence 跳过填充
    位置后逐层送入 LSTM，除末层外每层输出后接 Dropout；取末层 h_n[-1]
    （形状 (B, hidden_dims[-1])）作句子表示，再经 Dropout 输出。
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.input_dim = input_dim
        if input_dim is None:
            self.embedding = nn.Embedding(vocab_size, hidden_dims[0], padding_idx=pad_idx)
            lstm_in = hidden_dims[0]
        else:
            self.embedding = None
            lstm_in = input_dim
        self.lstms = nn.ModuleList()
        for out_dim in hidden_dims:
            self.lstms.append(nn.LSTM(lstm_in, out_dim, batch_first=True))
            lstm_in = out_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.input_dim is not None:
            # 连续向量序列：padding 位置是精确全零向量，长度 = 非零向量个数
            lengths = (x.abs().sum(dim=-1) > 0).sum(dim=1).clamp(min=1)
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            # clamp 防全 pad 样本导致 pack 报错（数据层已用 [unk] 兜底，双保险）
            lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1)
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
    vocab_size: int | None = None,
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout: float = 0.2,
    pad_idx: int = 0,
    input_dim: int | None = None,
) -> RNNEncoder:
    """构造 RNN 特征提取器。

    input_dim=None 时用 Embedding（维度 = hidden_dims[0]，vocab_size 必填）；
    input_dim=D 时为连续向量序列输入（跳过 Embedding，LSTM 输入 D 维）。
    """
    return RNNEncoder(vocab_size, hidden_dims, dropout, pad_idx, input_dim)
