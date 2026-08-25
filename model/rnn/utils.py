"""RNN 公共工具函数。"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    """多层 LSTM 文本编码器，输出句子表示 h。

    支持两种输入：
    - token id 序列（input_dim=None）：词嵌入维度取 hidden_dims 首元素，
      输入为 (B, L) 的 token id 张量（pad_idx 后填充），长度由
      (x != pad_idx).sum(1) 计算；
    - 连续向量序列（input_dim=D，如 BERT 逐 token 特征）：输入为
      (B, L, D) 浮点张量（全零向量后填充），长度由零向量检测
      （首个全零向量之前的位置数）计算。

    层数与各层隐层维度由 hidden_dims 列表决定（与 MLP/GNN 骨干约定一致，
    如 (512, 256)：两层 LSTM 512→256）。经 pack_padded_sequence 跳过填充
    位置后逐层送入 LSTM，除末层外每层输出后接 Dropout。无 pretrained_emb
    时词嵌入维度取 hidden_dims 首元素；有 pretrained_emb 时嵌入维度取自
    矩阵宽度、hidden_dims 仅作逐层 LSTM 隐层维度。

    读出方式由 pooling 决定："last"（默认）取末层末时刻隐状态 h_n[-1]；
    "max"/"mean" 把末层输出按时间做掩码最大/平均池化（STS-B 回归用 "max"，
    对相似度任务末时刻状态偏差大）。pretrained_emb 给出时以预训练词向量
    （如 GloVe）初始化嵌入层并冻结（小数据集上微调词向量会迅速过拟合）。
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.input_dim = input_dim
        self.pooling = pooling
        if input_dim is None:
            if pretrained_emb is not None:
                assert len(pretrained_emb) == vocab_size, (
                    f"预训练词向量行数 {len(pretrained_emb)} 与词表大小 {vocab_size} 不一致"
                )
                # 有预训练词向量时嵌入维度取自矩阵宽度，hidden_dims 仅作逐层
                # LSTM 隐层维度（否则 GloVe 100d 会强制出现 100→100 的窄层）
                emb_dim = pretrained_emb.size(1)
                layer_dims = hidden_dims
            else:
                emb_dim = hidden_dims[0]
                layer_dims = hidden_dims
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
            if pretrained_emb is not None:
                self.embedding.weight.data.copy_(pretrained_emb)
                # 冻结预训练词向量：小数据集上微调会迅速过拟合（STS-B 实验验证）
                self.embedding.weight.requires_grad_(False)
            lstm_in = emb_dim
        else:
            self.embedding = None
            lstm_in = input_dim
            layer_dims = hidden_dims
        self.lstms = nn.ModuleList()
        for out_dim in layer_dims:
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
        if self.pooling == "last":
            return self.dropout(h_n[-1])
        out, _ = pad_packed_sequence(packed, batch_first=True)
        mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        if self.pooling == "max":
            h = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        else:  # mean
            h = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths.float().unsqueeze(-1)
        return self.dropout(h)


def build_rnn_encoder(
    vocab_size: int | None = None,
    hidden_dims: tuple[int, ...] = (512, 256),
    dropout: float = 0.2,
    pad_idx: int = 0,
    input_dim: int | None = None,
    pretrained_emb: torch.Tensor | None = None,
    pooling: str = "last",
) -> RNNEncoder:
    """构造 RNN 特征提取器。

    input_dim=None 时用 Embedding（维度 = hidden_dims[0]，vocab_size 必填），
    pretrained_emb 给出时以预训练词向量（如 GloVe）初始化嵌入层并冻结；
    input_dim=D 时为连续向量序列输入（跳过 Embedding，LSTM 输入 D 维）。
    pooling 为读出方式（"last"/"max"/"mean"，见 RNNEncoder）。
    """
    return RNNEncoder(vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling)
