"""DVCCA（监督 β-DVCCA）RNN 模型定义。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mlp.utils import build_hidden_layers, kl_divergence, reparameterize
from .utils import build_rnn_encoder


class DVCCA(nn.Module):
    """监督 β-DVCCA（DVMIB 框架的监督退化形式，Abdelaleem et al., JMLR 2025）RNN 版。

    同 MLP/CNN/GNN 版：VIB 结构 + 重建解码器，损失 = CE + β·(KL + MSE_recon)
    （forward 返回 (logits, kl, recon_loss)，train.py 统一处理）。
    文本输入没有"原始像素"意义上的重建目标，Ĩ_D(X;Z) 取输入的时间池化
    表示：token 模式（IMDb/STS-B，input_dim=None）重建嵌入序列的 max 时间
    池化向量（B, emb_dim，IMDb 为可训练嵌入、STS-B 为冻结 GloVe 嵌入）；
    BERT 特征模式（AG News，input_dim=D）重建特征序列的 max 时间池化向量
    （B, D）。解码器为 z → 该向量的小 MLP，MSE 即高斯解码器的 −ln q(x|z)
    （与 GNN 版"从图级 z 广播重建节点特征"同为向输入表示的退化重建）。
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        dropout: float = 0.2,
        pad_idx: int = 0,
        input_dim: int | None = None,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.input_dim = input_dim
        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )
        # 重建目标维度：token 模式为嵌入维度（无预训练词向量时取 hidden_dims
        # 首元素，有 GloVe 时取矩阵宽度），BERT 特征模式为 input_dim
        if input_dim is None:
            recon_dim = pretrained_emb.size(1) if pretrained_emb is not None else hidden_dims[0]
        else:
            recon_dim = input_dim

        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        # 重建解码器：z → 输入表示空间（与 MLP 版同构的反向 MLP，高斯解码器）
        self.decoder = nn.Sequential(
            *build_hidden_layers(z_dim, tuple(reversed(hidden_dims)), dropout),
            nn.Linear(hidden_dims[0], recon_dim),
        )

        # 置零初始化：训练开始时 sigma = 1，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def _recon_target(self, x):
        """输入的时间池化表示（重建目标）：token 模式为 max 池化嵌入序列、
        BERT 特征模式为 max 池化特征序列。"""
        if self.input_dim is not None:
            feats = x
            mask = (x.abs().sum(dim=-1) > 0).unsqueeze(-1)
        else:
            feats = self.encoder.embedding(x)
            mask = (x != self.pad_idx).unsqueeze(-1)
        return feats.masked_fill(~mask, float("-inf")).max(dim=1).values

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl, recon_loss)；先验为 N(0, I)。labels 参数仅为统一接口。"""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)
        recon = F.mse_loss(self.decoder(z), self._recon_target(x))

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl, recon
