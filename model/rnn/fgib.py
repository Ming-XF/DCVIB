"""FGIB（Fixed-Geometry Information Bottleneck，固定几何信息瓶颈）RNN 模型定义。"""

import torch
import torch.nn as nn

from ..mlp.utils import ContinuousAnchorPrior, build_anchor_prior, kl_divergence, CosineClassifier
from .utils import build_rnn_encoder


class FGIB(nn.Module):
    """RNN 版 FGIB：确定性主路 + 固定锚点先验的旁路瓶颈。

    主路为确定性：编码器输出 h 后直接由分类器得到 logits，z 不参与前向。
    旁路从 h 引出 mu/logvar 头得到后验 q(z|x)；先验不是可学习的标签编码器，
    而是固定的高斯分布（register_buffer、不参与梯度）——分类
    （continuous_y=False）为按类别索引的固定高斯分布表（K 个正交方向缩放
    anchor_scale 作为各类均值，方差固定为 anchor_var），回归
    （continuous_y=True）为固定随机傅里叶特征连续锚点 mu_p(y) =
    anchor_scale·sqrt(2/z_dim)·cos(2π·ω·y+b)（锚点近似落在半径
    anchor_scale 的球面上，见 ContinuousAnchorPrior）——训练损失中加入
    KL(q(z|x) || r(z|y))，梯度只通过共享编码器 h 回传作为正则。
    KL 无法靠移动先验来减小，只要 z 未对齐锚点，正则力就持续存在。
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
        continuous_y: bool = False,
        anchor_scale: float = 4.0,
        anchor_var: float = 1.0,
        pretrained_emb: torch.Tensor | None = None,
        pooling: str = "last",
        cosine_classifier: bool = False,
        freeze_a: bool = False,
        a_identity: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.continuous_y = continuous_y

        self.encoder = build_rnn_encoder(
            vocab_size, hidden_dims, dropout, pad_idx, input_dim, pretrained_emb, pooling
        )

        # 主路：h 直接分类，无采样
        # --cosine-classifier 时用固定温度 cosine 分类器（理论验证实验）
        if cosine_classifier:
            self.classifier = CosineClassifier(hidden_dims[-1], num_classes)
        else:
            self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        # 旁路：h -> (mu, logvar)，参数化 z ~ N(mu, sigma^2)
        # a_identity 消融：A 固定为恒等映射（mu = h，要求 z_dim == hidden_dims[-1]），
        # 即固定正交侧头（FGIB-H）：部署表示 h 直接锚定、κ(AᵀA)=1；
        # freeze_a 消融：A 随机初始化后冻结，检验"旁路头可训练"通道是否被利用
        if a_identity:
            assert z_dim == hidden_dims[-1], "a_identity 要求 z_dim == hidden_dims[-1]"
            self.mu_head = nn.Identity()
        else:
            self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
            if freeze_a:
                self.mu_head.weight.requires_grad_(False)
                self.mu_head.bias.requires_grad_(False)
        self.logvar_head = nn.Linear(hidden_dims[-1], z_dim)

        # 固定类条件先验：分类为每类一个高斯分布表；回归为连续锚点映射
        if continuous_y:
            self.anchor_prior = ContinuousAnchorPrior(
                z_dim, anchor_scale, anchor_var
            )
        else:
            prior_mu, prior_logvar = build_anchor_prior(
                z_dim, num_classes, anchor_scale, anchor_var
            )
            self.register_buffer("prior_mu", prior_mu)
            self.register_buffer("prior_logvar", prior_logvar)

        # 置零初始化：训练起点 sigma = 1，初始 KL ≈ 0.5 * anchor_scale^2
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None):
        """返回 (logits, kl)，labels 为 None 时 kl 为 None。

        logits 仅由确定性主路产生；kl 来自旁路 z 与固定类条件先验的
        KL(q(z|x) || r(z|y))，只依赖 mu/logvar，无需采样。
        """
        h = self.encoder(x)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if self.continuous_y:
            mu_p, logvar_p = self.anchor_prior(labels.float().unsqueeze(-1))
        else:
            mu_p = self.prior_mu[labels]
            logvar_p = self.prior_logvar[labels]

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
