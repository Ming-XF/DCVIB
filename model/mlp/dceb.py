"""DCEB（deterministic-path CEB，确定性主路 CEB）模型定义。

消融模型（审稿人要求）：分类器吃确定性 h（同 FGIB），旁路 KL 的先验为
**可训练** prior_net（同 CEB 的反向编码器），即"确定性分类器 + 可训练
条件先验的 CEB-style side loss"。与 FGIB 的唯一差别是锚点是否冻结：
FGIB 冻结 buffer、DCEB 可训练 prior_net。用于隔离 FGIB 的健壮性到底
来自"冻结先验"还是"确定性分类路径"。
"""

import torch.nn as nn
import torch.nn.functional as F

from .fgib import FGIB
from .utils import flatten, kl_divergence


class DCEB(FGIB):
    """FGIB 结构（确定性主路 + 旁路 KL）但先验为可训练反向编码器。

    forward 与 FGIB 相同（返回 (logits, kl)，分类器吃 h），仅先验来源
    不同：mu_p/logvar_p 由 prior_net(one-hot y) 给出（置零初始化，训练
    起点 KL≈0；continuous_y=True 时输入连续 y，同 CEB 约定）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 移除固定锚点 buffer，换可训练 prior_net（同 CEB 的反向编码器）
        self._buffers.pop("prior_mu", None)
        self._buffers.pop("prior_logvar", None)
        prior_in = 1 if self.continuous_y else self.num_classes
        self.prior_net = nn.Linear(prior_in, 2 * self.z_dim)
        nn.init.zeros_(self.prior_net.weight)
        nn.init.zeros_(self.prior_net.bias)

    def forward(self, x, labels=None):
        """返回 (logits, kl)，kl 为 KL(q(z|x) || r(z|y))，r 可训练。"""
        h = self.encoder(flatten(x))
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if self.continuous_y:
            y_feat = labels.float().unsqueeze(-1)
        else:
            y_feat = F.one_hot(labels, num_classes=self.num_classes).float()
        mu_p, logvar_p = self.prior_net(y_feat).chunk(2, dim=1)

        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
