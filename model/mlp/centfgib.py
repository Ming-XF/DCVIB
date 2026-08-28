"""CentFGIB（center-loss FGIB，MSE-to-anchor FGIB）模型定义。

消融模型（审稿人要求）：固定锚点但直接使用 MSE/center loss 代替 KL。
旁路损失为 0.5·‖μ − a·v_y‖²（逐样本求和后 batch 平均，与 KL 的位移项
在 τ²=1 时的尺度一致），无 logvar 头参与、无方差项、无采样。若
CentFGIB 与 FGIB 表现一致，则 KL 机制可退化为普通原型正则；否则
说明高斯 KL 的形式（方差项 + 置零初始化约定）承载了部分效应。
"""

import torch.nn as nn

from .fgib import FGIB
from .utils import flatten


class CentFGIB(FGIB):
    """FGIB 的 center-loss 消融：旁路项为直接 MSE-to-anchor，无 KL。"""

    def forward(self, x, labels=None):
        """返回 (logits, center_loss)；labels 为 None 时返回 None。"""
        h = self.encoder(flatten(x))
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        if self.continuous_y:
            mu_p, _ = self.anchor_prior(labels.float().unsqueeze(-1))
        else:
            mu_p = self.prior_mu[labels]

        # 与 KL 位移项 0.5·‖μ−μ_p‖²/τ²（τ²=1）同尺度
        return logits, 0.5 * (mu - mu_p).pow(2).sum(dim=1).mean()
