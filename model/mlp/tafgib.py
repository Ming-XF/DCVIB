"""TAFGIB（trainable-anchor FGIB，可训练锚点 FGIB）模型定义。

消融模型（审稿人要求）：FGIB 的锚点均值表从固定 buffer 换成可训练参数
（初始化为固定 QR 正交锚点），其余完全一致（确定性主路、旁路 KL、
logvar 头置零初始化）。用于隔离"冻结先验"（vs 可训练先验）对 FGIB
健壮性的贡献；与 DCEB（先验为 prior_net 网络）互补：TAFGIB 保持锚点
表的几何结构、只放开均值可训练。
"""

import torch.nn as nn

from .fgib import FGIB


class TAFGIB(FGIB):
    """FGIB 的可训练锚点版本：prior_mu 为 nn.Parameter（锚点方差仍固定）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 固定 buffer 换成同初始化的可训练参数（分类模式才有 prior_mu 表）
        if not self.continuous_y:
            mu = self._buffers.pop("prior_mu")
            self.register_parameter("prior_mu", nn.Parameter(mu))
        # prior_logvar 保持固定（只放开均值，与 FGIB 的方差固定约定一致）

    def forward(self, x, labels=None):
        """与 FGIB 相同，仅 prior_mu 参与梯度。"""
        return super().forward(x, labels)
