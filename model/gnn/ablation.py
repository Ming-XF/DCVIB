"""GNN 消融试验变体（审稿人混杂因素分解，仅 cora 分类）。

- OPBFixedFrame（OPB-FF，仅分类）：GNN 版 GPB 完全同构（随机后验、KL 按
  mask、tied 锚点能量读出），仅冻结先验的全部可训练参数
  （prior_net.requires_grad_(False)，不重建先验分布）——均值块停在恒等帧
  [e_1..e_K]（锚点恒为 a·e_k）、方差块停在 τ²=1，先验恒为 N(a·e_k, I)。
- OPBFixedFrameVar（OPB-FV，仅分类）：帧冻结为恒等帧（fixed_frame 梯度钩子
  清零整个均值块 [:z_dim]），逐类先验方差 τ²_k 仍可学习。
- OPBRandVar（OPB-RV，仅分类）：先验全部冻结，方差块随机初始化后冻结
  （τ²_k 固定随机、非 1）。
"""

import torch
import torch.nn as nn

from .opb import OPB


class OPBFixedFrame(OPB):
    """GNN 版固定帧 OPB（cora 分类消融）：与 MLP 版 OPBFixedFrame 同构。"""

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("OPBFixedFrame 仅分类（cora）")
        kwargs["energy_classifier"] = True
        super().__init__(**kwargs)
        self.prior_net.requires_grad_(False)


class OPBFixedFrameVar(OPB):
    """GNN 版固定帧 + 可学习方差 OPB（cora 分类消融）：与 MLP 版同构。"""

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("OPBFixedFrameVar 仅分类（cora）")
        kwargs["energy_classifier"] = True
        kwargs["fixed_frame"] = True
        super().__init__(**kwargs)


class OPBRandVar(OPB):
    """GNN 版固定帧 + 随机冻结方差 OPB（OPB-RV，cora 分类消融）：先验参数
    全部冻结——均值块停在恒等帧 [e_1..e_K]（锚点恒为 a·e_k）、方差块重新
    随机化（Linear 默认初始化分布）后冻结——τ²_k 是固定但非 1 的随机值。

    与 OPB-FF 对照隔离"方差取值"因素（τ²=1 vs 随机固定 τ²_k），与 GPB 对照
    隔离"先验联合训练"因素。随机值按 run seed 可复现。
    """

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("OPBRandVar 仅分类（cora）")
        kwargs["energy_classifier"] = True
        super().__init__(**kwargs)
        import math
        z_dim = self.prior_net.weight.shape[0] // 2
        with torch.no_grad():
            # 覆盖置零初始化：方差块按 Linear 默认分布重新随机化，再整体冻结
            nn.init.kaiming_uniform_(
                self.prior_net.weight[z_dim:, :], a=math.sqrt(5))
            bound = 1.0 / math.sqrt(self.prior_net.weight.shape[1])
            nn.init.uniform_(self.prior_net.bias[z_dim:], -bound, bound)
        self.prior_net.requires_grad_(False)
