"""NCM 原型分类器变体（消融试验用）：无 IB 的受限读出对照。

审稿人混杂因素分解（paper/ablation 方案）的"固定原型但不用 IB / 纯 prototype
classifier"对照：

- NCMLearn：确定性 z = mu_head(encoder(x))，可训练原型 {m_k}（K×z_dim 参数，
  无几何约束），能量读出 logit_k = −‖z − m_k‖²/(2τ²)（τ²=1 固定），无后验、
  无 KL——纯原型分类器。
- NCMOrtho：同 NCMLearn，但原型固定为 anchor_scale·e_k（register_buffer，
  不参与训练）——"固定正交原型 + 不用 IB"；与 NCMLearn 之差 = 原型结构，
  与 OPB 之差 = IB 框架的增量（注：OPB 的帧可训练、此处原型固定，该对照
  附带帧可训练性混杂，表中如实注明）。

两者 forward 均返回 (logits, None)（kl=None，训练走基线损失路径，无 β 维度）。
仅分类任务（回归无类别原型表）。
"""

import torch
import torch.nn as nn

from .utils import build_hidden_layers, flatten


class _NCMBase(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
        anchor_scale: float = 4.0,
        learnable: bool = True,
    ):
        super().__init__()
        assert num_classes <= z_dim, "原型能量读出要求类别数不超过 z 维度"
        self.num_classes = num_classes
        self.anchor_scale = anchor_scale
        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        self.mu_head = nn.Linear(hidden_dims[-1], z_dim)
        if learnable:
            self.prototypes = nn.Parameter(torch.randn(num_classes, z_dim) * 0.01)
        else:
            proto = torch.zeros(num_classes, z_dim)
            proto[range(num_classes), range(num_classes)] = anchor_scale
            self.register_buffer("prototypes", proto)

    def energy_logits(self, z):
        """logit_k = −‖z − m_k‖²/(2τ²)，τ²=1 固定。"""
        d2 = (z.unsqueeze(1) - self.prototypes.unsqueeze(0)).pow(2).sum(-1)
        return -d2 / 2.0

    def forward(self, x, labels=None, stochastic=True):
        z = self.mu_head(self.encoder(flatten(x)))
        return self.energy_logits(z), None


class NCMLearn(_NCMBase):
    """可训练原型能量读出，无 IB（原型无几何约束）。"""

    def __init__(self, **kwargs):
        kwargs["learnable"] = True
        super().__init__(**kwargs)


class NCMOrtho(_NCMBase):
    """固定正交原型 a·e_k 能量读出，无 IB。"""

    def __init__(self, **kwargs):
        kwargs["learnable"] = False
        super().__init__(**kwargs)
