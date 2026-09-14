"""消融试验变体（审稿人混杂因素分解方案，结果保存于 output/ablation）。

- CEBEnergy（分类）：CEB 的可学习自由参考上挂 OPB 同款能量读出——
  logit_k = −‖z − mu_p,k‖²/(2τ²)，τ² 取参考的逐类 exp(logvar)。与 OPB 对照：
  同 IB、同受限读出，几何与尺度自由（隔离"几何 + 固定尺度"因素）。
- CEBTied（回归）：CEB 连续参考轴（prior_net 学出的直线方向 u = W/‖W‖）上挂
  tied 投影读出 ŷ = uᵀz/‖W‖（尺度与学出轴绑定、无自由参数）。与 EPB 对照：
  同 IB、同受限读出，轴非等距、尺度可学。初始化例外：prior_net 均值块置为
  e_1（CEB 置零约定在 tied 读出下除零，与 OPB-R 的 prior_direction 同理）。
- OPBFreeScale（分类 OPB-FS / 回归 EPB-FS）：OPB 同款 QR 正交帧 + 能量/投影
  读出 + IB，但锚点半径可学习（分类逐类 s_k、回归单标量 ρ），初始化等于
  anchor_scale——只关掉"固定尺度"一个因素，训练起点与 OPB 完全一致。
- OPBFixedFrame（OPB-FF，仅分类）：GPB 完全同构（随机后验、KL、tied 锚点能量
  读出、逐类可学习先验方差），仅先验帧冻结为恒等帧 [e_1..e_K]（fixed_frame
  梯度钩子，锚点恒为 a·e_k）——只关掉"帧可训练性"一个因素（审稿人关键消融）。
- EPBFixedAxis（EPB-FA，仅回归）：EPB 完全同构（随机后验、KL、tied 投影读出、
  可学习先验 logvar），仅先验轴冻结为 e_1（fixed_axis）——只关掉"轴可训练性"
  一个因素。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ceb import CEB
from .opb import OPB
from .utils import flatten, kl_divergence, reparameterize


class CEBEnergy(CEB):
    """CEB + 锚点能量读出（仅分类）：自由几何 + 受限读出 + IB。"""

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("CEBEnergy 仅分类（回归对照为 CEBTied）")
        super().__init__(**kwargs)

    def forward(self, x, labels=None, stochastic=True):
        h = self.encoder(flatten(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        # 全类别参考表（评估 labels=None 时同样构造，能量读出需要）
        eye = torch.eye(self.num_classes, device=x.device)
        mu_p_all, logvar_p_all = self.prior_net(eye).chunk(2, dim=1)  # (K, z_dim)
        tau2 = logvar_p_all.exp()
        d2 = (z.unsqueeze(1) - mu_p_all.unsqueeze(0)).pow(2)
        logits = -(d2 / (2.0 * tau2)).sum(dim=2)
        if labels is None:
            return logits, None
        kl = kl_divergence(mu, logvar, mu_p_all[labels], logvar_p_all[labels])
        return logits, kl


class CEBTied(CEB):
    """CEB 回归 + tied 投影读出（仅回归）：自由轴 + 受限读出 + IB。"""

    def __init__(self, **kwargs):
        if not kwargs.get("continuous_y"):
            raise ValueError("CEBTied 仅回归")
        super().__init__(**kwargs)
        # 置零约定在 tied 读出下 u=normalize(W) 除零——均值块置为 e_1
        # （与 OPB-R 的 prior_direction 初始化同理），方差块保持置零。
        z_dim = self.prior_net.weight.shape[0] // 2
        with torch.no_grad():
            self.prior_net.weight.zero_()
            self.prior_net.weight[0, 0] = 1.0

    def forward(self, x, labels=None, stochastic=True):
        h = self.encoder(flatten(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        z_dim = z.shape[1]
        W = self.prior_net.weight[:z_dim, 0]  # 参考轴方向（连续 y 分支的均值块）
        u = F.normalize(W, dim=0)
        logits = ((z @ u) / (W.norm() + 1e-12)).unsqueeze(-1)
        if labels is None:
            return logits, None
        mu_p, logvar_p = self.prior_net(labels.float().unsqueeze(-1)).chunk(2, dim=1)
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl


class OPBFixedFrame(OPB):
    """OPB 固定帧消融（OPB-FF，仅分类）：GPB 完全同构，仅冻结先验的全部
    可训练参数（prior_net.requires_grad_(False)，不重建先验分布）。

    均值块停在恒等帧 [e_1..e_K]（锚点恒为 a·e_k）、方差块停在置零初始值
    （τ²=1），先验恒为 N(a·e_k, I)——即论文 Eq. 7 声明的零参数实例。能量
    读出与 KL 共用同一套冻结锚点与方差，构造路径与 GPB 完全一致。
    """

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("OPBFixedFrame 仅分类（回归对照为 EPBFixedAxis）")
        kwargs["energy_classifier"] = True
        super().__init__(**kwargs)
        self.prior_net.requires_grad_(False)


class EPBFixedAxis(OPB):
    """EPB 固定轴消融（EPB-FA，仅回归）：EPB 完全同构，仅冻结先验的全部
    可训练参数（不重建先验分布）。

    prior_direction 停在 e_1（轴恒为第一坐标轴）、prior_logvar_net 停在
    置零初始值（τ²=1），先验恒为 N(ρ·ỹ·e_1, I)。tied 投影读出与 KL 共用
    同一条冻结等距轴，构造路径与 EPB 完全一致。
    """

    def __init__(self, **kwargs):
        if not kwargs.get("continuous_y"):
            raise ValueError("EPBFixedAxis 仅回归（分类对照为 OPBFixedFrame）")
        kwargs["tied_head"] = True
        super().__init__(**kwargs)
        self.prior_direction.requires_grad_(False)
        self.prior_logvar_net.requires_grad_(False)


class OPBFixedFrameVar(OPB):
    """OPB 固定帧 + 可学习方差消融（OPB-FV，仅分类）：帧冻结为恒等帧
    [e_1..e_K]（锚点恒为 a·e_k），逐类先验方差 τ²_k 仍可学习。

    与 GPB 对比隔离"帧可训练性"（两侧方差均可学习）、与 OPB-FF 对比隔离
    "方差可学习性"（两侧帧均固定）。均值块冻结用 fixed_frame 梯度钩子
    （清零权重行 [:z_dim]，即整个均值块；Linear 权重无法按行
    requires_grad_(False)，冻结参数而非重建先验）。
    """

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("OPBFixedFrameVar 仅分类（cora 消融）")
        kwargs["energy_classifier"] = True
        kwargs["fixed_frame"] = True
        super().__init__(**kwargs)


class EPBRandVar(OPB):
    """EPB 随机冻结方差消融（EPB-RV，仅回归，housing）：与 EPB-FA 同构——
    先验参数全部冻结（轴停在 e_1），但 prior_logvar_net 冻结在默认随机
    初始化上（而非置零）——τ²(y) 是固定但非 1 的随机函数。

    与 EPB-FA 对照隔离"方差取值"因素（τ²=1 vs 随机固定 τ²(y)），与 EPB
    对照隔离"先验联合训练"因素。随机值按 run seed 可复现（模型在
    manual_seed 之后构建）。
    """

    def __init__(self, **kwargs):
        if not kwargs.get("continuous_y"):
            raise ValueError("EPBRandVar 仅回归（housing 消融）")
        kwargs["tied_head"] = True
        super().__init__(**kwargs)
        # 覆盖置零初始化：按 Linear 默认初始化分布重新随机化方差网络
        # （fan_in=1 时 U(-1,1)），再冻结——τ²(y)=exp(w·y+b) 固定随机函数
        import math
        nn.init.kaiming_uniform_(self.prior_logvar_net.weight, a=math.sqrt(5))
        nn.init.uniform_(self.prior_logvar_net.bias, -1.0, 1.0)
        self.prior_direction.requires_grad_(False)
        self.prior_logvar_net.requires_grad_(False)


class OPBRandVar(OPB):
    """OPB 固定帧 + 随机冻结方差消融（OPB-RV，仅分类）：先验参数全部冻结——
    均值块停在恒等帧 [e_1..e_K]（锚点恒为 a·e_k）、方差块按 Linear 默认
    初始化分布重新随机化后冻结——τ²_k 是固定但非 1 的随机值。

    与 OPB-FF 对照隔离"方差取值"因素（τ²=1 vs 随机固定 τ²_k），与 GPB 对照
    隔离"先验联合训练"因素。随机值按 run seed 可复现。
    """

    def __init__(self, **kwargs):
        if kwargs.get("continuous_y"):
            raise ValueError("OPBRandVar 仅分类")
        kwargs["energy_classifier"] = True
        super().__init__(**kwargs)
        import math
        z_dim = self.prior_net.weight.shape[0] // 2
        with torch.no_grad():
            # 覆盖置零初始化：方差块重新随机化，再整体冻结
            nn.init.kaiming_uniform_(
                self.prior_net.weight[z_dim:, :], a=math.sqrt(5))
            bound = 1.0 / math.sqrt(self.prior_net.weight.shape[1])
            nn.init.uniform_(self.prior_net.bias[z_dim:], -bound, bound)
        self.prior_net.requires_grad_(False)


class OPBFreeScale(OPB):
    """OPB 同款几何 + 可学习锚点尺度（分类逐类 s_k、回归标量 ρ）。

    只关掉"固定尺度"一个因素；s_k/ρ 初始化为 anchor_scale，训练起点与
    OPB 完全一致。分类覆盖 _prior_table（能量读出与 KL 共用）；回归覆盖
    forward 的连续分支（rho 替换 anchor_scale）。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.continuous_y:
            self.rho = nn.Parameter(torch.tensor(float(self.anchor_scale)))
        else:
            self.scales = nn.Parameter(
                torch.full((self.num_classes,), float(self.anchor_scale))
            )

    def _prior_table(self):
        prior_mu, prior_logvar_table = super()._prior_table()
        if not self.continuous_y:
            factor = (self.scales / self.anchor_scale).unsqueeze(1)
            prior_mu = prior_mu * factor
        return prior_mu, prior_logvar_table

    def forward(self, x, labels=None, stochastic=True):
        if not self.continuous_y:
            return super().forward(x, labels, stochastic)
        h = self.encoder(flatten(x))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        u = F.normalize(self.prior_direction.weight.squeeze(-1), dim=0)
        rho = self.rho.clamp(min=1e-6)
        if self.tied_head:
            logits = ((z @ u) / rho).unsqueeze(-1)
        else:
            logits = self.classifier(z)
        if labels is None:
            return logits, None
        y_feat = labels.float().unsqueeze(-1)
        mu_p = rho * y_feat * u
        logvar_p = self.prior_logvar_net(y_feat)
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
