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
