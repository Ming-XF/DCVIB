"""FGIBS（FGIB-S，随机部署 canonical mode）模型定义。

FGIB 框架的第二个 canonical mode（第五轮，审稿人建议）：固定几何条件瓶颈的
**随机部署**实例——分类器消费采样 z（训练与随机推理都部署 z），使变分证书
的对象与实际部署对象一致（P4 的 certificate-object mismatch 对本模式失效）。
与 FGIB-H 共享固定正交侧头（A=I）、固定锚点先验 r(z|y)=N(a·v_y, τ²I) 与
零惩罚集设计；区别在于方差头**可训练**、随机 code z 就是部署表示。

先验方差 τ² 固定 ⇒ ∂σ²KL = ½(1/τ² − 1/σ²) 对后验方差有恢复力：CE 压 σ² 下、
KL 拉回 τ²——与 VIB 同构的机理 + 锚点几何，按构造没有 CEB 式 q/p 联合下移
的方差竞赛通道（rem:vib-prior 的对应物，见论文）。
"""

import torch

from .fgib import FGIB
from .utils import flatten, kl_divergence, reparameterize


class FGIBS(FGIB):
    """FGIB-S：随机部署的固定几何信息瓶颈。

    训练目标 CE(c(z), y) + β·KL(q(z|x)‖r(z|y))，z 由重参数化采样；推理同样
    使用采样 z（部署对象即证书对象）。stochastic=False 时用 z=μ 做均值部署
    对照（主协议评估仍走采样，见 train.py run_model 的 FGIBS 分支）。
    """

    def __init__(self, *args, **kwargs):
        # canonical：侧头固定为恒等映射（A=I 固定正交，κ=1），z_dim == hidden_dim
        kwargs.setdefault("a_identity", True)
        super().__init__(*args, **kwargs)
        # 方差头保持可训练（FGIB.__init__ 置零初始化：起点 σ²=τ²、KL≈½a²）

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl)；labels 为 None 时 kl 为 None。

        logits 由采样 z（stochastic=True）或后验均值 μ（stochastic=False，
        均值部署对照）经分类器产生；kl 为 KL(q(z|x)‖r(z|y))，与 FGIB 相同。
        """
        h = self.encoder(flatten(x))
        mu = self.mu_head(h)  # Identity：mu = h
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)
        if labels is None:
            return logits, None
        if self.continuous_y:
            mu_p, logvar_p = self.anchor_prior(labels.float().unsqueeze(-1))
        else:
            mu_p = self.prior_mu[labels]
            logvar_p = self.prior_logvar[labels]
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl
