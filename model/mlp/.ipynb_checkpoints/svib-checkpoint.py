"""SVIB（平方信息瓶颈，Squared-IB）模型定义。"""

from .vib import VIB


class SVIB(VIB):
    """平方信息瓶颈（Squared-IB）变体，基于 VIB 架构（ICLR 2019 "Caveats"）。

    与 VIB 的唯一区别在损失：squared-IB functional（Rodríguez Gálvez et al.,
    Eq. 8）为 max I(Y;T) − β·I(X;T)²，压缩项取平方；变分形式下 KL 是
    I(X;Z) 的上界，目标等价于 CE + β·KL²，故 forward 返回平方后的 KL。
    动机：分类任务 H(Y|X)=0 时标准 IB Lagrangian 只能取到硬聚类端点、无法
    扫出压缩-精度曲线，平方压缩项可恢复整条 IB 曲线。
    """

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl²)，先验为 N(0, I)。labels 参数仅为统一接口。"""
        logits, kl = super().forward(x, labels, stochastic=stochastic)
        return logits, kl ** 2
