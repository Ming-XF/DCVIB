"""SVIB（平方信息瓶颈，Squared-IB）GNN 模型定义。"""

from .vib import VIB


class SVIB(VIB):
    """平方信息瓶颈（Squared-IB）变体，基于 GNN 版 VIB 架构。

    与 VIB 的唯一区别在损失：squared-IB functional（ICLR 2019 "Caveats",
    Eq. 8）为 max I(Y;T) − β·I(X;T)²，变分形式下目标等价于 CE + β·KL²，
    故 forward 返回平方后的（mask 内）KL。详见 model/mlp/svib.py。
    """

    def forward(self, x, labels=None, stochastic=True, adj_norm=None, mask=None, batch=None):
        """返回 (logits, kl²)，先验为 N(0, I)。labels 参数仅为统一接口。"""
        logits, kl = super().forward(
            x, labels, stochastic=stochastic, adj_norm=adj_norm, mask=mask, batch=batch
        )
        return logits, kl ** 2
