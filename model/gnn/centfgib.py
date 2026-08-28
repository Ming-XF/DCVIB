"""CentFGIB（center-loss FGIB，MSE-to-anchor FGIB）GNN 模型定义。

消融模型（审稿人要求）：固定锚点但直接使用 MSE/center loss 代替 KL，
与 MLP 版 model/mlp/centfgib.py 同构。旁路损失为 0.5·‖μ − a·v_y‖²，
无 logvar 头参与、无方差项、无采样；分类器吃确定性 h（同 GNN 版
FGIB，图级任务在读出后的图级表示上计算）。转导设置（cora）下按
train mask 只在训练节点上平均，与 GNN 版 FGIB 的掩码 KL 约定一致。
"""

from .fgib import FGIB


class CentFGIB(FGIB):
    """GNN 版 center-loss 消融：旁路项为按 mask 的 MSE-to-anchor，无 KL。"""

    def forward(self, x, labels=None, adj_norm=None, mask=None, batch=None):
        """返回 (logits, center_loss)；labels 为 None 时返回 None。"""
        h = self.encoder(x, adj_norm)
        if batch is not None:
            from .utils import graph_readout
            h = graph_readout(h, batch, self.pooling)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        if self.continuous_y:
            mu_p, _ = self.anchor_prior(labels.float().unsqueeze(-1))
        else:
            mu_p = self.prior_mu[labels]

        mse = 0.5 * (mu - mu_p).pow(2).sum(dim=1)
        if mask is None:
            return logits, mse.mean()
        return logits, mse[mask].mean()
