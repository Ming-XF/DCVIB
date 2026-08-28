"""CentFGIB（center-loss FGIB，MSE-to-anchor FGIB）RNN 模型定义。

消融模型（审稿人要求）：固定锚点但直接使用 MSE/center loss 代替 KL，
与 MLP 版 model/mlp/centfgib.py 同构。旁路损失为 0.5·‖μ − a·v_y‖²
（或回归时 ‖μ − μ_p(y)‖²），无 logvar 头参与、无方差项、无采样；
分类器吃确定性 h（同 RNN 版 FGIB）。
"""

from .fgib import FGIB


class CentFGIB(FGIB):
    """RNN 版 center-loss 消融：旁路项为直接 MSE-to-anchor，无 KL。"""

    def forward(self, x, labels=None):
        """返回 (logits, center_loss)；labels 为 None 时返回 None。"""
        h = self.encoder(x)
        logits = self.classifier(h)

        if labels is None:
            return logits, None

        mu = self.mu_head(h)
        if self.continuous_y:
            mu_p, _ = self.anchor_prior(labels.float().unsqueeze(-1))
        else:
            mu_p = self.prior_mu[labels]

        return logits, 0.5 * (mu - mu_p).pow(2).sum(dim=1).mean()
