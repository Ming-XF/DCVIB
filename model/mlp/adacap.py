"""AdaCap（Adaptive Contrastive Approach，Belucci et al., ESANN 2026）MLP 模型。

审稿人对比基线（非 IB 回归方向，小样本对比正则）：两个组件——① Tikhonov
闭式输出层：每批表示 H(B,d) 的岭回归 β = (HᵀH + λI)⁻¹Hᵀy，λ = exp(log_alpha)
**可学习**（官方实现：HᵀH/Hᵀy 在 no_grad 下缓存，表示梯度仅经
y_hat = h @ beta 的外层 h 回传，与官方一致）；② 置换对比损失：P=10 个批内
打乱标签复用同一特征分解，L = MSE(y, ŷ) − (1/P)Σ_p MSE(π_p(y), ŷ^(p))
（拟合真实标签好、拟合乱标签差）。

评估读出（无标签泄漏）：训练集全部表示（eval 模式 BN）拟 β 存 buffer，
val/test 预测 ŷ = H·β——与官方代码 eval 复用最后训练批 β 的行为不同，论文
中声明。λ 初始值由 --beta 槽位传入（官方默认 1e2；原文的网格初始化由
调参网格替代）。仅回归任务。

forward 返回 (y_pred, loss)：训练时 loss 为 AdaCap 对比损失（训练管道对
adacap 直接用该项作损失、忽略 beta 与 criterion）；评估时（self.training
为 False）用拟合好的 beta 读出、返回 (y_pred, None)（Loss 列 = 纯 MSE）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_hidden_layers, flatten


def ridge_betas(h, y, y_perm, log_alpha, tol=1e-10):
    """Tikhonov 闭式解：beta = (HᵀH + λI)⁻¹Hᵀy（λ = exp(log_alpha) + tol）。

    返回 (beta (d,1), beta_perm (d,P))；y_perm 为 None 时 beta_perm 为 None。
    与官方实现一致：Gram 矩阵与交叉项在 no_grad 下缓存，表示梯度仅经
    y_hat = h @ beta 的外层 h 回传；λ 的梯度经求解器回传。
    """
    with torch.no_grad():
        G = h.t() @ h
        Xty = h.t() @ y
        Xty_p = None if y_perm is None else h.t() @ y_perm
    alpha = torch.exp(log_alpha) + tol
    G_reg = G + alpha * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    beta = torch.linalg.solve(G_reg, Xty)
    beta_perm = None if Xty_p is None else torch.linalg.solve(G_reg, Xty_p)
    return beta, beta_perm


class _AdaCapMixin:
    """AdaCap 训练/评估逻辑（四个骨干共用）。"""

    def _setup_shared(self, d: int, lambda_init: float, n_permuted: int = 10):
        self.bn = nn.BatchNorm1d(d, affine=False)  # 官方 TikhonovLayer 前置 BN
        self.log_alpha = nn.Parameter(torch.full((), math.log(lambda_init)))
        self.n_permuted = n_permuted
        # 评估读出（训练集拟合；persistent=False 不进 state_dict，
        # 评估前由训练管道的 fit_adacap_readout 重拟）
        self.register_buffer("beta", torch.zeros(d, 1), persistent=False)

    @torch.no_grad()
    def set_readout(self, H: torch.Tensor, y: torch.Tensor):
        """用训练集表示 H(n,d) 与标签 y(n,1) 拟 β（当前 λ），供评估预测。"""
        beta, _ = ridge_betas(H, y, None, self.log_alpha)
        self.beta.copy_(beta)

    def _train_step(self, h: torch.Tensor, y: torch.Tensor):
        """批内置换对比 + 岭闭式读出，返回 (y_hat, loss)。"""
        y = y.float().unsqueeze(-1)  # (B, 1)
        idx = torch.stack(
            [torch.randperm(y.size(0), device=y.device) for _ in range(self.n_permuted)]
        )  # (P, B)
        y_perm = y[idx].squeeze(-1).t()  # (B, P)：批内打乱标签（保留边际分布）
        beta, beta_perm = ridge_betas(h, y, y_perm, self.log_alpha)
        y_hat = h @ beta  # (B, 1)
        y_hat_perm = h @ beta_perm  # (B, P)
        loss = F.mse_loss(y_hat, y) - F.mse_loss(y_hat_perm, y_perm)
        return y_hat, loss


class AdaCap(_AdaCapMixin, nn.Module):
    """MLP 编码器 + AdaCap（Tikhonov 闭式输出 + 置换对比损失）。"""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: tuple[int, ...] = (512, 256),
        num_classes: int = 1,
        dropout: float = 0.2,
        lambda_init: float = 100.0,
        n_permuted: int = 10,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        self._setup_shared(hidden_dims[-1], lambda_init, n_permuted)

    def encode(self, x):
        """BN 后的隐藏表示（fit_adacap_readout 用，eval 模式）。"""
        return self.bn(self.encoder(flatten(x)))

    def forward(self, x, labels=None, stochastic=True):
        """训练返回 (y_hat, ada_loss)；评估返回 (h @ beta, None)。"""
        h = self.bn(self.encoder(flatten(x)))
        if not self.training:
            return h @ self.beta, None
        return self._train_step(h, labels)
