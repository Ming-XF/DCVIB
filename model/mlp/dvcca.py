"""DVCCA（监督 β-DVCCA）模型定义。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_hidden_layers, flatten, kl_divergence, reparameterize


class DVCCA(nn.Module):
    """监督 β-DVCCA（DVMIB 框架的监督退化形式，Abdelaleem et al., JMLR 2025）。

    论文的 β-DVCCA（Eq. 18）为 L = Ĩ_E(X;Z) − β(Ĩ_D(Y;Z) + Ĩ_D(X;Z))：压缩 X
    到 Z，并从 Z 同时"重建"X 和 Y。监督设置下 Y 为标签，Ĩ_D(Y;Z) 即分类
    交叉熵、Ĩ_D(X;Z) 即输入重建（高斯解码器 → MSE），故本实现 = VIB 结构 +
    一个重建解码器，损失 = CE + β·(KL + MSE_recon)（与项目 CE + β·KL 约定
    一致，β 同时加权压缩与重建两项，与论文 β 互为倒数重参数化）。
    forward 返回 (logits, kl, recon_loss)。
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256),
        z_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            *build_hidden_layers(input_dim, hidden_dims, dropout)
        )
        hidden_dim = hidden_dims[-1]

        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        self.classifier = nn.Linear(z_dim, num_classes)
        # 重建解码器：z → 输入空间（高斯解码器，MSE 即 −ln q(x|z) 的常数倍）
        self.decoder = nn.Sequential(
            *build_hidden_layers(z_dim, tuple(reversed(hidden_dims)), dropout),
            nn.Linear(hidden_dims[0], input_dim),
        )

        # 置零初始化：训练开始时 sigma = 1，KL 接近 0
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.zeros_(self.logvar_head.bias)

    def forward(self, x, labels=None, stochastic=True):
        """返回 (logits, kl, recon_loss)；先验为 N(0, I)。labels 参数仅为统一接口。"""
        x_flat = flatten(x)
        h = self.encoder(x_flat)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = reparameterize(mu, logvar, stochastic)
        logits = self.classifier(z)
        recon = F.mse_loss(self.decoder(z), x_flat)

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)
        kl = kl_divergence(mu, logvar, mu_p, logvar_p)
        return logits, kl, recon
