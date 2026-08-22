"""公共工具函数。"""

import torch
import torch.nn as nn


def flatten(x: torch.Tensor) -> torch.Tensor:
    """把图像输入 (B, C, H, W) 展平为 (B, C*H*W)。"""
    if x.dim() == 4 or x.dim() == 3:
        x = x.flatten(1)
    return x


def build_hidden_layers(
    input_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.0
) -> list[nn.Module]:
    """构造隐藏层序列：每层 Linear + ReLU + Dropout，不含输出层。"""
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return layers


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool):
    """重参数化采样：stochastic 时 z = mu + sigma*eps，否则 z = mu。"""
    if stochastic:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    return mu


def kl_divergence(mu, logvar, mu_p, logvar_p):
    """对角高斯 KL(q||p) 的闭式解，逐样本求和后对 batch 平均。

    clamp 防止方差过小导致数值发散。
    """
    logvar = logvar.clamp(-10.0, 10.0)
    logvar_p = logvar_p.clamp(-10.0, 10.0)
    var, var_p = logvar.exp(), logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar + (var + (mu - mu_p).pow(2)) / var_p - 1.0)
    return kl.sum(dim=1).mean()
