"""公共工具函数。"""

import math

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


def build_anchor_prior(
    z_dim: int, num_classes: int, anchor_scale: float = 4.0, anchor_var: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 FGIB 固定类条件锚点先验表，返回 (prior_mu, prior_logvar)。

    对随机矩阵做 QR 分解得到 K 个正交方向，缩放 anchor_scale 作为各类先验
    均值（(num_classes, z_dim)），方差固定为 anchor_var。要求类别数不超过
    z 维度。返回的均值/对数方差表以 register_buffer 挂到模型上、不参与梯度。
    """
    assert num_classes <= z_dim, "固定锚点先验要求类别数不超过 z 维度"
    g = torch.randn(z_dim, num_classes)
    q, _ = torch.linalg.qr(g)  # q 为 (z_dim, num_classes) 正交列
    prior_mu = (q * anchor_scale).t().contiguous()  # (num_classes, z_dim)
    prior_logvar = torch.full((num_classes, z_dim), math.log(anchor_var))
    return prior_mu, prior_logvar


class ContinuousAnchorPrior(nn.Module):
    """FGIB 回归用的固定连续锚点先验（随机傅里叶特征，零参数）。

    mu_p(y) = anchor_scale · sqrt(2/z_dim) · cos(2π·ω·y + b)，ω ~ N(0, bandwidth²)、
    b ~ U[0, 2π)，初始化时随机生成一次并固定为 buffer（受 run seed 控制，与
    分类版 QR 锚点一致）。||φ(y)|| ≈ 1，锚点近似落在半径 anchor_scale 的球面上，
    训练起点 KL ≈ 0.5·anchor_scale²，anchor_scale 语义与分类版一致；
    anchor_scale=0 时退化为各类相同的 N(0, I) 先验。
    """

    def __init__(
        self,
        z_dim: int,
        anchor_scale: float = 4.0,
        anchor_var: float = 1.0,
        bandwidth: float = 2.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.anchor_scale = anchor_scale
        omega = torch.randn(z_dim) * bandwidth
        bias = torch.rand(z_dim) * (2.0 * math.pi)
        self.register_buffer("omega", omega)
        self.register_buffer("bias", bias)
        self.register_buffer("anchor_logvar", torch.full((1, z_dim), math.log(anchor_var)))

    def forward(self, y):
        """y: (B, 1) 归一化连续标签 → ((B, z_dim) mu_p, (B, z_dim) logvar_p)。"""
        feat = math.sqrt(2.0 / self.z_dim) * torch.cos(
            2.0 * math.pi * self.omega * y + self.bias
        )
        mu_p = self.anchor_scale * feat
        return mu_p, self.anchor_logvar.expand_as(mu_p)
