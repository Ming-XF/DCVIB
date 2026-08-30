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


class CosineClassifier(nn.Module):
    """固定温度 cosine 分类器：logits = normalize(x) @ normalize(W).T / T。

    fgib_theory.tex 的 rem:c-scale(c) 指出 sweep 命题（prop:sweep）要求
    分类器尺度 c 固定，而默认 nn.Linear 无约束。本分类器对权重行做 L2
    归一化（无 bias），温度 T 固定（默认 1.0、不可学习），按构造满足
    固定 c 条件；特征侧也归一化（normalized features）。用于理论验证
    实验（--cosine-classifier）。
    """

    def __init__(self, in_features: int, out_features: int, temp: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.temp = temp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.nn.functional.normalize(self.weight, dim=1)
        x = torch.nn.functional.normalize(x, dim=1)
        return x @ w.t() / self.temp


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


def nib_mi_upper_bound(m: torch.Tensor, noise_var) -> torch.Tensor:
    """NIB 压缩项的非参数上界（NIB 论文 Eq. 16，同方差噪声 Σ=σ²I）。

    M = f(X) + ε（ε~N(0, noise_var·I)）时，混合分布各高斯分量之间的成对 KL
    化简为 ‖fᵢ−fⱼ‖²/(2·noise_var)，I(X;M) ≤ Î = −(1/N)Σᵢ ln[(1/N)Σⱼ
    exp(−‖fᵢ−fⱼ‖²/(2·noise_var))]（本实现用自然对数，单位为 nat；论文用
    log2/bit，扫参时注意 β 量纲差 ln2 倍）。noise_var 为模型的可学习参数
    （log 空间，初始 σ²=1 = 论文初始值）；初始 σ² 过小时 exp 项趋 0、
    Î→ln N 且梯度指数级消失（论文 Section IVA 的讨论）。
    """
    n = m.size(0)
    dists2 = torch.cdist(m, m).pow(2) / (2.0 * noise_var)
    logsum = torch.logsumexp(-dists2, dim=1)
    return -(logsum - math.log(n)).mean()


def build_anchor_prior(
    z_dim: int,
    num_classes: int,
    anchor_scale: float = 4.0,
    anchor_var: float = 1.0,
    geometry: str = "qr",
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 FGIB 固定类条件锚点先验表，返回 (prior_mu, prior_logvar)。

    锚点框架由 geometry 选择（几何消融，审稿人要求比较"同一表示上的
    orthogonal / simplex-ETF / 随机归一化"正则）：
    - "qr"：对随机矩阵做 QR 分解得到 K 个正交方向（默认，互余弦 0）；
    - "etf"：单纯形/ETF 框架（单位范数、互余弦 -1/(K-1) 的神经坍缩配置，
      嵌入 z_dim 后经随机正交旋转，几何不变）；
    - "random"：随机单位方向（不正交化）。
    各方向缩放 anchor_scale 作为各类先验均值（(num_classes, z_dim)），
    方差固定为 anchor_var。要求类别数不超过 z 维度。返回的均值/对数方差
    表以 register_buffer 挂到模型上、不参与梯度。
    """
    assert num_classes <= z_dim, "固定锚点先验要求类别数不超过 z 维度"
    if geometry == "qr":
        g = torch.randn(z_dim, num_classes)
        q, _ = torch.linalg.qr(g)  # q 为 (z_dim, num_classes) 正交列
        prior_mu = (q * anchor_scale).t().contiguous()  # (num_classes, z_dim)
    elif geometry == "etf":
        assert num_classes > 1, "ETF 框架要求类别数大于 1"
        frame = torch.eye(num_classes) - torch.ones(num_classes, num_classes) / num_classes
        frame *= (num_classes / (num_classes - 1)) ** 0.5  # 行范数 1、互余弦 -1/(K-1)
        embed = torch.zeros(num_classes, z_dim)
        embed[:, :num_classes] = frame  # K 维单纯形嵌入 z_dim
        rot, _ = torch.linalg.qr(torch.randn(z_dim, z_dim))  # 随机正交旋转（几何不变）
        prior_mu = anchor_scale * (embed @ rot).contiguous()  # (num_classes, z_dim)
    elif geometry == "random":
        g = torch.randn(z_dim, num_classes)
        prior_mu = (anchor_scale * g / g.norm(dim=0, keepdim=True)).t().contiguous()
    else:
        raise ValueError(f"未知锚点几何: {geometry}")
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
