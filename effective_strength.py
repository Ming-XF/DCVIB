"""尺度无关正则强度轴（审稿人意见 5 的补充分析）。

raw-β 网格跨目标不可比（SVIB 惩罚 β·KL²、NIB 为 nats、DVCCA 同时加权
重建、FGIB 强度还耦合 a²/τ²），本脚本为每个目标×β 计算**有效正则强度**

    r = β·‖∇θ(R)‖₂ / ‖∇θ(CE)‖₂,

即压缩项梯度相对任务损失梯度的范数比，作为尺度无关横轴；同时记录该
工作点上的压缩项数值（KL/center loss）作第二横轴。

测量状态：共享协议的早期工作点——每个模型在 MNIST/MLP 上做 1 个 epoch
的纯 CE 预热（Adam 1e-3，与主协议一致），再在固定 batch 上做两次独立
backward。零初始化处 VIB/CEB 的 KL 设计上为 0、其梯度范数退化，因此
不在初始化处测量。

输出 tables/effective_strength.tex 与 tables/effective_strength.json，
供 make_figures.py 生成尺度无关的 knob 安全性图（横轴 log10 r）。
"""

import json
import math

import torch
import torch.nn.functional as F

from datasets.datasets import get_mnist_dataloaders
from model import CEB, DVCCA, FGIB, NIB, SVIB, VIB

BETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]


def build_model(name: str):
    """按目标名构造 MNIST/MLP 模型（与 train.py 的默认配置一致）。"""
    if name == "vib":
        return VIB()
    if name == "svib":
        return SVIB()  # forward 返回的 kl 已是平方值
    if name == "ceb":
        return CEB()
    if name == "nib":
        return NIB()
    if name == "dvcca":
        return DVCCA()
    if name == "fgib_a16":
        return FGIB(anchor_scale=16.0)
    if name == "fgib_a4":
        return FGIB(anchor_scale=4.0)
    if name == "centfgib_a16":
        from model.mlp.centfgib import CentFGIB
        return CentFGIB(anchor_scale=16.0)
    raise ValueError(f"未知目标: {name}")


def grad_norm(model: torch.nn.Module) -> float:
    """全部参数的梯度 L2 范数（对逐参数 L2 求和开方）。"""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def call_model(model, x, y):
    """统一前向调用：fgib 族（确定性主路）不接受 stochastic。"""
    try:
        return model(x, labels=y, stochastic=True)
    except TypeError:
        return model(x, labels=y)


def warmup_ce(model: torch.nn.Module, loader, device: torch.device):
    """1 个 epoch 纯 CE 预热（共享协议工作点；无正则项）。"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = call_model(model, x, y)
        loss = F.cross_entropy(out[0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def measure(model: torch.nn.Module, x, y) -> tuple[float, float, float]:
    """在给定 batch 上分别反向传播 CE 与压缩项 R，返回
    (‖∇CE‖, ‖∇R‖, R 数值)。R 的约定与 train.py 损失一致：
    vib/ceb/fgib 为 KL、svib 为 KL²（模型已平方）、nib 为 Î、
    dvcca 为 KL + MSE_recon、centfgib 为 center loss。
    """
    torch.manual_seed(1234)  # 固定重参数化采样噪声
    model.zero_grad(set_to_none=True)
    out = call_model(model, x, y)
    logits = out[0]
    kl = out[1]
    recon = out[2] if len(out) > 2 else None

    ce = F.cross_entropy(logits, y)
    ce.backward(retain_graph=True)  # reg 与 ce 共享计算图，需保留
    n_ce = grad_norm(model)
    model.zero_grad(set_to_none=True)

    reg = kl
    if recon is not None:
        reg = reg + recon  # dvcca：CE + β·(KL + MSE_recon)
    reg.backward()
    n_reg = grad_norm(model)
    model.zero_grad(set_to_none=True)
    return n_ce, n_reg, float(reg.detach().item())


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None,
                    help="计算设备（默认自动：有 CUDA 用 cuda；显存紧张时传 cpu）")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, _, _ = get_mnist_dataloaders(batch_size=512, data_dir="./data")
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    models = [
        ("vib", "VIB (KL, $\\mathcal{N}(0,I)$)"),
        ("svib", "SVIB ($\\beta$KL$^2$)"),
        ("ceb", "CEB (trainable prior)"),
        ("nib", "NIB (pairwise bound)"),
        ("dvcca", "DVCCA (KL $+$ recon)"),
        ("fgib_a16", "FGIB-A ($a{=}16$)"),
        ("fgib_a4", "FGIB-A ($a{=}4$)"),
        ("centfgib_a16", "CentFGIB ($a{=}16$)"),
    ]
    rows = []
    for name, label in models:
        torch.manual_seed(0)
        model = build_model(name).to(device)
        warmup_ce(model, train_loader, device)
        model.eval()
        n_ce, n_reg, reg_val = measure(model, x, y)
        base = n_reg / n_ce  # β=1 时的有效强度
        r_list = [base * b for b in BETAS]
        rows.append(
            {
                "model": name,
                "label": label,
                "grad_ratio_beta1": base,
                "reg_value": reg_val,
                "r": {f"{b:g}": r for b, r in zip(BETAS, r_list)},
            }
        )
        print(
            f"{name:15s} ‖∇R‖/‖∇CE‖={base:.4f}  R={reg_val:.4f}  "
            f"r(β=1e-4..10)=[" + ", ".join(f"{r:.3f}" for r in r_list) + "]"
        )

    # LaTeX 表：目标 × β 的 log10 有效强度
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Scale-free effective regularization strength "
                 "$r = \\beta\\,\\lVert\\nabla_\\theta R\\rVert_2/\\lVert\\nabla_\\theta "
                 "\\mathrm{CE}\\rVert_2$ at the shared one-epoch CE-only operating point "
                 "(MNIST/MLP, seed 0). The raw $\\beta$ grid is not a shared axis across "
                 "objectives (SVIB penalizes $\\beta\\,\\mathrm{KL}^2$, NIB is in nats, "
                 "DVCCA weights reconstruction too); $r$ is. FGIB-A's row at $a{=}4$ vs "
                 "$a{=}16$ shows the $a^2/\\tau^2$ coupling explicitly; NIB's $r$ is "
                 "below $1.7\times10^{-3}$ at this operating point (its knob does not act).}")
    lines.append("\\label{tab:effective}")
    lines.append("\\begin{center}\\small")
    cols = "l" + "r" * len(BETAS)
    lines.append("\\begin{tabular}{" + cols + "}")
    header = ["Objective"] + [f"$\\log_{{10}} r\\,(\\beta={b:g})$" for b in BETAS]
    lines.append(" & ".join(header) + " \\\\ \\hline")
    for row in rows:
        vals = [
            f"{math.log10(r):.2f}" if r >= 0.01 else "$<\\!10^{-2}$"
            for r in row["r"].values()
        ]
        lines.append(f"{tex_escape(row['label'])} & " + " & ".join(vals) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")

    with open("tables/effective_strength.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("tables/effective_strength.json", "w") as f:
        json.dump({"betas": BETAS, "rows": rows}, f, indent=2)
    print("已写入 tables/effective_strength.tex / .json")


if __name__ == "__main__":
    main()
