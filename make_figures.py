"""生成论文图（matplotlib → PDF，供 paper.tex \\includegraphics）。

图 1  figures/beta_accuracy.pdf：MNIST/MLP 上测试 Acc 随 β（对数轴）的曲线。
    (a) 七目标：Det / VIB / SVIB / NIB / CEB / DVCCA / FGIB(a=16)；
    (b) 消融变体：Det / FGIB(a=16) / DCEB / TAFGIB / CentFGIB / frozen-A / A=I。
图 2  figures/diagnostics.pdf：(a) FGIB κ(AᵀA)（log10）随 epoch 的轨迹（各 β）；
    (b) CEB 后验/先验与 VIB 后验的 mean logvar 随 β 的变化（P3 测量）。

调色板为 dataviz 参考调色板 7 槽（已验证通过），按固定槽位顺序分配；
Det 为中性灰虚线参考线（纹理编码，非分类槽位）；FGIB 加粗强调（与论文一致）；
低对比槽位（黄/水蓝/品红）按 relief 规则加末端直接标签（文字用墨色、不着系列色）。

用法：python make_figures.py（需 MNIST 主实验与 tune_results_ablation 全部完成）
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_tables as mt

# 参考调色板 7 槽（固定顺序），Det 为中性灰参考线
SLOT = {
    1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100",
    5: "#e87ba4", 6: "#008300", 7: "#4a3aa7",
}
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"
DET_GRAY = "#8a8a86"

BETAS = mt.BETAS
# 纯 LaTeX 片段（不含 $ 定界符），由调用方按需包裹
BETA_LABELS = [r"10^{-4}", r"10^{-3}", r"10^{-2}", r"10^{-1}", r"1", r"10"]


def series(results_dir, model, anchor=None):
    """(beta, test acc) 列表；基线 det 返回单个值。"""
    if model == "mlp":
        from make_ablation import parse_log
        rec = parse_log(os.path.join("tune_results", "mnist_mlp", "train.log"))
        return rec["test"]["Acc"][0]
    from make_ablation import series_test_acc
    accs, _ = series_test_acc(results_dir, model, anchor)
    return [accs[b] for b in BETAS]


def style_axes(ax, ylim=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", axis="both", color="#d9d8d3", linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=INK2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#b5b4ae")
    if ylim is not None:
        ax.set_ylim(*ylim)
    # 横轴为 β 网格（对数尺度上的等间距 6 点），刻度显示 β 值
    ax.set_xticks(list(range(len(BETAS))), ["$" + l + "$" for l in BETA_LABELS])
    ax.set_xlabel(r"$\beta$", color=INK2, fontsize=10)


def plot_series(ax, xs, ys, color, lw, label, ls="-", marker="o"):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker=marker, markersize=6,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    # 末端直接标签（relief 规则：低对比槽位必须有可见标签；全部标签都用墨色）
    ax.annotate(label, xy=(xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=INK, va="center", zorder=4)


def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    xs = range(len(BETAS))

    # (a) 七目标
    ax = axes[0]
    det = series("tune_results", "mlp")
    ax.plot(xs, [100 * det] * 6, color=DET_GRAY, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.annotate("Det.", xy=(xs[-1], 100 * det), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=INK, va="center")
    for slot, (label, model, anchor) in enumerate([
        ("VIB", "vib", None), ("SVIB", "svib", None), ("NIB", "nib", None),
        ("CEB", "ceb", None), ("DVCCA", "dvcca", None), ("FGIB", "fgib", 16.0),
    ], start=1):
        ys = [100 * v for v in series("tune_results", model, anchor)]
        plot_series(ax, xs, ys, SLOT[slot], 3.0 if label == "FGIB" else 2.0, label)
    style_axes(ax, ylim=(0, 105))
    ax.set_ylabel("Test accuracy (%)", color=INK2, fontsize=10)
    ax.set_title("(a) Objectives on MNIST/MLP", color=INK, fontsize=10)

    # (b) 消融变体
    ax = axes[1]
    det = series("tune_results", "mlp")
    ax.plot(xs, [100 * det] * 6, color=DET_GRAY, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.annotate("Det.", xy=(xs[-1], 100 * det), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=INK, va="center")
    for slot, (label, model, anchor, resdir) in enumerate([
        ("FGIB", "fgib", 16.0, "tune_results"),
        ("DCEB", "dceb", None, "tune_results_ablation"),
        ("TAFGIB", "tafgib", 16.0, "tune_results_ablation"),
        ("CentFGIB", "centfgib", 16.0, "tune_results_ablation"),
        ("frozen $A$", "fgib", 16.0, "tune_results_ablation/freezea"),
        ("$A{=}I$", "fgib", 16.0, "tune_results_ablation/aid"),
    ], start=1):
        ys = [100 * v for v in series(resdir, model, anchor)]
        plot_series(ax, xs, ys, SLOT[slot], 3.0 if label == "FGIB" else 2.0, label)
    style_axes(ax, ylim=(0, 105))
    ax.set_title("(b) Confound ablations on MNIST/MLP ($a{=}16$)", color=INK, fontsize=10)

    fig.tight_layout()
    fig.savefig("figures/beta_accuracy.pdf", facecolor=SURFACE)
    print("已生成 figures/beta_accuracy.pdf")


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # (a) FGIB κ(AᵀA) 轨迹（各 beta，第一个 run）
    ax = axes[0]
    for slot, b in enumerate(["0.0001", "0.001", "0.01", "0.1", "1", "10"], start=1):
        path = glob.glob(f"tune_results_ablation/p3_diag/mnist_mlp_fgib_beta_{b}_anchor_16/train.log")
        if not path:
            continue
        traj = []
        run, last = 0, 0
        for line in open(path[0]).read().splitlines():
            if "===== Run" in line:
                run += 1
            m = re.search(r"Epoch\s+(\d+)/", line)
            if m:
                last = int(m.group(1))
            m = re.search(r"kappa_log10=([\d.\-]+)", line)
            if run == 1 and m:
                traj.append((last, float(m.group(1))))
        if traj:
            b_label = "$\\beta = " + BETA_LABELS[slot - 1] + "$"
            ax.plot([e for e, _ in traj], [v for _, v in traj], color=SLOT[slot], lw=2.0,
                    label=b_label)
            ax.annotate(b_label, xy=(traj[-1][0], traj[-1][1]),
                        xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#d9d8d3", linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=INK2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#b5b4ae")
    ax.set_xlabel("Epoch", color=INK2, fontsize=10)
    ax.set_ylabel(r"$\log_{10}\kappa(A^{\top}A)$", color=INK2, fontsize=10)
    ax.set_title("(a) FGIB side-head conditioning over training", color=INK, fontsize=10)

    # (b) P3 方差测量：mean logvar 随 beta
    ax = axes[1]
    xs = range(len(BETAS))

    def diag_stat(model, field):
        out = []
        for b in ["0.0001", "0.001", "0.01", "0.1", "1", "10"]:
            path = glob.glob(f"tune_results_ablation/p3_diag/mnist_mlp_{model}_beta_{b}/train.log")
            if not path:
                continue
            diags = []
            cur = None
            for line in open(path[0], errors="ignore").read().splitlines():
                if "===== Run" in line:
                    if cur:
                        diags.append(cur)
                    cur = None
                if "Diag " in line:
                    cur = line
            if cur:
                diags.append(cur)
            vals = [float(re.search(field + r"=([\d.\-]+)", d).group(1)) for d in diags]
            out.append(sum(vals) / len(vals))
        return out

    ax.axhline(0, color=DET_GRAY, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.annotate("init. ($\\log\\sigma^2{=}0$)", xy=(xs[-1], 0), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=INK, va="center")
    for slot, (label, model, field) in enumerate([
        ("CEB posterior", "ceb", "logvar_q_mean"),
        ("CEB prior", "ceb", "logvar_p_mean"),
        ("VIB posterior", "vib", "logvar_q_mean"),
    ], start=1):
        ys = diag_stat(model, field)
        plot_series(ax, xs, ys, SLOT[slot], 2.0, label)
    ax.set_ylim(-1.5, 1.5)
    style_axes(ax, ylim=(-1.5, 1.5))
    ax.set_ylabel(r"mean $\log\sigma^2$ (final epoch)", color=INK2, fontsize=10)
    ax.set_title("(b) P3 variance-race measurement", color=INK, fontsize=10)

    fig.tight_layout()
    fig.savefig("figures/diagnostics.pdf", facecolor=SURFACE)
    print("已生成 figures/diagnostics.pdf")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    fig1()
    fig2()
