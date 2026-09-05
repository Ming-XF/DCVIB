"""合成对齐试验论文图（codex_output.txt 方案第三节）：D̂–β scaling 诊断。

所有点 mean±std over seeds；series 按 (frame, a) 分离、绝不合并；失败行
（fail 非空）不计入均值并打印数量。C_G 水平线/参考斜率来自 cg csv
（旋转不变，每 a 一个值 + MC 标准误差）。β·D̂ > C_G 的点如实画上（不删除）。

输出 output/synthetic_align/：
    fig_dhat_beta.png       log-log D̂ vs β + 参考斜率 1/β（C_G/β 虚线）
    fig_beta_dhat.png       β·D̂ vs β + C_G 水平线
    fig_dhat_invbeta.png    D̂ vs 1/β 线性拟合（斜率+R²，仅诊断、非定理证明）
    fig_acc_beta.png        acc_det / acc_mc vs β
另打印 LaTeX 表格（每 β：D̂、β·D̂、CE、E[KL]、acc，mean±std）。

用法：
    python synthetic_align_plot.py [--csv output/synthetic_align/synthetic_align.csv]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compression_plot import SERIES_COLORS
from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "synthetic_align"

SERIES = [
    ("fixed", 6.0), ("fixed", 12.0), ("trainable", 6.0), ("trainable", 12.0),
]
SERIES_COLOR = {  # 与 paper 其余图一致的稳定配色（8 色表内取 4）
    ("fixed", 6.0): "#2a78d6",
    ("fixed", 12.0): "#eb6834",
    ("trainable", 6.0): "#4f9a5f",
    ("trainable", 12.0): "#9a6ac4",
}


def load(csv_path):
    """→ {frame: {a: {beta: {col: [values]}}}}，跳过 fail 行并计数。"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    fails = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fail"] or not r["D_hat_test"]:
                fails += 1
                continue
            key = data[r["frame_setting"]][float(r["anchor_scale"])][float(r["beta"])]
            for col in ["D_hat_test", "beta_D_hat_test", "CE_test", "KL_total_test",
                        "objective_test", "acc_det", "acc_mc"]:
                key[col].append(float(r[col]))
    return data, fails


def load_cg(cg_path):
    """→ {a: (C_G, se)}。"""
    cg = {}
    if cg_path.exists():
        with open(cg_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cg[float(r["anchor_scale"])] = (float(r["C_G"]), float(r["mc_se"]))
    return cg


def series_points(data, col):
    """→ [(label, frame, a, [(beta, mean, std), ...])]，β 升序。"""
    out = []
    for frame, a in SERIES:
        m = data.get(frame, {}).get(a)
        if not m:
            continue
        pts = []
        for b in sorted(m):
            vals = m[b][col]
            pts.append((b, sum(vals) / len(vals),
                        (sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                        if len(vals) > 1 else 0.0))
        if pts:
            out.append((f"{frame} frame, a={a:g}", frame, a, pts))
    return out


def plot_base(xlabel, ylabel, path, xlog, ylog=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax, path


def finish(ax, path):
    ax.legend(fontsize=8, frameon=False)
    fig_path = OUT_DIR / path
    ax.figure.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(ax.figure)
    print(f"论文图已保存：{fig_path}")


def fig_dhat_beta(data, cg):
    """log-log D̂ vs β + C_G/β 参考虚线（斜率 1/β）。"""
    fig, ax, path = plot_base("$\\beta$", "$\\hat{D}(Y)$ (nats, test)",
                              "fig_dhat_beta.png", True, True)
    for label, frame, a, pts in series_points(data, "D_hat_test"):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=errs, label=label, color=SERIES_COLOR[(frame, a)],
                    marker="o", markersize=4, linewidth=1.6, capsize=2)
        if a in cg:
            cg_val, _ = cg[a]
            ref = [cg_val / b for b in xs]
            ax.plot(xs, ref, linestyle="--", linewidth=1.0, alpha=0.6,
                    color=SERIES_COLOR[(frame, a)])
    finish(ax, path)


def fig_beta_dhat(data, cg):
    """β·D̂ vs β + C_G 水平虚线。"""
    fig, ax, path = plot_base("$\\beta$", "$\\beta\\,\\hat{D}(Y)$ (nats, test)",
                              "fig_beta_dhat.png", True)
    for label, frame, a, pts in series_points(data, "beta_D_hat_test"):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=errs, label=label, color=SERIES_COLOR[(frame, a)],
                    marker="o", markersize=4, linewidth=1.6, capsize=2)
    for a, (cg_val, se) in cg.items():
        ax.axhline(cg_val, linestyle="--", linewidth=1.0, alpha=0.6,
                   color=SERIES_COLOR[("fixed", a)])
        ax.text(ax.get_xlim()[1] * 0.98, cg_val * 1.15, f"$C_G(a={a:g})$",
                fontsize=7, ha="right", va="bottom", color="#52514e")
    finish(ax, path)


def fig_dhat_invbeta(data):
    """D̂ vs 1/β 线性拟合（仅 scaling 诊断，不称定理证明）。"""
    fig, ax, path = plot_base("$1/\\beta$", "$\\hat{D}(Y)$ (nats, test)",
                              "fig_dhat_invbeta.png", False)
    for label, frame, a, pts in series_points(data, "D_hat_test"):
        xs = [1.0 / p[0] for p in pts]
        ys = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=errs, label=label, color=SERIES_COLOR[(frame, a)],
                    marker="o", markersize=4, linewidth=1.6, capsize=2)
        # 线性拟合 + R²（诊断用）
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
        intercept = my - slope * mx
        ss_res = sum((ys[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
        ss_tot = sum((ys[i] - my) ** 2 for i in range(n))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"[拟合诊断] {label:26s} slope={slope:.4f} R²={r2:.3f}")
    finish(ax, path)


def fig_acc_beta(data):
    """acc_det / acc_mc vs β（确认 scaling 不以任务崩溃为代价）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, col, title in [(axes[0], "acc_det", "deterministic ($z=\\mu$)"),
                           (axes[1], "acc_mc", "stochastic (MC-10)")]:
        for label, frame, a, pts in series_points(data, col):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, label=label, color=SERIES_COLOR[(frame, a)],
                        marker="o", markersize=4, linewidth=1.6, capsize=2)
        ax.set_xscale("log")
        ax.set_xlabel("$\\beta$")
        ax.set_ylabel("test accuracy")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
        ax.set_facecolor("white")
        ax.legend(fontsize=7, frameon=False)
    fig.patch.set_facecolor("white")
    fig_path = OUT_DIR / "fig_acc_beta.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"论文图已保存：{fig_path}")


def print_table(data, cg):
    """LaTeX 可用表格：每 (frame, a, β) 的 D̂/βD̂/CE/E[KL]/acc mean±std。"""
    print("\n% LaTeX 表格（mean±std over seeds）：")
    print(r"\begin{tabular}{l l l c c c c c}")
    print(r"\hline")
    print(r"frame & $a$ & $\beta$ & $\hat D$ & $\beta\hat D$ & CE & $\E[\KL]$ & acc (MC) \\")
    print(r"\hline")
    for frame, a in SERIES:
        m = data.get(frame, {}).get(a)
        if not m:
            continue
        for b in sorted(m):
            row = [f"{frame}", f"{a:g}", f"{b:g}"]
            for col in ["D_hat_test", "beta_D_hat_test", "CE_test", "KL_total_test", "acc_mc"]:
                vals = m[b][col]
                mean = sum(vals) / len(vals)
                std = ((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                       if len(vals) > 1 else 0.0)
                row.append(f"${mean:.3f}_{{\\pm {std:.3f}}}$")
            print(" & ".join(row) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    if cg:
        print("% C_G：" + "，".join(f"a={a:g}: {v[0]:.3e}±{v[1]:.1e}" for a, v in sorted(cg.items())))


def main():
    parser = argparse.ArgumentParser(description="合成对齐试验论文图")
    parser.add_argument("--csv", default=str(OUT_DIR / "synthetic_align.csv"))
    parser.add_argument("--cg", default=str(OUT_DIR / "cg.csv"))
    args = parser.parse_args()
    data, fails = load(Path(args.csv))
    if not data:
        print(f"[跳过] 无有效数据：{args.csv}")
        return
    print(f"数据：{args.csv}（失败行 {fails}，已跳过）")
    cg = load_cg(Path(args.cg))
    _setup_rc()
    fig_dhat_beta(data, cg)
    fig_beta_dhat(data, cg)
    fig_dhat_invbeta(data)
    fig_acc_beta(data)
    print_table(data, cg)


if __name__ == "__main__":
    main()
