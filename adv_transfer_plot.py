"""迁移攻击曲线：x = ε（线性）、y = targeted 成功率（%）；L∞ 与 L2 各一张。

数据来自 adv_transfer_eval.py 的 output/adv_mnist/mnist_adv_transfer.csv。
每图 5 条线：CEB β=0.1 / OPB β=25, a=6 / MLP 基线的白盒自攻（实线）+ 两条
迁移 CEB→OPB / OPB→CEB（虚线，颜色随源模型）；ε=0 处补 (0,0) 锚点；成功率
跨 5 run 取均值、精确点不插值、无误差棒（与 fig_adv_targeted 风格一致）。

输出（白底、英文标签、无总标题）：
    paper/figures/fig_adv_transfer_linf_mnist.png
    paper/figures/fig_adv_transfer_l2_mnist.png

用法：
    python adv_transfer_plot.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "output" / "adv_mnist" / "mnist_adv_transfer.csv"
FIG_DIR = ROOT / "paper" / "figures"

CEB_NAME = "mnist_mlp_ceb_beta_0.1"
OPB_NAME = "mnist_mlp_opb_beta_25_anchor_6"
MLP_NAME = "mnist_mlp"

# (csv config, 显示标签, 颜色, 线型)：每条线独立配色、迁移统一虚线
SERIES = [
    (CEB_NAME, "CEB ($\\beta{=}0.1$)", "#2a78d6", "-"),
    (OPB_NAME, "OPB ($\\beta{=}25$, $a{=}6$)", "#eb6834", "-"),
    (MLP_NAME, "MLP (deterministic)", "#8a8880", "-"),
    (f"{CEB_NAME}→{OPB_NAME}", "CEB→OPB (transfer)", "#4f9a5f", "--"),
    (f"{OPB_NAME}→{CEB_NAME}", "OPB→CEB (transfer)", "#9a6ac4", "--"),
]


def load():
    """读 csv → {config: {norm: {eps: [succ per run]}}}。"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["norm"] == "none" or not r["succ"]:
                continue
            data[r["config"]][r["norm"]][float(r["eps"])].append(float(r["succ"]))
    return data


def plot_norm(norm, data):
    """单范数论文图：x = ε（线性），y = targeted 成功率（%）。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y_max = 0.0
    for key, label, color, ls in SERIES:
        eps_vals = sorted(data.get(key, {}).get(norm, {}))
        if not eps_vals:
            continue
        pts = [(0.0, 0.0)] + [(e, sum(data[key][norm][e]) / len(data[key][norm][e]) * 100)
                              for e in eps_vals]
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            label=label, color=color, linestyle=ls,
            marker="o", markersize=4, linewidth=1.6,
        )
        y_max = max(y_max, max(p[1] for p in pts))
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("targeted attack success (%)")
    ax.set_ylim(0.0, max(5.0, y_max * 1.15))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    path = FIG_DIR / f"fig_adv_transfer_{norm}_mnist.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"论文图已保存：{path}")
    return path


def main():
    data = load()
    _setup_rc()
    for norm in ("linf", "l2"):
        for key, label, _, _ in SERIES:
            eps_vals = sorted(data.get(key, {}).get(norm, {}))
            if eps_vals:
                vals = [sum(data[key][norm][e]) / len(data[key][norm][e]) * 100 for e in eps_vals]
                print(f"  {label:22s} succ[" + ", ".join(f"{v:.1f}" for v in vals) + "]")
        plot_norm(norm, data)


if __name__ == "__main__":
    main()
