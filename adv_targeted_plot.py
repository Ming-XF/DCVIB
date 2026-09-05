"""目标攻击鲁棒性曲线（CEB 论文 Fig. 4 底栏范式）：x = β（对数刻度）、
y = targeted PGD 成功率（%）；CEB 1 条线 + OPB 3 条线（a = 1/6/12）+
确定性 MLP 基线 1 条横线（无 β，虚线灰色，CEB 论文 Det. 同款参照）。
L∞ 与 L2 各一张论文图。

数据来自 adv_eval.py --targeted 的输出 output/adv_mnist/mnist_adv_targeted.csv
（长表 config/run/norm/eps/acc/succ，succ 为 targeted 成功率、条件化在干净正确
且非目标类的样本上）。每范数取 csv 中该范数最大的 ε 画图（本试验每范数单 ε：
L∞ 0.3、L2 6.0——L2 取 2.0 时攻击预算够不着目标类区域、成功率贴基率地板；
取 12.0 时基线打到 100% 天花板），成功率跨 run 取均值、精确点不插值、无误差
棒（与 fig_beta_* 风格一致）。

横轴截断按系列各自的干净精度有效区间（valid_beta_max_map，口径与 succ 相同
的 MC-10 随机路径，失效阈值统一 0.9）：CEB β=0.5 干净精度 0.857±0.22 低于
阈值、β≥1 到随机水平 → CEB 只画 β ≤ 0.1；OPB a=1 β≥5 精度逐步降至
0.59–0.84（后验 σ²→固定先验方差 τ²=1、半径 1 锚点上采样噪声淹没类别结构，
μ 路径不受影响但 adv 协议用 MC-10），β=5 的 0.836 跌破 0.9 → a=1 只画
β ≤ 1；a=6/12 全程精度有效、扫满网格。各系列超出其有效区间的 succ≈0 是
坍缩瓶颈阻断攻击梯度的平凡鲁棒性，不构成公平对比；基线横线跨整个区间。

输出（白底、英文标签、无总标题，ε 标注在左上角）：
    paper/figures/fig_adv_targeted_linf_mnist.png
    paper/figures/fig_adv_targeted_l2_mnist.png

用法：
    python adv_targeted_plot.py
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compression_eval import parse_combo_dir
from compression_plot import SERIES_COLORS
from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "output" / "adv_mnist" / "mnist_adv_targeted.csv"
FIG_DIR = ROOT / "paper" / "figures"

BASELINE_LABEL = "Deterministic MLP"
BASELINE_COLOR = "#8a8880"


def load_succ():
    """读 targeted csv → {norm: {eps: {(model, anchor): {beta: [succ per run]}}}}。

    基线（{dataset}_{backbone} 目录名）无 β，beta 用哨兵值 0.0 存储。
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["norm"] == "none" or not r["succ"]:
                continue
            info = parse_combo_dir(r["config"])
            if info is not None and info[0] == "mnist":
                _, _, model, beta, anchor = info
                key = (model, f"{anchor:g}" if anchor is not None else "")
                beta = float(beta)
            else:
                m = re.match(r"^mnist_(mlp|cnn|gcn|rnn)$", r["config"])
                if not m:
                    continue
                key, beta = (m.group(1), ""), 0.0
            data[r["norm"]][float(r["eps"])][key][beta].append(float(r["succ"]))
    return data


# 失效判定的干净精度阈值：统一 0.9（CEB 与 OPB a=1 同用；a=6/12 全程
# ≥ 0.98，不受阈值影响）
DEFAULT_VALIDITY_THRESHOLD = 0.9


def valid_beta_max_map():
    """{(model, anchor): 最大有效 β}：adv 口径（MC-10 随机路径，norm=none 行）
    干净精度仍 ≥ 失效阈值的最大 β。

    阈值统一 0.9（DEFAULT_VALIDITY_THRESHOLD）：CEB β=0.5 精度 0.857±0.22
    低于 0.9、β≥1 随机 → 有效区间 β ≤ 0.1；OPB a=1 β≥5 精度逐步降至
    0.59–0.84，β=5 的 0.836 跌破 0.9 → 有效区间 β ≤ 1（该区间精度下降的
    机制为后验 σ²→固定先验方差 τ²=1、半径 1 锚点上采样噪声淹没类别结构，
    μ 路径不受影响但 adv 协议用 MC-10）；a=6/12 锚点间距大、全程 ≥ 0.98 →
    全网格。
    """
    accs = defaultdict(lambda: defaultdict(list))
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["norm"] != "none":
                continue
            info = parse_combo_dir(r["config"])
            if info is None or info[0] != "mnist":
                continue
            key = (info[2], f"{info[4]:g}" if info[4] is not None else "")
            accs[key][info[3]].append(float(r["acc"]))
    out = {}
    for key, m in accs.items():
        valid = sorted(b for b, v in m.items()
                       if sum(v) / len(v) >= DEFAULT_VALIDITY_THRESHOLD)
        if not valid:
            raise ValueError(f"{key} 无干净精度 ≥ {DEFAULT_VALIDITY_THRESHOLD} 的 β（数据异常）")
        out[key] = max(valid)
    return out


def series_points(d, beta_max_map):
    """(model, anchor) → {beta: [succ]} → [(label, [(beta, succ_pct), ...])]，
    β 升序；各系列只取自身干净精度有效区间（β ≤ beta_max_map[(model, anchor)]：
    CEB 0.1、OPB a=1 的 1、a=6/12 全网格）；末尾追加基线横线
    （无 β，跨整个区间的一条常量虚线，CEB 论文 Det. 同款参照）。"""
    betas = sorted({b for key, m in d.items() for b in m})  # 全 β 网格
    out = []
    for model, anchors in (("ceb", [""]), ("opb", ["1", "6", "12"])):
        for a in anchors:
            m = d.get((model, a))
            if not m:
                continue
            limit = beta_max_map.get((model, a))
            pts = [(b, sum(m[b]) / len(m[b]) * 100) for b in betas
                   if b in m and m[b] and (limit is None or b <= limit)]
            if not pts:
                continue
            label = "CEB" if model == "ceb" else f"OPB (a={a})"
            out.append((label, pts))
    base = d.get(("mlp", ""))
    if base and betas:
        mean = sum(base[0.0]) / len(base[0.0]) * 100
        out.append((BASELINE_LABEL, [(betas[0], mean), (betas[-1], mean)]))
    return out


def plot_norm(norm, eps, series):
    """单范数论文图：x = β（对数刻度），y = targeted 成功率（%）。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y_max = 0.0
    for label, pts in series:
        if not pts:
            continue
        if label == BASELINE_LABEL:
            # 基线无 β：横跨整个区间的虚线常量（无圆点）
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    label=label, color=BASELINE_COLOR, linestyle="--", linewidth=1.6)
        else:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                label=label, color=SERIES_COLORS[label],
                marker="o", markersize=4, linewidth=1.6,
            )
        y_max = max(y_max, max(p[1] for p in pts))
    ax.set_xscale("log")
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("targeted attack success (%)")
    eps_s = f"$\\varepsilon_\\infty={eps:g}$" if norm == "linf" else f"$\\varepsilon_2={eps:g}$"
    ax.text(0.03, 0.96, eps_s, transform=ax.transAxes, fontsize=9,
            va="top", color="#52514e")
    ax.set_ylim(0.0, max(5.0, y_max * 1.15))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    path = FIG_DIR / f"fig_adv_targeted_{norm}_mnist.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"论文图已保存：{path}")
    return path


def main():
    data = load_succ()
    beta_max_map = valid_beta_max_map()
    for (model, a), lim in sorted(beta_max_map.items()):
        name = model.upper() + (f" a={a}" if a else "")
        print(f"{name:12s} 精度有效区间：β ≤ {lim:g}"
              "（超出区间干净精度失效，succ≈0 为坍缩/噪声瓶颈的平凡鲁棒性，不参与对比）")
    _setup_rc()
    for norm in ("linf", "l2"):
        if norm not in data:
            print(f"[跳过] 无 {norm} 数据")
            continue
        eps = max(data[norm])  # 该范数取 csv 中最大的 ε
        series = series_points(data[norm][eps], beta_max_map)
        if not series:
            print(f"[跳过] {norm} ε={eps:g} 无可解析配置")
            continue
        for label, pts in series:
            print(f"  {label:10s} n={len(pts)} succ[{min(p[1] for p in pts):.2f},"
                  f"{max(p[1] for p in pts):.2f}]%")
        plot_norm(norm, eps, series)


if __name__ == "__main__":
    main()
