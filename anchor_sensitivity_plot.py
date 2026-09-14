#!/usr/bin/env python
"""a/rho 超参数敏感性分析绘图脚本（只读日志、不重训）。

数据来源：tune_results 下 {dataset}_{backbone}_{model}_beta_{b}_anchor_{a}/train.log
（beta 6 值 x a/rho 9 值网格，每配置多 run），取每条日志最后一条
"Average over N runs | Test ..." 汇总行的 Acc（分类）/ R2（回归）均值±std。

在三个 beta（弱/中/强压缩）下绘制 a/rho–Acc / a/rho–R2 曲线，每个 setting
输出独立 PNG；全部解析结果另存 CSV 供复查。默认代表性两个 setting：
ImageNet-100 (MLP, 分类 OPB) 与 AgeDB (MLP, 回归 EPB)。

用法：
  python anchor_sensitivity_plot.py
  python anchor_sensitivity_plot.py --model-name opbl   # 换用 opbl 结果目录
  python anchor_sensitivity_plot.py --beta-low 0.01 --beta-mid 1 --beta-high 10
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 代表性两个 setting（与论文 sensitivity 表口径一致）
SETTINGS = [
    dict(dataset="imagenet100", backbone="mlp", task="cls",
         title="ImageNet-100 (MLP): OPB anchor-scale sensitivity",
         xlabel="anchor scale $a$", ylabel="Test Acc (%)", metric="Acc"),
    dict(dataset="agedb", backbone="mlp", task="reg",
         title="AgeDB (MLP): EPB isometric-scale sensitivity",
         xlabel="isometric scale $\\rho$", ylabel="Test $R^2$", metric="R2"),
]

ANCHOR_GRID = [1, 2, 4, 6, 8, 10, 12, 14, 16]  # a/rho 九值网格
BETA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]    # beta 六值网格

# 三条 beta 曲线配色（dataviz 验证调色板 slots 1–3，浅色模式）
BETA_COLORS = {1e-2: "#2a78d6", 1: "#eb6834", 10: "#1baf7a"}

CLS_RE = re.compile(
    r"Average over \d+ runs \| Test Loss [\d.eE+-]+±[\d.eE+-]+ "
    r"Acc ([\d.eE+-]+)±([\d.eE+-]+)")
REG_RE = re.compile(
    r"Average over \d+ runs \| Test Loss [\d.eE+-]+±[\d.eE+-]+ "
    r"MAE [\d.eE+-]+±[\d.eE+-]+ R2 ([\d.eE+-]+)±([\d.eE+-]+)")


def parse_summary(log_path):
    """解析 train.log 最后一条汇总行，返回 (mean, std) 或 None。"""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # 追加模式的日志可能有多条汇总行，取最后一条
    for line in reversed(lines):
        m = CLS_RE.search(line) or REG_RE.search(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def fmt_beta(beta):
    """网格值转目录名后缀：0.01 -> '0.01'，1 -> '1'，10 -> '10'。"""
    return f"{beta:g}"


def fmt_beta_label(beta):
    """图例/直标用：1e-2 -> 'β=10^{-2}'，1 -> 'β=1'，10 -> 'β=10'。"""
    if beta < 1 and beta >= 1e-4:
        return r"$\beta=10^{-%d}$" % abs(int(round(np.log10(beta))))
    return r"$\beta=%.4g$" % beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="tune_results")
    parser.add_argument("--figures-dir", default="paper/figures")
    parser.add_argument("--model-name", default="opb",
                        help="结果目录中的模型名（opb 主方法 / opbl 别名）")
    parser.add_argument("--beta-low", type=float, default=1e-2)
    parser.add_argument("--beta-mid", type=float, default=1)
    parser.add_argument("--beta-high", type=float, default=10)
    parser.add_argument("--csv", default="tune_results/anchor_sensitivity.csv")
    args = parser.parse_args()

    betas = [args.beta_low, args.beta_mid, args.beta_high]
    if any(b not in BETA_GRID for b in betas):
        print(f"警告: 所选 beta {betas} 不在六值网格 {BETA_GRID} 中，目录可能缺失")
    os.makedirs(args.figures_dir, exist_ok=True)

    rows = []  # CSV 行：(dataset, backbone, model, beta, anchor, metric, mean, std)
    for s in SETTINGS:
        stem = f"{s['dataset']}_{s['backbone']}"
        # 网格配置
        for beta in BETA_GRID:
            for a in ANCHOR_GRID:
                log = Path(args.results_dir) / (
                    f"{stem}_{args.model_name}_beta_{fmt_beta(beta)}_anchor_{a}"
                    f"/train.log")
                res = parse_summary(str(log))
                if res is None:
                    print(f"缺失: {log}")
                rows.append([s["dataset"], s["backbone"], args.model_name,
                             fmt_beta(beta), a, s["metric"],
                             res[0] if res else "", res[1] if res else ""])
        # Base 基线（无 a 维度）
        base_log = Path(args.results_dir) / f"{stem}/train.log"
        res = parse_summary(str(base_log))
        if res is None:
            print(f"缺失: {base_log}")
        rows.append([s["dataset"], s["backbone"], "base", "", "",
                     s["metric"], res[0] if res else "", res[1] if res else ""])

    # CSV 落盘（全部 6x9 网格，供复查）
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "backbone", "model", "beta", "anchor",
                    "metric", "mean", "std"])
        w.writerows(rows)

    # 每 setting 一张独立图
    for s in SETTINGS:
        stem = f"{s['dataset']}_{s['backbone']}"
        fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=300)

        # 三条 beta 曲线（不画误差棒，跨 seed 方差信息保留在 CSV 中）
        for i, beta in enumerate(betas):
            xs, ys = [], []
            for a in ANCHOR_GRID:
                r = next((r for r in rows
                          if r[0] == s["dataset"] and r[2] == args.model_name
                          and r[3] == fmt_beta(beta) and r[4] == a), None)
                if r and r[6]:
                    xs.append(a)
                    ys.append(float(r[6]))
            color = BETA_COLORS.get(beta, "#555550")
            ax.plot(xs, ys, color=color, linewidth=2,
                    marker="o", markersize=7, markeredgecolor="white",
                    markeredgewidth=1.0, alpha=0.9,
                    label=fmt_beta_label(beta))

        ax.set_xticks(ANCHOR_GRID)
        ax.set_xlim(ANCHOR_GRID[0] - 0.6, ANCHOR_GRID[-1] + 0.8)
        ax.set_xlabel(s["xlabel"], fontsize=10)
        ax.set_ylabel(s["ylabel"], fontsize=10)
        ax.set_title(s["title"], fontsize=11)
        ax.grid(axis="y", color="#d8d8d2", linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9, loc="best")
        ax.tick_params(labelsize=8)

        out = Path(args.figures_dir) / (
            f"fig_anchor_sensitivity_{stem}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"已输出: {out}")

    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
