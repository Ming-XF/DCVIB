"""信息平面图：每数据集两张论文图——信息平面（x = I(X;Z)、y = I(Y;Z)）与
证书平面（x = I(X;Z|Y)、y = I(Y;Z)）；CEB 1 条线 + OPB/EPB 3 条线
（a/ρ = 1/6/12），按 β 升序连线。另打印匹配压缩表
（CEB 崩溃点附近与其 I(X;Z) 最接近的 GPB 配置三量对比）。

数据来自 info_plane_eval.py 的 info_plane_{task}.csv（每 run 一行，跨 run 取均值）。
注意：I(X;Z) 为 InfoNCE 下界；I(Y;Z) 为随机路径 MC 估计（分类 ln K − CE_MC
下界、回归高斯近似）；分类 I(X;Z|Y) 为类条件高斯混合的精确 MC 估计
（logsumexp 对已知分量混合密度精确；mean-log 参考值 I_XZ_given_Y_upper
在 σ→0 时发散、不参与绘图）；回归 I(X;Z|Y) 为核加权 y-条件高斯混合的
直接 MC 估计（逐项 KL 非负）。x 轴数值非正时退化为线性刻度（兜底）。

输出（白底、英文标签、无总标题）：
    paper/figures/fig_info_plane_{task}.png
    paper/figures/fig_certificate_plane_{task}.png

用法：
    python info_plane_plot.py
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compression_plot import (
    SERIES_COLORS,
    PANEL_LABELSIZE,
    PANEL_TICKSIZE,
    PANEL_LEGENDSIZE,
    _panel_label,
)
from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
CSV_PATHS = {
    "mnist": ROOT / "output" / "adv_mnist" / "info_plane_mnist.csv",
    "imagenet100": ROOT / "output" / "compression_eval" / "info_plane_imagenet100.csv",
    "housing": ROOT / "output" / "compression_eval" / "info_plane_california.csv",
}
FIG_DIR = ROOT / "paper" / "figures"
TASK_DISPLAY = {"mnist": "MNIST", "imagenet100": "ImageNet-100", "housing": "Cal. Housing"}


def load(task):
    """读 csv → {(model, anchor): {beta: {i_xz: [..], i_yz: [..], cert: [..]}}}。"""
    data = defaultdict(lambda: defaultdict(dict))
    accs = defaultdict(list)
    with open(CSV_PATHS[task], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["task"] != task:
                continue
            key = (r["model"], f"{float(r['anchor']):g}" if r["anchor"] else "")
            b = float(r["beta"])
            d = data[key][b]
            d["i_xz"] = d.get("i_xz", []) + [float(r["I_XZ"])]
            d["i_yz"] = d.get("i_yz", []) + [float(r["I_YZ"])]
            d["cert"] = d.get("cert", []) + [float(r["I_XZ_given_Y"])]
            accs[key].append(float(r["Acc"]) if r["Acc"] else float(r["R2"]))
    return data, accs


def series_points(data, is_cls):
    """→ [(label, [(beta, I_XZ, I_YZ, cert), ...])]，β 升序；cert 为
    I(X;Z|Y) 点估计（分类 = logsumexp 精确 MC 估计、回归 = 核加权混合
    直接估计）。分类标 OPB、回归（housing）标 EPB，与论文实例名一致。"""
    out = []
    for model, anchors in (("ceb", [""]), ("opb", ["1", "6", "12"])):
        for a in anchors:
            m = data.get((model, a))
            if not m:
                continue
            pts = []
            for b in sorted(m):
                d = m[b]
                i_xz = sum(d["i_xz"]) / len(d["i_xz"])
                i_yz = sum(d["i_yz"]) / len(d["i_yz"])
                cert = sum(d["cert"]) / len(d["cert"])
                pts.append((b, i_xz, i_yz, cert))
            if not pts:
                continue
            if model == "ceb":
                label = "CEB"
            else:
                label = f"OPB (a={a})" if is_cls else f"EPB (ρ={a})"
            out.append((label, pts))
    return out


def _setup_axes(ax, xlabel, ylabel, paper_style):
    if paper_style:
        # 论文图 4：放大字号、取消网格线
        ax.set_xlabel(xlabel, fontsize=PANEL_LABELSIZE)
        ax.set_ylabel(ylabel, fontsize=PANEL_LABELSIZE)
        ax.tick_params(labelsize=PANEL_TICKSIZE)
        ax.legend(fontsize=PANEL_LEGENDSIZE, frameon=False)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")


def _draw(series, xs, ys, fig_path, xlabel, ylabel, xlog=True, paper_style=False,
          letter=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, pts in series:
        xv = [xs(p) for p in pts]
        yv = [ys(p) for p in pts]
        ax.plot(xv, yv, label=label, color=SERIES_COLORS[label],
                marker="o", markersize=4, linewidth=1.6)
    if xlog:
        ax.set_xscale("log")
    _setup_axes(ax, xlabel, ylabel, paper_style)
    if letter:
        _panel_label(ax, letter)
    fig.patch.set_facecolor("white")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"论文图已保存：{fig_path}")


def plot_task(task, data):
    series = series_points(data, is_cls=task != "housing")
    name = TASK_DISPLAY.get(task, task)
    paper_style = task == "mnist"  # 论文图 4（附录 imagenet100/housing 平面保持原样式）
    # 信息平面：x = I(X;Z)（对数），y = I(Y;Z)
    _draw(series, lambda p: p[1], lambda p: p[2],
          FIG_DIR / f"fig_info_plane_{task}.png",
          "$I(X;Z)$ (nats, InfoNCE lb.)" if paper_style
          else "$I(X;Z)$ (nats, InfoNCE lower bound)",
          "$I(Y;Z)$ (nats)" if task != "housing" else "$I(Y;Z)$ (nats, Gaussian approx.)",
          xlog=True, paper_style=paper_style,
          letter="(c)" if paper_style else None)
    # 证书平面：x = I(X;Z|Y)（分类为类条件混合直接估计、回归为核加权混合
    # 直接估计），y = I(Y;Z)；x 轴数值非正时退化为线性刻度
    cert_min = min(p[3] for _, pts in series for p in pts)
    cert_label = ("$I(X;Z|Y)$ (nats, direct est.)" if task != "housing"
                  else "$I(X;Z|Y)$ (nats, kernel-mixture est.)")
    _draw(series, lambda p: p[3], lambda p: p[2],
          FIG_DIR / f"fig_certificate_plane_{task}.png",
          cert_label,
          "$I(Y;Z)$ (nats)" if task != "housing" else "$I(Y;Z)$ (nats, Gaussian approx.)",
          xlog=cert_min > 0, paper_style=paper_style,
          letter="(d)" if paper_style else None)


def matched_compression_table(data):
    """CEB β ∈ {0.1, 0.5} 与其 I(X;Z) 最接近的 GPB 配置三量对比（打印）。"""
    ceb = data.get(("ceb", ""), {})
    for beta_ceb in (0.1, 0.5):
        if beta_ceb not in ceb:
            continue
        i_xz_ceb = sum(ceb[beta_ceb]["i_xz"]) / len(ceb[beta_ceb]["i_xz"])
        i_yz_ceb = sum(ceb[beta_ceb]["i_yz"]) / len(ceb[beta_ceb]["i_yz"])
        cert_ceb = sum(ceb[beta_ceb]["cert"]) / len(ceb[beta_ceb]["cert"])
        best = None
        for (model, a), m in data.items():
            if model != "opb":
                continue
            for b, d in m.items():
                i_xz = sum(d["i_xz"]) / len(d["i_xz"])
                if best is None or abs(i_xz - i_xz_ceb) < abs(best[1] - i_xz_ceb):
                    best = (f"GPB β={b:g} a={a}", i_xz,
                            sum(d["i_yz"]) / len(d["i_yz"]),
                            sum(d["cert"]) / len(d["cert"]))
        print(f"CEB β={beta_ceb:g}: I(X;Z)={i_xz_ceb:.3f} I(Y;Z)={i_yz_ceb:.3f} "
              f"I(X;Z|Y)={cert_ceb:.3f}")
        print(f"  最接近 {best[0]}: I(X;Z)={best[1]:.3f} I(Y;Z)={best[2]:.3f} "
              f"I(X;Z|Y)={best[3]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="*", default=["mnist", "imagenet100", "housing"],
        help="任务列表（默认全部三个；只重跑部分数据时指定对应任务）",
    )
    args = parser.parse_args()
    _setup_rc()
    for task in args.tasks:
        data, _ = load(task)
        if not data:
            print(f"[跳过] {task}：无数据（先运行 info_plane_eval.py）")
            continue
        plot_task(task, data)
        print(f"[{TASK_DISPLAY.get(task, task)}] 匹配压缩表：")
        matched_compression_table(data)
        print()


if __name__ == "__main__":
    main()
