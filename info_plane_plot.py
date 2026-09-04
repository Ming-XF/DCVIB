"""信息平面图：每数据集两张论文图——信息平面（x = I(X;Z)、y = I(Y;Z)）与
证书平面（x = I(X;Z|Y)、y = I(Y;Z)）；CEB 1 条线 + OPB/EPB 3 条线
（a/ρ = 1/6/12），按 β 升序连线、关键点标 β 值。另打印匹配压缩表
（CEB 崩溃点附近与其 I(X;Z) 最接近的 GPB 配置三量对比）。

数据来自 info_plane_eval.py 的 info_plane.csv（每 run 一行，跨 run 取均值）。
注意：I(X;Z) 为 InfoNCE 下界、分类 I(Y;Z) 为 H(Y)−CE 下界、回归 I(Y;Z) 为
高斯近似；I(X;Z|Y) 是两个下界之差，是启发式估计而非严格界（图与论文叙述
均如实标注）。I(X;Z|Y) 若出现负值（两下界偏差方向不同所致），证书平面
x 轴退化为线性刻度。

输出（白底、英文标签、无总标题）：
    paper/figures/fig_info_plane_{task}.png
    paper/figures/fig_certificate_plane_{task}.png

用法：
    python info_plane_plot.py
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compression_plot import SERIES_COLORS
from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
CSV_PATHS = {
    "mnist": ROOT / "output" / "adv_mnist" / "info_plane.csv",
    "imagenet100": ROOT / "output" / "compression_eval" / "info_plane.csv",
    "housing": ROOT / "output" / "compression_eval" / "info_plane.csv",
}
FIG_DIR = ROOT / "paper" / "figures"
TASK_DISPLAY = {"mnist": "MNIST", "imagenet100": "ImageNet-100", "housing": "Cal. Housing"}


def load(task):
    """读 csv → {(model, anchor): {beta: {I_XZ: mean, I_YZ: mean, ...}}}。"""
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
            accs[key].append(float(r["Acc"]) if r["Acc"] else float(r["R2"]))
    return data, accs


def series_points(data, is_cls):
    """→ [(label, [(beta, I_XZ, I_YZ, I_XZ_given_Y), ...])]，β 升序；
    分类标 OPB、回归（housing）标 EPB，与论文实例名一致。"""
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
                pts.append((b, i_xz, i_yz, i_xz - i_yz))
            if not pts:
                continue
            if model == "ceb":
                label = "CEB"
            else:
                label = f"OPB (a={a})" if is_cls else f"EPB (ρ={a})"
            out.append((label, pts))
    return out


def _setup_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")


def _draw(series, xs, ys, fig_path, xlabel, ylabel, xlog=True, annotate_betas=(5e-5, 0.1, 25)):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, pts in series:
        xv = [xs(p) for p in pts]
        yv = [ys(p) for p in pts]
        ax.plot(xv, yv, label=label, color=SERIES_COLORS[label],
                marker="o", markersize=4, linewidth=1.6)
        for p in pts:
            if p[0] in annotate_betas:
                ax.annotate(f"β={p[0]:g}", (xs(p), ys(p)), fontsize=7,
                            textcoords="offset points", xytext=(4, 3), color="#52514e")
    if xlog:
        ax.set_xscale("log")
    _setup_axes(ax, xlabel, ylabel)
    fig.patch.set_facecolor("white")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"论文图已保存：{fig_path}")


def plot_task(task, data):
    series = series_points(data, is_cls=task != "housing")
    name = TASK_DISPLAY.get(task, task)
    # 信息平面：x = I(X;Z)（对数），y = I(Y;Z)
    _draw(series, lambda p: p[1], lambda p: p[2],
          FIG_DIR / f"fig_info_plane_{task}.png",
          "$I(X;Z)$ (nats, InfoNCE lower bound)",
          "$I(Y;Z)$ (nats)" if task != "housing" else "$I(Y;Z)$ (nats, Gaussian approx.)",
          xlog=True)
    # 证书平面：x = I(X;Z|Y)，y = I(Y;Z)；负值退化为线性刻度
    cert_min = min(p[3] for _, pts in series for p in pts)
    _draw(series, lambda p: p[3], lambda p: p[2],
          FIG_DIR / f"fig_certificate_plane_{task}.png",
          "$I(X;Z|Y)$ (nats, heuristic estimate)",
          "$I(Y;Z)$ (nats)" if task != "housing" else "$I(Y;Z)$ (nats, Gaussian approx.)",
          xlog=cert_min > 0)


def matched_compression_table(data):
    """CEB β ∈ {0.1, 0.5} 与其 I(X;Z) 最接近的 GPB 配置三量对比（打印）。"""
    ceb = data.get(("ceb", ""), {})
    for beta_ceb in (0.1, 0.5):
        if beta_ceb not in ceb:
            continue
        i_xz_ceb = sum(ceb[beta_ceb]["i_xz"]) / len(ceb[beta_ceb]["i_xz"])
        i_yz_ceb = sum(ceb[beta_ceb]["i_yz"]) / len(ceb[beta_ceb]["i_yz"])
        best = None
        for (model, a), m in data.items():
            if model != "opb":
                continue
            for b, d in m.items():
                i_xz = sum(d["i_xz"]) / len(d["i_xz"])
                if best is None or abs(i_xz - i_xz_ceb) < abs(best[1] - i_xz_ceb):
                    best = (f"GPB β={b:g} a={a}", i_xz,
                            sum(d["i_yz"]) / len(d["i_yz"]))
        print(f"CEB β={beta_ceb:g}: I(X;Z)={i_xz_ceb:.3f} I(Y;Z)={i_yz_ceb:.3f} "
              f"I(X;Z|Y)={i_xz_ceb - i_yz_ceb:.3f}")
        print(f"  最接近 {best[0]}: I(X;Z)={best[1]:.3f} I(Y;Z)={best[2]:.3f} "
              f"I(X;Z|Y)={best[1] - best[2]:.3f}")


def main():
    _setup_rc()
    for task in ("mnist", "imagenet100", "housing"):
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
