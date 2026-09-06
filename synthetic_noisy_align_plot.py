"""噪声合成对齐试验论文图（codex_output1.txt 方案 §6）：2×2 主图 + 附录图 +
小表 + 核心数值。

主图（fixed frame、posterior_var=paper、population 模式、best checkpoint）：
  (a) D̂ vs β（log x）：每 r 一条 mean±std 曲线 + δ_noise 水平虚线 +
      δ_noise+C_comp/β 参考点线；
  (b) D̂−δ_noise vs 1/β：双 log、只画正值；
  (c) β·(D̂−δ_noise) vs β（log x）+ C_comp 水平虚线；
  (d) noisy-label acc（实线）与 clean-class acc（虚线）vs β。
附录图三面板：(i) posterior_var=fixed（最干净的标签歧义验证）、
  (ii) trainable frame 稳健性、(iii) 采样标签敏感性（经验 η 偏差另打印）。
所有点 mean±std over seeds（5 seeds）；失败行跳过并计数；opt_fail=1 的点
打叉标记（该点不得用于上界讨论）。r=0 与既有确定性实验
（output/synthetic_align/synthetic_align.csv，fixed a=6）交叉核对并打印。

输出 output/synthetic_noisy_align/：
    fig_noisy_alignment.png           2×2 主图
    fig_noisy_alignment_appendix.png  三面板附录图
    summary.csv                       每 (mode,frame,var,r,β) 的 mean±std 聚合
另打印：LaTeX 小表（方案 §6 表）、gating 统计、核心数值摘要。

用法：
    python synthetic_noisy_align_plot.py
    python synthetic_noisy_align_plot.py --csv output/synthetic_noisy_align/synthetic_noisy_align.csv
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "synthetic_noisy_align"

R_COLORS = {
    0.0: "#2a78d6", 0.1: "#eb6834", 0.2: "#4f9a5f", 0.4: "#9a6ac4",
}


def load(csv_path, mode="population", frame="fixed", var="paper"):
    """→ {r: {beta: {col: [values]}}}，跳过 fail 行；返回 (data, n_fail, opt_fail_rb)。"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    opt_fail = defaultdict(int)
    fails = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fail"]:
                fails += 1
                continue
            if (row["mode"] != mode or row["frame_type"] != frame
                    or row["posterior_var_mode"] != var or row["checkpoint"] != "best"):
                continue
            r = float(row["noise_rate"])
            b = float(row["beta"])
            for col in ["d_hat", "trained_risk_mc", "trained_objective_mc",
                        "L_hat_minus_L_comp", "excess_d", "beta_excess_d",
                        "noisy_label_acc", "clean_class_acc", "kl_total"]:
                data[r][b][col].append(float(row[col]))
            if row["opt_fail"] == "1":
                opt_fail[(r, b)] += 1
    return data, fails, opt_fail


def load_ref(csv_path):
    """reference.csv → {r: {beta: (delta, c_comp)}}。"""
    ref = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref.setdefault(float(row["noise_rate"]), {})[float(row["beta"])] = (
                float(row["delta_noise"]), float(row["comparator_risk"]))
    return ref


def mean_std(vals):
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return mean, std


def plot_series(ax, data, r, col, xlog=True, marker="o", linestyle="-", scale=1.0):
    m = data[r]
    xs, ys, errs = [], [], []
    for b in sorted(m):
        mean, std = mean_std(m[b][col])
        xs.append(b)
        ys.append(mean * scale)
        errs.append(std * scale)
    ax.errorbar(xs, ys, yerr=errs, label=f"$r={r:g}$", color=R_COLORS[r],
                marker=marker, markersize=4, linewidth=1.6, capsize=2,
                linestyle=linestyle)
    if xlog:
        ax.set_xscale("log")


def fig_main(data, ref, opt_fail):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    # (a) D̂ vs β + δ 水平虚线 + 参考曲线 δ+C_comp/β
    ax = axes[0, 0]
    for r in sorted(data):
        plot_series(ax, data, r, "d_hat")
        betas = sorted(ref[r])
        delta = ref[r][betas[0]][0]
        ax.axhline(delta, color=R_COLORS[r], linestyle="--", linewidth=1.0, alpha=0.7)
        c = ref[r][betas[0]][1]
        xs = sorted(data[r])
        ax.plot(xs, [delta + c / b for b in xs], color=R_COLORS[r], linestyle=":",
                linewidth=1.0, alpha=0.7)
    # 标注 opt_fail 点（×）
    for (r, b), n in opt_fail.items():
        if n:
            ax.scatter([b], [mean_std(data[r][b]["d_hat"])[0]], marker="x",
                       color=R_COLORS[r], s=40, linewidths=1.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\widehat D(Y)$ (nats)")
    ax.set_title("(a) anchor mismatch vs. $\\beta$; dashed: $\\delta_{\\mathrm{noise}}$, "
                 "dotted: $\\delta_{\\mathrm{noise}}+C_{\\mathrm{comp}}/\\beta$", fontsize=9)

    # (b) D̂−δ vs 1/β，双 log、只画正值
    ax = axes[0, 1]
    for r in sorted(data):
        if r == 0.0:
            continue  # δ=0，与 (a) 重复
        delta = ref[r][sorted(ref[r])[0]][0]
        xs, ys, errs = [], [], []
        for b in sorted(data[r]):
            mean, std = mean_std(data[r][b]["excess_d"])
            if mean > 0:
                xs.append(1.0 / b)
                ys.append(mean)
                errs.append(std)
        ax.errorbar(xs, ys, yerr=errs, label=f"$r={r:g}$", color=R_COLORS[r],
                    marker="o", markersize=4, linewidth=1.6, capsize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$1/\beta$")
    ax.set_ylabel(r"$\widehat D - \delta_{\mathrm{noise}}$ (nats, $>0$ only)")
    ax.set_title("(b) excess mismatch vs. $1/\\beta$ (log-log)", fontsize=9)

    # (c) β·(D̂−δ) vs β + C_comp 水平虚线
    ax = axes[1, 0]
    for r in sorted(data):
        if r == 0.0:
            continue
        plot_series(ax, data, r, "beta_excess_d")
        c = ref[r][sorted(ref[r])[0]][1]
        ax.axhline(c, color=R_COLORS[r], linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\beta\,(\widehat D - \delta_{\mathrm{noise}})$ (nats)")
    ax.set_title("(c) scaled excess vs. $\\beta$; dashed: $C_{\\mathrm{comp}}(r)$", fontsize=9)

    # (d) noisy acc（实线）与 clean acc（虚线）
    ax = axes[1, 1]
    for r in sorted(data):
        plot_series(ax, data, r, "noisy_label_acc", marker="o", linestyle="-")
        plot_series(ax, data, r, "clean_class_acc", marker="x", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(d) noisy-label acc (solid) / clean-class acc (dashed)", fontsize=9)

    for ax in axes.flat:
        ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
        ax.set_facecolor("white")
        ax.legend(fontsize=8, frameon=False, ncol=2 if ax is axes[0, 0] else 1)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = OUT_DIR / "fig_noisy_alignment.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"主图已保存：{path}")


def fig_appendix(datasets, ref):
    """三面板：var=fixed / trainable frame / sampled 敏感性（D̂ vs β + δ 虚线）。"""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    titles = [
        "(i) posterior variance fixed ($\\sigma^2\\equiv\\tau^2$)",
        "(ii) trainable polar frame",
        "(iii) sampled labels (finite-sample sensitivity)",
    ]
    for ax, (data, _), title in zip(axes, datasets, titles):
        for r in sorted(data):
            plot_series(ax, data, r, "d_hat")
            delta = ref[r][sorted(ref[r])[0]][0]
            ax.axhline(delta, color=R_COLORS[r], linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$\widehat D(Y)$ (nats)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
        ax.set_facecolor("white")
        ax.legend(fontsize=7, frameon=False)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    path = OUT_DIR / "fig_noisy_alignment_appendix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"附录图已保存：{path}")


def print_table(data, ref, opt_fail):
    """方案 §6 的小表：每 r：δ_noise、最大 β 处 D̂、excess、L̂−L_comp（mean±std）。"""
    b_max = max(sorted(data[sorted(data)[0]].keys()))
    print("\n% LaTeX 小表（mean±std over seeds；best checkpoint）：")
    print(r"\begin{tabular}{c c c c c}")
    print(r"\hline")
    print(r"$r$ & $\delta_{\mathrm{noise}}$ & $\widehat D(\beta{=}10)$ & "
          r"$\widehat D-\delta_{\mathrm{noise}}$ & $\widehat L-L_{\mathrm{comp}}$ \\")
    print(r"\hline")
    for r in sorted(data):
        delta = ref[r][sorted(ref[r])[0]][0]
        m1, s1 = mean_std(data[r][b_max]["d_hat"])
        m2, s2 = mean_std(data[r][b_max]["excess_d"])
        m3, s3 = mean_std(data[r][b_max]["L_hat_minus_L_comp"])
        print(f"${r:g}$ & ${delta:.3f}$ & ${m1:.3f}_{{\\pm{s1:.3f}}}$ & "
              f"${m2:.3f}_{{\\pm{s2:.3f}}}$ & ${m3:.3f}_{{\\pm{s3:.3f}}}$ \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


def print_core_numbers(data, ref):
    """5–8 个核心数值（主文可引用）。"""
    print("\n核心数值：")
    for r in sorted(data):
        delta = ref[r][sorted(ref[r])[0]][0]
        c = ref[r][sorted(ref[r])[0]][1]
        b_max = max(data[r])
        d10, s10 = mean_std(data[r][b_max]["d_hat"])
        ex10, _ = mean_std(data[r][b_max]["excess_d"])
        print(f"  r={r:g}: δ_noise={delta:.4f}, D̂(β={b_max:g})={d10:.4f}±{s10:.4f}, "
              f"excess(β={b_max:g})={ex10:+.4f}, C_comp={c:.4f}")


def cross_check_deterministic(data, ref):
    """r=0 与既有确定性实验（synthetic_align.csv, fixed a=6）交叉核对。"""
    path = ROOT / "output" / "synthetic_align" / "synthetic_align.csv"
    if not path.exists():
        print(f"[跳过] 未找到确定性实验 CSV：{path}")
        return
    old = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["frame_setting"] != "fixed" or float(row["anchor_scale"]) != 6.0 or row["fail"]:
                continue
            old[float(row["beta"])] = old.get(float(row["beta"]), []) + [float(row["D_hat_test"])]
    if 0.0 not in data:
        print("[跳过] 本 CSV 无 r=0 行")
        return
    print("\nr=0 交叉核对（新实验 D̂ vs 确定性实验 D_hat_test，mean）：")
    for b in sorted(data[0.0]):
        if b not in old:
            continue
        new_m, _ = mean_std(data[0.0][b]["d_hat"])
        old_m = sum(old[b]) / len(old[b])
        flag = "OK" if (new_m < 1e-3 and old_m < 1e-3) or abs(new_m - old_m) < 0.5 else "注意"
        print(f"  β={b:g}: 新 {new_m:.6f} | 旧 {old_m:.6f}  [{flag}]")


def write_summary(csv_path, ref):
    """summary.csv：每 (mode, frame, var, r, β) 的 mean±std 聚合（best checkpoint）。"""
    agg = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fail"] or row["checkpoint"] != "best":
                continue
            key = (row["mode"], row["frame_type"], row["posterior_var_mode"],
                   float(row["noise_rate"]), float(row["beta"]))
            for col in ["d_hat", "kl_total", "trained_risk", "trained_risk_mc",
                        "trained_objective_mc", "L_hat_minus_L_comp", "excess_d",
                        "beta_excess_d", "noisy_label_acc", "clean_class_acc"]:
                agg[key][col].append(float(row[col]))
            agg[key]["n_opt_fail"] = agg[key].get("n_opt_fail", 0) + (row["opt_fail"] == "1")
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "frame_type", "posterior_var_mode", "noise_rate", "beta",
                    "metric", "mean", "std", "n", "n_opt_fail"])
        for key in sorted(agg):
            m, frm, var, r, b = key
            d = agg[key]
            for col in sorted(d):
                if col == "n_opt_fail":
                    continue
                mean, std = mean_std(d[col])
                w.writerow([m, frm, var, f"{r:g}", f"{b:g}", col,
                            f"{mean:.6f}", f"{std:.6f}", len(d[col]), d["n_opt_fail"]])
    print(f"summary.csv 已保存：{OUT_DIR / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="噪声合成对齐试验论文图")
    parser.add_argument("--csv", default=str(OUT_DIR / "synthetic_noisy_align.csv"))
    parser.add_argument("--ref", default=str(OUT_DIR / "reference.csv"))
    args = parser.parse_args()

    ref = load_ref(args.ref)
    if not ref:
        print(f"[跳过] 无参考数据：{args.ref}")
        return
    _setup_rc()

    data, fails, opt_fail = load(args.csv, "population", "fixed", "paper")
    print(f"主图数据（fixed/paper/population/best）：失败行 {fails}，"
          f"opt_fail 点 {sum(opt_fail.values())}")
    if data:
        fig_main(data, ref, opt_fail)
        print_table(data, ref, opt_fail)
        print_core_numbers(data, ref)
        cross_check_deterministic(data, ref)
        # gating 统计
        bad = [(r, b) for (r, b), n in opt_fail.items() if n]
        print(f"\ngating（L̂≤L_comp+1.0）：{len(bad)} 个 (r,β) 组合含 opt_fail 行"
              + (f"：{bad}" if bad else "——全部通过"))

    d1, _, _ = load(args.csv, "population", "fixed", "fixed")
    d2, _, _ = load(args.csv, "population", "trainable", "paper")
    d3, f3, _ = load(args.csv, "sampled", "fixed", "paper")
    print(f"附录图数据：var=fixed {sum(len(v) for v in d1.values())} β 点、"
          f"trainable {sum(len(v) for v in d2.values())} β 点、"
          f"sampled {sum(len(v) for v in d3.values())} β 点（失败行 {f3}）")
    if d1 or d2 or d3:
        fig_appendix([(d1, None), (d2, None), (d3, None)], ref)

    write_summary(args.csv, ref)


if __name__ == "__main__":
    main()
