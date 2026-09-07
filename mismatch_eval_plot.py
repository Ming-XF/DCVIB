"""失配分析论文图（codex_output2.txt 方案 §5）：主 2×2 + 分类/回归附录图 +
一致性表 + 核心趋势数值。

主图（opb/epb、seed 级 mean±std、failed 行跳过并计数）：
  (a) ImageNet-100: D vs β 按 a 分曲线，虚线为 total KL（同 a 配色）；
  (b) ImageNet-100: MC-10 acc vs D 散点，颜色 = β（对数色标）、形状 = a；
  (c) Housing: D vs β 按 ρ 分曲线，并分画轴向/离轴两个贡献（D=两贡献之和）；
  (d) Housing: R² vs D 散点，颜色 = β、形状 = ρ。
附录图一（分类）：(i) 类间距离 mean(S_ij) 与原始下界 mean(LB_ij) vs β 按 a
  （CEB 的 D 作灰色虚线描述性参照）；(ii) P(LB>0) vs β；(iii) slack 的
  5/50/95 分位带 vs β。
附录图二（回归）：选定配置（ρ=6 的若干 β）的 bin 对 S_bc vs ρ|ȳ_b−ȳ_c| 散点
  + y=x 参考线（数据来自 detail npz）。
另打印：D/KL 一致性表（LaTeX）、核心趋势数值、slack sanity 统计。

用法：
    python mismatch_eval_plot.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prior_geometry import _setup_rc

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "output" / "mismatch_eval"

A_COLORS = {1.0: "#2a78d6", 6.0: "#eb6834", 12.0: "#4f9a5f"}
A_MARKERS = {1.0: "o", 6.0: "s", 12.0: "^"}

# 论文图 1 四面板（a/b/(i)/(ii)）的放大文字字号：画布尺寸保持默认 7.2×4.6 不变，
# 仅直接放大文字（轴标签/刻度/图例/色条）
PANEL_LABELSIZE = 33
PANEL_TICKSIZE = 29
PANEL_LEGENDSIZE = 20


def _panel_label(ax, letter):
    """论文面板角标：面板左上角外侧加粗字母（(a)/(b)/...）。"""
    ax.text(0.0, 1.04, letter, transform=ax.transAxes, fontsize=PANEL_LABELSIZE,
            fontweight="bold", va="bottom", ha="left")


def load(csv_path):
    """→ {model: {(beta, anchor): {col: [seed 值]}}}，跳过 failed 行。"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fails = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["failed"]:
                fails += 1
                continue
            anchor = float(r["anchor_scale"]) if r["anchor_scale"] else None
            key = (float(r["beta"]), anchor)
            for col, v in r.items():
                if col in ("dataset", "seed", "beta", "anchor_scale", "model", "failed"):
                    continue
                if v == "":
                    continue
                try:
                    data[r["model"]][key][col].append(float(v))
                except ValueError:
                    pass
    return data, fails


def mean_std(vals):
    v = np.array(vals)
    return v.mean(), v.std()


def series(model_data, col):
    """→ {anchor: [(beta, mean, std), ...]} β 升序。"""
    out = defaultdict(list)
    for (beta, a), cols in model_data.items():
        if col not in cols:
            continue
        m, s = mean_std(cols[col])
        out[a].append((beta, m, s))
    return {a: sorted(v) for a, v in out.items()}


def new_fig(w=7.2, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(OUT_ROOT / path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"图已保存：{OUT_ROOT / path}")


def fig_panel_a(cls_data):
    """(a) ImageNet-100: D vs β（实线）+ total KL（虚线），按 a 分曲线。"""
    fig, ax = new_fig()
    ax.grid(False)  # 论文图 1：取消网格线
    opb = cls_data["opb"]
    for a in (1.0, 6.0, 12.0):
        for col, ls in (("d_mean", "-"), ("kl_total", "--")):
            pts = series(opb, col).get(a)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=A_COLORS[a], linestyle=ls,
                    marker=A_MARKERS[a] if ls == "-" else None, markersize=4,
                    linewidth=1.6,
                    markerfacecolor="white", markeredgewidth=0.8,
                    label=f"$a={a:g}$: $D$" if ls == "-" else f"$a={a:g}$: total KL")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$", fontsize=PANEL_LABELSIZE)
    ax.set_ylabel("nats", fontsize=PANEL_LABELSIZE)
    ax.tick_params(labelsize=PANEL_TICKSIZE)
    # 图例锚定在 20pt 字号下 loc="best" 的实际位置（轴分数 bbox），
    # 只缩字号、不随 loc="best" 重排位（否则会左移压到曲线）
    ax.legend(fontsize=PANEL_LEGENDSIZE - 2, frameon=False,
              bbox_to_anchor=(0.4338, 0.3138, 0.5413, 0.7670), loc="upper left")
    _panel_label(ax, "(a)")
    save_fig(fig, "fig_mismatch_a_imagenet100_d_beta.png")


def fig_panel_b(cls_data):
    """(b) ImageNet-100: MC-10 acc vs D 散点，颜色 = β（对数色标）、形状 = a。"""
    fig, ax = new_fig()
    ax.grid(False)  # 论文图 1：取消网格线
    opb = cls_data["opb"]
    for a in (1.0, 6.0, 12.0):
        pts = series(opb, "d_mean").get(a, [])
        accs = series(opb, "mc10_metric").get(a, [])
        by_beta = {p[0]: p[1] for p in pts}
        by_beta_acc = {p[0]: p[1] for p in accs}
        xs = [by_beta[b] for b in by_beta if b in by_beta_acc]
        ys = [by_beta_acc[b] * 100 for b in by_beta if b in by_beta_acc]
        bs = [b for b in by_beta if b in by_beta_acc]
        sc = ax.scatter(xs, ys, c=bs, norm=matplotlib.colors.LogNorm(), cmap="viridis",
                        marker=A_MARKERS[a], s=45, edgecolors="white", linewidths=0.4,
                        label=f"$a={a:g}$")
    ax.set_xlabel(r"$D$ (nats)", fontsize=PANEL_LABELSIZE)
    ax.set_ylabel("MC-10 acc. (%)", fontsize=PANEL_LABELSIZE)
    ax.tick_params(labelsize=PANEL_TICKSIZE)
    ax.legend(fontsize=PANEL_LEGENDSIZE, frameon=False)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\beta$", fontsize=PANEL_LABELSIZE)
    cbar.ax.tick_params(labelsize=PANEL_TICKSIZE)
    _panel_label(ax, "(b)")
    save_fig(fig, "fig_mismatch_b_imagenet100_acc_vs_d.png")


def fig_panel_c(reg_data):
    """(c) Housing: D vs β（实线）+ 轴向（虚线）/离轴（点线）贡献，按 ρ 分曲线。"""
    fig, ax = new_fig()
    epb = reg_data["opb"]
    for rho in (1.0, 6.0, 12.0):
        for col, ls, mk in (("d_mean", "-", "o"), ("d_axial", "--", None), ("d_offaxis", ":", None)):
            pts = series(epb, col).get(rho)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            errs = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=errs, color=A_COLORS[rho], linestyle=ls,
                        marker=mk, markersize=4, linewidth=1.6, capsize=2,
                        markerfacecolor="white", markeredgewidth=0.8,
                        label=rf"$\rho={rho:g}$" if col == "d_mean" else None)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("nats")
    ax.legend(fontsize=8, frameon=False)
    save_fig(fig, "fig_mismatch_c_housing_d_beta.png")


def fig_panel_d(reg_data):
    """(d) Housing: R² vs D 散点，颜色 = β、形状 = ρ。"""
    fig, ax = new_fig()
    epb = reg_data["opb"]
    for rho in (1.0, 6.0, 12.0):
        pts = series(epb, "d_mean").get(rho, [])
        r2s = series(epb, "deterministic_metric").get(rho, [])
        by_beta = {p[0]: p[1] for p in pts}
        by_beta_r2 = {p[0]: p[1] for p in r2s}
        xs = [by_beta[b] for b in by_beta if b in by_beta_r2]
        ys = [by_beta_r2[b] for b in by_beta if b in by_beta_r2]
        bs = [b for b in by_beta if b in by_beta_r2]
        sc = ax.scatter(xs, ys, c=bs, norm=matplotlib.colors.LogNorm(), cmap="viridis",
                        marker=A_MARKERS[rho], s=45, edgecolors="white", linewidths=0.4,
                        label=rf"$\rho={rho:g}$")
    ax.set_xlabel(r"$D$ (nats)")
    ax.set_ylabel(r"test $R^2$")
    ax.legend(fontsize=8, frameon=False)
    plt.colorbar(sc, ax=ax, label=r"$\beta$")
    save_fig(fig, "fig_mismatch_d_housing_r2_vs_d.png")


def fig_appendix_cls(cls_data):
    """分类附录三个面板分别成图：(i) S/LB vs β、(ii) P(LB>0)、(iii) slack 分位带。"""
    opb = cls_data["opb"]
    ceb = cls_data.get("ceb", {})

    # (i) S 与 LB vs β（CEB 的 D 作灰色点线描述性参照）
    fig, ax = new_fig()
    ax.grid(False)  # 论文图 1：取消网格线
    for a in (1.0, 6.0, 12.0):
        for col, ls, mk in (("center_distance_mean", "-", "o"), ("lower_bound_mean", "--", None)):
            pts = series(opb, col).get(a)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=A_COLORS[a], linestyle=ls, marker=mk,
                    markersize=4, linewidth=1.6,
                    markerfacecolor="white", markeredgewidth=0.8,
                    label=f"$a={a:g}$" if col == "center_distance_mean" else None)
    pts = series(ceb, "d_mean").get(None)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#8a8a8a", linestyle=":", linewidth=1.4, label="CEB: $D$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$", fontsize=PANEL_LABELSIZE)
    ax.set_ylabel("dist. (nats$^{1/2}$)", fontsize=PANEL_LABELSIZE)
    ax.tick_params(labelsize=PANEL_TICKSIZE)
    ax.legend(fontsize=PANEL_LEGENDSIZE, frameon=False)
    _panel_label(ax, "(c)")
    save_fig(fig, "fig_mismatch_appendix_cls_i_separation_beta.png")

    # (ii) P(LB>0) vs β
    fig, ax = new_fig()
    ax.grid(False)  # 论文图 1：取消网格线
    for a in (1.0, 6.0, 12.0):
        pts = series(opb, "positive_bound_fraction").get(a)
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=A_COLORS[a], marker=A_MARKERS[a],
                markersize=4, linewidth=1.6,
                markerfacecolor="white", markeredgewidth=0.8, label=f"$a={a:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$", fontsize=PANEL_LABELSIZE)
    ax.set_ylabel(r"$P(\mathrm{LB}_{ij}>0)$", fontsize=PANEL_LABELSIZE)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=PANEL_TICKSIZE)
    ax.legend(fontsize=PANEL_LEGENDSIZE, frameon=False)
    _panel_label(ax, "(d)")
    save_fig(fig, "fig_mismatch_appendix_cls_ii_positive_fraction.png")

    # (iii) slack 分位带
    fig, ax = new_fig()
    ax.grid(False)  # 论文图 2：取消网格线
    for a in (1.0, 6.0, 12.0):
        p50 = series(opb, "slack_p50").get(a)
        p05 = series(opb, "slack_p05").get(a)
        p95 = series(opb, "slack_p95").get(a)
        if not p50:
            continue
        xs = [p[0] for p in p50]
        y50 = [p[1] for p in p50]
        lo = [q[1] for p, q in zip(p50, p05)]
        hi = [q[1] for p, q in zip(p50, p95)]
        ax.plot(xs, y50, color=A_COLORS[a], marker=A_MARKERS[a], markersize=4,
                linewidth=1.6, label=f"$a={a:g}$")
        ax.fill_between(xs, lo, hi, color=A_COLORS[a], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$", fontsize=PANEL_LABELSIZE)
    ax.set_ylabel("slack", fontsize=PANEL_LABELSIZE)
    ax.tick_params(labelsize=PANEL_TICKSIZE)
    ax.legend(fontsize=PANEL_LEGENDSIZE, frameon=False)
    _panel_label(ax, "(e)")
    save_fig(fig, "fig_mismatch_appendix_cls_iii_slack.png")


def fig_appendix_reg():
    """回归 bin 对散点：每个 β 一张独立图（ρ=6、seed 0），S_bc vs ρ|ȳ_b−ȳ_c| + y=x。"""
    betas = [1e-4, 1e-2, 1.0, 10.0]
    for beta in betas:
        path = OUT_ROOT / "detail_california_mlp" / f"california_mlp_opb_beta_{beta:g}_anchor_6_run1.npz"
        if not path.exists():
            print(f"[跳过] {path.name} 不存在")
            continue
        z = np.load(path)
        m_b, yb = z["m_b"], z["ybar_b"]
        rho = 6.0
        S = np.linalg.norm(m_b[:, None, :] - m_b[None, :, :], axis=-1)
        x = rho * np.abs(yb[:, None] - yb[None, :])
        triu = np.triu(np.ones_like(S, dtype=bool), 1)
        fig, ax = new_fig(6.4, 4.6)
        ax.scatter(x[triu], S[triu], s=14, color="#2a78d6", alpha=0.7)
        lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
        ax.plot(lims, lims, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(r"$\rho\,|\bar y_b - \bar y_c|$ (declared prior distance)")
        ax.set_ylabel(r"$S_{bc} = \|m_b - m_c\|$ (posterior bin distance)")
        ax.set_title(rf"Housing, $\rho=6$, $\beta={beta:g}$: bin separation vs. declared distance",
                     fontsize=9)
        save_fig(fig, f"fig_mismatch_appendix_reg_beta_{beta:g}.png")


def fig_appendix_reg_binscatter():
    """论文图 2(f)：回归 bin 对散点（ρ=6、seed 0），全部 β 合成一张，
    S_bc vs ρ|ȳ_b−ȳ_c|，颜色 = β（对数色标）+ y=x 参考线。"""
    betas = [1e-4, 1e-2, 1.0, 10.0]
    fig, ax = new_fig(6.4, 4.6)
    ax.grid(False)  # 论文图 2：取消网格线
    # 归一化必须在全部 β 上共享（每 scatter 新建 norm 会在单值数据上退化）
    norm = matplotlib.colors.LogNorm(vmin=min(betas), vmax=max(betas))
    sc = None
    for beta in betas:
        path = OUT_ROOT / "detail_california_mlp" / f"california_mlp_opb_beta_{beta:g}_anchor_6_run1.npz"
        if not path.exists():
            print(f"[跳过] {path.name} 不存在")
            continue
        z = np.load(path)
        m_b, yb = z["m_b"], z["ybar_b"]
        rho = 6.0
        S = np.linalg.norm(m_b[:, None, :] - m_b[None, :, :], axis=-1)
        x = rho * np.abs(yb[:, None] - yb[None, :])
        triu = np.triu(np.ones_like(S, dtype=bool), 1)
        sc = ax.scatter(x[triu], S[triu], s=14, c=np.full(int(triu.sum()), beta),
                        norm=norm, cmap="viridis", alpha=0.7)
    lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lims, lims, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"$\rho\,|\bar y_b - \bar y_c|$ (declared dist.)", fontsize=PANEL_LABELSIZE)
    ax.set_ylabel(r"$S_{bc}$", fontsize=PANEL_LABELSIZE)
    ax.tick_params(labelsize=PANEL_TICKSIZE)
    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r"$\beta$", fontsize=PANEL_LABELSIZE)
        cbar.ax.tick_params(labelsize=PANEL_TICKSIZE)
    _panel_label(ax, "(f)")
    save_fig(fig, "fig_mismatch_appendix_reg_binscatter.png")


def print_consistency_and_core(cls_data, reg_data):
    """一致性表（D vs kl_mean vs kl_cov）+ 核心趋势数值。"""
    print("\n% 一致性表（全部 OPB/EPB 配置 × seeds 的均值，nats）：")
    print(r"\begin{tabular}{l c c c}")
    print(r"\hline")
    print(r"task & $D$ & KL mean term & KL var term \\")
    print(r"\hline")
    for name, data, dcol in [("ImageNet-100", cls_data["opb"], "d_mean"),
                             ("Cal. Housing", reg_data["opb"], "d_mean")]:
        dm = np.concatenate([np.array(v[dcol]) for v in data.values() if dcol in v]).mean()
        km = np.concatenate([np.array(v["kl_mean"]) for v in data.values() if "kl_mean" in v]).mean()
        kv = np.concatenate([np.array(v["kl_cov"]) for v in data.values() if "kl_cov" in v]).mean()
        print(f"{name} & {dm:.3f} & {km:.3f} & {kv:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    print("\n核心趋势数值（seed 级 mean±std）：")
    for name, data in [("ImageNet-100 (OPB)", cls_data["opb"]),
                       ("Cal. Housing (EPB)", reg_data["opb"])]:
        for a in (1.0, 6.0, 12.0):
            d_lo = series(data, "d_mean").get(a)
            d_hi = d_lo
            if not d_lo:
                continue
            lo = min(d_lo)[1]
            hi = max(d_lo)[1]
            lo_b = min(d_lo)[0]
            hi_b = max(d_lo)[0]
            lb_hi = series(data, "lower_bound_mean").get(a)
            p_hi = series(data, "positive_bound_fraction").get(a)
            lb_txt = f", mean LB(β={hi_b:g})={lb_hi[-1][1]:.3f}" if lb_hi else ""
            pf_txt = f", P(LB>0)={p_hi[-1][1]:.3f}" if p_hi else ""
            print(f"  {name} a/ρ={a:g}: D(β={lo_b:g})={lo:.3f} → D(β={hi_b:g})={hi:.3f}{lb_txt}{pf_txt}")


def main():
    parser = argparse.ArgumentParser(description="失配分析论文图")
    args = parser.parse_args()
    _setup_rc()

    cls_data, f1 = load(OUT_ROOT / "mismatch_eval_imagenet100_mlp.csv")
    reg_data, f2 = load(OUT_ROOT / "mismatch_eval_california_mlp.csv")
    print(f"数据：分类 failed 行 {f1}、回归 failed 行 {f2}（已跳过）")
    if not cls_data.get("opb") or not reg_data.get("opb"):
        print("[跳过] 无 opb 数据，请先运行 mismatch_eval.py")
        return
    fig_panel_a(cls_data)
    fig_panel_b(cls_data)
    fig_panel_c(reg_data)
    fig_panel_d(reg_data)
    fig_appendix_cls(cls_data)
    fig_appendix_reg()
    fig_appendix_reg_binscatter()
    print_consistency_and_core(cls_data, reg_data)


if __name__ == "__main__":
    main()
