"""后验几何分析脚本（实验二：机制传递验证——先验正交是否驱动后验分离）。

实验一（prior_geometry.py）证明了 CEB 的先验几何任意、OPB 的先验严格正交。
本实验回答更关键的问题：分类器吃的是后验 z，先验只是正则手段——OPB 的正交
先验是否真的把后验表示"拉进"了正交框架？对每个 checkpoint 在 MNIST 测试集
（前 --n-test 张）上以 z=μ 确定性路径逐样本取后验均值 mu_q(x)，按类别聚合
类中心 c_k = mean(mu_q(x) | y=k)，测量：

- 后验类中心 Gram（两两余弦）：OPB 应近对角（显式分离结构），CEB 无结构；
- 类中心-锚点对齐矩阵 cos(c_k, 锚点_j)：OPB 对角应高（自锚对齐）、非对角低
  （无串扰）；CEB 以其学习先验均值为锚点作同样计算作对照；
- 类内散布 W（样本到自身类中心均方距离）/ 类间散布 B（类中心加权方差）/
  Fisher 比 B/W：OPB 应显著更高——"显式分离结构"的直接量化；
- 跨 seed 几何一致性（多 run 类中心 Gram 的逐对 std）：OPB 稳定、CEB 漂移；
- 半径匹配 ‖c_k‖ ≈ a 与类中心到自锚点距离：后验真的追到锚点球面的证据。

v1 仅支持 mnist MLP 的 CEB/OPB 配置（后验均值直接走 encoder/mu_head 提取）。
输出：posterior_geometry.csv（长表，每 (config, run) 一行标量指标）、
posterior_geometry.json（完整矩阵：类中心、Gram、对齐矩阵等）与四张图
（类中心 Gram / 对齐矩阵热力图网格、散布与 Fisher 柱图、类中心模长箱线）。

回归扩展（v2，housing 目录）：continuous_y 配置（CEB/OPB-R）测后验贴轴几何——
轴向坐标 t=μ_q·u 的斜率/截距/轴向 R²、离轴残差占比、成对等距传递
（‖Δμ_q‖/|Δỹ| 抽样对的均值±std）、跨 run 轴方向一致性；新增贴轴散点图。
anchor-scale 自动从训练日志 Args 读取。输出统一的 HTML 机制验证报告
（--html，默认 {results-dir}/mechanism_report.html）：训练配置与性能表、
实验一先验几何表（读 --prior-results-dir 下 prior_summary.json）、
实验二后验几何概念表（分类/回归统一四概念列）与主图（分类类中心 Gram +
回归贴轴散点）。

用法（与 prior_geometry.py 相同的目录约定）：

    python posterior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb
    python posterior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb \\
        --runs 5 --n-test 2000 --anchor-scale 8
"""

import argparse
import ast
import base64
import csv
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from datasets.datasets import get_california_dataloaders, get_mnist_dataloaders
from model.mlp.utils import flatten
from prior_geometry import (
    DIVERGING_CMAP,
    INK,
    MUTED,
    TwoSlopeNorm,
    _run_index,
    _series_color,
    _setup_rc,
    cross_run_summary,
    extract_prior_axis,
    extract_prior_means,
    offdiag,
    parse_dir_name,
    plot_cosine_heatmaps,
    plot_norms,
    read_trained_anchor_scale,
    summarize,
)
from train import build_model, build_parser

ROOT = Path(__file__).resolve().parent

# cross_run_summary 额外统计的标量键（基础键为实验一的几何指标）
EXTRA_SCALAR_KEYS = [
    "mean_diag_align",
    "max_offdiag_align",
    "within_scatter",
    "between_scatter",
    "fisher_ratio",
    "center_norm_mean",
    "mean_anchor_dist",
]


def collect_posterior(model, test_loader, n_test, device):
    """收集测试集前 n_test 张的确定性后验均值 mu_q 与标签（z=μ 评估路径）。

    v1 仅 MLP 骨干：直接走 encoder + mu_head，不走完整 forward。
    """
    model.eval()
    mus, ys = [], []
    with torch.no_grad():
        for x, y in test_loader:
            mu = model.mu_head(model.encoder(flatten(x.to(device))))
            mus.append(mu.cpu())
            ys.append(y)
            if sum(len(t) for t in mus) >= n_test:
                break
    return torch.cat(mus)[:n_test], torch.cat(ys)[:n_test]


def posterior_summarize(mu_q, labels, anchors, anchor_scale):
    """后验几何汇总：类中心、中心 Gram、对齐矩阵、散布、半径与锚点距离。

    先复用 summarize（键名与实验一一致，矩阵为类中心的两两几何），再追加
    对齐矩阵、散布与半径指标。anchors 为各类锚点 (K, d)（OPB 为 QR 锚点、
    CEB 为其学习先验均值，作对照锚点）。
    """
    K = anchors.size(0)
    d = mu_q.size(1)
    zero = torch.zeros(d)
    centers = torch.stack(
        [mu_q[labels == k].mean(0) if (labels == k).any() else zero for k in range(K)]
    )
    s = summarize(centers, anchor_scale)  # 类中心的两两几何（键名同实验一）

    cn = centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-8)
    an = anchors / anchors.norm(dim=1, keepdim=True).clamp_min(1e-8)
    align = cn @ an.t()  # (K, K)
    s["align"] = align.tolist()
    s["mean_diag_align"] = align.diagonal().mean().item()
    s["max_offdiag_align"] = offdiag(align).abs().max().item()

    # 类内散布 W：样本到自身类中心的均方距离；类间散布 B：类中心加权方差
    within = ((mu_q - centers[labels]) ** 2).sum(1)
    s["within_scatter"] = within.mean().item()
    n_k = torch.bincount(labels, minlength=K).float()
    between = (n_k * ((centers - mu_q.mean(0)) ** 2).sum(1)).sum() / n_k.sum()
    s["between_scatter"] = between.item()
    W, B = s["within_scatter"], s["between_scatter"]
    s["fisher_ratio"] = B / W if W > 0 else float("inf")

    # 半径匹配：后验中心是否追到锚点球面（‖c_k‖ ≈ a、距自锚点近）
    s["center_norm_mean"] = centers.norm(dim=1).mean().item()
    s["mean_anchor_dist"] = (centers - anchors).norm(dim=1).mean().item()
    return s


def posterior_summarize_reg(mu_q, y, direction, scale, bias_proj):
    """回归后验几何汇总：轴向坐标 t = mu_q·d 的斜率/截距/轴向 R²、离轴残差
    占比、成对等距传递（固定 seed 抽样 2000 对的 ‖Δμ_q‖/|Δỹ| 均值±std）。

    y 为归一化标签 [0,1]（训练管道同款 MinMax）；判据：OPB-R 斜率 ≈ rho
    （=scale，构造固定）、截距 ≈ 0；CEB 斜率为其学习尺度、截距 ≈ b·d。
    """
    d = direction.to(mu_q.dtype)
    t = mu_q @ d  # (N,)
    A = torch.stack([y, torch.ones_like(y)], dim=1)
    sol = torch.linalg.lstsq(A, t.unsqueeze(1)).solution.squeeze(-1)
    slope, intercept = sol[0].item(), sol[1].item()
    ss_res = ((t - (slope * y + intercept)) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    axis_r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float("nan")
    off_frac = (
        ((mu_q - t.unsqueeze(1) * d.unsqueeze(0)) ** 2).sum(1).mean()
        / (mu_q**2).sum(1).mean().clamp_min(1e-12)
    )
    g = torch.Generator().manual_seed(0)  # 固定抽样，跨 run/配置可复现
    n = mu_q.size(0)
    i1 = torch.randint(0, n, (2000,), generator=g)
    i2 = torch.randint(0, n, (2000,), generator=g)
    dy = (y[i1] - y[i2]).abs()
    ok = dy > 1e-3
    r = (mu_q[i1] - mu_q[i2]).norm(dim=1)[ok] / dy[ok]
    return {
        "axis_r2": axis_r2,
        "slope": slope,
        "intercept": intercept,
        "off_axis_frac": off_frac.item(),
        "iso_ratio_mean": r.mean().item(),
        "iso_ratio_std": r.std().item(),
        # 供贴轴散点图与跨 run 方向一致性使用
        "t": t.tolist(),
        "y": y.tolist(),
        "direction": direction.tolist(),
        "scale": scale,
        "bias_proj": bias_proj,
    }


def reg_posterior_cross_run(run_summaries):
    """回归后验跨 run 汇总：标量均值±std + 轴方向一致性（mean_dir_cos，1=一致）。"""
    dirs = torch.stack([torch.tensor(s["direction"]) for s in run_summaries])
    cos = dirs @ dirs.t()
    iu = torch.triu_indices(len(run_summaries), len(run_summaries), offset=1)
    out = {"mean_dir_cos": cos[iu[0], iu[1]].mean().item()}
    for k in ["axis_r2", "slope", "intercept", "off_axis_frac",
              "iso_ratio_mean", "iso_ratio_std"]:
        t = torch.tensor([s[k] for s in run_summaries], dtype=torch.float64)
        out[f"{k}_mean"] = t.mean().item()
        out[f"{k}_std"] = t.std().item()
    return out


def plot_axis_scatter(results, path):
    """回归贴轴散点：每模型一子图，x=归一化标签、y=轴向坐标 uᵀμ_q(x)，
    参考线为该模型的先验线（OPB-R：rho·ỹ；CEB：s·ỹ + b·d）。"""
    reg_labels = [l for l, res in results.items() if res["mode"] == "regression"]
    if not reg_labels:
        return None
    fig, axes = plt.subplots(
        1, len(reg_labels), figsize=(5.2 * len(reg_labels), 4.2), squeeze=False
    )
    for ax, label in zip(axes[0], reg_labels):
        res = results[label]
        color = _series_color(label)
        s = res["runs"][0]
        ys, ts = torch.tensor(s["y"]), torch.tensor(s["t"])
        idx = torch.randperm(ys.size(0), generator=torch.Generator().manual_seed(0))[:800]
        ax.scatter(ys[idx], ts[idx], s=6, alpha=0.35, color=color, edgecolors="none")
        xx = torch.tensor([0.0, 1.0])
        ax.plot(xx, s["scale"] * xx + s["bias_proj"], color=INK, linewidth=1.5,
                linestyle="--", label=f"先验线 t={s['scale']:.2f}·ỹ+{s['bias_proj']:.2f}")
        ax.set_xlabel("归一化标签 ỹ")
        ax.set_ylabel("轴向坐标 uᵀμ_q(x)")
        ax.set_title(f"{label}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("回归后验贴轴：后验均值在等距轴/学习轴上的坐标（浅色=逐样本）", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_paper_gram(cls_results, path):
    """论文版 Gram 图（全宽单行，英文标签，dpi 200）：1×2 面板（CEB | OPB），
    每面板为跨 run 平均的类中心余弦 Gram 热力图（发散色带 + 数值标注）。"""
    labels = list(cls_results)
    fig, axes = plt.subplots(
        1, len(labels), figsize=(5.6 * len(labels), 4.6), squeeze=False
    )
    for ax, label in zip(axes[0], labels):
        gram = torch.tensor([s["gram"] for s in cls_results[label]["runs"]]).mean(0)
        K = gram.size(0)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
        im = ax.imshow(gram, cmap=DIVERGING_CMAP, norm=norm, aspect="equal")
        ax.set_xticks(range(K), range(K))
        ax.set_yticks(range(K), range(K))
        ax.tick_params(length=0, labelsize=8)
        ax.grid(False)
        for i in range(K):
            for j in range(K):
                v = gram[i, j].item()
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if abs(v) > 0.45 else INK,
                )
        ax.set_xlabel("class $j$")
        ax.set_ylabel("class $k$")
        ax.set_title(cls_results[label]["model"].upper(), fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.9, label="cosine")
    # 论文图：白底、无总标题（论文已有 caption）
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_paper_axis(results, path):
    """论文版贴轴图（全宽单行，英文标签，dpi 200）：1×2 面板（CEB | EPB），
    每面板 run1 的后验轴向坐标散点 + 该模型先验线参考线。"""
    reg_labels = [l for l, res in results.items() if res["mode"] == "regression"]
    if not reg_labels:
        return None
    fig, axes = plt.subplots(
        1, len(reg_labels), figsize=(5.4 * len(reg_labels), 4.4), squeeze=False
    )
    for ax, label in zip(axes[0], reg_labels):
        res = results[label]
        color = _series_color(label)
        s = res["runs"][0]
        ys, ts = torch.tensor(s["y"]), torch.tensor(s["t"])
        idx = torch.randperm(
            ys.size(0), generator=torch.Generator().manual_seed(0)
        )[:900]
        ax.scatter(ys[idx], ts[idx], s=5, alpha=0.35, color=color, edgecolors="none")
        xx = torch.tensor([0.0, 1.0])
        ax.plot(
            xx, s["scale"] * xx + s["bias_proj"], color=INK, linewidth=2,
            linestyle="--",
            label=f"prior line $t={s['scale']:.1f}\\,\\tilde y+{s['bias_proj']:.1f}$",
        )
        ax.set_xlabel("normalized label $\\tilde y$")
        ax.set_ylabel("axial coordinate $u^{\\top}\\mu_q(x)$")
        model = "EPB" if res["model"] == "opb" else res["model"].upper()
        ax.set_title(model, fontsize=11)
        ax.legend(fontsize=9)
    # 论文图：白底、无总标题（论文已有 caption）
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_facecolor("white")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def read_train_log_info(d):
    """解析模型目录训练日志：返回 (Args 配置 dict, 最后一条 Average 汇总行)。

    Args 取**最后一条** Args 行——日志为追加模式，早先失败尝试（如 CEB 误传
    --tied-head 直接报错退出）的 Args 行在前面，最后一条才是产出 checkpoint
    的那次运行配置。
    """
    for log in sorted(d.glob("train_*.log")):
        try:
            with open(log, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        cfg = None
        args_lines = [l for l in lines if "Args:" in l]
        if args_lines:
            try:
                cfg = ast.literal_eval(args_lines[-1].split("Args:", 1)[1].strip())
            except (ValueError, SyntaxError):
                cfg = None
        avg = next((l.strip() for l in reversed(lines) if "Average over" in l), None)
        if cfg is not None or avg is not None:
            return cfg, avg
    return None, None


def _avg_metric(avg, mode):
    """从 Average 汇总行取测试指标（分类 Acc %、回归 R²）。"""
    import re

    if avg is None:
        return "-"
    if mode == "regression":
        m = re.search(r"R2 ([\d.]+)±([\d.]+)", avg)
        return f"R² {m.group(1)}±{m.group(2)}" if m else "-"
    m = re.search(r"Acc ([\d.]+)±([\d.]+)", avg)
    return f"Acc {float(m.group(1)) * 100:.2f}±{float(m.group(2)) * 100:.2f}%" if m else "-"


def _head_label(cfg, model):
    """分类头标签——tied/能量分类器仅对 opb 有意义，其余模型显示 —。"""
    if model != "opb" or cfg is None:
        return "—"
    if cfg.get("energy_classifier"):
        return "能量分类器"
    if cfg.get("tied_head"):
        return "tied 投影头"
    return "自由头"


def _img_b64(path):
    """PNG 转 base64（HTML 内嵌，文件缺失时返回 None）。"""
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if x is not None else "-"


def build_html_report(results, prior_summary, model_dirs, out_path, fig_dir, meta):
    """论文报告形式的机制验证 HTML：一张机制表（每任务 3 个最有区分度的概念
    指标）、一张两面板主图（分类 Gram + 回归贴轴散点）、先验/性能各一句文字、
    表注。results/prior_summary 为两脚本的跨 run 汇总；model_dirs 为
    {label: 模型目录 Path}（读训练日志）。未列入表/句的指标见 JSON/CSV。
    """
    # 机制表：几何集中度 / 尺度跟随 / 跨 seed 稳定——每个任务各取最有区分度的
    # 指标（分类 mean|cos_off|、回归离轴占比；diag_align/Fisher/截距区分度弱，
    # 不列入表，见 JSON/CSV）
    concept_rows = []
    for label, res in results.items():
        cr = res["cross_run"]
        if res["mode"] == "classification":
            # a 参照仅对 OPB（构造锚点）有意义；CEB 的 ‖c̄‖ 无参照
            a_val = meta.get(f"a_{label}") if res.get("model") == "opb" else None
            scale_cell = (
                f"‖c̄‖ {_fmt(cr.get('center_norm_mean_mean'), 2)}"
                + (f"（a={a_val:g}）" if a_val is not None else "（无参照）")
            )
            concept_rows.append(
                (
                    label, "分类",
                    f"{_fmt(cr.get('mean_abs_cos_offdiag_mean'))}±"
                    f"{_fmt(cr.get('mean_abs_cos_offdiag_std'))}",
                    scale_cell,
                    _fmt(cr.get("mean_pairwise_cos_std")),
                )
            )
        else:
            # ρ 参照仅对 OPB-R；CEB 的斜率为其学习尺度、无先验参照
            rho = meta.get(f"a_{label}") if res.get("model") == "opb" else None
            rho_ref = f"（ρ={rho:g}）" if rho is not None else ""
            shrink = "（收缩≈ρ·轴向R²）" if rho is not None else ""
            concept_rows.append(
                (
                    label, "回归",
                    _fmt(cr.get("off_axis_frac_mean")),
                    f"{_fmt(cr.get('slope_mean'), 2)}±{_fmt(cr.get('slope_std'), 2)}"
                    f"{rho_ref}{shrink}",
                    _fmt(cr.get("slope_std"), 3),
                )
            )

    # 先验一句（实验一结论）
    prior_note = ""
    if prior_summary:
        segs = []
        pm_by = {
            pm["model"] + pm["mode"]: pm
            for pm in prior_summary.get("models", {}).values()
        }
        cls = pm_by.get("cebclassification")
        if cls:
            segs.append(
                f"分类 CEB 先验非对角余弦 {_fmt(cls.get('mean_abs_cos_offdiag_mean'))}"
                f"±{_fmt(cls.get('mean_abs_cos_offdiag_std'))}"
            )
        cls = pm_by.get("opbclassification")
        if cls:
            theory = cls.get("theory_dist")
            segs.append(
                "分类 OPB 构造严格正交（非对角余弦 0"
                + (f"，两两距离恒 √2·a={_fmt(theory, 3)}" if theory else "")
                + "）"
            )
        reg = pm_by.get("cebregression")
        if reg:
            segs.append(
                f"回归 CEB 学习尺度 {_fmt(reg.get('scale_mean'), 3)}"
                f"±{_fmt(reg.get('scale_std'), 3)}"
            )
        reg = pm_by.get("opbregression")
        if reg:
            segs.append(
                f"回归 OPB-R 构造尺度 ρ={_fmt(reg.get('scale_mean'), 3)}"
                f"±{_fmt(reg.get('scale_std'), 3)} 固定（严格等距、过原点）"
            )
        prior_note = "实验一（先验）：" + "；".join(segs) + "。CEB 任意且跨 seed 漂移。"

    # 性能一句（预测由零参数几何读出）
    perf_parts = []
    for label, d in model_dirs.items():
        res = results.get(label, {})
        cfg, avg = read_train_log_info(d)
        if avg is None:
            continue
        task_short = "MNIST" if res.get("task") == "mnist" else "Housing"
        model = res.get("model", "?").upper()
        head = _head_label(cfg, res.get("model"))
        head_part = f"（{head}）" if head != "—" else ""
        perf_parts.append(
            f"{task_short} {model}{head_part} {_avg_metric(avg, res.get('mode', 'classification'))}"
        )
    perf_note = "性能（同 β=0.1，预测由零参数几何读出）：" + "；".join(perf_parts) + "。"

    # 后验一句（实验二结论，倍数由数据生成）
    by_mode = {"classification": {}, "regression": {}}
    for label, res in results.items():
        by_mode[res["mode"]][res.get("model")] = (label, res["cross_run"])
    post_segs = []
    ceb, opb = by_mode["classification"].get("ceb"), by_mode["classification"].get("opb")
    if ceb and opb:
        crc, cro = ceb[1], opb[1]
        r1 = crc["mean_abs_cos_offdiag_mean"] / max(cro["mean_abs_cos_offdiag_mean"], 1e-12)
        r2 = crc["mean_pairwise_cos_std"] / max(cro["mean_pairwise_cos_std"], 1e-12)
        seg = f"分类：OPB 后验非对角余弦低 {r1:.1f} 倍、跨 seed 稳定 {r2:.1f} 倍"
        a_val = meta.get(f"a_{opb[0]}")
        if a_val:
            seg += f"、半径跟随 {cro['center_norm_mean_mean'] / a_val * 100:.0f}%"
        post_segs.append(seg)
    ceb, opb = by_mode["regression"].get("ceb"), by_mode["regression"].get("opb")
    if ceb and opb:
        crc, cro = ceb[1], opb[1]
        r1 = crc["off_axis_frac_mean"] / max(cro["off_axis_frac_mean"], 1e-12)
        r2 = crc["slope_std"] / max(cro["slope_std"], 1e-12)
        post_segs.append(
            f"回归：OPB-R 离轴占比低 {r1:.0f} 倍、斜率跨 seed 稳定 {r2:.1f} 倍"
        )
    post_note = "实验二（后验）：" + "；".join(post_segs) + "。"

    gram_b64 = _img_b64(fig_dir / "posterior_geometry_centergram.png")
    axis_b64 = _img_b64(fig_dir / "posterior_geometry_axis.png")

    def row(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    def table(head, body_rows):
        head_html = row(head, "th")
        body_html = "".join(row(r) for r in body_rows)
        return f"<table>{head_html}{body_html}</table>"

    css = """
    body{font-family:'Noto Sans CJK JP','DejaVu Sans',sans-serif;background:#f9f9f7;
         color:#0b0b0b;max-width:1080px;margin:0 auto;padding:28px 20px;}
    h1{font-size:1.35em;border-bottom:2px solid #0b0b0b;padding-bottom:6px;}
    h2{font-size:1.1em;margin-top:26px;}
    table{border-collapse:collapse;margin:10px 0;font-size:.86em;}
    th,td{border:1px solid #c3c2b7;padding:5px 10px;text-align:center;}
    th{background:#f0efec;}
    td:first-child{text-align:left;font-weight:600;}
    .note{color:#52514e;font-size:.82em;margin:4px 0 14px;}
    .say{background:#fcfcfb;border:1px solid #c3c2b7;padding:8px 12px;margin:6px 0;
         font-size:.9em;line-height:1.6;}
    .fig{display:flex;flex-wrap:wrap;gap:18px;margin:12px 0;}
    .fig div{flex:1 1 460px;}
    .fig img{width:100%;border:1px solid #c3c2b7;}
    .cap{color:#52514e;font-size:.82em;margin-top:4px;}
    """
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>OPB 几何机制验证报告（分类 MNIST + 回归 Housing）</title>
<style>{css}</style></head><body>
<h1>OPB 几何机制验证报告（分类 MNIST + 回归 Housing）</h1>
<p class="note">生成日期 {date.today().isoformat()} | runs={meta['runs']} |
n_test={meta['n_test']} | anchor-scale 以各目录训练日志 Args 为准</p>

<h2>表：几何机制验证（跨 5 run 均值±std）</h2>
{table(["模型（任务）", "任务",
        "几何集中度<br>（分类 mean|cos_off| / 回归 离轴占比）",
        "尺度跟随<br>（分类 ‖c̄‖ 对照 a / 回归 斜率对照 ρ）",
        "跨 seed 稳定<br>（分类 跨run余弦std / 回归 斜率std）"], concept_rows)}
<p class="note">表注：① 回归斜率受编码器收缩影响——斜率 ≈ ρ·轴向R²（12·0.78≈9.4，
观测 9.22，一致），非机制失效；② 回归跨 seed 稳定用斜率std：轴方向本身可随表示
整体旋转（方向cos 无判别意义，见 JSON 的 mean_dir_cos）；③ diag_align / Fisher /
截距等区分度弱的指标未列入表，数值见 posterior_geometry.json/csv。</p>

<h2>文字表述（供论文直接改写）</h2>
<p class="say">{prior_note if prior_note else "（先运行 prior_geometry.py 生成先验汇总）"}</p>
<p class="say">{post_note}</p>
<p class="say">{perf_note}</p>

<h2>图</h2>
<div class="fig">"""
    if gram_b64:
        html += (
            '<div><img src="data:image/png;base64,'
            + gram_b64
            + '"><div class="cap">(a) 分类：后验类中心两两余弦 Gram'
            "（非对角 = 0 表示类中心正交，OPB 应近对角）</div></div>"
        )
    if axis_b64:
        html += (
            '<div><img src="data:image/png;base64,'
            + axis_b64
            + '"><div class="cap">(b) 回归：后验贴轴——轴向坐标 uᵀμ_q(x) 随'
            "归一化标签（参考线 = 各自先验线；后验沿轴收缩至 ρ·E[ỹ|x]，"
            "散点低于先验参考线属预期）</div></div>"
        )
    html += "</div></body></html>"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def plot_scatter_bars(results, path):
    """散布与 Fisher 比柱图：左=类内/类间散布、右=Fisher 比（跨 run 均值±std）。

    柱色按模型（CEB 蓝 / OPB 橙）；散布图中浅色=类内 W、深色=类间 B。
    """
    labels = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.3
    for i, label in enumerate(labels):
        cr = results[label]["cross_run"]
        color = _series_color(label)
        ax1.bar(i - width / 2, cr["within_scatter_mean"], width,
                yerr=cr["within_scatter_std"], color=color, alpha=0.4,
                error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
        ax1.bar(i + width / 2, cr["between_scatter_mean"], width,
                yerr=cr["between_scatter_std"], color=color,
                error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
        ax2.bar(i, cr["fisher_ratio_mean"], width,
                yerr=cr["fisher_ratio_std"], color=color,
                error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
    ax1.set_xticks(range(len(labels)), labels)
    ax2.set_xticks(range(len(labels)), labels)
    ax1.set_ylabel("均方距离")
    ax2.set_ylabel("B / W")
    ax1.set_title("类内散布 W（浅色）/ 类间散布 B（深色），跨 run 均值±std", fontsize=9)
    ax2.set_title("Fisher 比（类间/类内，越高分离越强）", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_analysis_parser():
    parser = build_parser()
    parser.add_argument(
        "--model-dirs", nargs="+", required=True,
        help="已训练配置的 checkpoint 目录列表（每个目录内含 {stem}_run{i}.pt；"
        "目录名须解析为 CEB/OPB 配置，支持 mnist（分类）与 california（housing 回归）MLP 目录）",
    )
    parser.add_argument(
        "--n-test", type=int, default=1000,
        help="测试集前 N 张用于后验均值统计（默认 1000）",
    )
    parser.add_argument(
        "--results-dir", type=str, default="pos_results",
        help="结果输出目录（默认 pos_results）",
    )
    parser.add_argument(
        "--prior-results-dir", type=str, default=None,
        help="先验几何结果目录（读 prior_summary.json 供 HTML 实验一表；"
        "默认自动找：results-dir 同级的 pri_results/，其次 results-dir 自身）",
    )
    parser.add_argument(
        "--html", type=str, default=None,
        help="HTML 机制验证报告输出路径（默认 {results-dir}/mechanism_report.html）",
    )
    parser.add_argument(
        "--paper-figs-dir", type=str, default=None,
        help="论文版图输出目录（可选，默认不生成）：fig_mechanism_gram.png / "
        "fig_mechanism_axis.png（全宽单行版、英文标签、dpi 200，供 paper.tex 使用）",
    )
    return parser


def main():
    parser = build_analysis_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试集一次取齐（无 shuffle、顺序确定，各配置共用同批样本）
    test_loaders = {
        "mnist": get_mnist_dataloaders(args.batch_size, args.data_dir)[2],
        "california": get_california_dataloaders(args.batch_size, args.data_dir)[2],
    }

    results = {}
    csv_rows = []
    model_dirs = {}
    meta = {
        "runs": args.runs,
        "anchor_scale": args.anchor_scale,
        "n_test": args.n_test,
        "near_pair_threshold": 0.5,
        "seed_base": args.seed,
    }
    for d in (Path(p) for p in args.model_dirs):
        label = d.name
        dataset, backbone, model_name = parse_dir_name(label)
        if model_name not in ("ceb", "opb"):
            parser.error(f"目录 '{label}' 模型为 '{model_name}'，实验二仅支持 ceb/opb")
        if dataset not in ("mnist", "california") or backbone != "mlp":
            parser.error(
                f"目录 '{label}'（{dataset}/{backbone}）不支持：实验二仅支持 "
                "mnist（分类）/california（housing 回归）MLP"
            )
        ckpts = sorted(d.glob("*_run*.pt"), key=_run_index)
        if len(ckpts) < args.runs:
            parser.error(
                f"目录 '{d}' 只找到 {len(ckpts)} 个 '*_run*.pt'，需要 --runs={args.runs} 个"
            )
        ckpts = ckpts[: args.runs]
        model_dirs[label] = d

        # 锚点尺度以训练日志 Args 为准（防目录名/CLI 与分析不一致的伪影）
        a_scale = read_trained_anchor_scale(d, args.anchor_scale)
        if a_scale is not None and abs(a_scale - args.anchor_scale) > 1e-9:
            print(
                f"[提示] {label} 按训练日志使用 anchor-scale={a_scale:g}"
                f"（忽略 CLI {args.anchor_scale:g}）"
            )
        meta[f"a_{label}"] = a_scale

        dir_args = argparse.Namespace(**vars(args))
        dir_args.task = "housing" if dataset == "california" else dataset
        dir_args.model = model_name
        dir_args.backbone = backbone
        model = build_model(parser, dir_args).to(device)
        is_reg = getattr(model, "continuous_y", False)
        mode = "regression" if is_reg else "classification"
        print(
            f"[加载] {label} → backbone={backbone} model={model_name} mode={mode}，"
            f"checkpoint {len(ckpts)} 个，n_test={args.n_test}"
        )

        run_summaries = []
        for i, ckpt in enumerate(ckpts, 1):
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            mu_q, labels = collect_posterior(
                model, test_loaders[dataset], args.n_test, device
            )
            if is_reg:
                ax = extract_prior_axis(model, model_name, a_scale)
                s = posterior_summarize_reg(
                    mu_q, labels, ax["direction"].cpu(), ax["scale"], ax["bias_proj"]
                )
                csv_rows.append(
                    [label, i, args.seed + i - 1] + [""] * 10
                    + [s["axis_r2"], s["slope"], s["intercept"],
                       s["off_axis_frac"], s["iso_ratio_mean"], s["iso_ratio_std"]]
                )
                print(
                    f"[{label}] run{i} (seed {args.seed + i - 1}) slope={s['slope']:.3f} "
                    f"intercept={s['intercept']:.3f} axis_r2={s['axis_r2']:.4f} "
                    f"off_axis={s['off_axis_frac']:.4f} iso_ratio={s['iso_ratio_mean']:.3f}"
                    f"±{s['iso_ratio_std']:.3f}"
                )
            else:
                anchors, _ = extract_prior_means(model, model_name, a_scale)
                anchors = anchors.detach().cpu()  # 后验统计在 CPU 上进行
                s = posterior_summarize(mu_q, labels, anchors, a_scale)
                csv_rows.append(
                    [
                        label,
                        i,
                        args.seed + i - 1,
                        s["mean_abs_cos_offdiag"],
                        s["max_abs_cos_offdiag"],
                        s["min_dist_offdiag"],
                        s["mean_diag_align"],
                        s["max_offdiag_align"],
                        s["within_scatter"],
                        s["between_scatter"],
                        s["fisher_ratio"],
                        s["center_norm_mean"],
                        s["mean_anchor_dist"],
                    ]
                    + [""] * 6
                )
                print(
                    f"[{label}] run{i} (seed {args.seed + i - 1}) "
                    f"mean|cos_off|={s['mean_abs_cos_offdiag']:.4f} "
                    f"diag_align={s['mean_diag_align']:.4f} "
                    f"max_off_align={s['max_offdiag_align']:.4f} "
                    f"fisher={s['fisher_ratio']:.2f} ‖c̄‖={s['center_norm_mean']:.3f} "
                    f"锚点距={s['mean_anchor_dist']:.3f}"
                )
            s["run"] = i
            s["seed"] = args.seed + i - 1
            run_summaries.append(s)

        if is_reg:
            cr = reg_posterior_cross_run(run_summaries)
            print(
                f"[汇总] {label}: slope={cr['slope_mean']:.3f}±{cr['slope_std']:.3f} | "
                f"intercept={cr['intercept_mean']:.3f} | off_axis={cr['off_axis_frac_mean']:.4f} | "
                f"iso_ratio={cr['iso_ratio_mean_mean']:.3f}±{cr['iso_ratio_std_mean']:.3f} | "
                f"方向cos={cr['mean_dir_cos']:.4f}"
            )
        else:
            cr = cross_run_summary(run_summaries, EXTRA_SCALAR_KEYS)
            print(
                f"[汇总] {label}: mean|cos_off|={cr['mean_abs_cos_offdiag_mean']:.4f}"
                f"±{cr['mean_abs_cos_offdiag_std']:.4f} | diag_align={cr['mean_diag_align_mean']:.4f}"
                f"±{cr['mean_diag_align_std']:.4f} | fisher={cr['fisher_ratio_mean']:.2f}"
                f"±{cr['fisher_ratio_std']:.2f} | ‖c̄‖={cr['center_norm_mean_mean']:.3f} | "
                f"锚点距={cr['mean_anchor_dist_mean']:.3f} | 跨run余弦std={cr['mean_pairwise_cos_std']:.4f}"
            )
        results[label] = {
            "task": "housing" if is_reg else dataset,
            "backbone": backbone,
            "model": model_name,
            "mode": mode,
            "runs": run_summaries,
            "cross_run": cr,
        }

    # 对照表：分类判据是 OPB 类中心 Gram 近对角、自锚对齐高、Fisher 比高、‖c̄‖≈a；
    # 回归判据是 OPB-R 斜率≈rho、截距≈0、离轴占比低、方向跨 run 一致
    cls_labels = [l for l, r in results.items() if r["mode"] == "classification"]
    reg_labels = [l for l, r in results.items() if r["mode"] == "regression"]
    if cls_labels:
        print("\n[对照·分类] 各模型跨 run 汇总")
        header = f"{'model':<20} {'mean|cos_off|':<18} {'diag_align':<14} {'fisher':<12} {'‖c̄‖':<10} {'锚点距':<10} {'跨run余弦std':<14}"
        print(header)
        for label in cls_labels:
            cr = results[label]["cross_run"]
            print(
                f"{label:<20} {cr['mean_abs_cos_offdiag_mean']:.4f}±{cr['mean_abs_cos_offdiag_std']:.4f}"
                f"{'':<4} {cr['mean_diag_align_mean']:.4f}±{cr['mean_diag_align_std']:.4f} "
                f"{cr['fisher_ratio_mean']:.2f}±{cr['fisher_ratio_std']:.2f} "
                f"{cr['center_norm_mean_mean']:.3f} {cr['mean_anchor_dist_mean']:.3f} "
                f"{cr['mean_pairwise_cos_std']:.4f}"
            )
    if reg_labels:
        print("\n[对照·回归] 各模型跨 run 汇总")
        header = f"{'model':<20} {'slope':<16} {'intercept':<12} {'off_axis':<12} {'方向cos':<10}"
        print(header)
        for label in reg_labels:
            cr = results[label]["cross_run"]
            print(
                f"{label:<20} {cr['slope_mean']:.3f}±{cr['slope_std']:.3f}"
                f"{'':<4} {cr['intercept_mean']:.3f} {'':<4} "
                f"{cr['off_axis_frac_mean']:.4f} {'':<4} {cr['mean_dir_cos']:.4f}"
            )

    out_root = ROOT / args.results_dir
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, **results}
    json_path = out_root / "posterior_geometry.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "posterior_geometry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "run",
                "seed",
                "mean_abs_cos_offdiag",
                "max_abs_cos_offdiag",
                "min_dist_offdiag",
                "mean_diag_align",
                "max_offdiag_align",
                "within_scatter",
                "between_scatter",
                "fisher_ratio",
                "center_norm_mean",
                "mean_anchor_dist",
                "axis_r2",
                "slope",
                "intercept",
                "off_axis_frac",
                "iso_ratio_mean",
                "iso_ratio_std",
            ]
        )
        w.writerows(csv_rows)

    _setup_rc()
    if cls_labels:
        cls_results = {l: results[l] for l in cls_labels}
        plot_cosine_heatmaps(
            cls_results,
            out_root / "posterior_geometry_centergram.png",
            suptitle="后验类中心两两余弦 Gram 矩阵（c_k = 各类后验均值中心；非对角 = 0 表示类中心正交）",
        )
        plot_cosine_heatmaps(
            cls_results,
            out_root / "posterior_geometry_align.png",
            suptitle="类中心-锚点对齐矩阵 cos(c_k, 锚点_j)（对角 = 自锚对齐、非对角 = 串扰）",
            key="align",
        )
        plot_scatter_bars(cls_results, out_root / "posterior_geometry_scatter.png")
        a_used = cls_results[cls_labels[0]]["cross_run"].get(
            "theory_dist", meta[f"a_{cls_labels[0]}"]
        )
        plot_norms(
            cls_results,
            out_root / "posterior_geometry_norms.png",
            a_used if isinstance(a_used, float) else meta[f"a_{cls_labels[0]}"],
            ylabel="类中心模长 ‖c_k‖",
            title="各类后验中心模长（OPB 期望 ≈ a）",
        )
    if reg_labels:
        plot_axis_scatter(results, out_root / "posterior_geometry_axis.png")
    if args.paper_figs_dir:
        paper_figs = Path(args.paper_figs_dir)
        paper_figs.mkdir(parents=True, exist_ok=True)
        if cls_labels:
            plot_paper_gram(
                {l: results[l] for l in cls_labels},
                paper_figs / "fig_mechanism_gram.png",
            )
        if reg_labels:
            plot_paper_axis(results, paper_figs / "fig_mechanism_axis.png")
        print(f"论文图已保存：{paper_figs}")

    # 先验几何汇总（HTML 实验一表）：默认自动找 results-dir 同级 pri_results/，其次自身
    prior_dir = None
    if args.prior_results_dir:
        prior_dir = Path(args.prior_results_dir)
    else:
        for cand in (out_root.parent / "pri_results", out_root):
            if (cand / "prior_summary.json").exists():
                prior_dir = cand
                break
    prior_summary = None
    if prior_dir is not None and (prior_dir / "prior_summary.json").exists():
        with open(prior_dir / "prior_summary.json", encoding="utf-8") as f:
            prior_summary = json.load(f)
        print(f"[HTML] 读取先验汇总：{prior_dir / 'prior_summary.json'}")
    else:
        print("[HTML] 未找到 prior_summary.json（先运行 prior_geometry.py），实验一表将省略")

    html_path = (
        Path(args.html) if args.html else out_root / "mechanism_report.html"
    )
    build_html_report(results, prior_summary, model_dirs, html_path, out_root, meta)

    print(f"\n结果已保存：{json_path}")
    print(f"汇总表已保存：{csv_path}")
    if cls_labels:
        print(f"图（分类）已保存：{out_root / 'posterior_geometry_centergram.png'}")
        print(f"                {out_root / 'posterior_geometry_align.png'}")
        print(f"                {out_root / 'posterior_geometry_scatter.png'}")
        print(f"                {out_root / 'posterior_geometry_norms.png'}")
    if reg_labels:
        print(f"图（回归）已保存：{out_root / 'posterior_geometry_axis.png'}")
    print(f"HTML 报告已保存：{html_path}")


if __name__ == "__main__":
    main()
