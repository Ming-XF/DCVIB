"""压缩-精度曲线（β 轴版）：每任务两张图——β-acc（横轴 β、纵轴 Acc/R²）与
β-KL（横轴 β、纵轴 E[KL] 对数刻度）；CEB 1 条线、GPB 三条线（分类 OPB a=1/6/12、
回归 EPB ρ=1/6/12）。imagenet100 为分类、housing（california）为回归，共四张图。

数据来自 compression_eval.py 的 output/compression_eval/compression_eval_summary.csv。
输出：
1. 论文图（白底、无总标题、英文标签，精确点不插值）：
   paper/figures/fig_beta_acc_imagenet100.png / fig_beta_acc_housing.png
   paper/figures/fig_beta_kl_imagenet100.png  / fig_beta_kl_housing.png
2. HTML 报告 output/compression_eval/compression_eval.html —— 每任务一张数据表
   （CE / KL 均值失配项 / KL 方差失配项 / E[KL] / 任务指标）与两张交互曲线卡
   （β-acc、β-KL，全系列共享 15 点 β 网格、无需插值）。

用法：
    python compression_plot.py
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prior_geometry import _setup_rc
from utils import curve_card, curve_css, curve_script, plot_bounds

ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "output" / "compression_eval" / "compression_eval_summary.csv"
HTML_PATH = ROOT / "output" / "compression_eval" / "compression_eval.html"
FIG_DIR = ROOT / "paper" / "figures"

# 每系列独立颜色（HTML 曲线卡按系列顺序自动取调色板，PNG 版用此表）
TASK_DISPLAY = {"imagenet100": "ImageNet-100", "housing": "Cal. Housing"}
SERIES_COLORS = {
    "CEB": "#2a78d6",
    "OPB (a=1)": "#eb6834",
    "OPB (a=6)": "#4f9a5f",
    "OPB (a=12)": "#9a6ac4",
    "EPB (ρ=1)": "#eb6834",
    "EPB (ρ=6)": "#4f9a5f",
    "EPB (ρ=12)": "#9a6ac4",
}


def load():
    """读汇总 csv → {(task, model, beta, anchor): dict}。"""
    rows = {}
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            anchor = f"{float(r['anchor']):g}" if r["anchor"] else ""
            key = (r["task"], r["model"], f"{float(r['beta']):g}", anchor)
            rows[key] = {
                "kl": float(r["KL_mean"]),
                "kl_std": float(r["KL_std"]),
                "acc": float(r["Acc_mean"]) if r["Acc_mean"] else None,
                "acc_std": float(r["Acc_std"]) if r["Acc_std"] else None,
                "r2": float(r["R2_mean"]) if r["R2_mean"] else None,
                "r2_std": float(r["R2_std"]) if r["R2_std"] else None,
                "ce": float(r["CE_mean"]),
                "ce_std": float(r["CE_std"]),
                "klm": float(r["KL_mean_mean"]) if r["KL_mean_mean"] else None,
                "klm_std": float(r["KL_mean_std"]) if r["KL_mean_std"] else None,
                "klv": float(r["KL_var_mean"]) if r["KL_var_mean"] else None,
                "klv_std": float(r["KL_var_std"]) if r["KL_var_std"] else None,
            }
    return rows


def series_points_beta(rows, task, metric):
    """各系列的点列表：[(label, [(beta, y), ...])]，按 β 升序连接（15 点网格）。

    metric ∈ {"acc", "kl"}：acc 为任务指标（分类 Acc×100、回归 R²），
    kl 为测试集 E[KL]（≤0 的数值伪影 clamp 到 1e-8，保证可对数化）。
    横轴为 β，不再做坍缩点/爆炸点/低精度端点过滤——β-acc 图上低精度端点
    正是压缩崩坏要展示的内容，β-KL 图对数纵轴天然容纳爆炸点。
    """
    is_cls = task == "imagenet100"
    betas = sorted({float(b) for (t, m, b, a) in rows if t == task})
    out = []
    for model, anchors in (("ceb", [""]), ("opb", ["1", "6", "12"])):
        for a in anchors:
            pts = []
            for b in betas:
                r = rows.get((task, model, f"{b:g}", a))
                if r is None:
                    continue
                if metric == "acc":
                    y = r["acc"] * 100 if is_cls else r["r2"]
                else:
                    y = max(r["kl"], 1e-8)
                pts.append((b, y))
            if not pts:
                continue
            if model == "ceb":
                label = "CEB"
            else:
                label = f"OPB (a={a})" if is_cls else f"EPB (ρ={a})"
            out.append((label, pts))
    return out


# ---------------- 论文 PNG（精确点，不插值） ----------------

def plot_task_metric(task, metric, rows):
    """单任务单指标论文图：x = β（对数刻度），y = Acc/R² 或 E[KL]（对数刻度）。"""
    is_cls = task == "imagenet100"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, pts in series_points_beta(rows, task, metric):
        color = SERIES_COLORS[label]
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            label=label, color=color,
            marker="o", markersize=4, linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("$\\beta$")
    if metric == "kl":
        ax.set_yscale("log")
        ax.set_ylabel("test $\\mathbb{E}[\\mathrm{KL}]$")
    elif is_cls:
        ax.set_ylabel("test accuracy (%)")
    else:
        ax.set_ylabel("test $R^2$")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    path = FIG_DIR / f"fig_beta_{metric}_{task}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"论文图已保存：{path}")
    return path


# ---------------- HTML 报告（表格 + 交互曲线） ----------------

def task_table_html(task, rows):
    is_cls = task == "imagenet100"
    metric = "Acc" if is_cls else "R2"
    metric_key = "acc" if is_cls else "r2"
    betas = sorted({float(b) for (t, m, b, a) in rows if t == task})

    def cell(r, k):
        m, s = r.get(k), r.get(k + "_std")
        if m is None:
            return "--"
        return f"{m:.4f}±{s:.4f}"

    body = []
    for model, anchors in (("ceb", [None]), ("opb", [1.0, 6.0, 12.0])):
        for a in anchors:
            for b in betas:
                r = rows.get((task, model, f"{b:g}", "" if a is None else f"{a:g}"))
                if r is None:
                    continue
                name = "CEB" if model == "ceb" else ("OPB" if is_cls else "EPB")
                sc = "a" if is_cls else "ρ"
                cfg = f"{name} (β={b:g}" + (f", {sc}={a:g})" if a is not None else ")")
                body.append(
                    f"<tr><td>{cfg}</td><td>{cell(r, 'ce')}</td>"
                    f"<td>{cell(r, 'klm')}</td><td>{cell(r, 'klv')}</td>"
                    f"<td>{cell(r, 'kl')}</td><td>{cell(r, metric_key)}</td></tr>"
                )
    return f"""
<h2>{task}（{metric}）测试集分解</h2>
<table>
<thead><tr><th>配置</th><th>CE</th><th>KL 均值失配项</th><th>KL 方差失配项</th><th>E[KL]</th><th>{metric}</th></tr></thead>
<tbody>
{''.join(body)}
</tbody>
</table>
"""


def task_curve_card(task, metric, rows):
    """交互曲线卡：全系列共享 15 点 β 网格（对数间距），精确点、无需插值。"""
    series_all = series_points_beta(rows, task, metric)
    is_cls = task == "imagenet100"
    if not series_all:
        return "<p>无数据</p>"

    betas = sorted({float(b) for (t, m, b, a) in rows if t == task})
    lo, hi = min(betas), max(betas)
    x0, x1, _, _, _ = plot_bounds(len(series_all))
    span = x1 - x0
    xs = [x0 + (math.log10(b) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * span
          for b in betas]
    x_labels = [f"{b:g}" for b in betas]

    series = []
    for label, pts in series_all:
        if metric == "kl":
            # 纵轴为 log10(KL)，tooltip 显示原始值
            vals = [math.log10(p[1]) for p in pts]
            texts = [f"{p[1]:.3g}" for p in pts]
        else:
            vals = [p[1] for p in pts]
            texts = [f"{p[1]:.2f}" for p in pts]
        # stds 传全 0（curve_card 要求等长列表；0/None 不画误差棒）
        series.append((label, vals, [0.0] * len(vals), texts))

    all_vals = [v for _, vals, _, _ in series for v in vals]
    if metric == "acc":
        y_min = min(0.0, min(all_vals)) - 0.08 * (max(all_vals) - min(all_vals))
        y_max = max(all_vals) + 0.08 * (max(all_vals) - min(all_vals))
    else:
        pad = 0.08 * (max(all_vals) - min(all_vals))
        y_min = min(all_vals) - pad
        y_max = max(all_vals) + pad

    if metric == "acc":
        y_label = "Acc（%）" if is_cls else "R²"
        subtitle = (
            f"横轴 = β（对数刻度，{'…'.join(x_labels[:1])}..{x_labels[-1]}）；纵轴 = {y_label}；"
            "CEB 1 条线 + GPB 3 条线（" + ("a" if is_cls else "ρ") + " = 1/6/12），"
            "各点均为测试集精确评估值"
        )
        title = f"{TASK_DISPLAY.get(task, task)}：β-acc（横轴 = β）"
    else:
        subtitle = (
            f"横轴 = β（对数刻度，{'…'.join(x_labels[:1])}..{x_labels[-1]}）；"
            "纵轴 = log₁₀ E[KL]（tooltip 显示原始值）；"
            "CEB 1 条线 + GPB 3 条线（" + ("a" if is_cls else "ρ") + " = 1/6/12）"
        )
        title = f"{TASK_DISPLAY.get(task, task)}：β-KL（横轴 = β）"
    card = curve_card(
        f"beta-{metric}-{task}", title, subtitle, xs, x_labels, series, y_min, y_max,
    )
    return f"""
<h2>{title}</h2>
{card}
"""


def build_html(rows):
    css = """
    body{font-family:'Noto Sans CJK JP','DejaVu Sans',sans-serif;background:#f9f9f7;
         color:#0b0b0b;max-width:1080px;margin:0 auto;padding:28px 20px;}
    h1{font-size:1.3em;border-bottom:2px solid #0b0b0b;padding-bottom:6px;}
    h2{font-size:1.05em;margin-top:24px;}
    table{border-collapse:collapse;margin:10px 0;font-size:.86em;}
    th,td{border:1px solid #c3c2b7;padding:4px 9px;text-align:center;}
    th{background:#f0efec;}
    td:first-child{text-align:left;}
    """
    html = [
        "<!DOCTYPE html>",
        '<html lang="zh"><head><meta charset="utf-8">',
        "<title>压缩-精度评估（横轴 = β）</title>",
        f"<style>{css}</style></head><body>",
        "<h1>压缩-精度评估：CE / KL 分解与 β-acc / β-KL 曲线</h1>",
        "<p style='color:#52514e;font-size:.85em;'>"
        "数据来自 output/compression_eval/compression_eval_summary.csv"
        "（compression_eval.py，跨 5 run 均值±std）；"
        "曲线横轴为 β（对数刻度），CEB 1 条线、OPB/EPB 各三条线（a/ρ = 1/6/12），"
        "共四张：imagenet100（Acc）与 housing（R²）各配 β-acc / β-KL 两张。</p>",
        curve_css(),
    ]
    for task in ("imagenet100", "housing"):
        html.append(task_table_html(task, rows))
        html.append(task_curve_card(task, "acc", rows))
        html.append(task_curve_card(task, "kl", rows))
    html.append(curve_script())
    html.append("</body></html>")
    HTML_PATH.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML 报告已保存：{HTML_PATH}")


def main():
    rows = load()
    _setup_rc()
    for task in ("imagenet100", "housing"):
        for metric in ("acc", "kl"):
            plot_task_metric(task, metric, rows)
    build_html(rows)


if __name__ == "__main__":
    main()
