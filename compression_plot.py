"""真实压缩-精度报告（审稿人建议版）：横轴 = 测试集实际 E[KL]（对数刻度）、
纵轴 = Acc（imagenet100）/ R²（housing）；CEB 1 条线、OPB a=1/6/12 三条线。

数据来自 compression_eval.py 的 output/compression_eval/compression_eval_summary.csv。
输出：
1. HTML 报告 output/compression_eval/compression_eval.html —— 每任务一张数据表
   （CE / KL 均值失配项 / KL 方差失配项 / E[KL] / 任务指标）与一张交互曲线卡
   （共享 log10(KL) 网格、各系列线性插值，误差棒 = 跨 run std，仅纵轴）；
2. 论文图（白底、无总标题、英文标签，精确点不插值）：
   paper/figures/fig_compression_imagenet100.png / fig_compression_housing.png。

用法：
    python compression_plot.py
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
}
INK = "#0b0b0b"


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


def series_points(rows, task):
    """各系列的点列表：[(label, [(kl, y, y_std, beta, collapsed), ...])]。

    KL < 1e-3 的坍缩角点（残渣级 KL，log 轴上制造回折悬崖）、预测死亡的端点
    （分类 acc < 0.05 / 回归 R² < 0.1）与 KL > 1e4 的方差爆炸端点
    （极小 β + 小 ρ 下离轴方差无约束膨胀）都不画；点按 KL 升序排列（IB 曲线
    惯例：β 增大 = 从右向左移动）。
    """
    is_cls = task == "imagenet100"
    betas = sorted({float(b) for (t, m, b, a) in rows if t == task})
    out = []
    for model, anchors in (("ceb", [""]), ("opb", ["1", "6", "12"])):
        for a in anchors:
            pts = []
            for b in betas:
                r = rows.get((task, model, f"{b:g}", a))
                if r is None or r["kl"] < 1e-3 or r["kl"] > 1e4:
                    continue
                y = r["acc"] if is_cls else r["r2"]
                if (y < 0.05) if is_cls else (y < 0.1):
                    continue  # 预测死亡的端点不画
                pts.append(
                    (r["kl"], y, r["acc_std"] if is_cls else r["r2_std"],
                     f"{b:g}", False)
                )
            pts.sort(key=lambda p: p[0])  # 按 KL 升序连接
            label = "CEB" if model == "ceb" else f"OPB (a={a})"
            out.append((label, pts))
    return out


# ---------------- 论文 PNG（精确点，不插值） ----------------

def plot_task(task, rows):
    is_cls = task == "imagenet100"
    ylabel = "test accuracy (%)" if is_cls else "test $R^2$"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, pts in series_points(rows, task):
        color = SERIES_COLORS[label]
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            label=label, color=color, alpha=1.0,
            marker="o", markersize=4, linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("realized compression $\\mathbb{E}[\\mathrm{KL}]$ on the test set")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", linewidth=0.6, color="#e1e0d9")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    path = FIG_DIR / f"fig_compression_{task}.png"
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
    for model, anchors, name in (("ceb", [None], "CEB"), ("opb", [1.0, 6.0, 12.0], "OPB")):
        for a in anchors:
            for b in betas:
                r = rows.get((task, model, f"{b:g}", "" if a is None else f"{a:g}"))
                if r is None:
                    continue
                cfg = f"{name} (β={b:g}" + (f", a={a:g})" if a is not None else ")")
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


def task_curve_card(task, rows):
    """交互曲线卡：共享 log10(KL) 网格（约 25 点），各系列线性插值。"""
    series_all = series_points(rows, task)
    is_cls = task == "imagenet100"
    if not series_all:
        return "<p>无数据</p>"

    all_kl = [p[0] for _, pts in series_all for p in pts]
    lo, hi = min(all_kl), max(all_kl)
    grid = np.logspace(math.log10(lo), math.log10(hi), 25)

    x0, x1, _, _, _ = plot_bounds(len(series_all))
    span = x1 - x0
    xs = [x0 + (math.log10(g) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * span
          for g in grid]
    x_labels = [f"{g:.2g}" for g in grid]

    series = []
    for label, pts in series_all:
        px = np.array([math.log10(p[0]) for p in pts])
        # 防重复 x（np.interp 要求严格递增；如 OPB a=1 的 β=1/10 几乎同 KL）
        px = px + np.arange(len(px)) * 1e-12
        py = np.array([p[1] for p in pts])
        gx = np.log10(grid)
        vals = np.interp(gx, px, py)
        texts = [f"{v:.4f}" for v in vals]
        # stds 传全 0（curve_card 要求等长列表；0/None 不画误差棒）
        series.append((label, list(vals), [0.0] * len(vals), texts))

    y_min = min(0.0, min(v for _, vals, _, _ in series for v in vals)) - 0.05
    y_max = max(1.0 if is_cls else 0.6, max(v for _, vals, _, _ in series for v in vals)) + 0.05
    card = curve_card(
        f"compression-{task}",
        f"{TASK_DISPLAY.get(task, task)}：真实压缩-精度曲线",
        f"横轴 = 测试集实际 E[KL]（对数刻度，各系列在共享网格上线性插值）；"
        f"纵轴 = {'Acc' if is_cls else 'R²'}；"
        "每条线 β ∈ {5e-5..25} 按 KL 升序连接；E[KL] < 1e-3 的坍缩角点、"
        "预测死亡端点（回归 R² < 0.1）与 E[KL] > 1e4 的方差爆炸端点省略",
        xs, x_labels, series, y_min, y_max,
    )
    return f"""
<h2>{task}：压缩-精度曲线（横轴 = E[KL]）</h2>
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
        "<title>压缩-精度评估（横轴 = 实际 E[KL]）</title>",
        f"<style>{css}</style></head><body>",
        "<h1>压缩-精度评估：CE / KL 分解与真实压缩-精度曲线</h1>",
        "<p style='color:#52514e;font-size:.85em;'>"
        "数据来自 output/compression_eval/compression_eval_summary.csv"
        "（compression_eval.py，跨 5 run 均值±std）；"
        "曲线横轴为测试集实际 E[KL]（对数刻度），CEB 1 条线、OPB a=1/6/12 三条线。</p>",
        curve_css(),
    ]
    for task in ("imagenet100", "housing"):
        html.append(task_table_html(task, rows))
        html.append(task_curve_card(task, rows))
    html.append(curve_script())
    html.append("</body></html>")
    HTML_PATH.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML 报告已保存：{HTML_PATH}")


def main():
    rows = load()
    _setup_rc()
    for task in ("imagenet100", "housing"):
        plot_task(task, rows)
    build_html(rows)


if __name__ == "__main__":
    main()
