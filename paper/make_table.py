#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""从 tune_results/*.html 收集调参结果，生成 main_result.tex 主结果表。

tune.py 生成的每个 html 对应一个 数据集×骨干 组合，内部每个模型一张表；
每行一个超参组合（指标 td 文本为 'mean±std'），各模型默认指标（分类 Acc、
回归 R²）的最优行已被 tune.py 标为 class="best"。

本脚本对每个 (数据集, 模型) 取 best 行（缺失时回退到该模型指标最大的行），
只保留 Acc/R² 一个指标：分类任务报告 Acc（×100 显示为百分比）、回归任务报告
R²（原始值），且只显示均值、不含标准差（表格较宽）；其余指标
（Loss/AUC/Pre/Rec/MAE）一律丢弃；每行（数据集）的最优指标加粗、次优
（并列次优全部）加下划线；表尾两行统计各方法跨数据集行的平均排名（并列取
平均名次，越低越好）与最优/并列最优次数（并列时各方法均计数）。输出可直接
\input{} 的 table* 浮动体，保存到 paper/main_result.tex；同源生成带标准差的
paper/main_result_std.tex（gen_table_std，单元格为均值±std，按列拆为
baselines 与 CEB/DVCCA+GPB/GPB-L 两张表以适配页宽）；以及配对差值表
paper/main_result_CI.tex（gen_result_ci，GPB − 各 baseline 的 bootstrap 95% CI，
同 seed 配对、按列拆两张表、表尾统计 */† 次数）。

同时生成超参数敏感性表 paper/result1.tex（gen_result1）：仅 ImageNet-100 (MLP，
分类 Acc ×100) 与 AgeDB (MLP，回归 R²) 两个任务，行 = CEB 与 OPB 的
a ∈ {1, 6, 12} 三条线，列 = β 网格 {1e-4, ..., 10}。

以及对抗鲁棒性表 paper/result2.tex（gen_result2）：读取
output/adv_mnist/mnist_adv.csv（adv_eval.py 输出），行 = 攻击强度
（clean + L∞/L2 各 ε，随 CSV 自适应），列 = CSV 中的模型配置（数量自适应），
值为跨 run 平均鲁棒精度（%）。

以及压缩-精度评估表 paper/result4.tex（gen_result4，CE/KL 分解项）；
以及几何机制表 paper/result3.tex（gen_result3）：读取 output/pri-pos 下
prior_geometry.py / posterior_geometry.py 的 JSON（posterior_geometry.json
与 prior_summary.json），行 = 四个机制对照配置（MNIST CEB/OPB、Housing
CEB/EPB），列 = 几何集中度 / 尺度跟随 / 跨 seed 稳定（每个任务取最有区分
度的指标）；解释性内容（先验任意 vs 构造固定、收缩效应、轴方向旋转）已写入 paper.tex 正文（sec:mechanism），本表只保留数据。
"""

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "tune_results"
OUT_PATH = Path(__file__).resolve().parent / "main_result.tex"
MAIN_RESULT_STD_PATH = Path(__file__).resolve().parent / "main_result_std.tex"
MAIN_RESULT_CI_PATH = Path(__file__).resolve().parent / "main_result_CI.tex"
RESULT1_PATH = Path(__file__).resolve().parent / "result1.tex"
RESULT2_PATH = Path(__file__).resolve().parent / "result2.tex"
RESULT3_PATH = Path(__file__).resolve().parent / "result3.tex"
RESULT4_PATH = Path(__file__).resolve().parent / "result4.tex"
# 对抗鲁棒性长表（adv_eval.py 输出）：config/run/norm/eps/acc
ADV_CSV_PATH = ROOT / "output" / "adv_mnist" / "mnist_adv.csv"
# 几何机制表数据源（prior_geometry.py / posterior_geometry.py 输出）
POSTERIOR_JSON = ROOT / "output" / "pri-pos" / "pos_results" / "posterior_geometry.json"
PRIOR_SUMMARY_JSON = ROOT / "output" / "pri-pos" / "pri_results" / "prior_summary.json"

# 压缩-精度表（result1.tex）：仅这两个 (task, backbone) 与固定 β 网格，
# 行 = CEB 与 OPB 的 a=1/6/12 三条线，列 = β
COMPRESSION_TASKS = [("imagenet100", "mlp"), ("agedb", "mlp")]
COMPRESSION_BETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
COMPRESSION_ANCHORS = [1.0, 6.0, 12.0]

# 压缩-精度评估数据（compression_eval.py 输出）：CE / KL / KL 分解项
COMPRESSION_EVAL_CSV = ROOT / "output" / "compression_eval" / "compression_eval_summary.csv"


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

# 分类任务集合（决定行顺序中分类块与回归块的分界）
CLASSIFICATION_TASKS = {"mnist", "imagenet100", "cora", "imdb", "agnews"}

# 数据集/骨干显示名
TASK_NAMES = {
    "mnist": "MNIST",
    "imagenet100": "ImageNet-100",
    "cora": "Cora",
    "imdb": "IMDb",
    "agnews": "AG News",
    "housing": "Cal. Housing",
    "stsb": "STS-B",
    "zinc": "ZINC",
    "agedb": "AgeDB",
}
BACKBONE_NAMES = {"mlp": "MLP", "cnn": "CNN", "gnn": "GCN", "gcn": "GCN", "rnn": "RNN"}

# 表格行顺序（分类在前、回归在后）；不在列表中的新组合追加在末尾
ROW_ORDER = [
    ("mnist", "mlp"), ("mnist", "cnn"),
    ("imagenet100", "mlp"), ("imagenet100", "cnn"),
    ("cora", "gnn"), ("imdb", "rnn"), ("agnews", "rnn"),
    ("housing", "mlp"), ("stsb", "rnn"), ("zinc", "gnn"),
    ("agedb", "mlp"), ("agedb", "cnn"),
]

# 表格列顺序与列名；html 里基础模型名为 mlp/cnn/gcn/rnn，统一归到 "base" 列。
# "opb" 列在表格中显示为 GPB（分类行为 OPB、回归行为 EPB，即几何先验瓶颈的
# 两个实例）；opbl 是 GPB 的 free-head 消融（GPB-L；结果目录 tune_results 中 opb 已改名
# 为 opbl），排在 opb 之后；"opb" 在 COLUMN_ORDER 中的位置是竖线插入点
# （其左侧加竖线，把 GPB/GPB-L 与前面的 baseline 方法隔开）。
COLUMN_ORDER = ["base", "vib", "svib", "nib", "ceb", "dvcca", "opb", "opbl"]
COLUMN_NAMES = {
    "base": "Base", "vib": "VIB", "svib": "SVIB", "nib": "NIB",
    "ceb": "CEB", "dvcca": "DVCCA", "opb": "GPB", "opbl": "GPB-L",
}


class TuneHTMLParser(HTMLParser):
    """解析单个调参结果 html：提取 meta 行与各模型 best 行的 Acc/R² 均值±标准差。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = ""
        self.rows = {}  # model -> [(is_best, key, mean, std), ...]

        self._in_meta = False
        self._in_h2 = False
        self._h2_text = ""
        self._in_thead = False
        self._th_texts = []
        self._in_th = False
        self._th_text = ""
        self._in_td = False
        self._td_text = ""
        self._tr_cells = []
        self._tr_best = False
        self._model = None
        self._metric_cols = []  # 当前表指标列名（去掉 beta/anchor 前两列）

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and "meta" in a.get("class", ""):
            self._in_meta = True
        elif tag == "h2":
            self._in_h2 = True
            self._h2_text = ""
        elif tag == "thead":
            self._in_thead = True
            self._th_texts = []
        elif tag == "th":
            self._in_th = True
            self._th_text = ""
        elif tag == "td":
            self._in_td = True
            self._td_text = ""
        elif tag == "tr":
            self._tr_best = "best" in a.get("class", "").split()
            self._tr_cells = []

    def handle_endtag(self, tag):
        if tag == "div" and self._in_meta:
            self._in_meta = False
        elif tag == "h2" and self._in_h2:
            self._in_h2 = False
            self._model = self._h2_text.strip()
        elif tag == "thead" and self._in_thead:
            self._in_thead = False
            # 去掉前两列（beta/anchor_scale），逐 th 收集避免单元格文本无空白分隔
            self._metric_cols = self._th_texts[2:]
        elif tag == "th" and self._in_th:
            self._in_th = False
            self._th_texts.append(self._th_text.strip())
        elif tag == "td" and self._in_td:
            self._in_td = False
            self._tr_cells.append(self._td_text.strip())
        elif tag == "tr":
            self._record_row()

    def handle_data(self, data):
        if self._in_meta:
            self.meta += data
        elif self._in_h2:
            self._h2_text += data
        elif self._in_th:
            self._th_text += data
        elif self._in_td:
            self._td_text += data

    def _record_row(self):
        # 行前两格为 beta/anchor，其后按 metric_cols 顺序为各指标；
        # FAILED 行（colspan 单元格）或指标缺失（"-"）的行长度不符/无 ±，直接跳过
        cells = self._tr_cells[2:]
        # 需排除 thead 的 tr（无 td、0 格）在首个表时与空 metric_cols 长度相等的情况
        if self._model is None or not self._metric_cols or len(cells) != len(self._metric_cols):
            return
        # 非结果表（如"最优精度汇总"表，列名为"最优 R2"而非 R2）直接跳过
        key = "Acc" if "Acc" in self._metric_cols else ("R2" if "R2" in self._metric_cols else None)
        if key is None:
            return
        cell = cells[self._metric_cols.index(key)]
        if "±" not in cell:
            return
        mean_s, std_s = cell.split("±", 1)
        try:
            mean, std = float(mean_s), float(std_s)
        except ValueError:
            return
        beta = _to_float(self._tr_cells[0])
        anchor = _to_float(self._tr_cells[1]) if len(self._tr_cells) > 1 else None
        self.rows.setdefault(self._model, []).append(
            (self._tr_best, key, mean, std, beta, anchor)
        )

    def best_values(self):
        """各模型取 best 行（缺失时回退指标最大行），返回 {model: (key, mean, std)}。"""
        out = {}
        for model, entries in self.rows.items():
            bests = [e for e in entries if e[0]]
            pool = bests or entries
            _, key, mean, std = max(pool, key=lambda e: e[2])[:4]
            out[model] = (key, mean, std)
        return out


def parse_file(path: Path):
    """解析一个 html 文件，返回 (task, backbone, {model: (key, mean, std)})。"""
    parser = TuneHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    m = re.search(r"task=(\S+)\s+backbone=(\S+)", parser.meta)
    if not m:
        raise ValueError(f"{path.name}: meta 行无法解析: {parser.meta!r}")
    task, backbone = m.group(1), m.group(2)
    return task, backbone, parser.best_values()


def parse_file_rows(path: Path):
    """解析一个 html 文件，返回 (task, backbone, {model: [(is_best, key, mean, std, beta, anchor)]})。"""
    parser = TuneHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    m = re.search(r"task=(\S+)\s+backbone=(\S+)", parser.meta)
    if not m:
        raise ValueError(f"{path.name}: meta 行无法解析: {parser.meta!r}")
    return m.group(1), m.group(2), parser.rows


def collect():
    """汇总全部 html：{(task, backbone): {列名: (key, mean, std)}}。"""
    grid = {}
    files = sorted(RESULTS_DIR.glob("*.html"))
    for path in files:
        task, backbone, values = parse_file(path)
        cells = {}
        for model, (key, mean, std) in values.items():
            col = "base" if model in BACKBONE_NAMES else model
            cells[col] = (key, mean, std)
        grid[(task, backbone)] = cells
    return grid


def collect_full(task_backbones):
    """收集压缩-精度表的完整行：{(task, backbone): {model: [(is_best, key, mean, std, beta, anchor)]}}。

    同一 (task, backbone) 存在多个 html（如 +opb+opbl+ 与 +opbl+ 两个版本）时，
    按文件名排序，后者覆盖前者（模型行整体替换）。
    """
    out = {tb: {} for tb in task_backbones}
    for path in sorted(RESULTS_DIR.glob("*.html")):
        task, backbone, rows = parse_file_rows(path)
        if (task, backbone) not in out:
            continue
        for model, entries in rows.items():
            if entries:
                out[(task, backbone)][model] = entries
    return out


def parse_adv_config(name):
    """对抗试验配置名 → (显示标签, beta, anchor)。

    mnist_mlp → ('Base', None, None)；
    mnist_mlp_{model}_beta{b}[_scale_{a}] → 如
    ('OPB ($\\beta=0.1$, $a=12$)', 0.1, 12.0)。无法解析时原样返回。
    """
    if name == "mnist_mlp":
        return "Base", None, None
    m = re.match(r"^mnist_mlp_([a-z]+)_beta([\d.]+)(?:_scale_?([\d.]+))?$", name)
    if not m:
        return name, None, None
    model = m.group(1).upper()
    beta = float(m.group(2))
    anchor = float(m.group(3)) if m.group(3) else None
    if anchor is not None:
        return f"{model} ($\\beta={beta:g}$, $a={anchor:g}$)", beta, anchor
    return f"{model} ($\\beta={beta:g}$)", beta, None


def gen_result2():
    r"""从 ADV_CSV_PATH 生成对抗鲁棒性表 result2.tex（CSV 缺失时返回 None）。

    行 = 攻击强度（clean + L∞/L2 各 ε，网格随 CSV 自适应），列 = CSV 中的
    模型配置（数量自适应、Base 居首其余按名排序）；值为跨 run 鲁棒精度
    （Acc ×100）均值±标准差，每行按均值最优加粗。tab:robustness，可直接
    \input{}。
    """
    if not ADV_CSV_PATH.exists():
        return None
    with open(ADV_CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    configs = sorted({r["config"] for r in rows},
                     key=lambda c: (c != "mnist_mlp", c))
    combos = [("none", 0.0)] \
        + [("linf", e) for e in sorted({float(r["eps"]) for r in rows if r["norm"] == "linf"})] \
        + [("l2", e) for e in sorted({float(r["eps"]) for r in rows if r["norm"] == "l2"})]
    accs = {}
    for r in rows:
        accs.setdefault((r["config"], r["norm"], float(r["eps"])), []).append(float(r["acc"]))

    def mean_std(c, norm, eps):
        vals = accs.get((c, norm, eps))
        if not vals:
            return None
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
        return m, s

    def fmt(c, norm, eps):
        ms = mean_std(c, norm, eps)
        return f"{ms[0] * 100:.2f}$\\pm${ms[1] * 100:.2f}" if ms else "--"

    lines = [
        "% 对抗鲁棒性表：由 paper/make_table.py 从 output/adv_mnist/mnist_adv.csv 自动生成，请勿手改。",
        "% 值为跨 run 鲁棒精度（%）均值±标准差，每行按均值最优加粗；行 = 攻击强度，列 = 模型配置（数量自适应）。",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Adversarial robustness on MNIST: test accuracy (\\%) under untargeted "
        "projected gradient descent (PGD) attacks in $L_\\infty$ and $L_2$ norms, "
        "evaluated on the deterministic ($z=\\mu$) path; mean $\\pm$ std. over runs.}",
        "\\label{tab:robustness}",
        "\\small",
        "{",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{l" + "c" * len(configs) + "}",
        "\\hline",
        " & " + " & ".join(parse_adv_config(c)[0] for c in configs) + " \\\\",
        "\\hline",
    ]
    for norm, eps in combos:
        if norm == "none":
            label = "Clean"
        else:
            math_norm = "\\ell_\\infty" if norm == "linf" else "\\ell_2"
            label = f"${math_norm}$ $\\varepsilon={eps:g}$"
        # 加粗按均值判定（带 ± 的字符串不能直接 float 比较）
        means = [mean_std(c, norm, eps) for c in configs]
        best = max((m[0] for m in means if m), default=None)
        cells = [
            f"\\textbf{{{fmt(c, norm, eps)}}}"
            if means[i] and means[i][0] == best
            else fmt(c, norm, eps)
            for i, c in enumerate(configs)
        ]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def _f4(x, nd=4):
    """None 安全格式化（数据缺失显示 -）。"""
    return "-" if x is None else f"{x:.{nd}f}"


def gen_result3():
    r"""从 output/pri-pos 的机制验证 JSON 生成几何机制表 result3.tex（table 浮动体）。

    行 = 四个机制对照配置（MNIST CEB/OPB、Housing CEB/EPB），列 = 三个概念
    指标：几何集中度（分类 mean|cos_off| / 回归离轴占比）、尺度跟随（分类 ‖c̄‖
    对照 a / 回归斜率对照 ρ）、跨 seed 稳定（分类跨run余弦std / 回归斜率std）；
    解释性内容（先验任意 vs 构造固定、收缩效应、轴方向旋转）在 paper.tex
    正文 sec:mechanism 中，本表只保留数据。数据缺失时返回 None（跳过）。
    """
    if not POSTERIOR_JSON.exists():
        return None
    pos = json.loads(POSTERIOR_JSON.read_text(encoding="utf-8"))
    results = {k: v for k, v in pos.items() if isinstance(v, dict) and "cross_run" in v}
    meta = pos.get("meta", {})

    def display(label, model, task):
        task_short = "MNIST" if task == "mnist" else "Housing"
        if model == "opb":
            name = "EPB" if task != "mnist" else "OPB"
        else:
            name = model.upper()
        return f"{name} ({task_short})"

    lines = [
        "% 几何机制表：由 paper/make_table.py 从 output/pri-pos 的机制验证 JSON 自动生成，请勿手改。",
        "% 行 = 机制对照配置；列 = 几何集中度 / 尺度跟随 / 跨 seed 稳定（跨 5 run 均值±std），",
        "% 列头括号注明分类（MNIST）与回归（Housing）各自使用的指标；解释性文字见 paper.tex 正文。",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Mechanism validation: the constructed geometry pulls the posterior "
        "into the orthogonal frame (classification) and onto the isometric axis "
        "(regression). Geometric concentration: mean off-diagonal cosine of the "
        "posterior class-center Gram matrix (classification) / off-axis residual "
        "fraction (regression). Scale following: class-center radius "
        "$\\lVert\\bar c\\rVert$ against the anchor scale $a$ (classification) / axial "
        "slope against $\\rho$ (regression). Cross-seed stability: std. over runs. "
        "Means $\\pm$ std. over five runs.}",
        "\\label{tab:mechanism}",
        "\\small",
        "{",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l l c c c}",
        "\\hline",
        "Model (task) & Task & Geometric concentration & Scale following & Cross-seed stability \\\\",
        "\\hline",
    ]

    for label, res in results.items():
        cr = res["cross_run"]
        task = res.get("task")
        model = res.get("model")
        a_val = meta.get(f"a_{label}") if model == "opb" else None
        if res["mode"] == "classification":
            conc = (
                f"{_f4(cr.get('mean_abs_cos_offdiag_mean'))}$\\pm$"
                f"{_f4(cr.get('mean_abs_cos_offdiag_std'))}"
            )
            scale = f"{_f4(cr.get('center_norm_mean_mean'), 2)}"
            if a_val is not None:
                scale += f" ($a={a_val:g}$)"
            stab = _f4(cr.get("mean_pairwise_cos_std"))
        else:
            conc = _f4(cr.get("off_axis_frac_mean"))
            scale = (
                f"{_f4(cr.get('slope_mean'), 2)}$\\pm${_f4(cr.get('slope_std'), 2)}"
            )
            if a_val is not None:
                scale += f" ($\\rho={a_val:g}$)"
            stab = _f4(cr.get("slope_std"), 3)
        lines.append(
            f"{display(label, model, task)} & {'Classif.' if task == 'mnist' else 'Regress.'} & "
            f"{conc} & {scale} & {stab} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")

    # 表注已移入正文（paper.tex sec:mechanism 的 prior/posterior 两段已含：
    # 先验任意 vs 构造固定、收缩解释、轴方向旋转）；本表只保留数据
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def gen_result1(full):
    r"""生成超参数敏感性表 result1.tex（table* 浮动体，可直接 \input{}）。

    行 = CEB 与 GPB（ImageNet-100 为 OPB 的 a、AgeDB 为 EPB 的 ρ）
    在 {1, 6, 12} 的三条线，列 = β 网格
    （COMPRESSION_BETAS）；仅 ImageNet-100 (MLP，分类 Acc ×100) 与
    AgeDB (MLP，回归 R²) 两个任务，各自成组、组间以横线分隔。
    """
    lines = [
        "% 超参数敏感性表：由 paper/make_table.py 自动生成，请勿手改。",
        "% ImageNet-100 (MLP) 报告测试 Acc（%）、AgeDB (MLP) 报告测试 R²；均为 5 次运行均值。",
        "% 行 = CEB 与 GPB 的三条线（ImageNet-100：OPB a=1/6/12；AgeDB：EPB ρ=1/6/12）；列 = β（对数网格）。",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Hyperparameter sensitivity: test metric as a function of $\\beta$ "
        "for CEB and for GPB (OPB at anchor scales $a\\in\\{1,6,12\\}$ on ImageNet-100; "
        "EPB at isometric scales $\\rho\\in\\{1,6,12\\}$ on AgeDB). ImageNet-100 (MLP) "
        "reports test accuracy (\\%); AgeDB (MLP) reports test $R^2$; means over five runs.}",
        "\\label{tab:compression}",
        "\\small",
        "{",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l l c c c c c c}",
        "\\hline",
        " & & \\multicolumn{6}{c}{$\\beta$} \\\\",
        " & & $10^{-4}$ & $10^{-3}$ & $10^{-2}$ & $10^{-1}$ & $1$ & $10$ \\\\",
        "\\hline",
    ]

    def fmt(entry):
        _, key, mean, _, _, _ = entry
        return f"{mean * 100:.2f}" if key == "Acc" else f"{mean:.4f}"

    def cell(entries, beta, anchor):
        for e in entries:
            if e[4] is not None and abs(e[4] - beta) < 1e-9 and (
                anchor is None or (e[5] is not None and abs(e[5] - anchor) < 1e-9)
            ):
                return fmt(e)
        return "--"

    for tb in COMPRESSION_TASKS:
        task, backbone = tb
        label = TASK_NAMES.get(task, task)
        rows = full.get(tb, {})
        # ImageNet-100（分类）用 OPB 的锚点尺度 a，AgeDB（回归）用 EPB 的等距尺度 ρ
        gpb_name = "OPB" if task in CLASSIFICATION_TASKS else "EPB"
        scale_name = "a" if task in CLASSIFICATION_TASKS else "\\rho"
        specs = [("CEB", rows.get("ceb", []), None)] + [
            (f"{gpb_name} (${scale_name}={a:g}$)", rows.get("opb", []), a)
            for a in COMPRESSION_ANCHORS
        ]
        for i, (name, entries, anchor) in enumerate(specs):
            first_col = label if i == 0 else ""
            cells = " & ".join(cell(entries, b, anchor) for b in COMPRESSION_BETAS)
            lines.append(f"{first_col} & {name} & {cells} \\\\")
        lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ])
    return "\n".join(lines) + "\n"


# 配对差值 95% CI：非劣界（分类 0.2 百分点 / 回归 R² 0.005）、bootstrap 次数与 seed
DIFF_CI_DELTA = {"Acc": 0.002, "R2": 0.005}
DIFF_CI_B = 10000
DIFF_CI_SEED = 0
# CI 表按列拆两张表：baselines（Base/VIB/SVIB/NIB）与 ceb/dvcca/opbl
CI_COL_GROUPS = [
    (["base", "vib"], "tab:main_result_ci_baselines"),
    (["svib", "nib"], "tab:main_result_ci_svib_nib"),
    (["ceb", "dvcca", "opbl"], "tab:main_result_ci_variants"),
]


def _combo_dir_name(task, backbone, model, beta, anchor):
    """调参结果目录名（与 tune.combo_name 同格式）：基线为 {task}_{model}，
    变体为 {task}_{backbone}_{model}，后缀 _beta_{b:g} / _anchor_{a:g}。"""
    name = f"{task}_{model}" if model in ("mlp", "cnn", "gcn", "rnn") else f"{task}_{backbone}_{model}"
    if beta is not None:
        name += f"_beta_{beta:g}"
    if anchor is not None:
        name += f"_anchor_{anchor:g}"
    return name


def parse_run_metrics(log_path: Path):
    """解析 train.log 逐 run 测试行（'时间戳 | Run i/N | Test ... | Acc/R2 ...'），
    返回 (key, [run 值列表])；无逐 run 行时返回 (None, None)。"""
    key, vals = None, []
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                heads = [p.strip() for p in line.split("|")]
                if len(heads) < 4 or not heads[1].startswith("Run") or "Test" not in heads[2]:
                    continue
                parts = heads[3].split()
                metrics = {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}
                k = "Acc" if "Acc" in metrics else ("R2" if "R2" in metrics else None)
                if k is None or (key is not None and k != key):
                    continue
                try:
                    v = float(metrics[k])
                except ValueError:
                    continue
                key, vals = k, vals + [v]
    except OSError:
        return None, None
    return (key, vals) if vals else (None, None)


def paired_diff_ci(a_vals, b_vals):
    """同 seed 配对的差值 bootstrap 95% CI（百分位法，固定 seed 可复现）。
    返回 (n, mean_diff, ci_low, ci_high)；n=0 时均值/CI 为 None。"""
    n = min(len(a_vals), len(b_vals))
    if n == 0:
        return 0, None, None, None
    diffs = np.array(a_vals[:n]) - np.array(b_vals[:n])
    rng = np.random.default_rng(DIFF_CI_SEED)
    means = diffs[rng.integers(0, n, size=(DIFF_CI_B, n))].mean(axis=1)
    return n, float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gen_result_ci():
    r"""生成 main_result_CI.tex：GPB − 各 baseline 的配对差值 95% CI 表。

    行 = 12 个 (task, backbone) setting（分类/回归块间双横线），单元格 =
    "均值差 [CI下限, CI上限]"（分类百分点、回归 R²）；CI 下限 > 0 加粗（显著
    更优）、> −δ（DIFF_CI_DELTA）标 $^{\dagger}$（tied）、否则原样。因 7 个
    差值列超页宽，按列拆两张 table*（CI_COL_GROUPS）。每表尾一行统计各列
    "X*/Y†" 次数。双方各取默认指标均值最优组合（与主表同规则），差值按同
    seed 配对（run 顺序即 seed 0..N−1）。数据缺失时返回 None（跳过）。
    """
    if not list(RESULTS_DIR.glob("*.html")):
        return None
    # 每个 (task, backbone) 的各模型 best 行（beta/anchor 供定位日志目录）
    grid = {}
    for path in sorted(RESULTS_DIR.glob("*.html")):
        task, backbone, rows = parse_file_rows(path)
        best = {}
        for model, entries in rows.items():
            bests = [e for e in entries if e[0]]
            pool = bests or entries
            _, key, mean, std, b, a = max(pool, key=lambda e: e[2])
            best[model] = (key, b, a)
        grid[(task, backbone)] = best
    if not grid:
        return None

    order = {k: i for i, k in enumerate(ROW_ORDER)}
    keys = sorted(grid, key=lambda k: (order.get(k, len(order)), k))
    cls_keys = [k for k in keys if k[0] in CLASSIFICATION_TASKS]
    reg_keys = [k for k in keys if k[0] not in CLASSIFICATION_TASKS]
    all_keys = cls_keys + reg_keys

    # 目录名用数据集名（如 housing → california），与 tune 的 get_dataset_name 一致
    dataset = {"housing": "california"}.get

    def diff_cell(task, backbone, best, col):
        """GPB − col 的 CI 单元格（含加粗/tied 标记）；异常返回 None。"""
        gp = best.get("opb")
        # base 列对应基础骨干模型（html 里模型名为 mlp/cnn/gcn/rnn；
        # gnn 骨干的基础模型名为 gcn）
        model_key = ("gcn" if backbone == "gnn" else backbone) if col == "base" else col
        base = best.get(model_key)
        if gp is None or base is None:
            return None
        key = "Acc" if gp[0] == "Acc" else "R2"
        gp_dir = RESULTS_DIR / _combo_dir_name(dataset(task, task), backbone, "opb", gp[1], gp[2])
        b_dir = RESULTS_DIR / _combo_dir_name(dataset(task, task), backbone, model_key, base[1], base[2])
        gp_key, gp_vals = parse_run_metrics(gp_dir / "train.log")
        b_key, b_vals = parse_run_metrics(b_dir / "train.log")
        if not gp_vals or not b_vals:
            return None
        n, mean_d, lo, hi = paired_diff_ci(gp_vals, b_vals)
        if mean_d is None:
            return None
        scale = 100.0 if key == "Acc" else 1.0
        delta = DIFF_CI_DELTA[key]
        # 小数位随指标收敛（分类 1 位、回归 3 位），控制表宽
        if key == "Acc":
            text = f"{mean_d * scale:+.1f} [{lo * scale:+.1f}, {hi * scale:+.1f}]"
        else:
            text = f"{mean_d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if lo > 0:
            return "\\textbf{" + text + "}", "*"
        if lo > -delta:
            return text + "$^{\\dagger}$", "†"
        return text, None

    def block_float(cols, caption, label):
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\small",
            "{",
            "\\setlength{\\tabcolsep}{0.5pt}",
            "\\begin{tabular}{ll" + "c" * len(cols) + "}",
            "\\hline",
            " & ".join(["Dataset", "Backbone"] + [COLUMN_NAMES[c] for c in cols]) + " \\\\",
            "\\hline",
        ]
        counts = {c: {"*": 0, "†": 0} for c in cols}
        for row_keys, last in ((cls_keys, False), (reg_keys, True)):
            for task, backbone in row_keys:
                best = grid[(task, backbone)]
                cells = []
                for col in cols:
                    r = diff_cell(task, backbone, best, col)
                    if r is None:
                        cells.append("--")
                    else:
                        text, mark = r
                        cells.append(text)
                        if mark:
                            counts[col][mark] += 1
                parts = [TASK_NAMES.get(task, task), BACKBONE_NAMES.get(backbone, backbone)] + cells
                lines.append(" & ".join(parts) + " \\\\")
            if not last:
                lines.append("\\hline\\hline")
        # 表尾汇总行：每列 "X*/Y†"（显著更优/tied 次数）
        summary = ["", ""]
        for col in cols:
            c = counts[col]
            summary.append(f"{c['*']}*/{c['†']}†")
        lines.append("\\hline")
        lines.append(" & ".join(summary) + " \\\\")
        lines.append("\\hline")
        lines.extend(["\\end{tabular}", "}", "\\end{table*}", "\\par\\smallskip"])
        return "\n".join(lines) + "\n"

    head = [
        "% 主结果差值 CI 表：由 paper/make_table.py 从 tune_results 自动生成，请勿手改。",
        "% 行 = 12 个 setting；单元格 = GPB − baseline 的配对差值 [bootstrap 95% CI]",
        "%（分类百分点、回归 R²）；加粗 = CI 下限 > 0（显著更优）、† = CI 下限 > −δ",
        f"%（非劣界 δ={DIFF_CI_DELTA['Acc'] * 100:g} 百分点 / {DIFF_CI_DELTA['R2']:g}）记 tied；",
        f"% 同 seed 配对（run 顺序即 seed 0..N−1），bootstrap B={DIFF_CI_B}（固定 seed {DIFF_CI_SEED}，百分位法）。",
        "% 因 7 个差值列超页宽，按列拆两张表（baselines / ceb+dvcca+GPB-L）；表尾一行统计各列 */† 次数。",
        "",
    ]
    out = "\n".join(head)
    captions = {
        "tab:main_result_ci_baselines": (
            "Paired-difference 95\\% confidence intervals of GPB over the plain "
            "backbone and VIB (GPB minus baseline, per setting; classification in "
            "percentage points, regression in $R^2$). Bold: lower CI bound above "
            "zero; $^{\\dagger}$: lower bound within the non-inferiority margin "
            "$\\delta$ (tied)."
        ),
        "tab:main_result_ci_svib_nib": (
            "Paired-difference 95\\% confidence intervals of GPB over SVIB and NIB "
            "(GPB minus baseline, per setting; classification in percentage points, "
            "regression in $R^2$). Bold/† as in the baselines table."
        ),
        "tab:main_result_ci_variants": (
            "Paired-difference 95\\% confidence intervals of GPB over CEB, DVCCA, "
            "and the free-head ablation GPB-L (GPB minus baseline, per setting; "
            "classification in percentage points, regression in $R^2$). Bold/† as in "
            "the baselines table."
        ),
    }
    for cols, label in CI_COL_GROUPS:
        out += block_float(cols, captions[label], label)
    return out


def gen_result4():
    r"""生成压缩-精度评估表 result4.tex：CE / KL 均值失配项 / KL 方差失配项 /
    E[KL] / 任务指标的测试集分解（compression_eval.py 输出，均值±std）。

    两个任务块（ImageNet-100 MLP 分类、AgeDB MLP 回归），行 = CEB（6 个 β）
    与 OPB（6 β × 3 a = 18 组合），块间双横线；tab:compression_decomp，
    可直接 \input{}。CSV 缺失时返回 None（跳过）。
    """
    if not COMPRESSION_EVAL_CSV.exists():
        return None
    rows = list(csv.DictReader(open(COMPRESSION_EVAL_CSV, newline="", encoding="utf-8")))

    def fmt(r, col):
        m, s = r.get(f"{col}_mean"), r.get(f"{col}_std")
        if not m:
            return "--"
        try:
            def g(x):
                # 大值（如方差爆炸点 KL≈1e8）用科学计数法控制表宽
                return f"{x:.1e}" if abs(x) >= 1e4 else f"{x:.3f}"
            if abs(float(m)) >= 1e4:
                return g(float(m))  # 方差爆炸行只报均值（std 与之同量级，见表注）
            return f"{g(float(m))}$\\pm${g(float(s))}"
        except ValueError:
            return "--"

    def block(task, is_cls):
        out = []
        rows_t = [r for r in rows if r["task"] == task]
        specs = [("ceb", None)] + [("opb", a) for a in COMPRESSION_ANCHORS]
        for model, a in specs:
            for r in rows_t:
                if r["model"] != model or (a is not None and float(r["anchor"]) != a):
                    continue
                beta = float(r["beta"])
                label = "CEB" if model == "ceb" else "OPB"
                cfg = f"{label} ($\\beta={beta:g}$" + (f", $a={a:g}$)" if a is not None else ")")
                metric = fmt(r, "Acc" if is_cls else "R2")
                out.append(
                    f"{cfg} & {fmt(r, 'CE')} & {fmt(r, 'KL_mean')} & "
                    f"{fmt(r, 'KL_var')} & {fmt(r, 'KL')} & {metric} \\\\"
                )
        return out

    lines = [
        "% 压缩-精度评估表：由 paper/make_table.py 从 compression_eval.csv 自动生成，请勿手改。",
        "% 测试集分解：CE / KL 均值失配项 / KL 方差失配项 / E[KL] / 任务指标（均值±std，跨 5 run）。",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test-set decomposition of the sweep configurations: prediction loss "
        "(CE or MSE), the mean-mismatch and variance-mismatch terms of the KL, the total "
        "$\\E[\\KL]$, and the task metric (accuracy on ImageNet-100, $R^2$ on AgeDB); "
        "means $\\pm$ std. over runs.}",
        "\\label{tab:compression_decomp}",
        "\\footnotesize",
        "{",
        "\\setlength{\\tabcolsep}{0.5pt}",
        "\\begin{tabular}{l c c c c c}",
        "\\hline",
        "Config & CE & KL mean term & KL var term & $\\E[\\KL]$ & Acc / $R^2$ \\\\",
        "\\hline",
    ]
    lines.extend(block("imagenet100", True))
    lines.append("\\hline\\hline")
    lines.extend(block("housing", False))
    lines.extend(["\\hline", "\\end{tabular}", "}", "\\end{table*}"])
    lines.append(
        "\\par\\smallskip\\textit{Note:} cells with $|\mathrm{value}| \ge 10^4$ "
        "(the variance-explosion rows at tiny $\\beta$ with $a=1$) show the mean only."
    )
    return "\n".join(lines) + "\n"


def rank_scores(scores):
    """按得分降序给排名（最优=1），并列取平均名次，返回与输入等长的排名列表。"""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2  # 名次 1 起，并列区间取平均
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def fmt(key, mean, std):
    """Acc 显示为百分比（两位小数）、R² 保留原值（四位小数）；不含标准差（省宽度）。"""
    if key == "Acc":
        return f"{mean * 100:.2f}"
    return f"{mean:.4f}"


def fmt_std(key, mean, std):
    """Acc 显示为百分比、R² 保留原值；均带标准差（均值±std，供 std 表使用）。"""
    if key == "Acc":
        return f"{mean * 100:.2f}$\\pm${std * 100:.2f}"
    return f"{mean:.4f}$\\pm${std:.4f}"


def _block_lines(grid, row_keys, fmt_fn, col_subset=None):
    """生成表格主体行；col_subset 缺省用全部列，子集用于拆表（std 版按列拆两表）。

    加粗/次优下划线始终按该行全部列（grid 全列）的均值判定，与主表一致——
    拆表不改变最优/次优归属。
    """
    cols = col_subset or COLUMN_ORDER
    out = []
    for task, backbone in row_keys:
        label = TASK_NAMES.get(task, task)
        backbone_label = BACKBONE_NAMES.get(backbone, backbone)
        cells = grid[(task, backbone)]
        best_mean = max(mean for _, mean, _ in cells.values())
        # 次优 = 去重得分的第二档；并列次优（同分）全部加下划线
        uniques = sorted({mean for _, mean, _ in cells.values()}, reverse=True)
        second_mean = uniques[1] if len(uniques) > 1 else None
        parts = [label, backbone_label]
        for col in cols:
            entry = cells.get(col)
            if entry is None:
                parts.append("--")
                continue
            key, mean, std = entry
            text = fmt_fn(key, mean, std)
            if mean == best_mean:
                text = "\\textbf{" + text + "}"
            elif mean == second_mean:
                text = "\\underline{" + text + "}"
            parts.append(text)
        out.append(" & ".join(parts) + " \\\\")
    return out


def _gen_table(grid, fmt_fn, with_std):
    """主结果表公共生成逻辑：gen_table（仅均值）与 gen_table_std（均值±标准差）
    共用；with_std 决定文件头注释、caption 与 label。"""
    order = {k: i for i, k in enumerate(ROW_ORDER)}
    keys = sorted(grid, key=lambda k: (order.get(k, len(order)), k))
    cls_keys = [k for k in keys if k[0] in CLASSIFICATION_TASKS]
    reg_keys = [k for k in keys if k[0] not in CLASSIFICATION_TASKS]

    std_note = "均值±标准差" if with_std else "多次运行均值（不含标准差）"
    lines = [
        "% 主结果表：由 paper/make_table.py 从 tune_results/*.html 自动生成，请勿手改。",
        f"% 分类任务为测试 Acc（%）、回归任务为测试 R²；均为 {std_note}。",
        "% 每格取该模型在 β / anchor-scale 调参网格上的最优配置（tune_results 各表绿色高亮行），",
        "% 每行（数据集）的最优指标加粗、次优（并列次优全部）加下划线；表尾两行统计各方法跨行的平均排名（越低越好）与最优/并列最优次数。",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test accuracy (\\%) on classification tasks and test $R^2$ on regression "
        + ("tasks (mean over runs" if not with_std else "tasks (mean $\\pm$ std. over runs")
        + "). Each entry reports the best configuration over "
        "the tuned $\\beta$ (and anchor-scale) grid; the best entry per dataset is bolded.}",
        "\\label{tab:main_result_std}" if with_std else "\\label{tab:main_result}",
        "\\small",
        "{",  # 花括号限定 tabcolsep 只在本表生效，不泄漏到论文其他表格
        "\\setlength{\\tabcolsep}{2pt}",  # 压缩列间距以收窄表格
        # opb 左侧插入竖线：GPB/GPB-L 与前面的 baseline 方法隔开
        "\\begin{tabular}{ll" + "c" * COLUMN_ORDER.index("opb") + "|"
        + "c" * (len(COLUMN_ORDER) - COLUMN_ORDER.index("opb")) + "}",
        "\\hline",
        " & ".join(["Dataset", "Backbone"] + [COLUMN_NAMES[c] for c in COLUMN_ORDER]) + " \\\\",
        "\\hline",
    ]

    lines.extend(_block_lines(grid, cls_keys, fmt_fn))
    lines.append("\\hline\\hline")  # 分类块与回归块的分界（上方 Acc、下方 R²）
    lines.extend(_block_lines(grid, reg_keys, fmt_fn))
    lines.append("\\hline")

    # 表尾两行汇总：各方法在所有数据集行上的平均排名（并列取平均名次，越小越好）
    # 与 最优/并列最优 次数（每行最优分相同时各并列方法均计数）
    all_keys = cls_keys + reg_keys
    rank_sums = [0.0] * len(COLUMN_ORDER)
    best_counts = [0] * len(COLUMN_ORDER)
    for kk in all_keys:
        cells = grid[kk]
        # 缺失列（如 opb 结果暂被改名为 opbl）以 -inf 参与排名（沉底、不计最优）
        means = [cells.get(c, (None, float("-inf"), 0.0))[1] for c in COLUMN_ORDER]
        ranks = rank_scores(means)
        best_mean = max(means)
        for j, mean in enumerate(means):
            rank_sums[j] += ranks[j]
            if mean == best_mean:
                best_counts[j] += 1
    n_rows = len(all_keys)
    lines.append("\\multicolumn{2}{l}{Mean rank (of %d)} & %s \\\\" % (
        n_rows, " & ".join(f"{r / n_rows:.2f}" for r in rank_sums)))
    lines.append("\\multicolumn{2}{l}{Best or tied-best (of %d)} & %s \\\\" % (
        n_rows, " & ".join(str(c) for c in best_counts)))
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def gen_table(grid):
    r"""生成 main_result.tex 内容（仅均值，table* 浮动体，可直接 \input{}）。"""
    return _gen_table(grid, fmt, False)


def gen_table_std(grid):
    r"""生成 main_result_std.tex 内容：主结果带标准差版，因 8 方法列 ×
    (均值±std) 超出页宽，按列拆成两张 table* 浮动体——
    (a) baselines（Base/VIB/SVIB/NIB，tab:main_result_std_baselines）与
    (b) CEB/DVCCA | GPB/GPB-L（tab:main_result_std_variants，竖线分隔同主表）；
    每表含全部 12 行（分类/回归块间双横线），加粗/下划线按该行全部 8 列的均值
    判定、与 main_result.tex 一致；表尾排名行省略（与主表重复）。"""
    order = {k: i for i, k in enumerate(ROW_ORDER)}
    keys = sorted(grid, key=lambda k: (order.get(k, len(order)), k))
    cls_keys = [k for k in keys if k[0] in CLASSIFICATION_TASKS]
    reg_keys = [k for k in keys if k[0] not in CLASSIFICATION_TASKS]

    def block_float(col_subset, caption, label):
        # 列规格：竖线插在 opb 左侧（与主表同位置）；无 opb 的子集无竖线
        colspec = (
            "ll" + "c" * len(col_subset)
            if "opb" not in col_subset
            else "ll" + "c" * col_subset.index("opb") + "|"
            + "c" * (len(col_subset) - col_subset.index("opb"))
        )
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            "\\small",
            "{",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\begin{tabular}{" + colspec + "}",
            "\\hline",
            " & ".join(["Dataset", "Backbone"] + [COLUMN_NAMES[c] for c in col_subset]) + " \\\\",
            "\\hline",
        ]
        lines.extend(_block_lines(grid, cls_keys, fmt_std, col_subset))
        lines.append("\\hline\\hline")
        lines.extend(_block_lines(grid, reg_keys, fmt_std, col_subset))
        lines.append("\\hline")
        lines.extend(["\\end{tabular}", "}", "\\end{table*}", "\\par\\smallskip"])
        return "\n".join(lines) + "\n"

    head = [
        "% 主结果表（带标准差）：由 paper/make_table.py 从 tune_results/*.html 自动生成，请勿手改。",
        "% 因 8 方法列 × (均值±std) 超出页宽，按列拆为两张表：",
        "% (a) baselines（Base/VIB/SVIB/NIB，tab:main_result_std_baselines）；",
        "% (b) CEB/DVCCA 与 GPB/GPB-L（tab:main_result_std_variants，竖线分隔同主表）。",
        "% 每表含全部 12 行；分类 Acc（%）、回归 R²，单元格为均值±std；",
        "% 加粗/下划线按该行全部 8 列的均值判定（与 main_result.tex 一致）；",
        "% 表尾排名行省略（与主表重复）。",
        "",
    ]
    out = "\n".join(head)
    out += block_float(
        ["base", "vib", "svib", "nib"],
        "Test accuracy (\\%) and test $R^2$ with standard deviations over runs: "
        "the plain backbone and the variational baselines; same configurations as "
        "Table~\\ref{tab:main_result}.",
        "tab:main_result_std_baselines",
    )
    out += block_float(
        ["ceb", "dvcca", "opb", "opbl"],
        "Test accuracy (\\%) and test $R^2$ with standard deviations over runs: "
        "CEB, DVCCA, and our method; same configurations as "
        "Table~\\ref{tab:main_result}. GPB and GPB-L are separated from the "
        "references by the vertical rule.",
        "tab:main_result_std_variants",
    )
    return out


def main():
    if not list(RESULTS_DIR.glob("*.html")):
        raise SystemExit(f"{RESULTS_DIR} 下未找到 html 文件")
    grid = collect()
    for (task, backbone), cells in sorted(grid.items()):
        detail = ", ".join(
            f"{COLUMN_NAMES.get(c, c)}={fmt(key, mean, std)}"
            for c, (key, mean, std) in sorted(cells.items())
        )
        print(f"{task:12s} {backbone:8s} | {detail}")
    OUT_PATH.write_text(gen_table(grid), encoding="utf-8")
    print(f"\n已生成 {OUT_PATH}")
    MAIN_RESULT_STD_PATH.write_text(gen_table_std(grid), encoding="utf-8")
    print(f"已生成 {MAIN_RESULT_STD_PATH}（主结果表带标准差版，均值±std，"
          f"加粗/下划线/排名与主表一致）")
    result_ci = gen_result_ci()
    if result_ci is None:
        print(f"警告：tune_results 无可用数据，跳过差值 CI 表")
    else:
        MAIN_RESULT_CI_PATH.write_text(result_ci, encoding="utf-8")
        print(f"已生成 {MAIN_RESULT_CI_PATH}（GPB − baseline 配对差值 95% CI，"
              f"按列拆两张表；加粗 = CI 下限 > 0、† = tied）")

    full = collect_full(COMPRESSION_TASKS)
    RESULT1_PATH.write_text(gen_result1(full), encoding="utf-8")
    print(f"已生成 {RESULT1_PATH}（压缩-精度表：CEB + GPB（OPB a / EPB ρ = 1/6/12）× β 网格，"
          f"仅 {', '.join(f'{TASK_NAMES.get(t, t)} ({b})' for t, b in COMPRESSION_TASKS)}）")

    result2 = gen_result2()
    if result2 is None:
        print(f"警告：{ADV_CSV_PATH} 不存在，跳过对抗鲁棒性表（先运行 adv_eval.py 生成）")
    else:
        RESULT2_PATH.write_text(result2, encoding="utf-8")
        print(f"已生成 {RESULT2_PATH}（对抗鲁棒性表：行 = 攻击强度、列 = 模型配置（自适应），"
              f"数据来自 {ADV_CSV_PATH}）")

    result4 = gen_result4()
    if result4 is None:
        print(f"警告：{COMPRESSION_EVAL_CSV} 不存在，跳过压缩-精度评估表"
              f"（先运行 compression_retrain.py 与 compression_eval.py 生成）")
    else:
        RESULT4_PATH.write_text(result4, encoding="utf-8")
        print(f"已生成 {RESULT4_PATH}（压缩-精度评估表：CE / KL 分解项，"
              f"数据来自 compression_eval.py）")

    result3 = gen_result3()
    if result3 is None:
        print(f"警告：{POSTERIOR_JSON} 不存在，跳过几何机制表"
              f"（先运行 prior_geometry.py 与 posterior_geometry.py 生成）")
    else:
        RESULT3_PATH.write_text(result3, encoding="utf-8")
        print(f"已生成 {RESULT3_PATH}（几何机制表：行 = 机制对照配置、列 = 三个概念指标，"
              f"数据来自 output/pri-pos 的机制验证 JSON）")


if __name__ == "__main__":
    main()
