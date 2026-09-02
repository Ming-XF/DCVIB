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
\input{} 的 table* 浮动体，保存到 paper/main_result.tex。
"""

import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "tune_results"
OUT_PATH = Path(__file__).resolve().parent / "main_result.tex"

# 分类任务集合（决定行顺序中分类块与回归块的分界）
CLASSIFICATION_TASKS = {"mnist", "imagenet100", "cora", "imdb", "agnews"}

# 数据集/骨干显示名
TASK_NAMES = {
    "mnist": "MNIST",
    "imagenet100": "ImageNet-100",
    "cora": "Cora",
    "imdb": "IMDb",
    "agnews": "AG News",
    "housing": "California Housing",
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
# opbl 是 opb 的 linear 消融（结果目录 tune_results 中 opb 已改名为 opbl），
# 排在 opb 之后；"opb" 在 COLUMN_ORDER 中的位置是竖线插入点（其左侧加竖线，
# 把 OPB/OPB-L 与前面的 baseline 方法隔开）。
COLUMN_ORDER = ["base", "vib", "svib", "nib", "ceb", "dvcca", "opb", "opbl"]
COLUMN_NAMES = {
    "base": "Base", "vib": "VIB", "svib": "SVIB", "nib": "NIB",
    "ceb": "CEB", "dvcca": "DVCCA", "opb": "OPB", "opbl": "OPB-L",
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
        self.rows.setdefault(self._model, []).append((self._tr_best, key, mean, std))

    def best_values(self):
        """各模型取 best 行（缺失时回退指标最大行），返回 {model: (key, mean, std)}。"""
        out = {}
        for model, entries in self.rows.items():
            bests = [e for e in entries if e[0]]
            pool = bests or entries
            _, key, mean, std = max(pool, key=lambda e: e[2])
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


def gen_table(grid):
    r"""生成 main_result.tex 内容（table* 浮动体，可直接 \input{}）。"""
    order = {k: i for i, k in enumerate(ROW_ORDER)}
    keys = sorted(grid, key=lambda k: (order.get(k, len(order)), k))
    cls_keys = [k for k in keys if k[0] in CLASSIFICATION_TASKS]
    reg_keys = [k for k in keys if k[0] not in CLASSIFICATION_TASKS]

    lines = [
        "% 主结果表：由 paper/make_table.py 从 tune_results/*.html 自动生成，请勿手改。",
        "% 分类任务为测试 Acc（%）、回归任务为测试 R²；均为 5 次运行均值（不含标准差）。",
        "% 每格取该模型在 β / anchor-scale 调参网格上的最优配置（tune_results 各表绿色高亮行），",
        "% 每行（数据集）的最优指标加粗、次优（并列次优全部）加下划线；表尾两行统计各方法跨行的平均排名（越低越好）与最优/并列最优次数。",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test accuracy (\\%) on classification tasks and test $R^2$ on regression "
        "tasks (mean over 5 runs). Each entry reports the best configuration over "
        "the tuned $\\beta$ (and anchor-scale) grid; the best entry per dataset is bolded.}",
        "\\label{tab:main_result}",
        "\\small",
        "{",  # 花括号限定 tabcolsep 只在本表生效，不泄漏到论文其他表格
        "\\setlength{\\tabcolsep}{3pt}",  # 压缩列间距以收窄表格
        # opb 左侧插入竖线：OPB/OPB-L 与前面的 baseline 方法隔开
        "\\begin{tabular}{ll" + "c" * COLUMN_ORDER.index("opb") + "|"
        + "c" * (len(COLUMN_ORDER) - COLUMN_ORDER.index("opb")) + "}",
        "\\hline",
        " & ".join(["Dataset", "Backbone"] + [COLUMN_NAMES[c] for c in COLUMN_ORDER]) + " \\\\",
        "\\hline",
    ]

    def block_lines(row_keys):
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
            for col in COLUMN_ORDER:
                entry = cells.get(col)
                if entry is None:
                    parts.append("--")
                    continue
                key, mean, std = entry
                text = fmt(key, mean, std)
                if mean == best_mean:
                    text = "\\textbf{" + text + "}"
                elif mean == second_mean:
                    text = "\\underline{" + text + "}"
                parts.append(text)
            out.append(" & ".join(parts) + " \\\\")
        return out

    lines.extend(block_lines(cls_keys))
    lines.append("\\hline\\hline")  # 分类块与回归块的分界（上方 Acc、下方 R²）
    lines.extend(block_lines(reg_keys))
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


if __name__ == "__main__":
    main()
