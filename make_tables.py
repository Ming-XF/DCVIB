"""从 tune_results/ 的训练日志生成 paper.tex 使用的 LaTeX 表格。

用法：python make_tables.py [--results-dir tune_results] [--out-dir tables]

流程：
1. 扫描 tune_results/<config>/train.log，解析每个配置的
   - 每 run 的最佳验证指标（`Val AUC/R2 improved to X` 行，按 `Run i/N | Test` 边界分段取 max）
   - 最后的 `Average over N runs | Test ...` 汇总行（测试指标均值±标准差）
2. 模型选择**只看验证集**：每个 (任务, 骨干, 模型) 取平均最佳验证分最高的配置，报告其测试分。
3. 输出四张表到 out-dir：
   - tab_main.tex        主结果（验证集选参）
   - tab_robust.tex      beta 鲁棒性（全网格最差值相对确定性基线的缺口）
   - tab_anchor.tex      beta=10 时 FGIB 随锚点尺度 a 的变化
   - tab_main_std.tex    主表的标准差与选中的超参（附录）

分类指标为 Acc、回归为 R²，统一乘 100 输出。
"""

import argparse
import collections
import os
import re
import statistics

DATASETS = ["mnist", "imagenet100", "cora", "imdb", "agnews", "california", "stsb", "zinc", "agedb"]
BOTTLENECKS = ["vib", "svib", "nib", "ceb", "dvcca", "fgib"]
HEADERS = ["VIB", "SVIB", "NIB", "CEB", "DVCCA", "\\textbf{FGIB}"]
BETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ANCHORS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
# 相差 0.005 个百分点以内视为并列（浮点与四舍五入噪声）
TIE_EPS = 5e-5
# 相对基线缺口超过 5 个百分点记为「该设定被摧毁」
DESTROY_EPS = 0.05

NAMES = {
    "mnist_mlp": ("MNIST", "MLP"), "mnist_cnn": ("MNIST", "CNN"),
    "imagenet100_mlp": ("ImageNet-100", "MLP"), "imagenet100_cnn": ("ImageNet-100", "CNN"),
    "cora_gnn": ("Cora", "GCN"), "imdb_rnn": ("IMDb", "LSTM"), "agnews_rnn": ("AG News", "LSTM"),
    "california_mlp": ("Cal.\\ Housing", "MLP"), "stsb_rnn": ("STS-B", "LSTM"),
    "zinc_gnn": ("ZINC-12k", "GCN"), "agedb_mlp": ("AgeDB", "MLP"), "agedb_cnn": ("AgeDB", "CNN"),
}
CLS = ["mnist_mlp", "mnist_cnn", "imagenet100_mlp", "imagenet100_cnn",
       "cora_gnn", "imdb_rnn", "agnews_rnn"]
REG = ["california_mlp", "stsb_rnn", "zinc_gnn", "agedb_mlp", "agedb_cnn"]

VAL_RE = re.compile(r"Val (?:AUC|R2) improved to ([\d.\-]+)")
RUN_RE = re.compile(r"Run \d+/\d+ \| Test \(best model @ Epoch (\d+)\)")
AVG_RE = re.compile(r"Average over (\d+) runs \| Test (.*)")
# 每 epoch 行：`Epoch  N/100 | Train ... | Val Loss L Acc A AUC U ...`（回归无 Acc/AUC，有 MAE/R2）
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/\d+ .*\| Val .*?(?:Acc ([\d.\-]+)|MAE ([\d.\-]+))")


def parse_dirname(name):
    """`{ds}_{bb}_{model}[_beta_b][_anchor_a]` 或基线 `{ds}_{model}` → (ds, bb, model, beta, anchor)。"""
    m = re.search(r"_beta_([\d.e\-+]+)", name)
    beta = float(m.group(1)) if m else None
    m = re.search(r"_anchor_([\d.e\-+]+)", name)
    anchor = float(m.group(1)) if m else None
    base = re.sub(r"_anchor_[\d.e\-+]+", "", re.sub(r"_beta_[\d.e\-+]+", "", name))
    parts = base.split("_")
    if parts[0] not in DATASETS:
        return None
    ds, rest = parts[0], parts[1:]
    if len(rest) == 1:  # 基线：目录名不含 backbone 段，gcn 属 gnn 骨干
        return ds, {"gcn": "gnn"}.get(rest[0], rest[0]), rest[0], beta, anchor
    return ds, rest[0], rest[1], beta, anchor


def load(results_dir):
    """返回每个配置一条记录：dict(setting, model, beta, anchor, val, test)。"""
    recs = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name, "train.log")
        if not os.path.isfile(path):
            continue
        parsed = parse_dirname(name)
        if parsed is None:
            continue
        ds, bb, model, beta, anchor = parsed
        text = open(path, errors="ignore").read()

        # 逐 run 收集：主选择指标（最佳验证 AUC / R²）与并列时的次级指标
        # （该最佳 epoch 上的验证 Acc / −MAE）。两者都只来自验证集。
        vals, seconds = [], []
        cur, epochs = None, {}
        for line in text.splitlines():
            hit = VAL_RE.search(line)
            if hit:
                v = float(hit.group(1))
                cur = v if cur is None else max(cur, v)
            hit = EPOCH_RE.search(line)
            if hit:
                acc, mae = hit.group(2), hit.group(3)
                epochs[int(hit.group(1))] = float(acc) if acc else -float(mae)
            hit = RUN_RE.search(line)
            if hit:
                if cur is not None:
                    vals.append(cur)
                    seconds.append(epochs.get(int(hit.group(1))))
                cur, epochs = None, {}
        vals = [v for v in vals if v is not None]

        avg = None
        for avg in AVG_RE.finditer(text):
            pass
        if avg is None or not vals:
            raise RuntimeError(f"{path}: 缺少 val 记录或 Average 汇总行")
        # runs 以汇总行声明的 N 为准，而非 `Run i/N` 行数：少数配置（imdb 的 4 个）
        # 首次尝试 CUDA OOM 中断后整组重跑，日志里残留被作废的 run 行，会多计。
        n_runs = int(avg.group(1))
        test = {k: (float(v), float(s))
                for k, v, s in re.findall(r"(\w+) ([\d.\-]+)±([\d.]+)", avg.group(2))}
        sec = [s for s in seconds[-n_runs:] if s is not None]
        recs.append(dict(setting=f"{ds}_{bb}", model=model, beta=beta, anchor=anchor,
                         val=statistics.fmean(vals[-n_runs:]),
                         val2=statistics.fmean(sec) if sec else float("-inf"),
                         runs=n_runs, test=test))
    return recs


def metric_key(setting):
    return "R2" if setting in REG else "Acc"


def pick_key(r):
    """验证集选模的排序键：主指标（最佳验证 AUC/R²）+ 次级验证指标 + 固定配置顺序。

    日志只到 4 位小数、分类任务上 AUC 常饱和，因此精确打平并不罕见
    （72 个格子中约 9 个）。打平时用次级**验证**指标（最佳 epoch 的验证
    Acc / −MAE）分胜负；仍然打平则按固定配置顺序（β 升序、a 升序）取
    网格中先出现的组合。**测试分数不参与选模**（审稿人要求：任何
    test-dependent selection 都会使测试集失去"只读一次"的地位）。
    """
    return (
        r["val"], r["val2"],
        -(r["beta"] if r["beta"] is not None else 0.0),
        -(r["anchor"] if r["anchor"] is not None else 0.0),
    )


def select_one(items):
    """按 pick_key 从配置列表中选择最佳记录（确定性、无测试信息）。"""
    return max(items, key=pick_key)


def build_index(recs):
    by_setting = collections.defaultdict(list)
    for r in recs:
        by_setting[r["setting"]].append(r)
    selected, baseline = {}, {}
    for setting, items in by_setting.items():
        key = metric_key(setting)
        best = {}
        for r in items:
            if r["model"] not in BOTTLENECKS:      # 确定性基线，无 beta 维度
                baseline[setting] = r
                continue
            cur = best.get(r["model"])
            if cur is None or pick_key(r) > pick_key(cur):
                best[r["model"]] = r
        selected[setting] = best
        assert setting in baseline, f"{setting} 缺少确定性基线"
        assert set(best) == set(BOTTLENECKS), f"{setting} 缺少模型 {set(BOTTLENECKS) - set(best)}"
    return by_setting, selected, baseline


def build_fixed_anchor_index(by_setting):
    """FGIB 的等预算对照：固定 a=16、只扫 6 个 beta（与其余瓶颈同预算），验证集选模。

    返回 {setting: 选中记录}，用于报告与其余瓶颈搜索预算相同时 FGIB 的 rank。
    """
    out = {}
    for setting, items in by_setting.items():
        cands = [r for r in items
                 if r["model"] == "fgib" and r["anchor"] == 16.0 and r["beta"] is not None]
        assert len(cands) == len(BETAS), f"{setting}: 固定 a=16 的 fgib 配置数 {len(cands)}"
        out[setting] = select_one(cands)
    return out


def fmt(v, bold=False):
    s = f"{100 * v:.2f}"
    return "\\textbf{" + s + "}" if bold else s


def rank_stats(vals_by_name):
    """按指标值计算各方法的 (mean_rank, wins)：vals_by_name[name] = 12 个设定值列表。"""
    # wins 预置全部键为 0：从未赢过任何设定的方法也要有输出
    ranks, wins = collections.defaultdict(list), {name: 0 for name in vals_by_name}
    for i in range(len(next(iter(vals_by_name.values())))):
        vals = [vals_by_name[n][i] for n in vals_by_name]
        best = max(vals)
        for name in vals_by_name:
            v = vals_by_name[name][i]
            ranks[name].append(1 + sum(1 for w in vals if w > v + TIE_EPS))
            if v >= best - TIE_EPS:
                wins[name] += 1
    mean_rank = {m: statistics.fmean(ranks[m]) for m in vals_by_name}
    return mean_rank, wins


def equal_budget_fgib_summary(by_setting, selected, baseline, out):
    """等预算对照：FGIB 固定 a=16（6 个 beta，与其余瓶颈同预算）的选模结果。

    打印 mean rank / wins 对比（全网格 FGIB vs 固定 a=16 FGIB），并写入
    out（LaTeX 片段）供论文正文引用。
    """
    fgib16 = build_fixed_anchor_index(by_setting)
    order = ["det"] + BOTTLENECKS
    full = {m: [selected[s][m]["test"][metric_key(s)][0] for s in CLS + REG] for m in BOTTLENECKS}
    full["det"] = [baseline[s]["test"][metric_key(s)][0] for s in CLS + REG]
    eq = dict(full)
    eq["fgib"] = [fgib16[s]["test"][metric_key(s)][0] for s in CLS + REG]
    mr_full, wins_full = rank_stats(full)
    mr_eq, wins_eq = rank_stats(eq)
    print(f"[等预算对照] FGIB 全网格(54 组) mean rank = {mr_full['fgib']:.2f}, wins = {wins_full['fgib']}/12")
    print(f"[等预算对照] FGIB 固定 a=16(6 组) mean rank = {mr_eq['fgib']:.2f}, wins = {wins_eq['fgib']}/12")
    print(f"[等预算对照] 其余方法（不受影响）：" + ", ".join(
        f"{m}={mr_eq[m]:.2f}({wins_eq[m]}/12)" for m in order if m != "fgib"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "\\emph{Equal-budget control.} Selected over the same 6-point $\\beta$ grid as "
            "every other bottleneck at a single globally fixed anchor scale $a=16$ (a post-hoc "
            "sensitivity control), FGIB's "
            f"mean rank is {mr_eq['fgib']:.2f} of 7 (vs.\\ {mr_full['fgib']:.2f} over the full "
            f"$6\\times9$ grid) and it is best or tied-best on {wins_eq['fgib']} of 12 settings "
            f"(vs.\\ {wins_full['fgib']}).\n"
        )
    return fgib16


def table_main(selected, baseline, out):
    lines = [
        "\\begin{table}[t]",
        "\\caption{Test performance of the validation-selected configuration of each objective, "
        "averaged over 5 runs (seeds 0--4). Accuracy (\\%) for classification, $R^2\\times100$ for "
        "regression; higher is better. \\emph{Det.}\\ is the deterministic backbone trained with the "
        "task loss alone. For every bottleneck $\\beta$ is chosen per cell by mean best-epoch "
        "validation score --- and for FGIB the anchor scale $a$ as well --- never by test score, so "
        "FGIB is selected over a $6\\times9$ grid and the others over a $6$-point grid; exact ties "
        "are broken by a second validation quantity and then by fixed grid order. Best per row "
        "in bold (ties within $0.01$ points both bolded); per-run standard deviations and the "
        "selected hyperparameters are in Table~\\ref{tab:main-std}, whose last column reports the "
        "equal-budget FGIB control.}",
        "\\label{tab:main}",
        "\\begin{center}\\small",
        "\\begin{tabular}{llccccccc}",
        "Task & Backbone & Det. & " + " & ".join(HEADERS) + " \\\\ \\hline \\\\[-1.8ex]",
    ]
    ranks, wins = collections.defaultdict(list), collections.Counter()
    for group, label in [(CLS, "\\emph{Classification (Acc \\%)}"),
                         (REG, "\\emph{Regression ($R^2\\times100$)}")]:
        lines.append("\\multicolumn{9}{l}{" + label + "} \\\\[0.3ex]")
        for setting in group:
            task, bb = NAMES[setting]
            key = metric_key(setting)
            vals = [baseline[setting]["test"][key][0]]
            vals += [selected[setting][m]["test"][key][0] for m in BOTTLENECKS]
            best = max(vals)
            lines.append(f"{task} & {bb} & "
                         + " & ".join(fmt(v, v >= best - TIE_EPS) for v in vals) + " \\\\")
            for name, v in zip(["det"] + BOTTLENECKS, vals):
                ranks[name].append(1 + sum(1 for w in vals if w > v + TIE_EPS))
                if v >= best - TIE_EPS:
                    wins[name] += 1
        lines.append("\\\\[-1.5ex]")
    order = ["det"] + BOTTLENECKS
    mean_rank = {m: statistics.fmean(ranks[m]) for m in order}
    top = min(mean_rank.values())
    best_wins = max(wins[m] for m in order)
    lines.append("\\hline \\\\[-1.8ex]")
    lines.append("\\multicolumn{2}{l}{Mean rank (of 7)} & " + " & ".join(
        ("\\textbf{" + f"{mean_rank[m]:.2f}" + "}") if mean_rank[m] <= top + 1e-9
        else f"{mean_rank[m]:.2f}" for m in order) + " \\\\")
    lines.append("\\multicolumn{2}{l}{Best or tied-best (of 12)} & " + " & ".join(
        ("\\textbf{" + str(wins[m]) + "}") if wins[m] == best_wins else str(wins[m])
        for m in order) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(out, "w").write("\n".join(lines) + "\n")


def series(by_setting, setting, model, anchor=None):
    """该模型在 6 个 beta 上的测试指标（fgib 需固定 anchor）。"""
    key = metric_key(setting)
    return {r["beta"]: r["test"][key][0] for r in by_setting[setting]
            if r["model"] == model and r["beta"] is not None
            and (anchor is None or r["anchor"] == anchor)}


def table_robust(by_setting, selected, fgib16, baseline, out):
    cols = BOTTLENECKS + ["fgib16"]
    stats = {}
    for setting in CLS + REG:
        base = baseline[setting]["test"][metric_key(setting)][0]
        stats[setting] = {"base": base}
        for m in BOTTLENECKS:
            anchor = selected[setting]["fgib"]["anchor"] if m == "fgib" else None
            s = series(by_setting, setting, m, anchor)
            assert len(s) == len(BETAS), f"{setting}/{m}: {len(s)} betas"
            # best-beta 行是**测试集上**取最大——探索性上界，不是选模结果
            # （选模只用验证集；此行的定位在 caption 与论文正文中说明）
            stats[setting][m] = {"worst": base - min(s.values()),
                                 "best_test": max(s.values()) - base,
                                 "val_gain": selected[setting][m]["test"][metric_key(setting)][0] - base}
        s16 = series(by_setting, setting, "fgib", 16.0)
        stats[setting]["fgib16"] = {"worst": base - min(s16.values()),
                                    "best_test": max(s16.values()) - base,
                                    "val_gain": fgib16[setting]["test"][metric_key(setting)][0] - base}
    lines = [
        "\\begin{table}[t]",
        "\\caption{Robustness of the compression knob. For each objective we sweep "
        "$\\beta\\in\\{10^{-4},10^{-3},10^{-2},10^{-1},1,10\\}$ and report the \\emph{worst} test "
        "score any $\\beta$ produces, as a deficit (in points) against the deterministic backbone of "
        "the same row; negative means the objective never falls below its backbone. Units as in "
        "Table~\\ref{tab:main}. FGIB is shown twice: at the anchor scale selected on validation in "
        "Table~\\ref{tab:main}, and at a single globally fixed anchor scale $a=16$ (a post-hoc sensitivity control, not selected a priori). Lower is better; best per row"
        "in bold. The summary rows take medians over the 12 settings and count settings where the "
        "worst case falls more than 5 points below the backbone. The ``best $\\beta$'' gain row is "
        "the maximum over the grid of the \\emph{test} score --- an exploratory upper bound, not a "
        "model-selection result; the validation-selected gain row is the selection result of "
        "Table~\\ref{tab:main} expressed relative to the backbone.}",
        "\\label{tab:robust}",
        "\\begin{center}\\small",
        "\\begin{tabular}{llccccc|cc}",
        "& & \\multicolumn{5}{c|}{} & \\multicolumn{2}{c}{\\textbf{FGIB}} \\\\",
        "Task & Backbone & " + " & ".join(HEADERS[:-1])
        + " & val-sel.\\ $a$ & $a{=}16$ \\\\ \\hline \\\\[-1.8ex]",
    ]
    for group, label in [(CLS, "\\emph{Classification (Acc \\%)}"),
                         (REG, "\\emph{Regression ($R^2\\times100$)}")]:
        lines.append("\\multicolumn{9}{l}{" + label + "} \\\\[0.3ex]")
        for setting in group:
            task, bb = NAMES[setting]
            vals = [stats[setting][c]["worst"] for c in cols]
            best = min(vals)
            lines.append(f"{task} & {bb} & "
                         + " & ".join(fmt(v, v <= best + 1e-9) for v in vals) + " \\\\")
        lines.append("\\\\[-1.5ex]")
    lines.append("\\hline \\\\[-1.8ex]")
    med_worst = [statistics.median([stats[s][c]["worst"] for s in CLS + REG]) for c in cols]
    destroyed = [sum(1 for s in CLS + REG if stats[s][c]["worst"] > DESTROY_EPS) for c in cols]
    med_best_test = [statistics.median([stats[s][c]["best_test"] for s in CLS + REG]) for c in cols]
    med_val_gain = [statistics.median([stats[s][c]["val_gain"] for s in CLS + REG]) for c in cols]
    lines.append("\\multicolumn{2}{l}{Median deficit} & "
                 + " & ".join(fmt(v) for v in med_worst) + " \\\\")
    lines.append("\\multicolumn{2}{l}{Settings destroyed (of 12)} & "
                 + " & ".join(str(v) for v in destroyed) + " \\\\")
    lines.append("\\multicolumn{2}{l}{Median gain at best $\\beta$ (test-selected upper bound)} & "
                 + " & ".join(fmt(v) for v in med_best_test) + " \\\\")
    lines.append("\\multicolumn{2}{l}{Median gain at validation-selected $\\beta$} & "
                 + " & ".join(fmt(v) for v in med_val_gain) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(out, "w").write("\n".join(lines) + "\n")


def table_anchor(by_setting, baseline, out):
    lines = [
        "\\begin{table}[t]",
        "\\caption{The anchor scale controls the fully compressed corner. Test score of FGIB at the "
        "strongest compression pressure in the grid ($\\beta=10$), as the anchor scale $a$ varies; "
        "$\\Delta$ is the gap from the deterministic backbone at the largest anchor ($a{=}16$). Units "
        "as in Table~\\ref{tab:main}. At small $a$ the anchors approach the single origin of a VIB "
        "prior and the model collapses to the trivial predictor (Remark~\\ref{rem:a0}); growing $a$ "
        "recovers the backbone, and the 100-class task needs the largest $a$, as the "
        "$\\ln(1+(K-1)e^{-ca})$ corner value predicts.}",
        "\\label{tab:anchor}",
        "\\begin{center}\\small",
        "\\begin{tabular}{llccccccccc|c}",
        "Task & Backbone & \\multicolumn{9}{c|}{anchor scale $a$} & $\\Delta$ \\\\",
        " & & " + " & ".join(str(a) for a in ANCHORS) + " & \\\\ \\hline \\\\[-1.8ex]",
    ]
    for group, label in [(CLS, "\\emph{Classification (Acc \\%)}"),
                         (REG, "\\emph{Regression ($R^2\\times100$)}")]:
        lines.append("\\multicolumn{12}{l}{" + label + "} \\\\[0.3ex]")
        for setting in group:
            task, bb = NAMES[setting]
            key = metric_key(setting)
            row = {r["anchor"]: r["test"][key][0] for r in by_setting[setting]
                   if r["model"] == "fgib" and r["beta"] == 10.0}
            assert len(row) == len(ANCHORS), f"{setting}: {len(row)} anchors at beta=10"
            delta = row[16.0] - baseline[setting]["test"][key][0]
            lines.append(f"{task} & {bb} & "
                         + " & ".join(f"{100 * row[a]:.1f}" for a in ANCHORS)
                         + f" & {100 * delta:+.1f} \\\\")
        lines.append("\\\\[-1.5ex]")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(out, "w").write("\n".join(lines) + "\n")


def table_main_std(selected, fgib16, baseline, out):
    lines = [
        "\\begin{table}[t]",
        "\\caption{Standard deviations over the 5 runs for every cell of Table~\\ref{tab:main}, with "
        "the validation-selected hyperparameters underneath each cell ($\\beta$, and $\\beta,a$ for "
        "FGIB; \\texttt{--} for the deterministic backbone, which has neither). The last column is "
        "the equal-budget control: FGIB selected over the same 6-point $\\beta$ grid as every other "
        "bottleneck at a single globally fixed anchor scale $a=16$ (a post-hoc sensitivity "
        "control), so its search budget matches the comparison (see Section~\\ref{sec:exp}). "
        "Units as in Table~\\ref{tab:main}.}",
        "\\label{tab:main-std}",
        "\\begin{center}\\scriptsize",
        "\\begin{tabular}{llcccccccc}",
        "Task & Backbone & Det. & " + " & ".join(h.replace("\\textbf{", "").replace("}", "")
                                                 for h in HEADERS) + " & FGIB ($a{=}16$) \\\\ \\hline \\\\[-1.8ex]",
    ]
    for group, label in [(CLS, "\\emph{Classification (Acc \\%)}"),
                         (REG, "\\emph{Regression ($R^2\\times100$)}")]:
        lines.append("\\multicolumn{10}{l}{" + label + "} \\\\[0.3ex]")
        for setting in group:
            task, bb = NAMES[setting]
            key = metric_key(setting)
            cells, hyper = [], []
            for r in [baseline[setting]] + [selected[setting][m] for m in BOTTLENECKS] + [fgib16[setting]]:
                mu, sd = r["test"][key]
                cells.append(f"{100 * mu:.2f}$\\pm${100 * sd:.2f}")
                if r["beta"] is None:
                    hyper.append("--")
                elif r["anchor"] is None:
                    hyper.append(f"{r['beta']:g}")
                else:
                    hyper.append(f"{r['beta']:g},{r['anchor']:g}")
            lines.append(f"{task} & {bb} & " + " & ".join(cells) + " \\\\")
            lines.append(" & & " + " & ".join("{\\tiny " + h + "}" for h in hyper) + " \\\\[0.4ex]")
        lines.append("\\\\[-1.5ex]")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(out, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="tune_results")
    ap.add_argument("--out-dir", default="tables")
    args = ap.parse_args()

    recs = load(args.results_dir)
    by_setting, selected, baseline = build_index(recs)
    os.makedirs(args.out_dir, exist_ok=True)
    fgib16 = equal_budget_fgib_summary(
        by_setting, selected, baseline,
        os.path.join(args.out_dir, "equal_budget.tex"),
    )
    table_main(selected, baseline, os.path.join(args.out_dir, "tab_main.tex"))
    table_robust(by_setting, selected, fgib16, baseline, os.path.join(args.out_dir, "tab_robust.tex"))
    table_anchor(by_setting, baseline, os.path.join(args.out_dir, "tab_anchor.tex"))
    table_main_std(selected, fgib16, baseline, os.path.join(args.out_dir, "tab_main_std.tex"))
    print(f"{len(recs)} configurations, {sum(r['runs'] for r in recs)} runs, "
          f"{len(by_setting)} task×backbone settings → {args.out_dir}/")


if __name__ == "__main__":
    main()
