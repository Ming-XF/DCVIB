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
   - tab_anchor.tex      beta=10 时 FGIB-A 随锚点尺度 a 的变化
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
HEADERS = ["VIB", "SVIB", "NIB", "CEB", "DVCCA", "\\textbf{FGIB-A}"]
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


def load(results_dir, strict=True):
    """返回每个配置一条记录：dict(setting, model, beta, anchor, val, test)。

    strict=False 时跳过尚未完成的日志（缺 Average 汇总行），用于新实验
    目录中还有子进程在跑的场景（如 table_hanchor）；主流程保持 strict。
    """
    recs = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name, "train.log")
        if not os.path.isfile(path):
            continue
        parsed = parse_dirname(name)
        if parsed is None:
            continue
        if not strict and "Average over" not in open(path, errors="ignore").read():
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


def build_fixed_anchor_index(by_setting, anchor=16.0):
    """FGIB-A 的等预算对照：固定 anchor（默认 a=16）、只扫 6 个 beta（与其余瓶颈
    同预算），验证集选模。

    返回 {setting: 选中记录}，用于报告与其余瓶颈搜索预算相同时 FGIB-A 的 rank；
    anchor 参数化后也用于逐 a 的固定锚点敏感性（审稿人第 5 条：a=16 是后验
    选择，应报告多个固定 a 的 rank）。
    """
    out = {}
    for setting, items in by_setting.items():
        cands = [r for r in items
                 if r["model"] == "fgib" and r["anchor"] == float(anchor)
                 and r["beta"] is not None]
        assert len(cands) == len(BETAS), f"{setting}: 固定 a={anchor} 的 fgib 配置数 {len(cands)}"
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
    """等预算对照：FGIB-A 固定 a=16（6 个 beta，与其余瓶颈同预算）的选模结果。

    打印 mean rank / wins 对比（全网格 FGIB-A vs 固定 a=16 FGIB-A），并写入
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
    print(f"[等预算对照] FGIB-A 全网格(54 组) mean rank = {mr_full['fgib']:.2f}, wins = {wins_full['fgib']}/12")
    print(f"[等预算对照] FGIB-A 固定 a=16(6 组) mean rank = {mr_eq['fgib']:.2f}, wins = {wins_eq['fgib']}/12")
    print(f"[等预算对照] 其余方法（不受影响）：" + ", ".join(
        f"{m}={mr_eq[m]:.2f}({wins_eq[m]}/12)" for m in order if m != "fgib"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "\\emph{Equal-budget control.} Selected over the same 6-point $\\beta$ grid as "
            "every other bottleneck at a single globally fixed anchor scale $a=16$ (a post-hoc "
            "sensitivity control), FGIB-A's "
            f"mean rank is {mr_eq['fgib']:.2f} of 7 (vs.\\ {mr_full['fgib']:.2f} over the full "
            f"$6\\times9$ grid) and it is best or tied-best on {wins_eq['fgib']} of 12 settings "
            f"(vs.\\ {wins_full['fgib']}).\n"
        )
    return fgib16


def _main_table_rows(selected, baseline, settings, caption, label, out):
    """生成一个"验证集选模测试分"表的共享实现（第四轮：主表分类 7 设置、
    回归 5 设置单独成附录表并标注 exploratory）。"""
    lines = ["\\begin{table}[t]", caption, "\\label{" + label + "}",
             "\\begin{center}\\footnotesize",
             "\\begin{tabular}{llccccccc}",
             "Task & Backbone & Det. & " + " & ".join(HEADERS) + " \\\\ \\hline \\\\[-1.8ex]"]
    ranks, wins = collections.defaultdict(list), collections.Counter()
    for setting in settings:
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
    order = ["det"] + BOTTLENECKS
    mean_rank = {m: statistics.fmean(ranks[m]) for m in order}
    top = min(mean_rank.values())
    best_wins = max(wins[m] for m in order)
    n = len(settings)
    lines.append("\\hline \\\\[-1.8ex]")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Mean rank (of {n})}} & " + " & ".join(
        ("\\textbf{" + f"{mean_rank[m]:.2f}" + "}") if mean_rank[m] <= top + 1e-9
        else f"{mean_rank[m]:.2f}" for m in order) + " \\\\")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Best or tied-best (of {n})}} & " + " & ".join(
        ("\\textbf{" + str(wins[m]) + "}") if wins[m] == best_wins else str(wins[m])
        for m in order) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(out, "w").write("\n".join(lines) + "\n")
    return {m: mean_rank[m] for m in order}, dict(wins), n


def table_main(selected, baseline, out):
    caption = (
        "\\caption{Test accuracy (\\%) of the validation-selected configuration of each objective on "
        "the seven classification settings, averaged over 5 runs (seeds 0--4); higher is better. "
        "\\emph{Det.}\\ is the deterministic backbone trained with the task loss alone. For every "
        "bottleneck $\\beta$ is chosen per cell by mean best-epoch validation score --- and for FGIB-A "
        "the anchor scale $a$ as well --- never by test score, so FGIB-A is selected over a $6\\times9$ "
        "grid and the others over a $6$-point grid; exact ties are broken by a second validation "
        "quantity and then by fixed grid order. The FGIB-A column is the optional trainable-head "
        "variant; the canonical fixed head (FGIB-H) is Table~\\ref{tab:hanchor}. The five regression "
        "settings are Table~\\ref{tab:main-reg}, reported as exploratory. Best per row in bold (ties "
        "within $0.01$ points both bolded); per-run standard deviations and the selected "
        "hyperparameters are in Table~\\ref{tab:main-std}, whose last column reports the equal-budget "
        "FGIB-A control.}"
    )
    return _main_table_rows(selected, baseline, CLS, caption, "tab:main", out)


def table_main_reg(selected, baseline, out):
    caption = (
        "\\caption{Regression settings ($R^2\\times100$), reported as \\emph{exploratory}: the "
        "random-Fourier anchor construction for continuous targets is a transfer, not a proof "
        "(Remark~\\ref{rem:caveats}), and the knob destroys FGIB-A on three of these five settings at "
        "some $\\beta$ (Section~\\ref{sec:knob}). Protocol and notation as in Table~\\ref{tab:main}.}"
    )
    return _main_table_rows(selected, baseline, REG, caption, "tab:main-reg", out)


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
        "\\caption{Robustness of the compression knob: the \\emph{worst} test score any "
        "$\\beta$ produces, as a deficit (in points) against the deterministic backbone of the same "
        "row; negative means the objective never falls below its backbone. Units as in "
        "Table~\\ref{tab:main}. FGIB-A is shown at the validation-selected anchor and at a single "
        "globally fixed $a=16$ (a post-hoc sensitivity control). Summary rows: medians over the 12 "
        "settings, the count of settings destroyed (worst case $>5$ points below the backbone), and "
        "two gain summaries --- ``best $\\beta$'' is a test-selected exploratory upper bound, "
        "``validation-selected'' is the selection result of Table~\\ref{tab:main} relative to the "
        "backbone.}",
        "\\label{tab:robust}",
        "\\begin{center}\\footnotesize",
        "\\begin{tabular}{llccccc|cc}",
        "& & \\multicolumn{5}{c|}{} & \\multicolumn{2}{c}{\\textbf{FGIB-A}} \\\\",
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
        "\\caption{The anchor scale controls the fully compressed corner. Test score of FGIB-A at the "
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
        "\\caption{Standard deviations over the 5 runs for every cell of "
        "Tables~\\ref{tab:main} and~\\ref{tab:main-reg}, with "
        "the validation-selected hyperparameters underneath each cell ($\\beta$, and $\\beta,a$ for "
        "FGIB-A; \\texttt{--} for the deterministic backbone, which has neither). The last column is "
        "the equal-budget control: FGIB-A selected over the same 6-point $\\beta$ grid as every other "
        "bottleneck at a single globally fixed anchor scale $a=16$ (a post-hoc sensitivity "
        "control), so its search budget matches the comparison (see Section~\\ref{sec:exp}). "
        "Units as in Table~\\ref{tab:main}.}",
        "\\label{tab:main-std}",
        "\\begin{center}\\scriptsize",
        "\\begin{tabular}{llcccccccc}",
        "Task & Backbone & Det. & " + " & ".join(h.replace("\\textbf{", "").replace("}", "")
                                                 for h in HEADERS) + " & FGIB-A ($a{=}16$) \\\\ \\hline \\\\[-1.8ex]",
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


def rank_split(selected, baseline, fgib16, out):
    """分类 / 回归 / 排除 AgeDB 的 mean rank 分解（审稿人第 4 条）。

    主表 12 设定联合 rank 混合了分类准确率与回归 R²；AgeDB 的 image-level
    划分有身份泄漏（论文已披露）。此函数报告：
    - CLS-only（7 设定）、REG-only（5 设定）
    - 排除 AgeDB 的 10 设定（CLS + 3 个回归）
    三种口径下全网格 FGIB-A 与等预算 FGIB-A(a=16) 的 mean rank 与 wins，
    写入 out（LaTeX 片段）供正文引用。
    """
    order = ["det"] + BOTTLENECKS
    subsets = {"CLS": CLS, "REG": REG,
               "exAgeDB": [s for s in CLS + REG if s not in ("agedb_mlp", "agedb_cnn")]}
    lines, prints = [], []
    for tag, ss in subsets.items():
        full = {m: [selected[s][m]["test"][metric_key(s)][0] for s in ss]
                for m in BOTTLENECKS}
        full["det"] = [baseline[s]["test"][metric_key(s)][0] for s in ss]
        eq = dict(full)
        eq["fgib"] = [fgib16[s]["test"][metric_key(s)][0] for s in ss]
        mr_full, wins_full = rank_stats(full)
        mr_eq, wins_eq = rank_stats(eq)
        prints.append(
            f"[rank 分解 {tag:7s} n={len(ss):2d}] FGIB-A 全网格 rank {mr_full['fgib']:.2f} "
            f"({wins_full['fgib']}/{len(ss)}) | FGIB-A a=16 rank {mr_eq['fgib']:.2f} "
            f"({wins_eq['fgib']}/{len(ss)}) | det {mr_eq['det']:.2f} | "
            + ", ".join(f"{m} {mr_eq[m]:.2f}" for m in BOTTLENECKS if m != "fgib"))
        lines.append(
            f"\\item[{tag}] full-grid FGIB-A mean rank {mr_full['fgib']:.2f} of 7 "
            f"(best or tied {wins_full['fgib']}/{len(ss)}); equal-budget FGIB-A ($a{{=}}16$) "
            f"mean rank {mr_eq['fgib']:.2f} (best or tied {wins_eq['fgib']}/{len(ss)}); "
            f"deterministic backbone {mr_eq['det']:.2f}.")
    for p in prints:
        print(p)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\\begin{itemize}\n" + "\n".join(lines) + "\n\\end{itemize}\n")


def table_anchor_ranks(by_setting, baseline, out_dir):
    """逐固定 a 的等预算 FGIB-A rank 敏感性（审稿人第 5 条：a=16 为后验选择）。

    对每个 a ∈ ANCHORS 用与其余瓶颈相同的 6 点 β 预算验证集选模，报告
    12 设定 / 仅分类 / 仅回归的 mean rank 与 wins，写入附录表。
    """
    rows = []
    for a in ANCHORS:
        fgib_a = build_fixed_anchor_index(by_setting, a)
        cells = []
        wins12 = None
        for tag, ss in [("12", CLS + REG), ("cls", CLS), ("reg", REG)]:
            vals = {m: [selected0(by_setting, s, m, a)["test"][metric_key(s)][0]
                        for s in ss] for m in BOTTLENECKS}
            vals["det"] = [baseline[s]["test"][metric_key(s)][0] for s in ss]
            vals["fgib"] = [fgib_a[s]["test"][metric_key(s)][0] for s in ss]
            mr, wins = rank_stats(vals)
            cells.append(mr["fgib"])
            if wins12 is None:
                wins12 = wins["fgib"]
        rows.append((a, cells, wins12))
        print(f"[anchor-ranks] a={a:2d} rank12={cells[0]:.2f} rankCLS={cells[1]:.2f} "
              f"rankREG={cells[2]:.2f} wins={wins12}/12")
    lines = [
        "\\begin{table}[t]",
        "\\caption{Sensitivity of the equal-budget FGIB-A rank to the fixed anchor scale. "
        "Each column reports FGIB-A selected over the same 6-point $\\beta$ grid as every other "
        "bottleneck at a \\emph{single globally fixed} anchor scale $a$, so its search budget "
        "matches the comparison exactly (the $a{=}16$ column is the post-hoc control of "
        "Table~\\ref{tab:main-std}). Mean rank of 7 entries (deterministic backbone + 6 "
        "bottlenecks) over all 12 settings, over the 7 classification settings, and over the 5 "
        "regression settings; ``best'' counts best-or-tied settings out of 12. The main-table "
        "number ($1.75$, full $6\\times9$ grid) is a ceiling only over the full grid, not over "
        "fixed $a$; the range of fixed-$a$ ranks quantifies how much of FGIB-A's edge is anchor "
        "search budget (Section~\\ref{sec:exp}).}",
        "\\label{tab:anchor-ranks}",
        "\\begin{center}\\small",
        "\\begin{tabular}{lccccccccc}",
        "anchor scale $a$ & " + " & ".join(str(a) for a in ANCHORS) + " \\\\ \\hline \\\\[-1.8ex]",
    ]
    for label, idx in [("mean rank (12 settings)", 0),
                       ("mean rank (classification, 7)", 1),
                       ("mean rank (regression, 5)", 2),
                       ("best or tied-best (of 12)", 3)]:
        if idx == 3:
            cells = [str(w) for _, _, w in rows]
        else:
            cells = [f"{c[idx]:.2f}" for _, c, _ in rows]
        lines.append("\\multicolumn{1}{l}{" + label + "} & " + " & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    with open(os.path.join(out_dir, "tab_anchor_ranks.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def selected0(by_setting, setting, model, anchor):
    """该设定下 model 的 val 选择记录（fgib 固定 anchor，其余模型无 anchor 维度）。"""
    cands = [r for r in by_setting[setting] if r["model"] == model
             and (model != "fgib" or r["anchor"] == float(anchor))]
    return select_one(cands)


def table_hanchor(selected, baseline, fgib16, out_dir, results_dir):
    """h 直锚（A=I 固定正交侧头，FGIB-H）七分类设置表（第四轮：canonical 主表）。

    比较 det / FGIB-A(a=16) / CentFGIB(a=16) / FGIB-H(A=I, a=16) /
    FGIB-H(A=I, a=4) 五种形态在全部 7 个分类设置上的验证集选模测试分，
    每种形态同一 6 点 β 预算、固定单一锚点尺度（等预算，非 6×9 网格）。
    a=16 为继承自 z 空间的锚点尺度，a=4 为 h 空间重调的尺度。数据来源：
    - FGIB-H a=16：mnist_mlp=aid/、mnist_cnn/imagenet100_mlp/cora_gnn=hdirect/、
      其余三设置=hdirect_all/（第四轮新跑）；
    - FGIB-H a=4：mnist_mlp 与三新设置=hdirect_all/、其余=hdirect_a48/。
    任一设置缺数据时整表跳过（防分母不一致的 rank 比较）。
    """
    import pathlib
    settings = ["mnist_mlp", "mnist_cnn", "imagenet100_mlp", "imagenet100_cnn",
                "cora_gnn", "imdb_rnn", "agnews_rnn"]
    h16_dir = {"mnist_mlp": "aid", "mnist_cnn": "hdirect", "imagenet100_mlp": "hdirect",
               "imagenet100_cnn": "hdirect_all", "cora_gnn": "hdirect",
               "imdb_rnn": "hdirect_all", "agnews_rnn": "hdirect_all"}
    h4_dir = {"mnist_mlp": "hdirect_all", "mnist_cnn": "hdirect_a48",
              "imagenet100_mlp": "hdirect_a48", "imagenet100_cnn": "hdirect_all",
              "cora_gnn": "hdirect_a48", "imdb_rnn": "hdirect_all",
              "agnews_rnn": "hdirect_all"}

    def load_aid(setting, dname, anchor):
        """从指定消融子目录读取某设置 FGIB-A a-identity 的选模测试分；缺失返回 None。"""
        d = pathlib.Path("tune_results_ablation") / dname
        if not d.is_dir():
            return None
        recs = load(str(d), strict=False)
        sub = [r for r in recs if r["setting"] == setting
               and r["model"] == "fgib" and r["anchor"] == float(anchor)]
        if len(sub) != len(BETAS):
            return None
        return select_one(sub)["test"][metric_key(setting)][0]

    rows, avail = {}, []
    for setting in settings:
        key = metric_key(setting)
        det = baseline[setting]["test"][key][0]
        f16 = fgib16[setting]["test"][key][0]
        cent_recs = load("tune_results_ablation/centfgib_all", strict=False)
        cent_recs = [r for r in cent_recs if r["setting"] == setting
                     and r["model"] == "centfgib" and r["anchor"] == 16.0]
        if len(cent_recs) != len(BETAS):
            continue
        cent = select_one(cent_recs)["test"][key][0]
        h16 = load_aid(setting, h16_dir[setting], 16)
        h4 = load_aid(setting, h4_dir[setting], 4)
        if h16 is None or h4 is None:
            print(f"[tab_hanchor] {setting} 缺 a-identity 数据（h16={h16 is not None}, "
                  f"h4={h4 is not None}），整表跳过")
            return
        rows[setting] = (det, f16, cent, h16, h4)
        avail.append(setting)
    lines = [
        "\\begin{table}[t]",
        "\\caption{The canonical fixed-orthogonal side head across the seven classification "
        "settings. FGIB-A, CentFGIB and FGIB-H differ only in the side-path regularizer --- "
        "trainable side head + KL, trainable side head + plain center loss, and fixed side "
        "head $A{=}I$ (equivalently the anchor matching applied directly to the deployed "
        "representation $h$, Result~\\ref{res:orth}) --- each selected over the same 6-point "
        "$\\beta$ budget at a single fixed anchor scale, on validation. All cells are the "
        "\\emph{validation-selected} configuration; the ``worst deficit'' row is the worst "
        "validation-selected deficit across the seven settings, not the worst across "
        "$\\beta$ --- the full per-$\\beta$ curves and their worst-across-$\\beta$ deficits "
        "are Tables~\\ref{tab:hanchor-sweep-a4} and~\\ref{tab:hanchor-sweep-a16}. $a{=}16$ "
        "is the scale inherited from the $z$-space sweep; $a{=}4$ is a re-tuned scale for "
        "$h$-space, chosen as the representative of the sensitivity sweep "
        "$a\\in\\{4,8,16\\}$ after seeing those results (each cell still selected on "
        "validation; no test score participates). Units as in Table~\\ref{tab:main}. At the "
        "inherited scale the fixed head trails by $\\approx 0.5$ points per setting; at the "
        "re-tuned scale it recovers most of the gap --- the price of the $\\kappa{=}1$ "
        "closure is small. All ranks and win counts below are computed over the same seven "
        "complete rows.}",
        "\\label{tab:hanchor}",
        "\\begin{center}\\small",
        "\\begin{tabular}{llccccc}",
        "Task & Backbone & Det. & FGIB-A ($a{=}16$) & CentFGIB ($a{=}16$) & "
        "FGIB-H ($A{=}I$, $a{=}16$) & FGIB-H ($A{=}I$, $a{=}4$) \\\\ \\hline \\\\[-1.8ex]",
    ]
    ranks = collections.defaultdict(list)
    wins = collections.Counter()
    worst = collections.defaultdict(float)
    for setting in settings:
        task, bb = NAMES[setting]
        vals = list(rows[setting])
        best = max(v for v in vals if v is not None)
        cells = [fmt(v, v is not None and v >= best - TIE_EPS) if v is not None else "--"
                 for v in vals]
        lines.append(f"{task} & {bb} & " + " & ".join(cells) + " \\\\")
        det = vals[0]
        for name, v in zip(["det", "fgib", "centfgib", "h16", "h4"], vals):
            if v is None:
                continue
            ranks[name].append(1 + sum(1 for w in vals if w is not None and w > v + TIE_EPS))
            if v >= best - TIE_EPS:
                wins[name] += 1
            worst[name] = max(worst[name], det - v)
    order = ["det", "fgib", "centfgib", "h16", "h4"]
    n = len(avail)
    lines.append("\\hline \\\\[-1.8ex]")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Mean rank (of {n})}} & " + " & ".join(
        f"{statistics.fmean(ranks[m]):.2f}" for m in order) + " \\\\")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Best or tied-best (of {n})}} & " + " & ".join(
        str(wins[m]) for m in order) + " \\\\")
    lines.append("\\multicolumn{2}{l}{Worst deficit vs.\ Det.} & -- & " + " & ".join(
        f"{fmt(worst[m])}" for m in order[1:]) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    with open(os.path.join(out_dir, "tab_hanchor.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    labels = {"det": "Det.", "fgib": "FGIB-A(a16)", "centfgib": "CentFGIB(a16)",
              "h16": "FGIB-H(a16)", "h4": "FGIB-H(a4)"}
    print("[tab_hanchor] mean rank: " + ", ".join(
        f"{labels[m]}={statistics.fmean(ranks[m]):.2f}" for m in order)
          + f" | wins: " + ", ".join(f"{labels[m]}={wins[m]}/{n}" for m in order))


def table_hanchor_sweep(baseline, out_dir):
    """FGIB-H（A=I）七分类设置的**完整 β 曲线**表（第五轮，审稿人要求：
    报告全网格曲线与跨 β 最坏赤字，而非仅验证选中单点）。

    两张表（a=4 与 a=16 各一），每行一个分类设置：Det + 6 个 β 的测试分
    （5 seeds 平均）+ 跨 β 最坏赤字（max_β Det−acc_β）。数据来自
    aid/hdirect/hdirect_a48/hdirect_all（见 table_hanchor 的目录映射）。
    """
    import pathlib
    settings = ["mnist_mlp", "mnist_cnn", "imagenet100_mlp", "imagenet100_cnn",
                "cora_gnn", "imdb_rnn", "agnews_rnn"]
    d16 = {"mnist_mlp": "aid", "mnist_cnn": "hdirect", "imagenet100_mlp": "hdirect",
           "imagenet100_cnn": "hdirect_all", "cora_gnn": "hdirect",
           "imdb_rnn": "hdirect_all", "agnews_rnn": "hdirect_all"}
    d4 = {"mnist_mlp": "hdirect_all", "mnist_cnn": "hdirect_a48",
          "imagenet100_mlp": "hdirect_a48", "imagenet100_cnn": "hdirect_all",
          "cora_gnn": "hdirect_a48", "imdb_rnn": "hdirect_all",
          "agnews_rnn": "hdirect_all"}
    a_dir = {}
    for s in settings:
        a_dir[(s, 16)] = d16[s]
        a_dir[(s, 4)] = d4[s]

    def per_beta(setting, anchor):
        """该设置该尺度下 {beta: 测试分}；数据缺失返回 None。"""
        d = pathlib.Path("tune_results_ablation") / a_dir[(setting, anchor)]
        if not d.is_dir():
            return None
        recs = load(str(d), strict=False)
        sub = [r for r in recs if r["setting"] == setting and r["model"] == "fgib"
               and r["anchor"] == float(anchor)]
        if len(sub) != len(BETAS):
            return None
        return {r["beta"]: r["test"][metric_key(setting)][0] for r in sub}

    lines = []
    for anchor, label in ((4, "tab:hanchor-sweep-a4"), (16, "tab:hanchor-sweep-a16")):
        lines.append("\\begin{table}[t]")
        lines.append(f"\\caption{{FGIB-H ($A{{=}}I$) full $\\beta$ sweep at $a{{=}}{anchor}$ on the "
                     f"seven classification settings: test accuracy (\\%) per $\\beta$ "
                     f"(5-seed mean), the deterministic backbone for reference, and the "
                     f"\\emph{{worst deficit across the six $\\beta$ values}} (max$_\\beta$ "
                     f"Det $-$ acc$_\\beta$). Validation-selected cells feed "
                     f"Table~\\ref{{tab:hanchor}}; the full curve is what "
                     f"Result~\\ref{{res:endpoint}} predicts over.}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("\\begin{center}\\footnotesize")
        lines.append("\\begin{tabular}{ll" + "c" * (1 + len(BETAS) + 1) + "}")
        header = ["Task", "Backbone", "Det."] + [f"$\\beta={b:g}$" for b in BETAS] + ["Worst"]
        lines.append(" & ".join(header) + " \\\\ \\hline \\\\[-1.8ex]")
        missing = 0
        for setting in settings:
            accs = per_beta(setting, anchor)
            if accs is None:
                missing += 1
                continue
            key = metric_key(setting)
            det = baseline[setting]["test"][key][0]
            vals = [accs[b] for b in BETAS]
            worst = max(det - v for v in vals)
            task, bb = NAMES[setting]
            cells = [task, bb, fmt(det)] + [fmt(v) for v in vals] + [fmt(worst)]
            lines.append(" & ".join(cells) + " \\\\")
        if missing:
            print(f"[tab_hanchor_sweep] a={anchor} 缺 {missing} 个设置，该表跳过")
            return
        lines.append("\\end{tabular}\\end{center}\\end{table}")
        lines.append("")
    with open(os.path.join(out_dir, "tab_hanchor_sweep.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[tab_hanchor_sweep] 已生成（a=4 与 a=16 两张完整曲线表）")


def table_fgibs_mnist(baseline, selected, out_dir):
    """FGIB-S 在 MNIST/MLP 上的主表对照（第五轮：审稿人要求随机部署 mode
    至少在一个视觉任务入主结果表）。FGIB-S 用与 FGIB-A 相同的 6×9 网格、
    验证集选模；对照 Det / VIB / CEB / FGIB-A / FGIB-H(a=4) 的验证选中分。"""
    recs = load("tune_results_fgibs", strict=False)
    sub = [r for r in recs if r["setting"] == "mnist_mlp" and r["model"] == "fgibs"]
    if not sub:
        print("[tab_fgibs_mnist] tune_results_fgibs 数据缺失，跳过")
        return
    best = select_one(sub)
    lines = [
        "\\begin{table}[t]",
        "\\caption{FGIB-S on MNIST/MLP: the stochastic canonical mode selected over the same "
        "$6\\times9$ grid as FGIB-A (validation-selected, 5 seeds), against the deterministic "
        "backbone and the deterministic-side objectives. FGIB-S is evaluated on its deployed "
        "sampled $z$; the $\\mu$-deployment figures are reported in the text "
        "(Section~\\ref{sec:suite}). Units as in Table~\\ref{tab:main}.}",
        "\\label{tab:fgibs-mnist}",
        "\\begin{center}\\footnotesize",
        "\\begin{tabular}{llllll}",
        "Det. & VIB & CEB & FGIB-A & FGIB-H ($a{=}4$) & FGIB-S \\\\ "
        "\\hline \\\\[-1.8ex]",
    ]
    key = "Acc"
    det = baseline["mnist_mlp"]["test"][key][0]
    vib = selected["mnist_mlp"]["vib"]["test"][key][0]
    ceb = selected["mnist_mlp"]["ceb"]["test"][key][0]
    fgib_a = selected["mnist_mlp"]["fgib"]["test"][key][0]
    s = best["test"][key][0]
    h4 = None
    import pathlib
    d = pathlib.Path("tune_results_ablation/hdirect_all")
    if d.is_dir():
        r4 = [r for r in load(str(d), strict=False)
              if r["setting"] == "mnist_mlp" and r["model"] == "fgib" and r["anchor"] == 4.0]
        if len(r4) == len(BETAS):
            h4 = select_one(r4)["test"][key][0]
    cells = [fmt(det), fmt(vib), fmt(ceb), fmt(fgib_a),
             fmt(h4) if h4 is not None else "--", fmt(s)]
    lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    with open(os.path.join(out_dir, "tab_fgibs_mnist.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[tab_fgibs_mnist] Det={100*det:.2f} VIB={100*vib:.2f} CEB={100*ceb:.2f} "
          f"FGIB-A={100*fgib_a:.2f} FGIB-H(a4)={100*h4:.2f} FGIB-S={100*s:.2f} "
          f"(FGIB-S 均值部署由 ceb_suite 报告)")


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
    table_main_reg(selected, baseline, os.path.join(args.out_dir, "tab_main_reg.tex"))
    table_robust(by_setting, selected, fgib16, baseline, os.path.join(args.out_dir, "tab_robust.tex"))
    table_anchor(by_setting, baseline, os.path.join(args.out_dir, "tab_anchor.tex"))
    table_main_std(selected, fgib16, baseline, os.path.join(args.out_dir, "tab_main_std.tex"))
    rank_split(selected, baseline, fgib16, os.path.join(args.out_dir, "rank_split.tex"))
    table_hanchor_sweep(baseline, args.out_dir)
    table_fgibs_mnist(baseline, selected, args.out_dir)
    table_anchor_ranks(by_setting, baseline, args.out_dir)
    table_hanchor(selected, baseline, fgib16, args.out_dir, args.results_dir)
    print(f"{len(recs)} configurations, {sum(r['runs'] for r in recs)} runs, "
          f"{len(by_setting)} task×backbone settings → {args.out_dir}/")


if __name__ == "__main__":
    main()
