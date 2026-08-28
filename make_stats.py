"""审稿人要求的统计补充（第 7 条）：按 seed 配对的差值、95% 置信区间与 Wilcoxon 检验。

从 tune_results/ 的 train.log 解析每个 run 的测试指标（seed 0..4 跨方法配对），
对每个 任务×骨干 设定：
1. 取各方法**验证集选中**配置（复用 make_tables 的选模逻辑）的逐 run 测试分，
   与同 seed 的确定性基线做配对差值，bootstrap 95% CI（10k 次重采样）；
2. 把 12 个设定的配对均值差汇总为 Wilcoxon signed-rank 检验（vs 0），
   以及 FGIB 与其他方法的逐对 Wilcoxon。
输出 tables/stats.tex（LaTeX 表）并打印论文正文用数字。

用法：python make_stats.py [--results-dir tune_results] [--out tables/stats.tex]
"""

import argparse
import os
import re

import numpy as np
from scipy import stats

import make_tables as mt

RUN_TEST_RE = re.compile(
    r"Run \d+/\d+ \| Test \(best model @ Epoch \d+\) \| (.*)$"
)
PAIR_RE = re.compile(r"(\w+) ([\d.\-]+)")


def per_run_metrics(log_path):
    """解析日志逐 run 的测试指标，返回 {run_index: {metric: value}}。

    与 make_tables.load 相同：以 Average 行声明的 N 为准取最后 N 个 run，
    规避残留的被作废 run 行（imdb 等 OOM 重跑的日志）。
    """
    text = open(log_path, errors="ignore").read()
    n_runs = 0
    for m in mt.AVG_RE.finditer(text):
        n_runs = int(m.group(1))
    runs = []
    for line in text.splitlines():
        m = RUN_TEST_RE.search(line)
        if m:
            runs.append(dict((k, float(v)) for k, v in PAIR_RE.findall(m.group(1))))
    runs = runs[-n_runs:]
    assert len(runs) == n_runs, f"{log_path}: 期望 {n_runs} 个 run 行，实际 {len(runs)}"
    return runs


def find_log(results_dir, setting, model, beta, anchor):
    """按 make_tables 的目录命名规则找 train.log（基线目录名 = setting 本身）。"""
    if model in ("mlp", "cnn", "gcn", "rnn"):
        ds = next(d for d in mt.DATASETS if setting.startswith(d))
        name = f"{ds}_{model}"  # 基线目录按 {dataset}_{model} 命名（cora_gcn 等）
    else:
        name = f"{setting}_{model}"
        if beta is not None:
            name += f"_beta_{beta:g}"
        if anchor is not None:
            name += f"_anchor_{anchor:g}"
    path = os.path.join(results_dir, name, "train.log")
    assert os.path.isfile(path), path
    return path


def paired_diff_ci(method_runs, base_runs, key):
    """同 seed 配对差值 (method - baseline) 的均值与 bootstrap 95% CI。"""
    assert len(method_runs) == len(base_runs) == 5
    d = np.array([m[key] - b[key] for m, b in zip(method_runs, base_runs)])
    rng = np.random.default_rng(0)
    boots = np.array([
        rng.choice(d, size=len(d), replace=True).mean() for _ in range(10_000)
    ])
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="tune_results")
    ap.add_argument("--out", default=os.path.join("tables", "stats.tex"))
    args = ap.parse_args()

    recs = mt.load(args.results_dir)
    by_setting, selected, baseline = mt.build_index(recs)
    fgib16 = mt.build_fixed_anchor_index(by_setting)

    settings = mt.CLS + mt.REG
    names = ["det"] + mt.BOTTLENECKS
    # {method: {setting: (mean_diff, ci_lo, ci_hi)}}，单位为百分点
    diffs = {m: {} for m in mt.BOTTLENECKS}
    fgib16_diffs = {}
    base_runs_cache = {}

    for setting in settings:
        key = mt.metric_key(setting)
        b = baseline[setting]
        base_runs_cache[setting] = per_run_metrics(
            find_log(args.results_dir, setting, b["model"], b["beta"], b["anchor"])
        )
        for m in mt.BOTTLENECKS:
            r = selected[setting][m]
            runs = per_run_metrics(
                find_log(args.results_dir, setting, m, r["beta"], r["anchor"])
            )
            diffs[m][setting] = paired_diff_ci(runs, base_runs_cache[setting], key)
        r16 = fgib16[setting]
        runs16 = per_run_metrics(
            find_log(args.results_dir, setting, "fgib", r16["beta"], r16["anchor"])
        )
        fgib16_diffs[setting] = paired_diff_ci(runs16, base_runs_cache[setting], key)

    # 设定级配对均值差（12 个），Wilcoxon signed-rank vs 0（打印为百分点）
    print("== 各方法 vs 确定性基线：12 设定配对均值差（百分点）的 Wilcoxon ==")
    wil = {}
    for m in mt.BOTTLENECKS:
        ds = np.array([100 * diffs[m][s][0] for s in settings])
        p = stats.wilcoxon(ds).pvalue if not np.allclose(ds, 0) else 1.0
        wil[m] = p
        print(f"{m:7s} mean={ds.mean():+.2f} median={np.median(ds):+.2f} "
              f"min={ds.min():+.2f} max={ds.max():+.2f} p={p:.4f}")
    d16 = np.array([100 * fgib16_diffs[s][0] for s in settings])
    p16 = stats.wilcoxon(d16).pvalue if not np.allclose(d16, 0) else 1.0
    print(f"fgib16  mean={d16.mean():+.2f} median={np.median(d16):+.2f} p={p16:.4f}")

    print("== FGIB 与其他方法的逐对 Wilcoxon（12 设定配对均值差，百分点）==")
    for m in mt.BOTTLENECKS:
        if m == "fgib":
            continue
        ds = np.array([100 * (diffs["fgib"][s][0] - diffs[m][s][0]) for s in settings])
        p = stats.wilcoxon(ds).pvalue if not np.allclose(ds, 0) else 1.0
        print(f"fgib vs {m:7s} mean={ds.mean():+.2f} p={p:.4f}")

    # LaTeX 表：每设定一行，各瓶颈 vs 基线的配对均值差 [95% CI]
    lines = [
        "\\begin{table}[t]",
        "\\caption{Paired per-seed differences against the deterministic backbone, at each "
        "objective's validation-selected configuration (Table~\\ref{tab:main}), with bootstrap "
        "95\\% confidence intervals over the 5 paired seeds (10{,}000 resamples). Units are "
        "percentage points (accuracy for classification, $R^2\\times100$ for regression); "
        "positive means the objective beats the backbone at that setting. The last column is the "
        "equal-budget FGIB control of Table~\\ref{tab:main-std}. Wilcoxon signed-rank tests over "
        "the 12 setting-level paired means against zero are reported in the text.}",
        "\\label{tab:paired}",
        "\\begin{center}\\small",
        "\\begin{tabular}{llccccccc}",
        "Task & Backbone & VIB & SVIB & NIB & CEB & DVCCA & FGIB & FGIB ($a{=}16$) \\\\ \\hline \\\\[-1.8ex]",
    ]
    for group, label in [(mt.CLS, "\\emph{Classification (Acc \\%)}"),
                         (mt.REG, "\\emph{Regression ($R^2\\times100$)}")]:
        lines.append("\\multicolumn{9}{l}{" + label + "} \\\\[0.3ex]")
        for setting in group:
            task, bb = mt.NAMES[setting]
            cells = []
            for m in mt.BOTTLENECKS:
                mu, lo, hi = diffs[m][setting]
                cells.append(f"{100 * mu:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}]")
            mu, lo, hi = fgib16_diffs[setting]
            cells.append(f"{100 * mu:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}]")
            lines.append(f"{task} & {bb} & " + " & ".join(cells) + " \\\\")
        lines.append("\\\\[-1.5ex]")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"已生成 {args.out}")


if __name__ == "__main__":
    main()
