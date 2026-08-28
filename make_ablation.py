"""MNIST/MLP 消融实验（审稿人第 1/2/3/4 条）的结果解析与表格生成。

输入：
- tune_results/（主实验：det / vib / ceb / fgib 各 beta，fgib a=16 对照）
- tune_results_ablation/main/（dceb / tafgib / centfgib）
- tune_results_ablation/freezea/、aid/、cosine/（fgib 的 A 约束消融与 cosine 分类器）
- tune_results_ablation/p3_diag/（vib/ceb/fgib 的 --log-variance-stats 诊断）

输出 tables/tab_ablation.tex（各变体在 6 个 beta 上的测试 Acc + 验证集选中列），
并打印 P3 方差测量与 κ(AᵀA) 测量的汇总数字。

用法：python make_ablation.py
"""

import glob
import os
import re

import make_tables as mt

BETAS = mt.BETAS

# (行名, 结果目录, 模型, anchor) —— 缺 anchor 为 None
ROWS = [
    ("Det.", "tune_results", "mlp", None, None),
    ("VIB", "tune_results", "vib", None, None),
    ("CEB", "tune_results", "ceb", None, None),
    ("FGIB ($a{=}16$)", "tune_results", "fgib", 16.0, None),
    ("DCEB (trainable prior)", "tune_results_ablation", "dceb", None, None),
    ("TAFGIB (trainable anchors)", "tune_results_ablation", "tafgib", 16.0, None),
    ("CentFGIB (MSE-to-anchor)", "tune_results_ablation", "centfgib", 16.0, None),
    ("FGIB frozen $A$", "tune_results_ablation/freezea", "fgib", 16.0, None),
    ("FGIB $A{=}I$", "tune_results_ablation/aid", "fgib", 16.0, None),
    ("FGIB cosine", "tune_results_ablation/cosine", "fgib", 16.0, None),
    ("VIB cosine", "tune_results_ablation/cosine", "vib", None, None),
    ("CEB cosine", "tune_results_ablation/cosine", "ceb", None, None),
]

AVG_RE = mt.AVG_RE
RUN_RE = mt.RUN_RE
VAL_RE = mt.VAL_RE


def parse_log(path):
    """返回 dict(beta=, anchor=, val=, test={metric: (mean,std)})，复刻 make_tables.load 的单文件版。"""
    text = open(path, errors="ignore").read()
    vals, cur = [], None
    for line in text.splitlines():
        m = VAL_RE.search(line)
        if m:
            v = float(m.group(1))
            cur = v if cur is None else max(cur, v)
        m = RUN_RE.search(line)
        if m and cur is not None:
            vals.append(cur)
            cur = None
    n_runs = 0
    avg = None
    for avg in mt.AVG_RE.finditer(text):
        n_runs = int(avg.group(1))
    if avg is None or not vals:
        raise RuntimeError(f"{path}: 缺少 val 或 Average 行")
    test = {k: (float(v), float(s))
            for k, v, s in re.findall(r"(\w+) ([\d.\-]+)±([\d.]+)", avg.group(2))}
    return {"val": sum(vals[-n_runs:]) / n_runs, "test": test}


def find_ablation_log(results_dir, model, beta, anchor):
    name = f"mnist_mlp_{model}"
    if beta is not None:
        name += f"_beta_{beta:g}"
    if anchor is not None:
        name += f"_anchor_{anchor:g}"
    path = os.path.join(results_dir, name, "train.log")
    assert os.path.isfile(path), path
    return path


def series_test_acc(results_dir, model, anchor):
    """6 个 beta 上的测试 Acc 列表（按 BETAS 顺序）+ 每配置的 (val, rec)。"""
    out, recs = {}, {}
    for b in BETAS:
        rec = parse_log(find_ablation_log(results_dir, model, b, anchor))
        out[b] = rec["test"]["Acc"][0]
        recs[b] = rec
    return out, recs


def val_select(recs):
    """验证集选模（均值 AUC + 固定配置顺序 tie-break），返回选中 beta。"""
    return max(recs, key=lambda b: (recs[b]["val"], -b))


def main():
    lines = [
        "\\begin{table}[t]",
        "\\caption{Confound ablations and fixed-scale checks on MNIST/MLP. Test accuracy (\\%) "
        "across the $\\beta$ grid for each variant (fixed $a=16$ where an anchor scale applies), "
        "plus the validation-selected configuration. \\emph{DCEB} keeps the deterministic "
        "classifier path of FGIB but trains the conditional prior (CEB's backward encoder); "
        "\\emph{TAFGIB} trains the anchor means instead of freezing them; \\emph{CentFGIB} "
        "replaces the KL with a direct MSE-to-anchor center loss; \\emph{frozen $A$} and "
        "\\emph{$A{=}I$} constrain the side head; \\emph{cosine} rows use the fixed-temperature "
        "cosine classifier of Remark~\\ref{rem:c-scale}(c). Protocol identical to the main "
        "experiments (100 epochs, patience 10, 5 seeds).}",
        "\\label{tab:ablation}",
        "\\begin{center}\\small",
        "\\begin{tabular}{lccccccc}",
        "Variant & $10^{-4}$ & $10^{-3}$ & $10^{-2}$ & $10^{-1}$ & $1$ & $10$ & val-sel. \\\\ \\hline \\\\[-1.8ex]",
    ]

    print("== MNIST/MLP 消融：各变体在 beta 网格上的测试 Acc（val-sel 列 = 验证集选模）==")
    for name, resdir, model, anchor, _ in ROWS:
        if model == "mlp":
            rec = parse_log(os.path.join("tune_results", "mnist_mlp", "train.log"))
            accs = None
            val_sel = rec["test"]["Acc"][0]
            cells = ["--"] * 6 + [f"{100 * val_sel:.2f}"]
            print(f"{name:28s} det = {100 * val_sel:.2f}")
        else:
            accs, recs = series_test_acc(resdir, model, anchor)
            bsel = val_select(recs)
            val_sel = recs[bsel]["test"]["Acc"][0]
            cells = [f"{100 * accs[b]:.2f}" for b in BETAS] + [f"\\textbf{{{100 * val_sel:.2f}}}"]
            print(f"{name:28s} " + " ".join(f"{100 * accs[b]:.2f}" for b in BETAS)
                  + f" | val-sel beta={bsel:g}: {100 * val_sel:.2f}")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    out = os.path.join("tables", "tab_ablation.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"已生成 {out}")

    # P3 方差测量（p3_diag）
    print("\n== P3 方差测量（每 run 最终 epoch Diag 行，5 runs 平均）==")
    for model in ["ceb", "vib"]:
        row = []
        for b in ["0.0001", "0.001", "0.01", "0.1", "1", "10"]:
            path = glob.glob(f"tune_results_ablation/p3_diag/mnist_mlp_{model}_beta_{b}/train.log")
            if not path:
                continue
            diags = extract_diags(path[0])
            def stat(field):
                vals = [float(re.search(field + r"=([\d.\-]+)", d).group(1)) for d in diags]
                return sum(vals) / len(vals)
            row.append(f"beta={b}: q={stat('logvar_q_mean'):+.2f}/clamp={stat('logvar_q_clamp'):.3f}"
                       + (f" p={stat('logvar_p_mean'):+.2f}" if model == "ceb" else ""))
        print(f"  {model}: " + " | ".join(row))

    # κ(AᵀA) 测量（p3_diag 的 fgib a=16）
    print("\n== FGIB κ(AᵀA)（log10，每 run 最终 epoch，5 runs 平均）==")
    for b in ["0.0001", "0.001", "0.01", "0.1", "1", "10"]:
        path = glob.glob(f"tune_results_ablation/p3_diag/mnist_mlp_fgib_beta_{b}_anchor_16/train.log")
        if not path:
            continue
        diags = extract_diags(path[0])
        kappas = [float(re.search(r"kappa_log10=([\d.\-]+)", d).group(1)) for d in diags if "kappa_log10=" in d]
        traj = []
        # 第一个 run 的逐 epoch 轨迹
        run, last = 0, 0
        for line in open(path[0]).read().splitlines():
            if "===== Run" in line:
                run += 1
            m = re.search(r"Epoch\s+(\d+)/", line)
            if m:
                last = int(m.group(1))
            m = re.search(r"kappa_log10=([\d.\-]+)", line)
            if run == 1 and m:
                traj.append((last, float(m.group(1))))
        print(f"  beta={b}: 最终 κ_log10 均值={sum(kappas)/len(kappas):.2f} "
              f"| run1 轨迹: " + " ".join(f"e{e}:{v:.1f}" for e, v in traj[::2]))


def extract_diags(path):
    """每 run 的最后一行 Diag。"""
    diags, cur = [], None
    for line in open(path, errors="ignore").read().splitlines():
        if "===== Run" in line:
            if cur:
                diags.append(cur)
            cur = None
        if "Diag " in line:
            cur = line
    if cur:
        diags.append(cur)
    return diags


if __name__ == "__main__":
    main()
