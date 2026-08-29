"""MNIST/MLP 消融实验（审稿人第 1/2/3/4 条）的结果解析与表格生成。

输入：
- tune_results/（主实验：det / vib / ceb / fgib 各 beta，fgib a=16 对照）
- tune_results_ablation/main/（dceb / tafgib / centfgib）
- tune_results_ablation/freezea/、aid/、cosine/（fgib 的 A 约束消融与 cosine 分类器）
- tune_results_ablation/p3_diag_full/（vib/ceb/fgib/tafgib 的全验证集 batch 聚合
  --log-variance-stats 诊断，修复旧版只读第一个 batch 的问题）
- tune_results_ablation/longtrain/（CEB/FGIB 400 epochs 长训练，P3 早停伪影检验）
- tune_results_ablation/agedb_disjoint/（AgeDB subject-disjoint 身份隔离对照）

输出 tables/tab_ablation.tex（各变体在 6 个 beta 上的测试 Acc + 验证集选中列）、
tables/tab_agedb_disjoint.tex，并打印 P3 方差测量、长训练与 κ(AᵀA) 测量的
汇总数字。

用法：python make_ablation.py
"""

import glob
import os
import re
import statistics

import make_tables as mt

BETAS = mt.BETAS

# 全验证集聚合的 P3 诊断目录（旧版 p3_diag/ 只读第一个 batch，仅留档）
P3_DIR = "tune_results_ablation/p3_diag_full"

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
    ("ETF (frozen classifier)", "tune_results_ablation/etf", "etf", None, None),
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
        if model in ("mlp", "etf"):
            # 基线无 beta 维度：mlp 在主目录，etf 在 ablation/etf（{ds}_{model} 命名）
            base_name = "mnist_mlp" if model == "mlp" else "mnist_etf"
            base_dir = "tune_results" if model == "mlp" else resdir
            path = os.path.join(base_dir, base_name, "train.log")
            if not os.path.isfile(path):
                print(f"{name:28s} 未完成（{path}），跳过行")
                continue
            rec = parse_log(path)
            accs = None
            val_sel = rec["test"]["Acc"][0]
            cells = ["--"] * 6 + [f"{100 * val_sel:.2f}"]
            print(f"{name:28s} single = {100 * val_sel:.2f}")
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

    # P3 方差测量（p3_diag_full：整个验证集聚合）
    print("\n== P3 方差测量（全验证集聚合，每 run 最终 epoch Diag 行，5 runs 平均）==")
    for model in ["ceb", "vib"]:
        row = []
        for b in ["0.0001", "0.001", "0.01", "0.1", "1", "10"]:
            path = glob.glob(f"{P3_DIR}/mnist_mlp_{model}_beta_{b}/train.log")
            if not path:
                continue
            diags = extract_diags(path[0])
            def stat(field):
                vals = [float(re.search(field + r"=([\d.\-]+)", d).group(1)) for d in diags]
                return sum(vals) / len(vals)
            row.append(f"beta={b}: q={stat('logvar_q_mean'):+.2f}/clamp={stat('logvar_q_clamp'):.3f}"
                       + (f" p={stat('logvar_p_mean'):+.2f}" if model == "ceb" else ""))
        print(f"  {model}: " + " | ".join(row))

    # κ(AᵀA) 测量（p3_diag_full 的 fgib a=16）
    print("\n== FGIB κ(AᵀA)（log10，每 run 最终 epoch，5 runs 平均）==")
    for b in ["0.0001", "0.001", "0.01", "0.1", "1", "10"]:
        path = glob.glob(f"{P3_DIR}/mnist_mlp_fgib_beta_{b}_anchor_16/train.log")
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


def centfgib_all_table():
    """CentFGIB 全 12 设定对照（tune_results_ablation/centfgib_all/，a=16、6 点 β 网格）。

    每设定报告 Det / FGIB(a=16) / CentFGIB(a=16) 的验证集选模测试分，并把
    CentFGIB 与主实验七目标 + fgib16 放在一起算 mean rank（9 个条目）。
    生成 tables/tab_centfgib.tex 并打印 rank 汇总。
    """
    import make_stats as ms

    lines = [
        "\\begin{table}[t]",
        "\\caption{CentFGIB across all 12 settings: does the geometry alone --- the same "
        "orthonormal anchors as FGIB but with the KL replaced by a plain "
        "$\\tfrac12\\|\\mu - a v_y\\|^2$ center loss --- reproduce FGIB's behavior beyond "
        "MNIST? Validation-selected configuration at $a=16$ (the same 6-point $\\beta$ "
        "grid and budget as the FGIB equal-budget control of Table~\\ref{tab:main-std}), "
        "5 seeds, protocol identical to the main sweep. Units as in Table~\\ref{tab:main}.}",
        "\\label{tab:centfgib}",
        "\\begin{center}\\small",
        "\\begin{tabular}{llccc}",
        "Task & Backbone & Det. & FGIB ($a{=}16$) & CentFGIB ($a{=}16$) \\\\ \\hline \\\\[-1.8ex]",
    ]

    recs = mt.load("tune_results")
    by_setting, selected, baseline = mt.build_index(recs)
    fgib16 = mt.build_fixed_anchor_index(by_setting)

    rows = {}
    for setting in mt.CLS + mt.REG:
        key = mt.metric_key(setting)
        b = baseline[setting]
        det = ms.per_run_metrics(ms.find_log("tune_results", setting, b["model"], b["beta"], b["anchor"]))
        det_val = statistics.mean(r[key] for r in det)
        f16 = fgib16[setting]
        # CentFGIB：a=16、6 个 beta，验证集选模
        recs_c = {}
        for beta in BETAS:
            path = os.path.join("tune_results_ablation", "centfgib_all",
                                f"{setting}_centfgib_beta_{beta:g}_anchor_16", "train.log")
            recs_c[beta] = parse_log(path)
        bsel = val_select(recs_c)
        rows[setting] = {
            "det": det_val,
            "fgib16": f16["test"][key][0],
            "cent": recs_c[bsel]["test"][key][0],
            "cent_beta": bsel,
        }
        lines.append(
            f"{mt.NAMES[setting][0]} & {mt.NAMES[setting][1]} & "
            f"{100 * det_val:.2f} & {100 * f16['test'][key][0]:.2f} & "
            f"\\textbf{{{100 * recs_c[bsel]['test'][key][0]:.2f}}} \\\\"
        )
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(os.path.join("tables", "tab_centfgib.tex"), "w").write("\n".join(lines) + "\n")
    print("已生成 tables/tab_centfgib.tex")

    # 9 条目 rank（det + 6 瓶颈 + fgib 全网格 + fgib16 + centfgib16）
    vals = {m: [selected[s][m]["test"][mt.metric_key(s)][0] for s in mt.CLS + mt.REG]
            for m in mt.BOTTLENECKS}
    vals["det"] = [baseline[s]["test"][mt.metric_key(s)][0] for s in mt.CLS + mt.REG]
    vals["fgib16"] = [fgib16[s]["test"][mt.metric_key(s)][0] for s in mt.CLS + mt.REG]
    vals["centfgib16"] = [rows[s]["cent"] for s in mt.CLS + mt.REG]
    mr, wins = mt.rank_stats(vals)
    for name in vals:
        print(f"[9 条目 rank] {name:12s} mean rank = {mr[name]:.2f}, best-or-tied = {wins[name]}/12")


def agedb_disjoint_table():
    """AgeDB subject-disjoint 对照（tune_results_ablation/agedb_disjoint/，审稿人第 4 条）。

    身份隔离划分下比较 Det / FGIB(a=16) / CentFGIB(a=16) / FGIB-H(A=I, a=16)
    的验证集选模测试 R²（6 点 β 网格，5 seeds，CNN 骨干）。
    生成 tables/tab_agedb_disjoint.tex 并打印四条目 rank。
    """
    base = "tune_results_ablation/agedb_disjoint"
    need = (["agedb_cnn"] + [f"agedb_cnn_{m}_beta_{b:g}_anchor_16"
                             for m in ("fgib", "centfgib") for b in BETAS]
            + [f"agedb_cnn_fgib_beta_{b:g}_anchor_16" for b in BETAS])
    done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(
        os.path.join(base, "**", "train.log"), recursive=True)}
    missing = [n for n in need if n not in done]
    if missing:
        print(f"agedb_disjoint 未完成（缺 {len(missing)} 个配置），跳过表格")
        return
    det = parse_log(os.path.join(base, "agedb_cnn", "train.log"))["test"]["R2"][0]
    rows = {}

    def select(dirname, prefix):
        recs = {}
        for b in BETAS:
            path = os.path.join(dirname, f"{prefix}_beta_{b:g}_anchor_16", "train.log")
            recs[b] = parse_log(path)
        bsel = val_select(recs)
        return recs[bsel]["test"]["R2"][0], bsel

    f16, f16_b = select(base, "agedb_cnn_fgib")
    cent, cent_b = select(base, "agedb_cnn_centfgib")
    aid, aid_b = select(os.path.join(base, "aid"), "agedb_cnn_fgib")
    vals = {"Det.": det, "FGIB ($a{=}16$)": f16, "CentFGIB ($a{=}16$)": cent,
            "FGIB-H ($A{=}I$, $a{=}16$)": aid}
    lines = [
        "\\begin{table}[t]",
        "\\caption{AgeDB under a subject-disjoint split. Identities are grouped so that no "
        "subject appears on both sides of any split boundary (60/20/20, seed 42); this removes "
        "the identity leakage of the image-level split used elsewhere in the paper, at the cost "
        "of comparability with other AgeDB literature. Validation-selected $\\beta$ at a fixed "
        "$a=16$ (6-point grid), 5 seeds, CNN backbone, $R^2\\times100$.}",
        "\\label{tab:agedb-disjoint}",
        "\\begin{center}\\small",
        "\\begin{tabular}{lcccc}",
        "AgeDB (subject-disjoint) & Det. & FGIB ($a{=}16$) & CentFGIB ($a{=}16$) & "
        "FGIB-H ($A{=}I$, $a{=}16$) \\\\ \\hline \\\\[-1.8ex]",
    ]
    best = max(vals.values())
    lines.append("$R^2\\times100$ (val-sel. $\\beta$) & "
                 + " & ".join(("\\textbf{" + f"{100 * v:.2f}" + "}") if v >= best - mt.TIE_EPS
                              else f"{100 * v:.2f}" for v in vals.values()) + " \\\\")
    lines.append("selected $\\beta$ & -- & " + f"{f16_b:g} & {cent_b:g} & {aid_b:g} \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    open(os.path.join("tables", "tab_agedb_disjoint.tex"), "w").write("\n".join(lines) + "\n")
    print("已生成 tables/tab_agedb_disjoint.tex | " +
          " ".join(f"{k} = {100 * v:.2f}" for k, v in vals.items()))


def longtrain_summary():
    """CEB/FGIB 400-epoch 长训练汇总（tune_results_ablation/longtrain/，P3 早停伪影检验）。

    打印每 run 最终 epoch 的后验（与 CEB 先验）logvar 均值与 clamp 占比，
    并把 run 1 的逐 epoch 轨迹导出 tables/longtrain_traj.json 供
    make_figures.py 画图（x=epoch，y=mean log σ²）。
    """
    import json
    base = "tune_results_ablation/longtrain"
    print("\n== CEB/FGIB 长训练（400 epochs，patience 400）最终 epoch logvar（5 runs 平均）==")
    traj_out = {}
    for name, sub in [("ceb_beta_1", "mnist_mlp_ceb_beta_1"),
                      ("ceb_beta_10", "mnist_mlp_ceb_beta_10"),
                      ("fgib_beta_1", "mnist_mlp_fgib_beta_1_anchor_16")]:
        path = glob.glob(os.path.join(base, sub, "train.log"))
        if not path:
            print(f"  {name}: 未完成")
            continue
        diags = extract_diags(path[0])
        def stat(field):
            vals = [float(re.search(field + r"=([\d.\-]+)", d).group(1))
                    for d in diags if field + "=" in d]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"  {name}: q={stat('logvar_q_mean'):+.4f} clamp={stat('logvar_q_clamp'):.4f}"
              + (f" p={stat('logvar_p_mean'):+.4f}" if "ceb" in name else ""))
        # run 1 逐 epoch 轨迹（epoch, logvar_q_mean[, logvar_p_mean]）
        traj, run, last, cur = [], 0, 0, None
        for line in open(path[0]).read().splitlines():
            if "===== Run" in line:
                run += 1
                if run > 1:
                    break
            m = re.search(r"Epoch\s+(\d+)/", line)
            if m:
                last = int(m.group(1))
            if "Diag " in line:
                cur = line
                q = re.search(r"logvar_q_mean=([\d.\-]+)", cur)
                if q and (not traj or traj[-1][0] != last):
                    p = re.search(r"logvar_p_mean=([\d.\-]+)", cur)
                    traj.append((last, float(q.group(1)),
                                 float(p.group(1)) if p else None))
        traj_out[name] = traj
        print(f"  {name} run1 轨迹端点: " + " ".join(
            f"e{e}:{q:+.3f}" for e, q, _ in traj if e in (1, 25, 50, 100, 200, 300, 400)))
    with open(os.path.join("tables", "longtrain_traj.json"), "w") as f:
        json.dump(traj_out, f)
    print("已导出 tables/longtrain_traj.json")


if __name__ == "__main__":
    main()
    # CentFGIB 全设定表：仅在 12 设定 × 6 β 全部完成后生成
    expected = len(mt.CLS + mt.REG) * len(BETAS)
    n_done = len(glob.glob(os.path.join("tune_results_ablation", "centfgib_all",
                                        "*_centfgib_beta_*_anchor_16", "train.log")))
    if n_done >= expected:
        centfgib_all_table()
    else:
        print(f"centfgib_all 未完成（{n_done}/{expected} 个配置），跳过全设定表")
    agedb_disjoint_table()
    longtrain_summary()
