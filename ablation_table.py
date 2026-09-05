"""消融试验结果表（审稿人混杂因素分解）：解析 output/ablation 各配置 train.log
的逐 run Test 行，生成：

1. 主表：每变体 × 3 设置（MNIST / ImageNet-100 / Housing）的最优配置
   （按跨 seed 平均测试指标选择，与 tune/rebuild 口径一致）与 mean±std；
2. 配对 bootstrap CI 表（同 seed 配对、B=10000、固定 seed 42，与
   main_result_CI 同口径）：因素归因阶梯
   - 读出：GPB − GPB-L（分类）、EPB − EPB-L（回归）、CEB-energy − CEB；
   - 几何：OPB-FS − CEB-energy（分类）、EPB-FS − CEB-tied（回归）；
   - 尺度：GPB − OPB-FS（分类）、EPB − EPB-FS（回归）；
   - IB 增量：GPB − NCM-ortho（分类，注明 NCM-ortho 帧固定、OPB 帧可训练，
     该对照附帧可训练性混杂）；
   - 原型结构：NCM-ortho − NCM-learn、NCM-learn − Base。

输出：output/ablation/ablation_table.csv（长表）+ 打印 LaTeX 两表。

用法：
    python ablation_table.py
"""

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "ablation"

CLS_LINE = re.compile(
    r"Run (\d+)/\d+ \| Test \(best model @ Epoch \d+\) \| "
    r"Loss ([\d.eE+-]+) Acc ([\d.eE+-]+) AUC ([\d.eE+-]+) "
    r"Pre ([\d.eE+-]+) Rec ([\d.eE+-]+)"
)
REG_LINE = re.compile(
    r"Run (\d+)/\d+ \| Test \(best model @ Epoch \d+\) \| "
    r"Loss ([\d.eE+-]+) MAE ([\d.eE+-]+) R2 ([\d.eE+-]+)"
)
DIR_RE = re.compile(
    r"^(?P<ds>\w+)_mlp_(?P<model>\S+?)(?:_beta_(?P<beta>[\d.eE+-]+))?"
    r"(?:_anchor_(?P<anchor>[\d.eE+-]+))?(?:_eg|_th)?$"
)

# 显示名（_eg/_th 后缀与 model 合成键；housing 的 opb 系显示为 EPB）
def display(ds, model, eg, th):
    if model == "mlp":
        return "Base"
    if model == "ncm":
        return "NCM-learn"
    if model == "ncmo":
        return "NCM-ortho"
    if model == "ceb":
        return "CEB"
    if model == "ceb-energy":
        return "CEB-energy"
    if model == "ceb-tied":
        return "CEB-tied"
    if model == "opb":
        prefix = "EPB" if ds == "california" else "GPB"
        if eg or th:
            return prefix
        return prefix + "-L"
    if model == "opb-free-scale":
        return "EPB-FS" if ds == "california" else "OPB-FS"
    return model


def parse_dir(d):
    """目录名 → (dataset, model, beta, anchor, eg, th)。"""
    m = DIR_RE.match(d.name)
    if not m:
        return None
    eg = d.name.endswith("_eg")
    th = d.name.endswith("_th")
    return (m.group("ds"), m.group("model"),
            float(m.group("beta")) if m.group("beta") else None,
            float(m.group("anchor")) if m.group("anchor") else None, eg, th)


def load_runs():
    """→ {(dataset, 显示键, beta, anchor): [per-run metric]}。

    日志为追加模式，同一 run 号取最后一次出现（补跑语义）。"""
    runs = defaultdict(list)
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        info = parse_dir(d)
        if info is None:
            continue
        ds, model, beta, anchor, eg, th = info
        log = d / "train.log"
        if not log.exists():
            continue
        best = {}  # run -> metric（分类 Acc / 回归 R2；同一 run 号取最后一次出现）
        for line in log.read_text(encoding="utf-8").splitlines():
            mc = CLS_LINE.search(line)
            if mc:
                best[int(mc.group(1))] = float(mc.group(3))  # Acc
                continue
            mr = REG_LINE.search(line)
            if mr:
                best[int(mr.group(1))] = float(mr.group(4))  # R2（group(3) 是 MAE）
        if not best:
            continue
        runs[(ds, display(ds, model, eg, th), beta, anchor)] = list(best.values())
    return runs


def best_config(runs):
    """每 (dataset, 变体) 按跨 seed 平均指标选最优配置。"""
    best = {}
    for (ds, name, beta, anchor), vals in runs.items():
        mean = sum(vals) / len(vals)
        if (ds, name) not in best or mean > best[(ds, name)][0]:
            best[(ds, name)] = (mean, beta, anchor, vals)
    return best


def mean_std(vals):
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return mean, std


def bootstrap_ci(diffs, b=10000, seed=42):
    """配对差值的 bootstrap 百分位区间（与 main_result_CI 同口径）。"""
    import random
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(b):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * b)], means[int(0.975 * b)]


def paired_row(runs, best, ds, a, b_name):
    """变体 a − b_name 的配对差值与 95% CI；缺失返回 None。"""
    ca, cb = best.get((ds, a)), best.get((ds, b_name))
    if ca is None or cb is None:
        return None
    va, vb = ca[3], cb[3]
    if len(va) != len(vb):
        return None
    diffs = [x - y for x, y in zip(va, vb)]
    mean = sum(diffs) / len(diffs)
    lo, hi = bootstrap_ci(diffs)
    return mean, lo, hi


def main():
    runs = load_runs()
    best = best_config(runs)
    settings = [("mnist", "MNIST"), ("imagenet100", "ImageNet-100"), ("california", "Housing")]
    is_reg = {"mnist": False, "imagenet100": False, "california": True}

    # 1. 主表
    print("% === 消融主表（最优配置，mean±std over 5 seeds）===")
    print(r"\begin{tabular}{l c c c}")
    print(r"\hline")
    print(r"variant & MNIST (Acc \%) & ImageNet-100 (Acc \%) & Housing ($R^2$) \\")
    print(r"\hline")
    variants = ["Base", "NCM-learn", "NCM-ortho", "CEB", "CEB-energy", "GPB-L",
                "GPB", "OPB-FS", "CEB-tied", "EPB-L", "EPB", "EPB-FS"]
    for name in variants:
        cells = []
        for ds, _ in settings:
            entry = best.get((ds, name))
            if entry is None:
                cells.append("--")
            else:
                mean, beta, anchor, vals = entry
                _, std = mean_std(vals)
                cfg = ""
                if beta is not None:
                    cfg = f"$\\beta={beta:g}$"
                if anchor is not None:
                    cfg += (", " if cfg else "") + f"$a={anchor:g}$"
                cfg = f" ({cfg})" if cfg else ""
                if is_reg[ds]:
                    cells.append(f"${mean:.3f}_{{\\pm {std:.3f}}}${cfg}")
                else:
                    cells.append(f"${mean * 100:.1f}_{{\\pm {std * 100:.1f}}}$" + cfg)
        print(name + " & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")

    # 2. 配对 CI 表
    print("\n% === 因素归因配对 CI（GPB − 对照，95% bootstrap，正值为 GPB 占优）===")
    print(r"\begin{tabular}{l l c c c}")
    print(r"\hline")
    print(r"factor & comparison & MNIST & ImageNet-100 & Housing \\")
    print(r"\hline")
    rows_ci = [
        ("readout (IB 变体)", "GPB − GPB-L", "mnist", "GPB", "GPB-L"),
        ("readout (IB 变体)", "GPB − GPB-L", "imagenet100", "GPB", "GPB-L"),
        ("readout (EPB)", "EPB − EPB-L", "california", "EPB", "EPB-L"),
        ("readout (自由几何)", "CEB-energy − CEB", "mnist", "CEB-energy", "CEB"),
        ("readout (自由几何)", "CEB-energy − CEB", "imagenet100", "CEB-energy", "CEB"),
        ("geometry", "OPB-FS − CEB-energy", "mnist", "OPB-FS", "CEB-energy"),
        ("geometry", "OPB-FS − CEB-energy", "imagenet100", "OPB-FS", "CEB-energy"),
        ("geometry (EPB)", "EPB-FS − CEB-tied", "california", "EPB-FS", "CEB-tied"),
        ("fixed scale", "GPB − OPB-FS", "mnist", "GPB", "OPB-FS"),
        ("fixed scale", "GPB − OPB-FS", "imagenet100", "GPB", "OPB-FS"),
        ("fixed scale (EPB)", "EPB − EPB-FS", "california", "EPB", "EPB-FS"),
        ("IB 增量*", "GPB − NCM-ortho", "mnist", "GPB", "NCM-ortho"),
        ("IB 增量*", "GPB − NCM-ortho", "imagenet100", "GPB", "NCM-ortho"),
        ("原型结构", "NCM-ortho − NCM-learn", "mnist", "NCM-ortho", "NCM-learn"),
        ("原型结构", "NCM-ortho − NCM-learn", "imagenet100", "NCM-ortho", "NCM-learn"),
        ("原型读出价值", "NCM-learn − Base", "mnist", "NCM-learn", "Base"),
        ("原型读出价值", "NCM-learn − Base", "imagenet100", "NCM-learn", "Base"),
    ]
    last_factor = None
    for factor, label, ds, a, b_name in rows_ci:
        res = paired_row(runs, best, ds, a, b_name)
        factor_cell = factor if factor != last_factor else ""
        last_factor = factor
        if res is None:
            print(f"{factor_cell} & {label} & -- & -- & -- \\\\")
            continue
        mean, lo, hi = res
        scale = 100 if not is_reg[ds] else 1
        cells = {ds: f"${mean*scale:+.2f}$ [$_{{{lo*scale:.2f}}}$, $_{{{hi*scale:.2f}}}$]"}
        row = f"{factor_cell} & {label} & "
        row += " & ".join(cells.get(s[0], "--") for s in settings) + r" \\"
        print(row)
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\multicolumn{4}{l}{* NCM-ortho 的原型固定为 $a\,e_k$、GPB 的帧可训练，该对照附带帧可训练性混杂。}")

    # 3. 长表 CSV
    csv_rows = []
    for (ds, name, beta, anchor), vals in sorted(runs.items()):
        mean, std = mean_std(vals)
        csv_rows.append([ds, name, beta, anchor, mean, std, len(vals)])
    with open(OUT / "ablation_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "beta", "anchor", "metric_mean", "metric_std", "n_runs"])
        w.writerows(csv_rows)
    print(f"\n长表已保存：{OUT / 'ablation_table.csv'}（{len(csv_rows)} 行）")


if __name__ == "__main__":
    main()
