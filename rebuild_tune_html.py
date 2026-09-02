"""从训练结果重建调参 HTML 结果表；--rerun 时先自动补跑缺失/失败的组合。

默认行为（不重新训练）：扫描结果目录、解析每个日志中**最后一条**汇总行
（train.py 的 FileHandler 为追加模式，补跑后日志尾部才是最新结果），并用与
tune.py 完全相同的逻辑重建 HTML（表格后附最优精度汇总表——每模型一行、
默认指标最优值及其 beta/anchor，再附一张 ceb/opb 的压缩-精度对比曲线：
Acc/R² 随 β 变化、x 为 log10(β)，ceb 画 1 条线、opb 画 anchor=1/6/12
三条线；opbl 结果在曲线中统一按 opb 显示）；缺失/失败的组合打印可直接执行的
train.py 命令。

--rerun 模式：自动补跑缺失/失败的组合（复用 tune.py 的每卡独立队列并行、每进程
OpenMP 线程限制与进度显示），全部完成后重新扫描并重建 HTML；仍失败的组合打印命令。

用法：
    python rebuild_tune_html.py --task mnist --backbone mlp --model vib ceb fgib \\
        --beta 1e-4 1e-3 1e-2 --anchor-scale 1 2 4 8 \\
        --results-dir tune_results --runs 5 --epochs 100
    python rebuild_tune_html.py ... --rerun --parallel 4   # 自动补跑后重建
参数与 tune.py 一致（复用其 build_tune_parser），--runs/--epochs 仅用于 HTML 元信息。
"""

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from train import get_dataset_name
from tune import (
    BASELINES,
    ProgressTracker,
    build_train_cmd,
    build_tune_parser,
    combo_name,
    cpu_count,
    detect_gpus,
    fmt_duration,
    gen_html,
    metric_mean,
    model_prefix,
    run_combo,
)
from utils import curve_card, curve_css, curve_script, plot_bounds

ROOT = Path(__file__).resolve().parent


def parse_summary_last(log_path: Path):
    """解析日志中最后一条 'Average over ... | Test ...' 汇总行。

    与 tune.parse_summary 相同，但取最后一个匹配（train.py 日志为追加模式，
    补跑后日志尾部才是最新结果）。
    """
    metrics = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if "Average over" in line and "| Test" in line:
                parts = line.split("| Test", 1)[1].strip().split()
                metrics = {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}
    return metrics


def parse_mean_std(text):
    """'0.9843±0.0007' → (0.9843, 0.0007)；'0.9843' → (0.9843, 0.0)；无法解析 → None。"""
    if text is None:
        return None
    try:
        if "±" in text:
            m, s = text.split("±", 1)
            return float(m), float(s)
        return float(text), 0.0
    except ValueError:
        return None


def build_summary_table(results):
    """最优精度汇总表：每个模型一行（默认指标最优值及其 beta/anchor）。

    最优与各模型表格的绿色高亮同规则（默认指标均值最大，metric_mean）；
    按最优值降序排列、缺失/失败模型沉底标 FAILED；复用页面已有的表格样式。
    """
    key = next((k for _, _, _, m in results if m for k in ("Acc", "R2") if k in m), None)
    if key is None:
        return ""
    best = {}
    for model, b, a, m in results:
        mean = metric_mean(m, key) if m else float("-inf")
        if model not in best or mean > best[model][0]:
            best[model] = (mean, b, a, m[key] if m and key in m else None)
    items = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    best_idx = 0 if items and items[0][1][0] != float("-inf") else -1
    rows = []
    for i, (model, (mean, b, a, val)) in enumerate(items):
        cls = ' class="best"' if i == best_idx else ""
        beta_cell = "-" if b is None else f"{b:g}"
        anchor_cell = "-" if a is None else f"{a:g}"
        val_cell = val if val is not None else "FAILED"
        rows.append(
            f"<tr{cls}><td>{model}</td><td>{beta_cell}</td><td>{anchor_cell}</td><td>{val_cell}</td></tr>"
        )
    return f"""
<h2>最优精度汇总</h2>
<table>
<thead><tr><th>模型</th><th>beta</th><th>anchor_scale</th><th>最优 {key}</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
"""


def build_curve_chart(results):
    """生成压缩-精度对比曲线：一张图，只绘制 ceb（1 条线）与 opb
    （anchor ∈ {1, 6, 12} 三条线）——追加在全部表格之后，位于最优精度
    汇总表之下。opbl 为 opb 的结果改名，曲线中统一按 opb 绘制。

    指标取 Acc（分类）或 R2（回归，与表格默认指标一致）；x 为 log10(β)
    （所绘模型 β 的并集）；缺失/失败组合的点断开。
    """
    by_model = {}
    for model, b, a, m in results:
        if model == "opbl":
            model = "opb"  # opbl 是 opb 的结果改名，曲线按 opb 显示
        by_model.setdefault(model, []).append((b, a, m))
    key = next((k for _, _, _, m in results if m for k in ("Acc", "R2") if k in m), None)
    if key is None:
        return ""

    betas = sorted(
        {b for model, rows in by_model.items() if model in ("ceb", "opb")
         for b, _, _ in rows if b is not None}
    )
    if not betas:
        return ""

    series = []
    for model, rows in by_model.items():
        if model not in ("ceb", "opb"):
            continue
        if model == "opb":
            grid_anchors = {a for _, a, _ in rows if a is not None}
            a_list = [a for a in (1.0, 6.0, 12.0) if a in grid_anchors]
            skipped = [a for a in (1.0, 6.0, 12.0) if a not in grid_anchors]
            if skipped:
                print(f"提示：{model} 曲线跳过 anchor {skipped}（网格中不存在）")
        else:
            a_list = [None]
        for a in a_list:
            vals, stds, texts = [], [], []
            for b in betas:
                m = next((mm for bb, aa, mm in rows if bb == b and aa == a), None)
                ms = parse_mean_std(m[key]) if m and key in m else None
                if ms is None:
                    vals.append(None)
                    stds.append(None)
                    texts.append(None)
                else:
                    mean, std = ms
                    vals.append(mean)
                    stds.append(std)
                    texts.append(f"{mean:.4f}±{std:.4f}" if std > 1e-12 else f"{mean:.4f}")
            label = model if a is None else f"{model}（a={a:g}）"
            series.append((label, vals, stds, texts))
    if not series:
        return ""

    lo, hi = math.log10(betas[0]), math.log10(betas[-1])
    x0, x1, _, _, _ = plot_bounds(len(series))
    span = x1 - x0
    xs = [x0 + (math.log10(b) - lo) / (hi - lo) * span for b in betas] if hi > lo else [x0] * len(betas)
    x_labels = [f"{b:g}" for b in betas]

    if key == "R2":
        all_vals = [v for _, vs, _, _ in series for v in vs if v is not None]
        y_min = min(0.0, min(all_vals)) - 0.05
        y_max = max(1.0, max(all_vals)) + 0.05
    else:
        y_min, y_max = 0.0, 1.0
    card = curve_card(
        "tune-all",
        "压缩-精度对比曲线（ceb / opb）",
        f"y 为测试集 {'Acc' if key == 'Acc' else 'R²'}（跨 run 均值±标准差），"
        "x 为 β（对数刻度）；opb 各画 anchor=1/6/12 三条线；单击图例可显隐系列",
        xs, x_labels, series, y_min, y_max,
        col_titles=[f"β={b:g}" for b in betas],
    )
    return f"""
<h2>压缩-精度对比曲线</h2>
{curve_css()}
{card}
{curve_script()}
"""


def build_combos(args):
    """与 tune.py main 相同的组合网格构造，返回 [(model, beta, anchor), ...]。"""
    models = args.model
    betas = args.beta
    anchors = args.anchor_scale
    if "fgib" not in models and "opb" not in models and "opbl" not in models and len(anchors) > 1:
        print(f"警告：模型列表中没有 fgib/opb/opbl，--anchor-scale 列表不会被使用")
    combos = []
    for model in models:
        if model in BASELINES:
            combos.append((model, None, None))
        elif model in ("fgib", "opb", "opbl"):
            combos.extend((model, b, a) for b in betas for a in anchors)
        else:
            combos.extend((model, b, None) for b in betas)
    return combos


def scan(args, combos, results_root):
    """扫描结果目录，返回 (results, failed)：
    results 为 gen_html 需要的 [(model, beta, anchor, metrics_or_None)]；
    failed 为 [(model, beta, anchor, name, dir)] 缺失/失败组合。
    """
    results = []
    failed = []
    for model, b, a in combos:
        name = combo_name(model_prefix(args, model), b, a)
        d = results_root / name
        log = d / "train.log"
        m = parse_summary_last(log) if log.exists() else None
        results.append((model, b, a, m))
        if m is None:
            failed.append((model, b, a, name, d))
    return results, failed


def train_cmd_args(args):
    """build_train_cmd 用的参数副本：去掉本脚本专属的 --rerun 开关，
    避免其被重建进 train.py 命令行（train.py 不认识该参数）。"""
    cmd_args = argparse.Namespace(**vars(args))
    cmd_args.rerun = False
    return cmd_args


def print_failed_commands(parser, args, failed):
    """打印缺失/失败组合的状态与可直接执行的补跑命令。"""
    if not failed:
        return
    print("\n缺失/失败的组合（手动补跑命令）：")
    cmd_args = train_cmd_args(args)
    for model, b, a, name, d in failed:
        d.mkdir(parents=True, exist_ok=True)
        cmd = build_train_cmd(parser, cmd_args, model, b, a, d)
        if (d / "error.log").exists():
            status = "失败（error.log 存在）"
        elif not (d / "train.log").exists():
            status = "缺失（无 train.log）"
        else:
            status = "日志无汇总行（可能中断，补跑将追加到旧日志尾部）"
        print(f"# {name}  [{status}]")
        print("  " + " ".join(cmd))


def main():
    parser = build_tune_parser()
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="自动补跑缺失/失败的组合（复用 tune.py 的每卡独立队列并行与 OpenMP 线程限制），"
        "完成后重新扫描并重建 HTML",
    )
    args = parser.parse_args()

    if args.rerun and "opbl" in args.model:
        parser.error(
            "--rerun 不支持 opbl：opbl 仅为结果目录/HTML 的显示别名"
            "（与 opb 同构的 beta×anchor 网格），train.py 不认识该模型"
        )

    results_root = ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)
    combos = build_combos(args)

    results, failed = scan(args, combos, results_root)
    n_ok = len(combos) - len(failed)
    print(f"扫描完成：{n_ok}/{len(combos)} 组有汇总结果，{len(failed)} 组缺失/失败")

    if args.rerun and failed:
        n_gpus = detect_gpus()
        n_slots = n_gpus if n_gpus > 0 else 1
        total_parallel = args.parallel * n_slots
        n_cpus = cpu_count()
        omp_threads = max(1, n_cpus // total_parallel)
        gpu_desc = f"{n_gpus} 张 GPU" if n_gpus > 0 else "无 GPU（CPU）"
        print(
            f"\n自动补跑 {len(failed)} 组：{gpu_desc}，每卡独立队列 × 单卡并行 "
            f"{args.parallel} = 总并行 {total_parallel}；每进程 OpenMP 线程数："
            f"{omp_threads}（容器 {n_cpus} 核 / 总并行 {total_parallel}）"
        )

        slots = list(range(n_gpus)) if n_gpus > 0 else [None]
        cmd_args = train_cmd_args(args)
        with ExitStack() as stack:
            pools = [
                stack.enter_context(ThreadPoolExecutor(max_workers=args.parallel))
                for _ in slots
            ]
            futures = {}
            tracker = ProgressTracker(len(failed))
            for i, (model, b, a, name, d) in enumerate(failed):
                d.mkdir(parents=True, exist_ok=True)
                cmd = build_train_cmd(parser, cmd_args, model, b, a, d)
                slot = i % len(slots)
                futures[pools[slot].submit(
                    run_combo, cmd, d, tracker, slots[slot], omp_threads
                )] = (model, b, a)
            status = {}
            for fut in as_completed(futures):
                key = futures[fut]
                _, ok = fut.result()
                status[key] = ok

        n_ok_retry = sum(status.values())
        print(
            f"补跑完成：{len(failed)} 组（成功 {n_ok_retry}，失败 "
            f"{len(failed) - n_ok_retry}），总用时 {fmt_duration(tracker.elapsed())}"
        )
        results, failed = scan(args, combos, results_root)

    if not args.rerun:
        print_failed_commands(parser, args, failed)

    meta = (
        f"task={args.task} backbone={args.backbone} model={'+'.join(args.model)} "
        f"runs={args.runs} epochs={args.epochs} | {len(combos)} 组组合 | "
        "表格后附最优精度汇总表与 ceb/opb 压缩-精度对比曲线（β 对数刻度；"
        "opb 各 a=1/6/12 一条线）"
    )
    if len(args.model) == 1:
        html_name = f"{model_prefix(args, args.model[0])}_tune_results.html"
    else:
        html_name = (
            f"{get_dataset_name(args.task)}_{args.backbone}_"
            f"{'+'.join(args.model)}_tune_results.html"
        )
    html_path = results_root / html_name
    gen_html(results, html_path, meta, extra_html=build_summary_table(results) + build_curve_chart(results))
    print(f"\n调参结果表格已生成：{html_path}")

    if args.rerun and failed:
        print("以下组合补跑后仍缺失/失败：")
        print_failed_commands(parser, args, failed)
    elif failed:
        print("提示：补跑上述命令后，重新执行本脚本即可补全表格")


if __name__ == "__main__":
    main()
