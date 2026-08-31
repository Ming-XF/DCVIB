"""从训练结果重建调参 HTML 结果表；--rerun 时先自动补跑缺失/失败的组合。

默认行为（不重新训练）：扫描结果目录、解析每个日志中**最后一条**汇总行
（train.py 的 FileHandler 为追加模式，补跑后日志尾部才是最新结果），并用与
tune.py 完全相同的逻辑重建 HTML；缺失/失败的组合打印可直接执行的 train.py 命令。

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
    model_prefix,
    run_combo,
)

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


def build_combos(args):
    """与 tune.py main 相同的组合网格构造，返回 [(model, beta, anchor), ...]。"""
    models = args.model
    betas = args.beta
    anchors = args.anchor_scale
    if "fgib" not in models and "opb" not in models and len(anchors) > 1:
        print(f"警告：模型列表中没有 fgib/opb，--anchor-scale 列表不会被使用")
    combos = []
    for model in models:
        if model in BASELINES:
            combos.append((model, None, None))
        elif model in ("fgib", "opb"):
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
        f"runs={args.runs} epochs={args.epochs} | {len(combos)} 组组合"
    )
    if len(args.model) == 1:
        html_name = f"{model_prefix(args, args.model[0])}_tune_results.html"
    else:
        html_name = (
            f"{get_dataset_name(args.task)}_{args.backbone}_"
            f"{'+'.join(args.model)}_tune_results.html"
        )
    html_path = results_root / html_name
    gen_html(results, html_path, meta)
    print(f"\n调参结果表格已生成：{html_path}")

    if args.rerun and failed:
        print("以下组合补跑后仍缺失/失败：")
        print_failed_commands(parser, args, failed)
    elif failed:
        print("提示：补跑上述命令后，重新执行本脚本即可补全表格")


if __name__ == "__main__":
    main()
