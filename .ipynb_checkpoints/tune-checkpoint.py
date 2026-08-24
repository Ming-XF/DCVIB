"""调参脚本：按 beta × anchor-scale 网格并行调用 train.py 训练，汇总生成 HTML 结果表。

用法与 train.py 相同（复用其 build_parser），区别如下：
- --beta / --anchor-scale 接受多个值（空格分隔），对两个列表做笛卡尔积；
- --parallel 指定并行训练进程数；
- --results-dir 指定结果根目录，每个参数组合一个子文件夹（模型与日志存于其中）；
- 全部训练完成后在结果根目录生成 tune_results.html 调参结果表。

示例：
    python tune.py --model fgib --beta 1e-4 1e-3 1e-2 --anchor-scale 1 2 4 8 \\
        --parallel 4 --runs 3 --epochs 100
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from train import build_parser

ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"

# 调参专属或由脚本接管路径的参数，重建 train.py 命令行时跳过
SKIP_DESTS = {"beta", "anchor_scale", "parallel", "results_dir", "save_path", "log_path"}


def replace_arg(parser, dest, option_strings, **kwargs):
    """移除解析器中 dest 对应的旧参数，并以新定义重新添加。"""
    for action in list(parser._actions):
        if action.dest == dest:
            parser._actions.remove(action)
            for opt in action.option_strings:
                parser._option_string_actions.pop(opt, None)
    parser.add_argument(*option_strings, **kwargs)


def build_tune_parser():
    """train.py 解析器 + beta/anchor-scale 列表化 + 并行/结果目录参数。"""
    parser = build_parser()
    replace_arg(
        parser, "beta", ["--beta"],
        type=float, nargs="+", default=[1e-3],
        help="KL 权重列表，调参网格的一维（默认 [1e-3]，即不调 beta）",
    )
    replace_arg(
        parser, "anchor_scale", ["--anchor-scale"],
        type=float, nargs="+", default=[4.0],
        help="fgib 锚点尺度列表，调参网格的一维；仅 --model fgib 生效（默认 [4.0]）",
    )
    parser.add_argument(
        "--parallel", type=int, default=2,
        help="并行训练进程数（默认 2）",
    )
    parser.add_argument(
        "--results-dir", type=str, default="tune_results",
        help="调参结果根目录，每个参数组合一个子文件夹（默认 tune_results）",
    )
    return parser


def combo_dir(results_root: Path, beta: float, anchor_scale: float) -> Path:
    return results_root / f"beta_{beta:g}_anchor_{anchor_scale:g}"


def build_train_cmd(parser, args, beta, anchor_scale, out_dir):
    """根据解析后的参数重建 train.py 命令行（跳过调参专属参数，指定组合路径）。"""
    cmd = [sys.executable, str(TRAIN_PY)]
    for action in parser._actions:
        if action.dest in SKIP_DESTS or action.dest == "help":
            continue
        value = getattr(args, action.dest, None)
        if value is None:
            continue
        if action.default is not None and value == action.default:
            continue
        opt = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                cmd.append(opt)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                cmd.append(opt)
        elif isinstance(value, (list, tuple)):
            cmd.extend([opt, *(str(v) for v in value)])
        else:
            cmd.extend([opt, str(value)])
    cmd.extend(["--beta", str(beta)])
    cmd.extend(["--anchor-scale", str(anchor_scale)])
    cmd.extend(["--save-path", str(out_dir / "model.pt")])
    cmd.extend(["--log-path", str(out_dir / "train.log")])
    return cmd


def run_combo(cmd, out_dir):
    """运行一次 train.py 训练，返回 (out_dir, 是否成功)。"""
    name = out_dir.name
    print(f"[启动] {name}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok = proc.returncode == 0
    if not ok:
        (out_dir / "error.log").write_text(
            " ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr,
            encoding="utf-8",
        )
        print(f"[失败] {name} (exit {proc.returncode})，详见 {out_dir / 'error.log'}", flush=True)
    else:
        print(f"[完成] {name}", flush=True)
    return out_dir, ok


def parse_summary(log_path: Path):
    """从训练日志解析 'Average over N runs | Test ...' 汇总行，返回指标字典。"""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if "Average over" in line and "| Test" in line:
                parts = line.split("| Test", 1)[1].strip().split()
                return {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}
    return None


def gen_html(results, out_path: Path, meta: str):
    """生成调参结果 HTML 表格；按最佳验证指标（回归 R2 / 分类 AUC）高亮最佳行。"""
    metric_cols = next((list(m) for _, _, m in results if m), [])
    best_key = "R2" if "R2" in metric_cols else "AUC"

    best_val, best_idx = None, -1
    for i, (_, _, m) in enumerate(results):
        if m and best_key in m:
            v = float(m[best_key].split("±")[0])
            if best_val is None or v > best_val:
                best_val, best_idx = v, i

    rows = []
    for i, (beta, anchor, m) in enumerate(results):
        cls = ' class="best"' if i == best_idx else ""
        cells = [f"<td>{beta:g}</td>", f"<td>{anchor:g}</td>"]
        if m is None:
            cells.append(f'<td colspan="{max(len(metric_cols), 1)}">FAILED</td>')
        else:
            cells.extend(f"<td>{m.get(col, '-')}</td>" for col in metric_cols)
        rows.append(f"<tr{cls}>" + "".join(cells) + "</tr>")

    best_text = ""
    if best_val is not None:
        best_text = f"，最佳 {best_key} = {best_val:.4f}"

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>调参结果</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
h1 {{ font-size: 1.4em; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #999; padding: 6px 12px; text-align: center; }}
th {{ background: #eee; }}
tr.best {{ background: #dfd; font-weight: bold; }}
.meta {{ color: #555; margin-bottom: 1em; }}
</style>
</head>
<body>
<h1>调参结果</h1>
<div class="meta">{meta}{best_text}</div>
<table>
<thead><tr><th>beta</th><th>anchor_scale</th>{''.join(f'<th>{c}</th>' for c in metric_cols)}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    parser = build_tune_parser()
    args = parser.parse_args()

    betas = args.beta
    anchors = args.anchor_scale
    if args.model != "fgib" and len(anchors) > 1:
        print(
            f"警告：--model 为 {args.model}（非 fgib），--anchor-scale 不影响训练，"
            f"网格退化为仅 beta 维（anchor 取首值 {anchors[0]:g}）"
        )
        anchors = anchors[:1]

    results_root = ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    combos = [(b, a) for b in betas for a in anchors]
    print(
        f"参数网格：beta × anchor_scale = {len(betas)} × {len(anchors)} = {len(combos)} 组训练，"
        f"并行度 {args.parallel}"
    )

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {}
        for b, a in combos:
            d = combo_dir(results_root, b, a)
            d.mkdir(parents=True, exist_ok=True)
            cmd = build_train_cmd(parser, args, b, a, d)
            futures[pool.submit(run_combo, cmd, d)] = (b, a)
        status = {}
        for fut in as_completed(futures):
            b, a = futures[fut]
            _, ok = fut.result()
            status[(b, a)] = ok

    results = []
    for b, a in combos:
        d = combo_dir(results_root, b, a)
        m = parse_summary(d / "train.log") if status[(b, a)] else None
        results.append((b, a, m))

    meta = (
        f"task={args.task} model={args.model} backbone={args.backbone} "
        f"runs={args.runs} epochs={args.epochs} | {len(combos)} 组组合"
    )
    html_path = results_root / "tune_results.html"
    gen_html(results, html_path, meta)
    print(f"调参结果表格已生成：{html_path}")


if __name__ == "__main__":
    main()
