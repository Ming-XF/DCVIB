"""调参脚本：按 模型 × beta × anchor-scale 网格并行调用 train.py 训练，汇总生成 HTML 结果表。

用法与 train.py 相同（复用其 build_parser），区别如下：
- --model / --beta / --anchor-scale 接受多个值（空格分隔）：
  - 基础模型（mlp/cnn/gcn/rnn）没有 beta 与 anchor-scale 维度，每个模型只训练一次；
  - vib/ceb/svib/nib/dvcca 只有 beta 维度；
  - fgib 有 beta × anchor-scale 两个维度；
  总试验数 = 各模型组合数之和；
- --parallel 指定**每张 GPU** 上的并行训练进程数；脚本自动检测可用 GPU
  （可用 `CUDA_VISIBLE_DEVICES` 环境变量限制），组合按轮转进入各 GPU 的独立队列
  （每卡并发严格 ≤ --parallel，互不影响），总并行 = GPU 数 × --parallel；
- --results-dir 指定结果根目录，每个参数组合一个子文件夹
  `{dataset}_{backbone}_{model}_beta_{b}_anchor_{a}/`（仅训练日志存于其中，**不保存模型参数**，按模型类型省略无维度后缀）；
- 全部训练完成后在结果根目录生成 HTML 调参结果表（每个模型一张表格，各表带排序下拉框，可自由选择排序指标）。

示例：
    python tune.py --model vib ceb fgib --beta 1e-4 1e-3 1e-2 --anchor-scale 1 2 4 8 \\
        --parallel 4 --runs 3 --epochs 100
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import torch

from train import build_parser, get_dataset_name

ROOT = Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train.py"

# 调参专属或由脚本接管路径的参数，重建 train.py 命令行时跳过
SKIP_DESTS = {"model", "beta", "anchor_scale", "parallel", "results_dir", "save_path", "log_path", "no_save"}

BASELINES = ("mlp", "cnn", "gcn", "rnn")


def replace_arg(parser, dest, option_strings, **kwargs):
    """移除解析器中 dest 对应的旧参数，并以新定义重新添加。"""
    for action in list(parser._actions):
        if action.dest == dest:
            parser._actions.remove(action)
            for opt in action.option_strings:
                parser._option_string_actions.pop(opt, None)
    parser.add_argument(*option_strings, **kwargs)


def build_tune_parser():
    """train.py 解析器 + model/beta/anchor-scale 列表化 + 并行/结果目录参数。"""
    parser = build_parser()
    replace_arg(
        parser, "model", ["--model"],
        nargs="+", choices=["mlp", "cnn", "gcn", "rnn", "vib", "ceb", "fgib", "svib", "nib", "dvcca"],
        default=["mlp"],
        help="模型列表，调参网格的一维；基础模型无 beta/anchor 维度，"
        "vib/ceb/svib/nib/dvcca 仅 beta 维度，fgib 为 beta × anchor-scale 两维（默认 [mlp]）",
    )
    replace_arg(
        parser, "beta", ["--beta"],
        type=float, nargs="+", default=[1e-3],
        help="KL 权重列表，调参网格的一维（默认 [1e-3]；基础模型不使用）",
    )
    replace_arg(
        parser, "anchor_scale", ["--anchor-scale"],
        type=float, nargs="+", default=[4.0],
        help="fgib 锚点尺度列表，调参网格的一维；仅 fgib 使用（默认 [4.0]）",
    )
    parser.add_argument(
        "--parallel", type=int, default=2,
        help="每张 GPU 上的并行训练进程数，总并行 = GPU 数 × 此值（默认 2）",
    )
    parser.add_argument(
        "--results-dir", type=str, default="tune_results",
        help="调参结果根目录，每个参数组合一个子文件夹（默认 tune_results）",
    )
    return parser


def detect_gpus():
    """检测可用 GPU 数量（受环境变量 CUDA_VISIBLE_DEVICES 限制）；无 GPU 返回 0。"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def model_prefix(args, model: str) -> str:
    """按 train.py 输出目录约定生成 {dataset}_{backbone}_{model} 前缀（基线为 {dataset}_{model}）。"""
    dataset_name = get_dataset_name(args.task)
    if model in BASELINES:
        return f"{dataset_name}_{model}"
    return f"{dataset_name}_{args.backbone}_{model}"


def combo_name(prefix: str, beta, anchor_scale) -> str:
    """组合子文件夹名：按模型类型省略无维度的后缀。"""
    name = prefix
    if beta is not None:
        name += f"_beta_{beta:g}"
    if anchor_scale is not None:
        name += f"_anchor_{anchor_scale:g}"
    return name


def build_train_cmd(parser, args, model, beta, anchor_scale, out_dir):
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
    cmd.extend(["--model", model])
    if beta is not None:
        cmd.extend(["--beta", str(beta)])
    if anchor_scale is not None:
        cmd.extend(["--anchor-scale", str(anchor_scale)])
    cmd.append("--no-save")
    cmd.extend(["--log-path", str(out_dir / "train.log")])
    return cmd


def run_combo(cmd, out_dir, gpu=None):
    """运行一次 train.py 训练，返回 (out_dir, 是否成功)。

    gpu 为分配给该进程的 GPU 序号（通过 CUDA_VISIBLE_DEVICES 指定）；None 表示 CPU 运行。
    """
    name = out_dir.name
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[启动] {name} (GPU {gpu})", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
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


def metric_mean(m, key):
    """取指标 'mean±std' 字符串中的均值；缺失或失败返回 -inf（排序时沉底）。"""
    if not m or key not in m:
        return float("-inf")
    return float(m[key].split("±")[0])


def gen_html(results, out_path: Path, meta: str):
    """生成调参结果 HTML：每个模型一张表格，各表可通过下拉框自由选择排序指标。

    results: [(model, beta, anchor, metrics_dict_or_None), ...]
    """
    metric_cols = next((list(m) for _, _, _, m in results if m), [])
    default_key = "Acc" if "Acc" in metric_cols else ("R2" if "R2" in metric_cols else None)

    groups = {}
    for model, beta, anchor, m in results:
        groups.setdefault(model, []).append((model, beta, anchor, m))

    sections = []
    for model, group in groups.items():
        if default_key:
            group = sorted(group, key=lambda r: -metric_mean(r[3], default_key))
        best_idx = next(
            (i for i, (_, _, _, m) in enumerate(group) if m and default_key in m), -1
        )

        rows = []
        for i, (_, beta, anchor, m) in enumerate(group):
            cls = ' class="best"' if i == best_idx else ""
            beta_cell = "-" if beta is None else f"{beta:g}"
            anchor_cell = "-" if anchor is None else f"{anchor:g}"
            data_attrs = (
                # 属性名统一小写：HTML 解析器会把属性名小写化，JS 侧同样用 key.toLowerCase() 查找
                "".join(f' data-{col.lower()}="{metric_mean(m, col):.6g}"' for col in metric_cols)
                if m else ""
            )
            cells = [f"<td>{beta_cell}</td>", f"<td>{anchor_cell}</td>"]
            if m is None:
                cells.append(f'<td colspan="{max(len(metric_cols), 1)}">FAILED</td>')
            else:
                cells.extend(f"<td>{m.get(col, '-')}</td>" for col in metric_cols)
            rows.append(f"<tr{cls}{data_attrs}>" + "".join(cells) + "</tr>")

        options = "".join(
            f'<option value="{col}"{" selected" if col == default_key else ""}>{col}</option>'
            for col in metric_cols
        )
        table_id = f"table-{model}"
        sections.append(f"""
<h2>{model}</h2>
<div class="sort-row">排序指标：
<select class="sort-select" data-table="{table_id}">
{options}
</select>
（Acc/AUC/Pre/Rec/R2 降序，Loss/MAE 升序）</div>
<table id="{table_id}">
<thead><tr><th>beta</th><th>anchor_scale</th>{''.join(f'<th>{col}</th>' for col in metric_cols)}</tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
""")

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>调参结果</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
h1 {{ font-size: 1.4em; }}
h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #999; padding: 6px 12px; text-align: center; }}
th {{ background: #eee; }}
tr.best {{ background: #dfd; font-weight: bold; }}
.meta {{ color: #555; margin-bottom: 1em; }}
.sort-row {{ margin-bottom: 0.5em; }}
</style>
</head>
<body>
<h1>调参结果</h1>
<div class="meta">{meta}（绿色高亮 = 各模型默认指标 {default_key or '-'} 的最佳行）</div>
{''.join(sections)}
<script>
function sortTable(tableId, key) {{
  const tbody = document.getElementById(tableId).querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const ascending = key === "Loss" || key === "MAE";
  rows.sort((a, b) => {{
    const k = key.toLowerCase();
    const va = parseFloat(a.dataset[k]);
    const vb = parseFloat(b.dataset[k]);
    const fa = isNaN(va) ? (ascending ? Infinity : -Infinity) : va;
    const fb = isNaN(vb) ? (ascending ? Infinity : -Infinity) : vb;
    return ascending ? fa - fb : fb - fa;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
document.querySelectorAll(".sort-select").forEach(sel => {{
  sel.addEventListener("change", () => sortTable(sel.dataset.table, sel.value));
}});
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    parser = build_tune_parser()
    args = parser.parse_args()

    models = args.model
    betas = args.beta
    anchors = args.anchor_scale
    if "fgib" not in models and len(anchors) > 1:
        print(f"警告：模型列表中没有 fgib，--anchor-scale 列表不会被使用")

    results_root = ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    combos = []
    for model in models:
        if model in BASELINES:
            combos.append((model, None, None))
        elif model == "fgib":
            combos.extend((model, b, a) for b in betas for a in anchors)
        else:
            combos.extend((model, b, None) for b in betas)

    per_model = {m: sum(1 for c in combos if c[0] == m) for m in models}
    n_gpus = detect_gpus()
    n_slots = n_gpus if n_gpus > 0 else 1
    total_parallel = args.parallel * n_slots
    gpu_desc = f"{n_gpus} 张 GPU" if n_gpus > 0 else "无 GPU（CPU）"
    print(
        f"参数网格：共 {len(combos)} 组训练，{gpu_desc}，每卡独立队列 × 单卡并行 {args.parallel} = 总并行 {total_parallel}；"
        f"各组数 " + " ".join(f"{m}×{n}" for m, n in per_model.items())
    )

    # 每 GPU 一个独立队列（各一个线程池，并发 ≤ --parallel），组合按轮转进入各卡队列
    slots = list(range(n_gpus)) if n_gpus > 0 else [None]
    with ExitStack() as stack:
        pools = [
            stack.enter_context(ThreadPoolExecutor(max_workers=args.parallel))
            for _ in slots
        ]
        futures = {}
        for i, (model, b, a) in enumerate(combos):
            d = results_root / combo_name(model_prefix(args, model), b, a)
            d.mkdir(parents=True, exist_ok=True)
            cmd = build_train_cmd(parser, args, model, b, a, d)
            slot = i % len(slots)
            futures[pools[slot].submit(run_combo, cmd, d, slots[slot])] = (model, b, a)
        status = {}
        for fut in as_completed(futures):
            key = futures[fut]
            _, ok = fut.result()
            status[key] = ok

    results = []
    for model, b, a in combos:
        d = results_root / combo_name(model_prefix(args, model), b, a)
        m = parse_summary(d / "train.log") if status[(model, b, a)] else None
        results.append((model, b, a, m))

    meta = (
        f"task={args.task} backbone={args.backbone} model={'+'.join(models)} "
        f"runs={args.runs} epochs={args.epochs} | {len(combos)} 组组合"
    )
    if len(models) == 1:
        html_name = f"{model_prefix(args, models[0])}_tune_results.html"
    else:
        html_name = (
            f"{get_dataset_name(args.task)}_{args.backbone}_"
            f"{'+'.join(models)}_tune_results.html"
        )
    html_path = results_root / html_name
    gen_html(results, html_path, meta)
    print(f"调参结果表格已生成：{html_path}")


if __name__ == "__main__":
    main()
