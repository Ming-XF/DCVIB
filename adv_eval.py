"""对抗鲁棒性评估脚本：对已训练的 MNIST 模型做 untargeted PGD（L∞/L2）攻击评估。

不训练任何参数：加载各配置目录下的最优 checkpoint（`{stem}_run{i}.pt`），在
测试集（前 `--n-test` 张）上以 z=μ 确定性路径（`stochastic=False`，与 CEB 论文
评估口径一致——评估时随机模型与确定性模型无异）做白盒攻击，报告 clean acc 与
各 ε 的鲁棒精度。ε 定义在归一化像素空间（MNIST mean 0.1307 / std 0.3081）。

用法（先用 `--save-path` 把每个配置训到独立目录，save-path 是 stem、自动生成
`{stem}_run{i}.pt`）：

    python train.py --model vib --beta 0.01 \\
        --save-path output/adv_mnist/mnist_mlp_vib_beta_0.01/model.pt
    python adv_eval.py \\
        --model-dirs output/adv_mnist/mnist_mlp_vib_beta_0.01 \\
                     output/adv_mnist/mnist_mlp_opb_beta_0.01_anchor_4 \\
        --eps-linf 0.1 0.2 0.3 --eps-l2 0.5 1.0 1.5 2.0 --pgd-steps 20 \\
        --n-test 1000 --runs 5

目录最后一级名字必须能解析出架构：`{dataset}_{backbone}_{model}`（变体，允许
自定义后缀如 `_beta_0.01`）或 `{dataset}_{model}`（基线）。checkpoint 须用默认
架构参数训练（hidden-dims 512 256、z-dim 256、dropout 0.2、anchor-scale 4.0）；
非默认值需用对应的全局参数（`--hidden-dims`/`--z-dim`/`--dropout`/`--anchor-scale`）
覆盖。opb 例外：checkpoint 须以 `--energy-classifier` 训练，评估时自动按能量
分类器重建（无需传该标志）；anchor_scale 仍须经 `--anchor-scale` 与训练值一致。
v1 仅支持 MNIST。

输出：`{results-dir}/{dataset}_adv.csv`（长表）与
`{results-dir}/{dataset}_adv.html`（汇总表，每配置一行、跨 run 均值±标准差；
表后附鲁棒精度-ε 对比曲线：L∞/L2 各一张，每配置一条折线、带均值±标准差误差棒，
图例可点击显隐、悬停/键盘方向键查看各配置数值）。
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.datasets import get_mnist_dataloaders
from train import build_model, build_parser, run_model
from tune import gen_html
from utils import curve_card, curve_css, curve_script, plot_bounds

ROOT = Path(__file__).resolve().parent

DATASETS = {"mnist", "imagenet100", "cora", "imdb", "agnews", "california", "stsb", "zinc", "agedb"}
BACKBONES = {"mlp", "cnn", "gnn", "rnn"}
VARIANTS = {"vib", "ceb", "fgib", "opb", "svib", "nib", "dvcca"}
BASELINE_MODELS = {"mlp", "cnn", "gcn", "rnn"}


def parse_dir_name(name: str):
    """从目录最后一级名字解析 (dataset, backbone, model)。"""
    parts = name.split("_")
    if not parts or parts[0] not in DATASETS:
        raise ValueError(f"无法解析目录名 '{name}'：首段必须是已知数据集名")
    dataset = parts[0]
    if len(parts) == 2 and parts[1] in BASELINE_MODELS:
        return dataset, parts[1], parts[1]
    if len(parts) >= 3 and parts[1] in BACKBONES and parts[2] in VARIANTS:
        return dataset, parts[1], parts[2]
    raise ValueError(
        f"无法解析目录名 '{name}'：应为 {{dataset}}_{{model}}（基线）或 "
        "{{dataset}}_{{backbone}}_{{model}}（变体，允许自定义后缀）"
    )


def _run_index(p: Path) -> int:
    m = re.search(r"_run(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def pgd_attack(model, images, labels, eps, norm, steps):
    """untargeted PGD（随机起点），返回对抗样本张量。

    模型走 stochastic=False（z=μ）确定性路径；labels=None 使 run_model
    跳过 KL/QR 计算。L∞：步长 2.5ε/steps·sign(grad)，clamp 到 [−ε,ε]；
    L2：梯度按样本归一化，步长 2.5ε/steps，超球时缩回。
    """
    alpha = 2.5 * eps / steps
    if norm == "linf":
        delta = torch.empty_like(images).uniform_(-eps, eps)
    else:  # l2：随机方向 × eps
        d = torch.randn_like(images)
        d = d / d.view(len(images), -1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        delta = d * eps
    delta = delta.detach().requires_grad_(True)

    for _ in range(steps):
        logits = run_model(model, images + delta, None, stochastic=False)[0]
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, delta)[0]
        if norm == "linf":
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
        else:
            d = delta + alpha * grad / grad.view(len(images), -1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            n = d.view(len(images), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
            delta = d * (eps / n.clamp_min(eps))
        delta = delta.detach().requires_grad_(True)

    return (images + delta).detach()


def fmt_mean_std(values):
    """[float] → 'mean±std' 字符串（与仓库日志汇总约定一致；单 run 时只给均值）。"""
    t = torch.tensor(values, dtype=torch.float64)
    if t.numel() == 1:
        return f"{t.item():.4f}"
    return f"{t.mean():.4f}±{t.std():.4f}"


# ---------------------------------------------------------------------------
# 鲁棒精度-ε 对比曲线：共享渲染器在 utils.py（curve_card/curve_css/curve_script/
# plot_bounds，dataviz 规范：2px 折线、≥8px 圆点带 2px 面环、发丝实线网格、
# 图例恒在（≥2 系列）、≤4 系列末端直标、十字准线 + tooltip、浅/深双模式）。
# ---------------------------------------------------------------------------

CURVE_NORM_TITLES = {"linf": "L∞ PGD 对比曲线", "l2": "L2 PGD 对比曲线"}


def _adv_curve_card(norm, eps_grid, prefix, curve_rows, n_runs):
    """单张范数对比图：把 eps 网格与 per-run 统计换算成曲线卡数据。"""
    keys = ["Acc"] + [f"{prefix}{e:g}" for e in eps_grid]
    x_max = max(eps_grid)
    x0, x1, _, _, _ = plot_bounds(len(curve_rows))
    eps_all = [0.0] + list(eps_grid)
    xs = [x0 + (e / x_max) * (x1 - x0) for e in eps_all]
    series = []
    for label, means, stds in curve_rows:
        vals = [means[k] for k in keys]
        errs = [stds[k] if n_runs > 1 else 0.0 for k in keys]
        texts = [f"{m:.4f}±{s:.4f}" if s > 1e-12 else f"{m:.4f}" for m, s in zip(vals, errs)]
        series.append((label, vals, errs, texts))
    title = CURVE_NORM_TITLES[norm]
    subtitle = (
        f"ε 定义在归一化像素空间；ε=0 为 clean 精度；误差棒为跨 {n_runs} run 的"
        "均值±标准差；单击图例可显隐配置"
    )
    return curve_card(
        norm, title, subtitle, xs,
        [f"{e:g}" for e in eps_all], series, 0.0, 1.0,
        x_sublabels=["clean"] + [None] * len(eps_grid),
        col_titles=["clean"] + [f"ε={e:g}" for e in eps_grid],
    )


def build_adv_curves_html(curve_rows, eps_linf, eps_l2, n_runs):
    """生成对比曲线区块 HTML：L∞/L2 各一张 SVG 折线图。

    curve_rows: [(label, {指标: mean}, {指标: std}), ...]，指标键为
    "Acc" / f"Linf{e:g}" / f"L2{e:g}"（与 main 的 metric_names 一致）。
    经 tune.gen_html(extra_html=...) 追加在汇总表之后。
    """
    if not curve_rows:
        return ""
    if len(curve_rows) > 8:
        print(f"警告：曲线调色板只有 8 色，{len(curve_rows)} 个配置将复用颜色（线型兜底）")
    cards = []
    for norm, eps_grid, prefix in (
        ("linf", [e for e in eps_linf if e != 0.0], "Linf"),
        ("l2", [e for e in eps_l2 if e != 0.0], "L2"),
    ):
        if not eps_grid:
            continue
        cards.append(_adv_curve_card(norm, eps_grid, prefix, curve_rows, n_runs))
    if not cards:
        return ""
    return f"""
<h2>鲁棒精度对比曲线</h2>
{curve_css()}
{''.join(cards)}
{curve_script()}
"""


def build_adv_parser():
    parser = build_parser()
    parser.add_argument(
        "--model-dirs", nargs="+", required=True,
        help="已训练配置的 checkpoint 目录列表（每个目录内含 {stem}_run{i}.pt；"
        "目录名须可解析出架构，见脚本 docstring）",
    )
    parser.add_argument(
        "--eps-linf", type=float, nargs="*", default=[0.1, 0.2, 0.3],
        help="L∞ 攻击 ε 网格（归一化像素空间，默认 [0.1 0.2 0.3]；传空值跳过该范数）",
    )
    parser.add_argument(
        "--eps-l2", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0],
        help="L2 攻击 ε 网格（归一化像素空间，默认 [0.5 1.0 1.5 2.0]；传空值跳过该范数）",
    )
    parser.add_argument("--pgd-steps", type=int, default=20, help="PGD 步数（默认 20）")
    parser.add_argument("--n-test", type=int, default=1000, help="测试集前 N 张（默认 1000）")
    parser.add_argument("--results-dir", type=str, default="adv_results", help="结果输出目录（默认 adv_results）")
    return parser


def main():
    parser = build_adv_parser()
    args = parser.parse_args()

    if args.task != "mnist":
        parser.error("adv_eval v1 仅支持 --task mnist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试集前 n_test 张（原图与标签一次取齐，全部配置共用）
    _, _, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
        if sum(len(t) for t in xs) >= args.n_test:
            break
    images = torch.cat(xs)[: args.n_test].to(device)
    labels = torch.cat(ys)[: args.n_test].to(device)

    combos = (
        [("none", 0.0)]
        + [("linf", e) for e in args.eps_linf if e != 0.0]
        + [("l2", e) for e in args.eps_l2 if e != 0.0]
    )
    metric_names = (
        ["Acc"]
        + [f"Linf{e:g}" for e in args.eps_linf if e != 0.0]
        + [f"L2{e:g}" for e in args.eps_l2 if e != 0.0]
    )

    results = []  # [(label, None, None, metrics)] 供 gen_html
    curve_rows = []  # [(label, {指标: mean}, {指标: std})] 供对比曲线图
    csv_rows = []

    for d in (Path(p) for p in args.model_dirs):
        label = d.name
        dataset, backbone, model = parse_dir_name(label)
        if dataset != "mnist":
            parser.error(f"目录 '{label}' 不是 MNIST 配置（adv_eval v1 仅支持 mnist）")
        ckpts = sorted(d.glob("*_run*.pt"), key=_run_index)
        if len(ckpts) < args.runs:
            parser.error(
                f"目录 '{d}' 只找到 {len(ckpts)} 个 '*_run*.pt'，需要 --runs={args.runs} 个"
            )
        ckpts = ckpts[: args.runs]

        # 用目录名解析出的 (backbone, model) 重建架构；非默认架构参数经命令行全局覆盖
        dir_args = argparse.Namespace(**vars(args))
        dir_args.model = model
        dir_args.backbone = backbone
        if model == "opb":
            # OPB 固定按能量分类器重建（paper/OPB.txt §12）：energy 训练下
            # self.classifier 是死参数（随机初始化），按默认普通分类器路径会得到
            # 随机输出（clean acc ≈ 1/K）。无需命令行传 --energy-classifier；
            # anchor_scale 仍经 --anchor-scale 全局覆盖，须与训练值一致（能量
            # logits 的锚点位置依赖它）。
            dir_args.energy_classifier = True
        model_net = build_model(parser, dir_args).to(device)
        print(f"[加载] {label} → backbone={backbone} model={model}，checkpoint {len(ckpts)} 个")

        per_run = {k: [] for k in metric_names}
        for run_i, ckpt in enumerate(ckpts, 1):
            model_net.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            model_net.eval()
            for (norm, eps), key in zip(combos, metric_names):
                if eps == 0.0:
                    x_in = images
                else:
                    x_in = pgd_attack(model_net, images, labels, eps, norm, args.pgd_steps)
                with torch.no_grad():
                    logits = run_model(model_net, x_in, None, stochastic=False)[0]
                    acc = (logits.argmax(1) == labels).float().mean().item()
                per_run[key].append(acc)
                csv_rows.append([label, run_i, norm, f"{eps:g}", acc])
                print(f"[{label}] run{run_i} {norm} ε={eps:g} acc {acc:.4f}")

        metrics = {k: fmt_mean_std(v) for k, v in per_run.items()}
        results.append((label, None, None, metrics))
        means, stds = {}, {}
        for k, v in per_run.items():
            t = torch.tensor(v, dtype=torch.float64)
            means[k] = float(t.mean())
            stds[k] = float(t.std())
        curve_rows.append((label, means, stds))
        print(f"[汇总] {label}: " + " ".join(f"{k}={v}" for k, v in metrics.items()))

    results_root = ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / f"{args.task}_adv.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "norm", "eps", "acc"])
        w.writerows(csv_rows)
    meta = (
        f"task={args.task} pgd_steps={args.pgd_steps} n_test={args.n_test} | "
        f"{len(results)} 个配置（每行跨 {args.runs} run 的均值±标准差；"
        "Acc 为 clean acc，Linf/L2 为对应 ε 的鲁棒精度）"
    )
    html_path = results_root / f"{args.task}_adv.html"
    curves_html = build_adv_curves_html(curve_rows, args.eps_linf, args.eps_l2, args.runs)
    gen_html(results, html_path, meta, extra_html=curves_html)
    print(f"\n结果已保存：{csv_path}\n调参结果表格已生成：{html_path}")


if __name__ == "__main__":
    main()
