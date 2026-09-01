"""几何使用分析脚本（实验三A：几何是否真正参与预测——分类器不能绕过正交结构）。

实验二证明了 OPB 的后验表示被拉进正交框架（方向对齐、半径跟随）。但 OPB 的
分类器是普通 Linear，理论上可以完全无视锚点——本实验回答几何是分类的真正
依据还是"装饰"：

1. 最近锚点分类准确率（paper/OPB.txt §13.3）：用无参数规则
   argmin_j ‖mu_q(x) − anchor_j‖ 分类测试集，与训练分类器准确率对比
   （acc_gap = 分类器 − 最近锚点）。gap 小 → 表示完全活在正交框架内，
   一个无参数的锚点规则就足以分类。
2. 分类器权重-锚点对齐：A_w[k,j] = cos(classifier.weight[k], anchor_j) 的
   对角/非对角结构——决策面是否沿锚点方向放置。

因果性说明：对 CEB，"最近锚点准确率高"是预期内的——它的先验均值会被 KL
拉向后验中心（锚点追后验，反向因果）；对 OPB，锚点是构造固定、不追后验的，
若 acc_gap 仍小则是后验追锚点——结合实验二的 diag_align 证据构成完整因果链。
判定以条目 2（+后续能量分类器消融）为主证据，条目 1 为辅证；若 acc_gap 为负
（锚点规则反超分类器）也如实报告。

v1 仅支持 mnist MLP 的 CEB/OPB 配置（同实验二）。输出：
{results-dir}/geometry_usage.csv（长表，每 (config, run) 一行标量指标）、
geometry_usage.json（对齐矩阵与最近锚点预测；不存逐样本 mu_q，过大）与两张图
（最近锚点 vs 分类器准确率柱图、分类器权重-锚点对齐热力图网格）。

用法（--anchor-scale 须与 checkpoint 训练时一致）：

    python geometry_usage.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb
    python geometry_usage.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb \\
        --runs 5 --n-test 1000 --anchor-scale 10
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from datasets.datasets import get_mnist_dataloaders
from model.mlp.utils import flatten
from prior_geometry import (
    MUTED,
    _run_index,
    _series_color,
    _setup_rc,
    extract_prior_means,
    offdiag,
    parse_dir_name,
    plot_cosine_heatmaps,
)
from train import build_model, build_parser, run_model

ROOT = Path(__file__).resolve().parent


def fmt(x):
    """None 安全格式化（能量模型无条目 2 指标时为 N/A）。"""
    return f"{x:.4f}" if x is not None else "N/A"


def collect_posterior_logits(model, test_loader, n_test, device):
    """单遍收集测试集前 n_test 张的 mu_q、logits 与标签（z=μ 确定性路径）。

    与 posterior_geometry.collect_posterior 同路径，额外经统一接口 run_model
    取分类器 logits（labels=None 跳过 KL/QR 计算）。
    """
    model.eval()
    mus, logits_list, ys = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = run_model(model, x, None, stochastic=False)[0]
            mu = model.mu_head(model.encoder(flatten(x)))
            mus.append(mu.cpu())
            logits_list.append(logits.cpu())
            ys.append(y)
            if sum(len(t) for t in mus) >= n_test:
                break
    return (
        torch.cat(mus)[:n_test],
        torch.cat(logits_list)[:n_test],
        torch.cat(ys)[:n_test],
    )


def nearest_anchor_acc(mu_q, labels, anchors):
    """无参数最近锚点规则：pred = argmin_j ‖mu_q(x) − anchor_j‖ 的准确率。

    用完整 cdist 而非点积捷径：OPB 锚点模长恒 a 点积等价，但 CEB 的学习先验
    模长不齐，‖anchor_j‖² 项不可忽略。
    """
    dists = torch.cdist(mu_q, anchors)  # (N, K)
    preds = dists.argmin(1)
    return (preds == labels).float().mean().item(), preds


def summarize_usage(mu_q, logits, labels, anchors, model):
    """条目 1+2 的汇总，返回可 JSON 序列化的 dict。

    anchors 为 (K, d)：OPB 为 QR 锚点、CEB 为其学习先验均值（对照锚点）。
    能量分类器模型（energy_classifier=True）的 classifier 权重是死参数，
    条目 2（align_w）置 None、仅保留条目 1。
    """
    nearest_acc, preds = nearest_anchor_acc(mu_q, labels, anchors)
    classifier_acc = (logits.argmax(1) == labels).float().mean().item()

    s = {
        "nearest_anchor_acc": nearest_acc,
        "classifier_acc": classifier_acc,
        "acc_gap": classifier_acc - nearest_acc,
        "nearest_preds": preds.tolist(),
    }
    if getattr(model, "energy_classifier", False):
        s.update({"align_w": None, "mean_diag_align_w": None, "max_offdiag_align_w": None})
        return s
    W = model.classifier.weight.detach().cpu()  # (K, d)
    wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    an = anchors / anchors.norm(dim=1, keepdim=True).clamp_min(1e-8)
    align_w = wn @ an.t()  # (K, K)
    s["align_w"] = align_w.tolist()
    s["mean_diag_align_w"] = align_w.diagonal().mean().item()
    s["max_offdiag_align_w"] = offdiag(align_w).abs().max().item()
    return s


def agg_cross_run(run_summaries):
    """跨 run（=跨 seed）标量指标的均值±标准差；None 值（能量模型无条目 2）跳过。"""
    keys = [
        "nearest_anchor_acc",
        "classifier_acc",
        "acc_gap",
        "mean_diag_align_w",
        "max_offdiag_align_w",
    ]
    out = {}
    for k in keys:
        vals = [s[k] for s in run_summaries if s[k] is not None]
        if not vals:
            out[f"{k}_mean"] = None
            out[f"{k}_std"] = None
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        out[f"{k}_mean"] = t.mean().item()
        out[f"{k}_std"] = t.std().item()
    return out


def plot_acc_bars(results, path):
    """最近锚点 vs 分类器准确率分组柱图（跨 run 均值±std；浅色=最近锚点）。"""
    labels = list(results)
    fig, ax = plt.subplots(figsize=(2.6 * len(labels), 4))
    width = 0.3
    handles = []
    for i, label in enumerate(labels):
        cr = results[label]["cross_run"]
        color = _series_color(label)
        b1 = ax.bar(i - width / 2, cr["nearest_anchor_acc_mean"], width,
                    yerr=cr["nearest_anchor_acc_std"], color=color, alpha=0.4,
                    error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
        b2 = ax.bar(i + width / 2, cr["classifier_acc_mean"], width,
                    yerr=cr["classifier_acc_std"], color=color,
                    error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
        handles.extend([b1, b2])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("准确率")
    ax.set_title("无参数最近锚点规则（浅色）vs 训练分类器（实色），跨 run 均值±std", fontsize=10)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#898781", alpha=0.4),
                 plt.Rectangle((0, 0), 1, 1, facecolor="#898781")],
        labels=["最近锚点规则", "训练分类器"], fontsize=8,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_analysis_parser():
    parser = build_parser()
    parser.add_argument(
        "--model-dirs", nargs="+", required=True,
        help="已训练配置的 checkpoint 目录列表（每个目录内含 {stem}_run{i}.pt；"
        "目录名须解析为分类 CEB/OPB 配置，v1 仅支持 mnist MLP 目录）",
    )
    parser.add_argument(
        "--n-test", type=int, default=1000,
        help="测试集前 N 张（默认 1000）",
    )
    parser.add_argument(
        "--results-dir", type=str, default="geo_results",
        help="结果输出目录（默认 geo_results）",
    )
    return parser


def main():
    parser = build_analysis_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试集一次取齐（无 shuffle、顺序确定，各配置共用同批样本）
    _, _, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)

    results = {}
    csv_rows = []
    for d in (Path(p) for p in args.model_dirs):
        label = d.name
        dataset, backbone, model_name = parse_dir_name(label)
        if model_name not in ("ceb", "opb"):
            parser.error(f"目录 '{label}' 模型为 '{model_name}'，实验三A 仅支持 ceb/opb")
        if dataset != "mnist" or backbone != "mlp":
            parser.error(
                f"目录 '{label}'（{dataset}/{backbone}）不支持：实验三A v1 仅支持 "
                "mnist MLP（后验均值直接走 encoder/mu_head 提取）"
            )
        ckpts = sorted(d.glob("*_run*.pt"), key=_run_index)
        if len(ckpts) < args.runs:
            parser.error(
                f"目录 '{d}' 只找到 {len(ckpts)} 个 '*_run*.pt'，需要 --runs={args.runs} 个"
            )
        ckpts = ckpts[: args.runs]

        dir_args = argparse.Namespace(**vars(args))
        dir_args.task = dataset
        dir_args.model = model_name
        dir_args.backbone = backbone
        model = build_model(parser, dir_args).to(device)
        print(
            f"[加载] {label} → backbone={backbone} model={model_name}，"
            f"checkpoint {len(ckpts)} 个，n_test={args.n_test}"
        )

        run_summaries = []
        for i, ckpt in enumerate(ckpts, 1):
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            anchors, _ = extract_prior_means(model, model_name, args.anchor_scale)
            anchors = anchors.detach().cpu()  # 分析在 CPU 上进行
            mu_q, logits, labels = collect_posterior_logits(
                model, test_loader, args.n_test, device
            )
            s = summarize_usage(mu_q, logits, labels, anchors, model)
            s["run"] = i
            s["seed"] = args.seed + i - 1
            run_summaries.append(s)
            csv_rows.append(
                [
                    label,
                    i,
                    s["seed"],
                    s["nearest_anchor_acc"],
                    s["classifier_acc"],
                    s["acc_gap"],
                    s["mean_diag_align_w"] if s["mean_diag_align_w"] is not None else "",
                    s["max_offdiag_align_w"] if s["max_offdiag_align_w"] is not None else "",
                ]
            )
            print(
                f"[{label}] run{i} (seed {s['seed']}) 最近锚点={s['nearest_anchor_acc']:.4f} "
                f"分类器={s['classifier_acc']:.4f} gap={s['acc_gap']:+.4f} "
                f"diag_align_w={fmt(s['mean_diag_align_w'])} max_off_align_w={fmt(s['max_offdiag_align_w'])}"
            )

        cr = agg_cross_run(run_summaries)
        results[label] = {
            "dataset": dataset,
            "backbone": backbone,
            "model": model_name,
            "runs": run_summaries,
            "cross_run": cr,
        }
        print(
            f"[汇总] {label}: 最近锚点={cr['nearest_anchor_acc_mean']:.4f}"
            f"±{cr['nearest_anchor_acc_std']:.4f} | 分类器={cr['classifier_acc_mean']:.4f}"
            f"±{cr['classifier_acc_std']:.4f} | gap={cr['acc_gap_mean']:+.4f} | "
            f"diag_align_w={fmt(cr['mean_diag_align_w_mean'])} | "
            f"max_off_align_w={fmt(cr['max_offdiag_align_w_mean'])}"
        )

    # 对照表：判据是 OPB 的 gap 小（锚点规则逼近分类器）且分类器权重沿锚点方向
    print("\n[对照] 各模型跨 run 汇总")
    header = (
        f"{'model':<20} {'最近锚点':<16} {'分类器':<16} {'gap':<12} "
        f"{'diag_align_w':<14} {'max_off_align_w':<16}"
    )
    print(header)
    for label, res in results.items():
        cr = res["cross_run"]
        print(
            f"{label:<20} {cr['nearest_anchor_acc_mean']:.4f}±{cr['nearest_anchor_acc_std']:.4f}"
            f"{'':<2} {cr['classifier_acc_mean']:.4f}±{cr['classifier_acc_std']:.4f}"
            f"{'':<2} {cr['acc_gap_mean']:+.4f}±{cr['acc_gap_std']:.4f} "
            f"{fmt(cr['mean_diag_align_w_mean'])} {fmt(cr['max_offdiag_align_w_mean'])}"
        )

    out_root = ROOT / args.results_dir
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "runs": args.runs,
            "anchor_scale": args.anchor_scale,
            "n_test": args.n_test,
            "seed_base": args.seed,
        },
        **results,
    }
    json_path = out_root / "geometry_usage.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "geometry_usage.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "run",
                "seed",
                "nearest_anchor_acc",
                "classifier_acc",
                "acc_gap",
                "mean_diag_align_w",
                "max_offdiag_align_w",
            ]
        )
        w.writerows(csv_rows)

    _setup_rc()
    plot_acc_bars(results, out_root / "geometry_usage_acc.png")
    if all(s["align_w"] is not None for res in results.values() for s in res["runs"]):
        plot_cosine_heatmaps(
            results,
            out_root / "geometry_usage_align_w.png",
            suptitle="分类器权重-锚点对齐矩阵 cos(W[k], 锚点_j)（对角 = 决策面沿自锚方向）",
            key="align_w",
        )
        align_fig = f"          {out_root / 'geometry_usage_align_w.png'}"
    else:
        align_fig = "          （能量分类器模型无 classifier 权重，跳过对齐矩阵图）"

    print(f"\n结果已保存：{json_path}")
    print(f"汇总表已保存：{csv_path}")
    print(f"图已保存：{out_root / 'geometry_usage_acc.png'}")
    print(align_fig)


if __name__ == "__main__":
    main()
