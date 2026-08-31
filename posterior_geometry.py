"""后验几何分析脚本（实验二：机制传递验证——先验正交是否驱动后验分离）。

实验一（prior_geometry.py）证明了 CEB 的先验几何任意、OPB 的先验严格正交。
本实验回答更关键的问题：分类器吃的是后验 z，先验只是正则手段——OPB 的正交
先验是否真的把后验表示"拉进"了正交框架？对每个 checkpoint 在 MNIST 测试集
（前 --n-test 张）上以 z=μ 确定性路径逐样本取后验均值 mu_q(x)，按类别聚合
类中心 c_k = mean(mu_q(x) | y=k)，测量：

- 后验类中心 Gram（两两余弦）：OPB 应近对角（显式分离结构），CEB 无结构；
- 类中心-锚点对齐矩阵 cos(c_k, 锚点_j)：OPB 对角应高（自锚对齐）、非对角低
  （无串扰）；CEB 以其学习先验均值为锚点作同样计算作对照；
- 类内散布 W（样本到自身类中心均方距离）/ 类间散布 B（类中心加权方差）/
  Fisher 比 B/W：OPB 应显著更高——"显式分离结构"的直接量化；
- 跨 seed 几何一致性（多 run 类中心 Gram 的逐对 std）：OPB 稳定、CEB 漂移；
- 半径匹配 ‖c_k‖ ≈ a 与类中心到自锚点距离：后验真的追到锚点球面的证据。

v1 仅支持 mnist MLP 的 CEB/OPB 配置（后验均值直接走 encoder/mu_head 提取）。
输出：posterior_geometry.csv（长表，每 (config, run) 一行标量指标）、
posterior_geometry.json（完整矩阵：类中心、Gram、对齐矩阵等）与四张图
（类中心 Gram / 对齐矩阵热力图网格、散布与 Fisher 柱图、类中心模长箱线）。

用法（与 prior_geometry.py 相同的目录约定）：

    python posterior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb
    python posterior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb \\
        --runs 5 --n-test 2000 --anchor-scale 8
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
    cross_run_summary,
    extract_prior_means,
    offdiag,
    parse_dir_name,
    plot_cosine_heatmaps,
    plot_norms,
    summarize,
)
from train import build_model, build_parser

ROOT = Path(__file__).resolve().parent

# cross_run_summary 额外统计的标量键（基础键为实验一的几何指标）
EXTRA_SCALAR_KEYS = [
    "mean_diag_align",
    "max_offdiag_align",
    "within_scatter",
    "between_scatter",
    "fisher_ratio",
    "center_norm_mean",
    "mean_anchor_dist",
]


def collect_posterior(model, test_loader, n_test, device):
    """收集测试集前 n_test 张的确定性后验均值 mu_q 与标签（z=μ 评估路径）。

    v1 仅 MLP 骨干：直接走 encoder + mu_head，不走完整 forward。
    """
    model.eval()
    mus, ys = [], []
    with torch.no_grad():
        for x, y in test_loader:
            mu = model.mu_head(model.encoder(flatten(x.to(device))))
            mus.append(mu.cpu())
            ys.append(y)
            if sum(len(t) for t in mus) >= n_test:
                break
    return torch.cat(mus)[:n_test], torch.cat(ys)[:n_test]


def posterior_summarize(mu_q, labels, anchors, anchor_scale):
    """后验几何汇总：类中心、中心 Gram、对齐矩阵、散布、半径与锚点距离。

    先复用 summarize（键名与实验一一致，矩阵为类中心的两两几何），再追加
    对齐矩阵、散布与半径指标。anchors 为各类锚点 (K, d)（OPB 为 QR 锚点、
    CEB 为其学习先验均值，作对照锚点）。
    """
    K = anchors.size(0)
    d = mu_q.size(1)
    zero = torch.zeros(d)
    centers = torch.stack(
        [mu_q[labels == k].mean(0) if (labels == k).any() else zero for k in range(K)]
    )
    s = summarize(centers, anchor_scale)  # 类中心的两两几何（键名同实验一）

    cn = centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-8)
    an = anchors / anchors.norm(dim=1, keepdim=True).clamp_min(1e-8)
    align = cn @ an.t()  # (K, K)
    s["align"] = align.tolist()
    s["mean_diag_align"] = align.diagonal().mean().item()
    s["max_offdiag_align"] = offdiag(align).abs().max().item()

    # 类内散布 W：样本到自身类中心的均方距离；类间散布 B：类中心加权方差
    within = ((mu_q - centers[labels]) ** 2).sum(1)
    s["within_scatter"] = within.mean().item()
    n_k = torch.bincount(labels, minlength=K).float()
    between = (n_k * ((centers - mu_q.mean(0)) ** 2).sum(1)).sum() / n_k.sum()
    s["between_scatter"] = between.item()
    W, B = s["within_scatter"], s["between_scatter"]
    s["fisher_ratio"] = B / W if W > 0 else float("inf")

    # 半径匹配：后验中心是否追到锚点球面（‖c_k‖ ≈ a、距自锚点近）
    s["center_norm_mean"] = centers.norm(dim=1).mean().item()
    s["mean_anchor_dist"] = (centers - anchors).norm(dim=1).mean().item()
    return s


def plot_scatter_bars(results, path):
    """散布与 Fisher 比柱图：左=类内/类间散布、右=Fisher 比（跨 run 均值±std）。

    柱色按模型（CEB 蓝 / OPB 橙）；散布图中浅色=类内 W、深色=类间 B。
    """
    labels = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.3
    for i, label in enumerate(labels):
        cr = results[label]["cross_run"]
        color = _series_color(label)
        ax1.bar(i - width / 2, cr["within_scatter_mean"], width,
                yerr=cr["within_scatter_std"], color=color, alpha=0.4,
                error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
        ax1.bar(i + width / 2, cr["between_scatter_mean"], width,
                yerr=cr["between_scatter_std"], color=color,
                error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
        ax2.bar(i, cr["fisher_ratio_mean"], width,
                yerr=cr["fisher_ratio_std"], color=color,
                error_kw=dict(elinewidth=1, ecolor=MUTED, capsize=3))
    ax1.set_xticks(range(len(labels)), labels)
    ax2.set_xticks(range(len(labels)), labels)
    ax1.set_ylabel("均方距离")
    ax2.set_ylabel("B / W")
    ax1.set_title("类内散布 W（浅色）/ 类间散布 B（深色），跨 run 均值±std", fontsize=9)
    ax2.set_title("Fisher 比（类间/类内，越高分离越强）", fontsize=9)
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
        help="测试集前 N 张用于后验均值统计（默认 1000）",
    )
    parser.add_argument(
        "--results-dir", type=str, default="pri_results",
        help="结果输出目录（默认 pri_results）",
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
            parser.error(f"目录 '{label}' 模型为 '{model_name}'，实验二仅支持 ceb/opb")
        if dataset != "mnist" or backbone != "mlp":
            parser.error(
                f"目录 '{label}'（{dataset}/{backbone}）不支持：实验二 v1 仅支持 "
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
            anchors = anchors.detach().cpu()  # 后验统计在 CPU 上进行
            mu_q, labels = collect_posterior(model, test_loader, args.n_test, device)
            s = posterior_summarize(mu_q, labels, anchors, args.anchor_scale)
            s["run"] = i
            s["seed"] = args.seed + i - 1
            run_summaries.append(s)
            csv_rows.append(
                [
                    label,
                    i,
                    s["seed"],
                    s["mean_abs_cos_offdiag"],
                    s["max_abs_cos_offdiag"],
                    s["min_dist_offdiag"],
                    s["mean_diag_align"],
                    s["max_offdiag_align"],
                    s["within_scatter"],
                    s["between_scatter"],
                    s["fisher_ratio"],
                    s["center_norm_mean"],
                    s["mean_anchor_dist"],
                ]
            )
            print(
                f"[{label}] run{i} (seed {s['seed']}) mean|cos_off|={s['mean_abs_cos_offdiag']:.4f} "
                f"diag_align={s['mean_diag_align']:.4f} max_off_align={s['max_offdiag_align']:.4f} "
                f"fisher={s['fisher_ratio']:.2f} ‖c̄‖={s['center_norm_mean']:.3f} "
                f"锚点距={s['mean_anchor_dist']:.3f}"
            )

        cr = cross_run_summary(run_summaries, EXTRA_SCALAR_KEYS)
        results[label] = {
            "dataset": dataset,
            "backbone": backbone,
            "model": model_name,
            "runs": run_summaries,
            "cross_run": cr,
        }
        print(
            f"[汇总] {label}: mean|cos_off|={cr['mean_abs_cos_offdiag_mean']:.4f}"
            f"±{cr['mean_abs_cos_offdiag_std']:.4f} | diag_align={cr['mean_diag_align_mean']:.4f}"
            f"±{cr['mean_diag_align_std']:.4f} | fisher={cr['fisher_ratio_mean']:.2f}"
            f"±{cr['fisher_ratio_std']:.2f} | ‖c̄‖={cr['center_norm_mean_mean']:.3f} | "
            f"锚点距={cr['mean_anchor_dist_mean']:.3f} | 跨run余弦std={cr['mean_pairwise_cos_std']:.4f}"
        )

    # 对照表：判据是 OPB 类中心 Gram 近对角、自锚对齐高、Fisher 比高、‖c̄‖≈a
    print("\n[对照] 各模型跨 run 汇总")
    header = f"{'model':<20} {'mean|cos_off|':<18} {'diag_align':<14} {'fisher':<12} {'‖c̄‖':<10} {'锚点距':<10} {'跨run余弦std':<14}"
    print(header)
    for label, res in results.items():
        cr = res["cross_run"]
        print(
            f"{label:<20} {cr['mean_abs_cos_offdiag_mean']:.4f}±{cr['mean_abs_cos_offdiag_std']:.4f}"
            f"{'':<4} {cr['mean_diag_align_mean']:.4f}±{cr['mean_diag_align_std']:.4f} "
            f"{cr['fisher_ratio_mean']:.2f}±{cr['fisher_ratio_std']:.2f} "
            f"{cr['center_norm_mean_mean']:.3f} {cr['mean_anchor_dist_mean']:.3f} "
            f"{cr['mean_pairwise_cos_std']:.4f}"
        )

    out_root = ROOT / args.results_dir
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "runs": args.runs,
            "anchor_scale": args.anchor_scale,
            "n_test": args.n_test,
            "near_pair_threshold": 0.5,
            "seed_base": args.seed,
        },
        **results,
    }
    json_path = out_root / "posterior_geometry.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "posterior_geometry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "run",
                "seed",
                "mean_abs_cos_offdiag",
                "max_abs_cos_offdiag",
                "min_dist_offdiag",
                "mean_diag_align",
                "max_offdiag_align",
                "within_scatter",
                "between_scatter",
                "fisher_ratio",
                "center_norm_mean",
                "mean_anchor_dist",
            ]
        )
        w.writerows(csv_rows)

    _setup_rc()
    plot_cosine_heatmaps(
        results,
        out_root / "posterior_geometry_centergram.png",
        suptitle="后验类中心两两余弦 Gram 矩阵（c_k = 各类后验均值中心；非对角 = 0 表示类中心正交）",
    )
    plot_cosine_heatmaps(
        results,
        out_root / "posterior_geometry_align.png",
        suptitle="类中心-锚点对齐矩阵 cos(c_k, 锚点_j)（对角 = 自锚对齐、非对角 = 串扰）",
        key="align",
    )
    plot_scatter_bars(results, out_root / "posterior_geometry_scatter.png")
    plot_norms(
        results,
        out_root / "posterior_geometry_norms.png",
        args.anchor_scale,
        ylabel="类中心模长 ‖c_k‖",
        title="各类后验中心模长（OPB 期望 ≈ a）",
    )

    print(f"\n结果已保存：{json_path}")
    print(f"汇总表已保存：{csv_path}")
    print(f"图已保存：{out_root / 'posterior_geometry_centergram.png'}")
    print(f"          {out_root / 'posterior_geometry_align.png'}")
    print(f"          {out_root / 'posterior_geometry_scatter.png'}")
    print(f"          {out_root / 'posterior_geometry_norms.png'}")


if __name__ == "__main__":
    main()
