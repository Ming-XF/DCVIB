"""先验几何分析脚本（实验一：问题存在性验证——OPB 动机的前提）。

实证 paper/OPB.txt §1 的前提："CEB 的类别先验均值之间没有明确的几何关系，
损失不直接区分这些情况"。只支持 CEB 与 OPB（分类配置）两种模型：对每个
checkpoint 把全类别 one-hot 送入先验编码器得到各类先验均值表 μ_p(k)，计算
两两余弦 Gram 矩阵、两两距离、逐类模长与跨 seed（run）几何一致性。

纯权重分析：不需要加载任何数据集，CPU 即可。OPB 的先验均值走与训练前向
相同的 qr_anchor_table 路径，其 Gram 应严格为对角 1 / 非对角 0、两两距离
恒为 √2·a——兼作 OPB 实现正确性的检验；CEB 的 Gram 则应表现为非零、任意、
跨 run 不一致（前提成立）。若 CEB 意外表现出结构性与跨 run 一致性，前提被
削弱，如实报告。

用法（目录须解析为分类 CEB/OPB 配置，checkpoint 用默认架构参数训练；
--anchor-scale 默认 4.0，与默认训练一致）：

    python prior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb
    python prior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb \\
        --runs 5 --anchor-scale 8 --results-dir pri_results

v1 支持 mnist/cora 目录（其他数据集的模型构建需要数据集维度，直接报错）。
输出：{results-dir}/prior_geometry.csv（长表：每个 (config, run) 一行标量指标，
矩阵类数据量大不进 csv）、prior_geometry.json（每 run 全部矩阵与跨 run 统计，
矩阵仅存于此）与三张图（余弦热力图网格 / 非对角余弦与距离分布 / 逐类模长箱线）。
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from adv_eval import _run_index, parse_dir_name
from model.mlp.utils import qr_anchor_table
from train import build_model, build_parser

ROOT = Path(__file__).resolve().parent

# 近重合类别对判定阈值（两两距离小于该值视为几乎共用的先验均值）
NEAR_PAIR_THRESHOLD = 0.5

# 画图配色（dataviz 参考调色板，浅色面）：CEB=槽位1蓝、OPB=槽位2橙
SERIES_COLORS = {"ceb": "#2a78d6", "opb": "#eb6834"}
# 发散色带（余弦 -1..1）：红极 #e34948 ↔ 中性灰 #f0efec ↔ 蓝极 #2a78d6
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "prior_diverging", ["#e34948", "#f0efec", "#2a78d6"]
)
INK = "#0b0b0b"          # 主墨色
SECONDARY = "#52514e"    # 次墨色（轴标签）
MUTED = "#898781"        # 弱化色（刻度）
BASELINE = "#c3c2b7"     # 轴线


def _setup_rc():
    """按浅色面设置 matplotlib 全局样式（面 #fcfcfb / 页 #f9f9f7）。"""
    plt.rcParams.update(
        {
            "figure.facecolor": "#f9f9f7",
            "savefig.facecolor": "#f9f9f7",
            "axes.facecolor": "#fcfcfb",
            "text.color": INK,
            "axes.labelcolor": SECONDARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            # Noto Sans CJK JP 为 ttc 在 matplotlib 中的注册名（含统一表意文字，
            # 中文标签正常渲染）；DejaVu Sans 兜底拉丁字符
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "legend.frameon": False,
        }
    )


def extract_prior_means(model, model_name, anchor_scale):
    """返回 (M: (K, d) CPU 张量, source: str)。

    CEB：先验编码器对全类别 one-hot 的均值块（学习表）；
    OPB：同 CEB 后走与训练前向相同的 qr_anchor_table 得到正交锚点表。
    回归配置（continuous_y）没有类别先验表，直接报错。
    """
    if getattr(model, "continuous_y", False):
        raise ValueError("回归配置没有类别先验表，实验一仅支持分类 CEB/OPB")
    # eye 须在模型所在设备上（实验二模型挂在 CUDA）
    eye = torch.eye(model.num_classes, device=next(model.parameters()).device)
    prior_out = model.prior_net(eye)  # (K, 2d)
    mu_raw = prior_out.chunk(2, dim=1)[0]  # (K, d)
    if model_name == "opb":
        M = qr_anchor_table(mu_raw.t(), anchor_scale)
        source = f"qr_anchor_table(prior_net(eye), a={anchor_scale:g})"
    else:  # ceb
        M = mu_raw
        source = "prior_net(eye) 学习表"
    return M.detach(), source


def pairwise_gram(M):
    """两两余弦矩阵 G (K, K)；零向量类别整行记 0（与自身亦记 0，避免 NaN）。"""
    K = M.size(0)
    norms = M.norm(dim=1)
    G = torch.zeros(K, K)
    ok = norms > 1e-8
    if ok.all():
        Mn = M / norms.unsqueeze(1)
        G = Mn @ Mn.t()
    elif ok.any():
        # 布尔掩码 G[ok] 返回副本、无法原地赋值，须用整数索引写回
        idx = ok.nonzero(as_tuple=False).flatten()
        Mn = M[ok] / norms[ok].unsqueeze(1)
        G[idx[:, None], idx[None, :]] = Mn @ Mn.t()
    return G


def offdiag(t):
    """取方阵上三角非对角元素（i<j）。"""
    idx = torch.triu_indices(t.size(0), t.size(0), offset=1)
    return t[idx[0], idx[1]]


def summarize(M, anchor_scale):
    """计算单个 run 的全部几何量，返回可 JSON 序列化的 dict。"""
    G = pairwise_gram(M)
    D = torch.cdist(M, M)
    g_off = offdiag(G)  # 带符号余弦
    d_off = offdiag(D)
    return {
        "gram": G.tolist(),
        "dist": D.tolist(),
        "norms": M.norm(dim=1).tolist(),
        "cos_offdiag": g_off.tolist(),
        "dist_offdiag": d_off.tolist(),
        "mean_abs_cos_offdiag": g_off.abs().mean().item(),
        "max_abs_cos_offdiag": g_off.abs().max().item(),
        "min_dist_offdiag": d_off.min().item(),
        "max_dist_offdiag": d_off.max().item(),
        "norm_std": M.norm(dim=1).std().item(),
        "n_near_pairs": int((d_off < NEAR_PAIR_THRESHOLD).sum().item()),
        "theory_dist": math.sqrt(2.0) * anchor_scale,
    }


def cross_run_summary(run_summaries, extra_keys=()):
    """跨 run（=跨 seed）几何一致性统计。

    mean_pairwise_cos_std：每个 (k,j) 类别对的余弦在 run 间的标准差再对类别对
    平均——"几何一致性指数"（0 = 各 run 几何完全一致）。其余为各汇总标量
    跨 run 的均值/标准差；extra_keys 为调用方追加的标量键（同样统计均值/标准差）。
    """
    grams = torch.stack([torch.tensor(s["gram"]) for s in run_summaries])  # (R,K,K)
    per_pair_std = offdiag(grams.std(dim=0))
    scalar_keys = [
        "mean_abs_cos_offdiag",
        "max_abs_cos_offdiag",
        "min_dist_offdiag",
        "max_dist_offdiag",
        "norm_std",
        "n_near_pairs",
        *extra_keys,
    ]
    agg = {k: torch.tensor([s[k] for s in run_summaries], dtype=torch.float64) for k in scalar_keys}
    out = {
        "mean_pairwise_cos_std": per_pair_std.mean().item(),
        "max_pairwise_cos_std": per_pair_std.max().item(),
    }
    for k, v in agg.items():
        out[f"{k}_mean"] = v.mean().item()
        out[f"{k}_std"] = v.std().item()
    return out


def _series_color(label):
    return SERIES_COLORS[parse_dir_name(label)[2]]


def plot_cosine_heatmaps(results, path, suptitle="类别先验均值两两余弦 Gram 矩阵（非对角 = 0 表示类别先验正交）", key="gram"):
    """K×K 余弦矩阵热力图网格（行=模型、列=run；发散色带，0 为中性灰）。

    key 指定读取每 run 的哪个 K×K 矩阵（默认 gram；实验二复用为 align 对齐矩阵）。
    """
    labels = list(results)
    runs = results[labels[0]]["runs"]
    K = len(runs[0][key])
    fig, axes = plt.subplots(
        len(labels), len(runs), figsize=(2.3 * len(runs), 2.7 * len(labels)), squeeze=False
    )
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    for row, label in enumerate(labels):
        for col, s in enumerate(results[label]["runs"]):
            ax = axes[row, col]
            im = ax.imshow(s[key], cmap=DIVERGING_CMAP, norm=norm, aspect="equal")
            ax.set_xticks(range(K), range(K))
            ax.set_yticks(range(K), range(K))
            ax.tick_params(length=0, labelsize=6)
            ax.grid(False)
            if K <= 10:
                for i in range(K):
                    for j in range(K):
                        v = s[key][i][j]
                        color = "#ffffff" if abs(v) > 0.45 else INK
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color=color)
            if col == 0:
                ax.set_ylabel(f"{label}\n类别 k", fontsize=8)
            if row == 0:
                ax.set_title(f"run{s['run']} (seed {s['seed']})", fontsize=8)
            if row == len(labels) - 1:
                ax.set_xlabel("类别 j", fontsize=8)
    fig.suptitle(suptitle, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="余弦")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_offdiag_dists(results, path):
    """图 2：非对角余弦与两两距离的分布（跨 run 合并；虚线为 OPB 理论参考）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, res in results.items():
        cos_all = [v for s in res["runs"] for v in s["cos_offdiag"]]
        dist_all = [v for s in res["runs"] for v in s["dist_offdiag"]]
        color = _series_color(label)
        ax1.hist(cos_all, bins=40, density=True, alpha=0.45, color=color,
                 edgecolor=color, linewidth=0.8, label=label)
        ax2.hist(dist_all, bins=40, density=True, alpha=0.45, color=color,
                 edgecolor=color, linewidth=0.8, label=label)
    ax1.axvline(0.0, color=SECONDARY, linestyle="--", linewidth=1.5)
    # theory_dist 在 summarize 中已按 √2·a 计算，这里直接取参考值
    theory = next(iter(results.values()))["runs"][0]["theory_dist"]
    ax2.axvline(theory, color=SECONDARY, linestyle="--", linewidth=1.5,
                label=f"√2·a = {theory:.3f}")
    ax1.set_xlabel("非对角余弦")
    ax2.set_xlabel("两两距离 ‖μ_p(i) − μ_p(j)‖")
    ax1.set_ylabel("密度（跨 run 合并）")
    ax1.set_title("类别先验两两余弦（0 = 正交）", fontsize=9)
    ax2.set_title("类别先验两两距离", fontsize=9)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_norms(results, path, anchor_scale, ylabel="先验均值模长 ‖μ_p(k)‖", title="各类先验均值模长（OPB 理论值恒为 a）"):
    """逐类模长箱线（跨 run；虚线为锚点尺度 a；ylabel/title 供实验二复用）。"""
    labels = list(results)
    K = len(results[labels[0]]["runs"][0]["norms"])
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * K), 4))
    width = 0.35
    handles = []
    for m_i, label in enumerate(labels):
        color = _series_color(label)
        per_class = [[s["norms"][k] for s in results[label]["runs"]] for k in range(K)]
        positions = [k + (m_i - (len(labels) - 1) / 2) * width for k in range(K)]
        bp = ax.boxplot(
            per_class, positions=positions, widths=width * 0.85, patch_artist=True,
            medianprops=dict(color=color, linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=color, markersize=2.5,
                            markeredgecolor="none", alpha=0.6),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#fcfcfb")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)
        handles.append(Patch(facecolor="#fcfcfb", edgecolor=color, label=label))
    ax.axhline(anchor_scale, color=SECONDARY, linestyle="--", linewidth=1.5,
               label=f"a = {anchor_scale:g}")
    handles.append(plt.Line2D([], [], color=SECONDARY, linestyle="--", linewidth=1.5,
                              label=f"a = {anchor_scale:g}"))
    ax.set_xticks(range(K), range(K))
    ax.set_xlabel("类别 k")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(handles=handles, fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_analysis_parser():
    parser = build_parser()
    parser.add_argument(
        "--model-dirs", nargs="+", required=True,
        help="已训练配置的 checkpoint 目录列表（每个目录内含 {stem}_run{i}.pt；"
        "目录名须解析为分类 CEB/OPB 配置，v1 仅支持 mnist/cora 数据集目录）",
    )
    parser.add_argument(
        "--results-dir", type=str, default="pri_results",
        help="结果输出目录（默认 pri_results）",
    )
    return parser


def main():
    parser = build_analysis_parser()
    args = parser.parse_args()

    results = {}
    csv_rows = []
    for d in (Path(p) for p in args.model_dirs):
        label = d.name
        dataset, backbone, model_name = parse_dir_name(label)
        if model_name not in ("ceb", "opb"):
            parser.error(f"目录 '{label}' 模型为 '{model_name}'，实验一仅支持 ceb/opb")
        if dataset not in ("mnist", "cora"):
            parser.error(
                f"目录 '{label}' 数据集为 '{dataset}'，实验一 v1 仅支持 mnist/cora "
                "（其他数据集的模型构建需要数据集维度）"
            )
        ckpts = sorted(d.glob("*_run*.pt"), key=_run_index)
        if len(ckpts) < args.runs:
            parser.error(
                f"目录 '{d}' 只找到 {len(ckpts)} 个 '*_run*.pt'，需要 --runs={args.runs} 个"
            )
        ckpts = ckpts[: args.runs]

        # 目录名解析出的架构重建模型；task 按目录名自动推导（纯权重分析、不加载数据）
        dir_args = argparse.Namespace(**vars(args))
        dir_args.task = dataset
        dir_args.model = model_name
        dir_args.backbone = backbone
        model = build_model(parser, dir_args)
        print(f"[加载] {label} → backbone={backbone} model={model_name}，checkpoint {len(ckpts)} 个")

        run_summaries = []
        for i, ckpt in enumerate(ckpts, 1):
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
            model.eval()
            M, source = extract_prior_means(model, model_name, args.anchor_scale)
            s = summarize(M, args.anchor_scale)
            s["run"] = i
            s["seed"] = args.seed + i - 1
            run_summaries.append(s)
            # CSV 长表只存标量指标（矩阵不进 csv，见 JSON 与图）
            csv_rows.append(
                [
                    label,
                    i,
                    s["seed"],
                    s["mean_abs_cos_offdiag"],
                    s["max_abs_cos_offdiag"],
                    s["min_dist_offdiag"],
                    s["max_dist_offdiag"],
                    s["norm_std"],
                    s["n_near_pairs"],
                ]
            )
            print(
                f"[{label}] run{i} (seed {s['seed']}) mean|cos_off|={s['mean_abs_cos_offdiag']:.4f} "
                f"minD={s['min_dist_offdiag']:.4f} norm_std={s['norm_std']:.4f} "
                f"近重合对={s['n_near_pairs']}"
            )

        cr = cross_run_summary(run_summaries)
        results[label] = {
            "dataset": dataset,
            "backbone": backbone,
            "model": model_name,
            "source": source,
            "runs": run_summaries,
            "cross_run": cr,
        }
        print(
            f"[汇总] {label} ({source}): mean|cos_off|={cr['mean_abs_cos_offdiag_mean']:.4f}"
            f"±{cr['mean_abs_cos_offdiag_std']:.4f} | minD={cr['min_dist_offdiag_mean']:.4f}"
            f"±{cr['min_dist_offdiag_std']:.4f} | 跨run余弦std={cr['mean_pairwise_cos_std']:.4f}"
        )

    # 对照表：前提成立的判据是 CEB 非对角余弦非零且跨 run 不一致，OPB 恒 0
    print("\n[对照] 各模型跨 run 汇总")
    print(f"{'model':<20} {'mean|cos_off|':<18} {'minD':<18} {'跨run余弦std':<14}")
    for label, res in results.items():
        cr = res["cross_run"]
        print(
            f"{label:<20} {cr['mean_abs_cos_offdiag_mean']:.4f}±{cr['mean_abs_cos_offdiag_std']:.4f}"
            f"{'':<4} {cr['min_dist_offdiag_mean']:.4f}±{cr['min_dist_offdiag_std']:.4f}"
            f"{'':<4} {cr['mean_pairwise_cos_std']:.4f}"
        )

    out_root = ROOT / args.results_dir
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "runs": args.runs,
            "anchor_scale": args.anchor_scale,
            "near_pair_threshold": NEAR_PAIR_THRESHOLD,
            "seed_base": args.seed,
        },
        **results,
    }
    json_path = out_root / "prior_geometry.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = out_root / "prior_geometry.csv"
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
                "max_dist_offdiag",
                "norm_std",
                "n_near_pairs",
            ]
        )
        w.writerows(csv_rows)

    _setup_rc()
    plot_cosine_heatmaps(results, out_root / "prior_geometry_cosine.png")
    plot_offdiag_dists(results, out_root / "prior_geometry_offdiag.png")
    plot_norms(results, out_root / "prior_geometry_norms.png", args.anchor_scale)
    print(f"\n结果已保存：{json_path}")
    print(f"汇总表已保存：{csv_path}")
    print(f"图已保存：{out_root / 'prior_geometry_cosine.png'}")
    print(f"          {out_root / 'prior_geometry_offdiag.png'}")
    print(f"          {out_root / 'prior_geometry_norms.png'}")


if __name__ == "__main__":
    main()
