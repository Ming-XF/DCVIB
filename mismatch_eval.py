"""真实 benchmark 直接失配分析（codex_output2.txt 方案）：连接几何机制与
无条件分离命题 prop:pgib-separation。

不做任何重训：从 output/compression_eval/ 的既有 checkpoint（目录解析与模型
重建口径同 compression_eval.py）在固定测试划分上直接测量——
  分类（ImageNet-100 MLP, OPB）：逐样本 d_i = ‖μ_i − a·q_{y_i}‖²/(2τ²_max)、
    D、逐类 D_k、后验类中心 m_k 与中心误差 e_k = ‖m_k − a·q_k‖、全部类别对的
    S_ij = ‖m_i−m_j‖ 与原始下界 LB_ij = √2·a − √(2τ²_max·D_i) − √(2τ²_max·D_j)
    （Jensen + 三角不等式，有限样本经验分布上 slack ≥ −1e-6 严格成立，仅浮点
    误差；违反即记 failed 并排查 τ/标签索引/锚点匹配）。分离比值截断
    max(0, LB) 仅作图时使用并注明。
  回归（California Housing MLP, EPB）：逐样本 d_i = ‖μ_i − ρ·ỹ_i·u‖²/(2τ²_max)、
    轴向/离轴分解（两项与 D 数值一致）、20 等宽 bins（<30 样本与相邻合并、
    记录边界）与 bin 对有限样本修正下界
    LB_bc = ρ|ȳ_b−ȳ_c| − √(2τ²_max·D_b) − √(2τ²_max·D_c) − ρ(diam_b+diam_c)
    （附录 rem:pgib-binning 的量化修正）。结果定位为 finite-sample diagnostic，
    不称定理统计验证。
CEB 只作描述性参照：报告 posterior-to-learned-reference 均值失配与 KL 三项，
不套用任何分离下界（其参考不正交、不满足固定分离条件，方案 §7 红线）。

τ²_max 定义：checkpoint 未存 qualified-family 上界，取实现的最大先验方差——
分类 max over (K,d) exp(logvar_p 表)、回归 max over 测试样本 exp(prior_logvar_net(ỹ))；
与 KL 分解的逐类真实方差分开报告（d_i 用 τ²_max、kl_mean 用真实 var_p）。
ỹ 为训练集拟合 MinMax 的归一化标签（数据管道既有口径，模型内不二次标准化）。

输出（output/mismatch_eval/）：
    mismatch_eval_imagenet100_mlp.csv   每 (dir, run) 一行聚合（方案 §6 字段 + 扩展）
    mismatch_eval_california_mlp.csv    同上（回归字段）
    mismatch_eval_imagenet100_mlp_classdetail.csv   逐类明细（n_k/D_k/e_k）
    mismatch_eval_california_mlp_bindetail.csv      逐 bin 明细（边界/n_b/ȳ_b/D_b/diam）
    detail_imagenet100_mlp/{dir}_run{i}.npz         复核下界用（centers/D_k/e_k/counts）
    detail_california_mlp/{dir}_run{i}.npz          （m_b/ybar_b/D_b/diam_b/n_b/edges/u）
    missing_matrix.csv                  缺失 checkpoint 显式矩阵
论文图由 mismatch_eval_plot.py 生成。

用法：
    python mismatch_eval.py --workers 32
"""

import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression_eval import (
    DATASET_TO_TASK,
    REGRESSION_TASKS,
    _prior_table_cls,
    parse_combo_dir,
)
from datasets.datasets import get_california_dataloaders, get_imagenet100_dataloaders
from model.mlp.utils import flatten
from train import build_model, build_parser, run_model

ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT / "output" / "compression_eval"
OUT_ROOT = ROOT / "output" / "mismatch_eval"

MC = 10           # 随机路径预测采样数
NBINS = 20        # 回归分箱数（方案 §4）
MIN_BIN_N = 30    # 少于该样本数的 bin 与相邻合并
SLACK_TOL = 1e-6  # slack ≥ −tol 的数值 sanity 界

CLS_HEADER = [
    "dataset", "seed", "beta", "anchor_scale", "model",
    "d_mean", "d_class_min", "d_class_max",
    "center_error_mean", "center_error_max", "jensen_ref_mean",
    "center_distance_mean", "center_distance_min",
    "lower_bound_mean", "lower_bound_min", "positive_bound_fraction",
    "slack_p05", "slack_p50", "slack_p95",
    "kl_mean", "kl_cov", "kl_total",
    "deterministic_metric", "mc10_metric", "ce", "tau2max",
    "n_classes_present", "failed",
]
REG_HEADER = [
    "dataset", "seed", "beta", "anchor_scale", "model",
    "d_mean", "d_axial", "d_offaxis", "axial_slope",
    "bin_count", "bin_distance_mean", "lower_bound_mean",
    "positive_bound_fraction", "slack_p05", "slack_p50", "slack_p95",
    "kl_mean", "kl_cov", "kl_total",
    "deterministic_metric", "mc10_metric", "ce", "tau2max", "failed",
]
CLS_DETAIL_HEADER = ["dataset", "seed", "beta", "anchor_scale", "model",
                     "class", "n_k", "d_k", "e_k"]
REG_DETAIL_HEADER = ["dataset", "seed", "beta", "anchor_scale", "model",
                     "bin", "n_b", "ybar_b", "d_b", "diam_b", "bin_start", "bin_end"]


def _pct(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


@torch.no_grad()
def _collect(model, loader, device):
    """测试集整遍收集：x、y、μ、KL 均值/方差失配项、μ 路径 logits。

    前向在 device（GPU）上执行、逐批回 CPU 累积（imagenet100 的 16384 维
    输入在 CPU 单线程上内存带宽受限，GPU 提速 ~100×）。
    """
    model.eval()
    xs, ys, mus, klm_s, klv_s, logits_s = [], [], [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h = model.encoder(flatten(x))
        mu = model.mu_head(h)
        logvar = model.logvar_head(h)
        if getattr(model, "continuous_y", False):
            yf = y.float().unsqueeze(-1)
            if model.__class__.__name__ == "OPB":
                u = F.normalize(model.prior_direction.weight.squeeze(-1), dim=0)
                mu_p = model.anchor_scale * yf * u.unsqueeze(0)
                logvar_p = model.prior_logvar_net(yf)
            else:  # CEB：学习条件参考（连续 y）
                mu_p, logvar_p = model.prior_net(yf).chunk(2, dim=1)
        else:
            mu_p_all, logvar_p_all = _prior_table_cls(model, device)
            mu_p = mu_p_all[y]
            logvar_p = logvar_p_all[y]
        var = logvar.exp()
        var_p = logvar_p.exp()
        klm = (0.5 * (mu - mu_p).pow(2) / var_p).sum(1)
        klv = 0.5 * (var / var_p - 1 - logvar + logvar_p).sum(1)
        logits = run_model(model, x, None, stochastic=False)[0]
        xs.append(x.cpu())
        ys.append(y.cpu())
        mus.append(mu.cpu())
        klm_s.append(klm.cpu())
        klv_s.append(klv.cpu())
        logits_s.append(logits.cpu())
    return (torch.cat(xs), torch.cat(ys), torch.cat(mus),
            torch.cat(klm_s), torch.cat(klv_s), torch.cat(logits_s))


@torch.no_grad()
def eval_cls(model, loader, device, mc=MC):
    """分类直接失配度量：D/D_k/中心误差/全部类别对 S 与原始 LB/slack。"""
    x_all, y, mu, klm, klv, logits = _collect(model, loader, device)
    K = model.num_classes
    mu_p_all, logvar_p_all = _prior_table_cls(model, device)  # (K,d)
    tau2max = float(logvar_p_all.exp().max().item())

    d_i = ((mu - mu_p_all[y].cpu()).pow(2).sum(1) / (2.0 * tau2max))  # (n,)
    counts = torch.bincount(y, minlength=K)
    present = (counts > 0).nonzero(as_tuple=True)[0].tolist()
    centers = torch.stack([mu[y == k].mean(0) for k in present])  # (P,d)
    anchors_p = mu_p_all[present].cpu()
    D_k = torch.stack([d_i[y == k].mean() for k in present])
    e_k = (centers - anchors_p).norm(dim=1)
    jensen = (2.0 * tau2max * D_k).sqrt()

    S = torch.cdist(centers, centers)
    is_opb = model.__class__.__name__ == "OPB"
    if is_opb:
        LB = math.sqrt(2.0) * model.anchor_scale - jensen.unsqueeze(0) - jensen.unsqueeze(1)
        triu = torch.triu(torch.ones(len(present), len(present), dtype=torch.bool), 1)
        S_p, LB_p = S[triu], LB[triu]
        slack = (S_p - LB_p).numpy()
    else:
        triu = torch.triu(torch.ones(len(present), len(present), dtype=torch.bool), 1)
        S_p = S[triu]
        LB_p = None
        slack = np.array([])

    ce = F.cross_entropy(logits, y).item()
    acc_det = (logits.argmax(1) == y).float().mean().item()
    xg = x_all.to(device)
    probs = torch.zeros(logits.size(0), K, device=device)
    for _ in range(mc):
        lm = run_model(model, xg, None, stochastic=True)[0]
        probs += lm.softmax(1)
    probs = (probs / mc).cpu()
    acc_mc = (probs.argmax(1) == y).float().mean().item()

    failed = ""
    if is_opb and float(slack.min()) < -SLACK_TOL:
        failed = f"slack<tol: {float(slack.min()):.3e}"

    metrics = {
        "d_mean": float(d_i.mean().item()),
        "d_class_min": float(D_k.min().item()), "d_class_max": float(D_k.max().item()),
        "center_error_mean": float(e_k.mean().item()), "center_error_max": float(e_k.max().item()),
        "jensen_ref_mean": float(jensen.mean().item()) if is_opb else "",
        "center_distance_mean": float(S_p.mean().item()), "center_distance_min": float(S_p.min().item()),
        "lower_bound_mean": float(LB_p.mean().item()) if is_opb else "",
        "lower_bound_min": float(LB_p.min().item()) if is_opb else "",
        "positive_bound_fraction": float((LB_p > 0).float().mean().item()) if is_opb else "",
        "slack_p05": _pct(slack, 5) if is_opb else "",
        "slack_p50": _pct(slack, 50) if is_opb else "",
        "slack_p95": _pct(slack, 95) if is_opb else "",
        "kl_mean": float(klm.mean().item()), "kl_cov": float(klv.mean().item()),
        "kl_total": float((klm + klv).mean().item()),
        "deterministic_metric": acc_det, "mc10_metric": acc_mc, "ce": ce,
        "tau2max": tau2max, "n_classes_present": len(present), "failed": failed,
    }
    detail = {
        "centers": centers.numpy(), "D_k": D_k.numpy(), "e_k": e_k.numpy(),
        "counts": counts.numpy(), "tau2max": tau2max,
    }
    return metrics, detail


@torch.no_grad()
def eval_reg(model, loader, device, mc=MC):
    """回归直接失配度量 + 有限样本分箱（方案 §4）。"""
    x_all, yt, mu, klm, klv, logits = _collect(model, loader, device)
    yt = yt.float()
    preds = logits.squeeze(-1)
    n = len(yt)

    is_opb = model.__class__.__name__ == "OPB"
    yf = yt.to(device).unsqueeze(-1)
    if is_opb:
        u = F.normalize(model.prior_direction.weight.squeeze(-1), dim=0)  # (d,)
        rho = model.anchor_scale
        logvar_p_all = model.prior_logvar_net(yf)
        tau2max = float(logvar_p_all.exp().max().item())
        mu_p = rho * yf * u.unsqueeze(0)
    else:  # CEB：学习条件参考
        mu_p_all, logvar_p_all = model.prior_net(yf).chunk(2, dim=1)
        tau2max = float(logvar_p_all.exp().max().item())
        mu_p = mu_p_all
    d_i = ((mu - mu_p.cpu()).pow(2).sum(1) / (2.0 * tau2max))  # (n,)
    d_mean = float(d_i.mean().item())

    if is_opb:
        u_cpu = u.cpu()
        axial = mu @ u_cpu
        axial_err = axial - rho * yt
        offaxis = mu - axial.unsqueeze(1) * u_cpu.unsqueeze(0)
        d_axial = float((axial_err.pow(2).sum() / (2.0 * tau2max * n)).item())
        d_offaxis = float((offaxis.pow(2).sum() / (2.0 * tau2max * n)).item())
        # float32 逐样本累加的相对误差 ~1e-7，容差取相对 1e-6
        if abs(d_axial + d_offaxis - d_mean) > 1e-6 * max(1.0, d_mean):
            raise RuntimeError(f"轴向分解与 D 不一致：{abs(d_axial + d_offaxis - d_mean):.3e}")
        my = float(yt.mean().item())
        slope = float(((axial - axial.mean()) * (yt - my)).sum() /
                      ((yt - my).pow(2)).sum().item()) / rho  # 归一化轴向斜率
    else:
        u = None
        axial = axial_err = offaxis = None
        d_axial = d_offaxis = slope = ""

    # 20 等宽 bins，<MIN_BIN_N 的 bin 与相邻合并（尾部并入左邻），记录边界
    if is_opb:
        edges = torch.linspace(float(yt.min()), float(yt.max()), NBINS + 1).tolist()
        assign = torch.bucketize(yt, torch.tensor(edges[1:-1]))
        bins = []  # (start, end, mask)
        b = 0
        while b < NBINS:
            start = edges[b]
            mask = assign == b
            end = edges[b + 1]
            b2 = b + 1
            while mask.sum() < MIN_BIN_N and b2 < NBINS:
                mask = mask | (assign == b2)
                end = edges[b2 + 1]
                b2 += 1
            if mask.sum() < MIN_BIN_N and bins:
                prev = bins[-1]
                bins[-1] = (prev[0], end, prev[2] | mask)
            elif mask.sum() > 0:
                bins.append((start, end, mask))
            b = b2
        if not bins:
            raise RuntimeError("无有效分箱")

        ybar_b, m_b, D_b, diam_b, n_b = [], [], [], [], []
        for start, end, mask in bins:
            ybar_b.append(float(yt[mask].mean().item()))
            m_b.append(mu[mask].mean(0))
            D_b.append(float(d_i[mask].mean().item()))
            diam_b.append(float(yt[mask].max().item() - yt[mask].min().item()))
            n_b.append(int(mask.sum().item()))
        m_b = torch.stack(m_b)
        B = len(bins)

        S = torch.cdist(m_b, m_b)
        yb = torch.tensor(ybar_b)
        corr = (2.0 * tau2max * torch.tensor(D_b)).sqrt() + rho * torch.tensor(diam_b)
        LB = rho * (yb.unsqueeze(0) - yb.unsqueeze(1)).abs() - corr.unsqueeze(0) - corr.unsqueeze(1)
        triu = torch.triu(torch.ones(B, B, dtype=torch.bool), 1)
        S_p, LB_p = S[triu], LB[triu]
        slack = (S_p - LB_p).numpy()
        B_res = {"S_mean": float(S_p.mean().item()),
                 "LB_mean": float(LB_p.mean().item()),
                 "pos_frac": float((LB_p > 0).float().mean().item()),
                 "slack": slack, "B": B}
    else:
        B_res = {"S_mean": "", "LB_mean": "", "pos_frac": "", "slack": np.array([]), "B": ""}

    mse = F.mse_loss(preds, yt).item()
    ss_res = ((preds - yt) ** 2).sum().item()
    ss_tot = ((yt - yt.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    xg = x_all.to(device)
    pmc = torch.zeros(n, device=device)
    for _ in range(mc):
        lm = run_model(model, xg, None, stochastic=True)[0].squeeze(-1)
        pmc += lm
    pmc = (pmc / mc).cpu()
    ss_res_mc = ((pmc - yt) ** 2).sum().item()
    r2_mc = 1 - ss_res_mc / ss_tot if ss_tot > 0 else float("nan")

    failed = ""
    if is_opb and float(B_res["slack"].min()) < -SLACK_TOL:
        failed = f"slack<tol: {float(B_res['slack'].min()):.3e}"

    metrics = {
        "d_mean": d_mean, "d_axial": d_axial, "d_offaxis": d_offaxis,
        "axial_slope": slope, "bin_count": B_res["B"],
        "bin_distance_mean": B_res["S_mean"],
        "lower_bound_mean": B_res["LB_mean"],
        "positive_bound_fraction": B_res["pos_frac"],
        "slack_p05": _pct(B_res["slack"], 5), "slack_p50": _pct(B_res["slack"], 50),
        "slack_p95": _pct(B_res["slack"], 95),
        "kl_mean": float(klm.mean().item()), "kl_cov": float(klv.mean().item()),
        "kl_total": float((klm + klv).mean().item()),
        "deterministic_metric": r2, "mc10_metric": r2_mc, "ce": mse,
        "tau2max": tau2max, "failed": failed,
    }
    detail = {
        "m_b": m_b.numpy() if is_opb else np.zeros((0, 0)),
        "ybar_b": np.array(ybar_b) if is_opb else np.array([]),
        "D_b": np.array(D_b) if is_opb else np.array([]),
        "diam_b": np.array(diam_b) if is_opb else np.array([]),
        "n_b": np.array(n_b) if is_opb else np.array([]),
        "bin_edges": [(float(s), float(e)) for s, e, _ in bins] if is_opb else [],
        "u": u.cpu().numpy() if is_opb else np.array([]),
        "tau2max": tau2max,
    }
    return metrics, detail


LOADERS = {}
PARSER = None
ARGS = None
DEVICE = "cpu"


def _init_worker(args, device):
    """spawn 子进程初始化：parser 含局部 lambda 不可 pickle，loaders 含 mmap，
    均在此重建（并行，每进程数秒）。"""
    global LOADERS, PARSER, ARGS, DEVICE
    PARSER = build_parser()
    ARGS = args
    DEVICE = device
    LOADERS = {
        "imagenet100": get_imagenet100_dataloaders(args.batch_size, args.data_dir),
        "california": get_california_dataloaders(args.batch_size, args.data_dir),
    }
    torch.set_num_threads(1)


def eval_task(task):
    """任务 = (dir_name, run)。返回 dict 供主进程写 CSV；npz 明细在进程内落盘。"""
    dir_name, run = task
    d = EVAL_ROOT / dir_name
    info = parse_combo_dir(dir_name)
    task_name, backbone, model_name, beta, anchor = info
    dataset = task_name
    task_label = DATASET_TO_TASK.get(task_name, task_name)
    is_reg = task_label in REGRESSION_TASKS
    ckpt = d / f"{dir_name}_run{run}.pt"
    if not ckpt.exists():
        return {"missing": True, "dir": dir_name, "run": run}
    seed = ARGS.seed + run - 1
    try:
        dir_args = argparse.Namespace(**vars(ARGS))
        dir_args.task = task_label
        dir_args.model = model_name
        dir_args.backbone = backbone
        dir_args.anchor_scale = anchor if anchor is not None else ARGS.anchor_scale
        dir_args.energy_classifier = (not is_reg) and model_name == "opb"
        dir_args.tied_head = is_reg and model_name == "opb"
        if dataset == "imagenet100":
            input_dim, feature_pool = LOADERS[dataset][3], LOADERS[dataset][4]
            model = build_model(PARSER, dir_args, imagenet100_input_dim=input_dim,
                                imagenet100_feature_pool=feature_pool)
        else:
            model = build_model(PARSER, dir_args)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=DEVICE))
        loader = LOADERS[dataset][2]
        out = {"missing": False, "dataset": dataset, "seed": seed, "beta": beta,
               "anchor": anchor, "model": model_name, "dir": dir_name, "run": run}
        if is_reg:
            m, detail = eval_reg(model, loader, DEVICE)
            row = [dataset, seed, beta, anchor, model_name,
                   m["d_mean"], m["d_axial"], m["d_offaxis"], m["axial_slope"],
                   m["bin_count"], m["bin_distance_mean"], m["lower_bound_mean"],
                   m["positive_bound_fraction"], m["slack_p05"], m["slack_p50"],
                   m["slack_p95"], m["kl_mean"], m["kl_cov"], m["kl_total"],
                   m["deterministic_metric"], m["mc10_metric"], m["ce"],
                   m["tau2max"], m["failed"]]
            out["row"] = row
            out["kind"] = "reg"
            det_rows = []
            if model_name == "opb":
                for b in range(m["bin_count"]):
                    s, e = detail["bin_edges"][b]
                    det_rows.append([dataset, seed, beta, anchor, model_name, b,
                                     int(detail["n_b"][b]), float(detail["ybar_b"][b]),
                                     float(detail["D_b"][b]), float(detail["diam_b"][b]), s, e])
            out["det_rows"] = det_rows
            np.savez(OUT_ROOT / "detail_california_mlp" / f"{dir_name}_run{run}.npz",
                     **detail)
        else:
            m, detail = eval_cls(model, loader, DEVICE)
            row = [dataset, seed, beta, anchor, model_name,
                   m["d_mean"], m["d_class_min"], m["d_class_max"],
                   m["center_error_mean"], m["center_error_max"], m["jensen_ref_mean"],
                   m["center_distance_mean"], m["center_distance_min"],
                   m["lower_bound_mean"], m["lower_bound_min"], m["positive_bound_fraction"],
                   m["slack_p05"], m["slack_p50"], m["slack_p95"],
                   m["kl_mean"], m["kl_cov"], m["kl_total"],
                   m["deterministic_metric"], m["mc10_metric"], m["ce"],
                   m["tau2max"], m["n_classes_present"], m["failed"]]
            out["row"] = row
            out["kind"] = "cls"
            det_rows = []
            if model_name == "opb":
                for i, k in enumerate(detail["counts"].nonzero()[0]):
                    det_rows.append([dataset, seed, beta, anchor, model_name, int(k),
                                     int(detail["counts"][k]), float(detail["D_k"][i]),
                                     float(detail["e_k"][i])])
            out["det_rows"] = det_rows
            np.savez(OUT_ROOT / "detail_imagenet100_mlp" / f"{dir_name}_run{run}.npz",
                     **detail)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out
    except Exception as e:
        row = [dataset, seed, beta, anchor, model_name] + [""] * (18 if is_reg else 23) + [str(e)]
        if "model" in dir() and model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"missing": False, "row": row, "kind": "reg" if is_reg else "cls",
                "det_rows": [], "failed": True}


def main():
    global LOADERS, PARSER, ARGS, DEVICE
    parser = build_parser()
    parser.add_argument("--eval-root", type=str, default=str(EVAL_ROOT))
    parser.add_argument("--workers", type=int, default=4,
                        help="GPU 评估进程数（imagenet100 每进程峰值显存 ~4GB，"
                        "8 并发会撑满 32GB 触发 OOM，默认 4）")
    args = parser.parse_args()
    PARSER = parser
    ARGS = args
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备：{DEVICE}（{args.workers} 进程，spawn）")
    # loaders 由各子进程在 initializer 内重建（spawn 无共享内存）

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "detail_imagenet100_mlp").mkdir(exist_ok=True)
    (OUT_ROOT / "detail_california_mlp").mkdir(exist_ok=True)

    tasks, missing = [], []
    for d in sorted(Path(args.eval_root).iterdir()):
        if not d.is_dir() or parse_combo_dir(d.name) is None:
            continue
        # 本试验只覆盖两个代表任务；其余数据集（agedb 等）的目录跳过
        if not d.name.startswith(("imagenet100_mlp_", "california_mlp_")):
            continue
        for run in range(1, 6):
            tasks.append((d.name, run))
            if not (d / f"{d.name}_run{run}.pt").exists():
                missing.append((d.name, run))

    print(f"评估矩阵：{len(tasks)} 个 (dir, run)（缺失 {len(missing)} 个 checkpoint）")

    cls_rows, reg_rows, cls_det, reg_det = [], [], [], []
    done = 0
    # CUDA 与 fork 不兼容（"Cannot re-initialize CUDA in forked subprocess"），
    # 用 spawn 启动；parser/loaders 在子进程 initializer 内重建
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args, DEVICE)) as pool:
        for out in pool.imap_unordered(eval_task, tasks, chunksize=1):
            if out.get("missing"):
                continue
            if out.get("failed"):
                (cls_rows if out["kind"] == "cls" else reg_rows).append(out["row"])
                done += 1
                continue
            if out["kind"] == "cls":
                cls_rows.append(out["row"])
                cls_det.extend(out["det_rows"])
            else:
                reg_rows.append(out["row"])
                reg_det.extend(out["det_rows"])
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"[进度] {done}/{len(tasks)}")

    def write(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write(OUT_ROOT / "mismatch_eval_imagenet100_mlp.csv", CLS_HEADER, cls_rows)
    write(OUT_ROOT / "mismatch_eval_california_mlp.csv", REG_HEADER, reg_rows)
    write(OUT_ROOT / "mismatch_eval_imagenet100_mlp_classdetail.csv", CLS_DETAIL_HEADER, cls_det)
    write(OUT_ROOT / "mismatch_eval_california_mlp_bindetail.csv", REG_DETAIL_HEADER, reg_det)
    write(OUT_ROOT / "missing_matrix.csv", ["dir", "run", "status"],
          [(d_, r_, "missing") for d_, r_ in missing])

    n_fail_cls = sum(1 for r_ in cls_rows if r_[-1])
    n_fail_reg = sum(1 for r_ in reg_rows if r_[-1])
    print(f"\n结果已保存：{OUT_ROOT}")
    print(f"  mismatch_eval_imagenet100_mlp.csv      {len(cls_rows)} 行（failed {n_fail_cls}）")
    print(f"  mismatch_eval_california_mlp.csv       {len(reg_rows)} 行（failed {n_fail_reg}）")
    print(f"  明细 CSV：分类 {len(cls_det)} 行、回归 {len(reg_det)} 行")
    print(f"  missing_matrix.csv                     {len(missing)} 行缺失")


if __name__ == "__main__":
    main()
