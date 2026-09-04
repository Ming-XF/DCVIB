"""压缩-精度评估：对补训 checkpoint 计算测试集 CE、E[KL] 与 KL 分解项
（均值失配项 / 方差失配项），输出长表与汇总 csv。

真实压缩-精度曲线的横轴是实际 E[KL]（证书 = KL 期望），本脚本产出该数据：
每个组合 × 每个 run 的 CE / kl / kl_mean / kl_var / 任务指标。
评估走 z=μ 确定性路径；CE 用模型的真实预测头（能量分类器/tied 头/自由头，
与训练一致）；KL 分解用对角高斯闭式（附录 KL 分解引理的通用版，先验方差
按模型实际——OPB 固定 τ²=1、CEB 逐类学习）。

输出（{eval-root} 下）：
    compression_eval.csv          （长表，每 run 一行）
    compression_eval_summary.csv  （每组合一行，均值±std）

用法：
    python compression_eval.py --batch-size 512 --data-dir ./data
    python compression_eval.py --eval-root output/adv_mnist   # MNIST 压缩-鲁棒性补训
"""

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.agedb import get_agedb_dataloaders
from datasets.datasets import (
    get_california_dataloaders,
    get_imagenet100_dataloaders,
    get_mnist_dataloaders,
)
from model.mlp.utils import flatten, qr_anchor_table
from train import build_model, build_parser, run_model

ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_ROOT = ROOT / "output" / "compression_eval"
# 目录名数据集 → train.py 任务名（tune 目录用数据集名：california 即 housing）
DATASET_TO_TASK = {"california": "housing"}
# 回归任务集合（CE = 标准化 MSE）
REGRESSION_TASKS = ("agedb", "housing")


def _prior_table_cls(model, device):
    """分类先验表：返回 (mu_p (K,d), logvar_p (K,d))。

    OPB 走与训练前向相同的 qr_anchor_table 正交锚点路径；CEB 为学习表。
    """
    eye = torch.eye(model.num_classes, device=device)
    out = model.prior_net(eye)
    mu_raw, logvar_p = out.chunk(2, dim=1)
    if model.__class__.__name__ == "OPB":
        mu_p = qr_anchor_table(mu_raw.t(), model.anchor_scale)  # (K, d)
    else:
        mu_p = mu_raw
    return mu_p, logvar_p


def batch_terms(model, x, y, device):
    """单批 KL 分解：返回 (mu, kl_mean, kl_var)；y 为归一化标签（回归）或类号。"""
    h = model.encoder(flatten(x.to(device)))
    mu = model.mu_head(h)
    logvar = model.logvar_head(h)
    y = y.to(device)
    if getattr(model, "continuous_y", False):
        yf = y.float().unsqueeze(-1)
        if model.__class__.__name__ == "OPB":
            u = F.normalize(model.prior_direction.weight.squeeze(-1), dim=0)
            mu_p = model.anchor_scale * yf * u.unsqueeze(0)
            logvar_p = model.prior_logvar_net(yf)
        else:
            mu_p, logvar_p = model.prior_net(yf).chunk(2, dim=1)
    else:
        mu_p_all, logvar_p_all = _prior_table_cls(model, device)
        mu_p = mu_p_all[y]
        logvar_p = logvar_p_all[y]
    var = logvar.exp()
    var_p = logvar_p.exp()
    kl_mean = (0.5 * (mu - mu_p).pow(2) / var_p).sum(1)  # 均值失配项
    kl_var = 0.5 * (var / var_p - 1 - logvar + logvar_p).sum(1)  # 方差失配项
    return mu, kl_mean, kl_var


def eval_checkpoint(model, loader, task, device):
    """测试集整遍：返回 (CE, kl, kl_mean, kl_var, 任务指标 acc/r2)。"""
    model.eval()
    sums = {"ce": 0.0, "kl": 0.0, "kl_mean": 0.0, "kl_var": 0.0}
    n = 0
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            y = y.to(device)
            mu, kl_mean, kl_var = batch_terms(model, x, y, device)
            kl = kl_mean + kl_var
            logits = run_model(model, x.to(device), None, stochastic=False)[0]
            if task in REGRESSION_TASKS:
                ce = F.mse_loss(logits.squeeze(-1), y.float(), reduction="none")
                preds.append(logits.squeeze(-1).cpu())
            else:
                ce = F.cross_entropy(logits, y, reduction="none")
                preds.append(logits.argmax(1).cpu())
            ys.append(y.cpu())
            sums["ce"] += ce.sum().item()
            sums["kl"] += kl.sum().item()
            sums["kl_mean"] += kl_mean.sum().item()
            sums["kl_var"] += kl_var.sum().item()
            n += y.numel()
    out = {k: v / n for k, v in sums.items()}
    preds = torch.cat(preds)
    ys = torch.cat(ys)
    if task in REGRESSION_TASKS:
        ss_res = ((preds - ys.float()) ** 2).sum().item()
        ss_tot = ((ys.float() - ys.float().mean()) ** 2).sum().item()
        out["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        out["acc"] = (preds == ys).float().mean().item()
    return out


def parse_combo_dir(name):
    """目录名 → (task, backbone, model, beta, anchor)。如
    'imagenet100_mlp_opb_beta_0.01_anchor_6'。"""
    m = re.match(r"^(\w+)_(\w+)_(ceb|opb)_beta_([\d.eE+-]+)(?:_anchor_([\d.eE+-]+))?$", name)
    if not m:
        return None
    task, backbone, model = m.group(1), m.group(2), m.group(3)
    beta = float(m.group(4))
    anchor = float(m.group(5)) if m.group(5) else None
    return task, backbone, model, beta, anchor


def main():
    parser = build_parser()
    parser.add_argument(
        "--eval-root", type=str, default=str(DEFAULT_EVAL_ROOT),
        help="checkpoint 目录根（默认 output/compression_eval；"
        "MNIST 补训用 output/adv_mnist），输出 csv 也写入该目录",
    )
    args = parser.parse_args()
    eval_root = Path(args.eval_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {
        "imagenet100": get_imagenet100_dataloaders(args.batch_size, args.data_dir),
        "agedb": get_agedb_dataloaders(args.batch_size, args.data_dir),
        "california": get_california_dataloaders(args.batch_size, args.data_dir),
        "mnist": get_mnist_dataloaders(args.batch_size, args.data_dir),
    }

    csv_rows = []
    summary_rows = []
    for d in sorted(eval_root.iterdir()):
        if not d.is_dir():
            continue
        info = parse_combo_dir(d.name)
        if info is None:
            print(f"[跳过] 无法解析目录名：{d.name}")
            continue
        task, backbone, model_name, beta, anchor = info
        dataset = task  # 目录名数据集（loaders 键）
        task = DATASET_TO_TASK.get(task, task)  # california → housing
        ckpts = sorted(d.glob(f"{d.name}_run*.pt"))
        if not ckpts:
            print(f"[跳过] 无 checkpoint：{d.name}")
            continue

        dir_args = argparse.Namespace(**vars(args))
        dir_args.task = task
        dir_args.model = model_name
        dir_args.backbone = backbone
        dir_args.anchor_scale = anchor if anchor is not None else args.anchor_scale
        # 与训练时一致的头：分类 OPB 能量分类器（imagenet100/mnist 均如此训练）、
        # 回归 OPB tied 投影头
        dir_args.energy_classifier = task not in REGRESSION_TASKS and model_name == "opb"
        dir_args.tied_head = task in REGRESSION_TASKS and model_name == "opb"
        if dataset == "imagenet100":
            input_dim, feature_pool = loaders[dataset][3], loaders[dataset][4]
            model = build_model(
                parser, dir_args,
                imagenet100_input_dim=input_dim,
                imagenet100_feature_pool=feature_pool,
            )
        else:
            model = build_model(parser, dir_args)
        model = model.to(device)
        test_loader = loaders[dataset][2]

        run_metrics = []
        for ckpt in ckpts:
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            m = eval_checkpoint(model, test_loader, task, device)
            m["run"] = int(re.search(r"run(\d+)", ckpt.stem).group(1))
            m["seed"] = args.seed + m["run"] - 1
            run_metrics.append(m)
            csv_rows.append(
                [d.name, task, model_name, beta, anchor, m["run"], m["seed"],
                 m["ce"], m["kl"], m["kl_mean"], m["kl_var"],
                 m.get("acc", ""), m.get("r2", "")]
            )
            print(
                f"[{d.name}] run{m['run']} CE={m['ce']:.4f} KL={m['kl']:.4f} "
                f"(mean {m['kl_mean']:.4f} / var {m['kl_var']:.4f}) "
                f"{'acc=' + format(m.get('acc'), '.4f') if 'acc' in m else 'r2=' + format(m.get('r2'), '.4f')}"
            )

        # 汇总：各量均值±std
        row = [d.name, task, model_name, beta, anchor]
        for k in ["ce", "kl", "kl_mean", "kl_var", "acc", "r2"]:
            vals = [m[k] for m in run_metrics if k in m]
            if vals:
                t = torch.tensor(vals, dtype=torch.float64)
                row += [t.mean().item(), t.std().item()]
            else:
                row += ["", ""]
        summary_rows.append(row)
        print(f"[汇总] {d.name}: KL={sum(m['kl'] for m in run_metrics) / len(run_metrics):.4f}")

    csv_path = eval_root / "compression_eval.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "task", "model", "beta", "anchor", "run", "seed",
                    "CE", "KL", "KL_mean", "KL_var", "Acc", "R2"])
        w.writerows(csv_rows)
    summary_path = eval_root / "compression_eval_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["combo", "task", "model", "beta", "anchor"]
        for k in ["CE", "KL", "KL_mean", "KL_var", "Acc", "R2"]:
            cols += [f"{k}_mean", f"{k}_std"]
        w.writerow(cols)
        w.writerows(summary_rows)
    print(f"\n长表已保存：{csv_path}")
    print(f"汇总已保存：{summary_path}")


if __name__ == "__main__":
    main()
