"""信息平面评估：对已训 checkpoint 估计三个信息论量，输出长表 csv。

- I(X;Z)：InfoNCE 下界（van den Oord et al., 2018）：I(X;Z) ≥ log B −
  L_NCE（B 为批内负样本数）。CPC 风格 critic s(x_j, z_i) = g_z(z_i)ᵀ
  g_h(h(x_j))（两个 128 维小 MLP 头，h 为模型编码器输出的 256 维特征）
  在训练划分上训练（每 run 独立训练，固定 seed 42），界在测试划分上分批
  计算：mean_i [log B + s_ii − logsumexp_j s_ij]（nats）。z 从 q(z|x)
  重参数化采样（每批新采样）：μ 路径的条件分布近 δ，InfoNCE 的有限批
  偏差为正、critic 目标奇异，随机路径下条件为真正的高斯、界行为良好。
  注意：极端压缩配置（β=25）σ→0、近确定性，其 InfoNCE 估计可能带正偏差
  （Poole et al. 2019 的有限批偏差），图中如实保留并在叙述中标注。
  因 z = Linear(h)，I(h(X);Z) 与 I(X;Z) 在 μ 路径上精确相等，critic 用 h
  作为 x 侧输入不损失信息。
- I(Y;Z)：分类 = H(Y) − CE(logits)（CE 上界 H(Y|Z)，故为下界，nats，
  H(Y)=ln K）；回归（housing）= 高斯近似 0.5·ln(Var(y)/Var(y−ŷ))
  （联合高斯假设下的闭式，启发式估计）。
- I(X;Z|Y) = I(X;Z) − I(Y;Z)：条件互信息恒等式。注意是两个下界之差，
  不是严格界（脚本 docstring 与论文叙述均如实标注）。

配置目录沿用 compression_eval.py 的解析与模型重建（opb 分类能量分类器 /
回归 tied 头自动处理）：MNIST 读 output/adv_mnist/、imagenet100 与 california
（housing）读 output/compression_eval/；基线模型（无瓶颈）跳过。

输出：{eval-root}/info_plane.csv（长表：combo/task/model/beta/anchor/run/
I_XZ/I_YZ/I_XZ_given_Y/CE/Acc/R2），论文图由 info_plane_plot.py 生成。

用法：
    python info_plane_eval.py                    # 全部三个数据集
    python info_plane_eval.py --datasets mnist   # 只跑 mnist（冒烟/补跑）
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from compression_eval import DATASET_TO_TASK, REGRESSION_TASKS, parse_combo_dir
from datasets.datasets import (
    get_california_dataloaders,
    get_imagenet100_dataloaders,
    get_mnist_dataloaders,
)
from model.mlp.utils import flatten
from train import build_model, build_parser, run_model

ROOT = Path(__file__).resolve().parent
EVAL_ROOTS = {
    "mnist": ROOT / "output" / "adv_mnist",
    "imagenet100": ROOT / "output" / "compression_eval",
    "california": ROOT / "output" / "compression_eval",
}

CRITIC_EPOCHS = 30  # 随机路径下 critic ~20 epoch 收敛（诊断：train loss 5.8 ≈ log B − 2.5）
CRITIC_BATCH = 4096
CRITIC_LR = 1e-3
CRITIC_SEED = 42


def loaders_for(dataset, args):
    """数据集名（目录名）→ (train_loader, val_loader, test_loader)。"""
    if dataset == "mnist":
        return get_mnist_dataloaders(args.batch_size, args.data_dir)
    if dataset == "imagenet100":
        return get_imagenet100_dataloaders(args.batch_size, args.data_dir)
    if dataset == "california":
        return get_california_dataloaders(args.batch_size, args.data_dir)
    raise ValueError(f"不支持的数据集：{dataset}")


def build_model_for_dir(parser, args, d, dataset, device):
    """按目录名重建模型（与 compression_eval.py 同口径：opb 分类能量分类器、
    回归 tied 头；imagenet100 传特征维度）。"""
    task = DATASET_TO_TASK.get(dataset, dataset)
    dir_args = argparse.Namespace(**vars(args))
    dir_args.task = task
    dir_args.model = parse_combo_dir(d.name)[2]
    dir_args.backbone = parse_combo_dir(d.name)[1]
    dir_args.anchor_scale = parse_combo_dir(d.name)[4]
    dir_args.energy_classifier = task not in REGRESSION_TASKS and dir_args.model == "opb"
    dir_args.tied_head = task in REGRESSION_TASKS and dir_args.model == "opb"
    if dataset == "imagenet100":
        loaders = loaders_for(dataset, args)
        model = build_model(
            parser, dir_args,
            imagenet100_input_dim=loaders[3],
            imagenet100_feature_pool=loaders[4],
        )
    else:
        model = build_model(parser, dir_args)
    return model.to(device), task


def encode(model, loader, device):
    """冻结前向收集 (h, mu, logvar, logits, y) 张量（训练/测试划分各自调用）。

    h 为编码器输出（critic 的 x 侧输入），z 在 critic 训练/估计时按
    q(z|x) = N(mu, exp(logvar)) 重参数化采样（每批新采样）。
    """
    hs, mus, lvs, yss, lgs = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.encoder(flatten(x))
            mu = model.mu_head(h)
            logvar = model.logvar_head(h)
            logits = run_model(model, x, None, stochastic=False)[0]
            hs.append(h.cpu())
            mus.append(mu.cpu())
            lvs.append(logvar.cpu())
            yss.append(y)
            lgs.append(logits.cpu())
    return (
        torch.cat(hs), torch.cat(mus), torch.cat(lvs),
        torch.cat(yss), torch.cat(lgs),
    )


def train_critic(h, mu, logvar, device, epochs=CRITIC_EPOCHS,
                 batch=CRITIC_BATCH, lr=CRITIC_LR, seed=CRITIC_SEED):
    """CPC 风格 critic：s(x_j, z_i) = g_z(z_i)ᵀ g_h(h(x_j))，InfoNCE 损失训练。

    每批从 q(z|x) 重参数化采样 z（新噪声），分数矩阵按 (B,128)@(128,B)
    计算、不显式展开 B² 对。返回 (gh, gz)。
    """
    torch.manual_seed(seed)
    dh, dz = h.shape[1], mu.shape[1]
    gh = torch.nn.Sequential(
        torch.nn.Linear(dh, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128)).to(device)
    gz = torch.nn.Sequential(
        torch.nn.Linear(dz, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128)).to(device)
    opt = torch.optim.Adam(list(gh.parameters()) + list(gz.parameters()), lr=lr)
    n = len(h)
    idx = torch.randperm(n)  # CPU 索引（h/mu/logvar 均为 CPU 张量）
    for _ in range(epochs):
        for s in range(0, n, batch):
            b = idx[s: s + batch]
            z = mu[b] + torch.exp(0.5 * logvar[b]) * torch.randn_like(mu[b])
            scores = gz(z.to(device)) @ gh(h[b].to(device)).T  # (B, B)
            loss = F.cross_entropy(scores, torch.arange(len(b), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    gh.eval()
    gz.eval()
    return gh, gz


def infonce_bound(h, mu, logvar, gh, gz, device, batch=CRITIC_BATCH):
    """测试划分 InfoNCE 下界（nats）：mean_i [log B + s_ii − logsumexp_j s_ij]。"""
    total, n = 0.0, len(h)
    with torch.no_grad():
        for s in range(0, n, batch):
            b = slice(s, s + batch)
            z = mu[b] + torch.exp(0.5 * logvar[b]) * torch.randn_like(mu[b])
            scores = gz(z.to(device)) @ gh(h[b].to(device)).T
            total += torch.log(torch.tensor(len(scores), dtype=torch.float64)).item() * len(scores)
            total += F.log_softmax(scores, dim=1).diag().double().sum().item()
    return total / n


def eval_combo(parser, args, d, dataset, device):
    """单配置：逐 run 估计三量，返回 csv 行列表。"""
    model, task = build_model_for_dir(parser, args, d, dataset, device)
    loaders = loaders_for(dataset, args)
    train_loader, _, test_loader = loaders[0], loaders[1], loaders[2]
    ckpts = sorted(d.glob(f"{d.name}_run*.pt"))
    rows = []
    for run_i, ckpt in enumerate(ckpts, 1):
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        model.eval()
        h_tr, mu_tr, lv_tr, _, _ = encode(model, train_loader, device)
        h_te, mu_te, lv_te, y_te, logits_te = encode(model, test_loader, device)
        gh, gz = train_critic(h_tr, mu_tr, lv_tr, device)
        i_xz = infonce_bound(h_te, mu_te, lv_te, gh, gz, device)

        if task in REGRESSION_TASKS:
            yhat = logits_te.squeeze(-1)
            ce = F.mse_loss(yhat, y_te.float(), reduction="mean").item()
            var_y = y_te.float().var(unbiased=False).item()
            var_res = (y_te.float() - yhat).var(unbiased=False).item()
            i_yz = 0.5 * math.log(var_y / var_res) if var_res > 0 else float("nan")
            ss_res = ((yhat - y_te.float()) ** 2).sum().item()
            ss_tot = ((y_te.float() - y_te.float().mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            acc = ""
        else:
            ce = F.cross_entropy(logits_te, y_te, reduction="mean").item()
            i_yz = math.log(model.num_classes) - ce
            acc = (logits_te.argmax(1) == y_te).float().mean().item()
            r2 = ""
        rows.append([d.name, dataset, task, parse_combo_dir(d.name)[2],
                     parse_combo_dir(d.name)[3], run_i,
                     f"{i_xz:.6f}", f"{i_yz:.6f}", f"{i_xz - i_yz:.6f}",
                     f"{ce:.6f}",
                     f"{acc:.6f}" if acc != "" else "",
                     f"{r2:.6f}" if r2 != "" else ""])
        print(f"[{d.name}] run{run_i} I(X;Z)={i_xz:.4f} I(Y;Z)={i_yz:.4f} "
              f"I(X;Z|Y)={i_xz - i_yz:.4f} CE={ce:.4f} "
              + (f"acc={acc:.4f}" if acc != "" else f"r2={r2:.4f}"))
    return rows


def main():
    parser = build_parser()
    parser.add_argument(
        "--datasets", nargs="*", default=["mnist", "imagenet100", "california"],
        help="数据集列表（目录名：mnist/imagenet100/california，默认全部三个）",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in args.datasets:
        root = EVAL_ROOTS[dataset]
        csv_path = root / "info_plane.csv"
        rows = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            info = parse_combo_dir(d.name)
            if info is None or info[0] != dataset:
                continue
            try:
                rows.extend(eval_combo(parser, args, d, dataset, device))
            except Exception as e:  # 单项失败不中断其余项
                print(f"[失败] {d.name}: {e}")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["combo", "dataset", "task", "model", "beta", "anchor",
                        "run", "I_XZ", "I_YZ", "I_XZ_given_Y", "CE", "Acc", "R2"])
            w.writerows(rows)
        print(f"\n结果已保存：{csv_path}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
