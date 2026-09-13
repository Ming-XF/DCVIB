"""信息平面评估：对已训 checkpoint 估计三个信息论量，输出长表 csv。

- I(X;Z)：InfoNCE 下界（van den Oord et al., 2018）：I(X;Z) ≥ log B −
  L_NCE（B 为批内负样本数）。CPC 风格 critic s(x_j, z_i) = g_z(z_i)ᵀ
  g_h(h(x_j))（两个 128 维小 MLP 头，h 为模型编码器输出的 256 维特征）
  在训练划分上训练（每 run 独立训练，固定 seed 42），界在测试划分上分批
  计算：mean_i [log B + s_ii − logsumexp_j s_ij]（nats）。z 从 q(z|x)
  重参数化采样（每批新采样）。注意：极端压缩配置（β=25）σ→0、近确定性，
  其 InfoNCE 估计可能带正偏差（Poole et al. 2019 的有限批偏差），图中如实
  保留并在叙述中标注。因 z = Linear(h)，critic 用 h 作为 x 侧输入不损失信息。
- I(Y;Z)：随机路径 MC 估计（方案 0）——与 I(X;Z) 同分布（此前 μ 路径与
  随机路径混用会使 I(X;Z|Y)=I(X;Z)−I(Y;Z) 失去一致性）。分类 =
  ln K − CE_MC，CE_MC = −(1/N)Σᵢ (1/M)Σₘ log p(yᵢ|zᵢ⁽ᵐ⁾)（每个 x
  采 M=MC_SAMPLES 个 z，仍为下界：CE ≥ H(Y|Z)）；回归 = 高斯近似
  0.5·ln(Var(y)/Var(y−ŷ_MC))，ŷ_MC 为 M 个采样 z 预测的均值（随机路径
  下的最优预测器，联合高斯假设的闭式，启发式估计）。
- I(X;Z|Y)（方案 1，分类）：类条件高斯混合直接估计——q(z|x) 为模型已知
  参数化后验（高斯），q(z|y) 为同类测试样本的经验混合 (1/N_c)Σⱼ q(z|xⱼ)，
  恒等式 I(X;Z|Y) = E_x,y E_{z~q(z|x)}[log q(z|x) − log q(z|y)]
  （z 仅经 x 依赖 y，q(z|x,y)=q(z|x)，对模型自身分布精确成立）。log q(z|y)
  用 logsumexp 精确计算：对已知分量的混合密度，logsumexp_j log N_j −
  log N_c 就是精确的 log q(z|y)（不是 Jensen 界），估计误差仅剩 z 采样
  MC 噪声与 N_c 有限样本近似——一致的精确 MC 估计。mean-log 方向
  （Jensen 下界）在小 β、σ→0、类内分量近似不相交时被远距离分量主导而
  发散，仅作参考写入 I_XZ_given_Y_upper 列、不参与点估计与绘图。回归
  （housing）无类别可条件，仍用 I(X;Z)−I(Y;Z) 差值（随机路径两估计之差，
  非严格界，如实标注）。

配置目录沿用 compression_eval.py 的解析与模型重建（opb 分类能量分类器 /
回归 tied 头自动处理）：MNIST 读 output/adv_mnist/、imagenet100 与 california
（housing）读 output/compression_eval/；基线模型（无瓶颈）跳过。

输出：{eval-root}/info_plane_{dataset}.csv（长表：combo/dataset/task/model/
beta/anchor/run/I_XZ/I_YZ/I_XZ_given_Y/I_XZ_given_Y_upper/CE/Acc/R2，
回归行无上界列），论文图由 info_plane_plot.py 生成。

用法：
    python info_plane_eval.py                    # 全部三个数据集
    python info_plane_eval.py --datasets mnist   # 只跑 mnist（冒烟/补跑）
    python info_plane_eval.py --datasets mnist --shard 0 --num-shards 16
        # 多进程并行：16 分片各处理一部分目录，输出 shard csv 后合并
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
CRITIC_STEPS_TARGET = 300  # 总更新步目标：小训练集按 epoch 数补齐（约 300 步收敛）
CRITIC_BATCH = 4096
CRITIC_LR = 1e-3
CRITIC_SEED = 42
MC_SAMPLES = 20  # I(Y;Z) 与条件互信息估计中每个 x 的 z 采样数
MC_CHUNK = 2048  # MC 预测的分块大小（限制显存）
COND_CHUNK = 256  # 类条件混合密度矩阵的行分块大小（限制显存）


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


def logits_from_z(model, z):
    """从采样 z 得到 logits（与训练前向一致的预测头路径）。

    OPB 分类为锚点能量分类器（_prior_table + _energy_logits）、OPB 回归
    tied 头为 uᵀz/ρ（均与 forward 内部一致）；其余模型直接 classifier(z)。
    z 须与模型同设备。
    """
    if model.__class__.__name__ == "OPB":
        if getattr(model, "energy_classifier", False):
            prior_mu, prior_logvar_table = model._prior_table()
            return model._energy_logits(z, prior_mu, prior_logvar_table)
        if getattr(model, "tied_head", False):
            u = F.normalize(model.prior_direction.weight.squeeze(-1), dim=0)
            return ((z @ u) / model.anchor_scale).unsqueeze(-1)
    return model.classifier(z)


def _std(logvar):
    """q(z|x) 的标准差：logvar clamp 到 [-10, 10]（与模型 KL 约定一致）。"""
    return torch.exp(0.5 * logvar.clamp(-10.0, 10.0))


def mc_i_yz(model, mu, logvar, y, device, is_regression):
    """随机路径 I(Y;Z)（方案 0）：对采样 z 做 MC 平均，与 I(X;Z) 同分布。

    分类：CE_MC = −(1/N)Σᵢ (1/M)Σₘ log p(yᵢ|zᵢ⁽ᵐ⁾)，
    I(Y;Z) = ln K − CE_MC（CE ≥ H(Y|Z)，仍为下界）；
    回归：ŷ_MC = (1/M)Σₘ f(z⁽ᵐ⁾)，残差 y − ŷ_MC 代入高斯近似
    0.5·ln(Var(y)/Var(y−ŷ_MC))。
    """
    n = len(y)
    y = y.to(device)
    total = 0.0
    preds = []
    with torch.no_grad():
        for s in range(0, n, MC_CHUNK):
            b = slice(s, s + MC_CHUNK)
            mu_b, lv_b, y_b = mu[b].to(device), logvar[b].to(device), y[b]
            if is_regression:
                out = torch.zeros(mu_b.size(0), 1, device=device)
                for _ in range(MC_SAMPLES):
                    z = mu_b + _std(lv_b) * torch.randn_like(mu_b)
                    out += logits_from_z(model, z)
                preds.append((out / MC_SAMPLES).cpu())
            else:
                for _ in range(MC_SAMPLES):
                    z = mu_b + _std(lv_b) * torch.randn_like(mu_b)
                    total += F.cross_entropy(
                        logits_from_z(model, z), y_b, reduction="sum").item()
    if is_regression:
        yhat = torch.cat(preds).squeeze(-1)
        yc = y.cpu().float()
        var_y = yc.var(unbiased=False).item()
        var_res = (yc - yhat).var(unbiased=False).item()
        return 0.5 * math.log(var_y / var_res) if var_res > 0 else float("nan")
    return math.log(model.num_classes) - total / (n * MC_SAMPLES)


def conditional_mi_estimate(mu, logvar, y, device):
    """分类 I(X;Z|Y) 直接估计（方案 1）：类条件高斯混合的精确 MC 估计。

    q(z|x) 为模型已知参数化后验（高斯），q(z|y) 为同类测试样本的经验混合
    (1/N_c)Σⱼ q(z|xⱼ)，恒等式 I(X;Z|Y) = E_x,y E_{z~q(z|x)}[log q(z|x) −
    log q(z|y)]（z 仅经 x 依赖 y）。log q(z|y) 用 logsumexp 精确计算：对
    已知分量的混合密度，logsumexp_j log N_j − log N_c 就是精确的
    log q(z|y)（不是 Jensen 界），估计误差仅剩 z 采样 MC 噪声与 N_c 有限
    样本近似。mean-log 方向（Jensen 下界）在 σ→0、类内分量近似不相交时
    被远距离分量主导而发散，仅作参考返回、不参与点估计与绘图。
    返回 (estimate, meanlog_ref)。
    """
    n = len(y)
    lo_sum = hi_sum = 0.0
    n_eff = 0
    for c in y.unique().tolist():
        idx = (y == c).nonzero(as_tuple=True)[0]
        nc = idx.numel()
        mu_c = mu[idx]  # (nc, d)，CPU
        lv_c = logvar[idx].clamp(-10.0, 10.0)
        mu_rep = mu_c.repeat_interleave(MC_SAMPLES, dim=0)
        lv_rep = lv_c.repeat_interleave(MC_SAMPLES, dim=0)
        z = mu_rep + torch.exp(0.5 * lv_rep) * torch.randn_like(mu_rep)  # (nc*M, d)
        var_rep = lv_rep.exp()
        # 自身密度 log q(z|x_i)（分析闭式，CPU 一次性算完）
        logq_x = -0.5 * (((z - mu_rep).pow(2) / var_rep + lv_rep
                           + math.log(2.0 * math.pi)).sum(1))
        mu_d, lv_d = mu_c.to(device), lv_c.to(device)
        var_d = lv_d.exp()
        const_d = (math.log(2.0 * math.pi) + lv_d).sum(1)  # (nc,)
        for s in range(0, z.size(0), COND_CHUNK):
            zb = z[s: s + COND_CHUNK].to(device)
            lqx = logq_x[s: s + COND_CHUNK].to(device)
            # 密度矩阵 S[a, j] = log N(zb_a; mu_j, var_j)
            diff2 = (zb.unsqueeze(1) - mu_d.unsqueeze(0)).pow(2)  # (A, nc, d)
            s_mat = -0.5 * ((diff2 / var_d.unsqueeze(0)).sum(-1)
                            + const_d.unsqueeze(0))  # (A, nc)
            logq_ub = s_mat.logsumexp(dim=1) - math.log(nc)  # log q(z|y) 上界
            logq_lb = s_mat.mean(dim=1)                      # log q(z|y) 下界
            lo_sum += (lqx - logq_ub).double().sum().item()
            hi_sum += (lqx - logq_lb).double().sum().item()
            n_eff += zb.size(0)
    return lo_sum / n_eff, hi_sum / n_eff


def train_critic(h, mu, logvar, device, epochs=CRITIC_EPOCHS,
                 batch=CRITIC_BATCH, lr=CRITIC_LR, seed=CRITIC_SEED,
                 steps_target=CRITIC_STEPS_TARGET):
    """CPC 风格 critic：s(x_j, z_i) = g_z(z_i)ᵀ g_h(h(x_j))，InfoNCE 损失训练。

    每批从 q(z|x) 重参数化采样 z（新噪声），分数矩阵按 (B,128)@(128,B)
    计算、不显式展开 B² 对。epoch 数按训练集大小补齐到约 steps_target 个
    更新步（小数据集固定 epoch 数会欠拟合 critic、下界过松）。返回 (gh, gz)。
    """
    torch.manual_seed(seed)
    steps_per_epoch = math.ceil(len(h) / batch)
    epochs = max(epochs, math.ceil(steps_target / steps_per_epoch))
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
            r2 = 1 - var_res / var_y if var_y > 0 else float("nan")
            acc = ""
            i_yz = mc_i_yz(model, mu_te, lv_te, y_te, device, is_regression=True)
            cert_hi = ""
            # 回归无类别可条件，仍为随机路径两估计之差（非严格界）
            cert = i_xz - i_yz
        else:
            ce = F.cross_entropy(logits_te, y_te, reduction="mean").item()
            acc = (logits_te.argmax(1) == y_te).float().mean().item()
            r2 = ""
            i_yz = mc_i_yz(model, mu_te, lv_te, y_te, device, is_regression=False)
            # cert 为 logsumexp 精确估计；cert_hi 为 mean-log 参考值（发散、不参与点估计）
            cert, cert_hi = conditional_mi_estimate(mu_te, lv_te, y_te, device)
        anchor = parse_combo_dir(d.name)[4]
        rows.append([d.name, dataset, task, parse_combo_dir(d.name)[2],
                     parse_combo_dir(d.name)[3],
                     f"{anchor:g}" if anchor is not None else "",
                     run_i,
                     f"{i_xz:.6f}", f"{i_yz:.6f}", f"{cert:.6f}",
                     f"{cert_hi:.6f}" if cert_hi != "" else "",
                     f"{ce:.6f}",
                     f"{acc:.6f}" if acc != "" else "",
                     f"{r2:.6f}" if r2 != "" else ""])
        print(f"[{d.name}] run{run_i} I(X;Z)={i_xz:.4f} I(Y;Z)={i_yz:.4f} "
              f"I(X;Z|Y)={cert:.4f}"
              + (f" (meanlog 参考 {cert_hi:.4f})" if cert_hi != "" else "")
              + f" CE={ce:.4f} "
              + (f"acc={acc:.4f}" if acc != "" else f"r2={r2:.4f}"))
    return rows


def main():
    parser = build_parser()
    parser.add_argument(
        "--datasets", nargs="*", default=["mnist", "imagenet100", "california"],
        help="数据集列表（目录名：mnist/imagenet100/california，默认全部三个）",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="目录分片索引（0 起，配合 --num-shards 多进程并行）",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="目录分片总数（1 = 不分片，输出 info_plane_{dataset}.csv；"
        ">1 时各分片输出 info_plane_{dataset}_shard{i}.csv，需另行合并）",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in args.datasets:
        root = EVAL_ROOTS[dataset]
        # imagenet100 与 california 共用同一 eval-root，须按数据集分文件
        suffix = f"_shard{args.shard}" if args.num_shards > 1 else ""
        csv_path = root / f"info_plane_{dataset}{suffix}.csv"
        rows = []
        # 分片：按目录序取模分配，各进程互不重叠、覆盖全部组合
        dirs = [
            d for d in sorted(root.iterdir())
            if d.is_dir() and (info := parse_combo_dir(d.name)) is not None
            and info[0] == dataset
        ]
        dirs = [d for i, d in enumerate(dirs) if i % args.num_shards == args.shard]
        for d in dirs:
            try:
                rows.extend(eval_combo(parser, args, d, dataset, device))
            except Exception as e:  # 单项失败不中断其余项
                print(f"[失败] {d.name}: {e}")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["combo", "dataset", "task", "model", "beta", "anchor",
                        "run", "I_XZ", "I_YZ", "I_XZ_given_Y",
                        "I_XZ_given_Y_upper", "CE", "Acc", "R2"])
            w.writerows(rows)
        print(f"\n结果已保存：{csv_path}（{len(rows)} 行）")


if __name__ == "__main__":
    main()
