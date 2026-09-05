"""合成对齐试验（synthetic exact-realizability task）：检验 Theorem 2.1 的
条件性 alignment scaling E[D(Y)] ≤ (C_align + ε)/β，方案见 codex_output.txt。

任务：K=10 类均匀、x = e_y（one-hot）、10000/2000/2000 独立生成、无增强/噪声；
Y = f(X) 严格成立，且单层线性编码器可实现任意 class-constant 后验均值，对齐
comparison solution q(z|x) = N(a·q_y, τ²I)（KL=0）在模型类内显式可实现。

模型：OPB（model/mlp/opb.py）+ 锚点能量分类器；τ²=1 固定（fixed_prior_var）；
frame 两设置——fixed（冻结恒等帧，锚点恒为 a·e_k，隔离优化误差）与 trainable
（论文主方法 stateless polar frame）；a ∈ {6, 12}；d = K（可 --d 16 冗余控制）；
β 对数网格 {1e-3..10} × 5 seeds；每 seed 记录最佳验证 checkpoint（按验证
objective 早停），失败/NaN 如实记录为 fail 行、不静默删除。

每个 checkpoint 记录（训练/测试）：D_hat = (1/N)Σ‖μ−a·q_y‖²/(2τ²)（τ²=1 下
即 KL 均值失配项）、β·D_hat、CE、KL 三项分解、objective = CE + β·E[KL]、
acc_det（μ 路径）/acc_mc（MC 采样）、frame 诊断（锚点成对距离、后验类中心
最小成对距离与平均成对余弦、先验均值原始矩阵最小奇异值）、best_epoch。

C_G = H(Y|Z_align)：锚点能量分类器在 q_align 下的交叉熵，对正交帧旋转不变
（每 a 一个值），每 a 1e6 MC + 标准误差。observed envelope L_min,obs 与
ε_obs = L − L_min,obs 在打印摘要中给出并明确标注非理论 ε（本试验不声称
验证定理上界，只作 scaling diagnostic）。

控制组（可选，单独输出 CSV、不与主实验合并）：
  --hidden-dims 2    容量控制（窄编码器，non-realizable）
  --label-noise 0.1  标签噪声（10% 翻转训练标签，测试干净）

输出：output/synthetic_align/synthetic_align{--out-name}.csv（列见 main 的
HEADER）+ cg_{out-name}.csv；论文图由 synthetic_align_plot.py 生成。

用法：
    python synthetic_align.py                          # 全部网格（2×2×9×5=180 run）
    python synthetic_align.py --frames fixed --a 6 --beta 1 3 --seeds 5
    python synthetic_align.py --label-noise 0.1 --out-name _noisy
"""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from model.mlp.opb import OPB
from model.mlp.utils import flatten

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "synthetic_align"

K = 10
BETAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
A_VALUES = [6.0, 12.0]
FRAMES = ["fixed", "trainable"]
SEEDS = 5

HEADER = [
    "frame_setting", "anchor_scale", "beta", "seed",
    "D_hat_train", "D_hat_test", "beta_D_hat_test",
    "CE_train", "CE_test",
    "KL_mean_test", "KL_var_test", "KL_total_test", "KL_total_train",
    "objective_train", "objective_test",
    "acc_det", "acc_mc",
    "anchor_pair_dist", "post_center_min_dist", "post_center_mean_cos",
    "min_singular_value", "best_epoch", "fail",
]


def make_data(seed=42, n_tr=10000, n_va=2000, n_te=2000, label_noise=0.0):
    """生成 (x, y) 三元组：x = one-hot e_y、均匀类别、独立生成；label_noise>0
    时训练标签按比例随机翻转（测试保持干净，仅控制组用）。"""

    def gen(n, flip=False):
        g = torch.Generator().manual_seed(seed + (7 if flip else 0))
        y = torch.arange(K).repeat((n + K - 1) // K)[:n]
        y = y[torch.randperm(n, generator=g)]
        x = F.one_hot(y, K).float()
        if flip:
            mask = torch.rand(n, generator=g) < label_noise
            idx = mask.nonzero(as_tuple=True)[0]
            y = y.clone()
            y[idx] = torch.randint(0, K, (len(idx),), generator=g)
        return x, y

    return gen(n_tr, flip=label_noise > 0), gen(n_va), gen(n_te)


def build_model(args, frame, a, device):
    """OPB：单层线性编码器（hidden_dims 默认 [d]）、能量分类器、τ²=1 固定；
    fixed 帧另冻结均值块。

    编码器换为纯 Linear(K, r)（无 ReLU/Dropout）：方案要求线性 encoder 保证
    任意 class-constant 后验均值可实现（对齐 comparison solution 显式在模型
    类内）；实测带 ReLU 的单层编码器会产生 dead-ReLU 方向的虚假对齐残差
    （D̂ 停在 ~3 nat 不收敛），线性编码器下 D̂ → 0。后验头、初始化、损失
    与正文 OPB 协议一致。
    """
    hidden = list(args.hidden_dims) if args.hidden_dims is not None else [args.d]
    model = OPB(
        input_dim=K, hidden_dims=tuple(hidden), z_dim=args.d, num_classes=K,
        dropout=0.2, anchor_scale=a, energy_classifier=True,
        fixed_frame=(frame == "fixed"), fixed_prior_var=True,
    )
    model.encoder = torch.nn.Sequential(torch.nn.Linear(K, hidden[-1]))
    return model.to(device)


def train_once(args, model, data, beta, seed, device):
    """单 seed 训练，返回 (best_state_dict, best_epoch)；NaN/发散抛异常。"""
    (x_tr, y_tr), (x_va, y_va), _ = data
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(x_tr)
    best_val, best_state, best_epoch = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for s in range(0, n, args.batch_size):
            b = slice(s, s + args.batch_size)
            logits, kl = model(x_tr[b], y_tr[b], stochastic=True)
            loss = F.cross_entropy(logits, y_tr[b]) + beta * kl
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch} 损失 NaN/Inf")
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits_v, kl_v = model(x_va, y_va, stochastic=False)
            val_obj = F.cross_entropy(logits_v, y_va).item() + beta * kl_v.item()
        if val_obj < best_val:
            best_val = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("无最佳验证 checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return best_epoch


@torch.no_grad()
def evaluate(model, x, y, device, mc_samples):
    """最佳 checkpoint 上的全量指标（整批计算，数据小）。返回指标 dict。"""
    model.eval()
    x, y = x.to(device), y.to(device)
    h = model.encoder(flatten(x))
    mu = model.mu_head(h)
    logvar = model.logvar_head(h)
    prior_mu, logvar_table = model._prior_table()
    mu_p = prior_mu[y]
    logvar_p = logvar_table[y]
    var, var_p = logvar.exp(), logvar_p.exp()
    kl_mean = (0.5 * (mu - mu_p).pow(2) / var_p).sum(1).mean().item()
    kl_var = (0.5 * (var / var_p - 1 - logvar + logvar_p).sum(1)).mean().item()
    logits, _ = model(x, None, stochastic=False)
    ce = F.cross_entropy(logits, y).item()
    acc_det = (logits.argmax(1) == y).float().mean().item()
    probs = torch.zeros_like(logits.softmax(1))
    for _ in range(mc_samples):
        lm, _ = model(x, None, stochastic=True)
        probs += lm.softmax(1)
    acc_mc = (probs.argmax(1) == y).float().mean().item()
    return {
        "d_hat": kl_mean,  # τ²=1 下 D̂ 即均值失配项
        "kl_mean": kl_mean, "kl_var": kl_var, "kl_total": kl_mean + kl_var,
        "ce": ce, "acc_det": acc_det, "acc_mc": acc_mc,
    }


@torch.no_grad()
def frame_diagnostics(model):
    """锚点成对距离、后验类中心几何、先验均值原始矩阵最小奇异值。"""
    prior_mu, _ = model._prior_table()  # (K, d)
    d2 = (prior_mu.unsqueeze(0) - prior_mu.unsqueeze(1)).pow(2).sum(-1)
    pair = d2[torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)]
    anchor_pair_dist = pair.sqrt().mean().item()

    raw = model.prior_net(model.class_eye)[:, : model.num_classes].t()  # (d, K)
    min_sv = torch.linalg.svdvals(raw).min().item()

    # 后验类中心：需输入样本；此处用先验表近似记录占位由 evaluate 传入。
    return anchor_pair_dist, min_sv


@torch.no_grad()
def posterior_centers(model, x, y, device):
    """后验类中心 c_k = mean_i:y_i=k μ(x_i)：最小成对距离与平均成对余弦。"""
    h = model.encoder(flatten(x.to(device)))
    mu = model.mu_head(h)
    y = y.to(device)
    centers = torch.stack([mu[y == k].mean(0) for k in range(K)])  # (K, d)
    c2 = (centers.unsqueeze(0) - centers.unsqueeze(1)).pow(2).sum(-1)
    triu = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    min_dist = c2[triu].sqrt().min().item()
    cn = F.normalize(centers, dim=1)
    cos = (cn @ cn.t())[triu]
    return min_dist, cos.mean().item()


def compute_cg(a, k=K, tau2=1.0, n=1_000_000, seed=1234, device="cpu"):
    """C_G = H(Y|Z_align)：y~Unif(K)、z~N(a·e_y, τ²I)，锚点能量分类器
    log p(y|z) 的负期望。正交帧旋转不变（q_k=e_k 即通用），1e6 MC + 标准误差。
    全 float64：C_G 量级极小（a=6 时 ≈ 9e^{−18}≈1.4e-7、a=12 时 ≈ 9e^{−72}，
    float32 会下溢为 0）。"""
    g = torch.Generator().manual_seed(seed)
    eye = torch.eye(k, dtype=torch.float64) * a  # 锚点均值 a·e_k
    tot, tot2, cnt = 0.0, 0.0, 0
    chunk = 100_000
    while cnt < n:
        m = min(chunk, n - cnt)
        y = torch.randint(0, k, (m,), generator=g)
        z = torch.randn((m, k), dtype=torch.float64, generator=g) + eye[y]
        diff2 = (z.unsqueeze(1) - eye.unsqueeze(0)).pow(2).sum(-1)  # (m, K)
        logits = -diff2 / (2.0 * tau2)
        lp = logits[torch.arange(m), y] - logits.logsumexp(1)
        tot += (-lp).sum().item()
        tot2 += (lp * lp).sum().item()
        cnt += m
    mean = tot / n  # = E[−log p(y|z)] = C_G
    var = tot2 / n - mean * mean  # log p(y|z) 的方差
    se = math.sqrt(max(var, 0.0) / n)
    return mean, se


def main():
    parser = argparse.ArgumentParser(description="合成对齐试验（codex_output.txt 方案）")
    parser.add_argument("--frames", nargs="+", default=FRAMES, choices=FRAMES)
    parser.add_argument("--a", nargs="+", type=float, default=A_VALUES)
    parser.add_argument("--beta", nargs="+", type=float, default=BETAS)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--d", type=int, default=K, help="瓶颈维度（默认 K=10）")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None,
                        help="编码器隐层维度（默认 [d]；容量控制传 2 等）")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="小 β 下 KL 梯度 ∝ β、对齐收敛慢（β=1e-3 约需 600 epoch），"
                        "按验证 objective 早停、收敛即提前结束")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="训练标签翻转比例（控制组，测试干净）")
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--cg-n", type=int, default=10_000_000,
                        help="C_G 为稀有事件期望（指数 n_j−n_y 差），1e6 时 SE≈9%，"
                        "1e7 时 SE≈3%")
    parser.add_argument("--out-name", default="", help="输出文件名后缀（控制组用）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"synthetic_align{args.out_name}.csv"
    cg_path = OUT_DIR / f"cg{args.out_name}.csv"
    data = make_data(label_noise=args.label_noise)
    x_te, y_te = data[2]
    print(f"任务：K={K} 类 one-hot、x=e_y；device={device}；"
          f"frames={args.frames} a={args.a} beta 网格 {len(args.beta)} 点 × {args.seeds} seeds"
          f"{'；label-noise=' + str(args.label_noise) if args.label_noise else ''}")

    rows = []
    objectives = []
    for frame in args.frames:
        for a in args.a:
            for beta in args.beta:
                for seed in range(args.seeds):
                    tag = f"{frame} a={a:g} β={beta:g} seed={seed}"
                    try:
                        model = build_model(args, frame, a, device)
                        best_epoch = train_once(args, model, data, beta, seed, device)
                        tr = evaluate(model, data[0][0], data[0][1], device, args.mc_samples)
                        te = evaluate(model, x_te, y_te, device, args.mc_samples)
                        pair_dist, min_sv = frame_diagnostics(model)
                        c_min_dist, c_mean_cos = posterior_centers(model, x_te, y_te, device)
                        obj_tr = tr["ce"] + beta * tr["kl_total"]
                        obj_te = te["ce"] + beta * te["kl_total"]
                        objectives.append(obj_te)
                        rows.append([
                            frame, a, beta, seed,
                            f"{tr['d_hat']:.6f}", f"{te['d_hat']:.6f}",
                            f"{beta * te['d_hat']:.6f}",
                            f"{tr['ce']:.6f}", f"{te['ce']:.6f}",
                            f"{te['kl_mean']:.6f}", f"{te['kl_var']:.6f}",
                            f"{te['kl_total']:.6f}", f"{tr['kl_total']:.6f}",
                            f"{obj_tr:.6f}", f"{obj_te:.6f}",
                            f"{te['acc_det']:.6f}", f"{te['acc_mc']:.6f}",
                            f"{pair_dist:.6f}", f"{c_min_dist:.6f}", f"{c_mean_cos:.6f}",
                            f"{min_sv:.6f}", str(best_epoch), "",
                        ])
                        print(f"[OK] {tag} D̂={te['d_hat']:.4f} βD̂={beta*te['d_hat']:.4f} "
                              f"CE={te['ce']:.4f} acc_det={te['acc_det']:.4f} "
                              f"acc_mc={te['acc_mc']:.4f} (best ep {best_epoch})")
                    except Exception as e:
                        rows.append([frame, a, beta, seed] + [""] * (len(HEADER) - 5)
                                    + [str(e)])
                        print(f"[失败] {tag}: {e}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"\n结果已保存：{csv_path}（{len(rows)} 行，失败 {sum(1 for r in rows if r[-1])} 行）")

    # C_G：仅主实验（控制组不需要），每 a 一个值 + MC 标准误差
    if not args.label_noise and args.hidden_dims is None:
        with open(cg_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["anchor_scale", "C_G", "mc_se"])
            for a in args.a:
                cg, se = compute_cg(a, n=args.cg_n)
                w.writerow([a, f"{cg:.6f}", f"{se:.6f}"])
                print(f"C_G(a={a:g}) = {cg:.6f} ± {se:.6f}（1e6 MC）")
        print(f"C_G 已保存：{cg_path}")

    if objectives:
        l_min = min(objectives)
        print(f"\nobserved envelope（非理论 L*）：L_min,obs = {l_min:.6f}；"
              f"各行 ε_obs = L − L_min,obs，仅作优化误差参考，不得写成定理 ε。")


if __name__ == "__main__":
    main()
