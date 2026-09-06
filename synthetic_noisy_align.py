"""噪声合成对齐试验（codex_output1.txt 方案）：检验近似可实现性下定理 2.1 的
非零对齐地板 δ_noise。

任务：K=10 类，干净类别 C~Unif(K)、输入 x=e_C（one-hot），观测标签 Y 按对称
噪声率 r 翻转：P(Y=C|C)=1−r、P(Y=j|C)=r/(K−1)。编码器只看 X。

主协议（mode=population）：对全部 K×K 个 (C,Y) 组合按条件概率 η_Y(C) 加权的
**总体目标**训练——L_pop = (1/K)Σ_C Σ_Y η_Y(C)[CE(logits(e_C),Y) + β·KL(q(e_C)‖p(Y))]，
无标签采样噪声、经验 η 与理论 η 严格一致（方案 §2.5 的优先选项）。
敏感性对照（mode=sampled）：按 η 分层采样 10000/2000/2000 样本逐样本训练，
单独报告经验 η 与理论的偏差（方案 §2.5 的后备选项）。

模型：OPB（model/mlp/opb.py）+ 锚点能量分类器；a=6、τ²=1 固定（fixed_prior_var）；
frame 两设置——fixed（冻结恒等帧，锚点恒为 a·e_k，主结果）与 trainable
（stateless polar frame，附录稳健性）；posterior_var 两模式——paper（可学习
logvar 头，论文模型）与 fixed（冻结 logvar 头于零，σ²≡τ²=1，V_0=0 的最干净
对比）；K=d=10、单层线性编码器（任意 class-constant 后验均值可实现）；β 九点
网格 {1e-3..10} × 5 seeds。

解析 comparator（方案 §4）：q_0(z|e_C)=N(h_0(e_C), τ²I)，
h_0(e_C) = a[(1−r)q_C + r/(K−1)Σ_{j≠C}q_j]，covariance mismatch 为 0；
δ_noise(r) = a²/(2τ²)·[1−(1−r)²−r²/(K−1)]（闭式）；
C_comp = 能量头在 q_0 下的预测风险（r=0 复用 synthetic_align.cg.csv 的 C_G，
r>0 用 float64 MC 1e6 + 标准误；logits_k = a·z_k/τ²，‖z‖² 项在 softmax 中消去）；
理论参考曲线 δ_noise + C_comp/β。每 run 记录 L̂ − L_comp，L̂ > L_comp + tol
（tol=0.02）记 opt_fail=1——该点不得用于上界讨论（方案 §7.3）。

checkpoint：主行 = 总体 objective 最小 checkpoint（方案 §3.6），final-epoch
checkpoint 作为同 CSV 的敏感性行；hit_epoch_cap=1 表示收敛前触顶（方案 §3.5
的"系统下降趋势"警告）。每类 D_k 存独立 detail CSV（方案 §5）。

输出：output/synthetic_noisy_align/ 下
  synthetic_noisy_align.csv   主结果（每 run × {best, final} 两行）
  synthetic_noisy_align_dk.csv 每类 D_k（population 模式 best checkpoint）
  reference.csv              每 (r, β)：δ_noise、C_comp±SE、L_comp
  summary.csv                每 (mode, frame, var, r, β) 的 mean±std 聚合
论文图由 synthetic_noisy_align_plot.py 生成。

用法：
    python synthetic_noisy_align.py                          # 全网格 675 run
    python synthetic_noisy_align.py --workers 24             # 并行进程数
    python synthetic_noisy_align.py --skip-trainable --skip-sampled   # 仅主网格
"""

import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn.functional as F

from model.mlp.opb import OPB
from model.mlp.utils import flatten, reparameterize

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "synthetic_noisy_align"

K = 10
BETAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
RATES = [0.0, 0.1, 0.2, 0.4]
A = 6.0
TAU2 = 1.0
OPT_TOL = 1.0  # L̂ − L_comp 判定容差（方案 §7.3：超出标 opt_fail）。
# 说明：随机路径 CE 的逐样本方差大（翻转标签事件一次付 ~28 nats），MC-2000 下
# SE≈0.25；1.0 nat 容差 = 4 SE，足以捕获真正的优化失败（如 smoke test 的 +6.5）
# 而不误伤收敛点。

HEADER = [
    "mode", "frame_type", "posterior_var_mode", "noise_rate", "beta",
    "anchor_scale", "seed", "checkpoint", "best_epoch", "total_epochs",
    "hit_epoch_cap",
    "d_hat", "kl_var", "kl_total", "consistency_check",
    "trained_risk", "trained_risk_mc", "trained_objective",
    "trained_objective_mc",
    "delta_noise", "comparator_risk", "comparator_objective",
    "L_hat_minus_L_comp", "opt_fail",
    "excess_d", "beta_excess_d",
    "noisy_label_acc", "clean_class_acc", "noisy_label_acc_mc",
    "min_center_distance", "mean_center_distance", "mean_center_anchor_err",
    "min_sigma_u", "raw_ortho_dev", "eta_dev", "fail",
]
DK_HEADER = ["seed", "noise_rate", "beta", "frame_type",
             "posterior_var_mode"] + [f"D_{k}" for k in range(K)]


def eta_matrix(r, dtype=torch.float32):
    """η[C,Y] = P(Y|C)：对角 1−r、非对角 r/(K−1)。"""
    eta = torch.full((K, K), r / (K - 1), dtype=dtype)
    eta.fill_diagonal_(1.0 - r)
    return eta


def delta_noise(r, a=A, tau2=TAU2, k=K):
    """闭式歧义地板：δ_noise(r) = a²/(2τ²)·[1−(1−r)²−r²/(k−1)]（方案 §4）。"""
    return a * a / (2.0 * tau2) * (1.0 - (1.0 - r) ** 2 - r * r / (k - 1))


def compute_c_comp_noisy(r, a=A, k=K, tau2=TAU2, n=1_000_000, seed=1234):
    """C_comp(r>0)：q_0(z|e_C)=N(h_0(e_C),τ²I) 在锚点能量头下的预测风险，
    float64 MC + 标准误。logits_k = a·z_k/τ²（‖z‖² 在 softmax 中消去）。"""
    eta = eta_matrix(r, dtype=torch.float64)
    h0 = a * eta  # (K,K)：h0[C] = a·η[C]
    g = torch.Generator().manual_seed(seed)
    tot = tot2 = 0.0
    cnt = 0
    chunk = 200_000
    while cnt < n:
        m = min(chunk, n - cnt)
        C = torch.randint(0, k, (m,), generator=g)
        Ys = torch.multinomial(eta, m, replacement=True, generator=g)  # (K,m)
        Y = Ys[C, torch.arange(m)]  # (m,)
        z = torch.randn((m, k), dtype=torch.float64, generator=g) * math.sqrt(tau2) + h0[C]
        logits = a * z / tau2
        lp = logits - logits.logsumexp(1, keepdim=True)
        v = -lp[torch.arange(m), Y]
        tot += v.sum().item()
        tot2 += (v * v).sum().item()
        cnt += m
    mean = tot / n
    var = tot2 / n - mean * mean
    return mean, math.sqrt(max(var, 0.0) / n)


def load_cg():
    """r=0 的 C_comp 复用确定性实验的 C_G（synthetic_align 的 cg.csv，
    存在则直接读，否则现算）。返回 {a: (C_G, se)}。"""
    from synthetic_align import compute_cg
    cg = {}
    path = ROOT / "output" / "synthetic_align" / "cg.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cg[float(row["anchor_scale"])] = (float(row["C_G"]), float(row["mc_se"]))
    if A not in cg:
        cg[A] = compute_cg(A, k=K, tau2=TAU2, n=10_000_000)
    return cg


def build_model(frame, var_mode, d=K, a=A):
    """OPB：单层线性编码器（class-constant 后验均值可实现）、能量分类器、
    τ²=1 固定；fixed 帧冻结均值块（锚点恒为 a·e_k）；posterior_var=fixed
    时冻结 logvar 头（σ²≡τ²=1、V_0=0）。"""
    model = OPB(
        input_dim=K, hidden_dims=(d,), z_dim=d, num_classes=K,
        dropout=0.0, anchor_scale=a, energy_classifier=True,
        fixed_frame=(frame == "fixed"), fixed_prior_var=True,
    )
    model.encoder = torch.nn.Sequential(torch.nn.Linear(K, d))
    if var_mode == "fixed":
        model.logvar_head.requires_grad_(False)
    return model


@torch.no_grad()
def pop_stats(model, eta):
    """μ 路径总体统计：返回 (ce, kl_mean, kl_var)。kl_mean 即 D̂（var_p=τ²=1）。"""
    model.eval()
    eye = torch.eye(K)
    h = model.encoder(eye)
    mu = model.mu_head(h)
    logvar = model.logvar_head(h)
    prior_mu, prior_lv = model._prior_table()
    logits = model._energy_logits(mu, prior_mu, prior_lv)
    logp = F.log_softmax(logits, dim=1)
    ce = -(logp * eta).sum(1).mean().item()
    var = logvar.exp()
    var_p = prior_lv.exp()
    dmat = 0.5 * (mu.unsqueeze(1) - prior_mu.unsqueeze(0)).pow(2).sum(-1) / var_p.unsqueeze(0)
    vmat = 0.5 * (var.unsqueeze(1) / var_p.unsqueeze(0) - 1 - logvar.unsqueeze(1)
                  + prior_lv.unsqueeze(0)).sum(-1)
    kl_mean = (dmat * eta).sum(1).mean().item()
    kl_var = (vmat * eta).sum(1).mean().item()
    return ce, kl_mean, kl_var


def train_pop(model, beta, eta, epochs, patience, lr):
    """总体目标全批训练（每 epoch 一步）。

    早停按 D̂（均值失配项）判定：它是本试验的标题量、确定性无噪，且在
    σ 收缩的陷阱区仍单调（KL 总和的 D̂↓ 与 V↑ 互相抵消会误停）。小 β
    下 CE 尾爬缓慢，epoch 上限由 main 按 β 缩放（β≤0.01 用 60000），
    未收敛时 hit_epoch_cap=1。配合 main 的降序 β 链式温启动使用。
    返回 (best_state, final_state, best_epoch, total_epochs)；NaN 抛异常。
    """
    eye = torch.eye(K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_d, best_state, best_epoch = math.inf, None, 0
    final_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        h = model.encoder(eye)
        mu = model.mu_head(h)
        logvar = model.logvar_head(h)
        z = reparameterize(mu, logvar, True)  # 每类采样一次、共享给全部 Y 项
        prior_mu, prior_lv = model._prior_table()
        logits = model._energy_logits(z, prior_mu, prior_lv)
        logp = F.log_softmax(logits, dim=1)
        ce = -(logp * eta).sum(1)
        var = logvar.exp()
        var_p = prior_lv.exp()
        kl_mat = torch.zeros(K, K)
        for y in range(K):
            kl_mat[:, y] = 0.5 * ((mu - prior_mu[y]).pow(2) / var_p[y]
                                  + var / var_p[y] - 1 - logvar + prior_lv[y]).sum(1)
        loss = ((ce + beta * (kl_mat * eta).sum(1)).mean())
        if not torch.isfinite(loss):
            raise RuntimeError(f"epoch {epoch} 损失 NaN/Inf")
        opt.zero_grad()
        loss.backward()
        opt.step()
        ce_e, kl_m, kl_v = pop_stats(model, eta)
        if kl_m < best_d - 1e-5:
            best_d = kl_m
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break
        final_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("无最佳 checkpoint")
    return best_state, final_state, best_epoch, epoch


@torch.no_grad()
def eval_pop(model, eta, mc):
    """总体分布上的全量指标（μ 路径 + MC 采样路径）。"""
    model.eval()
    eye = torch.eye(K)
    h = model.encoder(eye)
    mu = model.mu_head(h)
    logvar = model.logvar_head(h)
    prior_mu, prior_lv = model._prior_table()
    var = logvar.exp()
    var_p = prior_lv.exp()
    dmat = 0.5 * (mu.unsqueeze(1) - prior_mu.unsqueeze(0)).pow(2).sum(-1) / var_p.unsqueeze(0)
    vmat = 0.5 * (var.unsqueeze(1) / var_p.unsqueeze(0) - 1 - logvar.unsqueeze(1)
                  + prior_lv.unsqueeze(0)).sum(-1)
    d_hat = (dmat * eta).sum(1).mean().item()
    kl_var = (vmat * eta).sum(1).mean().item()
    # 每类 D_k：条件在观测类别 k（方案 §5）
    w = eta.sum(0)
    d_k = ((dmat * eta).sum(0) / w).tolist()
    logits = model._energy_logits(mu, prior_mu, prior_lv)
    logp = F.log_softmax(logits, dim=1)
    ce = -(logp * eta).sum(1).mean().item()
    pred = logits.argmax(1)
    noisy_acc = eta[torch.arange(K), pred].mean().item()
    clean_acc = (pred == torch.arange(K)).float().mean().item()
    ce_mc = 0.0
    probs = torch.zeros(K, K)
    for _ in range(mc):
        z = reparameterize(mu, logvar, True)
        lm = model._energy_logits(z, prior_mu, prior_lv)
        lp = F.log_softmax(lm, dim=1)
        ce_mc += -(lp * eta).sum(1).mean().item()
        probs += lm.softmax(1)
    ce_mc /= mc
    probs /= mc
    pred_mc = probs.argmax(1)
    noisy_acc_mc = eta[torch.arange(K), pred_mc].mean().item()
    # 后验类中心几何 + center-to-conditional-anchor 误差
    c2 = (mu.unsqueeze(0) - mu.unsqueeze(1)).pow(2).sum(-1)
    triu = torch.triu(torch.ones(K, K, dtype=torch.bool), 1)
    h0 = A * eta  # (K,K)：h0[C] = a·η[C]
    return {
        "d_hat": d_hat, "kl_var": kl_var, "kl_total": d_hat + kl_var,
        "ce": ce, "ce_mc": ce_mc, "noisy_acc": noisy_acc,
        "clean_acc": clean_acc, "noisy_acc_mc": noisy_acc_mc,
        "min_cd": c2[triu].sqrt().min().item(),
        "mean_cd": c2[triu].sqrt().mean().item(),
        "ca_err": (mu - h0).norm(dim=1).mean().item(),
        "d_k": d_k,
    }


@torch.no_grad()
def frame_diag(model):
    """QR 前原始均值矩阵的最小奇异值与正交偏离（fixed 帧恒为 1/0）。"""
    raw = model.prior_net(model.class_eye)[:, :K].t()  # (d,K)
    min_sv = torch.linalg.svdvals(raw.double()).min().item()
    ortho_dev = (raw.t() @ raw - torch.eye(K)).norm().item()
    return min_sv, ortho_dev


def make_sampled_data(r, seed, n_tr=10000, n_va=2000, n_te=2000):
    """分层采样数据：C~Unif、Y~η(C) 逐样本采样，测试同时保留 C 与 Y。"""
    g = torch.Generator().manual_seed(1000 + seed)

    def gen(n):
        C = torch.randint(0, K, (n,), generator=g)
        x = F.one_hot(C, K).float()
        # 逐样本从各自类别行抽样（勿用 [C] 索引共享的 K 行抽签表：那会让
        # 同类别样本共享同一标签，η_emp 退化为置换矩阵）
        Y = torch.multinomial(eta_matrix(r)[C], 1, generator=g).squeeze(-1)
        return x, Y, C

    return gen(n_tr), gen(n_va), gen(n_te)


def train_sampled(model, data, beta, epochs, patience, lr):
    """采样标签逐样本训练（minibatch），按验证 objective 早停。"""
    (x_tr, y_tr, _), (x_va, y_va, _), _ = data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, best_epoch = math.inf, None, 0
    final_state = None
    n = len(x_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        for s in range(0, n, 512):
            b = slice(s, s + 512)
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
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            best_epoch = epoch
        if epoch - best_epoch >= patience:
            break
        final_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("无最佳 checkpoint")
    return best_state, final_state, best_epoch, epoch


@torch.no_grad()
def eval_sampled(model, x, y, c, mc):
    """经验测试分布上的指标（μ 路径 + MC）。"""
    model.eval()
    h = model.encoder(x)
    mu = model.mu_head(h)
    logvar = model.logvar_head(h)
    prior_mu, prior_lv = model._prior_table()
    mu_p = prior_mu[y]
    lv_p = prior_lv[y]
    var = logvar.exp()
    var_p = lv_p.exp()
    d_hat = 0.5 * ((mu - mu_p).pow(2) / var_p).sum(1).mean().item()
    kl_var = 0.5 * (var / var_p - 1 - logvar + lv_p).sum(1).mean().item()
    logits = model._energy_logits(mu, prior_mu, prior_lv)
    ce = F.cross_entropy(logits, y).item()
    noisy_acc = (logits.argmax(1) == y).float().mean().item()
    clean_acc = (logits.argmax(1) == c).float().mean().item()
    ce_mc = 0.0
    probs = torch.zeros_like(logits.softmax(1))
    for _ in range(mc):
        z = reparameterize(mu, logvar, True)
        lm = model._energy_logits(z, prior_mu, prior_lv)
        ce_mc += F.cross_entropy(lm, y).item()
        probs += lm.softmax(1)
    ce_mc /= mc
    probs /= mc
    noisy_acc_mc = (probs.argmax(1) == y).float().mean().item()
    return {"d_hat": d_hat, "kl_var": kl_var, "kl_total": d_hat + kl_var,
            "ce": ce, "ce_mc": ce_mc, "noisy_acc": noisy_acc,
            "clean_acc": clean_acc, "noisy_acc_mc": noisy_acc_mc}


def make_row(mode, frame, var_mode, r, beta, seed, checkpoint, best_epoch,
             total_epochs, hit_cap, m, delta, c_comp, l_comp, diag, eta_dev, fail=""):
    l_hat = m["ce_mc"] + beta * m["kl_total"]
    excess = m["d_hat"] - delta
    opt_fail = 1 if (l_hat - l_comp) > OPT_TOL else 0
    return [
        mode, frame, var_mode, f"{r:g}", f"{beta:g}", A, seed, checkpoint,
        str(best_epoch), str(total_epochs), str(hit_cap),
        f"{m['d_hat']:.6f}", f"{m['kl_var']:.6f}", f"{m['kl_total']:.6f}",
        f"{m['kl_total'] - m['d_hat'] - m['kl_var']:.2e}",
        f"{m['ce']:.6f}", f"{m['ce_mc']:.6f}",
        f"{m['ce'] + beta * m['kl_total']:.6f}",
        f"{l_hat:.6f}",
        f"{delta:.6f}", f"{c_comp:.6f}", f"{l_comp:.6f}",
        f"{l_hat - l_comp:.6f}", str(opt_fail),
        f"{excess:.6f}", f"{beta * excess:.6f}",
        f"{m['noisy_acc']:.6f}", f"{m['clean_acc']:.6f}", f"{m['noisy_acc_mc']:.6f}",
        f"{m.get('min_cd', float('nan')):.6f}", f"{m.get('mean_cd', float('nan')):.6f}",
        f"{m.get('ca_err', float('nan')):.6f}",
        f"{diag[0]:.6f}", f"{diag[1]:.6f}", f"{eta_dev:.6f}", fail,
    ]


def run_pop_chain(task, ref):
    """降序 β 链式温启动：每个 (frame, var, r, seed) 从 β=10 训练到 β=1e-3，
    每级以前一级的 best checkpoint 为起点。小 β 的 σ→0 捷径会让 CE 梯度
    饱和、μ 对齐只能靠 β·KL 项以 ~1e-6/步爬行；温启动使每级只需在邻域
    内微调（同种子同族目标的续训，每级最终目标只可能更低）。
    返回 (rows, dk_rows)。"""
    frame, var_mode, r, seed = task
    torch.manual_seed(seed)
    delta, c_comp, _ = ref[r]
    eta = eta_matrix(r)
    rows, dk = [], []
    try:
        model = build_model(frame, var_mode)
        state = None
        for beta in sorted(BETAS, reverse=True):
            l_comp = c_comp + beta * delta
            epochs = EPOCHS_SMALL if beta <= 0.01 else (EPOCHS_MID if beta <= 0.03 else EPOCHS)
            if state is not None:
                model.load_state_dict(state)
            best_s, fin_s, best_ep, tot_ep = train_pop(
                model, beta, eta, epochs, PATIENCE, LR)
            hit = 1 if tot_ep >= epochs else 0
            for cp, st in (("best", best_s), ("final", fin_s)):
                model.load_state_dict(st)
                m = eval_pop(model, eta, MC)
                diag = frame_diag(model)
                rows.append(make_row("population", frame, var_mode, r, beta, seed,
                                     cp, best_ep, tot_ep, hit, m, delta, c_comp,
                                     l_comp, diag, 0.0))
            model.load_state_dict(best_s)
            m = eval_pop(model, eta, MC)
            dk.append([seed, f"{r:g}", f"{beta:g}", frame, var_mode] + m["d_k"])
            state = best_s  # 温启动：下一级 β 从这里续训
    except Exception as e:
        beta = 0.0  # 失败行占位（链中途失败时剩余 β 不重复记行）
        rows.append(["population", frame, var_mode, f"{r:g}", str(beta), A, seed,
                     "", "", "", ""] + [""] * (len(HEADER) - 11) + [str(e)])
    return rows, dk


def run_sampled_task(task, ref):
    """采样标签敏感性单任务（按 (r, β, seed) 独立训练）。"""
    r, beta, seed = task
    torch.manual_seed(seed)
    delta, c_comp, _ = ref[r]
    l_comp = c_comp + beta * delta
    try:
        model = build_model("fixed", "paper")
        data = make_sampled_data(r, seed)
        best_s, fin_s, best_ep, tot_ep = train_sampled(
            model, data, beta, EPOCHS_S, PATIENCE_S, LR)
        hit = 1 if tot_ep >= EPOCHS_S else 0
        # 经验 η 与理论 η 的偏差（方案 §2.5）：逐对计数须用
        # index_put_(accumulate=True)——fancy-index 原地赋值/= 对重复
        # 索引不累加（经典缓冲问题），会得到错误的 η_emp
        (x_tr, y_tr, c_tr), _, (x_te, y_te, c_te) = data
        eta_emp = torch.zeros(K, K)
        eta_emp.index_put_((c_tr, y_tr), torch.ones_like(c_tr, dtype=eta_emp.dtype),
                           accumulate=True)
        eta_emp = eta_emp / eta_emp.sum(1, keepdim=True)
        eta_dev = (eta_emp - eta_matrix(r)).abs().max().item()
        rows = []
        for cp, st in (("best", best_s), ("final", fin_s)):
            model.load_state_dict(st)
            m = eval_sampled(model, x_te, y_te, c_te, MC)
            diag = frame_diag(model)
            rows.append(make_row("sampled", "fixed", "paper", r, beta, seed, cp,
                                 best_ep, tot_ep, hit, m, delta, c_comp,
                                 l_comp, diag, eta_dev))
        return rows, []
    except Exception as e:
        return [["sampled", "fixed", "paper", f"{r:g}", f"{beta:g}", A, seed,
                 "", "", "", ""] + [""] * (len(HEADER) - 11) + [str(e)]], []


def run_task(task, ref):
    """任务分派：population 走降序 β 链式温启动、sampled 走独立单任务。"""
    if task[0] == "population":
        return run_pop_chain(task[1:], ref)
    return run_sampled_task(task[1:], ref)


EPOCHS = 20000      # 总体全批（1 步/epoch）的 β≥0.1 上限；更小 β 用缩放上限
EPOCHS_MID = 30000  # β=0.03 的 epoch 上限
EPOCHS_SMALL = 60000  # β≤0.01 的 epoch 上限（CE 尾爬缓慢）
PATIENCE = 2000     # 早停按确定性的 D̂（KL 总和的 D̂↓ 与 σ 收缩的 V↑ 互相抵消
                    # 会误停）；小 β 尾爬 2000 步无新低即停
EPOCHS_S = 2000     # 采样模式（20 步/epoch，与原合成实验同预算）
PATIENCE_S = 100    # 采样模式按验证 objective 早停（β=10 下 30 步会误停）
LR = 1e-3
MC = 2000


def _init_worker():
    torch.set_num_threads(1)


def main():
    global EPOCHS, PATIENCE, EPOCHS_S, PATIENCE_S, MC
    parser = argparse.ArgumentParser(description="噪声合成对齐试验（codex_output1.txt 方案）")
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--beta", nargs="+", type=float, default=BETAS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=MC,
                        help="随机路径风险估计的采样数（翻转标签事件方差大，"
                        "默认 2000）")
    parser.add_argument("--skip-trainable", action="store_true",
                        help="跳过 trainable frame 附录网格")
    parser.add_argument("--skip-sampled", action="store_true",
                        help="跳过采样标签敏感性网格")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--epochs-sampled", type=int, default=EPOCHS_S)
    parser.add_argument("--patience-sampled", type=int, default=PATIENCE_S)
    args = parser.parse_args()
    EPOCHS, PATIENCE = args.epochs, args.patience
    EPOCHS_S, PATIENCE_S = args.epochs_sampled, args.patience_sampled
    MC = args.mc_samples

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 解析参考量（方案 §4）：δ_noise 闭式、C_comp（r=0 用 C_G、r>0 MC）
    cg = load_cg()
    cg_val, cg_se = cg[A]
    ref = {}
    ref_rows = []
    print("解析 comparator 参考量：")
    for r in args.rates:
        delta = delta_noise(r)
        if r == 0.0:
            c_comp, c_se = cg_val, cg_se
        else:
            c_comp, c_se = compute_c_comp_noisy(r)
        ref[r] = (delta, c_comp, c_se)
        for beta in args.beta:
            ref_rows.append([f"{r:g}", f"{beta:g}", f"{delta:.8f}",
                             f"{c_comp:.8f}", f"{c_se:.3e}",
                             f"{c_comp + beta * delta:.8f}"])
        print(f"  r={r:g}: δ_noise={delta:.6f}  C_comp={c_comp:.6f}±{c_se:.2e}")

    # 运行矩阵：population 按 (frame, var, r, seed) 分链（链内 β 降序温启动），
    # sampled 按 (r, β, seed) 独立训练
    tasks = []
    for frame in ["fixed"] + ([] if args.skip_trainable else ["trainable"]):
        var_modes = ["paper", "fixed"] if frame == "fixed" else ["paper"]
        for var in var_modes:
            for r in args.rates:
                for seed in range(args.seeds):
                    tasks.append(("population", frame, var, r, seed))
    if not args.skip_sampled:
        for r in args.rates:
            if r == 0.0:
                continue  # r=0 采样模式即既有确定性实验，跳过
            for beta in args.beta:
                for seed in range(args.seeds):
                    tasks.append(("sampled", r, beta, seed))

    print(f"运行矩阵：{len(tasks)} run（{args.workers} 进程并行，CPU）")
    global REF
    REF = ref
    rows, dk_rows = [], []
    done = 0
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_run_wrapper, tasks, chunksize=1):
            rows.extend(res[0])
            dk_rows.extend(res[1])
            done += 1
            if done % 50 == 0 or done == len(tasks):
                fails = sum(1 for r0 in rows if r0[-1])
                print(f"[进度] {done}/{len(tasks)}（累计失败 {fails}）")

    with open(OUT_DIR / "reference.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_rate", "beta", "delta_noise", "comparator_risk",
                    "comparator_risk_se", "comparator_objective"])
        w.writerows(ref_rows)
    with open(OUT_DIR / "synthetic_noisy_align.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    with open(OUT_DIR / "synthetic_noisy_align_dk.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DK_HEADER)
        w.writerows(dk_rows)
    n_fail = sum(1 for r0 in rows if r0[-1])
    n_opt = sum(1 for r0 in rows if r0[23] == "1")
    print(f"\n结果已保存：{OUT_DIR}")
    print(f"  synthetic_noisy_align.csv   {len(rows)} 行（失败 {n_fail}，opt_fail {n_opt}）")
    print(f"  synthetic_noisy_align_dk.csv {len(dk_rows)} 行（每类 D_k）")
    print(f"  reference.csv               {len(ref_rows)} 行（δ_noise/C_comp/L_comp）")


def _run_wrapper(task):
    """进程池入口：ref 由全局 REF 传入（fork 继承）。"""
    return run_task(task, REF)


REF = {}


if __name__ == "__main__":
    main()
