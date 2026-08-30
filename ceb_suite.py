"""CEB 匹配套件（第五轮，审稿人建议的最小 matched experiment package，
按用户约束收窄为 MNIST/MLP、CEB vs FGIB-S 两模型 + Det 参照）：

分析性试验，回答"FGIB 相比 CEB 的 robust-generalization 性质发生了什么"：
1. clean accuracy（多 β；CEB 均值部署、FGIB-S 随机部署 + 均值部署对照）；
2. random-label 记忆（--random-labels，100 epochs 无早停，最终 train acc）；
3. targeted / untargeted PGD-L∞（ε=0.3，40 步）；
4. OOD（均匀噪声，无标签分数：max-softmax / CEB 预测标签 rate /
   FGIB-S anchor-distance min_y KL(q‖N(a v_y, τ²I))）；
5. calibration（ECE 15 bins / NLL / Brier）。

用法：
  python ceb_suite.py --train   # 并发启动训练（train.py 子进程，输出 output/ceb_suite/）
  python ceb_suite.py --eval    # 加载 checkpoint 评估并生成 tables/ceb_suite.tex
"""

import argparse
import concurrent.futures
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from datasets.datasets import get_mnist_dataloaders
from model import CEB, FGIBS, MLP
from model.mlp.utils import flatten, kl_divergence

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "ceb_suite"
BETAS = ["0.001", "0.01", "0.1", "1"]
BETA_F = {"0.001": 1e-3, "0.01": 1e-2, "0.1": 1e-1, "1": 1.0}
A = 16.0
N_CLS = 10


# ---------------------------------------------------------------- 训练阶段
def build_cmds():
    """(任务名, save 前缀, train.py 参数列表)。"""
    cmds = []
    cmds.append(("det", ["--model", "mlp"]))
    for b in BETAS:
        cmds.append((f"ceb_b{b}", ["--model", "ceb", "--beta", b]))
        cmds.append((f"fgibs_b{b}", ["--model", "fgibs", "--beta", b,
                                    "--anchor-scale", "16"]))
    for m in ("ceb", "fgibs"):
        for b in ("0.001", "0.1"):
            cmds.append((f"rand_{m}_b{b}", ["--model", m, "--beta", b,
                                            "--anchor-scale", "16"] if m == "fgibs"
                         else ["--model", m, "--beta", b]))
    return cmds


def train_phase(parallel: int):
    OUT.mkdir(parents=True, exist_ok=True)
    cmds = build_cmds()
    gpu = 0

    def run(job):
        nonlocal gpu
        name, extra = job
        gpu = (gpu + 1) % max(1, _n_gpus())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        if name.startswith("rand_"):
            cmd = [sys.executable, "train.py", "--task", "mnist", "--runs", "5",
                   "--epochs", "100", "--patience", "1000", "--random-labels",
                   "--log-path", str(OUT / f"{name}.log")] + extra
        else:
            cmd = [sys.executable, "train.py", "--task", "mnist", "--runs", "5",
                   "--save-path", str(OUT / f"{name}.pt"),
                   "--log-path", str(OUT / f"{name}.log")] + extra
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        return name, r.returncode, (r.stderr or r.stdout)[-300:]

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
        for name, code, tail in ex.map(run, cmds):
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"[train] {name:16s} {status}")
            if code != 0:
                print("   ", tail.replace(chr(10), " | ")[:200])
    print("训练阶段完成 →", OUT)


def _n_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        return max(1, len(out.splitlines()))
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------- 评估阶段
def load_model(name, run=1):
    if name == "det":
        model = MLP()
    elif name.startswith("ceb"):
        model = CEB()
    else:  # fgibs
        model = FGIBS(anchor_scale=A)
    path = OUT / f"{name}_run{run}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 checkpoint: {path}")
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def eval_acc(model, loader, device, stochastic):
    """CEB 用 μ 部署（stochastic=False），FGIB-S 用采样 z（stochastic=True）。"""
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, FGIBS):
            logits, _ = model(x, None, stochastic=stochastic)
        elif isinstance(model, CEB):
            logits, _ = model(x, None, stochastic=False)
        else:
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def forward_logits(model, x, stochastic):
    if isinstance(model, FGIBS):
        return model(x, None, stochastic=stochastic)[0]
    if isinstance(model, CEB):
        return model(x, None, stochastic=False)[0]
    return model(x)


def pgd(model, x, y, device, targeted, eps=0.3, alpha=0.01, steps=40):
    """PGD-L∞；untargeted 最大化 CE(logits, y)，targeted 最小化 CE(logits, (y+1)%10)。
    攻击部署对象：CEB 攻 μ，FGIB-S 攻采样 z。返回 (robust_acc, success_rate)。"""
    x_adv = x.clone().detach()
    target = (y + 1) % N_CLS
    model.eval()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = forward_logits(model, x_adv, stochastic=isinstance(model, FGIBS))
        loss = (F.cross_entropy(logits, target) if targeted
                else -F.cross_entropy(logits, y))
        g, = torch.autograd.grad(loss, x_adv)
        with torch.no_grad():
            x_adv = x_adv + (alpha * g.sign() if not targeted else -alpha * g.sign())
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0, 1)
    with torch.no_grad():
        logits = forward_logits(model, x_adv, stochastic=isinstance(model, FGIBS))
        pred = logits.argmax(1)
    robust = (pred == y).float().mean().item()
    success = 1.0 - robust
    return robust, success


@torch.no_grad()
def ood_scores(models, loader_in, loader_out, device):
    """OOD=均匀噪声；分数：max-softmax（所有模型）、CEB 预测标签 rate、
    FGIB-S anchor-distance（min_y KL，无标签）。返回 {模型名: {分数名: AUROC}}。"""
    out = {}
    for mname, model in models.items():
        msp_in, msp_out = [], []
        rate_in, rate_out = [], []
        anch_in, anch_out = [], []
        for loader, is_in in ((loader_in, True), (loader_out, False)):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = forward_logits(model, x, stochastic=isinstance(model, FGIBS))
                msp = logits.softmax(1).max(1).values.cpu()
                (msp_in if is_in else msp_out).append(msp)
                if isinstance(model, CEB):
                    # 逐样本 rate（CEB.forward 的 kl 是 batch 平均标量，OOD 需要逐样本）
                    h = model.encoder(flatten(x))
                    mu_q = model.mu_head(h)
                    lv_q = model.logvar_head(h)
                    y_feat = torch.nn.functional.one_hot(
                        logits.argmax(1), num_classes=N_CLS).float()
                    mu_p, lv_p = model.prior_net(y_feat).chunk(2, dim=1)
                    lv_q = lv_q.clamp(-10, 10)
                    lv_p = lv_p.clamp(-10, 10)
                    per_sample = 0.5 * (lv_p - lv_q + (lv_q - lv_p).exp()
                                        + (mu_q - mu_p).pow(2) * (-lv_p).exp() - 1).sum(1)
                    (rate_in if is_in else rate_out).append(per_sample.cpu())
                if isinstance(model, FGIBS):
                    # 逐样本 anchor-distance：min_y KL(q‖N(a·v_y, I))（无标签分数）
                    xf = flatten(x)
                    mu = model.mu_head(model.encoder(xf))
                    lv = model.logvar_head(mu).clamp(-10, 10)
                    per_k = []
                    for k in range(N_CLS):
                        d_k = 0.5 * (lv.exp() + (mu - model.prior_mu[k]).pow(2)
                                     - 1 - lv).sum(1)  # τ²=1, lv_p=0
                        per_k.append(d_k)
                    dist = torch.stack(per_k).min(0).values  # (B,)
                    (anch_in if is_in else anch_out).append(dist.cpu())
        msp_in_t, msp_out_t = torch.cat(msp_in), torch.cat(msp_out)
        labels = torch.cat([torch.ones(len(msp_in_t)), torch.zeros(len(msp_out_t))])
        scores = torch.cat([msp_in_t, msp_out_t]).numpy()
        res = {"maxsoftmax": roc_auc_score(labels.numpy(), scores)}
        if rate_in:
            r_in, r_out = torch.cat(rate_in), torch.cat(rate_out)
            rs = torch.cat([r_in, r_out]).numpy()
            res["rate"] = roc_auc_score(labels.numpy(), -rs)  # 异常分数越高越 OOD
        if anch_in:
            a_in, a_out = torch.cat(anch_in), torch.cat(anch_out)
            as_ = torch.cat([a_in, a_out]).numpy()
            res["anchor"] = roc_auc_score(labels.numpy(), as_)
        out[mname] = res
    return out


@torch.no_grad()
def calibration(model, loader, device):
    """ECE（15 bins）/ NLL / Brier，按模型部署约定。"""
    n_bins = 15
    bin_conf = torch.zeros(n_bins)
    bin_acc = torch.zeros(n_bins)
    bin_n = torch.zeros(n_bins)
    nll, brier, total = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = forward_logits(model, x, stochastic=isinstance(model, FGIBS))
        prob = logits.softmax(1)
        conf, pred = prob.max(1)
        idx = (conf * n_bins).long().clamp(0, n_bins - 1).cpu()
        onehot = F.one_hot(y, N_CLS).float()
        bin_conf.scatter_add_(0, idx, conf.cpu())
        bin_acc.scatter_add_(0, idx, (pred == y).float().cpu())
        bin_n.scatter_add_(0, idx, torch.ones_like(conf).cpu())
        nll += F.cross_entropy(logits, y, reduction="sum").item()
        brier += ((prob - onehot) ** 2).sum().item()
        total += y.numel()
    ece = ((bin_conf - bin_acc).abs() / bin_n.clamp(min=1)).sum().item() / n_bins
    return ece, nll / total, brier / total


def parse_clean_acc(name):
    """从训练日志取 5 runs 平均测试 Acc。"""
    log = OUT / f"{name}.log"
    if not log.is_file():
        return None
    for line in reversed(log.read_text().splitlines()):
        m = re.search(r"Average over 5 runs \| Test Loss [\d.]+±[\d.]+ Acc ([\d.]+)±", line)
        if m:
            return 100 * float(m.group(1))
    return None


def parse_rand_train_acc(name):
    """random-label 日志：最终 epoch 的 Train Acc 均值±标准差（train.py 汇总行
    "Average over 5 runs | Train Acc (final epoch) ...±..."）。"""
    log = OUT / f"rand_{name}.log"
    if not log.is_file():
        return None
    txt = log.read_text()
    m = re.search(r"Train Acc \(final epoch\) ([\d.]+)±([\d.]+)", txt)
    if m:
        return float(m.group(1)), float(m.group(2))
    # 回退：最后一个 epoch 的 Train Acc（单 run 末行）
    accs = re.findall(r"Epoch\s+\d+/\d+ \| Train Loss [\d.]+ Acc ([\d.]+)", txt)
    return (100 * float(accs[-1]), None) if accs else None


def eval_phase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, test_loader = get_mnist_dataloaders(batch_size=512, data_dir="./data")
    # OOD：均匀噪声（与 MNIST 同分布形状）
    x_in, _ = next(iter(test_loader))
    noise = torch.rand(x_in.size(0), 1, 28, 28)

    class NoiseLoader:
        def __init__(self, x):
            self.x = x

        def __iter__(self):
            for i in range(0, self.x.size(0), 512):
                yield self.x[i:i + 512], torch.zeros(min(512, self.x.size(0) - i), dtype=torch.long)

    noise_loader = NoiseLoader(noise)

    rows = {}
    # 1) clean acc（5 seeds 平均，来自日志）+ FGIB-S 均值部署对照（seed 0 重算）
    print("== clean accuracy（5 seeds 平均；FGIB-S 另报 μ 部署对照 seed0）==")
    for mname in ["det"] + [f"ceb_b{b}" for b in BETAS] + [f"fgibs_b{b}" for b in BETAS]:
        acc = parse_clean_acc(mname)
        rows[mname] = {"clean": acc}
        print(f"  {mname:12s} clean={acc}")
    # 2) FGIB-S μ 部署对照
    for b in BETAS:
        try:
            m = load_model(f"fgibs_b{b}").to(device)
            acc_mu = eval_acc(m, test_loader, device, stochastic=False)
            rows[f"fgibs_b{b}"]["clean_mu"] = 100 * acc_mu
            print(f"  fgibs_b{b:8s} clean(μ 部署, seed0)={100 * acc_mu:.2f}")
        except FileNotFoundError as e:
            print("  ", e)
    # 3) PGD（seed 0；1000 样本）
    xs, ys = next(iter(test_loader))
    xs, ys = xs[:1000].to(device), ys[:1000].to(device)
    print("== PGD-L∞ ε=0.3 40 步（seed 0，前 1000 测试样本）==")
    for mname in ["det", "ceb_b0.001", "fgibs_b0.001", "ceb_b0.1", "fgibs_b0.1"]:
        try:
            m = load_model(mname).to(device)
            clean = (forward_logits(m, xs, isinstance(m, FGIBS)).argmax(1) == ys).float().mean().item()
            for tgt, tag in ((False, "untargeted"), (True, "targeted")):
                robust, success = pgd(m, xs, ys, device, targeted=tgt)
                rows[mname][f"pgd_{tag}"] = robust
                print(f"  {mname:12s} {tag:10s} clean={clean:.4f} robust={robust:.4f} "
                      f"succ={success:.4f}")
        except FileNotFoundError as e:
            print("  ", e)
    # 4) OOD
    print("== OOD（均匀噪声）AUROC ==")
    models = {}
    for mname in ["det", "ceb_b0.001", "fgibs_b0.001", "ceb_b0.1", "fgibs_b0.1"]:
        try:
            models[mname] = load_model(mname).to(device)
        except FileNotFoundError:
            pass
    ood = ood_scores(models, test_loader, noise_loader, device)
    for mname, res in ood.items():
        for sname, v in res.items():
            rows.setdefault(mname, {})[f"ood_{sname}"] = v
            print(f"  {mname:12s} {sname:10s} AUROC={v:.4f}")
    # 5) calibration
    print("== calibration（ECE/NLL/Brier，seed 0）==")
    for mname in ["det", "ceb_b0.001", "fgibs_b0.001", "ceb_b0.1", "fgibs_b0.1"]:
        if mname not in models:
            try:
                models[mname] = load_model(mname).to(device)
            except FileNotFoundError:
                continue
        ece, nll, brier = calibration(models[mname], test_loader, device)
        rows[mname]["ece"] = ece
        rows[mname]["nll"] = nll
        rows[mname]["brier"] = brier
        print(f"  {mname:12s} ECE={ece:.4f} NLL={nll:.4f} Brier={brier:.4f}")
    # 6) random-label 记忆
    print("== random-label 最终 Train Acc（100 epochs，5 seeds）==")
    rand = {}
    for m in ("ceb", "fgibs"):
        for b in ("0.001", "0.1"):
            r = parse_rand_train_acc(f"{m}_b{b}")
            rand[f"{m}_b{b}"] = r
            print(f"  rand_{m}_b{b:8s} final train acc={r}")

    write_table(rows, rand)
    print("已生成 tables/ceb_suite.tex")


def write_table(rows, rand):
    lines = [
        "\\begin{table}[t]",
        "\\caption{Matched CEB suite on MNIST/MLP (Section~\\ref{sec:suite}): clean accuracy "
        "(5-seed mean), PGD-$L_\\infty$ ($\\varepsilon{=}0.3$, 40 steps, 1000 test samples, "
        "seed 0; untargeted/targeted), OOD detection with uniform noise (AUROC; label-free "
        "scores: max-softmax, CEB's rate with its \\emph{predicted} label, and FGIB-S's "
        "anchor distance $\\min_y\\KLd(q(z|x)\\|\\Ncal(a v_y,\\tau^2I))$), calibration "
        "(ECE 15 bins, seed 0), and random-label memorization (final train accuracy, "
        "100 epochs, 5 seeds). FGIB-S is evaluated on its \\emph{deployed} sampled "
        "$z$; the $\\mu$-deployment control is reported in the text.}",
        "\\label{tab:suite}",
        "\\begin{center}\\footnotesize",
        "\\begin{tabular}{lcccccccc}",
        "Model & Clean & PGD-un & PGD-t & OOD-msp & OOD-rate & OOD-anchor & ECE & Rand-lab "
        "\\\\ \\hline \\\\[-1.8ex]",
    ]
    def f(v, nd=2):
        if v is None:
            return "--"
        if isinstance(v, tuple):
            return f"{v[0]:.1f}$\\pm${v[1]:.1f}" if v[1] is not None else f"{v[0]:.1f}"
        return f"{v:.{nd}f}"

    for name, label, rand_key in [
        ("det", "Det.", None),
        ("ceb_b0.001", "CEB $\\beta{=}10^{-3}$", "ceb_b0.001"),
        ("ceb_b0.1", "CEB $\\beta{=}10^{-1}$", "ceb_b0.1"),
        ("fgibs_b0.001", "FGIB-S $\\beta{=}10^{-3}$", "fgibs_b0.001"),
        ("fgibs_b0.1", "FGIB-S $\\beta{=}10^{-1}$", "fgibs_b0.1"),
    ]:
        r = rows.get(name, {})
        cells = [
            f(r.get("clean")),
            f(r.get("pgd_untargeted")),
            f(r.get("pgd_targeted")),
            f(r.get("ood_maxsoftmax")),
            f(r.get("ood_rate")),
            f(r.get("ood_anchor")),
            f(r.get("ece"), 3),
            f(rand.get(rand_key) if rand_key else None),
        ]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\end{tabular}\\end{center}\\end{table}")
    Path("tables/ceb_suite.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="训练阶段（train.py 子进程并发）")
    ap.add_argument("--eval", action="store_true", help="评估阶段（加载 checkpoint）")
    ap.add_argument("--parallel", type=int, default=8, help="训练并发数（默认 8）")
    args = ap.parse_args()
    if args.train:
        train_phase(args.parallel)
    elif args.eval:
        eval_phase()
    else:
        ap.print_help()
