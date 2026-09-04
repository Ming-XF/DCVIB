"""对抗鲁棒性评估脚本：对已训练的 MNIST 模型做 targeted PGD（L∞/L2）攻击评估
（CEB 论文 Fig. 4 底栏范式）。

不训练任何参数：加载各配置目录下的最优 checkpoint（`{stem}_run{i}.pt`），在
测试集（前 `--n-test` 张）上做白盒攻击。攻击目标为单一目标类（`--target-class`，
默认 0 = MNIST 最具区分度的数字，CEB 论文选 trouser 的同款理由），succ 为
targeted 成功率（干净预测正确且 ≠ 目标类的样本中、对抗预测 == 目标类的比例）。
评估走随机路径：预测与攻击每步梯度均为 `--mc-samples`（默认 10）次采样
（z~q(z|x) 重参数化）的平均——预测为 MC 平均概率、攻击梯度为 EOT 平均梯度
（白盒攻击者知道随机性的标准做法）；确定性模型（基线/FGIB）采样无效果、
自动退化为单次前向。ε 定义在归一化像素空间（MNIST mean 0.1307 / std 0.3081）。

用法（先用 `--save-path` 把每个配置训到独立目录，save-path 是 stem、自动生成
`{stem}_run{i}.pt`）：

    python adv_eval.py \\
        --model-dirs output/adv_mnist/mnist_mlp_ceb_beta_0.1 \\
                     output/adv_mnist/mnist_mlp_opb_beta_0.1_anchor_12 \\
        --target-class 0 --eps-linf 0.3 --eps-l2 6.0 --pgd-steps 20 \\
        --n-test 1000 --runs 5

目录最后一级名字必须能解析出架构：`{dataset}_{backbone}_{model}`（变体，允许
自定义后缀如 `_beta_0.01`）或 `{dataset}_{model}`（基线）。checkpoint 须用默认
架构参数训练（hidden-dims 512 256、z-dim 256、dropout 0.2）；非默认值需用对应的
全局参数（`--hidden-dims`/`--z-dim`/`--dropout`）覆盖。opb 例外：checkpoint 须以
`--energy-classifier` 训练，评估时自动按能量分类器重建（无需传该标志）；
anchor_scale 从目录名 `_anchor_{a}` 后缀解析（缺失时用 `--anchor-scale` 全局值）。
v1 仅支持 MNIST。

并行：`--parallel` 为每张 GPU 的并发进程数（默认 0 = 进程内串行）；>0 时按
tune.py 的每卡独立队列模式把每个工作项（配置）作为子进程（`CUDA_VISIBLE_DEVICES`
指定 GPU）执行，各子进程写 parts_targeted/ 分片 csv，父进程合并后统一生成
汇总与 HTML。

输出：`{results-dir}/{dataset}_adv_targeted.csv`（长表：config/run/norm/eps/acc/
succ，clean 行 succ 为空）与 `{results-dir}/{dataset}_adv_targeted.html`
（汇总表，每配置一行、跨 run 均值±标准差、按 clean acc 排序）。论文图由
adv_targeted_plot.py 从 csv 生成。
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.datasets import get_mnist_dataloaders
from train import build_model, build_parser, run_model
from tune import detect_gpus, gen_html

ROOT = Path(__file__).resolve().parent

DATASETS = {"mnist", "imagenet100", "cora", "imdb", "agnews", "california", "stsb", "zinc", "agedb"}
BACKBONES = {"mlp", "cnn", "gnn", "rnn"}
VARIANTS = {"vib", "ceb", "fgib", "opb", "svib", "nib", "dvcca"}
BASELINE_MODELS = {"mlp", "cnn", "gcn", "rnn"}
_STOCHASTIC_CLS = {"VIB", "SVIB", "CEB", "NIB", "OPB", "DVCCA"}

# 子进程命令行重建时跳过的参数（父进程按工作项另行构造）
CHILD_SKIP = {"model_dirs", "part_csv", "parallel", "results_dir", "help"}


def parse_dir_name(name: str):
    """从目录最后一级名字解析 (dataset, backbone, model)。"""
    parts = name.split("_")
    if not parts or parts[0] not in DATASETS:
        raise ValueError(f"无法解析目录名 '{name}'：首段必须是已知数据集名")
    dataset = parts[0]
    if len(parts) == 2 and parts[1] in BASELINE_MODELS:
        return dataset, parts[1], parts[1]
    if len(parts) >= 3 and parts[1] in BACKBONES and parts[2] in VARIANTS:
        return dataset, parts[1], parts[2]
    raise ValueError(
        f"无法解析目录名 '{name}'：应为 {{dataset}}_{{model}}（基线）或 "
        "{{dataset}}_{{backbone}}_{{model}}（变体，允许自定义后缀）"
    )


def parse_anchor(name: str):
    """从目录名解析 `_anchor_{a}` 后缀的锚点尺度；无则返回 None。"""
    m = re.search(r"_anchor_([\d.eE+-]+)", name)
    return float(m.group(1)) if m else None


def _run_index(p: Path) -> int:
    m = re.search(r"_run(\d+)\.pt$", p.name)
    return int(m.group(1)) if m else -1


def _is_stochastic(model):
    """模型是否有随机瓶颈（z~q(z|x) 采样路径）；确定性模型（基线/FGIB）采样无效果。"""
    return model.__class__.__name__ in _STOCHASTIC_CLS


def _mc_logits(model, x, k):
    """k 次采样的平均 softmax 概率：每次前向从 q(z|x) 重参数化采样 z
    （stochastic=True）；确定性模型单次前向（采样无效果）。返回 (B, C) 概率。"""
    if not _is_stochastic(model):
        k = 1
    probs = 0.0
    for _ in range(k):
        logits = run_model(model, x, None, stochastic=True)[0]
        probs = probs + F.softmax(logits, 1)
    return probs / k


def pgd_attack(model, images, labels, eps, norm, steps, target_cls, mc_samples=10):
    """targeted PGD（随机起点，EOT 梯度）：最大化目标类 logits，返回对抗样本。

    随机模型每步梯度为 mc_samples 次采样（z~q(z|x)）的平均梯度（EOT：白盒
    攻击者知道随机性），确定性模型自动退化为单前向；labels=None 使 run_model
    跳过 KL/QR 计算。L∞：步长 2.5ε/steps·sign(grad)，clamp 到 [−ε,ε]；
    L2：梯度按样本归一化，步长 2.5ε/steps，超球时缩回。
    """
    labels = torch.full_like(labels, target_cls)
    k = mc_samples if _is_stochastic(model) else 1
    alpha = 2.5 * eps / steps
    if norm == "linf":
        delta = torch.empty_like(images).uniform_(-eps, eps)
    else:  # l2：随机方向 × eps
        d = torch.randn_like(images)
        d = d / d.view(len(images), -1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        delta = d * eps
    delta = delta.detach().requires_grad_(True)

    for _ in range(steps):
        grad = None
        for _ in range(k):
            logits = run_model(model, images + delta, None, stochastic=True)[0]
            loss = -F.cross_entropy(logits, labels)  # 沿上升方向移动即朝目标类靠近
            g = torch.autograd.grad(loss, delta)[0]
            grad = g if grad is None else grad + g
        grad = grad / k
        if norm == "linf":
            delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
        else:
            d = delta + alpha * grad / grad.view(len(images), -1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
            n = d.view(len(images), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
            delta = d * (eps / n.clamp_min(eps))
        delta = delta.detach().requires_grad_(True)

    return (images + delta).detach()


def fmt_mean_std(values):
    """[float] → 'mean±std' 字符串（与仓库日志汇总约定一致；单 run 时只给均值）。"""
    t = torch.tensor(values, dtype=torch.float64)
    if t.numel() == 1:
        return f"{t.item():.4f}"
    return f"{t.mean():.4f}±{t.std():.4f}"


def attack_combos(args):
    """攻击网格 [(norm, eps), ...]：clean（none, 0）打头。"""
    return (
        [("none", 0.0)]
        + [("linf", e) for e in args.eps_linf if e != 0.0]
        + [("l2", e) for e in args.eps_l2 if e != 0.0]
    )


def acc_succ_keys(eps_linf, eps_l2):
    """指标键（含成功率）：Acc + Linf*/L2* + SuccLinf*/SuccL2*，顺序即表格列序。"""
    linf = [e for e in eps_linf if e != 0.0]
    l2 = [e for e in eps_l2 if e != 0.0]
    acc = ["Acc"] + [f"Linf{e:g}" for e in linf] + [f"L2{e:g}" for e in l2]
    succ = [f"SuccLinf{e:g}" for e in linf] + [f"SuccL2{e:g}" for e in l2]
    return acc, succ


def _build_net(parser, args, d, device):
    """按目录名重建并返回模型（opb 固定能量分类器、anchor 从目录名解析）。"""
    label = d.name
    dataset, backbone, model = parse_dir_name(label)
    if dataset != "mnist":
        raise ValueError(f"目录 '{label}' 不是 MNIST 配置（adv_eval v1 仅支持 mnist）")
    ckpts = sorted(d.glob("*_run*.pt"), key=_run_index)
    if len(ckpts) < args.runs:
        raise ValueError(f"目录 '{d}' 只找到 {len(ckpts)} 个 '*_run*.pt'，需要 --runs={args.runs} 个")
    ckpts = ckpts[: args.runs]
    dir_args = argparse.Namespace(**vars(args))
    dir_args.model = model
    dir_args.backbone = backbone
    if model == "opb":
        # OPB 固定按能量分类器重建（paper/design/OPB.txt §12）：energy 训练下
        # self.classifier 是死参数（随机初始化），按默认普通分类器路径会得到
        # 随机输出（clean acc ≈ 1/K）。anchor_scale 从目录名解析（a=1/6/12
        # 网格混跑时全局值不可用），缺失时回退 --anchor-scale。
        dir_args.energy_classifier = True
        a = parse_anchor(label)
        if a is not None:
            dir_args.anchor_scale = a
    model_net = build_model(parser, dir_args).to(device)
    return model_net, ckpts


def _eval_preds(model_net, x, labels, mc_samples):
    """MC 采样平均概率的预测：返回 (preds, acc)。"""
    with torch.no_grad():
        probs = _mc_logits(model_net, x, mc_samples)
    preds = probs.argmax(1)
    acc = (preds == labels).float().mean().item()
    return preds, acc


def evaluate_config(parser, args, d, images, labels, device):
    """单配置主协议评估：返回 (label, csv 行列表)。

    行格式 [config, run, norm, eps, acc, succ]；clean 行 succ 为空串。
    succ 为 targeted 成功率：干净正确且 ≠ 目标类的样本翻转到目标类的比例。
    """
    label = d.name
    model_net, ckpts = _build_net(parser, args, d, device)
    print(f"[加载] {label} → checkpoint {len(ckpts)} 个")
    rows = []
    for run_i, ckpt in enumerate(ckpts, 1):
        model_net.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        model_net.eval()
        with torch.no_grad():
            clean_probs = _mc_logits(model_net, images, args.mc_samples)
        clean_preds = clean_probs.argmax(1)
        clean_correct = clean_preds == labels
        clean_acc = clean_correct.float().mean().item()
        rows.append([label, run_i, "none", "0", f"{clean_acc:.6f}", ""])
        for norm, eps in attack_combos(args)[1:]:
            x_in = pgd_attack(model_net, images, labels, eps, norm, args.pgd_steps,
                              args.target_class, mc_samples=args.mc_samples)
            preds, acc = _eval_preds(model_net, x_in, labels, args.mc_samples)
            eligible = clean_correct & (clean_preds != args.target_class)
            n_e = eligible.sum().item()
            succ = (preds[eligible] == args.target_class).float().mean().item() if n_e else float("nan")
            rows.append([label, run_i, norm, f"{eps:g}", f"{acc:.6f}", f"{succ:.6f}"])
            print(f"[{label}] run{run_i} {norm} ε={eps:g} acc {acc:.4f} succ {succ:.4f}")
    return label, rows


def summarize(rows_by_label, eps_linf, eps_l2, verbose=True):
    """长表行 → results（gen_html 汇总表行）。

    rows_by_label: {label: [csv 行]}（行序 = 最终 csv 序）；每 label 的指标为
    跨 run 均值±std 字符串（fmt_mean_std）。
    """
    acc_keys, succ_keys = acc_succ_keys(eps_linf, eps_l2)
    results = []
    for label, rows in rows_by_label.items():
        per = {k: [] for k in acc_keys + succ_keys}
        for r in rows:
            norm, eps, acc, succ = r[2], r[3], float(r[4]), r[5]
            if norm == "none":
                per["Acc"].append(acc)
            else:
                key = f"{'Linf' if norm == 'linf' else 'L2'}{float(eps):g}"
                per[key].append(acc)
                per[f"Succ{key}"].append(float(succ) if succ != "" else float("nan"))
        metrics = {k: fmt_mean_std(per[k]) for k in acc_keys + succ_keys if per[k]}
        results.append((label, None, None, metrics))
        if verbose:
            print(f"[汇总] {label}: " + " ".join(f"{k}={v}" for k, v in metrics.items()))
    return results


def load_test_images(args, device):
    """测试集前 n_test 张（原图与标签一次取齐，各工作项共享同一确定性切片）。"""
    _, _, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
        if sum(len(t) for t in xs) >= args.n_test:
            break
    return (
        torch.cat(xs)[: args.n_test].to(device),
        torch.cat(ys)[: args.n_test].to(device),
    )


def build_adv_parser():
    parser = build_parser()
    parser.add_argument(
        "--model-dirs", nargs="*", default=None,
        help="已训练配置的 checkpoint 目录列表（每个目录内含 {stem}_run{i}.pt；"
        "目录名须可解析出架构，见脚本 docstring）",
    )
    parser.add_argument(
        "--eps-linf", type=float, nargs="*", default=[0.3],
        help="L∞ 攻击 ε 网格（归一化像素空间，默认 [0.3]；传空值跳过该范数）",
    )
    parser.add_argument(
        "--eps-l2", type=float, nargs="*", default=[6.0],
        help="L2 攻击 ε 网格（归一化像素空间，默认 [6.0]——2.0 时攻击预算"
        "够不着目标类区域、成功率贴基率地板；传空值跳过该范数）",
    )
    parser.add_argument("--pgd-steps", type=int, default=20, help="PGD 步数（默认 20）")
    parser.add_argument("--n-test", type=int, default=1000, help="测试集前 N 张（默认 1000）")
    parser.add_argument(
        "--mc-samples", type=int, default=10,
        help="随机模型的 MC 评估采样数（默认 10）：预测为 k 次采样平均概率、攻击"
        "每步梯度为 k 次采样平均（EOT）；确定性模型自动退化为单次前向",
    )
    parser.add_argument(
        "--target-class", type=int, default=0,
        help="targeted 攻击的目标类（默认 0：MNIST 最具区分度的数字）",
    )
    parser.add_argument("--results-dir", type=str, default="adv_results", help="结果输出目录（默认 adv_results）")
    parser.add_argument(
        "--parallel", type=int, default=0,
        help="每张 GPU 的并发子进程数（默认 0 = 进程内串行；>0 按每卡独立队列并行，"
        "总并行 = GPU 数 × 此值）",
    )
    parser.add_argument(
        "--part-csv", type=str, default=None,
        help="[内部] 子进程模式：只跑一个工作项（--model-dirs 单目录），"
        "把长表行写入该 csv 后退出，不生成汇总/HTML",
    )
    return parser


def _child_cmd(parser, args, extra):
    """重建子进程命令行：转发父进程所有非默认参数 + 工作项专属 extra。"""
    cmd = [sys.executable, str(ROOT / "adv_eval.py")]
    for action in parser._actions:
        if action.dest in CHILD_SKIP:
            continue
        value = getattr(args, action.dest, None)
        if value is None:
            continue
        if action.default is not None and value == action.default:
            continue
        opt = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                cmd.append(opt)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                cmd.append(opt)
        elif isinstance(value, (list, tuple)):
            cmd.extend([opt, *(str(v) for v in value)])
        else:
            cmd.extend([opt, str(value)])
    cmd.extend(extra)
    return cmd


def _run_child(cmd, env, part_path):
    """执行一个子进程工作项，返回 (part_path, ok, stderr)。"""
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(part_path).exists():
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        return part_path, False, tail
    return part_path, True, ""


def _write_part(part_path, rows):
    Path(part_path).parent.mkdir(parents=True, exist_ok=True)
    with open(part_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "norm", "eps", "acc", "succ"])
        w.writerows(rows)


def _read_part(part_path):
    with open(part_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.reader(f)][1:]  # 去掉表头


def main():
    parser = build_adv_parser()
    args = parser.parse_args()

    if args.task != "mnist":
        parser.error("adv_eval v1 仅支持 --task mnist")

    results_root = ROOT / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    # ---------------- 子进程模式：单个配置 → part csv ----------------
    if args.part_csv:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images, labels = load_test_images(args, device)
        if not args.model_dirs or len(args.model_dirs) != 1:
            parser.error("--part-csv 模式需要 --model-dirs 恰好一个目录")
        label, rows = evaluate_config(parser, args, Path(args.model_dirs[0]), images, labels, device)
        _write_part(args.part_csv, rows)
        print(f"[分片完成] {label} → {args.part_csv}")
        return

    # ---------------- 父进程：构造工作项 ----------------
    if not args.model_dirs:
        parser.error("没有评估目标：--model-dirs 至少给一个目录")

    items = []
    for p in args.model_dirs:
        d = Path(p)
        parse_dir_name(d.name)  # 提前校验，失败即报错
        items.append((d.name, ["--model-dirs", str(d)]))

    parts_dir = results_root / "parts_targeted"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = []  # (label, part_path, full_extra)
    for i, (label, base) in enumerate(items):
        safe = re.sub(r"\W+", "_", label).strip("_")
        p = parts_dir / f"{i:04d}_{safe}.csv"
        part_paths.append((label, p, base + ["--part-csv", str(p)]))

    # ---------------- 执行 ----------------
    if args.parallel > 0:
        n_gpus = detect_gpus()
        slots = list(range(n_gpus)) if n_gpus > 0 else [None]
        total = n_gpus * args.parallel if n_gpus > 0 else args.parallel
        print(f"并行模式：GPU {n_gpus} 张 × --parallel {args.parallel} = {total} 并发，共 {len(items)} 个工作项")
        with ExitStack() as stack:
            pools = [stack.enter_context(ThreadPoolExecutor(max_workers=args.parallel)) for _ in slots]
            futures = {}
            for i, (label, p, extra) in enumerate(part_paths):
                cmd = _child_cmd(parser, args, extra)
                env = dict(os.environ)
                gpu = slots[i % len(slots)]
                if gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                futures[pools[i % len(pools)].submit(_run_child, cmd, env, p)] = (label, p)
            n_ok = 0
            for fut in as_completed(futures):
                label, p = futures[fut]
                _, ok, err = fut.result()
                n_ok += ok
                if not ok:
                    print(f"[失败] {label}\n{err}")
            print(f"并行执行完成：{n_ok}/{len(items)} 个工作项成功")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images, labels = load_test_images(args, device)
        for label, p, extra in part_paths:
            try:
                d = Path(extra[1])
                _, rows = evaluate_config(parser, args, d, images, labels, device)
                _write_part(p, rows)
            except Exception as e:  # 单项失败不中断其余项
                print(f"[失败] {label}: {e}")

    # ---------------- 合并分片 → 长表 + 汇总 ----------------
    rows_by_label = {}
    missing = []
    for label, p, _extra in part_paths:
        if not Path(p).exists():
            missing.append(label)
            continue
        rows_by_label[label] = _read_part(p)
    if missing:
        print(f"缺失分片（跳过汇总）：{missing}")

    all_rows = [r for _, rows in rows_by_label.items() for r in rows]
    csv_path = results_root / f"{args.task}_adv_targeted.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "norm", "eps", "acc", "succ"])
        w.writerows(all_rows)

    results = summarize(rows_by_label, args.eps_linf, args.eps_l2)
    meta = (
        f"task={args.task} pgd_steps={args.pgd_steps} n_test={args.n_test} "
        f"mc={args.mc_samples} targeted={args.target_class} | {len(rows_by_label)} 个配置"
        "（每行跨 run 的均值±标准差；Acc 为 clean acc，Linf/L2 为对应 ε 的鲁棒精度，"
        "Succ* 为 targeted 成功率（干净正确且非目标类的样本翻转到目标类的比例））"
    )
    html_path = results_root / f"{args.task}_adv_targeted.html"
    gen_html(results, html_path, meta)
    print(f"\n结果已保存：{csv_path}\n汇总表格已生成：{html_path}")


if __name__ == "__main__":
    main()
