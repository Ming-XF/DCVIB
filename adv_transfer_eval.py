"""迁移对抗鲁棒性评估（CEB 论文 Fig. 6 迁移攻击范式）：用源模型生成 targeted
PGD 扰动、在目标模型上评估，检验白盒鲁棒性是否为梯度掩蔽伪影。

与主 targeted 试验同条件：目标类 0、20 步、步长 2.5ε/steps、随机起点、源模型
EOT-10 攻击梯度、目标模型 MC-10 预测；succ 按目标模型口径（其干净正确且
≠ 0 的样本中翻转到 0 的比例）。

组合（5 个，每组合 5 run）：
- 白盒自攻：CEB β=0.1（有效区间末端）、OPB β=25, a=1（最大 β、最压缩最鲁棒，
  最严格的梯度掩蔽检验对象）、确定性 MLP 基线（对照）；
- 迁移：CEB→OPB、OPB→CEB（源生成扰动、目标评估）。

ε 网格：L∞ [0.05..0.3]、L2 [1..6]（覆盖地板到饱和区间）。输出长表
output/adv_mnist/mnist_adv_transfer.csv（config/run/norm/eps/acc/succ，
config 为单配置目录名或 src→dst），论文图由 adv_transfer_plot.py 生成。

用法：
    python adv_transfer_eval.py
"""

import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adv_eval import (  # noqa: E402
    _build_net, _eval_preds, _mc_logits, build_adv_parser,
    load_test_images, pgd_attack,
)

CEB_DIR = ROOT / "output" / "adv_mnist" / "mnist_mlp_ceb_beta_0.1"
OPB_DIR = ROOT / "output" / "adv_mnist" / "mnist_mlp_opb_beta_25_anchor_1"
MLP_DIR = ROOT / "output" / "adv_mnist" / "mnist_mlp"
CSV_PATH = ROOT / "output" / "adv_mnist" / "mnist_adv_transfer.csv"

EPS_LINF = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
EPS_L2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def run_combo(parser, args, src_dir, dst_dir, label, images, labels, device):
    """源模型生成扰动、目标模型评估：返回 csv 行列表（clean 行为目标口径）。"""
    src_net, src_ckpts = _build_net(parser, args, src_dir, device)
    if src_dir == dst_dir:
        dst_net, dst_ckpts = src_net, src_ckpts
    else:
        dst_net, dst_ckpts = _build_net(parser, args, dst_dir, device)
    if len(src_ckpts) != len(dst_ckpts):
        raise ValueError(f"{label} 两模型 run 数不一致：{len(src_ckpts)} vs {len(dst_ckpts)}")
    print(f"[加载] {label}（{len(dst_ckpts)} runs）")
    rows = []
    for run_i in range(len(dst_ckpts)):
        src_net.load_state_dict(torch.load(src_ckpts[run_i], weights_only=True, map_location=device))
        src_net.eval()
        if src_dir != dst_dir:
            dst_net.load_state_dict(torch.load(dst_ckpts[run_i], weights_only=True, map_location=device))
            dst_net.eval()
        with torch.no_grad():
            clean_probs = _mc_logits(dst_net, images, args.mc_samples)
        clean_preds = clean_probs.argmax(1)
        clean_correct = clean_preds == labels
        clean_acc = clean_correct.float().mean().item()
        rows.append([label, run_i + 1, "none", "0", f"{clean_acc:.6f}", ""])
        for norm, eps_grid in (("linf", args.eps_linf), ("l2", args.eps_l2)):
            for eps in eps_grid:
                if eps == 0.0:
                    continue
                x_in = pgd_attack(src_net, images, labels, eps, norm, args.pgd_steps,
                                  args.target_class, mc_samples=args.mc_samples)
                preds, acc = _eval_preds(dst_net, x_in, labels, args.mc_samples)
                eligible = clean_correct & (clean_preds != args.target_class)
                n_e = eligible.sum().item()
                succ = (preds[eligible] == args.target_class).float().mean().item() if n_e else float("nan")
                rows.append([label, run_i + 1, norm, f"{eps:g}", f"{acc:.6f}", f"{succ:.6f}"])
                print(f"[{label}] run{run_i + 1} {norm} ε={eps:g} acc {acc:.4f} succ {succ:.4f}")
    return rows


def main():
    parser = build_adv_parser()
    args = parser.parse_args([])
    args.eps_linf = EPS_LINF
    args.eps_l2 = EPS_L2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = load_test_images(args, device)

    combos = [
        (CEB_DIR, CEB_DIR, CEB_DIR.name),
        (OPB_DIR, OPB_DIR, OPB_DIR.name),
        (MLP_DIR, MLP_DIR, MLP_DIR.name),
        (CEB_DIR, OPB_DIR, f"{CEB_DIR.name}→{OPB_DIR.name}"),
        (OPB_DIR, CEB_DIR, f"{OPB_DIR.name}→{CEB_DIR.name}"),
    ]
    all_rows = []
    for src_dir, dst_dir, label in combos:
        all_rows.extend(run_combo(parser, args, src_dir, dst_dir, label,
                                  images, labels, device))

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "norm", "eps", "acc", "succ"])
        w.writerows(all_rows)
    print(f"\n结果已保存：{CSV_PATH}（{len(all_rows)} 行）")


if __name__ == "__main__":
    main()
