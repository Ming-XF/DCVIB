"""MNIST 分类 / ImageNet-100 特征分类 / Cora 节点分类 / IMDb 情感分析 / AG News BERT 特征分类 / California Housing 回归训练脚本：MLP / CNN / GCN / RNN 基线及 VIB / CEB / FGIB 等变体。

用法：
    python train.py                 # 使用默认参数训练 MLP
    python train.py --model cnn     # 训练 CNN 基线
    python train.py --model vib     # 训练 MLP 版 VIB
    python train.py --backbone cnn --model vib    # 训练 CNN 版 VIB
    python train.py --backbone cnn --model ceb      # 训练 CNN 版 CEB
    python train.py --task imagenet100 --model vib     # ImageNet-100 特征分类（MLP 骨干）
    python train.py --task imagenet100 --model ceb     # CEB 特征分类
    python train.py --task cora --model gcn            # Cora 节点分类（GCN 基线）
    python train.py --task cora --backbone gnn --model vib    # GNN 版 VIB
    python train.py --task cora --backbone gnn --model ceb    # GNN 版 CEB
    python train.py --task zinc --model gcn            # ZINC-12k 分子图回归（GCN 基线）
    python train.py --task zinc --backbone gnn --model vib    # GNN 版 VIB 图级回归（ceb/fgib/svib/nib/dvcca 同理）
    python train.py --task zinc --backbone gnn --model ceb    # GNN 版 CEB 回归（连续 y 条件先验）
    python train.py --task zinc --backbone gnn --model fgib   # GNN 版 FGIB 回归（固定 RFF 连续锚点先验）
    python train.py --task imdb --model rnn            # IMDb 情感分析（RNN 基线）
    python train.py --task imdb --backbone rnn --model vib    # RNN 版 VIB
    python train.py --task imdb --backbone rnn --model ceb    # RNN 版 CEB
    python train.py --task agnews --model rnn          # AG News BERT 特征分类（RNN 基线）
    python train.py --task agnews --backbone rnn --model vib  # RNN 版 VIB（ceb/fgib 同理）
    python train.py --task stsb --model rnn             # STS-B 文本相似度回归（RNN 基线）
    python train.py --task stsb --backbone rnn --model fgib   # RNN 版 FGIB 回归（固定 RFF 连续锚点先验）
    python train.py --task housing --model vib     # California Housing 回归（MLP 骨干）
    python train.py --task housing --model ceb     # CEB 回归（连续 y 条件先验）
    python train.py --task housing --model fgib    # FGIB 回归（固定 RFF 连续锚点先验）
    python train.py --task agedb --model cnn                 # AgeDB 年龄回归（CNN 基线，RGB 64×64）
    python train.py --task agedb --model mlp                 # AgeDB 年龄回归（MLP 基线）
    python train.py --task agedb --backbone cnn --model vib  # CNN 版 VIB 回归（ceb/fgib/svib/nib/dvcca 同理）
    python train.py --task agedb --backbone cnn --model fgib # CNN 版 FGIB 回归（固定 RFF 连续锚点先验）
    python train.py --model fgib    # FGIB（固定几何信息瓶颈：确定性主路 + 固定正交锚点先验，MLP 骨干）
    python train.py --model fgib --anchor-scale 8   # 自定义锚点尺度（0 为各类相同锚点）
    python train.py --backbone cnn --model fgib     # CNN 版 FGIB
    python train.py --task imagenet100 --model fgib # ImageNet-100 特征分类（MLP 骨干）
    python train.py --task cora --backbone gnn --model fgib    # GNN 版 FGIB
    python train.py --task imdb --backbone rnn --model fgib    # RNN 版 FGIB
    python train.py --task agnews --backbone rnn --model fgib  # AG News 版 FGIB
    python train.py --model opb     # OPB（正交先验信息瓶颈：先验编码器全类别均值每前向 QR 正交化，仅分类任务）
    python train.py --model opb --anchor-scale 8   # 自定义锚点尺度
    python train.py --backbone cnn --model opb     # CNN 版 OPB（gnn/rnn 同理）
    python train.py --model ceb     # CEB（标签条件先验 r(z|y)，对角高斯实现）
    python train.py --model svib    # SVIB（平方信息瓶颈：基于 VIB 的模型变体，CE + β·KL²）
    python train.py --backbone cnn --model svib   # CNN 版 SVIB（其他骨干同理）
    python train.py --model nib     # NIB（非线性信息瓶颈：噪声注入 + 无先验成对距离上界，CE + β·Î）
    python train.py --model dvcca   # DVCCA（监督 β-DVCCA：VIB + 重建解码器，CE + β·KL + β·MSE_recon；仅 MLP/CNN/GNN 骨干）
    python train.py --random-labels --patience 1000   # 随机标签记忆实验（信息自由数据集）
    python train.py --epochs 20 --batch-size 128 --lr 1e-3
    python train.py --runs 5 --seed 0   # 重复训练 5 次并报告测试集平均指标
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from datasets.agedb import get_agedb_dataloaders
from datasets.agnews import get_agnews_dataloaders
from datasets.cora import get_cora_data
from datasets.datasets import (
    get_california_dataloaders,
    get_imagenet100_dataloaders,
    get_mnist_dataloaders,
)
from datasets.imdb import get_imdb_dataloaders
from datasets.stsb import get_stsb_dataloaders
from datasets.zinc import get_zinc_dataloaders
from model import CEB, CNN, DVCCA, FGIB, GCN, MLP, NIB, OPB, SVIB, VIB
from model.cnn import CEB as CNNCEB, DVCCA as CNNDVCCA, FGIB as CNNFGIB, NIB as CNNNIB, OPB as CNNOPB, SVIB as CNNSVIB, VIB as CNNVIB
from model.gnn import CEB as GNNCEB, DVCCA as GNNDVCCA, FGIB as GNNFGIB, NIB as GNNNIB, OPB as GNNOPB, SVIB as GNNSVIB, VIB as GNNVIB
from model.rnn import (
    CEB as RNNCEB,
    DVCCA as RNNDVCCA,
    FGIB as RNNFGIB,
    NIB as RNNNIB,
    OPB as RNNOPB,
    RNN,
    SVIB as RNNSVIB,
    VIB as RNNVIB,
)


def run_model(model, images, labels, stochastic, adj=None, mask=None, batch=None):
    """统一 MLP / CNN / GNN / RNN 骨干下各模型的前向接口，返回 (logits, kl, recon)。

    recon 为 DVCCA 的输入重建损失（损失中同样乘以 beta），其余模型为 None。
    batch 为图批的节点→图索引（仅 ZINC 图级任务传入，GNN 分支按图读出）。
    """
    if isinstance(model, (VIB, CEB, NIB, OPB, CNNVIB, CNNCEB, CNNNIB, CNNOPB, RNNVIB, RNNCEB, RNNNIB, RNNOPB)):
        logits, kl = model(images, labels, stochastic=stochastic)
        return logits, kl, None
    if isinstance(model, (GNNVIB, GNNCEB, GNNNIB, GNNOPB)):
        logits, kl = model(images, labels, stochastic=stochastic, adj_norm=adj, mask=mask, batch=batch)
        return logits, kl, None
    if isinstance(model, (DVCCA, CNNDVCCA, RNNDVCCA)):
        return model(images, labels, stochastic=stochastic)
    if isinstance(model, GNNDVCCA):
        return model(images, labels, stochastic=stochastic, adj_norm=adj, mask=mask, batch=batch)
    if isinstance(model, (FGIB, CNNFGIB, RNNFGIB)):
        logits, kl = model(images, labels)
        return logits, kl, None
    if isinstance(model, GNNFGIB):
        logits, kl = model(images, labels, adj_norm=adj, mask=mask, batch=batch)
        return logits, kl, None
    if isinstance(model, GCN):
        return model(images, adj_norm=adj, batch=batch), None, None
    return model(images), None, None


def train_one_epoch(model, loader, optimizer, criterion, beta, device, task="classification"):
    """训练一个 epoch，返回 (平均损失, 训练准确率)；回归任务准确率为 None。"""
    model.train()
    total_loss, total, correct = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, kl, recon = run_model(model, images, labels, stochastic=True)
        if task in ("housing", "stsb", "agedb"):
            logits = logits.squeeze(-1)
        loss = criterion(logits, labels)
        if kl is not None:
            loss = loss + beta * kl
        if recon is not None:
            loss = loss + beta * recon
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        if task not in ("housing", "stsb", "agedb"):
            correct += (logits.argmax(dim=1) == labels).sum().item()

    acc = None if task in ("housing", "stsb", "agedb") else correct / total
    return total_loss / total, acc


@torch.no_grad()
def evaluate(model, loader, criterion, beta, device, task="classification", target_scaler=None):
    """分类返回 (loss, acc, 宏平均 AUC / precision / recall)；回归返回 (loss, MAE, R2)。

    回归的 MAE 用 target_scaler 逆归一化回原始单位计算（R2 与量纲无关）。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, kl, recon = run_model(model, images, labels, stochastic=False)
        if task in ("housing", "stsb", "agedb"):
            logits = logits.squeeze(-1)
        loss = criterion(logits, labels)
        if kl is not None:
            loss = loss + beta * kl
        if recon is not None:
            loss = loss + beta * recon

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        if task in ("housing", "stsb", "agedb"):
            preds = logits
        else:
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels)

    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    if task in ("housing", "stsb", "agedb"):
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        labels = target_scaler.inverse_transform(labels.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        return total_loss / total, mae, r2

    probs = torch.cat(all_probs).cpu().numpy()
    if probs.shape[1] == 2:
        probs = probs[:, 1]  # 二分类时 sklearn 要求 (n,) 的阳性类概率
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return total_loss / total, correct / total, auc, precision, recall


def train_one_epoch_cora(model, x, adj, y, train_mask, optimizer, criterion, beta, device):
    """Cora 全图批训练：CE 与 KL 都只在 train mask 节点上计算（转导式）。

    返回 (loss, 训练节点准确率)。
    """
    model.train()
    optimizer.zero_grad()
    logits, kl, recon = run_model(model, x, y, stochastic=True, adj=adj, mask=train_mask)
    loss = criterion(logits[train_mask], y[train_mask])
    if kl is not None:
        loss = loss + beta * kl
    if recon is not None:
        loss = loss + beta * recon
    loss.backward()
    optimizer.step()
    correct = (logits.argmax(dim=1)[train_mask] == y[train_mask]).sum().item()
    return loss.item(), correct / int(train_mask.sum())


@torch.no_grad()
def evaluate_cora(model, x, adj, y, mask, criterion, beta, device):
    """Cora 评估：分类指标只在 mask 节点上计算，返回 (loss, acc, 宏平均 AUC, pre, rec)。"""
    model.eval()
    logits, kl, recon = run_model(model, x, y, stochastic=False, adj=adj, mask=mask)
    loss = criterion(logits[mask], y[mask])
    if kl is not None:
        loss = loss + beta * kl
    if recon is not None:
        loss = loss + beta * recon

    probs = logits.softmax(dim=1)[mask].cpu().numpy()
    preds = logits.argmax(dim=1)[mask].cpu().numpy()
    labels = y[mask].cpu().numpy()
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    correct = (preds == labels).sum()

    return loss.item(), correct / len(labels), auc, precision, recall


def train_one_epoch_zinc(model, loader, optimizer, criterion, beta, device):
    """ZINC 图回归训练一个 epoch（图批四元组：x / 块对角 adj / batch_idx / y）。

    返回 (平均损失, None)，损失按图数加权（MSE 与 KL 均为批内平均）。
    """
    model.train()
    total_loss, total = 0.0, 0

    for x, adj_block, batch_idx, y in loader:
        x, adj_block, batch_idx, y = (
            x.to(device), adj_block.to(device), batch_idx.to(device), y.to(device)
        )
        optimizer.zero_grad()
        logits, kl, recon = run_model(
            model, x, y, stochastic=True, adj=adj_block, batch=batch_idx
        )
        logits = logits.squeeze(-1)
        loss = criterion(logits, y)
        if kl is not None:
            loss = loss + beta * kl
        if recon is not None:
            loss = loss + beta * recon
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

    return total_loss / total, None


@torch.no_grad()
def evaluate_zinc(model, loader, criterion, beta, device, target_scaler):
    """ZINC 图回归评估，返回 (loss, MAE, R2)；MAE 经 y_scaler 逆归一化回原始单位。"""
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for x, adj_block, batch_idx, y in loader:
        x, adj_block, batch_idx, y = (
            x.to(device), adj_block.to(device), batch_idx.to(device), y.to(device)
        )
        logits, kl, recon = run_model(
            model, x, y, stochastic=False, adj=adj_block, batch=batch_idx
        )
        logits = logits.squeeze(-1)
        loss = criterion(logits, y)
        if kl is not None:
            loss = loss + beta * kl
        if recon is not None:
            loss = loss + beta * recon

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        all_preds.append(logits)
        all_labels.append(y)

    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    labels = target_scaler.inverse_transform(labels.reshape(-1, 1)).ravel()
    return total_loss / total, mean_absolute_error(labels, preds), r2_score(labels, preds)


MODEL_CLASSES = {
    ("mlp", "mlp"): MLP,
    ("cnn", "cnn"): CNN,
    ("gcn", "gnn"): GCN,
    ("rnn", "rnn"): RNN,
    ("vib", "mlp"): VIB,
    ("vib", "cnn"): CNNVIB,
    ("vib", "gnn"): GNNVIB,
    ("vib", "rnn"): RNNVIB,
    ("svib", "mlp"): SVIB,
    ("svib", "cnn"): CNNSVIB,
    ("svib", "gnn"): GNNSVIB,
    ("svib", "rnn"): RNNSVIB,
    ("nib", "mlp"): NIB,
    ("nib", "cnn"): CNNNIB,
    ("nib", "gnn"): GNNNIB,
    ("nib", "rnn"): RNNNIB,
    ("dvcca", "mlp"): DVCCA,
    ("dvcca", "cnn"): CNNDVCCA,
    ("dvcca", "gnn"): GNNDVCCA,
    ("dvcca", "rnn"): RNNDVCCA,
    ("ceb", "mlp"): CEB,
    ("ceb", "cnn"): CNNCEB,
    ("ceb", "gnn"): GNNCEB,
    ("ceb", "rnn"): RNNCEB,
    ("fgib", "mlp"): FGIB,
    ("fgib", "cnn"): CNNFGIB,
    ("fgib", "gnn"): GNNFGIB,
    ("fgib", "rnn"): RNNFGIB,
    ("opb", "mlp"): OPB,
    ("opb", "cnn"): CNNOPB,
    ("opb", "gnn"): GNNOPB,
    ("opb", "rnn"): RNNOPB,
}


def get_dataset_name(task: str) -> str:
    """任务名 → 数据集命名（输出目录 / 调参结果命名前缀用）。"""
    return (
        "california" if task == "housing"
        else "imagenet100" if task == "imagenet100"
        else "cora" if task == "cora"
        else "imdb" if task == "imdb"
        else "agnews" if task == "agnews"
        else "stsb" if task == "stsb"
        else "zinc" if task == "zinc"
        else "agedb" if task == "agedb"
        else "mnist"
    )


def build_parser():
    """构建 argparse 解析器（train.py 与 tune.py 共享，tune.py 会覆盖部分参数）。"""
    parser = argparse.ArgumentParser(description="Train MNIST MLP/CNN baseline and VIB / CEB / FGIB / OPB variants")
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn", "gcn", "rnn", "vib", "ceb", "fgib", "opb", "svib", "nib", "dvcca"],
        default="mlp",
        help="svib is Squared-IB (ICLR 2019 Caveats): a VIB subclass whose forward "
        "returns the squared KL (loss becomes CE + β·KL²), mainly for classification; "
        "nib is Nonlinear IB (Entropy 2019): noise-injected bottleneck (sigma^2 "
        "trainable, initialized at the paper value 1.0) with a prior-free "
        "pairwise-distance upper bound on I(X;M) (loss = CE + β·Î); "
        "dvcca is supervised beta-DVCCA (DVMIB, JMLR 2025): VIB plus an input "
        "reconstruction decoder (loss = CE + β·KL + β·MSE_recon), "
        "not available for the RNN backbone; "
        "opb is Orthogonal-Prior Bottleneck (paper/OPB.txt): a CEB variant whose "
        "prior means are the QR-orthonormalized per-class outputs of the prior net "
        "(classification tasks only)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mlp", "cnn", "gnn", "rnn"],
        default="mlp",
        help="backbone of VIB variants; ignored when --model is mlp, cnn, gcn or rnn",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mnist", "housing", "imagenet100", "cora", "imdb", "agnews", "stsb", "zinc", "agedb"],
        default="mnist",
        help="mnist is the default (MNIST classification); housing is California "
        "Housing regression (MLP backbone only); imagenet100 uses pretrained "
        "ResNet50 features (MLP backbone only); cora uses GNN backbone only; "
        "imdb, agnews and stsb use RNN backbone only (agnews uses BERT token "
        "features; stsb is STS-B text similarity regression, RNN backbone only); "
        "zinc is ZINC-12k molecular graph regression (GNN backbone only); "
        "agedb is AgeDB face-image age regression (RGB 64×64, MLP/CNN backbones; "
        "ceb/fgib condition the prior on the continuous scaled y)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate（未指定时默认 1e-3；cora 任务默认 0.01）",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="hidden layer dims of the MLP / GNN backbone; "
        "RNN backbone uses the first element as word embedding dim and the list "
        "as per-layer LSTM hidden dims (layer count = list length)",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--z-dim", type=int, default=256, help="dimension of z in the VIB-family models")
    parser.add_argument(
        "--anchor-scale",
        type=float,
        default=4.0,
        help="scale of the anchor prior means in fgib/opb. fgib classification: "
        "orthogonal per-class directions scaled by this value, 0 = identical N(0,I) "
        "anchors; fgib regression: anchors lie on a sphere of this radius via fixed "
        "random Fourier features, 0 = N(0,I) prior; opb: scale of the QR-orthonormal "
        "per-class prior means (classification only)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=250,
        help="max sequence length of IMDb reviews / STS-B sentence pairs "
        "(imdb/stsb only; stsb 中两个句子各自截断到 (max_len-1)//2 后以 [SEP] 拼接)",
    )
    parser.add_argument(
        "--random-labels",
        action="store_true",
        help="randomize training labels with a fixed seed (information-free dataset "
        "memorization experiment; classification tasks only, val/test labels stay real)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-3,
        help="weight of the KL term in VIB/CEB/FGIB; for svib the penalty "
        "is β·KL², so the meaningful β range differs (sweep with tune.py)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early stopping patience: stop after N epochs without val AUC improvement",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="number of complete training runs, test metrics are averaged over them",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed of the first run (run i uses seed+i-1)",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="model save path (default output/{model}/mnist_{model}.pt)",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="training log file path (default output/{model}/train_{model}.log)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存模型 checkpoint，最优模型仅保留在内存中用于测试评估（默认保存）",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.task == "housing" and (args.model == "cnn" or args.backbone == "cnn"):
        parser.error("CNN backbone is not supported for housing")
    if args.task in ("housing", "stsb", "zinc", "agedb") and args.model == "opb":
        parser.error(
            "OPB 仅支持分类任务（v1 定义需要有限 K 个类别构造正交锚点表，见 paper/OPB.txt）"
        )
    if args.task in ("cora", "zinc"):
        ok = args.model == "gcn" or (
            args.backbone == "gnn" and args.model in ("vib", "ceb", "fgib", "opb", "svib", "nib", "dvcca")
        )
        if not ok:
            parser.error("Cora/ZINC 任务仅支持 --model gcn 或 --backbone gnn 加 vib/ceb/fgib/opb/svib/nib/dvcca")
    elif args.model == "gcn" or args.backbone == "gnn":
        parser.error("GNN backbone is only supported for cora/zinc")
    if args.task in ("imdb", "agnews", "stsb"):
        ok = args.model == "rnn" or (
            args.backbone == "rnn" and args.model in ("vib", "ceb", "fgib", "opb", "svib", "nib", "dvcca")
        )
        if not ok:
            parser.error(
                f"{args.task} task only supports --model rnn or "
                "--backbone rnn with vib/ceb/fgib/opb/svib/nib/dvcca"
            )
    elif args.model == "rnn" or args.backbone == "rnn":
        parser.error("RNN backbone is only supported for imdb/agnews/stsb")

    # 图批拼块对角后 512 批约 1.2 万节点、稠密邻接约 566MB；128 批约 3 千
    # 节点 / 36MB（benchmarking-gnns 的 ZINC 约定批大小），未显式设置时覆写
    if args.task == "zinc" and args.batch_size == parser.get_default("batch_size"):
        args.batch_size = 128
    # 学习率任务级默认值：cora 全图批训练在 1e-3 下收敛过慢，用 0.01；
    # 显式指定 --lr 时始终尊重用户值
    if args.lr is None:
        args.lr = 0.01 if args.task == "cora" else 1e-3

    # mlp/cnn/gcn/rnn 为自带骨干的基线；变体由 --backbone 指定骨干
    if args.model in ("mlp", "cnn", "rnn"):
        backbone = args.model
    elif args.model == "gcn":
        backbone = "gnn"
    else:
        backbone = args.backbone

    # 输出目录按 {dataset}_{backbone}_{model} 命名（基线为 {dataset}_{model}），
    # 如 mnist_mlp_vib、mnist_cnn_vib、california_mlp_ceb、cora_gnn_vib、imdb_rnn_vib
    dataset_name = get_dataset_name(args.task)
    if args.model in ("mlp", "cnn", "gcn", "rnn"):
        output_name = f"{dataset_name}_{args.model}"
    else:
        output_name = f"{dataset_name}_{backbone}_{args.model}"

    output_dir = os.path.join("output", output_name)
    # 仅当 save_path/log_path 的默认值需要落在 output 目录内时才创建它；
    # tune.py 的 --no-save + 显式 --log-path 调用不会在 output/ 留下空子文件夹
    if args.save_path is None or args.log_path is None:
        os.makedirs(output_dir, exist_ok=True)
    if args.save_path is None:
        args.save_path = os.path.join(output_dir, f"{output_name}.pt")
    if args.log_path is None:
        args.log_path = os.path.join(output_dir, f"train_{output_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_path, encoding="utf-8"),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Args: {vars(args)}")

    if args.task == "cora":
        # 图任务为全图批（转导式），无 DataLoader；数据只需加载一次（划分固定）
        x, adj, y, train_mask, val_mask, test_mask = get_cora_data(
            args.data_dir, random_labels=args.random_labels
        )
        x, adj, y = x.to(device), adj.to(device), y.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        target_scaler = None
    elif args.task == "zinc":
        train_loader, val_loader, test_loader, target_scaler, zinc_input_dim = (
            get_zinc_dataloaders(args.batch_size, args.data_dir)
        )
    elif args.task == "agedb":
        train_loader, val_loader, test_loader, target_scaler = (
            get_agedb_dataloaders(args.batch_size, args.data_dir)
        )
    elif args.task == "housing":
        train_loader, val_loader, test_loader, target_scaler = (
            get_california_dataloaders(args.batch_size, args.data_dir)
        )
    elif args.task == "imdb":
        train_loader, val_loader, test_loader, vocab_size = get_imdb_dataloaders(
            args.batch_size, args.data_dir, max_len=args.max_len,
            random_labels=args.random_labels,
        )
        target_scaler = None
    elif args.task == "agnews":
        train_loader, val_loader, test_loader, agnews_input_dim = get_agnews_dataloaders(
            args.batch_size, args.data_dir, random_labels=args.random_labels
        )
        target_scaler = None
    elif args.task == "stsb":
        train_loader, val_loader, test_loader, vocab_size, target_scaler, glove_matrix = (
            get_stsb_dataloaders(
                args.batch_size, args.data_dir, max_len=args.max_len
            )
        )
    else:
        if args.task == "imagenet100":
            train_loader, val_loader, test_loader, imagenet100_input_dim, imagenet100_feature_pool = (
                get_imagenet100_dataloaders(
                    args.batch_size, args.data_dir,
                    random_labels=args.random_labels, spatial=(backbone == "cnn"),
                )
            )
        else:
            train_loader, val_loader, test_loader = get_mnist_dataloaders(
                args.batch_size, args.data_dir, random_labels=args.random_labels
            )
        target_scaler = None

    model_cls = MODEL_CLASSES[(args.model, backbone)]
    model_kwargs = dict(dropout=args.dropout)
    if backbone in ("mlp", "gnn", "rnn"):
        model_kwargs["hidden_dims"] = tuple(args.hidden_dims)
    if args.model not in ("mlp", "cnn", "gcn", "rnn"):
        model_kwargs["z_dim"] = args.z_dim
    if args.model in ("fgib", "opb"):
        model_kwargs["anchor_scale"] = args.anchor_scale
    if args.task == "housing":
        model_kwargs["input_dim"] = 8
        model_kwargs["num_classes"] = 1
        if args.model in ("ceb", "fgib"):
            model_kwargs["continuous_y"] = True
    elif args.task == "stsb":
        model_kwargs["vocab_size"] = vocab_size
        model_kwargs["num_classes"] = 1
        model_kwargs["pretrained_emb"] = glove_matrix
        model_kwargs["pooling"] = "max"
        # 词嵌入维度取自 GloVe 矩阵（100），--hidden-dims 仅作逐层 LSTM 隐层
        # 维度（模型内对 pretrained_emb 自动解耦嵌入维度与层维度）；
        # 未显式指定时 stsb 默认单层 LSTM 256（多层在小数据集上过拟合，
        # 实验验证：单层 val R²≈0.27 vs 两层 0.07）
        if tuple(args.hidden_dims) == tuple(parser.get_default("hidden_dims")):
            model_kwargs["hidden_dims"] = (256,)
        if args.model in ("ceb", "fgib"):
            model_kwargs["continuous_y"] = True
    elif args.task == "imagenet100":
        model_kwargs["num_classes"] = 100
        if backbone == "cnn":
            # CNN 需要保留空间结构的池化特征；回退到全局池化（1×1）文件时无法卷积
            if imagenet100_feature_pool == 1:
                parser.error(
                    "imagenet100 CNN 骨干需要空间池化特征文件（缺失 "
                    "imagenet100_resnet50_layer3_pool4_features.npz，"
                    "当前回退为全局池化特征），请先运行 "
                    "datasets/extract_imagenet100_features.py --pool 4"
                )
            model_kwargs["input_channels"] = imagenet100_input_dim // (imagenet100_feature_pool ** 2)
            model_kwargs["input_size"] = imagenet100_feature_pool
        else:
            model_kwargs["input_dim"] = imagenet100_input_dim
    elif args.task == "cora":
        model_kwargs["input_dim"] = 1433
        model_kwargs["num_classes"] = 7
    elif args.task == "zinc":
        model_kwargs["input_dim"] = zinc_input_dim
        model_kwargs["num_classes"] = 1
        model_kwargs["pooling"] = "mean"  # 图级读出（编码器后 mean 池化）
        if args.model in ("ceb", "fgib"):
            model_kwargs["continuous_y"] = True
    elif args.task == "agedb":
        model_kwargs["num_classes"] = 1
        if backbone == "mlp":
            model_kwargs["input_dim"] = 3 * 64 * 64  # RGB 64×64 展平
        else:  # backbone == "cnn"（gnn/rnn 已被约束块拒绝）
            model_kwargs["input_channels"] = 3
            model_kwargs["input_size"] = 64
        if args.model in ("ceb", "fgib"):
            model_kwargs["continuous_y"] = True
    elif args.task == "imdb":
        model_kwargs["vocab_size"] = vocab_size
        model_kwargs["num_classes"] = 2
    elif args.task == "agnews":
        model_kwargs["input_dim"] = agnews_input_dim
        model_kwargs["num_classes"] = 4

    criterion = nn.MSELoss() if args.task in ("housing", "stsb", "zinc", "agedb") else nn.CrossEntropyLoss()
    test_results = []
    run_train_accs = []

    for run in range(1, args.runs + 1):
        seed = args.seed + run - 1
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logging.info(f"===== Run {run}/{args.runs} (seed {seed}) =====")

        model = model_cls(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        stem, ext = os.path.splitext(args.save_path)
        run_save_path = f"{stem}_run{run}{ext}"

        # 初始 -inf：回归任务 val R² 可任意负（如 MLP 前几轮远低于 -1），
        # 首个 epoch 必定保存，避免全部 epoch 无改善时无 checkpoint 可加载
        best_score, best_epoch, patience_counter = -float("inf"), 0, 0
        best_state = None
        metric_name = "R2" if args.task in ("housing", "stsb", "zinc", "agedb") else "AUC"
        train_acc = None

        for epoch in range(1, args.epochs + 1):
            if args.task == "cora":
                train_loss, train_acc = train_one_epoch_cora(
                    model, x, adj, y, train_mask, optimizer, criterion, args.beta, device
                )
                val_loss, val_acc, val_auc, val_pre, val_rec = evaluate_cora(
                    model, x, adj, y, val_mask, criterion, args.beta, device
                )
                val_score = val_auc
                logging.info(
                    f"Epoch {epoch:>3}/{args.epochs} | "
                    f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f} "
                    f"Pre {val_pre:.4f} Rec {val_rec:.4f}"
                )
            elif args.task == "zinc":
                train_loss, _ = train_one_epoch_zinc(
                    model, train_loader, optimizer, criterion, args.beta, device
                )
                val_loss, val_mae, val_r2 = evaluate_zinc(
                    model, val_loader, criterion, args.beta, device, target_scaler
                )
                val_score = val_r2
                logging.info(
                    f"Epoch {epoch:>3}/{args.epochs} | "
                    f"Train Loss {train_loss:.4f} | "
                    f"Val Loss {val_loss:.4f} MAE {val_mae:.4f} R2 {val_r2:.4f}"
                )
            elif args.task in ("housing", "stsb", "agedb"):
                train_loss, _ = train_one_epoch(
                    model, train_loader, optimizer, criterion, args.beta, device, args.task
                )
                val_loss, val_mae, val_r2 = evaluate(
                    model, val_loader, criterion, args.beta, device, args.task,
                    target_scaler,
                )
                val_score = val_r2
                logging.info(
                    f"Epoch {epoch:>3}/{args.epochs} | "
                    f"Train Loss {train_loss:.4f} | "
                    f"Val Loss {val_loss:.4f} MAE {val_mae:.4f} R2 {val_r2:.4f}"
                )
            else:
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion, args.beta, device, args.task
                )
                val_loss, val_acc, val_auc, val_pre, val_rec = evaluate(
                    model, val_loader, criterion, args.beta, device, args.task
                )
                val_score = val_auc
                logging.info(
                    f"Epoch {epoch:>3}/{args.epochs} | "
                    f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f} "
                    f"Pre {val_pre:.4f} Rec {val_rec:.4f}"
                )

            if val_score > best_score:
                best_score, best_epoch = val_score, epoch
                patience_counter = 0
                if args.no_save:
                    # 保留在内存（拷贝到 CPU 省显存），测试评估时直接加载
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    torch.save(model.state_dict(), run_save_path)
                logging.info(f"  Val {metric_name} improved to {best_score:.4f}, model saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info(
                        f"Val {metric_name} did not improve for {args.patience} consecutive epochs, "
                        f"early stopping at Epoch {epoch} "
                        f"(best {metric_name} {best_score:.4f} @ Epoch {best_epoch})"
                    )
                    break

        if args.random_labels and train_acc is not None:
            run_train_accs.append(train_acc)
        if args.no_save:
            model.load_state_dict(best_state)
        else:
            model.load_state_dict(torch.load(run_save_path, weights_only=True))
        if args.task == "cora":
            test_loss, test_acc, test_auc, test_pre, test_rec = evaluate_cora(
                model, x, adj, y, test_mask, criterion, args.beta, device
            )
            logging.info(
                f"Run {run}/{args.runs} | Test (best model @ Epoch {best_epoch}) | "
                f"Loss {test_loss:.4f} Acc {test_acc:.4f} AUC {test_auc:.4f} "
                f"Pre {test_pre:.4f} Rec {test_rec:.4f}"
            )
            test_results.append((test_loss, test_acc, test_auc, test_pre, test_rec))
        elif args.task in ("housing", "stsb", "zinc", "agedb"):
            if args.task == "zinc":
                test_loss, test_mae, test_r2 = evaluate_zinc(
                    model, test_loader, criterion, args.beta, device, target_scaler
                )
            else:
                test_loss, test_mae, test_r2 = evaluate(
                    model, test_loader, criterion, args.beta, device, args.task,
                    target_scaler,
                )
            logging.info(
                f"Run {run}/{args.runs} | Test (best model @ Epoch {best_epoch}) | "
                f"Loss {test_loss:.4f} MAE {test_mae:.4f} R2 {test_r2:.4f}"
            )
            test_results.append((test_loss, test_mae, test_r2))
        else:
            test_loss, test_acc, test_auc, test_pre, test_rec = evaluate(
                model, test_loader, criterion, args.beta, device, args.task
            )
            logging.info(
                f"Run {run}/{args.runs} | Test (best model @ Epoch {best_epoch}) | "
                f"Loss {test_loss:.4f} Acc {test_acc:.4f} AUC {test_auc:.4f} "
                f"Pre {test_pre:.4f} Rec {test_rec:.4f}"
            )
            test_results.append((test_loss, test_acc, test_auc, test_pre, test_rec))

    means = torch.tensor(test_results).mean(dim=0)
    stds = torch.tensor(test_results).std(dim=0)
    if args.task in ("housing", "stsb", "zinc", "agedb"):
        logging.info(
            f"Average over {args.runs} runs | Test "
            f"Loss {means[0]:.4f}±{stds[0]:.4f} MAE {means[1]:.4f}±{stds[1]:.4f} "
            f"R2 {means[2]:.4f}±{stds[2]:.4f}"
        )
    else:
        logging.info(
            f"Average over {args.runs} runs | Test "
            f"Loss {means[0]:.4f}±{stds[0]:.4f} Acc {means[1]:.4f}±{stds[1]:.4f} "
            f"AUC {means[2]:.4f}±{stds[2]:.4f} Pre {means[3]:.4f}±{stds[3]:.4f} "
            f"Rec {means[4]:.4f}±{stds[4]:.4f}"
        )
    if args.random_labels and run_train_accs:
        train_accs = torch.tensor(run_train_accs)
        logging.info(
            f"Average over {args.runs} runs | Train Acc (final epoch) "
            f"{train_accs.mean():.4f}±{train_accs.std():.4f}"
        )


if __name__ == "__main__":
    main()
