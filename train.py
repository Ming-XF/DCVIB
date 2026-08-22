"""MNIST 分类 / California Housing 回归训练脚本：MLP / CNN 基线及 VIB / SCVIB / DCVIB 变体。

用法：
    python train.py                 # 使用默认参数训练 MLP
    python train.py --model cnn     # 训练 CNN 基线
    python train.py --model vib     # 训练 MLP 版 VIB
    python train.py --backbone cnn --model vib    # 训练 CNN 版 VIB
    python train.py --backbone cnn --model scvib  # 训练 CNN 版 SCVIB
    python train.py --backbone cnn --model dcvib  # 训练 CNN 版 DCVIB
    python train.py --task regression --model vib     # California Housing 回归（MLP 骨干）
    python train.py --task regression --model scvib   # SCVIB 回归（连续标签条件先验）
    python train.py --task regression --model dcvib   # DCVIB 回归（连续标签条件先验）
    python train.py --epochs 20 --batch-size 128 --lr 1e-3
    python train.py --runs 5 --seed 0   # 重复训练 5 次并报告测试集平均指标
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from model import CNN, DCVIB, MLP, SCVIB, VIB
from model.cnn import DCVIB as CNDCVIB, SCVIB as CNNSCVIB, VIB as CNNVIB


def get_dataloaders(batch_size: int, data_dir: str = "./data"):
    """加载 MNIST 数据集并返回训练/验证/测试 DataLoader。

    MNIST 官方只有训练集（60k）和测试集（10k），验证集从训练集中切出 10k。
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的均值与标准差
        ]
    )

    full_train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [50000, 10000], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_california_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """加载 California Housing 并返回训练/验证/测试 DataLoader 及目标 scaler。

    划分 60/20/20（seed 固定），特征用 StandardScaler 标准化、目标用 MinMaxScaler
    归一化到 [0,1]（均仅在训练集上拟合）；评估时用返回的 y_scaler 逆归一化指标。
    数据缓存在 data_dir/california_housing/cal_housing_py3.pkz（sklearn data_home）。
    """
    data_home = os.path.join(data_dir, "california_housing")
    X, y = fetch_california_housing(data_home=data_home, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio / (1 - test_ratio), random_state=seed
    )

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))

    def make_dataset(X, y):
        return TensorDataset(
            torch.tensor(x_scaler.transform(X), dtype=torch.float32),
            torch.tensor(
                y_scaler.transform(y.reshape(-1, 1)).ravel(), dtype=torch.float32
            ),
        )

    train_dataset = make_dataset(X_train, y_train)
    val_dataset = make_dataset(X_val, y_val)
    test_dataset = make_dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_scaler


def run_model(model, images, labels, stochastic):
    """统一 MLP / CNN 骨干下各模型的前向接口，返回 (logits, kl)。"""
    if isinstance(model, (VIB, SCVIB, CNNVIB, CNNSCVIB)):
        return model(images, labels, stochastic=stochastic)
    if isinstance(model, (DCVIB, CNDCVIB)):
        return model(images, labels)
    return model(images), None


def train_one_epoch(model, loader, optimizer, criterion, beta, device, task="classification"):
    model.train()
    total_loss, total = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, kl = run_model(model, images, labels, stochastic=True)
        if task == "regression":
            logits = logits.squeeze(-1)
        loss = criterion(logits, labels)
        if kl is not None:
            loss = loss + beta * kl
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return total_loss / total


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
        logits, kl = run_model(model, images, labels, stochastic=False)
        if task == "regression":
            logits = logits.squeeze(-1)
        loss = criterion(logits, labels)
        if kl is not None:
            loss = loss + beta * kl

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        if task == "regression":
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

    if task == "regression":
        preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
        labels = target_scaler.inverse_transform(labels.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        return total_loss / total, mae, r2

    probs = torch.cat(all_probs).cpu().numpy()
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")

    return total_loss / total, correct / total, auc, precision, recall


MODEL_CLASSES = {
    ("mlp", "mlp"): MLP,
    ("cnn", "cnn"): CNN,
    ("vib", "mlp"): VIB,
    ("vib", "cnn"): CNNVIB,
    ("scvib", "mlp"): SCVIB,
    ("scvib", "cnn"): CNNSCVIB,
    ("dcvib", "mlp"): DCVIB,
    ("dcvib", "cnn"): CNDCVIB,
}


def main():
    parser = argparse.ArgumentParser(description="Train MNIST MLP/CNN baseline and VIB / SCVIB / DCVIB variants")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn", "vib", "scvib", "dcvib"], default="mlp")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mlp", "cnn"],
        default="mlp",
        help="backbone of VIB variants; ignored when --model is mlp or cnn",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="regression uses California Housing (MLP backbone only)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="hidden layer dims of the MLP backbone",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--z-dim", type=int, default=256, help="dimension of z in VIB/SCVIB/DCVIB")
    parser.add_argument("--beta", type=float, default=1e-3, help="weight of the KL term in VIB/SCVIB/DCVIB")
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
    args = parser.parse_args()

    if args.task == "regression" and (args.model == "cnn" or args.backbone == "cnn"):
        parser.error("CNN backbone is not supported for regression")

    # mlp/cnn 为自带骨干的基线；变体由 --backbone 指定骨干
    if args.model in ("mlp", "cnn"):
        backbone = args.model
    else:
        backbone = args.backbone

    # 输出目录按 {dataset}_{backbone}_{model} 命名（基线为 {dataset}_{model}），
    # 如 mnist_mlp_vib、mnist_cnn_vib、california_mlp_scvib
    dataset_name = "california" if args.task == "regression" else "mnist"
    if args.model in ("mlp", "cnn"):
        output_name = f"{dataset_name}_{args.model}"
    else:
        output_name = f"{dataset_name}_{backbone}_{args.model}"

    output_dir = os.path.join("output", output_name)
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

    if args.task == "regression":
        train_loader, val_loader, test_loader, target_scaler = (
            get_california_dataloaders(args.batch_size, args.data_dir)
        )
    else:
        train_loader, val_loader, test_loader = get_dataloaders(
            args.batch_size, args.data_dir
        )
        target_scaler = None

    model_cls = MODEL_CLASSES[(args.model, backbone)]
    model_kwargs = dict(dropout=args.dropout)
    if backbone == "mlp":
        model_kwargs["hidden_dims"] = tuple(args.hidden_dims)
    if args.model not in ("mlp", "cnn"):
        model_kwargs["z_dim"] = args.z_dim
    if args.task == "regression":
        model_kwargs["input_dim"] = 8
        model_kwargs["num_classes"] = 1
        if args.model in ("scvib", "dcvib"):
            model_kwargs["continuous_y"] = True

    criterion = nn.MSELoss() if args.task == "regression" else nn.CrossEntropyLoss()
    test_results = []

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

        best_score, best_epoch, patience_counter = -1.0, 0, 0
        metric_name = "R2" if args.task == "regression" else "AUC"

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, args.beta, device, args.task
            )
            if args.task == "regression":
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
                val_loss, val_acc, val_auc, val_pre, val_rec = evaluate(
                    model, val_loader, criterion, args.beta, device, args.task
                )
                val_score = val_auc
                logging.info(
                    f"Epoch {epoch:>3}/{args.epochs} | "
                    f"Train Loss {train_loss:.4f} | "
                    f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f} "
                    f"Pre {val_pre:.4f} Rec {val_rec:.4f}"
                )

            if val_score > best_score:
                best_score, best_epoch = val_score, epoch
                patience_counter = 0
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

        model.load_state_dict(torch.load(run_save_path, weights_only=True))
        if args.task == "regression":
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
    if args.task == "regression":
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


if __name__ == "__main__":
    main()
