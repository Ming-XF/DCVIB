"""MNIST 多层感知机（MLP）训练脚本。

用法：
    python train.py                 # 使用默认参数训练
    python train.py --epochs 20 --batch-size 128 --lr 1e-3
"""

import argparse

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MLP


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """返回 (loss, accuracy, 宏平均 one-vs-rest ROC AUC)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = outputs.softmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        all_probs.append(probs)
        all_labels.append(labels)

    probs = torch.cat(all_probs).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    return total_loss / total, correct / total, auc


def main():
    parser = argparse.ArgumentParser(description="训练 MNIST MLP 模型")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值：验证集 AUC 连续多少轮无提升则停止")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="./mnist_mlp.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, args.data_dir
    )

    model = MLP(
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc, best_epoch, patience_counter = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  验证集 AUC 提升至 {best_auc:.4f}，模型已保存")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"验证集 AUC 连续 {args.patience} 轮无提升，"
                    f"于 Epoch {epoch} 提前停止（最佳 AUC {best_auc:.4f} @ Epoch {best_epoch}）"
                )
                break

    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device)
    print(
        f"测试集（最佳模型 @ Epoch {best_epoch}） | "
        f"Loss {test_loss:.4f} Acc {test_acc:.4f} AUC {test_auc:.4f}"
    )
    print(f"模型已保存至 {args.save_path}")


if __name__ == "__main__":
    main()
