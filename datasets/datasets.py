"""MNIST / ImageNet-100 特征 / California Housing 数据集加载与预处理。

各任务提供 DataLoader 构造函数，统一返回 (train_loader, val_loader, test_loader)
（回归额外返回目标 scaler）：
- get_mnist_dataloaders：MNIST 官方训练/测试集，验证集从训练集切 10k（seed 42）
- get_imagenet100_dataloaders：预训练 ResNet50 特征（datasets/extract_imagenet100_features.py
  生成的 npz），60/20/20 分层划分 + StandardScaler（仅训练集拟合），供 MLP 骨干
- get_california_dataloaders：sklearn California Housing，60/20/20 划分，特征标准化、
  目标 MinMaxScaler 归一化到 [0,1]
"""
import logging
import os

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    batch_size: int, data_dir: str = "./data", random_labels: bool = False
):
    """加载 MNIST 数据集并返回训练/验证/测试 DataLoader。

    MNIST 官方只有训练集（60k）和测试集（10k），验证集从训练集中切出 10k。
    random_labels=True 时把训练集标签按固定 seed 随机化（信息自由数据集
    I(X;Y)=0 的记忆实验），验证/测试集保持真实标签。
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

    if random_labels:
        # 只随机化训练切分的标签（固定 seed 42，各模型/各 run 看到同一组随机标签）
        rng = torch.Generator().manual_seed(42)
        train_dataset.dataset.targets[train_dataset.indices] = torch.randint(
            0, 10, (len(train_dataset),), generator=rng
        )
        logging.info("随机标签实验：训练集标签已随机化（I(X;Y)=0，固定 seed 42）")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ImageNet-100 特征的空间池化尺寸（提取脚本 --pool 的对应值）：pool4 文件
# 存在时优先使用（保留 4×4 空间结构），否则回退全局池化的旧文件
IMAGENET100_FEATURE_POOL = 4


def get_imagenet100_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    random_labels: bool = False,
    spatial: bool = False,
):
    """加载 ImageNet-100 预训练特征并返回训练/验证/测试 DataLoader、特征维度与池化尺寸。

    特征来自 datasets/extract_imagenet100_features.py 生成的 npz：优先使用 layer3
    池化到 pool×pool 的文件（默认 pool=4，1024×4×4=16384 维，保留空间结构），
    不存在时回退全局平均池化的旧文件（1024 维）。60/20/20 分层划分（seed 固定），
    特征用 StandardScaler 标准化（仅在训练集上拟合）。random_labels=True 时把
    训练集标签按固定 seed 随机化（信息自由数据集 I(X;Y)=0 的记忆实验）。
    spatial=True（CNN 骨干）且特征带空间结构时，样本重排为 (B, C, pool, pool)
    四维张量；否则保持展平向量。
    返回 (train_loader, val_loader, test_loader, input_dim, feature_pool)，
    feature_pool 为特征的空间池化尺寸（全局池化回退时为 1，CNN 骨干据此判断
    特征是否保留空间结构）。
    """
    root = os.path.join(data_dir, "imagenet100")
    pool_file = os.path.join(root, "imagenet100_resnet50_layer3_features.npz")
    feature_pool = 1
    if IMAGENET100_FEATURE_POOL > 1:
        preferred = os.path.join(
            root, f"imagenet100_resnet50_layer3_pool{IMAGENET100_FEATURE_POOL}_features.npz"
        )
        if os.path.exists(preferred):
            pool_file = preferred
            feature_pool = IMAGENET100_FEATURE_POOL
            logging.info("ImageNet-100 特征：使用 %d×%d 空间池化文件 %s",
                         IMAGENET100_FEATURE_POOL, IMAGENET100_FEATURE_POOL, preferred)
        else:
            logging.warning(
                "未找到 %s，回退全局池化特征 %s（特征维度减为 1024）",
                preferred, pool_file,
            )
    data = np.load(pool_file)
    X, y = data["features"], data["labels"]
    input_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio / (1 - test_ratio),
        random_state=seed, stratify=y_train,
    )
    logging.info(
        "ImageNet-100 划分（60/20/20 分层）：train %d / val %d / test %d",
        len(X_train), len(X_val), len(X_test),
    )

    if random_labels:
        rng = torch.Generator().manual_seed(42)
        y_train = torch.randint(
            0, 100, (len(y_train),), generator=rng
        ).numpy()
        logging.info("随机标签实验：训练集标签已随机化（I(X;Y)=0，固定 seed 42）")

    x_scaler = StandardScaler().fit(X_train)

    def make_dataset(X, y):
        x = torch.tensor(x_scaler.transform(X), dtype=torch.float32)
        if spatial and feature_pool > 1:
            x = x.view(-1, input_dim // (feature_pool ** 2), feature_pool, feature_pool)
        return TensorDataset(x, torch.tensor(y, dtype=torch.long))

    train_dataset = make_dataset(X_train, y_train)
    val_dataset = make_dataset(X_val, y_val)
    test_dataset = make_dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim, feature_pool


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
