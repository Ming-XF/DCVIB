"""MNIST / ImageNet-100 特征 / California Housing 数据集加载与预处理。

各任务提供 DataLoader 构造函数，统一返回 (train_loader, val_loader, test_loader)
（回归额外返回目标 scaler）：
- get_mnist_dataloaders：MNIST 官方训练/测试集，验证集从训练集切 10k（seed 42）
- get_imagenet100_dataloaders：预训练 ResNet50 特征（datasets/extract_imagenet100_features.py
  生成的 npz），60/20/20 分层划分 + StandardScaler（仅训练集拟合），供 MLP 骨干
- get_california_dataloaders：sklearn California Housing，60/20/20 划分，特征标准化、
  目标 MinMaxScaler 归一化到 [0,1]
"""
import fcntl
import json
import logging
import os
import warnings

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

# mmap 只读数组转 tensor 会触发"非可写"告警：DataLoader 只读使用缓存、从不回写，忽略该噪音
warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")


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


def _build_imagenet100_cache(pool_file, cache_dir, seed, test_ratio, val_ratio):
    """加载原始 npz 并一次性完成 60/20/20 划分与 StandardScaler 变换，落盘 float32 .npy。

    仅在缓存缺失或 npz 更新时执行（调用方持 flock 保证多进程只有一个构建者），
    之后各进程用 mmap 只读映射共享同一份页缓存，避免每进程重复持有约 8GB 特征副本。
    meta.json 记录 npz 的 mtime，特征重新提取（mtime 变化）时缓存自动失效重建；
    各 .npy 先写 .tmp 再原子改名，构建中途被杀不会留下半成品缓存。
    """
    logging.info("ImageNet-100 预处理缓存缺失，首次构建（npz 加载 + 划分 + 标准化，内存峰值较高）...")
    data = np.load(pool_file)
    X, y = data["features"], data["labels"]
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
    x_scaler = StandardScaler().fit(X_train)
    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in (
        ("X_train.npy", x_scaler.transform(X_train).astype(np.float32)),
        ("X_val.npy", x_scaler.transform(X_val).astype(np.float32)),
        ("X_test.npy", x_scaler.transform(X_test).astype(np.float32)),
        ("y_train.npy", y_train.astype(np.int64)),
        ("y_val.npy", y_val.astype(np.int64)),
        ("y_test.npy", y_test.astype(np.int64)),
    ):
        path = os.path.join(cache_dir, name)
        tmp = path + ".tmp.npy"  # np.save 对不以 .npy 结尾的路径会自动补后缀
        np.save(tmp, arr)
        os.replace(tmp, path)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"npz_mtime": os.path.getmtime(pool_file)}, f)
    logging.info("ImageNet-100 预处理缓存已写入 %s", cache_dir)


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
    特征用 StandardScaler 标准化（仅在训练集上拟合）。划分 + 标准化结果首次加载时
    缓存为 float32 .npy（imagenet100/preprocessed_pool{p}/），之后各进程用
    mmap 只读映射共享页缓存（多进程零拷贝、物理内存只占一份），npz mtime 变化时
    自动重建。random_labels=True 时把
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
    # 预处理缓存（划分 + StandardScaler 后的 float32 .npy）：各进程 mmap 共享
    # 同一份页缓存（多进程零拷贝，物理内存只占一份）；flock 保证首次构建只有一个进程执行
    cache_dir = os.path.join(root, f"preprocessed_pool{feature_pool}")
    lock_fd = os.open(os.path.join(root, "preprocess.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        meta_path = os.path.join(cache_dir, "meta.json")
        cache_ready = os.path.exists(meta_path)
        if cache_ready:
            with open(meta_path) as f:
                cache_ready = json.load(f).get("npz_mtime") == os.path.getmtime(pool_file)
        if not cache_ready:
            _build_imagenet100_cache(pool_file, cache_dir, seed, test_ratio, val_ratio)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    logging.info("ImageNet-100 预处理缓存：mmap 共享 %s", cache_dir)
    input_dim = np.load(os.path.join(cache_dir, "X_train.npy"), mmap_mode="r").shape[1]

    def make_dataset(x_name, y_name, y_override=None):
        x = torch.from_numpy(np.load(os.path.join(cache_dir, x_name), mmap_mode="r"))
        if spatial and feature_pool > 1:
            x = x.view(-1, input_dim // (feature_pool ** 2), feature_pool, feature_pool)
        if y_override is None:
            y_override = torch.from_numpy(np.load(os.path.join(cache_dir, y_name), mmap_mode="r"))
        return TensorDataset(x, y_override)

    y_train = torch.from_numpy(np.load(os.path.join(cache_dir, "y_train.npy"), mmap_mode="r"))
    if random_labels:
        rng = torch.Generator().manual_seed(42)
        y_train = torch.randint(0, 100, (len(y_train),), generator=rng)
        logging.info("随机标签实验：训练集标签已随机化（I(X;Y)=0，固定 seed 42）")

    train_dataset = make_dataset("X_train.npy", None, y_train)
    val_dataset = make_dataset("X_val.npy", "y_val.npy")
    test_dataset = make_dataset("X_test.npy", "y_test.npy")

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
