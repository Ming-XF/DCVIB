"""Cora 引文网络数据集：下载、解析与 60/20/20 分层划分。

数据源为 pygcn 仓库的 cora.content（2708 节点：paper_id + 1433 维词袋特征 + 标签）
与 cora.cites（5429 条边），首次使用时下载并缓存到 data_dir/cora/，之后离线可用。

解析遵循 Kipf & Welling GCN 的经典配方：特征行归一化（sum=1）、邻接加自环后
对称归一化 Â = D^(-1/2)(A+I)D^(-1/2)，返回稠密张量（2708x2708，约 29MB）。
"""
import logging
import os
import urllib.request

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATA_URLS = [
    # 主源 GitHub raw，备源 jsdelivr 镜像
    "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/{}",
    "https://cdn.jsdelivr.net/gh/tkipf/pygcn@master/data/cora/{}",
]
FILES = ("cora.content", "cora.cites")


def download_cora(data_dir: str = "./data") -> str:
    """确保 cora.content / cora.cites 已缓存在 data_dir/cora/，返回缓存目录。"""
    cache_dir = os.path.join(data_dir, "cora")
    os.makedirs(cache_dir, exist_ok=True)
    for fname in FILES:
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        for base in DATA_URLS:
            try:
                logger.info("下载 %s 到 %s", fname, path)
                urllib.request.urlretrieve(base.format(fname), path)
                break
            except Exception as exc:
                logger.warning("下载失败（%s）：%s", base.format(fname), exc)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"无法下载 {fname}，请手动放置到 {cache_dir}/")
    return cache_dir


def parse_cora(cache_dir: str):
    """解析 Cora 数据，返回 (x, adj_norm, y)：行归一化特征、对称归一化邻接、标签 0..6。"""
    content = np.genfromtxt(os.path.join(cache_dir, "cora.content"), dtype=np.dtype(str))
    features = sp.csr_matrix(content[:, 1:-1], dtype=np.float32)
    labels_raw = content[:, -1]
    classes = sorted(set(labels_raw))
    class_map = {c: i for i, c in enumerate(classes)}
    labels = np.array([class_map[c] for c in labels_raw], dtype=np.int64)

    # 边表 -> 对称邻接；丢弃 content 中不存在的节点 id 的边
    idx = np.array(content[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(cache_dir, "cora.cites"), dtype=np.int32)
    mapped = np.array(
        [idx_map.get(e, -1) for e in edges_unordered.flatten()], dtype=np.int64
    ).reshape(edges_unordered.shape)
    valid = (mapped >= 0).all(axis=1)
    dropped = (~valid).sum()
    if dropped:
        logger.warning("丢弃 %d 条含未知节点的边", dropped)
    mapped = mapped[valid]
    n = labels.shape[0]
    adj = sp.coo_matrix(
        (np.ones(mapped.shape[0], dtype=np.float32), (mapped[:, 0], mapped[:, 1])),
        shape=(n, n),
    )
    adj = adj + adj.T - adj.multiply(adj.T)  # 无向化（去重）

    # 特征行归一化（sum=1）
    rowsum = np.asarray(features.sum(1)).ravel()
    rowsum[rowsum == 0] = 1.0
    features = sp.diags(1.0 / rowsum) @ features

    # 加自环 + 对称归一化 Â = D^(-1/2)(A+I)D^(-1/2)
    adj = adj + sp.eye(n, dtype=np.float32)
    deg = np.asarray(adj.sum(1)).ravel()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = sp.diags(deg_inv_sqrt) @ adj @ sp.diags(deg_inv_sqrt)

    x = torch.tensor(np.asarray(features.todense(), dtype=np.float32))
    adj_norm = torch.tensor(np.asarray(adj_norm.todense(), dtype=np.float32))
    y = torch.tensor(labels, dtype=torch.long)
    return x, adj_norm, y


def get_cora_data(
    data_dir: str = "./data",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    random_labels: bool = False,
):
    """加载 Cora 并返回 (x, adj_norm, y, train_mask, val_mask, test_mask)。

    60/20/20 分层划分（seed 固定），mask 为 bool 张量；GCN 训练为全图批
    （转导式），DataLoader 不适用，由 train.py 的 cora 分支直接消费。
    random_labels=True 时把训练节点标签按固定 seed 随机化（信息自由数据集
    I(X;Y)=0 的记忆实验），验证/测试节点保持真实标签。
    """
    cache_dir = download_cora(data_dir)
    x, adj_norm, y = parse_cora(cache_dir)

    y_np = y.numpy()
    indices = np.arange(len(y_np))
    train_idx, test_idx, y_train, _ = train_test_split(
        indices, y_np, test_size=test_ratio, random_state=seed, stratify=y_np
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_idx, y_train, test_size=val_ratio / (1 - test_ratio),
        random_state=seed, stratify=y_train,
    )
    logging.info(
        "Cora 划分（60/20/20 分层）：train %d / val %d / test %d（共 %d 节点，%d 类）",
        len(train_idx), len(val_idx), len(test_idx), len(y_np), y.max().item() + 1,
    )

    def make_mask(idx):
        mask = torch.zeros(len(y_np), dtype=torch.bool)
        mask[idx] = True
        return mask

    train_mask, val_mask, test_mask = (
        make_mask(train_idx), make_mask(val_idx), make_mask(test_idx)
    )

    if random_labels:
        rng = torch.Generator().manual_seed(42)
        y = y.clone()
        y[train_mask] = torch.randint(
            0, int(y.max().item()) + 1, (int(train_mask.sum()),), generator=rng
        )
        logging.info("随机标签实验：训练节点标签已随机化（I(X;Y)=0，固定 seed 42）")

    return x, adj_norm, y, train_mask, val_mask, test_mask
