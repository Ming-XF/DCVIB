"""ZINC-12k 分子图回归数据集：下载、解析与图批组装。

数据源为 DGL 托管的 benchmarking-gnns 官方预处理 pickle（约 60.4MB），
首次使用时下载到 data_dir/zinc/ZINC.pkl，之后离线可用（下载需代理）。
官方划分 10000/1000/1000 固定，无需自行切分，不需要 RDKit。

pickle 结构（protocol 3）：[train, val, test, num_atom_type=28, num_bond_type=4]，
train/val/test 为内嵌的 MoleculeDGL 对象（data 属性为分子 dict 列表），每个
分子 dict 含 num_atom（节点数 N）、atom_type（torch LongTensor (N,)，值域
[0,28)）、bond_type（torch LongTensor (N,N)，值域 [0,4)）、
logP_SA_cycle_normalized（float，惩罚 logP 目标，即回归目标 y）。

本环境无 dgl 且不需要其中的图对象，加载时用自定义 Unpickler：不可导入的
全局引用（dgl、data.molecules 等）用占位类恢复，torch/numpy 引用走真实
导入（tensor 由 torch 旧版 reducer 反序列化）。

图转换：节点特征为 atom_type 的 one-hot（28 维 float32）；邻接由
bond_type != 0 得到（对称化去重、加自环）后按 D^-1/2(A+I)D^-1/2 对称
归一化（同 cora 配方）；边特征 bond_type 不使用（与 benchmarking-gnns
的 GCN 基线一致）。目标用 MinMaxScaler 归一化到 [0,1]（仅训练集拟合），
评估时用返回的 y_scaler 逆归一化回原始单位。collate 用 torch.block_diag
拼块对角邻接并给出 batch_idx 节点→图索引。
"""

import logging
import os
import pickle
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

ZINC_URL = "https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl"
ZINC_PKL = "ZINC.pkl"
PROCESSED_NAME = "zinc_processed.npz"
NUM_ATOM_TYPE = 28
NUM_BOND_TYPE = 4
# 官方划分（ZINC.pkl 内置，论文可比，不自行切分）
TRAIN_GRAPHS, VAL_GRAPHS, TEST_GRAPHS = 10000, 1000, 1000


class _DummyMeta(type):
    """占位元类：任意类属性访问都返回占位类本身。

    内嵌对象（如 dgl.scheme.Scheme）以 getattr(cls, '_reconstruct_scheme')
    形式重建时，返回占位类作为可调用对象即可恢复。
    """

    def __getattr__(cls, name):
        return _DummyClass


class _DummyClass(metaclass=_DummyMeta):
    """占位类：ZINC.pkl 内嵌的 dgl/data.molecules 对象在本环境不可导入且用不到。"""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        # state 可能为 dict（常规 __dict__）或 tuple（__slots__ 类），后者直接丢弃
        if isinstance(state, dict):
            self.__dict__.update(state)


class _ZincUnpickler(pickle.Unpickler):
    """ZINC.pkl 加载器：只取 MoleculeDGL 对象的 data 属性（分子 dict 列表），
    内嵌的 dgl 图对象等不可导入的全局引用用 _DummyClass 恢复。"""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _DummyClass


def download_zinc(data_dir: str = "./data") -> str:
    """确保 ZINC.pkl 在 data_dir/zinc/，返回缓存目录。"""
    cache_dir = os.path.join(data_dir, "zinc")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, ZINC_PKL)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return cache_dir
    logger.info("下载 ZINC 数据到 %s", path)
    try:
        urllib.request.urlretrieve(ZINC_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"无法下载 ZINC 数据集（{ZINC_URL}）：{exc}，"
            f"请先执行 source /etc/network_turbo 开启代理，"
            f"或手动放置 {ZINC_PKL} 到 {cache_dir}/ 后重试"
        )
    return cache_dir


def _molecule_to_graph(mol: dict) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """分子 dict → (x (N,28) one-hot, adj_norm (N,N), y)。"""
    n = int(mol["num_atom"])
    atom_type = mol["atom_type"].long()
    bond_type = mol["bond_type"].long()
    assert atom_type.max().item() < NUM_ATOM_TYPE, "atom_type 越界"
    assert bond_type.max().item() < NUM_BOND_TYPE, "bond_type 越界"

    x = F.one_hot(atom_type, NUM_ATOM_TYPE).float()
    adj = (bond_type != 0).float()
    adj = torch.clamp(adj + adj.t(), max=1.0)  # 无向化去重（bond_type 本身对称）
    adj = adj + torch.eye(n)  # 自环：单原子分子 deg=1，归一化不除零
    deg_inv_sqrt = adj.sum(dim=1).pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
    y = np.float32(mol["logP_SA_cycle_normalized"])
    return x.numpy(), adj_norm.numpy(), y


def _load_processed(cache_dir: str) -> tuple[list, list, list]:
    """加载（或解析并缓存）ZINC 图列表，返回 (train_graphs, val_graphs, test_graphs)，
    每个图为 (x, adj_norm, y) 元组。"""
    processed_path = os.path.join(cache_dir, PROCESSED_NAME)
    if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
        data = np.load(processed_path, allow_pickle=True)
        return data["train"].tolist(), data["val"].tolist(), data["test"].tolist()

    pkl_path = os.path.join(cache_dir, ZINC_PKL)
    with open(pkl_path, "rb") as f:
        train_ds, val_ds, test_ds, num_atom_type, num_bond_type = _ZincUnpickler(f).load()
    assert num_atom_type == NUM_ATOM_TYPE, f"atom 类型数异常：{num_atom_type}"
    assert num_bond_type == NUM_BOND_TYPE, f"bond 类型数异常：{num_bond_type}"
    splits = (train_ds.data, val_ds.data, test_ds.data)
    assert [len(s) for s in splits] == [TRAIN_GRAPHS, VAL_GRAPHS, TEST_GRAPHS], (
        f"划分规模异常：{[len(s) for s in splits]}"
    )
    mol0 = splits[0][0]
    assert {"num_atom", "atom_type", "bond_type", "logP_SA_cycle_normalized"} <= set(mol0), (
        f"分子 dict 键异常：{sorted(mol0)}"
    )

    graphs = []
    for mol_list in splits:
        graphs.append([_molecule_to_graph(mol) for mol in mol_list])
    np.savez(
        processed_path,
        train=np.array(graphs[0], dtype=object),
        val=np.array(graphs[1], dtype=object),
        test=np.array(graphs[2], dtype=object),
    )
    logger.info("ZINC 图解析完成并缓存到 %s（%d/%d/%d）",
                processed_path, len(graphs[0]), len(graphs[1]), len(graphs[2]))
    return graphs[0], graphs[1], graphs[2]


class ZINCDataset(Dataset):
    """ZINC 图数据集：item 为 (x (N,28), adj_norm (N,N), y 缩放后标量)。"""

    def __init__(self, graphs: list):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        x, adj, y = self.graphs[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(adj),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_zinc(batch):
    """图批组装：节点特征纵向拼接、邻接块对角拼接、batch_idx 标记节点所属图。"""
    xs, adjs, ys = zip(*batch)
    x = torch.cat(xs, dim=0)
    adj = torch.block_diag(*adjs)
    counts = torch.tensor([a.size(0) for a in adjs], dtype=torch.long)
    batch_idx = torch.repeat_interleave(counts)  # 每图至少 1 原子，覆盖 0..B-1
    y = torch.stack(ys)
    return x, adj, batch_idx, y


def get_zinc_dataloaders(batch_size: int, data_dir: str = "./data"):
    """返回 (train_loader, val_loader, test_loader, y_scaler, input_dim)。

    y_scaler 为仅训练集拟合的 MinMaxScaler（评估时逆归一化 MAE 用）；
    input_dim 为节点特征维度（28）。
    """
    cache_dir = download_zinc(data_dir)
    train_graphs, val_graphs, test_graphs = _load_processed(cache_dir)

    y_scaler = MinMaxScaler().fit(
        np.array([g[2] for g in train_graphs], dtype=np.float32).reshape(-1, 1)
    )

    def scale_split(graphs):
        ys = y_scaler.transform(
            np.array([g[2] for g in graphs], dtype=np.float32).reshape(-1, 1)
        ).ravel()
        return [(x, adj, y) for (x, adj, _), y in zip(graphs, ys)]

    train_ds = ZINCDataset(scale_split(train_graphs))
    val_ds = ZINCDataset(scale_split(val_graphs))
    test_ds = ZINCDataset(scale_split(test_graphs))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_zinc)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_zinc)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_zinc)
    logger.info(
        "ZINC-12k 加载完成：%d/%d/%d 图（batch %d，官方划分），"
        "目标 MinMax 归一化到 [0,1]，节点特征 %d 维",
        len(train_ds), len(val_ds), len(test_ds), batch_size, NUM_ATOM_TYPE,
    )
    return train_loader, val_loader, test_loader, y_scaler, NUM_ATOM_TYPE
