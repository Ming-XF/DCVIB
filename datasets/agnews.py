"""AG News BERT 逐 token 特征数据集加载与划分。

特征与标签由 datasets/extract_agnews_features.py 生成（冻结 bert-base-uncased
提取每 token 最后一层隐状态，fp16、全零向量 padding），缓存在
data_dir/agnews/agnews_bert_base_{features,labels}.npy（不划分）。

加载时按 60/20/20 分层划分（seed 固定）。特征在加载时一次性读入 RAM 并
转 fp32（12GB fp16 → 25.5GB fp32，本机 RAM 754GB 充裕），逐样本零拷贝
返回——相比 mmap 逐样本读盘，数据加载从 ~13s/epoch 降到 ~1s。collate 用
pad_sequence 对 (L, 768) 序列后填充全零向量，长度由模型内部零向量检测
计算。
"""
import logging
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

PCA_DIM = 50  # reduce_agnews_features.py 的默认降维维度，加载端自动检测对应文件


class AGNewsDataset(Dataset):
    """AG News BERT 特征容器：每项为 (特征序列 (L, 768) fp32, label)。"""

    def __init__(self, indices, features, labels):
        self.indices = indices
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        feat = torch.from_numpy(self.features[i])
        return feat, int(self.labels[i])


def collate_agnews(batch):
    """批量组装：pad_sequence 对 (L, 768) 序列后填充全零向量。

    返回 (features [B, Lmax, 768], labels [B])；长度由模型内部按
    零向量检测计算。
    """
    feats = pad_sequence(
        [item[0] for item in batch], batch_first=True, padding_value=0.0
    )
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return feats, labels


def get_agnews_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    random_labels: bool = False,
    layer: int = 10,
):
    """加载 AG News BERT 特征并返回训练/验证/测试 DataLoader 及 input_dim。

    官方 train/test 合并后 60/20/20 分层划分（seed 固定），特征一次性读入
    RAM 并转 fp32（原始 768 维约 25.5GB，本机 RAM 充裕）。layer 指定 BERT
    层号（默认 10 = 倒数第三层）。特征文件自适应选择：若存在 PCA 降维文件
    agnews_bert_base_layer{layer}_pca{PCA_DIM}_features.npy（由
    reduce_agnews_features.py 生成，默认 PCA_DIM=100）则使用之，否则回退
    原始 768 维特征；返回的第 4 个值为模型输入维度 input_dim（与所用文件
    一致，无需外部指定）。random_labels=True 时把训练集标签按固定 seed
    随机化（信息自由数据集 I(X;Y)=0 的记忆实验），验证/测试集保持真实标签。
    """
    base = os.path.join(data_dir, "agnews")
    pca_path = os.path.join(base, f"agnews_bert_base_layer{layer}_pca{PCA_DIM}_features.npy")
    if os.path.exists(pca_path):
        feat_name = os.path.basename(pca_path)
        input_dim = PCA_DIM
    else:
        feat_name = f"agnews_bert_base_layer{layer}_features.npy"
        input_dim = 768
    logging.info("AG News 特征文件：%s（input_dim=%d）", feat_name, input_dim)
    features = np.load(os.path.join(base, feat_name))
    features = features.astype(np.float32)
    labels = np.load(os.path.join(base, "agnews_bert_base_labels.npy"))
    n = len(labels)

    indices = np.arange(n)
    train_idx, test_idx, y_train, _ = train_test_split(
        indices, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_idx, y_train, test_size=val_ratio / (1 - test_ratio),
        random_state=seed, stratify=y_train,
    )
    logging.info(
        "AG News BERT 特征划分（60/20/20 分层）：train %d / val %d / test %d（共 %d 条）",
        len(train_idx), len(val_idx), len(test_idx), n,
    )

    train_labels = labels.copy()
    if random_labels:
        rng = torch.Generator().manual_seed(42)
        train_labels[train_idx] = torch.randint(
            0, 4, (len(train_idx),), generator=rng
        ).numpy()
        logging.info("随机标签实验：训练集标签已随机化（I(X;Y)=0，固定 seed 42）")

    train_dataset = AGNewsDataset(train_idx, features, train_labels)
    val_dataset = AGNewsDataset(val_idx, features, labels)
    test_dataset = AGNewsDataset(test_idx, features, labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_agnews
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_agnews
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_agnews
    )

    return train_loader, val_loader, test_loader, input_dim
