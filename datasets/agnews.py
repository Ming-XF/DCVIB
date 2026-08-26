"""AG News BERT 逐 token 特征数据集加载与划分。

特征与标签由 datasets/extract_agnews_features.py 生成（冻结 bert-base-uncased
提取每 token 最后一层隐状态，fp16、全零向量 padding），缓存在
data_dir/agnews/agnews_bert_base_{features,labels}.npy（不划分）。

加载时按 60/20/20 分层划分（seed 固定）。特征首次加载时转 fp32 并落盘
*_fp32.npy 预处理缓存（flock 保证多进程只有一个构建者，meta 记录源文件
mtime、特征重新提取时自动重建），之后各进程 mmap 只读映射共享页缓存、
逐样本零拷贝——多进程调参时物理内存只占一份（否则原始 768 维特征每进程
约 25.5GB）。collate 用 pad_sequence 对 (L, input_dim) 序列后填充全零
向量，长度由模型内部零向量检测计算。
"""
import fcntl
import json
import logging
import os
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# mmap 只读数组转 tensor 会触发"非可写"告警：DataLoader 只读使用缓存、从不回写，忽略该噪音
warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")

logger = logging.getLogger(__name__)

PCA_DIM = 50  # reduce_agnews_features.py 的默认降维维度，加载端自动检测对应文件


class AGNewsDataset(Dataset):
    """AG News BERT 特征容器：每项为 (特征序列 (L, input_dim) fp32, label)。

    features 为 fp32 预处理缓存上的 mmap torch 张量（跨进程共享页缓存），
    索引返回单样本视图，collate 时由 pad_sequence 组装新张量。
    """

    def __init__(self, indices, features, labels):
        self.indices = indices
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        feat = self.features[i]
        return feat, int(self.labels[i])


def _build_agnews_cache(src_path, cache_path, meta_path, input_dim):
    """把 fp16 特征源文件转 fp32 落盘缓存，meta.json 最后写入标记完成。

    仅在缓存缺失或源文件更新时执行（调用方持 flock 保证多进程只有一个
    构建者）；缓存先写 .tmp.npy 再原子改名，构建中途被杀不留半成品。
    """
    logging.info("AG News 预处理缓存缺失，首次构建（fp16 → fp32 转换，内存峰值较高）...")
    features = np.load(src_path).astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp.npy"  # np.save 对不以 .npy 结尾的路径会自动补后缀
    np.save(tmp, features)
    os.replace(tmp, cache_path)
    with open(meta_path, "w") as f:
        json.dump({"src_mtime": os.path.getmtime(src_path), "input_dim": input_dim}, f)
    logging.info("AG News 预处理缓存已写入 %s（fp32）", cache_path)


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

    官方 train/test 合并后 60/20/20 分层划分（seed 固定）。特征首次加载时
    转 fp32 落盘 *_fp32.npy 缓存（flock 串行化构建、源文件 mtime 变化自动
    重建），之后各进程 mmap 只读映射共享页缓存——多进程调参时物理内存只占
    一份。layer 指定 BERT 层号（默认 10 = 倒数第三层）。特征文件自适应
    选择：若存在 PCA 降维文件 agnews_bert_base_layer{layer}_pca{PCA_DIM}
    _features.npy（由 reduce_agnews_features.py 生成，默认 PCA_DIM=100）
    则使用之，否则回退原始 768 维特征；返回的第 4 个值为模型输入维度
    input_dim（与所用文件一致，无需外部指定）。random_labels=True 时把
    训练集标签按固定 seed 随机化（信息自由数据集 I(X;Y)=0 的记忆实验），
    验证/测试集保持真实标签。
    """
    base = os.path.join(data_dir, "agnews")
    pca_path = os.path.join(base, f"agnews_bert_base_layer{layer}_pca{PCA_DIM}_features.npy")
    if os.path.exists(pca_path):
        src_name = os.path.basename(pca_path)
        input_dim = PCA_DIM
    else:
        src_name = f"agnews_bert_base_layer{layer}_features.npy"
        input_dim = 768
    logging.info("AG News 特征文件：%s（input_dim=%d）", src_name, input_dim)

    src_path = os.path.join(base, src_name)
    cache_path = os.path.join(base, src_name.replace(".npy", "_fp32.npy"))
    meta_path = cache_path + ".meta.json"
    lock_fd = os.open(os.path.join(base, "preprocess.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        cache_ready = os.path.exists(cache_path) and os.path.exists(meta_path)
        if cache_ready:
            with open(meta_path) as f:
                cache_ready = json.load(f).get("src_mtime") == os.path.getmtime(src_path)
        if not cache_ready:
            _build_agnews_cache(src_path, cache_path, meta_path, input_dim)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    logging.info("AG News 预处理缓存：mmap 共享 %s", cache_path)
    features = torch.from_numpy(np.load(cache_path, mmap_mode="r"))
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
