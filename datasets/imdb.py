"""IMDb 情感分析数据集：下载、解析与 60/20/20 分层划分。

数据源为斯坦福官方 aclImdb_v1.tar.gz（官方 train/test 各 25k 已标注影评，
另有 train/unsup 5 万无标签文档被忽略），首次使用时下载解压到
data_dir/imdb/，之后离线可用。

预处理：文本去 HTML 标签后按 basic_english 风格分词（lower + 纯字母），
官方 train/test 合并后 60/20/20 分层划分（seed 固定），词表仅从训练集
构建（pad_idx=0、unk_idx=1 为保留位，其余按词频编号），序列统一截断到
max_len，collate 用 pad_sequence 后填充（长度由模型内部按 pad 掩码计算）。

分词 + 划分 + 词表 + id 转换首次加载时一次性落盘缓存
（imdb_processed_maxlen{max_len}/：定长 token_id 矩阵、labels、划分索引、
vocab.json，meta.json 记录 tarball mtime 与划分参数、变化自动重建；
flock 保证多进程只有一个构建者），之后各进程 mmap 共享 token_id 矩阵，
跳过 5 万影评文件的逐进程解析。
"""
import fcntl
import json
import logging
import os
import re
import tarfile
import urllib.request
import warnings
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# mmap 只读数组转 tensor 会触发"非可写"告警：DataLoader 只读使用缓存、从不回写，忽略该噪音
warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*")

logger = logging.getLogger(__name__)

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
PAD_IDX, UNK_IDX = 0, 1
_HTML_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    """去 HTML 标签（如 <br />）、小写化后按纯字母分词（basic_english 风格）。"""
    return _TOKEN_RE.findall(_HTML_RE.sub(" ", text).lower())


def download_imdb(data_dir: str = "./data") -> str:
    """确保 aclImdb 已解压到 data_dir/imdb/，返回解压目录。"""
    cache_dir = os.path.join(data_dir, "imdb")
    extract_dir = os.path.join(cache_dir, "aclImdb")
    if os.path.isdir(extract_dir):
        return extract_dir
    os.makedirs(cache_dir, exist_ok=True)

    tarball = os.path.join(cache_dir, "aclImdb_v1.tar.gz")
    if not os.path.exists(tarball) or os.path.getsize(tarball) == 0:
        logger.info("下载 IMDb 数据集到 %s", tarball)
        try:
            urllib.request.urlretrieve(DATA_URL, tarball)
        except Exception as exc:
            raise RuntimeError(
                f"无法下载 IMDb 数据集（{DATA_URL}）：{exc}，"
                f"请手动放置 aclImdb_v1.tar.gz 到 {cache_dir}/ 后重试"
            )
    logger.info("解压 %s 到 %s", tarball, cache_dir)
    with tarfile.open(tarball) as tar:
        tar.extractall(cache_dir, filter="data")
    return extract_dir


def read_imdb(extract_dir: str):
    """读取 aclImdb/{train,test}/{pos,neg}/ 四个目录，返回 (texts, labels)。

    显式遍历四个已标注目录（忽略 train/unsup/ 的 5 万无标签文档）；
    标签 pos=1、neg=0。
    """
    texts, labels = [], []
    for split in ("train", "test"):
        for label, subdir in ((1, "pos"), (0, "neg")):
            folder = os.path.join(extract_dir, split, subdir)
            for fname in sorted(os.listdir(folder)):
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(label)
    return texts, labels


class IMDbDataset(Dataset):
    """IMDb 样本容器：每项为 (token_ids 张量 [L], label)。

    token_ids 为预处理缓存上的 mmap torch 张量（跨进程共享页缓存），
    indices 指定样本子集（train/val/test），索引返回单样本视图。
    """

    def __init__(self, indices, token_ids, labels):
        self.indices = indices
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.token_ids[i].long(), int(self.labels[i])


def collate_imdb(batch):
    """批量组装：pad_sequence 后填充（pad_idx=0），返回 (tokens [B,L], labels [B])。

    缓存为定长 max_len 矩阵，pad 后裁掉全 batch 共有的尾部零（与原变长
    pad_sequence 行为一致，避免 500 长序列拖慢 LSTM）；序列长度由模型
    内部按 (x != pad_idx).sum(1) 计算。
    """
    tokens = pad_sequence(
        [item[0] for item in batch], batch_first=True, padding_value=PAD_IDX
    )
    tokens = tokens[:, : (tokens != PAD_IDX).sum(1).max()]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return tokens, labels


def _build_imdb_cache(extract_dir, cache_dir, max_len, seed, test_ratio, val_ratio, tarball_mtime):
    """读取全部影评并一次性完成 tokenize + 划分 + 词表 + id 转换，落盘缓存。

    仅在缓存缺失或数据更新时执行（调用方持 flock 保证多进程只有一个构建者）；
    之后各进程 mmap 共享定长 token_id 矩阵，跳过 5 万文件的逐进程解析。
    meta.json 最后写入作为完成标记，各 .npy 先写 .tmp 再原子改名。
    """
    logging.info("IMDb 预处理缓存缺失，首次构建（读取影评 + 分词 + 词表 + id 转换）...")
    texts, labels = read_imdb(extract_dir)
    tokens_list = [_tokenize(text)[:max_len] for text in texts]

    idx_all = np.arange(len(labels))
    train_idx, test_idx, _, _ = train_test_split(
        idx_all, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )
    y_train_split = np.asarray(labels)[train_idx]
    train_idx, val_idx, _, _ = train_test_split(
        train_idx, y_train_split, test_size=val_ratio / (1 - test_ratio),
        random_state=seed, stratify=y_train_split,
    )
    logging.info(
        "IMDb 划分（60/20/20 分层）：train %d / val %d / test %d（共 %d 条影评）",
        len(train_idx), len(val_idx), len(test_idx), len(labels),
    )

    train_tokens = [tokens_list[i] for i in train_idx]
    counter = Counter(tok for tokens in train_tokens for tok in tokens)
    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common())}
    logging.info(
        "IMDb 词表：%d 个词（含 pad/unk 保留位共 %d，仅由训练集构建）",
        len(vocab), len(vocab) + 2,
    )

    mat = np.zeros((len(tokens_list), max_len), dtype=np.int32)
    for i, tokens in enumerate(tokens_list):
        ids = [vocab.get(tok, UNK_IDX) for tok in tokens] or [UNK_IDX]
        mat[i, : len(ids)] = ids

    os.makedirs(cache_dir, exist_ok=True)
    for name, arr in (
        ("token_ids.npy", mat),
        ("labels.npy", np.asarray(labels, dtype=np.int32)),
        ("train_idx.npy", train_idx),
        ("val_idx.npy", val_idx),
        ("test_idx.npy", test_idx),
    ):
        path = os.path.join(cache_dir, name)
        tmp = path + ".tmp.npy"  # np.save 对不以 .npy 结尾的路径会自动补后缀
        np.save(tmp, arr)
        os.replace(tmp, path)
    with open(os.path.join(cache_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "tarball_mtime": tarball_mtime,
                "max_len": max_len,
                "seed": seed,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
            },
            f,
        )
    logging.info("IMDb 预处理缓存已写入 %s", cache_dir)


def get_imdb_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    max_len: int = 250,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    random_labels: bool = False,
):
    """加载 IMDb 并返回 (train_loader, val_loader, test_loader, vocab_size)。

    官方 train/test 共 50k 已标注影评合并后 60/20/20 分层划分（seed 固定），
    词表仅从训练集构建（pad_idx=0、unk_idx=1，其余按词频编号 2..V+1），
    序列截断到 max_len（先截断后建词表，词表只含模型实际可见的词）。
    返回的 vocab_size 含 pad/unk 两个保留位，供模型构造 Embedding。
    分词 + 划分 + 词表 + id 转换结果首次加载时落盘缓存（flock 串行化、
    tarball mtime 与划分参数变化自动重建），之后各进程 mmap 共享。
    random_labels=True 时把训练集标签按固定 seed 随机化（信息自由数据集
    I(X;Y)=0 的记忆实验），验证/测试集保持真实标签。
    """
    extract_dir = download_imdb(data_dir)
    base = os.path.join(data_dir, "imdb")
    tarball = os.path.join(base, "aclImdb_v1.tar.gz")
    cache_dir = os.path.join(base, f"imdb_processed_maxlen{max_len}")
    meta_path = os.path.join(cache_dir, "meta.json")

    lock_fd = os.open(os.path.join(base, "preprocess.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        cache_ready = os.path.exists(meta_path)
        if cache_ready:
            with open(meta_path) as f:
                m = json.load(f)
            cache_ready = (
                m.get("tarball_mtime") == os.path.getmtime(tarball)
                and m.get("seed") == seed
                and m.get("val_ratio") == val_ratio
                and m.get("test_ratio") == test_ratio
            )
        if not cache_ready:
            _build_imdb_cache(
                extract_dir, cache_dir, max_len, seed, test_ratio, val_ratio,
                os.path.getmtime(tarball),
            )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    logging.info("IMDb 预处理缓存：mmap 共享 %s", cache_dir)
    token_ids = torch.from_numpy(np.load(os.path.join(cache_dir, "token_ids.npy"), mmap_mode="r"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"))
    train_idx = np.load(os.path.join(cache_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(cache_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(cache_dir, "test_idx.npy"))
    with open(os.path.join(cache_dir, "vocab.json")) as f:
        vocab = json.load(f)
    vocab_size = len(vocab) + 2  # pad_idx=0、unk_idx=1 为保留位

    train_labels = labels.copy()
    if random_labels:
        rng = torch.Generator().manual_seed(42)
        train_labels[train_idx] = torch.randint(
            0, 2, (len(train_idx),), generator=rng
        ).numpy()
        logging.info("随机标签实验：训练集标签已随机化（I(X;Y)=0，固定 seed 42）")

    train_dataset = IMDbDataset(train_idx, token_ids, train_labels)
    val_dataset = IMDbDataset(val_idx, token_ids, labels)
    test_dataset = IMDbDataset(test_idx, token_ids, labels)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_imdb
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_imdb
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_imdb
    )

    return train_loader, val_loader, test_loader, vocab_size
