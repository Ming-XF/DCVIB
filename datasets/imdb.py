"""IMDb 情感分析数据集：下载、解析与 60/20/20 分层划分。

数据源为斯坦福官方 aclImdb_v1.tar.gz（官方 train/test 各 25k 已标注影评，
另有 train/unsup 5 万无标签文档被忽略），首次使用时下载解压到
data_dir/imdb/，之后离线可用。

预处理：文本去 HTML 标签后按 basic_english 风格分词（lower + 纯字母），
官方 train/test 合并后 60/20/20 分层划分（seed 固定），词表仅从训练集
构建（pad_idx=0、unk_idx=1 为保留位，其余按词频编号），序列统一截断到
max_len，collate 用 pad_sequence 后填充（长度由模型内部按 pad 掩码计算）。
"""
import logging
import os
import re
import tarfile
import urllib.request
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

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
    """IMDb 样本容器：每项为 (token_ids 张量, label)。"""

    def __init__(self, token_ids, labels):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.token_ids[idx], dtype=torch.long), self.labels[idx]


def collate_imdb(batch):
    """批量组装：pad_sequence 后填充（pad_idx=0），返回 (tokens [B,L], labels [B])。

    序列长度不随 batch 返回，由模型内部按 (x != pad_idx).sum(1) 计算。
    """
    tokens = pad_sequence(
        [item[0] for item in batch], batch_first=True, padding_value=PAD_IDX
    )
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return tokens, labels


def get_imdb_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    max_len: int = 500,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """加载 IMDb 并返回 (train_loader, val_loader, test_loader, vocab_size)。

    官方 train/test 共 50k 已标注影评合并后 60/20/20 分层划分（seed 固定），
    词表仅从训练集构建（pad_idx=0、unk_idx=1，其余按词频编号 2..V+1），
    序列截断到 max_len（先截断后建词表，词表只含模型实际可见的词）。
    返回的 vocab_size 含 pad/unk 两个保留位，供模型构造 Embedding。
    """
    extract_dir = download_imdb(data_dir)
    texts, labels = read_imdb(extract_dir)
    tokens_list = [_tokenize(text)[:max_len] for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        tokens_list, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio / (1 - test_ratio),
        random_state=seed, stratify=y_train,
    )
    logging.info(
        "IMDb 划分（60/20/20 分层）：train %d / val %d / test %d（共 %d 条影评）",
        len(X_train), len(X_val), len(X_test), len(tokens_list),
    )

    counter = Counter(tok for tokens in X_train for tok in tokens)
    vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common())}
    logging.info(
        "IMDb 词表：%d 个词（含 pad/unk 保留位共 %d，仅由训练集构建）",
        len(vocab), len(vocab) + 2,
    )
    vocab_size = len(vocab) + 2  # pad_idx=0、unk_idx=1 为保留位

    def make_dataset(token_lists, ys):
        ids = [
            [vocab.get(tok, UNK_IDX) for tok in tokens] or [UNK_IDX]
            for tokens in token_lists
        ]
        return IMDbDataset(ids, ys)

    train_dataset = make_dataset(X_train, y_train)
    val_dataset = make_dataset(X_val, y_val)
    test_dataset = make_dataset(X_test, y_test)

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
