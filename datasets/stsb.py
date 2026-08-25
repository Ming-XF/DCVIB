"""STS-B 文本语义相似度回归数据集：下载、解析与划分。

数据源为 HuggingFace GLUE 的 stsb 配置（nyu-mll/glue，parquet 分片），
首次使用时下载到 data_dir/stsb/，之后离线可用（下载需代理）。

预处理：两个句子分别去 HTML 标签后按 basic_english 风格分词（lower +
纯字母），各自截断后以 [SEP]（idx=2）拼接成单条序列送入 LSTM；官方
train/dev（共 7249 条已标注句子对，test 分片无标签、GLUE 评测服务器
专用，不使用）合并后 60/20/20 划分（seed 固定，回归不分层，同 IMDb
约定）；词表仅从训练集构建（pad_idx=0、unk_idx=1、sep_idx=2 为保留位，
其余按词频编号）；目标用 MinMaxScaler 归一化到 [0,1]（仅训练集拟合），
评估时用返回的 y_scaler 逆归一化回原始相似度分数（0~5）。词向量用
GloVe 100d 预训练矩阵（load_glove_matrix，嵌入层冻结）。collate 用
pad_sequence 后填充。
"""
import logging
import os
import shutil
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .imdb import _tokenize

logger = logging.getLogger(__name__)

PAD_IDX, UNK_IDX, SEP_IDX = 0, 1, 2
_DATA_FILES = {
    # HuggingFace datasets-server 提供的 nyu-mll/glue stsb 配置 parquet 分片
    "stsb_train.parquet": (
        "https://huggingface.co/datasets/nyu-mll/glue/resolve/"
        "refs%2Fconvert%2Fparquet/stsb/train/0000.parquet"
    ),
    "stsb_dev.parquet": (
        "https://huggingface.co/datasets/nyu-mll/glue/resolve/"
        "refs%2Fconvert%2Fparquet/stsb/validation/0000.parquet"
    ),
}
GLOVE_ZIP_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_ZIP = "glove.6B.zip"
GLOVE_TXT = "glove.6B.100d.txt"
GLOVE_DIM = 100


def download_stsb(data_dir: str = "./data") -> str:
    """确保 STS-B 两个 parquet 分片在 data_dir/stsb/，返回缓存目录。"""
    cache_dir = os.path.join(data_dir, "stsb")
    os.makedirs(cache_dir, exist_ok=True)
    for fname, url in _DATA_FILES.items():
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        logger.info("下载 STS-B 数据到 %s", path)
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise RuntimeError(
                f"无法下载 STS-B 数据集（{url}）：{exc}，"
                f"请先执行 source /etc/network_turbo 开启代理，"
                f"或手动放置 {fname} 到 {cache_dir}/ 后重试"
            )
    return cache_dir


def load_glove_matrix(data_dir: str, vocab: dict, dim: int = GLOVE_DIM) -> torch.Tensor:
    """构造与 vocab 对齐的 GloVe 预训练词向量矩阵（stsb 专用）。

    首次使用时从 HF 下载 glove.6B.zip（约 820MB）并解出 glove.6B.100d.txt
    缓存到 data_dir/stsb/；矩阵行 0..2（pad/unk/sep 保留位）为全零，词表内
    词取 GloVe 向量，未登录词用 N(0, 0.1) 随机初始化。模型侧加载该矩阵后
    会冻结嵌入层（小数据集上微调词向量会迅速过拟合，STS-B 实验验证）。
    """
    cache_dir = download_stsb(data_dir)
    txt_path = os.path.join(cache_dir, GLOVE_TXT)
    if not os.path.exists(txt_path):
        zip_path = os.path.join(cache_dir, GLOVE_ZIP)
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            logger.info("下载 GloVe 预训练词向量到 %s", zip_path)
            try:
                urllib.request.urlretrieve(GLOVE_ZIP_URL, zip_path)
            except Exception as exc:
                raise RuntimeError(
                    f"无法下载 GloVe（{GLOVE_ZIP_URL}）：{exc}，"
                    f"请先执行 source /etc/network_turbo 开启代理，"
                    f"或手动放置 {GLOVE_ZIP} 到 {cache_dir}/ 后重试"
                )
        logger.info("解出 %s", txt_path)
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(GLOVE_TXT) as src, open(txt_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    glove = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)

    vocab_size = len(vocab) + 3
    matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    hit = 0
    for word, idx in vocab.items():
        vec = glove.get(word)
        if vec is not None:
            matrix[idx] = vec
            hit += 1
    oov_rows = [i for i in range(3, vocab_size) if not matrix[i].any()]
    matrix[oov_rows] = np.random.default_rng(0).normal(0, 0.1, (len(oov_rows), dim))
    logging.info(
        "GloVe %dd 词向量：%d/%d 词命中（%.1f%%），未登录词随机初始化",
        dim, hit, len(vocab), 100 * hit / len(vocab),
    )
    return torch.from_numpy(matrix)


class STSBDataset(Dataset):
    """STS-B 样本容器：每项为 (token_ids 张量, 归一化相似度分数)。"""

    def __init__(self, token_ids, labels):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.token_ids[idx], dtype=torch.long), self.labels[idx]


def collate_stsb(batch):
    """批量组装：pad_sequence 后填充（pad_idx=0），返回 (tokens [B,L], labels [B] 浮点)。"""
    tokens = pad_sequence(
        [item[0] for item in batch], batch_first=True, padding_value=PAD_IDX
    )
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return tokens, labels


def get_stsb_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    max_len: int = 250,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """加载 STS-B 并返回 (train_loader, val_loader, test_loader, vocab_size,
    y_scaler, glove_matrix)。

    官方 train/dev（共 7249 条句子对，相似度 0~5）合并后 60/20/20 划分
    （seed 固定，回归不分层，同 IMDb 约定）；两个句子各自截断到
    (max_len-1)//2 后以 [SEP] 拼接，保证两个句子都对序列有贡献；词表仅
    从训练集构建（pad_idx=0、unk_idx=1、sep_idx=2 为保留位，其余按词频
    编号 3..V+2）。目标用 MinMaxScaler 归一化到 [0,1]（仅训练集拟合），
    返回的 y_scaler 供评估时把 MAE 逆归一化回原始 0~5 分。glove_matrix
    为 GloVe 100d 预训练词向量矩阵（vocab_size × 100），供模型初始化并
    冻结嵌入层。
    """
    cache_dir = download_stsb(data_dir)
    frames = [
        pd.read_parquet(os.path.join(cache_dir, fname))
        for fname in _DATA_FILES
    ]
    df = pd.concat(frames, ignore_index=True)

    cap = (max_len - 1) // 2  # 每句上限，留 1 位给 [SEP]
    tokens_list = [
        _tokenize(s1)[:cap] + [SEP_IDX] + _tokenize(s2)[:cap]
        for s1, s2 in zip(df["sentence1"], df["sentence2"])
    ]
    labels = df["label"].to_numpy(dtype="float32")

    X_train, X_test, y_train, y_test = train_test_split(
        tokens_list, labels, test_size=test_ratio, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio / (1 - test_ratio), random_state=seed
    )
    logging.info(
        "STS-B 划分（60/20/20）：train %d / val %d / test %d（共 %d 条句子对）",
        len(X_train), len(X_val), len(X_test), len(tokens_list),
    )

    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))

    counter = Counter(tok for tokens in X_train for tok in tokens if tok != SEP_IDX)
    vocab = {word: i + 3 for i, (word, _) in enumerate(counter.most_common())}
    logging.info(
        "STS-B 词表：%d 个词（含 pad/unk/sep 保留位共 %d，仅由训练集构建）",
        len(vocab), len(vocab) + 3,
    )
    vocab_size = len(vocab) + 3  # pad_idx=0、unk_idx=1、sep_idx=2 为保留位
    glove_matrix = load_glove_matrix(data_dir, vocab)

    def make_dataset(token_lists, ys):
        ids = [
            [vocab.get(tok, UNK_IDX) if tok != SEP_IDX else SEP_IDX for tok in tokens]
            for tokens in token_lists
        ]
        scaled = y_scaler.transform(ys.reshape(-1, 1)).ravel()
        return STSBDataset(ids, scaled)

    train_dataset = make_dataset(X_train, y_train)
    val_dataset = make_dataset(X_val, y_val)
    test_dataset = make_dataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_stsb
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_stsb
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_stsb
    )

    return train_loader, val_loader, test_loader, vocab_size, y_scaler, glove_matrix
