"""AG News BERT 逐 token 特征的 PCA 降维脚本（基于已提取特征，无需重跑 BERT）。

把 data/agnews/agnews_bert_base_layer{layer}_features.npy（127.6k × max_len × 768
fp16）降维为 data/agnews/agnews_bert_base_layer{layer}_pca{dim}_features.npy
（127.6k × max_len × dim fp16），用于减小 RAM 占用（768→256 时 fp32 预载从
25.5GB 降到 8.4GB，30 个并行试验约 250GB）与每 epoch 的 GPU 搬运量（15GB
降到 5GB），并降低 LSTM 首层计算量。

PCA 仅在训练划分的 token 子样本上拟合（与 datasets/agnews.py 相同的
60/20/20 分层划分 seed 42，避免测试信息进入无监督变换），全量变换在 GPU
上分块完成。降维只做线性投影 + 去均值，token 序列结构不变（RNN 骨干照常
使用，input_dim 改为 dim）。

用法：
    python datasets/reduce_agnews_features.py --dim 256 --layer 10
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "agnews")
FIT_TOKENS = 300_000  # PCA 拟合用的训练集 token 数（子采样上限）
FIT_SAMPLES = 5000  # 从中采样的训练样本数上限
TRANSFORM_CHUNK = 2048  # GPU 分块变换的样本数


def parse_args():
    parser = argparse.ArgumentParser(description="AG News BERT 特征 PCA 降维")
    parser.add_argument("--dim", type=int, default=50, help="降维后的维度（默认 50，加载端自动检测该文件）")
    parser.add_argument("--layer", type=int, default=10, help="BERT 层号（默认 10 = 倒数第三层）")
    parser.add_argument("--max-len", type=int, default=64, help="输出序列长度（默认 64，与提取时一致）")
    return parser.parse_args()


def main():
    args = parse_args()
    src_path = os.path.join(
        FEAT_DIR, f"agnews_bert_base_layer{args.layer}_features.npy"
    )
    labels_path = os.path.join(FEAT_DIR, "agnews_bert_base_labels.npy")
    dst_path = os.path.join(
        FEAT_DIR, f"agnews_bert_base_layer{args.layer}_pca{args.dim}_features.npy"
    )
    assert os.path.exists(src_path), f"特征文件不存在：{src_path}（先运行 extract_agnews_features.py）"

    labels = np.load(labels_path)
    n = len(labels)

    # 与 datasets/agnews.py 相同的 60/20/20 分层划分（seed 42），仅用训练集 token 拟合 PCA
    indices = np.arange(n)
    train_idx, _, y_train, _ = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_idx, _, _, _ = train_test_split(
        train_idx, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    logger.info("训练划分 %d 样本（仅用其 token 拟合 PCA）", len(train_idx))

    # 1) 拟合 PCA：训练集随机样本的 token 子集
    t0 = time.time()
    src = np.load(src_path, mmap_mode="r")
    rng = np.random.default_rng(42)
    sel_samples = rng.choice(train_idx, size=min(FIT_SAMPLES, len(train_idx)), replace=False)
    tokens = np.asarray(src[sel_samples], dtype=np.float32).reshape(-1, src.shape[2])
    rng.shuffle(tokens)
    tokens = tokens[:FIT_TOKENS]
    logger.info(
        "PCA 拟合样本：%d 个 token（%d×%d），读取用时 %.1fs",
        len(tokens), tokens.shape[0], tokens.shape[1], time.time() - t0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(tokens).to(device)
    mean = x.mean(dim=0, keepdim=True)
    x = x - mean
    _, s, v = torch.linalg.svd(x, full_matrices=False)
    w = v.t()[:, : args.dim]  # (768, dim) 投影矩阵（右奇异向量 = 主成分方向）
    mean = mean.squeeze(0)
    explained = (s[: args.dim] ** 2).sum().item() / (s**2).sum().item()
    logger.info(
        "PCA %d→%d：前 %d 维解释方差 %.1f%%，SVD 用时 %.1fs",
        src.shape[2], args.dim, args.dim, 100 * explained, time.time() - t0,
    )

    # 2) 分块变换并写 fp16 输出（memmap 预分配，边算边写）
    t0 = time.time()
    out = np.lib.format.open_memmap(
        dst_path, mode="w+", dtype=np.float16,
        shape=(n, args.max_len, args.dim),
    )
    for start in range(0, n, TRANSFORM_CHUNK):
        stop = min(start + TRANSFORM_CHUNK, n)
        chunk = np.asarray(src[start:stop], dtype=np.float32)
        proj = (torch.from_numpy(chunk).to(device) - mean) @ w
        out[start:stop] = proj.cpu().numpy().astype(np.float16)
        if (start // TRANSFORM_CHUNK) % 8 == 0:
            logger.info("  已变换 %d/%d 样本", stop, n)
    out.flush()
    logger.info("保存 %s（%.1fGB），变换用时 %.1fs",
                dst_path, os.path.getsize(dst_path) / 1e9, time.time() - t0)


if __name__ == "__main__":
    main()
