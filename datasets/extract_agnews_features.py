"""使用预训练 BERT 提取 AG News 逐 token 特征。

流程：
1. 经 HuggingFace 下载 AG News（官方 train 120k / test 7.6k，4 类新闻主题，
   需先 source /etc/network_turbo 开启代理）
2. 冻结 bert-base-uncased，提取每 token 指定层隐状态 (N, L, 768)
   （--layer 默认 10 = 倒数第三层；hidden_states 索引 0 为嵌入输出，
   1..12 为第 1..12 层输出，12 为最后一层）
3. 序列截断到 --max-len，不足部分用全零向量 padding（加载端按首个全零向量识别长度）
4. fp16 保存 agnews_bert_base_layer{layer}_features.npy / labels 到 data/agnews/

用法：
    source /etc/network_turbo
    python datasets/extract_agnews_features.py --smoke-samples 2000   # 冒烟：前 2000 条
    python datasets/extract_agnews_features.py                        # 全量提取（倒数第三层）
"""
import argparse
import logging
import os
import sys

# 本地 datasets/ 包与 HF 的 datasets 库同名：以脚本方式运行时 sys.path[0] 是
# 脚本所在目录，会让 import datasets 解析到本地 datasets/datasets.py。
# 本脚本不使用本地包，先把脚本目录移出 sys.path，确保导入 HF 的库。
if __package__ is None and sys.path and os.path.basename(sys.path[0]) == "datasets":
    sys.path.pop(0)

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

MODEL_NAME = "bert-base-uncased"
HIDDEN_DIM = 768


def parse_args():
    parser = argparse.ArgumentParser(description="AG News BERT 逐 token 特征提取")
    parser.add_argument("--max-len", type=int, default=64,
                        help="序列截断长度（token 数）")
    parser.add_argument("--layer", type=int, default=10,
                        help="取 BERT 哪一层的隐状态（1..12，默认 10 = 倒数第三层）")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke-samples", type=int, default=0,
                        help=">0 时只提取前 N 条，用于冒烟测试")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    logging.info("下载 AG News 数据集（需代理）...")
    ds = load_dataset("ag_news")
    texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    labels = list(ds["train"]["label"]) + list(ds["test"]["label"])
    logging.info("共 %d 条（train %d / test %d）", len(texts), len(ds["train"]), len(ds["test"]))
    if args.smoke_samples > 0:
        texts, labels = texts[:args.smoke_samples], labels[:args.smoke_samples]
        logging.info("冒烟模式：仅前 %d 条", len(texts))

    logging.info("加载 %s（冻结，eval）...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 先统计未截断长度分布（仅 tokenizer，代价小），供选择 max_len 参考
    tok_lens = []
    for i in tqdm(range(0, len(texts), 1000), desc="统计长度"):
        chunk = tokenizer(texts[i:i + 1000], truncation=False)["input_ids"]
        tok_lens.extend(len(t) for t in chunk)
    tok_lens = np.array(tok_lens)
    logging.info(
        "长度分布（未截断）：mean %d / 50%% %d / 75%% %d / 90%% %d / 95%% %d / max %d",
        int(tok_lens.mean()), *[int(np.percentile(tok_lens, p)) for p in (50, 75, 90, 95)],
        tok_lens.max(),
    )
    logging.info(
        "max_len=%d 完整保留 %.1f%% 文档，保留 %.1f%% 总 token",
        args.max_len, (tok_lens <= args.max_len).mean() * 100,
        np.minimum(tok_lens, args.max_len).sum() / tok_lens.sum() * 100,
    )

    out_dir = os.path.join(args.data_dir, "agnews")
    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(
        out_dir, f"agnews_bert_base_layer{args.layer}_features.npy"
    )
    labels_path = os.path.join(out_dir, "agnews_bert_base_labels.npy")

    features = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float16,
        shape=(len(texts), args.max_len, HIDDEN_DIM),
    )
    labels_arr = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64, shape=(len(texts),),
    )
    labels_arr[:] = labels

    logging.info("提取第 %d 层（%s）逐 token 特征 -> %s（fp16，约 %.1fGB）",
                 args.layer,
                 "倒数第三层" if args.layer == 10 else f"第 {args.layer} 层",
                 features_path, features.nbytes / 1e9)
    start = 0
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="提取特征"):
            batch_texts = texts[i:i + args.batch_size]
            enc = tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=args.max_len, return_tensors="pt",
            ).to(device)
            out = model(**enc, output_hidden_states=True).hidden_states[args.layer]
            # 把 padding 位置的输出写成精确全零向量（加载端靠它识别真实长度）
            out = out * enc["attention_mask"].unsqueeze(-1)
            b = out.size(0)
            features[start:start + b] = out.cpu().numpy().astype(np.float16)
            start += b

    features.flush()
    logging.info("已保存 %s / %s", features_path, labels_path)

    # 读回验证
    data = np.load(features_path, mmap_mode="r")
    logging.info("读回验证：features %s %s，labels %s", data.shape, data.dtype,
                 np.load(labels_path).shape)
    row = data[0]
    nonzero = (np.abs(row).sum(axis=1) > 0).sum()
    logging.info("样例 0 有效长度（零向量检测）：%d / %d", nonzero, row.shape[0])


if __name__ == "__main__":
    main()
