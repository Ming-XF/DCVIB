r"""AgeDB 人脸年龄回归数据集：下载、解压与划分。

数据源为 GitHub HarmonLiu05/agedb 的完整镜像（agedb_images.tar.gz，约
171.5MB，16,488 张图、567 人、年龄 0~96），首次使用时下载到
data_dir/agedb/ 并解压为单个 jpg 文件目录，之后离线可用（下载需
source /etc/network_turbo 代理，缺文件自动重下）。

年龄标注嵌在文件名中（AgeDB 官方命名约定：{id}_{name}_{age}_{fm}.jpg，
如 10622_LindaEvans_47_f.jpg），加载时用正则 _(\d+)_([fm])\.jpg$ 从全部
文件名提取并做全量覆盖校验——任一行不匹配直接报错并列出示例，不做静默
过滤，保证与基准数据集规模一致；年龄越界检查用宽松范围 0~120。

预处理：RGB 64×64 + ImageNet 均值/标准差归一化；全部图像合并后 60/20/20
划分（seed 42，回归不分层，同 stsb/imdb/housing 约定）；目标年龄用
MinMaxScaler 归一化到 [0,1]（仅训练集拟合），评估时用返回的 y_scaler
逆归一化回年龄（岁）。

性能：首次加载把全部图像解码为 (N, 64, 64, 3) uint8 数组（约 202MB）并
缓存 npy（_load_image_array，行数与排序文件名顺序一致），之后每次加载
直接读缓存、每 epoch 只做微秒级张量归一化——避免单进程每 epoch 重新
JPEG 解码 ~13k 张图（数据迭代 6.7s/epoch → 0.6s/epoch，GPU 训练本身仅
约 0.6s/epoch）。
"""

import logging
import os
import re
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

AGEDB_URL = "https://media.githubusercontent.com/media/HarmonLiu05/agedb/main/agedb_images.tar.gz"
AGEDB_TARBALL = "agedb_images.tar.gz"
EXTRACT_DIR = "agedb_images"
PROCESSED_NAME = "agedb_64x64_uint8.npy"
IMG_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AGE_PATTERN = re.compile(r"_(\d+)_([fm])\.jpg$")
# 身份 = 文件名第二段（姓名），如 10622_LindaEvans_47_f.jpg → LindaEvans。
# 首段数字是图像编号（全数据集唯一），不是身份 id，不能用它做 subject 划分。
SUBJECT_PATTERN = re.compile(r"^\d+_([^_]+)_\d+_[fm]\.jpg$")
MIN_AGE, MAX_AGE = 0, 120


def download_agedb(data_dir: str = "./data") -> str:
    """确保 tar 包已下载并解压到 data_dir/agedb/，返回缓存目录。"""
    cache_dir = os.path.join(data_dir, "agedb")
    os.makedirs(cache_dir, exist_ok=True)
    tarball = os.path.join(cache_dir, AGEDB_TARBALL)
    if not (os.path.exists(tarball) and os.path.getsize(tarball) > 0):
        logger.info("下载 AgeDB 数据到 %s", tarball)
        try:
            urllib.request.urlretrieve(AGEDB_URL, tarball)
        except Exception as exc:
            raise RuntimeError(
                f"无法下载 AgeDB 数据集（{AGEDB_URL}）：{exc}，"
                f"请先执行 source /etc/network_turbo 开启代理，"
                f"或手动放置 {AGEDB_TARBALL} 到 {cache_dir}/ 后重试"
            )
    extract_dir = os.path.join(cache_dir, EXTRACT_DIR)
    if not os.path.isdir(extract_dir):
        logger.info("解压 AgeDB 图像到 %s", extract_dir)
        # tar 包为平铺结构（无顶层目录），解压到独立子目录避免与 tar 包混杂
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tarball) as tf:
            tf.extractall(extract_dir, filter="data")
    return cache_dir


def _extract_ages(filenames: pd.Series) -> np.ndarray:
    """从文件名提取年龄并做全量覆盖校验，返回 float32 年龄数组。

    AgeDB 官方命名：`{id}_{name}_{age}_{fm}.jpg`，年龄为最后一个 `_数字_`
    段。任一行不匹配直接报错并列出示例（不静默过滤，保证规模与基准一致）。
    """
    matched = filenames.str.extract(AGE_PATTERN)
    bad = matched[0].isna()
    if bad.any():
        examples = filenames[bad].head(5).tolist()
        raise ValueError(
            f"{int(bad.sum())}/{len(filenames)} 个文件名无法提取年龄"
            f"（正则 {AGE_PATTERN.pattern}），示例：{examples}"
        )
    ages = matched[0].astype("float32").to_numpy()
    if ages.min() < MIN_AGE or ages.max() > MAX_AGE:
        raise ValueError(
            f"年龄越界：min={ages.min():.0f} max={ages.max():.0f}"
            f"（期望 {MIN_AGE}~{MAX_AGE}）"
        )
    return ages


def _extract_subjects(filenames: pd.Series) -> pd.Series:
    """从文件名提取身份（姓名段，如 LindaEvans）并做全量覆盖校验。

    AgeDB 命名 `{id}_{name}_{age}_{fm}.jpg`：首段数字是图像编号（每张图
    唯一），身份是第二段姓名（全库 567 人、每人多张照片）。与 _extract_ages
    相同的严谨度：任一行不匹配直接报错，不静默过滤。
    """
    matched = filenames.str.extract(SUBJECT_PATTERN)
    bad = matched[0].isna()
    if bad.any():
        examples = filenames[bad].head(5).tolist()
        raise ValueError(
            f"{int(bad.sum())}/{len(filenames)} 个文件名无法提取身份"
            f"（正则 {SUBJECT_PATTERN.pattern}），示例：{examples}"
        )
    return matched[0]


def _load_image_array(cache_dir: str, paths: list) -> np.ndarray:
    """解码全部图像为 (N, 64, 64, 3) uint8 数组，首次解码后缓存 npy。

    16.5k 张 JPEG 每次训练都现场解码会让数据加载（单进程）耗时 ~7s/epoch、
    远超 GPU 训练 ~0.6s/epoch，故一次性预解码缓存（同 agnews RAM 预载 /
    zinc 处理缓存惯例），之后每 epoch 只需微秒级张量归一化。
    """
    processed_path = os.path.join(cache_dir, PROCESSED_NAME)
    if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
        images = np.load(processed_path, mmap_mode="r")
        if images.shape[0] == len(paths):
            logger.info("AgeDB 预解码缓存命中：%s", processed_path)
            return images
        logger.info("预解码缓存行数不匹配（%d != %d），重新解码", images.shape[0], len(paths))

    logger.info("预解码 AgeDB 图像（%d 张，一次性，约需数十秒）...", len(paths))
    images = np.empty((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        images[i] = np.asarray(img, dtype=np.uint8)
    np.save(processed_path, images)
    logger.info("预解码完成并缓存到 %s（%.0fMB）", processed_path, images.nbytes / 1e6)
    return images


class AgeDBDataset(Dataset):
    """AgeDB 样本容器：每项为 (RGB 64×64 归一化张量 (3,64,64), 缩放后年龄标量)。

    图像数据为共享的预解码 uint8 数组（images），按样本下标索引，
    __getitem__ 只做张量级 ToTensor + Normalize（无需 JPEG 解码）。
    """

    def __init__(self, images: np.ndarray, indices: np.ndarray, ages: np.ndarray, transform):
        self.images = images
        self.indices = indices
        self.ages = ages
        self.transform = transform

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        img = self.images[self.indices[idx]]
        return self.transform(img), torch.tensor(self.ages[idx], dtype=torch.float32)


def get_agedb_dataloaders(
    batch_size: int,
    data_dir: str = "./data",
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    subject_disjoint: bool = False,
):
    """加载 AgeDB 并返回 (train_loader, val_loader, test_loader, y_scaler)。

    全部图像合并后 60/20/20 划分（seed 固定，回归不分层）；图像 RGB 64×64
    + ImageNet 均值/标准差归一化；目标年龄 MinMaxScaler 归一化到 [0,1]
    （仅训练集拟合），y_scaler 供评估时逆归一化 MAE 到岁。

    subject_disjoint=True 时按身份（subject id）分组划分：同一人的所有
    照片只落在一个划分内，防止人脸身份跨 train/val/test 泄漏年龄信息
    （AgeDB 平均每人 ~29 张照片，图像级划分必然同人跨划分）。划分在
    身份级别做 60/20/20（seed 固定，身份不按年龄分层，同图像级约定）。
    """
    cache_dir = download_agedb(data_dir)
    extract_dir = os.path.join(cache_dir, EXTRACT_DIR)
    filenames = sorted(
        f for f in os.listdir(extract_dir) if f.lower().endswith(".jpg")
    )
    if not filenames:
        raise RuntimeError(f"{extract_dir} 下没有 jpg 图像，解压可能不完整")
    df = pd.DataFrame({"filename": filenames})
    df["path"] = df["filename"].map(lambda f: os.path.join(extract_dir, f))
    df["age"] = _extract_ages(df["filename"])
    df["subject"] = _extract_subjects(df["filename"])
    logger.info(
        "AgeDB 图像：%d 张，年龄 %.0f~%.0f 岁", len(df), df["age"].min(), df["age"].max()
    )

    if subject_disjoint:
        subjects = np.sort(df["subject"].unique())
        sub_train, sub_test = train_test_split(
            subjects, test_size=test_ratio, random_state=seed
        )
        sub_train, sub_val = train_test_split(
            sub_train, test_size=val_ratio / (1 - test_ratio), random_state=seed
        )
        df_train = df[df["subject"].isin(sub_train)]
        df_val = df[df["subject"].isin(sub_val)]
        df_test = df[df["subject"].isin(sub_test)]
        logger.info(
            "AgeDB 划分（subject-disjoint 60/20/20）：train %d 人/%d 张，"
            "val %d 人/%d 张，test %d 人/%d 张（共 %d 人/%d 张）",
            len(sub_train), len(df_train), len(sub_val), len(df_val),
            len(sub_test), len(df_test), len(subjects), len(df),
        )
    else:
        df_train, df_test = train_test_split(df, test_size=test_ratio, random_state=seed)
        df_train, df_val = train_test_split(
            df_train, test_size=val_ratio / (1 - test_ratio), random_state=seed
        )
        logger.info(
            "AgeDB 划分（图像级 60/20/20）：train %d / val %d / test %d（共 %d 张）",
            len(df_train), len(df_val), len(df_test), len(df),
        )

    y_train = df_train["age"].to_numpy(dtype="float32")
    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))

    # 预解码缓存（一次性 JPEG 解码 → uint8 数组），之后每项只做张量归一化
    images = _load_image_array(cache_dir, df["path"].tolist())

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    def make_dataset(frame):
        scaled = y_scaler.transform(
            frame["age"].to_numpy(dtype="float32").reshape(-1, 1)
        ).ravel()
        return AgeDBDataset(images, frame.index.to_numpy(), scaled, transform)

    train_loader = DataLoader(
        make_dataset(df_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(make_dataset(df_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(make_dataset(df_test), batch_size=batch_size, shuffle=False)

    logger.info(
        "AgeDB 加载完成：%d/%d/%d（batch %d，RGB %dx%d，ImageNet 归一化），"
        "年龄 MinMax 归一化到 [0,1]（%.0f~%.0f 岁）",
        len(df_train), len(df_val), len(df_test),
        batch_size, IMG_SIZE, IMG_SIZE, y_train.min(), y_train.max(),
    )
    return train_loader, val_loader, test_loader, y_scaler
