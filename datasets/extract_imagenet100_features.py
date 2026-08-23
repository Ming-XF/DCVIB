"""使用 ImageNet-1k 预训练的 ResNet50 对 ImageNet-100 提取中间层特征。

流程：
1. 解压 imagenet100.zip 到 data/imagenet100/images/（幂等，已完整解压则跳过）
2. 加载 torchvision resnet50（IMAGENET1K_V2 官方权重），截断到指定层
3. 按官方 eval 预处理遍历全部图片，逐批提取该层输出（全局平均池化后展平）作为特征
4. 保存 features（float32 N×dim）/ labels（int64 N）/ classes（类别文件夹名）到 .npz

--layer 可选 penultimate（avgpool 输出 2048 维，即倒数第二层）或
layer1/layer2/layer3/layer4（对应残差块输出，池化后 256/512/1024/2048 维），默认 layer3。

用法：
    python datasets/extract_imagenet100_features.py                 # 全量提取（默认 layer3）
    python datasets/extract_imagenet100_features.py --layer penultimate   # 提取倒数第二层
    python datasets/extract_imagenet100_features.py --smoke-classes 4   # 冒烟测试：仅前 4 个类
"""
import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

logger = logging.getLogger("extract_features")

EXPECTED_NUM_IMAGES = 126689
IMG_SUFFIXES = {".jpeg", ".jpg", ".png"}
# 各截断层池化后的特征维度（顶层模块序：conv1, bn1, relu, maxpool,
# layer1, layer2, layer3, layer4, avgpool, fc）
LAYER_DIMS = {
    "penultimate": 2048,
    "layer1": 256,
    "layer2": 512,
    "layer3": 1024,
    "layer4": 2048,
}


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet-100 预训练 ResNet50 特征提取")
    parser.add_argument("--zip-path", type=Path,
                        default="/root/autodl-pub/ImageNet100/imagenet100.zip")
    parser.add_argument("--extract-dir", type=Path, default="data/imagenet100/images",
                        help="解压父目录，类别文件夹位于 <extract-dir>/imagenet100/ 下")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出 npz 路径（默认 data/imagenet100/imagenet100_resnet50_features.npz）")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke-classes", type=int, default=0,
                        help=">0 时只解压/提取前 N 个类别，用于冒烟测试")
    parser.add_argument("--layer", type=str, default="layer3",
                        choices=["penultimate", "layer1", "layer2", "layer3", "layer4"],
                        help="截断到 ResNet50 的哪一层（默认 layer3，全局平均池化后展平）")
    parser.add_argument("--log-path", type=Path, default="data/imagenet100/extract_features.log")
    return parser.parse_args()


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def count_images(root):
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def extract_zip(zip_path, extract_dir, smoke_classes):
    """解压 zip（幂等）。冒烟测试时用 zipfile 只抽前 N 类，速度快。"""
    dataset_root = extract_dir / "imagenet100"
    if smoke_classes <= 0:
        class_dirs = [d for d in dataset_root.iterdir() if d.is_dir()] if dataset_root.is_dir() else []
        if len(class_dirs) == 100 and count_images(dataset_root) == EXPECTED_NUM_IMAGES:
            logger.info("已存在完整解压目录 %s，跳过解压", dataset_root)
            return
    extract_dir.mkdir(parents=True, exist_ok=True)
    if smoke_classes > 0:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            class_names = sorted({n.split("/")[1] for n in names if n.count("/") == 2})
            selected = class_names[:smoke_classes]
            members = [n for n in names if n.count("/") == 2 and n.split("/")[1] in selected]
            logger.info("冒烟测试：仅解压 %d 个类别 %s", len(selected), selected)
            for m in tqdm(members, desc="解压(冒烟)"):
                zf.extract(m, extract_dir)
    else:
        logger.info("开始解压 %s -> %s", zip_path, extract_dir)
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract_dir)], check=True)
    logger.info("解压完成，当前共 %d 张图片", count_images(dataset_root))


class SafeImageFolder(ImageFolder):
    """容错版 ImageFolder：损坏图片返回黑图占位并记录，保证特征/标签索引对齐。"""

    def __init__(self, *args, **kwargs):
        self.bad_images = []
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as exc:
            self.bad_images.append(path)
            logger.warning("图片加载失败，用黑图占位：%s（%s）", path, exc)
            sample = Image.new("RGB", (224, 224))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def load_backbone(device, layer="layer3"):
    """加载官方 V2 权重，截断到指定层并接全局平均池化，输出展平特征 (B, dim)。

    penultimate 截断到 avgpool（含）；layer1~4 截断到对应残差块输出，
    再接 AdaptiveAvgPool2d(1) 池化。
    """
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device).eval()
    cut = {"layer1": 5, "layer2": 6, "layer3": 7, "layer4": 8, "penultimate": 9}[layer]
    children = list(model.children())[:cut]
    if layer != "penultimate":
        children.append(nn.AdaptiveAvgPool2d(1))
    children.append(nn.Flatten())
    return nn.Sequential(*children)


@torch.inference_mode()
def extract_features(backbone, loader, device, n_total, feature_dim):
    features = np.empty((n_total, feature_dim), dtype=np.float32)
    labels = np.empty((n_total,), dtype=np.int64)
    start = 0
    for x, y in tqdm(loader, desc="提取特征"):
        feat = backbone(x.to(device, non_blocking=True)).cpu().numpy()
        b = feat.shape[0]
        features[start:start + b] = feat
        labels[start:start + b] = y.numpy()
        start += b
    return features, labels


def verify_and_save(out, features, labels, classes, feature_dim):
    assert features.shape == (len(labels), feature_dim)
    assert np.isfinite(features).all(), "特征存在非有限值"
    assert labels.min() >= 0 and labels.max() == len(classes) - 1
    np.savez(out, features=features, labels=labels, classes=np.array(classes))
    logger.info("已保存 %s（%.2f GB）", out, out.stat().st_size / 1e9)
    # 读回验证
    data = np.load(out)
    logger.info("读回验证：features %s %s，labels %s %s，classes %d 个",
                data["features"].shape, data["features"].dtype,
                data["labels"].shape, data["labels"].dtype, len(data["classes"]))
    counts = np.bincount(data["labels"], minlength=len(data["classes"]))
    logger.info("每类样本数：min %d / max %d", counts.min(), counts.max())
    logger.info("特征每维均值 %.4f±%.4f，样本均值 %.4f±%.4f",
                data["features"].mean(), data["features"].std(),
                data["features"].mean(axis=1).mean(), data["features"].mean(axis=1).std())


def main():
    args = parse_args()
    setup_logging(args.log_path)

    extract_zip(args.zip_path, args.extract_dir, args.smoke_classes)
    dataset_root = args.extract_dir / "imagenet100"

    weights = ResNet50_Weights.IMAGENET1K_V2
    full_dataset = SafeImageFolder(root=dataset_root, transform=weights.transforms())
    if args.smoke_classes > 0:
        # 目录已完整解压时 ImageFolder 会看到全部类别，这里显式只保留前 N 类
        indices = [i for i, (_, c) in enumerate(full_dataset.samples)
                   if c < args.smoke_classes]
        dataset = Subset(full_dataset, indices)
        classes = full_dataset.classes[:args.smoke_classes]
    else:
        dataset = full_dataset
        classes = full_dataset.classes
    logger.info("数据集：%d 张图片，%d 个类别（%s ...）",
                len(dataset), len(classes), classes[:3])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("CUDA 不可用，回退到 CPU")
    backbone = load_backbone(device, args.layer)
    feature_dim = LAYER_DIMS[args.layer]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    features, labels = extract_features(backbone, loader, device, len(dataset), feature_dim)

    if full_dataset.bad_images:
        logger.warning("共 %d 张损坏图片被黑图占位：%s", len(full_dataset.bad_images),
                       [str(p) for p in full_dataset.bad_images[:5]])

    out = args.out
    if out is None:
        name = f"imagenet100_resnet50_{args.layer}_features"
        if args.smoke_classes > 0:
            name += f"_smoke{args.smoke_classes}"
        out = Path("data/imagenet100") / f"{name}.npz"
    verify_and_save(out, features, labels, classes, feature_dim)


if __name__ == "__main__":
    main()
