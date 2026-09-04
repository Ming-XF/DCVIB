#!/bin/bash
# 两类补训（为后续评估保存 checkpoint，均带 --save-root 生成 _run{i}.pt）：
# 1. 压缩-精度评估补训：保存到 output/compression_eval（imagenet100/housing，
#    分类 OPB 用 --energy-classifier、回归 OPB 用 --tied-head，与主模型一致）；
# 2. 压缩-鲁棒性评估补训（文末）：MNIST β 网格保存到 output/adc_mnist。
# tune.py 无跳过逻辑——已存在的结果目录也会整体重训，按需注释分组。

# 0. 清理旧补训残留（含无 .pt 后缀修复前的错误命名 checkpoint）
# rm -rf output/compression_eval

# # 1. ImageNet-100 CEB（6 个 β）
# python tune.py --task imagenet100 --backbone mlp --model ceb \
#   --beta 1e-4 1e-3 1e-2 1e-1 1 10 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# python tune.py --task imagenet100 --backbone mlp --model ceb \
#   --beta 5e-5 5e-4 5e-3 5e-2 5e-1 5 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# python tune.py --task imagenet100 --backbone mlp --model ceb \
#   --beta 15 20 25 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# # 2. ImageNet-100 OPB（6 β × 3 a，能量分类器——与主模型一致）
# python tune.py --task imagenet100 --backbone mlp --model opb \
#   --beta 1e-4 1e-3 1e-2 1e-1 1 10 --anchor-scale 1 6 12 \
#   --energy-classifier \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 18 --runs 5 --epochs 100

# python tune.py --task imagenet100 --backbone mlp --model opb \
#   --beta 5e-5 5e-4 5e-3 5e-2 5e-1 5 --anchor-scale 1 6 12 \
#   --energy-classifier \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 18 --runs 5 --epochs 100

# python tune.py --task imagenet100 --backbone mlp --model opb \
#   --beta 15 20 25 --anchor-scale 1 6 12 \
#   --energy-classifier \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 18 --runs 5 --epochs 100

# 3. AgeDB CEB（6 个 β）
# python tune.py --task housing --backbone mlp --model ceb \
#   --beta 1e-4 1e-3 1e-2 1e-1 1 10 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# python tune.py --task housing --backbone mlp --model ceb \
#   --beta 5e-5 5e-4 5e-3 5e-2 5e-1 5 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# python tune.py --task housing --backbone mlp --model ceb \
#   --beta 15 20 25 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# 4. AgeDB OPB（6 β × 3 a，tied 投影头——与主模型一致）
# python tune.py --task housing --backbone mlp --model opb \
#   --beta 1e-4 1e-3 1e-2 1e-1 1 10 --anchor-scale 1 6 12 \
#   --tied-head \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 18 --runs 5 --epochs 100

# python tune.py --task housing --backbone mlp --model opb \
#   --beta 5e-5 5e-4 5e-3 5e-2 5e-1 5 --anchor-scale 1 6 12 \
#   --tied-head \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 18 --runs 5 --epochs 100

# python tune.py --task housing --backbone mlp --model opb \
#   --beta 15 20 25 --anchor-scale 1 6 12 \
#   --tied-head \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 18 --runs 5 --epochs 100

# ================================================================
# 压缩-鲁棒性评估补训：MNIST β 网格 → 压缩-鲁棒性曲线（E[KL] × 对抗精度）
# checkpoint 保存到 output/adc_mnist/{combo}/{combo}_run{i}.pt（+ train.log）。
# OPB 分类必须带 --energy-classifier（adv_eval 重建 OPB 固定按能量分类器，
# 且与主模型/论文口径一致）——train.py 对该 flag 限定 opb 模型，故与
# ceb/mlp 分两次调用。MNIST 全默认超参（batch 512、hidden 512 256、z-dim 256、
# dropout 0.2、100 epochs、5 runs、seed 0），与论文鲁棒性配置
# （CEB β=0.1 / OPB β=0.1 a=12）同口径。β 网格与压缩-精度试验一致
# {5e-5..25}；a ∈ {1, 6, 12}。共 1 + 15 + 45 = 61 组合 × 5 runs。
# 注：output/adv_mnist 的旧鲁棒性 checkpoint（β ∈ {1e-3,1e-2,1e-1,1} 子集）
# 是主网格复制件、命名不规则；本目录全部重训、命名统一（compression_eval 的
# 目录解析要求 {dataset}_{backbone}_{model}_beta_{b}_anchor_{a}），勿混用。
# ================================================================

# 1. 基线 + CEB（1 + 15 β）
python tune.py --task mnist --backbone mlp --model mlp ceb \
  --beta 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 10 15 20 25 \
  --results-dir output/adc_mnist --save-root output/adc_mnist \
  --parallel 8 --runs 5 --epochs 100

# 2. OPB（15 β × 3 a，能量分类器——与主模型一致）
python tune.py --task mnist --backbone mlp --model opb \
  --beta 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 10 15 20 25 \
  --anchor-scale 1 6 12 \
  --energy-classifier \
  --results-dir output/adc_mnist --save-root output/adc_mnist \
  --parallel 8 --runs 5 --epochs 100
