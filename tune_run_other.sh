#!/bin/bash
# 压缩-精度评估补训：为扫参组合保存 checkpoint 到 output/compression_eval。
# tune.py 需带 --save-root（保存 _run{i}.pt checkpoint、跳过 --no-save）；
# 分类 OPB 用 --energy-classifier、回归 OPB 用 --tied-head（与主模型一致）。

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

python tune.py --task imagenet100 --backbone mlp --model ceb \
  --beta 15 20 25 \
  --results-dir output/compression_eval --save-root output/compression_eval \
  --parallel 8 --runs 5 --epochs 100

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

python tune.py --task imagenet100 --backbone mlp --model opb \
  --beta 15 20 25 --anchor-scale 1 6 12 \
  --energy-classifier \
  --results-dir output/compression_eval --save-root output/compression_eval \
  --parallel 18 --runs 5 --epochs 100

# 3. AgeDB CEB（6 个 β）
# python tune.py --task housing --backbone mlp --model ceb \
#   --beta 1e-4 1e-3 1e-2 1e-1 1 10 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

# python tune.py --task housing --backbone mlp --model ceb \
#   --beta 5e-5 5e-4 5e-3 5e-2 5e-1 5 \
#   --results-dir output/compression_eval --save-root output/compression_eval \
#   --parallel 8 --runs 5 --epochs 100

python tune.py --task housing --backbone mlp --model ceb \
  --beta 15 20 25 \
  --results-dir output/compression_eval --save-root output/compression_eval \
  --parallel 8 --runs 5 --epochs 100

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

python tune.py --task housing --backbone mlp --model opb \
  --beta 15 20 25 --anchor-scale 1 6 12 \
  --tied-head \
  --results-dir output/compression_eval --save-root output/compression_eval \
  --parallel 18 --runs 5 --epochs 100
