#!/bin/bash
# 第四轮修订实验：
#   1. hdirect_all/   FGIB-H（--a-identity）补全 7 个分类设置：hdirect/（a=16）
#                     已有 mnist_cnn/imagenet100_mlp/housing/cora，aid/ 有 mnist_mlp
#                     a=16；本次补 mnist_mlp a=4 与 imagenet100_cnn/imdb/agnews 的
#                     a=4/16（各 6 点 β × 5 seeds），使 tab_hanchor 成为无缺格的
#                     7 设置表（审稿人意见 4：canonical 方法与主实验对齐）。
#   2. dceb_diag/     DCEB 长训练诊断（400 epochs、patience 400 无早停，β=1/10，
#                     --log-variance-stats）：核验修正后的符号——可训练先验方差被
#                     拉**升**而非压下（审稿人意见 1）。
#   3. vib_longtrain/ VIB 长训练对照（400 epochs，β=1/10，--log-variance-stats）：
#                     固定先验下后验方差被恢复力拉回 1、高 β 失败模式是 μ→0 的
#                     预测信息消失而非方差钳位竞赛（审稿人意见 3）。
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'

# 1) FGIB-H 补全（a-identity 与普通 fgib 目录名相同，独立 results-dir 防覆盖）
python tune.py --task mnist --backbone mlp --model fgib --a-identity --anchor-scale 4 \
  --beta $B --results-dir tune_results_ablation/hdirect_all --runs 5 --parallel 8
python tune.py --task imagenet100 --backbone cnn --model fgib --a-identity --anchor-scale 4 16 \
  --beta $B --results-dir tune_results_ablation/hdirect_all --runs 5 --parallel 8
python tune.py --task imdb --backbone rnn --model fgib --a-identity --anchor-scale 4 16 \
  --beta $B --results-dir tune_results_ablation/hdirect_all --runs 5 --parallel 8
python tune.py --task agnews --backbone rnn --model fgib --a-identity --anchor-scale 4 16 \
  --beta $B --results-dir tune_results_ablation/hdirect_all --runs 5 --parallel 8

# 2) DCEB 长训练诊断（验证先验方差方向修正）
python tune.py --task mnist --backbone mlp --model dceb --beta 1 10 \
  --epochs 400 --patience 400 --log-variance-stats \
  --results-dir tune_results_ablation/dceb_diag --runs 5 --parallel 8

# 3) VIB 长训练对照（固定先验 → 无钳位竞赛）
python tune.py --task mnist --backbone mlp --model vib --beta 1 10 \
  --epochs 400 --patience 400 --log-variance-stats \
  --results-dir tune_results_ablation/vib_longtrain --runs 5 --parallel 8

echo "=== 第四轮补充实验（hdirect_all / dceb_diag / vib_longtrain）完成 ==="
