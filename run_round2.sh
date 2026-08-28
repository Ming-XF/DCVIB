#!/bin/bash
# 第二轮补充实验：
#   1. tafgib_diag/  TAFGIB 可训练锚点的最终几何测量（MNIST，审稿人第 2 条）
#   2. centfgib_all/ CentFGIB 全 12 设定（6 β × a=16 × 5 seeds，与等预算 FGIB 同预算）
# 协议与主实验一致：100 epochs、patience 10、5 seeds、验证集选模。
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'

python tune.py --model tafgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/tafgib_diag --runs 5 --parallel 12 --log-variance-stats

# CentFGIB：按便宜到贵的顺序，每任务每骨干一次调用
python tune.py --task mnist --backbone mlp --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task mnist --backbone cnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task housing --backbone mlp --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task cora --backbone gnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task zinc --backbone gnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task stsb --backbone rnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task agnews --backbone rnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task agedb --backbone mlp --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task agedb --backbone cnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task imdb --backbone rnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task imagenet100 --backbone mlp --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12
python tune.py --task imagenet100 --backbone cnn --model centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12

echo "=== 第二轮补充实验完成 ==="
