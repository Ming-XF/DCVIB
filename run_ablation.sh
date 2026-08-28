#!/bin/bash
# MNIST/MLP 审稿人补充实验（第 1/2/3/4 条）：
#   p3_diag/  P3 方差竞赛测量（vib/ceb 后验-先验 logvar + clamp 占比）与 FGIB κ(AᵀA)
#   主目录    dceb / tafgib / centfgib 混淆消融（第 4 条）
#   freezea/  FGIB 冻结旁路头 A
#   aid/      FGIB A = I（固定正交）
#   cosine/   固定温度 cosine 分类器下的 fgib/vib/ceb（第 2 条理论验证）
# 协议与主实验一致：100 epochs、patience 10、5 seeds（0..4）、验证集选模。
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'
COMMON="--results-dir tune_results_ablation --runs 5 --parallel 12"

python tune.py --model vib ceb --beta $B --results-dir tune_results_ablation/p3_diag --runs 5 --parallel 12 --log-variance-stats
python tune.py --model fgib --beta $B --anchor-scale 16 --results-dir tune_results_ablation/p3_diag --runs 5 --parallel 12 --log-variance-stats

python tune.py --model dceb --beta $B $COMMON
python tune.py --model tafgib --beta $B --anchor-scale 4 16 $COMMON
python tune.py --model centfgib --beta $B --anchor-scale 4 16 $COMMON

python tune.py --model fgib --beta $B --anchor-scale 16 --results-dir tune_results_ablation/freezea --runs 5 --parallel 12 --freeze-a
python tune.py --model fgib --beta $B --anchor-scale 16 --results-dir tune_results_ablation/aid --runs 5 --parallel 12 --a-identity

python tune.py --model fgib --beta $B --anchor-scale 16 --results-dir tune_results_ablation/cosine --runs 5 --parallel 12 --cosine-classifier
python tune.py --model vib --beta $B --results-dir tune_results_ablation/cosine --runs 5 --parallel 12 --cosine-classifier
python tune.py --model ceb --beta $B --results-dir tune_results_ablation/cosine --runs 5 --parallel 12 --cosine-classifier

echo "=== 全部补充实验完成 ==="
