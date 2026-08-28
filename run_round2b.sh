#!/bin/bash
# 修复 cosine_classifier 参数回归后，重跑 8 个非 MLP 骨干的 CentFGIB 设定
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'
RD="--results-dir tune_results_ablation/centfgib_all --runs 5 --parallel 12 --model centfgib --beta $B --anchor-scale 16"
python tune.py --task mnist --backbone cnn $RD
python tune.py --task cora --backbone gnn $RD
python tune.py --task zinc --backbone gnn $RD
python tune.py --task stsb --backbone rnn $RD
python tune.py --task agnews --backbone rnn $RD
python tune.py --task imdb --backbone rnn $RD
python tune.py --task agedb --backbone cnn $RD
python tune.py --task imagenet100 --backbone cnn $RD
echo "=== 补跑完成 ==="
