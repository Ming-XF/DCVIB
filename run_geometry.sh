#!/bin/bash
# 第四轮补充 b：几何消融（审稿人"最有价值的补实验"第 2 条）——
# 在同一表示（--a-identity，A=I，锚点直接作用于部署表示 h）上比较
# 锚点几何：qr 正交（默认）/ etf 单纯形框架 / random 随机单位方向
# 与 learned-center（TAFGIB 可训练锚点均值），MNIST/MLP、a=16、6 点 β。
# 冻结 ETF 分类头不是 fixed-prototype representation loss 的等价基线，
# 本消融补齐真正的同表示几何对照。
# 等 run_round4.sh 完成后才启动（imdb 8×3.7GB/卡 + 本组 8×~1GB 会超 32GB）。
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'

while pgrep -f "bash run_round4.sh" > /dev/null; do sleep 60; done
echo "=== run_round4 已完成，开始几何消融 ==="

for g in qr etf random; do
  python tune.py --task mnist --backbone mlp --model centfgib --a-identity \
    --anchor-geometry $g --anchor-scale 16 --beta $B \
    --results-dir tune_results_ablation/geometry/$g --runs 5 --parallel 8
done
# learned-center：可训练锚点均值（TAFGIB）+ A=I，同表示
python tune.py --task mnist --backbone mlp --model tafgib --a-identity \
  --anchor-scale 16 --beta $B \
  --results-dir tune_results_ablation/geometry/learned --runs 5 --parallel 8

echo "=== 几何消融完成 ==="
