#!/bin/bash
# 第三轮补充 b：FGIB-H（--a-identity）锚点尺度敏感性。
# tab_hanchor 显示 A=I 在 a=16 下跨设置略逊（rank 3.40 vs FGIB 1.00），
# 但 prop:orth 只承诺固定正交头与 h 直锚等距，a 的最优值未必继承 z 空间
# 的 16。跑 a=4/8 看重调锚点尺度能否收复差距（a=16 已有 hdirect/）。
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'
P='--runs 5 --parallel 8'
for s in "--task mnist --backbone cnn" "--task imagenet100 --backbone mlp" \
         "--task housing --backbone mlp" "--task cora --backbone gnn"; do
  python tune.py $s --model fgib --a-identity --anchor-scale 4 8 --beta $B \
    --results-dir tune_results_ablation/hdirect_a48 $P
done
echo "=== FGIB-H 锚点敏感性完成 ==="
