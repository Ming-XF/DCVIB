#!/bin/bash
# 第三轮补充实验（审稿人修订 v2：方法升级为固定正交侧头 = h 直锚）：
#   1. p3_diag_full/   全验证集 batch 聚合的 P3 方差诊断重跑（MNIST，修复只读
#                      第一个 batch 的问题；vib/ceb/tafgib/fgib a=16）
#   2. etf/            冻结 ETF 分类头基线（审稿人第 7 条，MNIST/MLP）
#   3. hdirect/        h 直锚（--a-identity，A=I 固定正交侧头）跨设置验证：
#                      mnist_cnn / imagenet100_mlp / housing_mlp / cora_gnn
#                      （mnist_mlp 已有 tune_results_ablation/aid/）
#   4. longtrain/      CEB 长训练（400 epochs、patience 400 无早停，β=1/10）
#                      + fgib β=1/10 验证 ℓ_q≡0 不变点持续
#   5. agedb_disjoint/ AgeDB subject-disjoint 身份隔离划分：cnn 基线 + fgib +
#                      centfgib + a-identity（fgib 与 a-identity 组合名相同，
#                      后者放独立子目录 aid/ 防 dirname 覆盖）
# 协议与主实验一致（除 longtrain 外）：100 epochs、patience 10、5 seeds、验证集选模。
set -e
cd /root/autodl-tmp/FGIB
export PYTHONUNBUFFERED=1
B='1e-4 1e-3 1e-2 1e-1 1 10'

# 1) P3 全批诊断（MNIST，最便宜，先跑以验证 diag 修复）
python tune.py --task mnist --backbone mlp --model vib ceb --beta $B \
  --log-variance-stats --results-dir tune_results_ablation/p3_diag_full --runs 5 --parallel 8
python tune.py --task mnist --backbone mlp --model fgib tafgib --beta $B --anchor-scale 16 \
  --log-variance-stats --results-dir tune_results_ablation/p3_diag_full --runs 5 --parallel 8

# 2) ETF 基线
python tune.py --task mnist --backbone mlp --model etf \
  --results-dir tune_results_ablation/etf --runs 5 --parallel 8

# 3) h 直锚跨设置（A=I 即 --a-identity）
python tune.py --task mnist --backbone cnn --model fgib --a-identity --anchor-scale 16 --beta $B \
  --results-dir tune_results_ablation/hdirect --runs 5 --parallel 8
python tune.py --task imagenet100 --backbone mlp --model fgib --a-identity --anchor-scale 16 --beta $B \
  --results-dir tune_results_ablation/hdirect --runs 5 --parallel 8
python tune.py --task housing --backbone mlp --model fgib --a-identity --anchor-scale 16 --beta $B \
  --results-dir tune_results_ablation/hdirect --runs 5 --parallel 8
python tune.py --task cora --backbone gnn --model fgib --a-identity --anchor-scale 16 --beta $B \
  --results-dir tune_results_ablation/hdirect --runs 5 --parallel 8

# 4) CEB 长训练（patience 400 = 无早停，检验 P3 是否早停伪影）
python tune.py --task mnist --backbone mlp --model ceb fgib --beta 1 10 --anchor-scale 16 \
  --epochs 400 --patience 400 --log-variance-stats \
  --results-dir tune_results_ablation/longtrain --runs 5 --parallel 8

# 5) AgeDB subject-disjoint（最重，最后；a-identity 单独子目录）
python tune.py --task agedb --backbone cnn --subject-disjoint \
  --model cnn fgib centfgib --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/agedb_disjoint --runs 5 --parallel 6
python tune.py --task agedb --backbone cnn --subject-disjoint \
  --model fgib --a-identity --beta $B --anchor-scale 16 \
  --results-dir tune_results_ablation/agedb_disjoint/aid --runs 5 --parallel 6

echo "=== 第三轮补充实验完成 ==="
