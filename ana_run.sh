#!/bin/bash


#进行random labels实验分析，验证OPB在无信息的情况下，通过压缩可以不记忆数据
python train.py --task mnist --backbone mlp --model opb --beta 10 --anchor-scale 1 --random-labels


#先训练模型保存模型参数
python train.py --task mnist --backbone mlp --model opb --beta 10 --anchor-scale 1


#加载模型参数，使用扰动数据集部分数据，验证对抗鲁棒性泛化
python adv_eval.py --model-dirs output/adv_mnist/mnist_mlp_vib_beta_0.01 output/adv_mnist/mnist_mlp_opb_beta_0.01_anchor_4 --eps-linf 0.1 0.2 0.3 --eps-l2 0.5 1 1.5 2 --pgd-steps 20 --n-test 1000 --runs 5 --results-dir adv_results


#加载模型参数，验证先验分布几何
python prior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb --runs 5 --anchor-scale 10 --results-dir pri_results

#加载模型参数，使用数据集部分数据，验证后验分布几何
python posterior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb --runs 5 --n-test 2000 --anchor-scale 10


#能量分类器消融试验，验证几何真正参与预测
python train.py --model opb --energy-classifier --beta 10 --anchor-scale 10


#最近锚点分类准确率（两个不同的分类器），分类器权重-锚点对齐，验证几何真正参与预测
python geometry_usage.py --model-dirs output/ablation/mnist_mlp_opb_ec_a10 --anchor-scale 10 --energy-classifier --results-dir geo_results_ec