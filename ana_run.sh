#!/bin/bash


#主实验

#自由分类/回归预测头消融

#精度/压缩 权衡


#先训练模型保存模型参数
python train.py --task mnist --backbone mlp --model opb --beta 10 --anchor-scale 1


#加载模型参数，使用扰动数据集部分数据，验证对抗鲁棒性泛化
python adv_eval.py --model-dirs output/adv_mnist/mnist_mlp output/adv_mnist/mnist_mlp_opb_beta0.1_scale12 output/adv_mnist/mnist_mlp_ceb_beta0.1 --eps-linf 0.05 0.1 0.15 0.2 0.25 0.3 --eps-l2 1 2 3 4 5 6 --pgd-steps 20 --n-test 1000 --runs 5 --results-dir output/adv_mnist


#加载模型参数，验证先验分布几何
python prior_geometry.py --model-dirs output/mnist_mlp_ceb --runs 5 --anchor-scale 10 --results-dir output/pri-pos/pri_results

#加载模型参数，使用数据集部分数据，验证后验分布几何
python posterior_geometry.py --model-dirs output/mnist_mlp_ceb output/mnist_mlp_opb --runs 5 --n-test 2000 --anchor-scale 10 --results-dir output/pri-pos/pos_results



#进行random labels实验分析，验证OPB在无信息的情况下，通过压缩可以不记忆数据
python train.py --task mnist --backbone mlp --model opb --beta 10 --anchor-scale 1 --random-labels



