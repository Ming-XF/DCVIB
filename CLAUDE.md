# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

CVIB：在多个数据集（MNIST / ImageNet-100 特征 / Cora / IMDb / AG News BERT 特征 / California Housing）上对比 MLP / CNN / GCN / RNN 基线与其变分信息瓶颈变体（VIB / CEB / DCVIB / FGIB）。代码注释、文档字符串和提交信息均为中文。

## 常用命令

```bash
python train.py                     # 默认训练 MLP（100 epochs, 5 runs）
python train.py --model cnn                     # CNN 基线
python train.py --model vib         # VIB（N(0,I) 先验）
python train.py --model ceb         # CEB（标签条件先验 r(z|y)，对角高斯）
python train.py --model dcvib       # DCVIB（确定性主路 + 旁路瓶颈）
python train.py --model fgib        # FGIB（固定几何信息瓶颈：DCVIB + 固定正交锚点先验）
python train.py --model fgib --anchor-scale 8   # 自定义锚点尺度（0 为各类相同锚点）
python train.py --backbone cnn --model vib      # CNN 版 VIB（ceb/dcvib/fgib 同理）
python train.py --task imagenet100 --model vib      # ImageNet-100 特征分类（MLP 骨干）
python train.py --task imagenet100 --model fgib    # ImageNet-100 特征分类（FGIB）
python train.py --task cora --model gcn             # Cora 节点分类（GCN 基线）
python train.py --task cora --backbone gnn --model vib     # GNN 版 VIB（ceb/dcvib/fgib 同理）
python train.py --task imdb --model rnn             # IMDb 情感分析（RNN 基线）
python train.py --task imdb --backbone rnn --model vib     # RNN 版 VIB（ceb/dcvib/fgib 同理）
python train.py --task agnews --model rnn           # AG News BERT 逐 token 特征分类（RNN 基线）
python train.py --task agnews --backbone rnn --model vib   # RNN 版 VIB（ceb/dcvib/fgib 同理）
python train.py --task housing                          # California Housing 回归（MLP 基线）
python train.py --task housing --model vib              # VIB 回归
python train.py --task housing --model ceb              # CEB 回归（连续 y 条件先验）
python train.py --task housing --model dcvib            # DCVIB 回归（连续 y 条件先验）
python train.py --task housing --model fgib             # FGIB 回归（固定 RFF 连续锚点先验）
python train.py --model mlp --epochs 20 --batch-size 128 --lr 1e-3 --beta 1e-3
python train.py --model vib --runs 5 --seed 0   # 多 run 取测试集平均指标
python train.py --random-labels --patience 1000   # 随机标签记忆实验（信息自由数据集，看 Train Acc）
python tune.py --model vib ceb fgib --beta 1e-4 1e-3 1e-2 --anchor-scale 1 2 4 8 --parallel 4   # 调参：模型 × beta × anchor-scale 网格并行训练并生成 HTML 结果表
```

- 没有测试套件、linter 配置或 requirements.txt。环境：Python 3.12，torch 2.9.0+cu128，torchvision 0.24.0，scikit-learn 1.7.2（CUDA 可用）。
- 完整参数见 `train.py` 的 argparse：`--task`（默认 mnist（MNIST 分类）；housing 为 California Housing 回归、仅 MLP 骨干；imagenet100 用预训练 ResNet50 特征、仅 MLP 骨干；cora 仅 GNN 骨干；imdb/agnews 仅 RNN 骨干，其中 agnews 用 BERT 逐 token 特征）、`--backbone`（默认 mlp；`--model` 为 mlp/cnn/gcn/rnn 时被忽略）、`--hidden-dims`（默认 512 256，MLP/GNN 的各层维度；RNN 首元素为词嵌入维度、列表为逐层 LSTM 隐层维度，层数 = 列表长度）、`--dropout`（0.2）、`--z-dim`（256，仅 VIB 系）、`--max-len`（500，仅 IMDb）、`--random-labels`（随机化训练标签，固定 seed 42，仅分类任务；信息自由数据集记忆实验）、`--beta`（KL 权重，1e-3）、`--anchor-scale`（4.0，仅 fgib；固定锚点先验均值的尺度，0 为 N(0,I) 先验；分类为各类正交锚点距离、回归为锚点球面半径）、`--patience`（早停，10）、`--data-dir`、`--save-path`、`--log-path`。

## 架构

**模型**：四个骨干各有一组同构的五个变体（基线/VIB/CEB/DCVIB/FGIB），接口约定相同（见下）。

- MLP 骨干（`model/mlp/`；`model/__init__.py` 从 `.mlp` 重导出，`train.py` 的 `from model import ...` 接口不变）：
  - `MLP`：纯基线，`forward(x)` 只返回 logits。
  - `VIB`：编码器 h → mu/logvar 头 → 重参数化采样 z → 分类器；先验为固定 N(0, I)。
  - `CEB`：条件熵瓶颈（Fischer 2020），同 VIB，但先验改为可学习的 `prior_net`（反向编码器 b(z|y)，无激活线性层，分类时 one-hot 标签、`continuous_y=True` 时连续 y → (mu_p, logvar_p)），KL 为 KL(q(z|x) || r(z|y))；目标 VCEB = KL + γ·CE，项目损失约定下等价 CE + β·KL（β = 1/γ）。本实现为对角高斯版本（Fischer 2020 原版为全协方差小瓶颈 D=4，对角化使瓶颈可扩展到高维 z）。
  - `DCVIB`：主路确定性（h 直接分类，z 不参与前向）；旁路从共享编码器 h 引出 mu/logvar 头计算标签条件 KL，梯度只通过共享编码器回传作为正则。forward 只需 `(x, labels)`，无采样。
  - `FGIB`：固定几何信息瓶颈，同 DCVIB，但标签条件先验换成 `register_buffer` 的固定锚点先验（不参与梯度），KL 无法靠移动先验减小，只能靠 z 对齐锚点，正则力持续存在。**分类**（`continuous_y=False`）：`build_anchor_prior`（`model/mlp/utils.py`）对随机矩阵做 QR 分解得到 K 个正交方向，缩放 `anchor_scale` 作为各类先验均值、方差固定为 `anchor_var`（1.0），要求 `num_classes <= z_dim`（imagenet100 的 100 ≤ 256 满足）。**回归**（`continuous_y=True`，仅 housing/MLP）：`ContinuousAnchorPrior` 用固定随机傅里叶特征 mu_p(y) = anchor_scale·√(2/z_dim)·cos(2π·ω·y+b)，ω~N(0,2²)、b~U[0,2π) 固定 buffer（零参数，受 run seed 控制），锚点近似落在半径 anchor_scale 的球面上，anchor_scale 语义与分类版一致。两种模式 `anchor_scale=0` 均退化为 N(0,I) 先验。
- CNN 骨干（`model/cnn/`）：同名五个变体 `CNN`/`VIB`/`CEB`/`DCVIB`/`FGIB`，结构与 MLP 版一一对应，仅把 MLP 隐藏层换成卷积特征提取器（`build_cnn_encoder`：Conv3x3+ReLU+MaxPool2x2 堆叠，默认通道 (32, 64)，28×28 → 64×7×7=3136 展平后接 Linear 到 hidden_dim=256）；瓶颈头与先验部分和 MLP 版完全一致，`reparameterize`/`kl_divergence` 复用自 `model/mlp/utils.py`。因与 MLP 版类名冲突，顶层 `model/` 只重导出 `CNN`，CNN 版 VIB 系列需 `from model.cnn import VIB` 等导入。
- GNN 骨干（`model/gnn/`，Cora 转导式节点分类）：同名五个变体 `GCN`/`VIB`/`CEB`/`DCVIB`/`FGIB`，`forward` 多收 `adj_norm`（对称归一化邻接，稠密张量）与 `mask` 参数；CEB/DCVIB/FGIB 用 `kl_divergence_masked`（`model/gnn/utils.py`）把 KL 限制在 train mask 节点上，防止标签条件先验接触验证/测试节点标签（标签泄漏）；GNN 版无 `continuous_y`。
- RNN 骨干（`model/rnn/`，IMDb 情感二分类 / AG News BERT 特征分类）：同名五个变体 `RNN`/`VIB`/`CEB`/`DCVIB`/`FGIB`，瓶颈头与先验部分和 MLP 版完全一致，仅编码器换成 LSTM 文本编码器（`build_rnn_encoder`：Embedding(vocab_size, hidden_dims[0], padding_idx=0) → 多层 LSTM（层数与各层隐层维度由 `--hidden-dims` 列表决定，默认 (512, 256) 两层，逐层 feed 前一层的 packed 输出，除末层外每层输出后接 Dropout）→ 取末层 `h_n[-1]` → Dropout → h (B, hidden_dims[-1])）。**两种输入模式**：`input_dim=None`（IMDb）时输入为 (B, L) 的 token id 张量（pad_idx=0 后填充），长度由 `(x != pad_idx).sum(1)` 计算；`input_dim=768`（AG News）时跳过 Embedding，输入为 (B, L, 768) 连续向量序列（BERT 逐 token 特征，全零向量后填充），长度由零向量检测（`(x.abs().sum(-1) > 0).sum(1)`）计算；均 pack 后送入 LSTM（lengths 需 `.cpu()`）。RNN 版 VIB 系列同样需 `from model.rnn import VIB` 等导入。

**统一前向接口**（`train.py:run_model`）：所有模型返回 `(logits, kl)`，基线返回 `kl=None`；VIB/CEB 额外接收 `stochastic` 标志（训练 True 采样，评估 False 用 z=mu），GNN 版另传 `adj_norm`/`mask`。损失 = CrossEntropy + beta·KL（kl 为 None 时省略）。DCVIB/FGIB 训练路径必须传入 labels（否则 kl=None 退化为纯基线）。

**关键约定**：

- 所有 VIB 系模型的 logvar 头和 `prior_net` 做置零初始化，使训练起点 KL≈0、sigma=1 —— 修改模型初始化时需保持这一点。FGIB 例外：先验锚点均值非零，置零 logvar 头下训练起点 sigma=1、KL ≈ 0.5·anchor_scale²。
- `model/mlp/utils.py` 提供共享工具：`flatten`（(B,C,H,W)→(B,C·H·W)）、`build_hidden_layers`（Linear+ReLU+Dropout）、`reparameterize`、`kl_divergence`（对角高斯闭式解，logvar clamp 到 [-10, 10]，逐样本求和后 batch 平均）、`build_anchor_prior`（FGIB 固定锚点先验表构造，QR 正交方向）；CNN/RNN/GNN 模型跨包复用其中的工具。
- CEB/DCVIB/FGIB 在 `labels is None` 时返回 `kl=None`（评估或推理时无标签），训练路径必须传入 labels。
- 回归时 CEB/DCVIB/FGIB 用 `continuous_y=True`、`num_classes=1`（输出 (B,1)）；先验条件在归一化后的连续 y 上（housing 标签为 (B,) 一维浮点，`labels.float().unsqueeze(-1)` → `prior_net` / RFF 锚点映射），仍保持置零初始化。仅 MLP 骨干（CNN 不支持回归，GNN/RNN 骨干无回归任务）。

**训练流程**（`train.py`）：

- 数据（分类）：MNIST，训练集 60k 切出 10k 验证（`random_split`，generator 固定 seed 42）。
- 数据（imagenet100）：预训练 ResNet50 特征的 npz，60/20/20 分层划分 + StandardScaler（仅训练集拟合），仅 MLP 骨干。
- 数据（cora）：全图批转导训练，无 DataLoader（`datasets/cora.py`，划分固定 seed 42）。
- 数据（imdb）：斯坦福官方 aclImdb_v1.tar.gz（官方 train/test 各 25k 已标注影评）合并后 60/20/20 分层划分（seed 42）→ 30k/10k/10k；词表仅从训练集构建（pad_idx=0、unk_idx=1 保留位，其余按词频编号，约 9 万词）；去 HTML 标签后纯字母分词、截断到 `--max-len`（500）；collate 用 `pad_sequence` 后填充。二分类评估时 `evaluate` 把 (n,2) 概率取第 1 列再算 AUC（sklearn 二分类要求 (n,)）。
- 数据（agnews）：冻结 bert-base-uncased 提取的逐 token **倒数第三层（layer 10）**隐状态（`datasets/extract_agnews_features.py`，`--layer` 可选 1..12，需 `source /etc/network_turbo` 代理 + `HF_HUB_DISABLE_XET=1`），127.6k × 64 × 768 fp16 存于 `data/agnews/agnews_bert_base_layer{layer}_features.npy`（全零向量后填充，长度由零向量检测）；加载端 60/20/20 分层划分 + RAM 预载 fp32（`datasets/agnews.py`，layer 默认 10）。注意：本地 `datasets/` 包与 HF `datasets` 库同名，提取脚本里先移除脚本目录再导入 HF 库。
- 数据（回归）：`fetch_california_housing`（20640 样本、8 特征），60/20/20 划分（seed 42），特征用 StandardScaler 标准化、目标用 MinMaxScaler 归一化到 [0,1]（均仅在训练集上拟合）。损失 MSELoss 在归一化目标上计算，MAE 用 y_scaler 逆归一化回原始单位（十万美元；R² 与量纲无关），按验证 R² 早停。CNN 骨干不支持回归（argparse 直接报错）。
- 每 run 使用 seed+i-1；分类按验证集宏平均 AUC、回归按验证 R² 早停（patience=10），保存最优 state dict 到 `output/<name>/<name>_run{i}.pt`，训练日志写入 `output/<name>/train_<name>.log`（logging 同时输出到控制台和文件）。训练日志每 epoch 报告 Train Loss + **Train Acc**（回归无 Acc）——随机标签实验靠 Train Acc 爬升识别记忆（val/test 始终 ~随机水平；MLP 约 40 epoch 开始记忆，实验时用 `--patience 1000` 避免早停）。开启 `--random-labels` 时结束后额外报告所有 run 的**最终 epoch Train Acc 均值±标准差**。`<name>` 按 `{dataset}_{backbone}_{model}` 命名（基线为 `{dataset}_{model}`），dataset 为 `mnist`/`imagenet100`/`cora`/`imdb`/`california`，如 `output/mnist_mlp_vib/`、`output/mnist_cnn_vib/`、`output/cora_gnn_vib/`、`output/imdb_rnn_vib/`、`output/california_mlp_ceb/`——每个 任务×骨干×模型 组合对应唯一目录。
- 早停结束后加载最优 checkpoint 在测试集评估（分类报告 Loss/Acc/宏平均 AUC/Precision/Recall，回归报告 Loss/MAE/R²，其中 Loss 含 beta·KL 项、MAE/R² 为纯预测指标），最后对全部 runs 取均值±标准差。
- 调参脚本（`tune.py`）：复用 `train.py` 的 `build_parser()` 与 `get_dataset_name()`（train.py 的 argparse 与数据集命名均已提取为函数），`--model`/`--beta`/`--anchor-scale` 均为 nargs="+" 列表：基础模型（mlp/cnn/gcn/rnn）无 beta/anchor 维度（每模型 1 组），vib/ceb/dcvib 仅 beta 维度，fgib 为 beta × anchor-scale 两维，总试验数 = 各模型组合数之和；`--parallel`（默认 2）个并发子进程（`sys.executable train.py ...`，cwd 为项目根目录），每组参数在 `--results-dir`（默认 `tune_results/`）下一个独立子文件夹 `{dataset}_{backbone}_{model}_beta_{b:g}_anchor_{a:g}/`（按模型类型省略无维度后缀，基线无 backbone 段，如 `mnist_mlp/`、`mnist_mlp_vib_beta_0.001/`、`mnist_cnn_fgib_beta_0.001_anchor_4/`；含 `model_run{i}.pt`、`train.log`，失败时写 `error.log` 并在表格标 FAILED）；全部完成后解析各日志 `Average over N runs | Test ...` 汇总行生成 HTML 结果表（**每个模型一张表格**，各表带排序下拉框可自由选择排序指标——JS 按行 data 属性排序，Acc/AUC/Pre/Rec/R² 降序、Loss/MAE 升序，默认 Acc（回归 R²）、绿色高亮各模型默认指标最佳行，失败组合沉底标 FAILED；基础模型行 beta/anchor 显示 "-"；文件名单模型为 `{dataset}_{backbone}_{model}_tune_results.html`、多模型为 `{dataset}_{backbone}_{m1+m2}_tune_results.html`）。模型列表无 fgib 且 anchor-scale 列表多于 1 个值时打印警告。
- 评估指标用 sklearn 计算（`roc_auc_score(multi_class="ovr")`、`precision_score`/`recall_score` 均 macro 平均、`mean_absolute_error`/`r2_score`），因此 `evaluate` 需在 no_grad 下收集整个 loader 的预测。
- 数据集：MNIST 在 `./data/MNIST/`；California Housing 缓存在 `./data/california_housing/cal_housing_py3.pkz`，训练时以 `--data-dir` 下的 `california_housing/` 子目录作为 sklearn 的 data_home 加载（数据来自 OpenML 44977，与 sklearn 官方数据一致）。figshare 源在本机被墙，`~/scikit_learn_data/cal_housing_py3.pkz` 留有原始备份，项目内副本丢失时可从备份恢复。IMDb 下载解压缓存在 `./data/imdb/`（下载源 ai.stanford.edu 本机可达；数据管道与词表构建见 `datasets/imdb.py`）。
