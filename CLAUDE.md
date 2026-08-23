# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

CVIB：在多个数据集（MNIST / ImageNet-100 特征 / Cora / IMDb / California Housing）上对比 MLP / CNN / GCN / RNN 基线与其三种变分信息瓶颈（VIB）变体（VIB / SCVIB / DCVIB）。代码注释、文档字符串和提交信息均为中文。

## 常用命令

```bash
python train.py                     # 默认训练 MLP（100 epochs, 5 runs）
python train.py --model cnn                     # CNN 基线
python train.py --model vib         # VIB（N(0,I) 先验）
python train.py --model scvib       # SCVIB（随机主路 + 标签条件先验 r(z|y)）
python train.py --model dcvib       # DCVIB（确定性主路 + 旁路瓶颈）
python train.py --backbone cnn --model vib      # CNN 版 VIB（scvib/dcvib 同理）
python train.py --task imagenet100 --model vib      # ImageNet-100 特征分类（MLP 骨干）
python train.py --task cora --model gcn             # Cora 节点分类（GCN 基线）
python train.py --task cora --backbone gnn --model vib     # GNN 版 VIB（scvib/dcvib 同理）
python train.py --task imdb --model rnn             # IMDb 情感分析（RNN 基线）
python train.py --task imdb --backbone rnn --model vib     # RNN 版 VIB（scvib/dcvib 同理）
python train.py --task regression                       # California Housing 回归（MLP 基线）
python train.py --task regression --model vib           # VIB 回归
python train.py --task regression --model scvib         # SCVIB 回归（连续 y 条件先验）
python train.py --task regression --model dcvib         # DCVIB 回归（连续 y 条件先验）
python train.py --model mlp --epochs 20 --batch-size 128 --lr 1e-3 --beta 1e-3
python train.py --model vib --runs 5 --seed 0   # 多 run 取测试集平均指标
```

- 没有测试套件、linter 配置或 requirements.txt。环境：Python 3.12，torch 2.9.0+cu128，torchvision 0.24.0，scikit-learn 1.7.2（CUDA 可用）。
- 完整参数见 `train.py` 的 argparse：`--task`（默认 classification；imagenet100 用预训练 ResNet50 特征、仅 MLP 骨干；cora 仅 GNN 骨干；imdb 仅 RNN 骨干；regression 使用 California Housing、仅 MLP 骨干）、`--backbone`（默认 mlp；`--model` 为 mlp/cnn/gcn/rnn 时被忽略）、`--hidden-dims`（默认 512 256，MLP/GNN 的各层维度；RNN 首元素为词嵌入维度、列表为逐层 LSTM 隐层维度，层数 = 列表长度）、`--dropout`（0.2）、`--z-dim`（256，仅 VIB 系）、`--max-len`（500，仅 IMDb）、`--beta`（KL 权重，1e-3）、`--patience`（早停，10）、`--data-dir`、`--save-path`、`--log-path`。

## 架构

**模型**：四个骨干各有一组同构的四个变体（基线/VIB/SCVIB/DCVIB），接口约定相同（见下）。

- MLP 骨干（`model/mlp/`；`model/__init__.py` 从 `.mlp` 重导出，`train.py` 的 `from model import ...` 接口不变）：
  - `MLP`：纯基线，`forward(x)` 只返回 logits。
  - `VIB`：编码器 h → mu/logvar 头 → 重参数化采样 z → 分类器；先验为固定 N(0, I)。
  - `SCVIB`：同 VIB，但先验改为可学习的 `prior_net`（无激活线性层，分类时 one-hot 标签、`continuous_y=True` 时连续 y → (mu_p, logvar_p)），KL 为 KL(q(z|x) || r(z|y))。
  - `DCVIB`：主路确定性（h 直接分类，z 不参与前向）；旁路从共享编码器 h 引出 mu/logvar 头计算标签条件 KL，梯度只通过共享编码器回传作为正则。forward 只需 `(x, labels)`，无采样。
- CNN 骨干（`model/cnn/`）：同名四个变体 `CNN`/`VIB`/`SCVIB`/`DCVIB`，结构与 MLP 版一一对应，仅把 MLP 隐藏层换成卷积特征提取器（`build_cnn_encoder`：Conv3x3+ReLU+MaxPool2x2 堆叠，默认通道 (32, 64)，28×28 → 64×7×7=3136 展平后接 Linear 到 hidden_dim=256）；瓶颈头与先验部分和 MLP 版完全一致，`reparameterize`/`kl_divergence` 复用自 `model/mlp/utils.py`。因与 MLP 版类名冲突，顶层 `model/` 只重导出 `CNN`，CNN 版 VIB 系列需 `from model.cnn import VIB` 等导入。
- GNN 骨干（`model/gnn/`，Cora 转导式节点分类）：同名四个变体 `GCN`/`VIB`/`SCVIB`/`DCVIB`，`forward` 多收 `adj_norm`（对称归一化邻接，稠密张量）与 `mask` 参数；SCVIB/DCVIB 用 `kl_divergence_masked`（`model/gnn/utils.py`）把 KL 限制在 train mask 节点上，防止标签条件先验接触验证/测试节点标签（标签泄漏）；GNN 版无 `continuous_y`。
- RNN 骨干（`model/rnn/`，IMDb 情感二分类）：同名四个变体 `RNN`/`VIB`/`SCVIB`/`DCVIB`，瓶颈头与先验部分和 MLP 版完全一致，仅编码器换成 LSTM 文本编码器（`build_rnn_encoder`：Embedding(vocab_size, hidden_dims[0], padding_idx=0) → 多层 LSTM（层数与各层隐层维度由 `--hidden-dims` 列表决定，默认 (512, 256) 两层，逐层 feed 前一层的 packed 输出，除末层外每层输出后接 Dropout）→ 取末层 `h_n[-1]` → Dropout → h (B, hidden_dims[-1])）。输入为 (B, L) 的 token id 张量（pad_idx=0 后填充），长度由模型内部 `(x != pad_idx).sum(1)` 计算后 pack（lengths 需 `.cpu()`）；RNN 版 VIB 系列同样需 `from model.rnn import VIB` 等导入。

**统一前向接口**（`train.py:run_model`）：所有模型返回 `(logits, kl)`，基线返回 `kl=None`；VIB/SCVIB 额外接收 `stochastic` 标志（训练 True 采样，评估 False 用 z=mu），GNN 版另传 `adj_norm`/`mask`。损失 = CrossEntropy + beta·KL（kl 为 None 时省略）。

**关键约定**：

- 所有 VIB 系模型的 logvar 头和 `prior_net` 做置零初始化，使训练起点 KL≈0、sigma=1 —— 修改模型初始化时需保持这一点。
- `model/mlp/utils.py` 提供共享工具：`flatten`（(B,C,H,W)→(B,C·H·W)）、`build_hidden_layers`（Linear+ReLU+Dropout）、`reparameterize`、`kl_divergence`（对角高斯闭式解，logvar clamp 到 [-10, 10]，逐样本求和后 batch 平均）；CNN/RNN 模型跨包复用其中的 `reparameterize`/`kl_divergence`。
- SCVIB/DCVIB 在 `labels is None` 时返回 `kl=None`（评估或推理时无标签），训练路径必须传入 labels。
- 回归时 SCVIB/DCVIB 用 `continuous_y=True`、`num_classes=1`（输出 (B,1)）；先验条件在归一化后的连续 y 上（`labels.float().unsqueeze(-1)` → `prior_net`），仍保持置零初始化。

**训练流程**（`train.py`）：

- 数据（分类）：MNIST，训练集 60k 切出 10k 验证（`random_split`，generator 固定 seed 42）。
- 数据（imagenet100）：预训练 ResNet50 特征的 npz，60/20/20 分层划分 + StandardScaler（仅训练集拟合），仅 MLP 骨干。
- 数据（cora）：全图批转导训练，无 DataLoader（`datasets/cora.py`，划分固定 seed 42）。
- 数据（imdb）：斯坦福官方 aclImdb_v1.tar.gz（官方 train/test 各 25k 已标注影评）合并后 60/20/20 分层划分（seed 42）→ 30k/10k/10k；词表仅从训练集构建（pad_idx=0、unk_idx=1 保留位，其余按词频编号，约 9 万词）；去 HTML 标签后纯字母分词、截断到 `--max-len`（500）；collate 用 `pad_sequence` 后填充。二分类评估时 `evaluate` 把 (n,2) 概率取第 1 列再算 AUC（sklearn 二分类要求 (n,)）。
- 数据（回归）：`fetch_california_housing`（20640 样本、8 特征），60/20/20 划分（seed 42），特征用 StandardScaler 标准化、目标用 MinMaxScaler 归一化到 [0,1]（均仅在训练集上拟合）。损失 MSELoss 在归一化目标上计算，MAE 用 y_scaler 逆归一化回原始单位（十万美元；R² 与量纲无关），按验证 R² 早停。CNN 骨干不支持回归（argparse 直接报错）。
- 每 run 使用 seed+i-1；分类按验证集宏平均 AUC、回归按验证 R² 早停（patience=10），保存最优 state dict 到 `output/<name>/<name>_run{i}.pt`，训练日志写入 `output/<name>/train_<name>.log`（logging 同时输出到控制台和文件）。`<name>` 按 `{dataset}_{backbone}_{model}` 命名（基线为 `{dataset}_{model}`），dataset 为 `mnist`/`imagenet100`/`cora`/`imdb`/`california`，如 `output/mnist_mlp_vib/`、`output/mnist_cnn_vib/`、`output/cora_gnn_vib/`、`output/imdb_rnn_vib/`、`output/california_mlp_scvib/`——每个 任务×骨干×模型 组合对应唯一目录。
- 早停结束后加载最优 checkpoint 在测试集评估（分类报告 Loss/Acc/宏平均 AUC/Precision/Recall，回归报告 Loss/MAE/R²，其中 Loss 含 beta·KL 项、MAE/R² 为纯预测指标），最后对全部 runs 取均值±标准差。
- 评估指标用 sklearn 计算（`roc_auc_score(multi_class="ovr")`、`precision_score`/`recall_score` 均 macro 平均、`mean_absolute_error`/`r2_score`），因此 `evaluate` 需在 no_grad 下收集整个 loader 的预测。
- 数据集：MNIST 在 `./data/MNIST/`；California Housing 缓存在 `./data/california_housing/cal_housing_py3.pkz`，训练时以 `--data-dir` 下的 `california_housing/` 子目录作为 sklearn 的 data_home 加载（数据来自 OpenML 44977，与 sklearn 官方数据一致）。figshare 源在本机被墙，`~/scikit_learn_data/cal_housing_py3.pkz` 留有原始备份，项目内副本丢失时可从备份恢复。IMDb 下载解压缓存在 `./data/imdb/`（下载源 ai.stanford.edu 本机可达；数据管道与词表构建见 `datasets/imdb.py`）。
