# GPB/OPB/EPB 文献调研结论

检索日期：2026-09-05。检索接口：OpenAlex Works API。脚本执行了 40 个短语查询，
合并得到 723 条候选记录；排序使用 `relevance_score`，并按标题、关键词和摘要做了二次
启发式排序。完整的原始返回和候选列表见 `openalex_raw.json`、`openalex_results.json`、
`openalex_results.csv`，检索脚本见 `search_openalex.py`。

## 结论

在本次 OpenAlex 检索和 `paper/` 中已有参考文献范围内，没有发现一篇论文同时具备本文的
全部关键组合：

1. 以 CEB/条件信息瓶颈的样本级条件高斯 KL 作为压缩项；
2. 将类别条件先验的均值限制为由构造得到的正交类别框架（OPB）；
3. 将连续标签先验限制为保持标签距离的等距轴（EPB）；
4. 使用与该先验几何绑定、没有额外自由分类器/回归头的预测规则；
5. 给出从先验几何、KL 对齐到后验类别分离的统一理论和实验验证。

因此，较稳妥的 novelty 表述是：

> “To the best of our OpenAlex search, we found no prior work that combines a
> conditional entropy-bottleneck objective with a constructed orthogonal
> class-prior/isometric regression geometry and geometry-tied prediction heads
> in one framework.”

不要写成“绝对首次”或“没有任何相关工作”。OpenAlex 不是完整的引文数据库，检索结果
还包含预印本、同一工作的重复记录和少量与关键词仅有表面匹配的论文；投稿前仍应核对
标题相近论文的全文、Google Scholar/Semantic Scholar 和最新会议论文。

## 最相关的基础工作

| 文章题目 | 与本文的关系 | 是否直接先例 |
|---|---|---|
| *The Information Bottleneck Method* | IB 原始目标和信息平面 | 否，理论基础 |
| *Deep Variational Information Bottleneck* | 用神经网络和重参数化实现 VIB，固定边缘先验 | 否，无条件几何先验 |
| *The Conditional Entropy Bottleneck* | 用标签条件后验/先验估计残差信息，并讨论鲁棒泛化 | 否，是本文直接出发点 |
| *CEB Improves Model Robustness* | CEB 的大规模分类和对抗鲁棒性实验 | 否，是鲁棒性比较基线 |
| *Caveats for Information Bottleneck in Deterministic Scenarios* | 说明确定性标签下 IB/CEB 扫描可能退化 | 否，是本文关于压缩旋钮的理论背景 |

## 部分重叠：几何与信息瓶颈

- *Explainable Recommender With Geometric Information Bottleneck*（2024）：在变分推荐
  系统中学习几何先验并用于解释；摘要没有显示条件类别高斯、正交类别框架、等距回归轴
  或几何绑定预测头。因此是“几何先验 + IB”近邻，不是本文的统一方法。
- *Geometry as a Missing Axis of Representation Quality: The Variational Geometric
  Information Bottleneck under Data Scarcity*（2025）：把曲率和内在维度作为 VIB 的几何
  惩罚项，研究数据稀缺下的泛化；没有把标签条件先验限制为 OPB/EPB 几何。
- *A Geometric Information Bottleneck for Activation Steering*（2026）：用几何 IB 观点
  选择大语言模型激活层，并保持正交子空间；任务是激活干预，不是条件 KL 瓶颈或类别/回归
  标签先验。
- *Geometric and Information Compression of Representations in Deep Learning*（2026）：
  研究 CEB 中信息压缩与类内几何压缩的关系；它是分析性工作，没有提出本文的构造先验。
- *Multimodal Conditional Information Bottleneck for Generalizable AI-Generated Image
  Detection*（2025）：使用条件瓶颈和动态文本正交化；正交化作用于文本特征，未见 OPB 的
  条件高斯先验、全类别正交框架或几何绑定分类器。

## 部分重叠：分类几何、正交原型和 ETF

- *An Orthogonal Classifier for Improving the Adversarial Robustness of Neural Networks*
  （2021/2022）：构造正交分类器权重以提升对抗鲁棒性。它与 OPB 的正交分类几何相关，
  但不是 CEB 条件先验，也没有通过 KL 将后验拉向正交类先验。
- *Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training*
  （2020）和 *A Geometric Analysis of Neural Collapse with Unconstrained Features*
  （2021）：分析类均值/分类器趋向 ETF 或相关几何，是本文应引用的理论背景，不是
  “由条件先验构造几何”的方法。
- *Towards Understanding Neural Collapse in Supervised Contrastive Learning with the
  Information Bottleneck Method*（2023）：把神经坍缩和 IB 联系起来，并讨论 simplex ETF；
  没有 CEB 条件高斯 KL、OPB 的逐步正交先验或 EPB 回归轴。
- *Early Exit with Disentangled Representation and Equiangular Tight Frame*（2023）、
  *No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic
  and Fixed Classifier*（2023）以及 *Inducing Neural Collapse to a Fixed Hierarchy-Aware
  Frame for Reducing Mistake Severity*（2023）：使用固定/非参数 ETF 或框架分类器，但
  研究目标和损失不同，不能作为本文完整方法的先例。
- *Equidistant Prototypes Embedding for Single Sample Based Face Recognition with Generic
  Learning and Incremental Learning*（2014）：使用等距原型做分类表示，属于原型分类
  先例，不涉及信息瓶颈条件先验或回归轴。

## 部分重叠：回归几何

- *Deep Isometric Maps*（2022）：学习保持流形距离的深度映射，但属于无监督非线性降维，
  不是标签条件高斯瓶颈。
- *Discriminative Noise Robust Sparse Orthogonal Label Regression-Based Domain Adaptation*
  （2021/2023）：使用正交标签回归嵌入做域适配，和“标签几何约束”相关，但没有 CEB/GPB
  的条件 KL 先验和几何绑定回归头。
- *Distance Preserving Machine Learning for Uncertainty Aware Accelerator Capacitance
  Predictions*（2024）：关注距离保持和不确定性回归中的特征提取，未提出 EPB 的
  `rho * standardized_label * unit_axis` 条件先验。
- *Cauchy-Schwarz Divergence Information Bottleneck for Regression*（2024）：是回归信息
  瓶颈近邻，但摘要未显示等距标签先验或几何绑定预测规则。

## 部分重叠：信息瓶颈与对抗鲁棒性

- *The Conditional Entropy Bottleneck*、*CEB Improves Model Robustness*：本文最直接的
  CEB/鲁棒性基线。
- *Revisiting Hilbert-Schmidt Information Bottleneck for Adversarial Robustness*（2021）、
  *Adversarial Information Bottleneck*（2022）、*Improving the Adversarial Robustness of
  NLP Models by Information Bottleneck*（2022）、*Causal Information Bottleneck Boosts
  Adversarial Robustness of Deep Neural Network*（2022）：分别用 HSIC、对抗 IB、NLP 特征
  筛选或因果分解提升鲁棒性，没有本文的几何条件先验构造。
- *RobustBench: A Standardized Adversarial Robustness Benchmark*（2020）：是攻击与评估
  协议参考，不是表示学习方法。

## 建议在 related work 中明确的边界

本文应把贡献拆成“组合创新”而不是声称每个零件都首次：

1. CEB/条件残差瓶颈来自 Fischer 的工作；
2. 正交分类器、ETF 和 neural collapse 已有大量先例；
3. 等距映射、正交标签回归也有独立文献；
4. 本文的差异在于把这些几何约束放进**条件先验的参数化**，用单个 KL 连接后验，
   并把预测头绑定到同一先验几何，同时给出分类 OPB 和回归 EPB 的统一分析。

## 可复核性说明

`search_openalex.py` 不保存或输出 API 密钥，只从环境变量 `OPENALEX_API_KEY` 读取。
`openalex_raw.json` 是本次查询的原始候选快照；候选标题不能替代全文阅读，尤其是标题中
含有 “geometric”、“orthogonal”、“information bottleneck” 的文章，必须检查其先验
分布、训练目标和预测头是否真的与本文一致。
