# Chico：第22实验 JASA 论文蓝图与证据整理计划

## 一、目标与论文定位

整理后的问题是：如何将第22实验从 v3–v7 的探索过程收敛为一篇面向 JASA 的论文，并用最少的补充实验支撑“无需 MIM 和 CCC，异构网络结构与物理约束本身即可形成有效因素分离”的结论。

- 目标期刊：The Journal of the Acoustical Society of America，Regular Research Article，优先定位 Animal Bioacoustics。
- 论文任务：中华白海豚回声定位 click 与困难噪声的二分类检测。
- 主方法：保留异构 \(X/H\) 编码器、解码器、卷积重建和三个物理先验；删除 CLUB、Q 网络、MIM 和 CCC。
- 术语采用“convolutional factorization”“source-like component”“response-like component”和“latent dependence”，不把 CCA/KCCA 直接等同于因果解耦，也不声称 \(\hat h\) 是已验证的真实传播信道。
- JASA 官方约束将在蓝图中注明：英文标题不超过约17词、摘要不超过200词，正文强调完整性和可复核性。[JASA投稿说明](https://pubs.aip.org/asa/jasa/pages/manuscript)

建议主标题：

> **Physics-Structured Heterogeneous Convolutional Factorization for Indo-Pacific Humpback Dolphin Echolocation Click Detection**

中文译题：

> **用于中华白海豚回声定位 click 检测的物理结构化异构卷积分解**

本任务按以下技能顺序组织：

1. `ccf-humanization`：控制证据边界，避免把工程探索写成防御性论文叙事。
2. `ccf-paper-writer`：建立 JASA 论文主线和段落级章节框架。
3. `ccf-experiment-designer`：建立主张—实验—指标对应关系及最小补实验方案。
4. `ccf-visual-composer`：设计完整网络 SVG、结果图和重建图。
5. `ccf-pipeline-orchestrator`：区分已有证据、待补证据和成稿门槛。

## 二、最终 Markdown 文档

生成：

[JASA_experiment22_paper_blueprint_zh.md](D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\JASA_experiment22_paper_blueprint_zh.md)

文档采用中文论文蓝图，包含英文标题、英文摘要写作骨架和以下内容：

1. **一页式论文判断**
   - 当前最可靠的科学结论。
   - 哪些结论已有充分支持。
   - 哪些只能作为单种子探索结果。
   - 哪些表述必须避免。

2. **v3–v7 实验复盘**
   - v3：多种子证明 MIM 未改善分类，并使 CCA/KCCA 略升。
   - v4：正交约束未明显降低潜变量依赖。
   - v5：物理损失实际占比过低，不能据此支持物理约束有效。
   - v6：异构结构带来最大的 CCA/KCCA 降幅，物理先验主要改善 \(X/H\) 形态。
   - v7：去掉 MIM 后，CCC 未改善 AUROC/AUPRC，且提高正类 CCA/KCCA。
   - 明确区分不同架构和协议，禁止将其拼接成不成立的直接性能排名。

3. **统一论文主线**
   - 科学起点：接收 click 同时包含发声结构、传播效应和背景干扰。
   - 核心问题：显式互信息抑制是否是形成可用因素分离的必要条件。
   - 核心观察：异构分支的不同归纳偏置、卷积重建和弱物理约束已经限制了解空间。
   - 方法逻辑：让 \(X\) 分支表达短时、带限的 click-like 成分，让 \(H\) 分支表达带固定直达项的稀疏响应。
   - 有界结论：该设计在当前来源隔离数据上保持检测性能并降低潜变量统计依赖，但不证明唯一物理解卷积。

4. **段落级论文框架**
   - Abstract
   - Introduction
   - Related work：海豚 click 检测、物理引导声学学习、盲卷积因素分解。
   - Signal formulation and task definition
   - Heterogeneous convolutional factorization
   - Experimental protocol
   - Detection performance
   - Factor dependence and physical morphology
   - Reconstruction analysis
   - Discussion and limitations
   - Conclusion
   - Data availability、Author declarations 和 Supplementary Material 占位。

## 三、方法、图表与证据表达

### 3.1 最终方法

文档给出唯一主体损失：

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{recon}}
+
\mathcal L_{\mathrm{BCE}}
+
\lambda_H\mathcal L_H
+
\lambda_C\mathcal L_{\mathrm{compact}}
+
\lambda_B\mathcal L_{\mathrm{band}},
\]

其中

\[
\mathcal L_{\mathrm{recon}}
=
\frac{1}{B}\sum_i
\frac{\lVert \hat r_i-r_i\rVert_2}
{\lVert r_i\rVert_2+\epsilon},
\qquad
\hat r=\hat x * \hat h.
\]

固定权重采用现有校准值：

\[
(\lambda_H,\lambda_C,\lambda_B)
=
(0.3845476,\ 0.01394295,\ 0.01368216).
\]

正文不出现 CLUB、Q、MIM 或 CCC；它们只作为“未被保留的显式正则化消融”出现在实验分析中。

### 3.2 完整网络框图

生成纯矢量、可编辑 SVG：

[jasa_experiment22_no_mim_architecture.svg](D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\figures\jasa_experiment22_no_mim_architecture.svg)

图中完整展示：

- 输入 \(r\in\mathbb R^{5761}\)。
- \(X\) Encoder：
  - AvgPool anti-alias prefilter；
  - 四个 stride-2 Residual1d blocks；
  - 通道数 \(1\rightarrow16\rightarrow32\rightarrow64\rightarrow64\)；
  - global mean/max pooling、Linear \(128\rightarrow64\)、LayerNorm；
  - 输出 \(z_x\in\mathbb R^{64}\)。
- \(X\) Decoder：
  - Linear \(64\rightarrow32\times181\)；
  - 插值长度 \(181\rightarrow361\rightarrow721\rightarrow1441\rightarrow2881\rightarrow5761\)；
  - Residual blocks \(32\rightarrow24\rightarrow16\rightarrow8\rightarrow8\rightarrow8\)；
  - Conv1d、tanh、Tukey window；
  - 输出 \(\hat x\in\mathbb R^{5761}\)。
- \(H\) Encoder：
  - 四个 dilation \(1,2,4,8\) 的不降采样 Residual1d blocks；
  - Linear \(5761\rightarrow64\)、LayerNorm；
  - 输出 \(z_h\in\mathbb R^{64}\)。
- \(H\) Decoder：
  - Linear \(64\rightarrow128\rightarrow2880\)；
  - SiLU、Soft-Threshold、幅度限制；
  - 拼接固定直达项 \(h[0]=1\)；
  - 输出 \(\hat h\in\mathbb R^{2881}\)。
- 因果 FFT 卷积得到 \(\hat r\)。
- 分类头仅读取 \(z_x\)，输出 click 概率。
- 实线表示推理数据流，虚线表示训练损失；三个物理先验注明仅作用于真实正类。
- 图中不出现 MIM、CCC、CLUB 或 Q。
- 采用 Okabe–Ito 色板并验证灰度打印、双栏宽度可读性和 SVG 文本可编辑性。

### 3.3 主体表格和图形

正文规划以下证据：

- **表1：数据和来源隔离协议**
  - 样本量、正负类、录音来源、SNR范围、采样率、信号长度、五折划分和OOF定义。

- **表2：主体检测性能**
  - AUROC、AUPRC、平衡准确率、F1、敏感度、特异度。
  - 按录音来源配对 Bootstrap 给出95%区间。
  - 单种子 v7 数值标记为“现有探索证据”，多种子单元格在补实验前标为 TBD。

- **表3：必要消融**
  - X-only BCE 分类器。
  - 同构但不共享的双编码器。
  - 异构结构、无物理先验。
  - 异构结构加三个物理先验，即最终方法。
  - MIM 和 CCC 放在独立的“显式正则化未被采用”分组，避免暗示它们属于最终模型。

- **图2：实验22证据演进图**
  - 用时间轴展示 v3–v7 每次实验回答的问题和结论，不把异构协议的分数画成连续性能提升曲线。

- **图3：CCA/KCCA典型相关谱**
  - 展示前十个典型相关，而非只给均值。
  - 正类为主要分析，全体样本为补充。
  - 配对差值和录音来源 Bootstrap 区间同时呈现。
  - 图注明确：低相关表示较弱统计依赖，不等价于可辨识的物理解耦。

- **图4：检测性能—潜变量依赖联合图**
  - 横轴为正类 KCCA 或 CCA，纵轴为 AUROC/AUPRC。
  - 显示最终无 MIM/CCC 条件能够维持分类性能，同时具有较低的潜变量依赖。
  - 不连接不同实验协议的点，避免制造虚假演进趋势。

- **表4：\(X/H\) 物理形态诊断**
  - \(H\) 尾部非零率、Hoyer sparsity、尾部能量、能量加权时延。
  - \(X\) 中心能量比例、边界能量、20–190 kHz带外能量。
  - 这些指标用于“结构合理性”，不作为真实 \(h\) 恢复误差。

### 3.4 重建任务评估

重建部分分为三个证据层级：

1. **接收信号重建**
   - 在完整 OOF 上计算 relative \(L_2\)、NMSE、Pearson相关、对数谱距离。
   - 按 SNR 和正负类别分层。
   - 当前训练历史中平均重建损失约0.964，只能作为训练诊断，不能替代 OOF 重建结果。

2. **source-like 成分**
   - 对真实正类，从数据构造记录恢复经过相同对齐与幅度归一化的干净 click 贡献。
   - 计算尺度不变相关、SI-SDR、峰值时间误差、包络相关和带内谱距离。
   - 明确这种比较评价“与已知干净 click 的一致性”，不证明 \(\hat x\) 是唯一真实声源。

3. **response-like 成分**
   - 当前无真实 \(h\)，不报告不存在的恢复误差。
   - 只报告稀疏性、直达项、延迟和尾部衰减等形态指标。
   - 不把合成通道实验列为当前必做项。

重建图采用预注册式选样而非随机挑选最好结果：

- 低、中、高 SNR代表样本；
- OOF重建误差中位样本；
- 一个高误差失败样本；
- 每个样本依次显示 \(r\)、\(\hat r\)、残差、干净click参考、\(\hat x\) 和 \(\hat h\)；
- 同时给出对应数值指标，理性说明相位、幅度或局部细节不能完全恢复。

## 四、最小补实验方案

不要求外部数据，也不要求合成传播通道。主体补充限制为：

1. **最终方法多种子稳定性**
   - 将当前无 MIM/CCC 方法从1个种子扩展到3个种子、每种子5折。
   - 现有种子保留，因此新增10个正式模型。
   - 若三种子置信区间稳定，不再机械扩展到五种子。

2. **三个匹配对照**
   - X-only分类器：相同 \(X\) Encoder 和分类头，只用 BCE。
   - 同构双分支：两个独立但结构相同的编码器，用于区分“参数独立”与“结构异构”。
   - 异构无先验：保留重建和BCE，删除三个物理先验。
   - 三个条件均使用相同的3种子、5折、初始化映射和样本顺序。

3. **无需重新训练的分析**
   - 完整OOF重建指标。
   - 正类干净click一致性。
   - SNR分层敏感度和噪声类别假阳性率。
   - 录音来源级Bootstrap。
   - CCA/KCCA全谱及物理形态指标。

4. **不追加的实验**
   - 不再搜索新的显式解耦损失。
   - 不扩大 MIM/CCC 超参数网格。
   - 不要求外部海域数据或真实信道真值。
   - MIM/CCC沿用已有证据，仅作为方法筛选历史和负消融，不作普遍性结论。

如果匹配的 X-only 基线与最终方法分类性能相当，论文主张将收敛为“结构化因素分解在不牺牲检测性能的情况下提供更可解释的表征”，而不是声称性能显著提升。

## 五、验证与完成标准

- 所有现有数字从 v3–v7 原始 CSV/JSON 重新核对，禁止从旧报告手工抄录后直接使用。
- 不同架构、种子数量或训练协议的结果分表展示。
- 所有未完成实验均标记 `TBD`，不生成预测值。
- SVG逐层核对代码中的模块、长度、通道数、卷积方向和训练/推理路径。
- 网络图在 JASA 双栏宽度和灰度打印下仍可辨认。
- 最终方法及主体损失中不存在 MIM、CCC、CLUB 或 Q。
- 重建章节同时展示代表结果和失败结果。
- \(\hat h\) 始终表述为 response-like factor，不声称为已验证的真实传播通道。
- 文档最终形成“科学问题—结构性洞察—方法—分类证据—因素依赖—重建边界—最小补实验”的闭合论文脉络。
