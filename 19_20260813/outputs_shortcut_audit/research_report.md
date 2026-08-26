# 信号—噪声可分性与背景捷径确认研究报告

## 摘要

本实验独立于原BCSC–Transformer实验，重新从 `00_Data\original_audio` 截取5761点、576 kHz波形，并以严格匹配队列隔离“信号本身可分”“背景差异捷径”与“BCSC残差泄漏”。历史基准只读取原实验波形结果，不重新解释。

- 正样本原始时间映射：3500/3500。
- 严格匹配对数：3370，最低验收3150对；匹配可行性：不可行（按实际规模继续）。
- 遮罩合成测试：通过；注入模板NCC下降 98.74%。

## 七条件AUROC

| 条件 | AUROC [录音组Bootstrap 95% CI] |
|---|---:|
| 历史基准 | 0.9797 [0.9713, 0.9863] |
| 宽窗口去信号背景 | 0.4976 [0.4910, 0.5045] |
| 严格匹配全波形 | 0.6346 [0.5603, 0.7215] |
| 随机遮罩对照 | 0.5176 [0.5105, 0.5365] |
| BCSC残差 | 0.5017 [0.4937, 0.5099] |
| 严格匹配信号区 | 0.8279 [0.7528, 0.8838] |
| 对称重截取全波形 | 0.8759 [0.7995, 0.9297] |

## 特征基线AUROC

| 条件 | AUROC [录音组Bootstrap 95% CI] |
|---|---:|
| 浅层梯度提升特征 | 0.7256 [0.6911, 0.7654] |
| 逻辑回归特征 | 0.6172 [0.5770, 0.6536] |

![ROC与PR](figures/roc_pr_all.png)
![AUROC森林图](figures/auroc_forest.png)

## 匹配平衡

匹配变量为未峰值归一化带通波形的背景对数RMS、局部SNR、峰均比、谱质心、−10 dB带宽与100–190 kHz高频能量比例。

![匹配前后平衡](figures/matching_balance.png)

## 预注册对比

| 比较 | 配对 | 指标 | 差值 [95% CI] | 显著 |
|---|---|---|---:|---|
| 制作流程影响 | historical vs symmetric_full | auroc | +0.1039 [+0.0461, +0.1820] | 是 |
| 制作流程影响 | historical vs symmetric_full | auprc | +0.0950 [+0.0413, +0.1671] | 是 |
| 制作流程影响 | historical vs symmetric_full | balanced_accuracy | +0.1147 [+0.0611, +0.1770] | 是 |
| 背景提供的额外信息 | matched_full vs matched_signal | auroc | -0.1933 [-0.3049, -0.0457] | 是 |
| 背景提供的额外信息 | matched_full vs matched_signal | auprc | -0.1583 [-0.2702, -0.0268] | 是 |
| 背景提供的额外信息 | matched_full vs matched_signal | balanced_accuracy | -0.1016 [-0.1686, -0.0264] | 是 |
| BCSC残差中的click泄漏 | matched_residual vs matched_background | auroc | +0.0040 [-0.0031, +0.0121] | 否 |
| BCSC残差中的click泄漏 | matched_residual vs matched_background | auprc | +0.0030 [-0.0106, +0.0167] | 否 |
| BCSC残差中的click泄漏 | matched_residual vs matched_background | balanced_accuracy | +0.0194 [+0.0008, +0.0382] | 是 |
| 遮罩与拼接伪影影响 | matched_background vs matched_randommask | auroc | -0.0199 [-0.0399, -0.0096] | 是 |
| 遮罩与拼接伪影影响 | matched_background vs matched_randommask | auprc | -0.0250 [-0.0450, -0.0122] | 是 |
| 遮罩与拼接伪影影响 | matched_background vs matched_randommask | balanced_accuracy | -0.0813 [-0.1089, -0.0496] | 是 |
| SNR及粗统计差异影响 | symmetric_full vs matched_full | auroc | +0.2412 [+0.1482, +0.3248] | 是 |
| SNR及粗统计差异影响 | symmetric_full vs matched_full | auprc | +0.2242 [+0.1352, +0.3145] | 是 |
| SNR及粗统计差异影响 | symmetric_full vs matched_full | balanced_accuracy | +0.1706 [+0.1136, +0.2312] | 是 |
| 清除信号后背景是否仍有类别信息 | matched_background vs 0.5 | auroc | -0.0024 [-0.0088, +0.0040] | 否 |

![SNR分层](figures/snr_strata.png)
![混淆矩阵](figures/confusion_matrices.png)

## 残差click审计

BCSC残差审计特征：直达模板NCC、±0.175 ms中心窗能量、剩余包络峰、残差频谱余弦与残余突出度。

![残差click审计](figures/residual_click_audit.png)

## 结论证据

- 信号区AUROC置信区间下限高于0.5：信号形态本身可分。
- 宽窗口背景AUROC置信区间下限高于0.5：存在背景类别信息。
- BCSC残差显著优于宽窗口背景且残差NCC/中心能量存在类别差异：原0.9265部分来自信号泄漏。
- 严格匹配后全波形显著下降：原任务受到SNR、幅度或频谱不平衡影响。
- 不设置必须下降或必须显著的结果门槛，以上结论按95% CI实际方向陈述。

## 可复现信息

- 配置哈希：`f047e9f9acace257`
- 随机种子：20260813
- Python：`D:\Python_env\toothwhale\python.exe`
- GPU：NVIDIA GeForce RTX 4060
