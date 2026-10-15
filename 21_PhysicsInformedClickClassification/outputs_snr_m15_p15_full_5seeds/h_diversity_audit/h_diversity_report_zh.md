# 既有模型信道响应 h 多样性审计报告

## 审计范围

- 模式：`full`。
- 只读检查点目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_full_5seeds`。
- 审计输出目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_full_5seeds\h_diversity_audit`。
- 已核验检查点：50 个；数据 SHA-256：`70204b5ac4484bb4a08a684e29083390cec974502061c45ba48ea2ef9ac130ed`。
- 数据固定清单：640 个 Click train、475 个多 Click train、3335 个可比较 Click、3285 个完整匹配（98.5%）。
- 响应不跨随机种子直接比较；距离和交换误差均先在单个检查点内计算。

## 主检验

### Recon + BCE

| 指标 | 估计值 | 95% CI | Holm校正p | 结果 |
|---|---:|---:|---:|---|
| M_train：不同 train 距离 − 同 train 距离 | -0.001857 | [-0.003605, 0.000195] | 1 | 未通过 |
| M_noise：不同 train 距离 − 混合/干净距离 | 0.008016 | [0.007579, 0.011421] | 0.00149925 | 通过 |
| M_swap：不同 train 交换误差 − 同 train 交换误差 | -0.002524 | [-0.004832, -0.001598] | 1 | 未通过 |
### Recon + Physics + BCE

| 指标 | 估计值 | 95% CI | Holm校正p | 结果 |
|---|---:|---:|---:|---|
| M_train：不同 train 距离 − 同 train 距离 | -0.003230 | [-0.003633, -0.001190] | 1 | 未通过 |
| M_noise：不同 train 距离 − 混合/干净距离 | 0.005464 | [0.002892, 0.011894] | 0.00149925 | 通过 |
| M_swap：不同 train 交换误差 − 同 train 交换误差 | -0.000568 | [-0.002993, -0.000519] | 1 | 未通过 |

置信区间按“随机种子×录音来源”分层 Bootstrap；单侧置换检验在录音、背景类型和 5 dB SNR 层内破坏差值方向，并对每个条件的三个主检验执行 Holm 校正。

## 混杂探针

| 条件 | 探针 | 测试估计 | 95% CI | 结果 |
|---|---|---:|---:|---|
| Recon + BCE | snr_r2 | 0.260011 | [0.175697, 0.309883] | 未通过 |
| Recon + BCE | background_balanced_accuracy | 0.582971 | [0.568471, 0.597081] | 未通过 |
| Recon + Physics + BCE | snr_r2 | 0.254064 | [0.121156, 0.332962] | 未通过 |
| Recon + Physics + BCE | background_balanced_accuracy | 0.606663 | [0.581915, 0.629003] | 未通过 |

Ridge 和多项 Logistic 探针只读取冻结响应；训练折拟合、验证折选择正则、测试折报告。

## Physics 项归因

| 指标 | Physics − Recon+BCE | 95%配对区间 | 区间下限>0 |
|---|---:|---:|---|
| M_train：不同 train 距离 − 同 train 距离 | -0.001373 | [-0.002800, +0.001013] | 否 |
| M_noise：不同 train 距离 − 混合/干净距离 | -0.002552 | [-0.007344, +0.002967] | 否 |
| M_swap：不同 train 交换误差 − 同 train 交换误差 | +0.001957 | [-0.000697, +0.003690] | 否 |

## 验收结论

**Physics响应未通过全部结构化多样性验收规则**

该结论严格限于“响应具有结构化多样性”。本审计不宣称估计响应 h 等于真实水声传播信道，也不使用测试结果修改阈值、配对规则或模型。

## 图件与明细

- `figures/h_distance_hierarchy.png`：同 train、混合/干净、不同 train 距离层级。
- `figures/h_swap_reconstruction.png`：信道交换重建误差。
- `figures/h_collapse_effective_rank.png`：尾部能量与有效秩。
- `figures/h_representative_edp.png`：代表性归一化能量时延分布。
- `metrics/`：样本诊断、配对指标、Bootstrap/置换结果和混杂探针。
