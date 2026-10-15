# 实验23-v4：BCSC可靠性门控最小消融实验报告

## 实际运行证据

核心三条件均完成五折，核心OOF共21,000行。主终点是低SNR AUROC；Bootstrap以34个录音组为配对重采样单元。

| condition | mean_low_snr_auroc | mean_auroc | mean_auprc | mean_parameters |
| --- | --- | --- | --- | --- |
| C0_waveform | 0.7713 | 0.8418 | 0.8320 | 39873.0000 |
| C1_full_zero | 0.7874 | 0.8568 | 0.8653 | 58631.0000 |
| C5_structure_dropout | 0.7967 | 0.8663 | 0.8722 | 58631.0000 |

## 核心配对Bootstrap

| left | right | endpoint | point_difference | ci_low | ci_high | bootstrap_runs_requested | bootstrap_runs_valid | supports_positive_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C5_structure_dropout | C0_waveform | low_snr_auroc | 0.0373 | 0.0040 | 0.0720 | 2000 | 2000 | True |
| C5_structure_dropout | C0_waveform | auroc | 0.0358 | 0.0067 | 0.0668 | 2000 | 2000 | True |
| C5_structure_dropout | C1_full_zero | low_snr_auroc | 0.0055 | -0.0050 | 0.0172 | 2000 | 2000 | False |
| C5_structure_dropout | C1_full_zero | auroc | 0.0057 | -0.0021 | 0.0154 | 2000 | 2000 | False |

## 核心结论

至少一个主要比较的95%置信区间下限不大于0。因此只能报告提升趋势，不能声称BCSC结构增益成立。

## 阶段A最小消融筛选

中间组件仅在fold 0和fold 2运行，用于方向诊断；不报告统计显著性。

| step | fold | from_condition | to_condition | from_low_snr_auroc | to_low_snr_auroc | delta_low_snr_auroc | from_auroc | to_auroc | delta_auroc | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reliability | 0 | C2_v2_scalar | C3_reliability | 0.7761 | 0.7783 | 0.0022 | 0.8499 | 0.8529 | 0.0030 | nan |
| reliability | 2 | C2_v2_scalar | C3_reliability | 0.8093 | 0.8139 | 0.0046 | 0.8839 | 0.8862 | 0.0023 | nan |
| local_gate | 0 | C3_reliability | C4_local_gate | 0.7783 | 0.7551 | -0.0233 | 0.8529 | 0.8371 | -0.0158 | nan |
| local_gate | 2 | C3_reliability | C4_local_gate | 0.8139 | 0.8434 | 0.0295 | 0.8862 | 0.8987 | 0.0125 | nan |
| structure_dropout | 0 | C4_local_gate | C5_structure_dropout | 0.7551 | 0.7369 | -0.0182 | 0.8371 | 0.8323 | -0.0047 | nan |
| structure_dropout | 2 | C4_local_gate | C5_structure_dropout | 0.8434 | 0.8299 | -0.0135 | 0.8987 | 0.8911 | -0.0076 | nan |
| reliability | summary | nan | nan | nan | nan | 0.0034 | nan | nan | nan | 一致的改善方向 |
| local_gate | summary | nan | nan | nan | nan | 0.0031 | nan | nan | nan | 效果不稳定 |
| structure_dropout | summary | nan | nan | nan | nan | -0.0159 | nan | nan | nan | 当前配置下可能无效 |

## 证据等级

C0/C1/C5：完整五折核心证据。C2/C3/C4：两个fold的最小消融证据，仅可表述为可能有效、结果不稳定或当前配置下可能无效。
