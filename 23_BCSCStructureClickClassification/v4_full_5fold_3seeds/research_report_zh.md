# 实验23-v4：完整五折、三随机种子确认

## 设计

C0--C5均完成五折，随机种子为20260826、20260827、20260828，共90个模型折。主统计以每个样本三个种子预测概率的均值计算，并采用34个录音组配对Bootstrap。逐种子五折均值另行保留。

## 三种子五折汇总

| condition | seeds | low_snr_auroc_mean | low_snr_auroc_seed_std | auroc_mean | auroc_seed_std | auprc_mean | auprc_seed_std | balanced_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_waveform | 3 | 0.7676 | 0.0032 | 0.8366 | 0.0057 | 0.8332 | 0.0015 | 0.7550 |
| C1_full_zero | 3 | 0.7665 | 0.0180 | 0.8395 | 0.0151 | 0.8416 | 0.0217 | 0.7591 |
| C2_v2_scalar | 3 | 0.7870 | 0.0066 | 0.8567 | 0.0073 | 0.8605 | 0.0079 | 0.7677 |
| C3_reliability | 3 | 0.7864 | 0.0065 | 0.8556 | 0.0079 | 0.8581 | 0.0076 | 0.7690 |
| C4_local_gate | 3 | 0.7842 | 0.0151 | 0.8531 | 0.0132 | 0.8501 | 0.0213 | 0.7670 |
| C5_structure_dropout | 3 | 0.7822 | 0.0134 | 0.8519 | 0.0129 | 0.8523 | 0.0181 | 0.7666 |

## 顺序组件消融

| step | model_folds | mean_delta_low_snr_auroc | std_delta_low_snr_auroc | positive_folds | direction |
| --- | --- | --- | --- | --- | --- |
| reliability | 15 | -0.0006 | 0.0119 | 8 | 结果不稳定 |
| local_gate | 15 | -0.0022 | 0.0310 | 7 | 结果不稳定 |
| structure_dropout | 15 | -0.0020 | 0.0109 | 7 | 结果不稳定 |

## 三种子平均概率的配对Bootstrap

| left | right | endpoint | point_difference | ci_low | ci_high | bootstrap_runs_requested | bootstrap_runs_valid | supports_positive_gain | probability_aggregation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C5_structure_dropout | C0_waveform | low_snr_auroc | 0.0107 | -0.0031 | 0.0250 | 2000 | 2000 | False | three_seed_mean |
| C5_structure_dropout | C0_waveform | auroc | 0.0137 | 0.0024 | 0.0244 | 2000 | 2000 | True | three_seed_mean |
| C5_structure_dropout | C1_full_zero | low_snr_auroc | 0.0099 | -0.0038 | 0.0217 | 2000 | 2000 | False | three_seed_mean |
| C5_structure_dropout | C1_full_zero | auroc | 0.0100 | 0.0021 | 0.0180 | 2000 | 2000 | True | three_seed_mean |
| C3_reliability | C2_v2_scalar | low_snr_auroc | 0.0023 | -0.0040 | 0.0083 | 2000 | 2000 | False | three_seed_mean |
| C3_reliability | C2_v2_scalar | auroc | 0.0013 | -0.0028 | 0.0051 | 2000 | 2000 | False | three_seed_mean |
| C4_local_gate | C3_reliability | low_snr_auroc | -0.0002 | -0.0084 | 0.0074 | 2000 | 2000 | False | three_seed_mean |
| C4_local_gate | C3_reliability | auroc | -0.0004 | -0.0069 | 0.0055 | 2000 | 2000 | False | three_seed_mean |
| C5_structure_dropout | C4_local_gate | low_snr_auroc | 0.0016 | -0.0043 | 0.0070 | 2000 | 2000 | False | three_seed_mean |
| C5_structure_dropout | C4_local_gate | auroc | 0.0014 | -0.0025 | 0.0053 | 2000 | 2000 | False | three_seed_mean |

## 核心结论

至少一个C5核心比较的主终点95%置信区间下限不大于0；因此只能报告提升趋势，不能声称BCSC结构增益成立。

说明：此Bootstrap反映录音组重采样不确定性，三种子不被当作独立Bootstrap单元；种子间离散度见`full_3seed_seed_summary.csv`。
