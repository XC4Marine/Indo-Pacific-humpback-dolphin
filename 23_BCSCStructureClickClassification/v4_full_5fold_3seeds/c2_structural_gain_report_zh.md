# C2结构增益的三种子配对Bootstrap检验

C2保留原始BCSC结构图并经Path-CNN与波形特征全局融合。C0仅使用波形；C1保留同构结构分支但将结构输入置零。概率先在三个随机种子间取均值，再以34个录音组为重采样单元进行2,000次配对Bootstrap。

## Bootstrap结果

| left | right | endpoint | point_difference | ci_low | ci_high | bootstrap_runs_requested | bootstrap_runs_valid | supports_positive_gain | probability_aggregation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2_v2_scalar | C0_waveform | low_snr_auroc | 0.0071 | -0.0081 | 0.0232 | 2000 | 2000 | False | three_seed_mean |
| C2_v2_scalar | C0_waveform | auroc | 0.0113 | -0.0001 | 0.0243 | 2000 | 2000 | False | three_seed_mean |
| C2_v2_scalar | C1_full_zero | low_snr_auroc | 0.0063 | -0.0070 | 0.0193 | 2000 | 2000 | False | three_seed_mean |
| C2_v2_scalar | C1_full_zero | auroc | 0.0076 | -0.0005 | 0.0162 | 2000 | 2000 | False | three_seed_mean |

## 种子级五折平均增量

| left | right | seed | delta_low_snr_auroc | delta_auroc |
| --- | --- | --- | --- | --- |
| C2_v2_scalar | C0_waveform | 20260826 | 0.0151 | 0.0154 |
| C2_v2_scalar | C0_waveform | 20260827 | 0.0281 | 0.0331 |
| C2_v2_scalar | C0_waveform | 20260828 | 0.0151 | 0.0117 |
| C2_v2_scalar | C1_full_zero | 20260826 | -0.0010 | 0.0004 |
| C2_v2_scalar | C1_full_zero | 20260827 | 0.0386 | 0.0321 |
| C2_v2_scalar | C1_full_zero | 20260828 | 0.0239 | 0.0193 |

## 结论

至少一个低SNR AUROC比较的95%置信区间下限不大于0。因此结果仅支持结构增益趋势，不能声称结构增益成立。
