# 人工标注集多径参数调优与验证报告

## 数据与验证设计

- 有效人工标注：199条。
- 有多径：172条；无多径：27条。
- 人工标注多径总数：256条。
- 参数候选：15120组。
- 验证：按航次与脉冲串分组的5折交叉验证，同一脉冲串不跨折。
- 匹配判据：检测位置与人工位置相差不超过±50 μs；一对一匹配。
- 每个外层验证折只使用其余四折选择F2最高的参数；同分依次比较Precision、时延MAE、路径数量完全正确率和无多径识别率。

## 折外验证结果

```json
{
  "files": 199,
  "tp": 192,
  "fp": 156,
  "fn": 64,
  "exact_count": 93,
  "negative_files": 27,
  "true_negative_files": 1,
  "main_within_tolerance": 198,
  "delay_abs_error_sum_samples": 1052.0,
  "delay_abs_error_count": 192,
  "precision": 0.5517241379310345,
  "recall": 0.75,
  "f1": 0.6357615894039735,
  "f2": 0.6997084548104957,
  "exact_count_accuracy": 0.46733668341708545,
  "no_multipath_specificity": 0.037037037037037035,
  "main_within_50us_rate": 0.9949748743718593,
  "main_mae_us": 8.44500279173646,
  "main_median_ae_us": 8.680555555555555,
  "delay_mae_us": 9.51244212962963,
  "delay_median_ae_us": 6.944444444444445,
  "delay_p95_ae_us": 24.305555555555557
}
```

折外结果由每一折各自在其训练部分选出的参数产生，是本次主要效果估计。

## 结果解释与使用边界

- 主到达定位较稳定：±50 μs内比例为99.50%，MAE为8.445 μs。
- 多径定位在已匹配路径上较准确：时延MAE为9.512 μs，P95绝对误差为24.306 μs。
- 路径级召回率为75.00%，但Precision为55.17%，路径数量完全正确率为46.73%。
- 无多径识别率仅为3.70%（1/27）；F2优先参数存在明显过检。
- 因此该参数适合作为高召回候选生成器；若全量结果要直接作为最终多径标签，应增加人工复核或改用更强调Precision/无多径识别率的选参目标。

## 各折选参

| validation_fold | config_name | training_f2 | training_precision | training_recall | validation_f2 | validation_precision | validation_recall | segment_us | center_search_half_width_ms | direct_guard_us | max_delay_ms | candidate_min_distance_us | peak_height_mad_factor | peak_prominence_mad_factor | min_candidate_snr_db | min_abs_ncc | min_spectral_cosine | min_accept_score | weight_energy | weight_ncc | weight_spectral | max_multipaths |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | config_04091 | 0.762631 | 0.686695 | 0.784314 | 0.636704 | 0.576271 | 0.653846 | 35 | 0.1 | 30 | 1 | 35 | 4 | 2 | 0 | 0.2 | 0.25 | 0.25 | 0.3 | 0.4 | 0.3 | 2 |
| 1 | config_04175 | 0.75361 | 0.579861 | 0.814634 | 0.625 | 0.5 | 0.666667 | 35 | 0.1 | 30 | 1.5 | 35 | 4 | 2 | 0 | 0.2 | 0.25 | 0.25 | 0.3 | 0.4 | 0.3 | 2 |
| 2 | config_00311 | 0.732914 | 0.558219 | 0.795122 | 0.784672 | 0.614286 | 0.843137 | 25 | 0.1 | 30 | 1 | 25 | 4 | 2 | 0 | 0.2 | 0.25 | 0.25 | 0.3 | 0.4 | 0.3 | 2 |
| 3 | config_00311 | 0.750452 | 0.58042 | 0.809756 | 0.714286 | 0.526316 | 0.784314 | 25 | 0.1 | 30 | 1 | 25 | 4 | 2 | 0 | 0.2 | 0.25 | 0.25 | 0.3 | 0.4 | 0.3 | 2 |
| 4 | config_00326 | 0.743427 | 0.579505 | 0.8 | 0.734767 | 0.546667 | 0.803922 | 25 | 0.1 | 30 | 1 | 25 | 4 | 2 | 4 | 0.45 | 0.5 | 0.45 | 0.2 | 0.6 | 0.2 | 2 |

## 折分布

| fold | files | groups | true_paths | no_multipath |
|---|---|---|---|---|
| 0 | 40 | 25 | 52 | 6 |
| 1 | 38 | 24 | 51 | 6 |
| 2 | 40 | 25 | 51 | 5 |
| 3 | 41 | 25 | 51 | 5 |
| 4 | 40 | 25 | 51 | 5 |

## 后续全量应用参数

以下参数使用全部199条标注选择，供后续应用到3500条全量样本；其全标注集指标属于回代结果，不作为独立验证分数。

```json
{
  "segment_us": 25.0,
  "center_search_half_width_ms": 0.1,
  "direct_guard_us": 30.0,
  "max_delay_ms": 1.0,
  "candidate_min_distance_us": 25.0,
  "peak_height_mad_factor": 4.0,
  "peak_prominence_mad_factor": 2.0,
  "min_candidate_snr_db": 0.0,
  "min_abs_ncc": 0.2,
  "min_spectral_cosine": 0.25,
  "min_accept_score": 0.25,
  "weight_energy": 0.3,
  "weight_ncc": 0.4,
  "weight_spectral": 0.3,
  "max_multipaths": 2
}
```

全标注集回代指标：

```json
{
  "files": 199,
  "tp": 206,
  "fp": 156,
  "fn": 50,
  "exact_count": 97,
  "negative_files": 27,
  "true_negative_files": 1,
  "main_within_tolerance": 198,
  "delay_abs_error_sum_samples": 1054.0,
  "delay_abs_error_count": 206,
  "precision": 0.569060773480663,
  "recall": 0.8046875,
  "f1": 0.6666666666666666,
  "f2": 0.7431457431457431,
  "exact_count_accuracy": 0.48743718592964824,
  "no_multipath_specificity": 0.037037037037037035,
  "main_within_50us_rate": 0.9949748743718593,
  "main_mae_us": 7.598757677275265,
  "main_median_ae_us": 8.680555555555555,
  "delay_mae_us": 8.88282092772384,
  "delay_median_ae_us": 6.944444444444445,
  "delay_p95_ae_us": 24.305555555555557
}
```
