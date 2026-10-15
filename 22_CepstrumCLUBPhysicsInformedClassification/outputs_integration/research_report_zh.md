# 第22个轻量化完整五折消融实验报告

## 数据与边界

- 固定数据哈希：`224ec22c07267873dc7a045fd4a339bfc2164f129d144acc702828ac61506615`；只读复用初版独立峰对齐缓存。
- 任务：中华白海豚 click 与困难噪声二分类；第21个实验不作为严格配对基线。
- 五个种子、五折、六条件；每折使用下一折验证，最佳验证AUROC检查点用于测试。
- 倒谱仅作为弱结构监督，以下诊断不等同于真实物理信道证明。

## 主表

| condition | condition_label | total_parameters | trainable_parameters | auroc | auroc_seed_sd | auprc | auprc_seed_sd | balanced_accuracy | balanced_accuracy_seed_sd | f1 | f1_seed_sd | sensitivity | sensitivity_seed_sd | specificity | specificity_seed_sd | training_seconds | peak_memory_mb | mean_batch_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Full | 609531 | 609531 | 0.55343 | NA | 0.52307 | NA | 0.55304 | NA | 0.62515 | NA | 0.74540 | NA | 0.36068 | NA | 8.78902 | 84.86279 | 0.02415 |

## 消融表（Full−Ablation）

_当前运行范围无对应比较。_

## SNR表

| condition | snr_lower_db | snr_upper_db | positive_count | auroc | auroc_seed_sd | sensitivity | sensitivity_seed_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | -15.00000 | -10.00000 | 94 | 0.48829 | NA | 0.67021 | NA |
| full | -10.00000 | -5.00000 | 129 | 0.49251 | NA | 0.60465 | NA |
| full | -5.00000 | 0.00000 | 140 | 0.49362 | NA | 0.65000 | NA |
| full | 0.00000 | 5.00000 | 123 | 0.57640 | NA | 0.78862 | NA |
| full | 5.00000 | 10.00000 | 101 | 0.62917 | NA | 0.85149 | NA |
| full | 10.00000 | 15.00000 | 120 | 0.65243 | NA | 0.93333 | NA |

## 解耦诊断表

| condition | diagnostic | mean | seed_sd | finite_count |
| --- | --- | --- | --- | --- |
| full | mean_mim | 0.00514 | NA | 1 |
| full | mean_ncc | -0.00016 | NA | 1 |
| full | mean_channel_tail_energy | 6.25080 | NA | 1 |
| full | mean_response_diversity | 0.00179 | NA | 1 |

所有数值均来自实际运行；置信区间跨0者已标记为“证据不充分”。
