# 第22实验 v7：去 MIM 的交叉卷积一致性初筛

本实验为单随机种子五折初筛，不能替代多随机种子复验。

## 分类与损失

| auroc | auprc | balanced_accuracy | f1 | sensitivity | specificity | threshold | condition |
|---|---|---|---|---|---|---|---|
| 0.865331 | 0.875110 | 0.799151 | 0.774960 | 0.691655 | 0.906648 | 0.432207 | cross_conv_consistency |
| 0.880366 | 0.888997 | 0.803395 | 0.782813 | 0.708628 | 0.898161 | 0.528390 | no_mim_reference |

CCC 原始/加权损失占比见 `metrics/loss_summary.csv`；录音来源配对 Bootstrap 区间见 `metrics/paired_source_bootstrap.csv`。

## X/H 解耦诊断

| condition | subset | method | corr_1 | corr_2 | corr_3 | corr_4 | corr_5 | corr_6 | corr_7 | corr_8 | corr_9 | corr_10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_conv_consistency | all | kernel_cca_nystrom | 0.903944 | 0.739810 | 0.600786 | 0.576307 | 0.526986 | 0.475898 | 0.460355 | 0.402561 | 0.378288 | 0.357081 |
| cross_conv_consistency | all | linear_cca | 0.925115 | 0.814588 | 0.774149 | 0.727259 | 0.698612 | 0.675679 | 0.656975 | 0.619252 | 0.603129 | 0.594791 |
| cross_conv_consistency | positive | kernel_cca_nystrom | 0.815054 | 0.730093 | 0.631887 | 0.568702 | 0.528408 | 0.504791 | 0.441955 | 0.419507 | 0.411061 | 0.393310 |
| cross_conv_consistency | positive | linear_cca | 0.856925 | 0.837669 | 0.774105 | 0.747959 | 0.699065 | 0.673613 | 0.665176 | 0.643109 | 0.613242 | 0.598411 |
| no_mim_reference | all | kernel_cca_nystrom | 0.820099 | 0.722150 | 0.637749 | 0.579661 | 0.495615 | 0.472595 | 0.444974 | 0.406124 | 0.381884 | 0.370444 |
| no_mim_reference | all | linear_cca | 0.933971 | 0.817749 | 0.762990 | 0.746004 | 0.677364 | 0.651304 | 0.644282 | 0.636243 | 0.625130 | 0.599083 |
| no_mim_reference | positive | kernel_cca_nystrom | 0.806717 | 0.752897 | 0.653106 | 0.585901 | 0.528105 | 0.518841 | 0.484127 | 0.443029 | 0.437216 | 0.394516 |
| no_mim_reference | positive | linear_cca | 0.882655 | 0.851866 | 0.770426 | 0.733509 | 0.719476 | 0.692535 | 0.680663 | 0.648991 | 0.633997 | 0.617329 |

CCA/KCCA 为折内诊断，不能单独证明统计独立或真实物理信道恢复。
