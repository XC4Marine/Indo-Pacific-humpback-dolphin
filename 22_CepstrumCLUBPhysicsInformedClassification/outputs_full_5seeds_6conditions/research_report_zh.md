# 第22个轻量化完整五折消融实验报告

## 数据与边界

- 固定数据哈希：`224ec22c07267873dc7a045fd4a339bfc2164f129d144acc702828ac61506615`；只读复用初版独立峰对齐缓存。
- 任务：中华白海豚 click 与困难噪声二分类；第21个实验不作为严格配对基线。
- 五个种子、五折、六条件；每折使用下一折验证，最佳验证AUROC检查点用于测试。
- 倒谱仅作为弱结构监督，以下诊断不等同于真实物理信道证明。

## 主表

| condition | condition_label | total_parameters | trainable_parameters | auroc | auroc_seed_sd | auprc | auprc_seed_sd | balanced_accuracy | balanced_accuracy_seed_sd | f1 | f1_seed_sd | sensitivity | sensitivity_seed_sd | specificity | specificity_seed_sd | training_seconds | peak_memory_mb | mean_batch_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | Full | 609531 | 609531 | 0.89983 | 0.00347 | 0.90494 | 0.00221 | 0.82694 | 0.00287 | 0.81913 | 0.00330 | 0.78394 | 0.01837 | 0.86994 | 0.02147 | 2985.20468 | 85.64600 | 0.01925 |
| without_recon | w/o Recon | 609531 | 231634 | 0.89490 | 0.00383 | 0.89913 | 0.00700 | 0.82340 | 0.00432 | 0.81334 | 0.00515 | 0.76960 | 0.01140 | 0.87720 | 0.01071 | 2109.25545 | 50.00049 | 0.01530 |
| without_channel | w/o Channel | 609531 | 609531 | 0.89853 | 0.00479 | 0.90521 | 0.00498 | 0.82654 | 0.00454 | 0.81761 | 0.00477 | 0.77771 | 0.01646 | 0.87537 | 0.01948 | 2783.42420 | 84.10352 | 0.01721 |
| without_mim | w/o MIM | 584699 | 584699 | 0.90512 | 0.00399 | 0.91271 | 0.00513 | 0.82923 | 0.00426 | 0.82223 | 0.00599 | 0.79011 | 0.01462 | 0.86834 | 0.00952 | 2177.40162 | 84.44629 | 0.01516 |
| bce_only | BCE-only | 584699 | 14345 | 0.89781 | 0.00665 | 0.90137 | 0.01005 | 0.82609 | 0.00247 | 0.81684 | 0.00368 | 0.77571 | 0.01113 | 0.87646 | 0.00979 | 500.02930 | 30.27588 | 0.00425 |
| direct_classifier | Direct classifier | 14345 | 14345 | 0.89781 | 0.00665 | 0.90137 | 0.01005 | 0.82609 | 0.00247 | 0.81684 | 0.00368 | 0.77571 | 0.01113 | 0.87646 | 0.00979 | 478.66245 | 28.08154 | 0.00404 |

## 消融表（Full−Ablation）

| variant | removed_mechanism | metric | full_minus_ablation | ci_low | ci_high | evidence | full_seed_wins | bootstrap_runs_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| without_recon | w/o Recon | auroc | 0.00492 | -0.00059 | 0.01053 | 证据不充分 | 4 | 2000 |
| without_recon | w/o Recon | auprc | 0.00582 | -0.00162 | 0.01434 | 证据不充分 | 4 | 2000 |
| without_recon | w/o Recon | balanced_accuracy | 0.00354 | -0.00253 | 0.00946 | 证据不充分 | 4 | 2000 |
| without_recon | w/o Recon | f1 | 0.00579 | -0.00000 | 0.01213 | 证据不充分 | 5 | 2000 |
| without_recon | w/o Recon | sensitivity | 0.01434 | -0.00869 | 0.03604 | 证据不充分 | 4 | 2000 |
| without_recon | w/o Recon | specificity | -0.00726 | -0.03431 | 0.01921 | 证据不充分 | 2 | 2000 |
| without_channel | w/o Channel | auroc | 0.00130 | -0.00385 | 0.00720 | 证据不充分 | 2 | 2000 |
| without_channel | w/o Channel | auprc | -0.00027 | -0.00503 | 0.00500 | 证据不充分 | 3 | 2000 |
| without_channel | w/o Channel | balanced_accuracy | 0.00040 | -0.00492 | 0.00579 | 证据不充分 | 2 | 2000 |
| without_channel | w/o Channel | f1 | 0.00152 | -0.00273 | 0.00620 | 证据不充分 | 3 | 2000 |
| without_channel | w/o Channel | sensitivity | 0.00623 | -0.00557 | 0.01922 | 证据不充分 | 4 | 2000 |
| without_channel | w/o Channel | specificity | -0.00543 | -0.02368 | 0.01305 | 证据不充分 | 1 | 2000 |
| without_mim | w/o MIM | auroc | -0.00530 | -0.01244 | 0.00072 | 证据不充分 | 1 | 2000 |
| without_mim | w/o MIM | auprc | -0.00776 | -0.01355 | -0.00145 | CI不跨0 | 0 | 2000 |
| without_mim | w/o MIM | balanced_accuracy | -0.00229 | -0.00955 | 0.00443 | 证据不充分 | 2 | 2000 |
| without_mim | w/o MIM | f1 | -0.00310 | -0.01045 | 0.00372 | 证据不充分 | 1 | 2000 |
| without_mim | w/o MIM | sensitivity | -0.00617 | -0.02344 | 0.01615 | 证据不充分 | 1 | 2000 |
| without_mim | w/o MIM | specificity | 0.00160 | -0.02503 | 0.02206 | 证据不充分 | 3 | 2000 |
| bce_only | BCE-only | auroc | 0.00201 | -0.00374 | 0.00806 | 证据不充分 | 4 | 2000 |
| bce_only | BCE-only | auprc | 0.00357 | -0.00407 | 0.01283 | 证据不充分 | 3 | 2000 |
| bce_only | BCE-only | balanced_accuracy | 0.00086 | -0.00520 | 0.00714 | 证据不充分 | 3 | 2000 |
| bce_only | BCE-only | f1 | 0.00229 | -0.00371 | 0.00849 | 证据不充分 | 3 | 2000 |
| bce_only | BCE-only | sensitivity | 0.00823 | -0.01316 | 0.02671 | 证据不充分 | 4 | 2000 |
| bce_only | BCE-only | specificity | -0.00651 | -0.03202 | 0.01575 | 证据不充分 | 2 | 2000 |
| direct_classifier | Direct classifier | auroc | 0.00201 | -0.00374 | 0.00806 | 证据不充分 | 4 | 2000 |
| direct_classifier | Direct classifier | auprc | 0.00357 | -0.00407 | 0.01283 | 证据不充分 | 3 | 2000 |
| direct_classifier | Direct classifier | balanced_accuracy | 0.00086 | -0.00520 | 0.00714 | 证据不充分 | 3 | 2000 |
| direct_classifier | Direct classifier | f1 | 0.00229 | -0.00371 | 0.00849 | 证据不充分 | 3 | 2000 |
| direct_classifier | Direct classifier | sensitivity | 0.00823 | -0.01316 | 0.02671 | 证据不充分 | 4 | 2000 |
| direct_classifier | Direct classifier | specificity | -0.00651 | -0.03202 | 0.01575 | 证据不充分 | 2 | 2000 |

## SNR表

| condition | snr_lower_db | snr_upper_db | positive_count | auroc | auroc_seed_sd | sensitivity | sensitivity_seed_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bce_only | -15.00000 | -10.00000 | 566 | 0.72296 | 0.00601 | 0.38693 | 0.01446 |
| bce_only | -10.00000 | -5.00000 | 604 | 0.83316 | 0.00855 | 0.58344 | 0.01731 |
| bce_only | -5.00000 | 0.00000 | 589 | 0.92074 | 0.00745 | 0.82309 | 0.01267 |
| bce_only | 0.00000 | 5.00000 | 585 | 0.95637 | 0.00759 | 0.91453 | 0.02331 |
| bce_only | 5.00000 | 10.00000 | 574 | 0.97220 | 0.00723 | 0.96307 | 0.00753 |
| bce_only | 10.00000 | 15.00000 | 582 | 0.97952 | 0.00682 | 0.98110 | 0.00739 |
| direct_classifier | -15.00000 | -10.00000 | 566 | 0.72296 | 0.00601 | 0.38693 | 0.01446 |
| direct_classifier | -10.00000 | -5.00000 | 604 | 0.83316 | 0.00855 | 0.58344 | 0.01731 |
| direct_classifier | -5.00000 | 0.00000 | 589 | 0.92074 | 0.00745 | 0.82309 | 0.01267 |
| direct_classifier | 0.00000 | 5.00000 | 585 | 0.95637 | 0.00759 | 0.91453 | 0.02331 |
| direct_classifier | 5.00000 | 10.00000 | 574 | 0.97220 | 0.00723 | 0.96307 | 0.00753 |
| direct_classifier | 10.00000 | 15.00000 | 582 | 0.97952 | 0.00682 | 0.98110 | 0.00739 |
| full | -15.00000 | -10.00000 | 566 | 0.72800 | 0.00519 | 0.39541 | 0.03800 |
| full | -10.00000 | -5.00000 | 604 | 0.83486 | 0.00739 | 0.60132 | 0.02744 |
| full | -5.00000 | 0.00000 | 589 | 0.92394 | 0.00424 | 0.83769 | 0.02163 |
| full | 0.00000 | 5.00000 | 585 | 0.95763 | 0.00321 | 0.91829 | 0.01924 |
| full | 5.00000 | 10.00000 | 574 | 0.97312 | 0.00287 | 0.96794 | 0.00876 |
| full | 10.00000 | 15.00000 | 582 | 0.97955 | 0.00246 | 0.98041 | 0.00464 |
| without_channel | -15.00000 | -10.00000 | 566 | 0.71894 | 0.01544 | 0.39081 | 0.05182 |
| without_channel | -10.00000 | -5.00000 | 604 | 0.83335 | 0.00721 | 0.58974 | 0.02542 |
| without_channel | -5.00000 | 0.00000 | 589 | 0.92472 | 0.00753 | 0.83022 | 0.02002 |
| without_channel | 0.00000 | 5.00000 | 585 | 0.95840 | 0.00299 | 0.91350 | 0.01453 |
| without_channel | 5.00000 | 10.00000 | 574 | 0.97415 | 0.00279 | 0.96202 | 0.00743 |
| without_channel | 10.00000 | 15.00000 | 582 | 0.97953 | 0.00195 | 0.97766 | 0.00210 |
| without_mim | -15.00000 | -10.00000 | 566 | 0.73785 | 0.00891 | 0.42862 | 0.03624 |
| without_mim | -10.00000 | -5.00000 | 604 | 0.84449 | 0.00377 | 0.61424 | 0.02261 |
| without_mim | -5.00000 | 0.00000 | 589 | 0.92790 | 0.00493 | 0.83260 | 0.01787 |
| without_mim | 0.00000 | 5.00000 | 585 | 0.96013 | 0.00296 | 0.91521 | 0.00789 |
| without_mim | 5.00000 | 10.00000 | 574 | 0.97632 | 0.00337 | 0.96934 | 0.00623 |
| without_mim | 10.00000 | 15.00000 | 582 | 0.98216 | 0.00225 | 0.97869 | 0.00551 |
| without_recon | -15.00000 | -10.00000 | 566 | 0.71210 | 0.01303 | 0.35230 | 0.03544 |
| without_recon | -10.00000 | -5.00000 | 604 | 0.82605 | 0.00779 | 0.57550 | 0.02970 |
| without_recon | -5.00000 | 0.00000 | 589 | 0.92147 | 0.00492 | 0.83056 | 0.01695 |
| without_recon | 0.00000 | 5.00000 | 585 | 0.95701 | 0.00198 | 0.91590 | 0.00759 |
| without_recon | 5.00000 | 10.00000 | 574 | 0.97202 | 0.00130 | 0.95749 | 0.00990 |
| without_recon | 10.00000 | 15.00000 | 582 | 0.97878 | 0.00172 | 0.98282 | 0.00471 |

## 解耦诊断表

| condition | diagnostic | mean | seed_sd | finite_count |
| --- | --- | --- | --- | --- |
| full | mean_mim | -0.00654 | 0.00733 | 5 |
| full | mean_ncc | 0.05709 | 0.02336 | 5 |
| full | mean_channel_tail_energy | 3.15355 | 0.38203 | 5 |
| full | mean_response_diversity | 0.00160 | 0.00069 | 5 |
| without_recon | mean_mim | -0.02699 | 0.01244 | 5 |
| without_recon | mean_ncc | NA | NA | 0 |
| without_recon | mean_channel_tail_energy | 12.82772 | 2.20581 | 5 |
| without_recon | mean_response_diversity | 0.00709 | 0.00179 | 5 |
| without_channel | mean_mim | -0.01461 | 0.01022 | 5 |
| without_channel | mean_ncc | 0.09747 | 0.04340 | 5 |
| without_channel | mean_channel_tail_energy | 2.65544 | 0.82529 | 5 |
| without_channel | mean_response_diversity | 0.00338 | 0.00228 | 5 |
| without_mim | mean_mim | NA | NA | 0 |
| without_mim | mean_ncc | 0.04744 | 0.01838 | 5 |
| without_mim | mean_channel_tail_energy | 4.53650 | 1.06518 | 5 |
| without_mim | mean_response_diversity | 0.00046 | 0.00007 | 5 |
| bce_only | mean_mim | NA | NA | 0 |
| bce_only | mean_ncc | NA | NA | 0 |
| bce_only | mean_channel_tail_energy | NA | NA | 0 |
| bce_only | mean_response_diversity | NA | NA | 0 |
| direct_classifier | mean_mim | NA | NA | 0 |
| direct_classifier | mean_ncc | NA | NA | 0 |
| direct_classifier | mean_channel_tail_energy | NA | NA | 0 |
| direct_classifier | mean_response_diversity | NA | NA | 0 |

所有数值均来自实际运行；置信区间跨0者已标记为“证据不充分”。
