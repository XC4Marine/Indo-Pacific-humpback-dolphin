# 双分支物理约束海豚 Click 分类实验

本目录是独立实验流水线，统一比较 Waveform Transformer、`Recon + BCE` 和
`Recon + Physics + BCE`。数据按原始录音编号使用相同的五折划分，验证阈值只在
验证折选择，最终统计按录音组执行 2000 次 Bootstrap。

## 运行

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\pipeline.py' `
  --mode smoke
```

模式：

- `--mode smoke`：检查7000条数据、张量尺寸、因果卷积、损失和 BCE 梯度路径。
- `--mode quick`：折2–4训练、折1验证、折0测试，运行三个实验条件。
- `--mode full`：三个条件的完整五折录音来源隔离评估。
- `--force`：忽略已有检查点与结果并重新计算。
- `--seeds 20260826 20260827 ...`：设置一个或多个独立训练随机种子。

完整五折、五种子示例：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\pipeline.py' `
  --mode full `
  --seeds 20260826 20260827 20260828 20260829 20260830 `
  --output-dir 'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_full_5seeds'
```

固定低SNR噪声混合的完整实验示例：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\pipeline.py' `
  --mode full `
  --positive-noise-mix `
  --snr-min-db -15 `
  --snr-max-db 15 `
  --mix-seed 20260826 `
  --seeds 20260826 20260827 20260828 20260829 20260830 `
  --output-dir 'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_full_5seeds'
```

启用混合时，正样本与同录音来源的唯一未选背景配对；负样本与原五折保持不变。
混合配对、目标SNR、混合种子和最终波形均纳入数据哈希。

默认输出写入本目录下的 `outputs/`。可用 `--output-dir` 指定其他绝对目录。
训练窗口使用嵌套进度条显示总体阶段、实验条件、折、epoch、batch、分项损失、
梯度范数和验证 AUROC。

## 主要产物

- `run_config.json`、`dataset_manifest.csv`、`data_hash.json`
- `models/*.pt`、`logs/*_history.csv`、`logs/*_batch.csv`
- `metrics/fold_metrics.csv`、`metrics/bootstrap_summary.csv`
- `metrics/paired_differences.csv`、`metrics/low_minus_high_paired_auroc.csv`
- `metrics/snr_bin_metrics.csv`、`metrics/snr_bin_summary.csv`
- `predictions/sample_predictions.csv`、`artifact_validation.json`
- `figures/roc_pr.png`、`figures/confusion_matrices.png`
- `ablation_results.csv`、`research_report_zh.md`

物理解耦不由分类性能自动证明。报告同时给出 `NCC(x_hat, y)`、信道尾部能量及
传播响应多样性；若出现 `x_hat≈y, h_hat≈delta`，结论会明确限制为分类有效。

## 既有检查点的 h 多样性审计

`h_diversity_audit.py` 只读取固定低 SNR 实验的清单、波形缓存和既有物理模型
检查点，不训练模型、不修改权重或分类结果。完整审计命令：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\h_diversity_audit.py' `
  --mode full `
  --checkpoint-output 'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_full_5seeds' `
  --output-dir 'D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_full_5seeds\h_diversity_audit' `
  --seeds 20260826 20260827 20260828 20260829 20260830 `
  --bootstrap-runs 2000 `
  --permutation-runs 2000 `
  --device auto `
  --force
```

`--mode smoke` 固定运行 Physics 条件的首个种子和折0。`--force` 仅清理并重建
明确指定的审计输出目录；程序拒绝将审计目录设为检查点目录。完整结果包括
`responses/`、五个指标 CSV、四类诊断图、`h_diversity_acceptance.json` 和
`h_diversity_report_zh.md`。
