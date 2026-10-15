# 第20实验 v8：全自动多径时延拓扑提取报告

## 协议

- 数据：第20实验 110 个 Bellhop 合成 clicktrain。
- 条件：clean、+5 dB、−5 dB；评价阶段独立读取物理到达数据。
- 协议：按 69 个 `train_id` 分组的 5 折 OOF；共享门控仅用训练组 clean 波形自监督训练。
- 匹配：跨事件覆盖至少 6/10 的反射结构，±0.03 ms 一对一匹配；绝对幅值不计入主结论。
- 结果状态：正式 OOF 汇总：覆盖全部 110 个 clicktrain。
- 下表只由本次已生成预测计算，不预设任何方法优劣。

## clean

### 稳定结构

| 方法 | stable_precision | stable_recall | stable_f1 | stable_delay_mae_ms | stable_exact_count |
|---|---:|---:|---:|---:|---:|
| cepstrum_cfar | 0.3333 | 0.5000 | 0.4444 | 0.0063 | 0.0000 |
| med | 0.0238 | 1.0000 | 0.0465 | 0.0077 | 0.0000 |
| sparse_gate | 0.0000 | 0.0000 | 0.0000 | 0.0109 | 0.0000 |

### 辅助指标

| 方法 | event_f1 | reconstruction_nrmse | runtime_ms |
|---|---:|---:|---:|
| cepstrum_cfar | 0.1913 | N/A | 103.4828 |
| med | 0.0530 | N/A | 40.1035 |
| sparse_gate | 0.0000 | 0.9495 | 32.8210 |

## snr_m05

### 稳定结构

| 方法 | stable_precision | stable_recall | stable_f1 | stable_delay_mae_ms | stable_exact_count |
|---|---:|---:|---:|---:|---:|
| cepstrum_cfar | 0.0000 | 0.0000 | 0.0000 | 0.0193 | 0.0000 |
| med | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0000 |
| sparse_gate | 0.0000 | 0.0000 | 0.0000 | 0.0169 | 0.0000 |

### 辅助指标

| 方法 | event_f1 | reconstruction_nrmse | runtime_ms |
|---|---:|---:|---:|
| cepstrum_cfar | 0.0000 | N/A | 176.1621 |
| med | 0.0000 | N/A | 25.4646 |
| sparse_gate | 0.0000 | 0.9723 | 6.7298 |

## snr_p05

### 稳定结构

| 方法 | stable_precision | stable_recall | stable_f1 | stable_delay_mae_ms | stable_exact_count |
|---|---:|---:|---:|---:|---:|
| cepstrum_cfar | 0.0000 | 0.0000 | 0.0000 | 0.0073 | 0.0000 |
| med | 0.0000 | 0.0000 | 0.0000 | N/A | 0.0000 |
| sparse_gate | 0.0000 | 0.0000 | 0.0000 | 0.0109 | 0.0000 |

### 辅助指标

| 方法 | event_f1 | reconstruction_nrmse | runtime_ms |
|---|---:|---:|---:|
| cepstrum_cfar | 0.0953 | N/A | 174.5280 |
| med | 0.0000 | N/A | 26.0770 |
| sparse_gate | 0.0000 | 0.9568 | 6.6351 |

## 观察到的失败模式（clean）

| 方法 | 漏检样本 | 过检样本 | 数量正确但时延失配 |
|---|---:|---:|---:|
| cepstrum_cfar | 65 | 95 | 14 |
| med | 41 | 110 | 4 |
| sparse_gate | 109 | 109 | 43 |

## 配对比较

`paired_deltas.csv` 保存全部条件、方法对和指标的配对中位数差及95% bootstrap区间。当区间跨越零时，报告不将该差异解释为稳定优势。

## 产物

- `summary_metrics.csv`：逐条件、逐方法的中位数、四分位数和95%区间。
- `per_clicktrain_metrics.csv` 与 `event_metrics.csv`：可审计的逐样本结果。
- `summary.png`：主指标图；`diagnostic_cases.png`：按 CFAR F1 自动选择的困难、中位和最佳案例。
