# 定向滤波与多径参数选择报告

## 最终滤波

- 20 kHz Butterworth高通（原型阶数2）后串联 185–215 kHz带阻（原型阶数2）。
- 两级均使用SOS与sosfiltfilt零相位处理。
- 检出强200 kHz背景样本：383。
- 原始主到达疑似由200 kHz人工脉冲主导的样本：11；滤波前后主到达移动超过50 μs的样本：16。
- 这些强干扰样本的185–215 kHz实测衰减中位数：-22.778 dB。
- 背景RMS变化中位数：-4.699 dB。
- 30–180 kHz主体频谱余弦相似度第5百分位：0.999145。
- 选择185–215 kHz而非190–210 kHz，是因为实测干扰峰覆盖约188–206 kHz；二阶原型比四阶带阻的脉冲振铃更小。

## 最终检测参数

```json
{
  "segment_us": 50.0,
  "center_search_half_width_ms": 0.5,
  "direct_guard_us": 100.0,
  "max_delay_ms": 1.0,
  "candidate_min_distance_us": 50.0,
  "peak_height_mad_factor": 5.0,
  "peak_prominence_mad_factor": 3.0,
  "min_candidate_snr_db": 6.0,
  "min_abs_ncc": 0.55,
  "min_spectral_cosine": 0.7,
  "min_accept_score": 0.6,
  "weight_energy": 0.2,
  "weight_ncc": 0.5,
  "weight_spectral": 0.3,
  "max_multipaths": 2
}
```

选择理由：50 μs覆盖主要click振荡且保持相邻路径分辨率；0.5 ms中心窗口降低其他脉冲抢占主到达的风险；100 μs保护区保留旧结果中大量100–150 μs近距离路径；1 ms覆盖旧方法约94.3%的延迟分布，并显著降低下一次独立click进入搜索区的风险。MAD 5/3、SNR 6 dB、NCC 0.55、频谱余弦0.70及评分0.60采用精确率优先设置。

## 固定审查集与视觉验收边界

- 固定审查集：200条；每条生成滤波/检测图与路径叠加图，共400张诊断图。
- 参数筛选同时参考诊断图、旧ACF结果的一致性代理及候选指标；旧ACF结果不作为真值。
- 诊断图已经完成单人目视复核，用于排除明显的主到达错位、200 kHz人工脉冲和后续独立click。
- 由于当前数据没有逐路径人工真值或双人盲审标签，98%主到达正确率和90%多径可信率只能作为操作性验收目标，不能表述为独立统计准确率；Notebook保留完整诊断图与候选表，供后续正式标注复核。

## 全量结果

- 输入WAV：3500。
- 处理失败：0。
- 选中多径总数：1213。
- 0/1/2条多径分布：{0: 2426, 1: 935, 2: 139}。
- 合成延迟定位误差：0个采样点。
- ±50 μs外振铃比例：0.004419。
- 自动验收：通过。

## 旧ACF参考一致性最高的实验（旧结果不作为真值）

| experiment | legacy_precision_proxy | legacy_recall_proxy | selected_total |
|---|---:|---:|---:|
| geometry_seg100_center0.5 | 0.833333 | 0.054054 | 12 |
| geometry_seg100_center1 | 0.833333 | 0.054054 | 12 |
| geometry_seg100_center1.5 | 0.833333 | 0.054054 | 12 |
| geometry_seg75_center0.5 | 0.750000 | 0.064865 | 16 |
| geometry_seg75_center1 | 0.750000 | 0.064865 | 16 |
