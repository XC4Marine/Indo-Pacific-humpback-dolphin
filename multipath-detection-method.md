# 中华白海豚 Click 多径检测与筛选方法

## 概述

本文档描述从海豚 click 脉冲片段中自动检测直达脉冲与多径（海底/海面反射）信号、
计算信道参数（时延、幅度比、相干性），并校准锚点偏差的完整方法。

## 流水线总览

```
09_Locate (Click 切分)
    │  18116 个 Pulse_YYY.wav (±5ms, 576kHz)
    ▼
10_AutoCorrMultipath (多径检测)
    │  ACF 次峰初筛 → Teager 验证 → Pearson 过滤
    │  → all_pulse_times.txt
    ▼
11_ChannelMetrics (信道参数)
    │  Delay_us, AmpRatio, Coherence
    │  → pulse_params.txt
    ▼
recalibrate_fix_v2.py (锚点校准)
    │  AmpRatio<0.9 异常 Click → RMS 质心重新定位
    │  → 更新 all_pulse_times.txt + pulse_params.txt
```

## 第一层：归一化自相关 (ACF) 次峰初筛

### 原理
多径信号与直达信号具有高度波形相似性，仅存在时间延迟与幅度衰减。
归一化自相关函数 (ACF) 的主峰 (Lag=0) 对应直达信号的自匹配，
次峰位置对应多径的时间延迟。

### 计算步骤
1. 对 5761 点信号 (10ms, 576kHz) 计算自相关:
   `acf = numpy.correlate(signal, signal, mode="full")`
2. 取右半侧 (Lag ≥ 0)，除以 Lag=0 处值归一化至 1.0
3. 在 |lag| ∈ [0.1ms, 5ms) 区间内 `scipy.signal.find_peaks` 检测次峰
4. 筛选条件:
   - 峰高 ≥ 主峰的 **10%**
   - 峰间距 ≥ **30μs** (distance)
   - prominence ≥ 阈值的 50%
5. 按相关系数降序排列，选取前 **3** 个候选多径

### 参数选择理由
- 10% 阈值: 100 样本预实验对比 2%-10% 六个档位，10% 下平均 1.15 个/样本
- 最小滞后 0.1ms: 排除直达峰自身邻近区域的旁瓣
- 最大 3 个: 覆盖绝大部分真实多径场景，避免噪声误检

## 第二层：Teager 能量交叉验证

### 原理
Teager Energy Operator (TEO) 对信号瞬态变化敏感，真 click 脉冲具有高 TEO 值。
若自相关选出的候选位置 TEO 能量过低，可能是噪声而非真实多径。

### 计算步骤
1. 计算全信号 TEO: `te[n] = x[n]^2 - x[n-1] * x[n+1]`
2. 取绝对值，得全局最大值 `T_max`
3. 对每个候选滞后位置，取 ±50μs 窗内最大 TEO 绝对值 `T_cand`
4. 若 `T_cand < 0.001 * T_max` (0.1%)，移出候选列表

> 注：Teager 验证仅在 `autocorr_detect.py` 实验脚本中使用，
> `batch_detect.py` 全量批处理跳过此层以提升速度。

## 第三层：Pearson 相干性过滤

### 原理
若多径候选确为直达信号的反射副本，其局部波形应与直达窗波形高度相关；
若窗口内为随机噪声，则 Pearson 相关系数应接近 0。

### 计算步骤
1. 取直达锚点附近 ±50μs 窗内信号 `direct_win`
2. 取候选多径位置 ±50μs 窗内信号 `mp_win`
3. 计算 Pearson r: `r = Σ((d_i - μ_d)(m_i - μ_m)) / sqrt(Σ(d_i - μ_d)^2 * Σ(m_i - μ_m)^2)`
4. 若 `|r| < 0.02`，该候选视为不相干噪声，移出列表

## 信道参数计算

### Delay_us (μs)
`Delay = (t_mp - t_direct) * 10^6`

多径相对直达的时间延迟，以微秒为单位。

### AmpRatio
`AmpRatio = RMS(direct_win) / RMS(mp_win)`

其中 RMS 在 ±50μs 窗内计算。AmpRatio > 1 表示直达强于多径（正常情况）；
AmpRatio < 1 表示多径窗内信号强于直达窗（异常，需要校准）。

### Coherence
`Coherence = |Pearson r(direct_win, mp_win)|`

±50μs 窗内直达与多径的波形线性相关程度，值越接近 1 表示多径特征越可靠。

## RMS 质心法锚点校准

### 问题
原始方法假设直达脉冲位于窗口中心 (t=5.0000ms)，但实际信号中直达脉冲可能
偏离中心位置。当偏离较大时，原始锚点可能落在噪声区或弱信号区，导致
AmpRatio 严重低估（< 0.9）。

### 算法
1. 对信号逐点计算 50μs 滑动窗内的 RMS^2
2. 取 `argmax(RMS^2)` 作为新直达锚点 (样本索引)
3. 以新锚点为基准，重新运行三层过滤检测多径
4. 重新计算 Delay_us、AmpRatio、Coherence
5. 值替换写入 `all_pulse_times.txt` 和 `pulse_params.txt`

### 校准效果
- 355 个 AmpRatio<0.9 异常 Click 全部成功校准
- 新检出 558 个多径（旧锚点未能检出的）
- 校准后 AmpRatio<0.9 异常降为 0

## 输出文件格式

### all_pulse_times.txt
```
PulseTrain_XXX	Click_YYY	直达ms	多径1ms	多径2ms	多径3ms
PulseTrain_001	Click_001	5.0000	5.1441	5.4986	NaN
```

### pulse_params.txt
```
PulseTrain	ClickName	PathType	Delay_us	AmpRatio	Coherence
PulseTrain_001	Click_001	Direct	0.00	1.000000	1.000000
PulseTrain_001	Click_001	Multipath_1	144.10	3.110994	0.492873
PulseTrain_001	Click_001	Multipath_2	498.60	2.602653	0.447567
```

### anomalous_ampratio_final.txt
校准后仍 AmpRatio<0.9 的残留异常 (当前为空)。

## 最终统计 (10% 阈值, MAX=3)

| 指标 | 值 |
|------|-----|
| 总 Click | 18116 |
| 检出多径 Click | 10519 (58.1%) |
| 多径总数 | 15260 |
| 平均每 Click | 0.84 |
| AmpRatio<0.9 残留 | 0 |

## 代码清单

| 文件 | 功能 |
|------|------|
| `10_AutoCorrMultipath/autocorr_detect.py` | 100 样本实验 (含 Teager 验证) |
| `10_AutoCorrMultipath/batch_detect.py` | 全量批处理 (ACF + Pearson) |
| `11_ChannelMetrics/calc_pulse_params.py` | 信道参数计算 (Delay_us, AmpRatio, Coherence) |
| `11_ChannelMetrics/recalibrate_fix_v2.py` | RMS 质心法锚点校准 |
