# 实验记录

## 2026-06-26 — 09_Locate: MATLAB → Python 迁移

### 完成内容
- 将 `09_Locate/locate.m` 移植为 Python 脚本 `09_Locate/locate.py`
- 功能：基于 IPI 模板互相关定位 click 中心，截取 ±5ms 窗口
- 处理结果：832 个 PulseTrain，共 18116 个 Click 文件夹

### 算法流程
1. 读取 PulseTrain.wav + PulseParameters.txt
2. 从 IPI 构造理想脉冲模板向量
3. 计算 TEO 并归一化
4. 全局互相关对齐（scipy.signal.correlate）
5. 每 pulse 截取 ±5ms 保存 wav/波形图/分析图

### 输出结构
```
09_Locate/PulseTrain_XXX/Click_YYY/
├── Pulse_YYY.wav          (576kHz, 5761 samples, ±5ms)
├── Waveform_YYY.png
└── Analysis_YYY.png
```

## 2026-06-26 — 10_AutoCorrMultipath & 11_ChannelMetrics: 多径检测与信道参数计算

### 多径检测方法 (ACF + Teager + Pearson 三层过滤)

#### 第一层：自相关 (ACF) 次峰初筛
- `numpy.correlate` 归一化自相关，Lag=0 处为 1.0
- 在 |lag| > 0.1ms、< 5ms 区间内 `scipy.signal.find_peaks` 检测次峰
- 阈值 10%（次峰 ≥ 主峰高度的 10%），峰间距 ≥ 30μs
- 按相关系数降序选取，最多 3 个多径候选

#### 第二层：Teager 能量交叉验证 (仅 autocorr_detect.py)
- 计算全信号 Teager Energy Operator: x(n)^2 - x(n-1)x(n+1)
- 验证候选位置 Teager 能量是否 ≥ 全局最大值的 0.1%
- 不通过则移出候选

#### 第三层：Pearson 相干性过滤
- 直达窗 (±50μs) 与多径候选窗 (±50μs) 计算 Pearson r
- 若 |r| < 0.02 则视为不相干噪声，移出候选

#### 信道参数计算 (calc_pulse_params.py)
- 窗宽 ±50μs RMS 计算：直达 RMS / 多径 RMS = AmpRatio
- Pearson 相干性：直达窗与多径窗的 Pearson r
- Delay_us: 多径时间相对直达时间的延迟

### 全量批处理结果 (10%, MAX=3)
| 指标 | 值 |
|------|-----|
| 总 Click | 18116 |
| 检出多径 Click | 10519 (58.1%) |
| 多径总数 | 15307 |
| 平均每 Click | 0.84 |
| 100 样本验证 | 65.0% 检出, 平均 1.13 个/样本 |

### 输出文件
- `10_AutoCorrMultipath/autocorr_detect.py`: 100 样本实验脚本
- `10_AutoCorrMultipath/batch_detect.py`: 全量批处理
- `10_AutoCorrMultipath/all_pulse_times.txt`: PulseTrain/Click/直达ms/多径1ms/多径2ms/多径3ms
- `11_ChannelMetrics/calc_pulse_params.py`: 信道参数计算
- `11_ChannelMetrics/pulse_params.txt`: 33376 行 (Direct + Multipath_X, Delay_us/AmpRatio/Coherence)

## 2026-06-27 — 11_ChannelMetrics: RMS 质心法锚点校准

### 背景
- 使用 `calc_pulse_params.py` 计算后检测到 355 个 Click 含 AmpRatio < 0.9 异常径
- 异常原因：原始窗口中心 ≠ 真实直达脉冲位置，导致 AmpRatio 被低估
- 编写 `recalibrate_fix_v2.py` 实现自动校准

### 校准算法 (recalibrate_fix_v2.py)
1. 读入 pulse_params.txt，筛选 AmpRatio < 0.9 的 Click
2. 对每个异常 Click 读原始 Pulse_YYY.wav
3. **RMS 质心法**定位：50μs 滑动窗求 RMS²，取 argmax 作为新直达锚点
4. 以新锚点为基准重新 ACF 检测 (10% 阈值, MAX=3)
5. Pearson 相干性过滤 (|r| ≥ 0.02)
6. 边界保护：多径绝对时间 < 音频总长度
7. 重算 Delay_us、AmpRatio、Coherence
8. 值替换写入 all_pulse_times.txt 和 pulse_params.txt

### 校准结果
| 指标 | 值 |
|------|-----|
| 校准前异常 Click | 355 |
| 成功校准 | 355 |
| 新检出多径 | 558 |
| 最终直达径 | 18116 |
| 最终多径 | 15260 |
| 校准后 AmpRatio<0.9 | 0 |
| 耗时 | 2.6s |

### 输出文件
- `11_ChannelMetrics/recalibrate_fix_v2.py`: 校准脚本
- `11_ChannelMetrics/anomalous_ampratio_final.txt`: 残留异常明细 (空)
- `10_AutoCorrMultipath/all_pulse_times.txt`: 已更新
- `11_ChannelMetrics/pulse_params.txt`: 已更新

## 2026-06-27 — 参数回退: 删除多维测距、恢复检测条件

- 删除 `12_PulseClustering/` 文件夹 (多维距离测度聚类)
- MAX_MULTIPATH 回归 8→3, ACF 阈值回归 2%→10%
- 以上全部记录为回退后最终状态
