# 实验记录

## 2026-08-06：Positive_Clicks_3500召回型多径二次实验

### 原因

用户在第一版诊断图中发现多条明显延迟到达未被选中。统计表明SNR 6 dB并非主要瓶颈，主要限制来自MAD峰门限5/3、评分0.60以及1 ms最大时延。

### 新配置

- 最大时延：1 ms → 2 ms。
- MAD峰高/显著性：5/3 → 4/2。
- NCC：0.55 → 0.50；频谱余弦：0.70 → 0.60；综合评分：0.60 → 0.55。
- SNR仍为6 dB；片段50 μs、中心半宽0.5 ms、保护区100 μs、权重0.20/0.50/0.30保持不变。
- 原精确率优先Notebook和结果不覆盖；新增 `17_20260805/multiExtract_tuned_3500_recall.ipynb` 与 `17_20260805/multipath_results_3500_recall`。

### 全量消融与结果

| 配置 | 多径数 | 0条文件 | 1条文件 | 2条文件 |
|---|---:|---:|---:|---:|
| 严格门限+1 ms | 1213 | 2426 | 935 | 139 |
| 严格门限+2 ms | 2278 | 1761 | 1200 | 539 |
| 放宽门限+1 ms | 2015 | 1804 | 1377 | 319 |
| 放宽门限+2 ms | 3625 | 1029 | 1317 | 1154 |

- 3500/3500条处理成功，错误0；导出7125个576 kHz、单通道PCM16、29采样点路径WAV，自动结构验收通过。
- 相对1 ms严格基线，1864个文件增加路径、6个文件减少路径。
- 1–2 ms区间有339条路径；335条可取得下一次IPI，均未达到下一次click时间，最小已知下一次IPI为8.8 ms。
- 固定200条审查集生成400张诊断图并完成联系表目视检查。新配置提高了可见多径召回，但旧ACF精确率代理下降；由于无人工逐路径真值，不宣称正式准确率。
- 合成测试仍为0采样点延迟误差、±50 μs外振铃0.004419。

### 新增输出

- `full_ablation.csv`：全量1/2 ms与严格/放宽门限消融。
- `baseline_comparison.csv`：逐文件新旧多径数变化。
- `delay_ipi_audit.csv`：路径延迟与下一次IPI对照。

## 2026-08-06：Positive_Clicks_3500定向滤波与多径批量提取

### 目的

在主到达—延迟到达提取前抑制低频底噪和已知200 kHz人工定位脉冲，同时保留宽带click主体并导出滤波后的识别片段。

### 实现与选择

- 新建 `D:\Project_Github\Indo-Pacific-humpback-dolphin\17_20260805\multiExtract_tuned_3500.ipynb`，未覆盖 `multiExtract.ipynb`。
- 滤波采用20 kHz二阶原型Butterworth高通与185–215 kHz二阶原型Butterworth带阻，SOS形式、`sosfiltfilt`零相位；选择较宽带阻是因为人工干扰峰实测约188–206 kHz，低阶方案的带外振铃更小。
- 参数实验覆盖片段40/50/75/100 μs、中心窗0.5/1.0/1.5 ms、保护区75/100/150 μs、最大时延1/2/3/4 ms及多组门限和权重。
- 最终使用50 μs、中心窗0.5 ms、保护区100 μs、最大时延1 ms、MAD 5/3、SNR 6 dB、NCC 0.55、频谱余弦0.70、评分0.60、能量/NCC/频谱权重0.20/0.50/0.30。选择1 ms而非2–4 ms，主要为了降低下一次独立click被当作多径的风险；选择严格相似度门限和较低能量权重，采用精确率优先策略。
- 备用局部峰仅写入候选表并标记 `fallback_diagnostic_only`，不参与最终路径选择。

### 验证结果

- 合成宽带click及延迟副本定位误差0采样点；±50 μs外滤波振铃比例0.004419，未超过0.005门限。
- 全量3500条全部成功，无错误；共18317条候选、1213条多径，0/1/2条多径文件数为2426/935/139。
- 导出4713个滤波路径WAV，全部为576 kHz、单通道PCM16、29采样点，CSV索引和文件一致，缺失或格式错误均为0。
- 383条强200 kHz背景样本的185–215 kHz衰减中位数为22.778 dB；背景RMS中位下降4.699 dB；30–180 kHz频谱余弦第5百分位为0.999145。
- 固定200条审查集生成400张诊断图并完成单人目视筛查。由于没有逐路径人工真值/双人盲审，98%主到达正确率和90%多径视觉可信率未作为独立统计准确率宣称，后续正式标注可直接基于诊断图、`all_candidates.csv`和`selected_paths.csv`复核。

### 输出

`D:\Project_Github\Indo-Pacific-humpback-dolphin\17_20260805\multipath_results_3500`

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

## 2026-06-30 — 13_Negative: Hard Negatives 自动挖掘脚本

### 完成内容
- 新建 `13_Negative/` 模块，用于从 `00_Data/original_audio/*.wav` 长录音中自动挖掘难负样本脉冲
- 默认保护索引改为 `00_Data/AllTrain.csv`，按 `Ori_file_num(No.)` 分组构建保护区间，覆盖 click、buzz、burst pulse 等不可作为噪声的片段
- 新增三份代码：
  - `13_Negative/masking.py`: 解析 AllTrain.csv，按 `Train_start(s)` / `Train_end(s)` 扩展前后 10ms，构建 IntervalTree 保护带
  - `13_Negative/filtering.py`: 带通滤波、TEO、频谱质心、归一化 ACF 次峰和 Pearson 相干性校验
  - `13_Negative/mine_negatives.py`: 分块读取 2GB 级长录音，多进程文件级调度，保存 `Negative_Samples/*.wav` 和 `negative_samples_log.csv`

### 算法流程
1. AllTrain.csv 中每条记录转换为毫秒区间，并前后扩展 10ms 作为排他保护带
2. 对长录音分块估计平滑 TEO 第 40 百分位数，使用 3 倍底噪阈值触发候选锚点
3. 对候选锚点执行保护区间排除，命中 AllTrain 保护带则跳过
4. 截取锚点 ±50μs 计算频谱质心，剔除低频易负样本
5. 在锚点后 5ms 窗口计算 ACF，检测 0.1ms 到 5ms 次峰，并用 Pearson |r| >= 0.02 校验
6. 有合格次峰标记为 `Type_3_Multipath_Noise`，否则标记为 `Type_4_Isolated_Noise`
7. 以锚点为中心保存 ±5ms wav，并汇总元数据 CSV

### 验证结果
- `D:\Python_env\toothwhale\python.exe -m compileall 13_Negative` 通过
- `mine_negatives.py --help` 通过
- `AllTrain.csv` 解析通过：34 个原始录音编号，共 897 个保护区间
- 已安装 `intervaltree==3.2.1`
- 全量运行完成：35 个原始 wav，目标总量 90,580（18116 click 的 5 倍），每个 wav 均衡保存 2,588 个
- 运行参数：`--workers 2 --block-seconds 5 --target-total 90580 --teo-threshold-factor 5 --max-candidates-per-block 20 --min-free-gb 10`
- 输出统计：`Type_3_Multipath_Noise` 64,133 个，`Type_4_Isolated_Noise` 26,447 个
- 输出体积：90,580 个 wav，共约 1.05 GB；运行耗时 1,199.3 s
- 结果一致性检查：`negative_samples_log.csv` 90,580 行，`Negative_Samples/*.wav` 90,580 个

### 上下游关系
- 上游依赖：`00_Data/AllTrain.csv`、`00_Data/original_audio/*.wav`
- 下游用途：为后续正负样本分类、困难负样本扩充和误检分析提供噪声脉冲片段

## 2026-07-01 — 13_Negative: 20-190 kHz 带通与去除 HFER 后重挖

### 完成内容
- 在负样本挖掘预处理中加入 Butterworth 带通滤波，默认保留 `20-190 kHz`，滤除 `20 kHz` 以下与 `190 kHz` 以上成分
- `mine_negatives.py` 支持 `--sample-rate 0` 自动从原始 wav 读取采样率；本轮读取结果为 `576000 Hz`
- 根据带通后 HFER 失去区分度的问题，移除高低频能量比指标：
  - `filtering.py` 不再计算或返回 HFER
  - `mine_negatives.py` 不再在候选、日志 CSV 中保存 HFER
  - `negative_samples_log.csv` 列变为 `Audio_File, Noise_Timestamp(s), Centroid, Noise_Type`

### 运行参数
- 输出目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin\13_Negative_Bandpass_20k_190k_NoHFER`
- 命令参数：`--workers 2 --block-seconds 5 --target-total 90580 --teo-threshold-factor 5 --max-candidates-per-block 20 --min-free-gb 10 --sample-rate 0 --bandpass-low-hz 20000 --bandpass-high-hz 190000 --bandpass-order 6`

### 验证结果
- `D:\Python_env\toothwhale\python.exe -m compileall 13_Negative` 通过
- `mine_negatives.py --help` 通过
- 全量运行完成：35 个原始 wav，输出 `90,580` 个负样本 wav
- 类型统计：`Type_3_Multipath_Noise` 62,356 个，`Type_4_Isolated_Noise` 28,224 个
- 耗时：2,111.4 s
- 一致性检查：`negative_samples_log.csv` 90,580 行，`Negative_Samples/*.wav` 90,580 个
- CSV 列检查：无 HFER，仅保留 `Audio_File`、`Noise_Timestamp(s)`、`Centroid`、`Noise_Type`
- wav 抽检：采样率 `576000 Hz`，单通道

### 上下游关系
- 上游依赖：`00_Data/AllTrain.csv`、`00_Data/original_audio/*.wav`
- 替代 2026-06-30 版本作为当前推荐负样本集；2026-07-01 早先输出到 `13_Negative_Bandpass_20k_190k` 的中间结果是在 HFER 尚未移除时产生，不作为最终结果

## 2026-07-18 — 10_AutoCorrMultipath & 11_ChannelMetrics: 前置 RMS 质心锚点校准后重算

### 完成内容
- 将 `11_ChannelMetrics/recalibrate_fix_v2.py` 从“仅修复 AmpRatio<0.9 异常 Click”的后处理脚本，调整为全量前置直达锚点校准脚本
- 新生成 `10_AutoCorrMultipath/calibrated_anchors.txt`，为每个 Click 保存 RMS 质心法得到的校准直达时间、锚点样本索引和采样率
- 更新 `10_AutoCorrMultipath/batch_detect.py`，多径检测不再假设直达固定为窗口中心 5.0000ms，而是读取 `calibrated_anchors.txt` 中的校准锚点
- 更新 `11_ChannelMetrics/calc_pulse_params.py`，信道参数计算使用 `all_pulse_times.txt` 第三列中的校准直达时间，按校准锚点计算 Delay_us、AmpRatio 和 Coherence
- 更新 `multipath-detection-method.md` 与 `14/README.md`，说明当前流水线与正样本表统计

### 关键指标与结果
| 指标 | 值 |
|------|-----|
| 校准 Click 数 | 18116 |
| 缺失锚点 | 0 |
| 总 Click | 18116 |
| 检出多径 Click | 10510 (58.0%) |
| 孤立 Click | 7606 |
| 多径总数 | 15373 |
| 平均每 Click 多径数 | 0.85 |
| `pulse_params.txt` 直达径行数 | 18116 |
| `pulse_params.txt` 多径行数 | 15373 |
| `pulse_params.txt` 数据行数 | 33489 |
| AmpRatio<0.9 残留异常径 | 0 |

### 输出文件
- `10_AutoCorrMultipath/calibrated_anchors.txt`: 18116 行校准锚点记录
- `10_AutoCorrMultipath/all_pulse_times.txt`: 已基于校准锚点重跑多径检测
- `11_ChannelMetrics/pulse_params.txt`: 已基于校准直达时间重算信道参数
- `11_ChannelMetrics/anomalous_ampratio_final.txt`: 残留 AmpRatio<0.9 异常径统计，当前为 0
- `14/README.md`: 已记录同步正样本表的统计信息
- `multipath-detection-method.md`: 已记录“先校准锚点，再检测多径，再计算信道参数”的当前方法

### 上下游关系
- 上游依赖：`09_Locate/PulseTrain_XXX/Click_YYY/Pulse_YYY.wav`
- 当前流水线顺序：`09_Locate` → `11_ChannelMetrics/recalibrate_fix_v2.py` 生成锚点 → `10_AutoCorrMultipath/batch_detect.py` 检测多径 → `11_ChannelMetrics/calc_pulse_params.py` 计算信道参数
- 下游用途：`14/` 正样本表、后续正负样本分类、混响/多径分析与方法文档均应使用本轮校准锚点后的 `pulse_params.txt`

## 2026-07-21 — 15: Click 分段聚类 CSV 参数时序绘图脚本

### 完成内容
- 新建 `15/plot_csv_parameters_over_time.py`，用于读取 click 分段聚类 CSV，并绘制 RMS、归一化互相关峰值、谱质心、频谱带宽和 DBSCAN 原始标签随片段中心时间变化的图。
- 默认输入为 `12_ClickSegmentClustering/outputs_100us/Pulse_027_segments_dbscan.csv`，输出为 `15/outputs/Pulse_027_parameters_over_time.png`。
- 新增 `15/.gitignore`，忽略输出图、Matplotlib 缓存和 `__pycache__`。

### 关键指标与结果
- 运行环境：`D:\Python_env\toothwhale\python.exe`。
- 输入 CSV 行数：99。
- 时间范围：0.050 ms 到 9.918 ms。
- DBSCAN 标签计数：`-1: 22`, `0: 44`, `1: 28`, `2: 5`。
- 语法检查通过：`python -m py_compile 15/plot_csv_parameters_over_time.py`。

### 上下游关系
- 上游依赖：`12_ClickSegmentClustering` 生成的分段聚类 CSV。
- 下游用途：用于快速检查分段参数时序变化和聚类标签在时间轴上的分布。

## 2026-08-08 — 09_Locate：Pulse 定位约束、批量误差审计与精确位置清单

### 任务目标
- 在不重新生成裁剪 WAV、逐 Pulse 波形图和 TEO/自相关分析图的前提下，评估 Pulse 定位方法的样本误差和毫秒误差。
- 将单个 PulseTrain 的校验逻辑扩展到批量处理，定位误差较大的 PulseTrain，并输出可追溯到 `Voyage_Ori`、PulseTrain 和原始 `Pulse_N.wav` 的精确位置清单。
- 将批量脚本转换为可独立运行的 Jupyter Notebook。

### MATLAB 单文件定位与 Report
- 修改 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\locate.m`，使用块注释保留但停用第 7、8 模块中的裁剪 WAV、逐 Pulse PNG 和逐 Pulse TEO/自相关 PNG 保存代码。
- 保留理论位置生成、TEO、互相关和可视化计算；Report 使用原始 `Pulse_N.wav` 在 `PulseTrain.wav` 中的唯一匹配位置及其 TEO 峰作为参考，不修改 `theoretical_samples`。
- 在 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\Locate_Position_Report` 生成 `pulse_position_report.csv`、`pulse_position_summary.png` 和 `pulse_position_summary.txt`，并在 MATLAB 输出窗口显示总体进度。
- 修正互相关滞后选择：候选 lag 必须使完整 theoretical pulse 模板落在 PulseTrain 范围内，因此首个位置不得小于 1，最后一个 pulse 不得超过 PulseTrain 总样本数。
- `PulseTrain_545` 修正前因无边界约束选择 lag `-8418`，产生约 `-14.7147 ms` 偏移；修正后 lag 为 `58`，预测位置 `[59, 15231, 30270]`，参考位置 `[57, 15231, 30271]`，误差为 `[2, 0, -1]` samples，MAE 为 `0.001736111 ms`，最大绝对误差为 `0.003472222 ms`。
- `PulseTrain_737` 验证得到 `43/43` 个唯一参考位置，MAE 为 `0.010578165 ms`，RMSE 为 `0.012367179 ms`，最大绝对误差为 `0.022569444 ms`。

### Python 批量定位与误差报告
- 修改 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\locate.py`，应用相同的完整模板边界约束，并注释全部旧有切片和逐 Pulse 图片保存调用。
- 针对 FLOAT PulseTrain 与 PCM16 Pulse 文件无法逐样本精确匹配的情况，增加受限局部归一化互相关（NCC）参考定位：搜索半径为最近 IPI 的 45%，下限 2 ms、上限 50 ms，最低接受分数为 0.99；Report 中分别记录 `exact` 和 `ncc` 方法及匹配分数。
- 增加每个 PulseTrain 的误差标准差、每 Pulse 误差斜率、系统性固定偏移标记、缺失参考数量、可用参考数量和基于可用参考的匹配率。
- 在 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\Locate_Batch_Report` 生成：
  - `pulse_location_pulse_report.csv`
  - `pulse_location_train_summary.csv`
  - `pulse_location_batch_summary.png`
  - `pulse_location_batch_summary.txt`
- 最终批量运行处理 `832` 个 PulseTrain，生成 `9405` 个 exact 参考和 `8998` 个 NCC 参考，且完整模板边界约束阻止了 `29` 次旧逻辑可能产生的越界/不完整对齐。
- `PulseTrain_436` 缺少的 `14` 个 WAV 按要求忽略：总 pulse 数 `37`，可用参考 `23`，匹配 `23/23`，`match_rate=1.0`，状态为 `ok`，不列为问题 PulseTrain。
- 当前数据中 `PulseTrain_648` 有 `50/53` 个参考位置可用，是唯一参考不完整的 PulseTrain；`PulseTrain_822` 当前检查正常。

### 高误差诊断与排除规则
- 使用稳健阈值 `median + 3 × 1.4826 × MAD` 识别高 MAE，当前阈值为 `0.013086921 ms`；共识别 `50` 个高 MAE PulseTrain，其中 `32` 个表现为系统性固定偏移。
- 前 7 个高误差 PulseTrain 为：`PulseTrain_513`、`PulseTrain_511`、`PulseTrain_691`、`PulseTrain_039`、`PulseTrain_067`、`PulseTrain_290`、`PulseTrain_286`。
- 对应 MAE 分别为 `0.322569`、`0.289236`、`0.200893`、`0.188657`、`0.178627`、`0.144495`、`0.093056 ms`；误差标准差分别为 `0.005424`、`0.001770`、`0.003679`、`0.003374`、`0.006257`、`0.006037`、`0.002355 ms`。
- 诊断结果表明这些误差主要是稳定的正向固定偏移，不是随 Pulse 序号累积的 IPI 漂移；NCC 均高于 `0.9993`。Delta TEO 模板倾向选择延后的重复 TEO 结构，个别强峰会主导总相关分数。
- 在 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\locate.py` 中加入固定 `EXCLUDED_PULSE_TRAINS` 集合；仅在精确位置清单中排除上述 7 个 PulseTrain，不删除或改动原始数据。

### 精确位置清单
- 扫描 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate` 下的 `22` 个 `Voyage_Ori_*` 目录，建立了覆盖 `832/832` 个 PulseTrain、无重复归属的 Voyage 映射。
- 生成 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\Locate_Batch_Report\pulse_precise_positions.txt`，采用 UTF-8 BOM 和制表符分隔格式。
- 每条记录包含 `Voyage_Ori`、`PulseTrain`、`PulseNumber`、`PositionSample_1based`、`PositionTime_s`、`PositionTime_ms`、`SampleRate_Hz`、`ReferenceFile`、`MatchMethod` 和 `MatchScore`。
- 精确位置定义为原始 `Pulse_N.wav` 的 TEO 峰在对应 `PulseTrain.wav` 中的位置。
- 最终保留 `825` 个 PulseTrain 的 `18301` 个有效 Pulse 位置，其中 `9405` 个来自 exact 匹配、`8896` 个来自 NCC 匹配；排除的 7 个 PulseTrain 共含 `102` 个 Pulse，另有 `17` 个不可用位置（`PulseTrain_436` 缺少 14 个 WAV，`PulseTrain_648` 有 3 个参考不可用）。

### Notebook 转换与验证
- 新建可独立运行的 `D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\locate.ipynb`，共 `37` 个单元格（`18` 个代码单元格、`19` 个 Markdown 单元格），完整包含 `locate.py` 的 `16` 个顶层函数，不依赖 `%run locate.py`。
- Notebook 最后一个单元格调用 `main()`；通过 `nbformat 5.10.4` 格式验证，并在 `D:\Python_env\toothwhale\python.exe` 环境中成功执行全部定义单元格。
- 验证 Voyage 映射数量为 `832`，固定排除列表为 7 个目标 PulseTrain；Python 全量批处理耗时约 `1 分 47 秒至 1 分 57 秒`。
- 运行期间未新建或更新裁剪 WAV、逐 Pulse 波形图或逐 Pulse 分析图，已有切片和分析目录内容保持不变。

## 2026-08-14 — 19_20260813：BCSC–Transformer 二分类与捷径学习实验

### 任务与实现
- 新建并完整执行 `D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\bcsc_transformer_binary_classification.ipynb`；核心可复用实现位于 `D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\bcsc_transformer_pipeline.py`。
- 使用 `00_Data\Positive_Clicks_3500` 的3500条正样本；按 `Ori_file_num` 数量及录音内 `Noise_Type` 分层，从 `14\negative_samples_log.csv` 匹配抽取3500条负样本，并从原始长录音重新截取5761点波形。
- 两类统一执行去直流、20–190 kHz五阶零相位带通、Tukey边缘渐变和峰值归一化。数据按原始录音编号隔离为5折；同一来源不跨训练、验证和测试集合。
- BCSC使用85 μs原子、`lambda_ratio=0.075`、35轮外迭代、120轮FISTA、30轮字典更新、6–72 kHz原子带宽；每条样本执行8次固定原子残差扰动复核，按7/8复现及±0.03 ms匹配判定稳定到达。
- 完成波形端到端、直达径+多径显式特征、仅直达径显式特征三组Transformer的来源隔离五折训练，并执行录音组Bootstrap、配对差值、波形反事实、梯度×输入归因和多径数值清零消融。

### 数据与BCSC验收
| 指标 | 结果 |
|------|------|
| 正/负样本 | 3500 / 3500 |
| 采样率、通道、点数 | 576 kHz、单通道、5761点，全部通过 |
| 负样本最小时间间隔 | 10.444 ms |
| 来源组泄漏 | 0 |
| BCSC可追踪记录 | 7000/7000 |
| BCSC状态为 `ok` | 6995/7000（99.93%） |
| 不稳定直达候选 | 5条，均保留状态，未静默删除 |
| 高置信度次到达候选 | 5237条 |
| 正样本高置信度多径检出率 | 57.17% |
| 负样本稳定次到达候选检出率 | 92.46% |
| 合成时延定位误差 | 0.0000 ms（阈值≤0.03 ms） |

### 三组主结果
| 模型 | AUROC均值 [录音组Bootstrap 95% CI] | AUPRC | 平衡准确率 | F1 |
|------|------:|------:|------:|------:|
| 波形端到端 Transformer | 0.9803 [0.9713, 0.9864] | 0.9779 | 0.9355 | 0.9360 |
| 直达径+多径显式特征 Transformer | 0.9403 [0.9112, 0.9474] | 0.9143 | 0.8854 | 0.8914 |
| 仅直达径显式特征 Transformer | 0.9301 [0.9062, 0.9429] | 0.8990 | 0.8720 | 0.8793 |

### 研究结论与消融
- 波形模型仅背景残差五折平均AUROC为0.9265；跨类别且RMS匹配的背景交换后AUROC为0.8506，预测翻转率18.19%，平均绝对概率变化0.1785。梯度×输入归因的直达径、多径、背景平均贡献为47.49%、7.74%、44.77%。这些结果提供背景/多径捷径证据。
- 移除高置信度多径后的波形AUROC为0.9803，与完整输入基本相同；当前捷径证据主要来自背景残差与背景交换，而不是单独移除所选多径分量。
- 直达径+多径相对仅直达径的配对汇总差值：AUROC +0.0067，95% CI [-0.0039, 0.0164]；AUPRC +0.0076，95% CI [-0.0089, 0.0278]；平衡准确率 +0.0134，95% CI [0.0065, 0.0205]。AUROC和AUPRC置信区间包含0，不能宣称统计上明确的增益。
- 保留多径掩码但清零多径数值特征后，五折平均AUROC由0.9403降至0.9163；稳定次到达检出子集上由0.9468降至0.9219。该消融支持多径数值属性含有辅助信息，但不能把算法稳定性解释为物理路径真实性。

### 产物
- 已执行Notebook：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\bcsc_transformer_binary_classification.ipynb`，8个单元格，3个代码单元格均有执行计数和输出。
- 研究报告：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs\research_report.md`。
- 输出包括数据清单/折分、BCSC路径表、75维显式特征、21000条测试预测、15个模型权重、折指标、配对差值、反事实预测与指标、归因、多径消融及3张PNG图。
- 配置哈希：`33ce6ffa3ab921bc`；运行环境：`D:\Python_env\toothwhale\python.exe`；GPU：NVIDIA GeForce RTX 4060。

## 2026-08-18：信号—噪声可分性与背景捷径确认实验

### 目的
独立于原BCSC–Transformer实验，区分当前高AUROC来自“click信号本身可分”“正负背景差异捷径”还是“BCSC残差泄漏”。

### 交付与位置
- Notebook：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\signal_noise_shortcut_audit.ipynb`
- 流水线：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\shortcut_audit_pipeline.py`
- 报告：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs_shortcut_audit\research_report.md`
- 输出目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs_shortcut_audit`

### 关键结果
- 正样本原始时间映射3500/3500；全部波形576 kHz、单通道、5761点且有限。
- 原队列与严格匹配队列来源组泄漏均为0；匹配负样本10 ms重叠违反为0。
- 严格匹配3370对（最低验收3150对）；6个匹配变量中局部SNR绝对SMD=0.193>0.10，按预注册标记严格匹配不可行并按实际规模继续。
- 七条件AUROC（录音组Bootstrap 95% CI）：历史0.9797、对称重截取0.8759、严格匹配全波形0.6346、信号区0.8279、BCSC残差0.5017、宽窗口背景0.4976、随机遮罩0.5176。
- 预注册对比：历史vs对称+0.104（制作流程捷径）；对称vs严格匹配+0.241（SNR/粗统计不平衡）；严格匹配全波形vs信号区-0.193（背景未提供额外信息、信号形态可分）；BCSC残差vs宽窗口背景AUROC差值不显著；宽窗口背景vs 0.5不显著（无背景类别信息）。
- BCSC合成直达+0.35 ms副本误差0.0007 ms≤0.03 ms；遮罩合成测试通过（未遮罩不变、边界连续、NCC下降98.74%、随机遮罩不覆盖注入路径）。
- 原实验文件与 `outputs` 未修改；历史基准只读。