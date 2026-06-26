# 实验记录

## 2026-06-26 — 09_Locate: MATLAB → Python 迁移

### 完成内容
- 将 `09_Locate/locate.m` 的 MATLAB 逻辑移植为 Python 脚本 `09_Locate/locate.py`
- 脚本功能：基于 IPI 模板互相关定位每个 click 的理论中心位置，截取 ±5ms 窗口
- 输出：每个 click 的 .wav 音频片段、波形图（灰曲线+红色中心线）、TEO+自相关分析图
- 处理结果：832 个 PulseTrain 全部处理完毕，共 18116 个 click 文件夹

### 算法流程
1. 读取 PulseTrain.wav + PulseParameters.txt（自动适应 UTF-8/GBK 编码）
2. 从 IPI(ms) 列构造理想脉冲模板向量
3. 计算 TEO（Teager Energy Operator）：x(n)^2 - x(n-1)*x(n+1)，并归一化
4. 全局互相关对齐（scipy.signal.correlate），得最佳滞后偏移
5. 对每个 pulse 中心位置截取 ±5ms，保存 wav/波形图/分析图

### 输出结构
```
09_Locate/
├── locate.py
└── PulseTrain_XXX/
    └── Click_YYY/
        ├── Pulse_YYY.wav
        ├── Waveform_YYY.png
        └── Analysis_YYY.png
```

### Python 环境
- Python 3.11.15 @ D:\Python_env\toothwhale\python.exe
- 依赖：numpy, scipy, matplotlib, soundfile, pandas, tqdm

### 与之前实验的关系
- 本模块位于数据处理流水线的后半段（脉冲串定位）
- 上游依赖：00_Data/ClickTrains 中的数据（由更早的 click 检测与参数提取步骤生成）
- 下游：定位后的 click 片段可用于后续的 04_Classification 分类实验或更深层分析
- 本脚本替代了 MATLAB 的手动单文件夹处理方式，实现了 832 个文件夹的批量自动处理
## 2026-06-26 — 10_AutoCorrMultipath: 基于自相关次峰法的直达/多径脉冲检测

### 完成内容
- 编写 `10_AutoCorrMultipath/autocorr_detect.py`：自相关次峰法实验脚本
  - 自相关计算（`numpy.correlate`，归一化使 Lag=0 处为 1.0）
  - 直达脉冲定位在 t=0（窗口中心，±100μs 标注）
  - 多径检测：|lag| > 0.1ms 区间内 `scipy.signal.find_peaks` 检测次峰
  - 自适应阈值判定：对比 5% 和 10% 两个候选阈值，选定 10%（平均 1.15 个/样本）
  - 多径数量约束：按相关系数降序，最多取 3 个
  - 可视化：波形图（上）+ 自相关图（下），绿色标注直达、蓝色标注多径，±100μs 半透明色带
- 100 样本实验：随机种子 12025，生成 100 张双图 + results.csv + summary.html
- 编写 `10_AutoCorrMultipath/batch_detect.py`：全量批处理脚本
- 对全部 18116 个 Click 片段运行检测，生成 `all_pulse_times.txt`

### 关键结果
| 指标 | 100 样本实验 | 全量 (18116) |
|------|-------------|-------------|
| 阈值 | 10% | 10% |
| 检出≥1个多径 | 69/100 (69.0%) | 10603/18116 (58.5%) |
| 多径总数 | 115 | 17153 |
| 平均每片段 | 1.15 | 0.95 |
| 耗时 | ~22s | ~51s |

### 算法参数
- 阈值：次峰高度 ≥ 主峰的 10%
- 最小滞后：0.1ms（排除直达峰邻近区域）
- 最小峰间距：30μs（find_peaks distance）
- 最大多径数：3（按相关系数降序选取）
- 前后标注范围：±100μs

### 输出结构
```
10_AutoCorrMultipath/
├── autocorr_detect.py          # 100 样本实验脚本
├── batch_detect.py             # 全量批处理脚本
├── results.csv                 # 100 样本详细结果
├── summary.html                # 100 样本可视化汇总
├── all_pulse_times.txt         # 全量结果（每行一个 Click）
└── plots/                      # 100 张波形+自相关双图
```

### all_pulse_times.txt 格式
- Tab 分隔：`PulseTrain_XXX Click_YYY 直达时间(ms) 多径1(ms) 多径2(ms) 多径3(ms)`
- 直达时间固定为 5.0000 ms（窗口中心）
- 多径时间 = 5.0000 + 自相关滞后偏移，无多径填 NaN

### 与之前实验的关系
- 上游依赖：`09_Locate` 切分的 18116 个 Click 片段（Pulse_*.wav, 576kHz, 5761 samples, ±5ms 窗口）
- 下游用途：脉冲时间信息可用于混响分类（直达 vs 多径分离）、多径结构统计、声传播路径分析