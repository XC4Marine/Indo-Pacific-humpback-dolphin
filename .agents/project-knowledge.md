# 项目认知文档

## 项目概述
本项目聚焦于**中华白海豚（Indo-Pacific humpback dolphin, Sousa chinensis）的水下声学信号处理与机器学习分析**。核心任务是检测、分类、定位海豚发出的高频脉冲信号（clicks），并通过深度学习等方法进行混响分类与脉冲串定位。

## 目录结构与功能分布

```
00_Data/                         # 原始数据目录（不入 Git）
├── ClickTrains/PulseTrain_001/  # 832个脉冲串文件夹
│   ├── PulseTrain.wav           # 完整脉冲串音频
│   ├── PulseParameters.txt      # 每个脉冲的特征参数（SNR/IPI/SPL等）
│   ├── Pulse_N.wav              # 单个脉冲音频（MATLAB预提取）
│   └── Pulse_N_Waveform.png / _Psd.png  # 波形图和功率谱图
├── ClickTrains.csv              # 834条脉冲串元数据汇总
├── AllTrain.csv                 # 897条全部脉冲串记录（含时间间隔）
├── Results.csv                  # 野外调查记录（日期/季节/坐标/水深/海豚行为等）
├── original_audio/              # 原始水听器录音文件（Ori_Recording_XX.wav）
└── 02_ClickDetection/           # 假脉冲检测负样本

01_ClickDuration/                # Click持续时间分析（Jupyter Notebook）
02_ClickDetection/               # 假脉冲检测（Jupyter Notebook）
03_Distinguish/                  # 混响级别分类：Clean / Moderate / Severe
04_Classification/               # 正负样本分类实验（SVM/XGBoost/聚类等）
05_DistinguishbyEnergy/          # 基于能量的混响区分
06_DataSplit/                    # 数据集划分（10折交叉验证）
07_PreTrial/                     # 预实验与错误分析
08_ReverbSeparation/             # 混响分离
09_Locate/                       # 脉冲串定位（MATLAB→Python 迁移）
10_AutoCorrMultipath/            # 多径检测 (ACF + Teager + Pearson三层过滤, 10%阈值, MAX=3)
├── autocorr_detect.py           # 100样本实验脚本
├── batch_detect.py              # 全量批处理脚本
└── all_pulse_times.txt          # 18116 Click 直达/多径时间表
11_ChannelMetrics/               # 信道参数计算与异常校准
├── calc_pulse_params.py         # 信道参数计算 (Delay_us, AmpRatio, Coherence)
├── recalibrate_fix_v2.py        # RMS质心法锚点校准脚本
├── pulse_params.txt             # 33376行信道参数 (18116直达 + 15259多径)
└── anomalous_ampratio_final.txt # AmpRatio<0.9 残留异常 (空)
13_Negative/                     # 长录音 Hard Negatives 自动挖掘
├── masking.py                   # 读取 AllTrain.csv 并构建按原始录音编号分组的保护区间树
├── filtering.py                 # 带通滤波、TEO、频谱质心、ACF次峰与Pearson相干性校验
└── mine_negatives.py            # 多进程文件级调度, 导出 negative_samples_log.csv 与 Negative_Samples/*.wav
```

## 数据特点

| 特征 | 描述 |
|------|------|
| 采样格式 | .wav 单通道音频, 576kHz |
| 物种 | 中华白海豚 (Sousa chinensis) |
| 信号类型 | 高频 click 脉冲串 |
| 每条脉冲串 | 含 2~100+ 个脉冲，由 PulseParameters.txt 记录各脉冲参数 |
| 关键参数 | SNR、IPI（脉冲间隔）、SPLpp、Duration、Fpeak、带宽 |
| 混响分类 | 三级：Clean（无混响）/ Moderate（中度）/ Severe（严重） |
| 野外数据 | Results.csv 包含日期/季节/经纬度/水深/海豚群体大小/行为 |
| 负样本挖掘保护索引 | 13_Negative 默认使用 AllTrain.csv，覆盖 click、buzz、burst pulse 所在时段，避免将其误当噪声 |
| 当前负样本预处理 | 20-190 kHz Butterworth 带通；不再使用 HFER 高低频能量比 |

## 项目目标
1. 从原始水听器录音中检测出海豚的 click 脉冲串
2. 对每个 click 进行参数提取（持续时间、峰值频率、带宽等）
3. 区分真实 click 与海洋环境噪声（假阳性检测）
4. 对 click 进行混响严重程度分类
5. 最终建立自动化的海豚声学监测与分类系统

## 多径检测流水线 (10_AutoCorrMultipath + 11_ChannelMetrics)

### 检测三层过滤
1. **ACF 次峰初筛**: 归一化自相关, 10% 阈值, |lag|∈[0.1ms, 5ms], 最多 3 个
2. **Teager 交叉验证** (仅实验脚本): Teager 能量 ≥ 全局最大值 0.1%
3. **Pearson 相干性**: 直达窗与多径候窗的 Pearson r, |r| ≥ 0.02

### 锚点校准
- RMS 质心法: 50μs 滑动窗 RMS² argmax 定位最优直达锚点

### 信道三个参数
- Delay_us: 多径相对直达延迟
- AmpRatio: RMS(直达) / RMS(多径)
- Coherence: Pearson r (直达窗, 多径窗)

## 编码环境
- Python 3.11.15 @ D:\Python_env\toothwhale\python.exe
- 依赖：numpy, scipy, matplotlib, soundfile, pandas, tqdm, pathlib, intervaltree

## 当前进度（2026-07-01）
- [x] 01~09 模块：完成
- [x] 10_AutoCorrMultipath: 三层过滤多径检测 (10%阈值, MAX=3), 10519/18116 Click 检出多径, 15259 多径
- [x] 11_ChannelMetrics: 18116直达 + 15260多径信道参数, 355异常Click RMS质心校准完成, AmpRatio<0.9 → 0
- [x] 13_Negative: 长录音 Hard Negatives 自动挖掘完成，默认以 AllTrain.csv 构建排他保护带；当前推荐结果为 `13_Negative_Bandpass_20k_190k_NoHFER`，输出 90,580 个负样本，CSV 不含 HFER
