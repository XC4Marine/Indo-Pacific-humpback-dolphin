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
```

## 数据特点

| 特征 | 描述 |
|------|------|
| 采样格式 | .wav 单通道音频 |
| 物种 | 中华白海豚 (Sousa chinensis) |
| 信号类型 | 高频 click 脉冲串 |
| 每条脉冲串 | 含 2~100+ 个脉冲，由 PulseParameters.txt 记录各脉冲参数 |
| 关键参数 | SNR、IPI（脉冲间隔）、SPLpp、Duration、Fpeak、带宽 |
| 混响分类 | 三级：Clean（无混响）/ Moderate（中度）/ Severe（严重） |
| 野外数据 | Results.csv 包含日期/季节/经纬度/水深/海豚群体大小/行为 |

## 项目目标（推断）
1. 从原始水听器录音中检测出海豚的 click 脉冲串
2. 对每个 click 进行参数提取（持续时间、峰值频率、带宽等）
3. 区分真实 click 与海洋环境噪声（假阳性检测）
4. 对 click 进行混响严重程度分类
5. 最终建立自动化的海豚声学监测与分类系统

## 当前进度（2026-06-26）
- [x] 01~06 模块：已有 Jupyter Notebook 实现，任务基本完成
- [x] 07 模块：预实验已完成，含错误分析
- [x] 08 模块：混响分离（待验证）
- [x] 09 模块：locate.m MATLAB 代码已移植为 locate.py（Python），处理完 832 个 PulseTrain 共 18116 个 click