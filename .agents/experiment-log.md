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