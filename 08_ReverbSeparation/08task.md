# 数据描述：
代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data

文件夹D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains\PulseTrain_020中，PulseTrain.wav中含有很多个Pulse，已经提取出来了，文件命名方式为：Pulse_001.wav。该文件夹下除了PulseTrain.wav，其余都是单个Pulse的wav文件。
请使用互相关的方式定位出每一个pulse在pulse_train中的位置，并在pulse_train的波形图中标注出来。
同时，画出一张pulse_train.wav的Teager算子图，统计采用如下方式进行信号检测的话，会检测出哪些pulse，并统计出pulse数量，在一张新的PulseTrain.wav上标注出来：

1. **频率粗筛**：
	- 使用64 点快速傅里叶变换（FFT）计算频谱，50%重叠，hann窗。
	- 采用**谱均值减法**（Spectral means subtraction）获得噪声阈值。
	- 设定阈值：如果在 15–95 kHz 频带内，超过 12.5% 的频率箱（frequency bins）强度超过阈值 13 dB，则标记为疑似信号。
	- 候选click的前后各7.5ms都被记录下来，方便后续计算谱均值。重叠的click被融合。
2. **时域精确定位（寻找开始、结束点）**：
    - **能量算子**：使用[[Teager能量算子]](Teager Energy Operator, TEO)进行计算。传统的能量计算通常取平方和，而 Teager 算子能够捕捉信号的**瞬时能量变化**，对脉冲性极强的点击信号非常敏感。
    - **确定噪声基准**：选取 Teager 能量值的**第 40 百分位数**（基于对数据的经验分析）作为该段数据的噪声基准。
    - **寻找信号的核心与边界**：
	    - **核心**：所有 Teager 能量超过噪声基准 **100 倍**的点被标记为信号点。如果两个点之间的时间距离小于 **200 微秒**，则被合并视为同一个点击信号。
	    - **Click选择**：为防止信号重叠、降低冗余度，需要对Click信号优中选优。每15ms数据中选择能量最强的click。
	    - **确定起止点**：为包含混响时间、考虑波形的不对成性，首先对 Teager 能量进行 **10 点滑动平均**；系统向信号核心的前后搜索，直到能量值降至噪声基准的 **3 倍**以下。这两个点即被定义为该次点击信号的精确**开始**和**结束**位置。