# 数据描述：
代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data
数据结构：
1.00_Data\original_audio文件夹下的数据
├── original_audio/
│   ├── Ori_Recording_01.wav        
│   ├── Ori_Recording_02.wav          
│   ├── ……
│   ├── Ori_Recording_35.wav

2. D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\AllTrain.csv表头及前三行
Train_num(No.),Ori_file_num(No.),Train_start(s),Train_end(s),Interval(s),Train_duration(ms),Num_of_pulses,Mean_SNR(dB),Mean_IPI(ms),Mean_SPLpp(dB),Mean_Duration(μs),Mean_Fpeak(kHz),Mean_Bandwidth_-3dB(kHz),Mean_Bandwidth_-10dB(kHz)
1,1,278.2543,278.7155,18.1177,461.2,4,30.2,152.99,156.66,38.19,58.33,57.21,103.42
2,1,296.8332,297.3997,62.7266,566.51,9,33,70.4,163.5,46.96,98.07,40.33,114.6
3,1,360.1263,360.5878,485.0108,461.45,3,32.7,230.62,158.07,33.54,118.17,36.26,83.98



# 任务描述与需求
AllTrain.csv文件中的Train_start(s),Train_end(s)记录了所有clickTrain的位置，对应的Ori_file_num(No.)保存在original_audio文件夹下；Num_of_pulses记录了每一个clickTrain中的click数量。现在用户需要找到这些音频文件中的瞬时但非click信号，以制作负标签供后续的分类。

请使用click检测算法，先检测出所有类似click的信号以及起始和结束位置。然后避开AllTrain.csv文件中的所有clickTrain（找到Train_start(s),Train_end(s)，分别向前向后扩展10s，在区域内以外检测负样本信号），每段音频recording中，都找到和真值click信号数量相等的负样本click。最终的负样本信号，保留以峰值为中心的200us，作为负样本。

由于original_audio文件夹下的音频时长都较长，采样率为576kHz，建议先读取一段音频，再将其进行5kHz-160kHz的带通滤波，再切分为1min的音频进行处理（不足1min就补零）。依次循环，直至完成所有音频中click的检测和负样本信号的提取。

由于Ori_Recording_12.wav中没有真值click，因此跳过该音频

# Click检测算法
1. **频率粗筛**：
	- 使用 1024 点（5.33ms）快速傅里叶变换（FFT）计算频谱，50%重叠，hann窗。
	- 采用**谱均值减法**（Spectral means subtraction）去除周围 3 秒内的背景环境噪声。
	- 设定阈值：如果在 15–95 kHz 频带内，超过 12.5% 的频率箱（frequency bins）强度超过阈值 13 dB，则标记为疑似信号。
	- 候选click的前后各7.5ms都被记录下来，方便后续计算谱均值。重叠的click被融合。
2. **时域精确定位（寻找开始、结束点）**：
    - **能量算子**：使用[[Teager能量算子]](Teager Energy Operator, TEO)进行计算。传统的能量计算通常取平方和，而 Teager 算子能够捕捉信号的**瞬时能量变化**，对脉冲性极强的点击信号非常敏感。
    - **确定噪声基准**：选取 Teager 能量值的**第 40 百分位数**（基于对数据的经验分析）作为该段数据的噪声基准。
    - **寻找信号的核心与边界**：
	    - **核心**：所有 Teager 能量超过噪声基准 **100 倍**的点被标记为信号点。如果两个点之间的时间距离小于 **200 微秒**，则被合并视为同一个点击信号。
	    - **Click选择**：为防止信号重叠、降低冗余度，需要对Click信号优中选优。每15ms数据中选择能量最强的click。
	    - **确定起止点**：为包含混响时间、考虑波形的不对成性，首先对 Teager 能量进行 **10 点滑动平均**；系统向信号核心的前后搜索，直到能量值降至噪声基准的 **3 倍**以下。这两个点即被定义为该次点击信号的精确**开始**和**结束**位置。



# 代码要求
使用.ipynb代码，不同cell执行不同功能。
cell1：import需要的包，读取需要的数据；
cell2：检测click信号并筛选负样本，用进度条告知用户进度
cell3：计算负样本的统计特征：来自哪一个Ori_Recording、Duration(μs)和Fpeak(kHz)，起始时间，整理为txt文件，以及所有200ms负样本的.wav音频,波形文件.png(保持原有采样率576kHz、仅滤除5kHz以下的噪声)，都存放在对应的Ori_Recording文件夹下。文件保存位置00_Data\02_ClickDetection\FalseClick。

