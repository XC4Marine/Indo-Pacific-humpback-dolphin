# 数据描述：
代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data
数据结构：
1.所有正样本位于00_Data\ClickTrains文件夹
├── ClickTrains/
│   ├── PulseTrain_001
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_4.wav
│   ├── PulseTrain_002
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_9.wav       
│   ├── ……
│   ├── PulseTrain_832
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_21.wav




# 任务描述与需求
1.已知正样本有897个，请根据文件命名，取出所有正样本的音频文件，每个文件都是200us。
2.以50us为边界，区分click信号混响的严重性：小于50us的信号的属于无混响，大于50us，小于100us的属于中度混响，大于100us的属于混响严重。
3.如何界定信号边界：
（1）先算出信号 Teager ，对 Teager 能量进行 **10 点滑动平均**；
（2）选取 Teager 能量值的**第 40 百分位数**（基于对数据的经验分析）作为该段数据的噪声基准。
（3）对 Teager 能量进行 **10 点滑动平均**；系统从100us为中心，向前后搜索，直到能量值降至噪声基准的 **3 倍**以下。这两个点即被定义为该次点击信号的精确**开始**和**结束**位置。
（4）输出三种类型的文件夹，和每一段音频的png保存在各自的文件夹中。输出每种类型的音频数量。


# 代码要求
使用.ipynb代码，不同cell执行不同功能。


