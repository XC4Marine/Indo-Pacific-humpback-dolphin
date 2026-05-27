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
│   	├──PulseTrain.wav
│   ├── PulseTrain_002
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_9.wav     
│   	├──PulseTrain.wav  
│   ├── ……
│   ├── PulseTrain_832
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_21.wav
│   	├──PulseTrain.wav




# 任务描述与需求
1.禁止读取所有PulseTrain.wav文件。取出PulseTrain.wav文件以外的所有正样本音频文件，每个文件都是200us。
2.算出每个音频文件的Teager算子。
2.以能量比例指标来定义混响程度：以Teager算子的峰值为中点，前25us和后25us的信号组成核心脉冲。计算核心脉冲能量与信号总能量的比值。
3.输出能量比值的频率分布直方图。
4.输出能量比值小于0.7的音频的文件位置和命名。
5.在05_DistinguishbyEnergy文件夹下新建一个数据集，能量比值大于0.7的音频位弱混响，能量比值小于0.7的为强混响，音频保存在各自文件夹。



# 代码要求
使用.ipynb代码，不同cell执行不同功能。


