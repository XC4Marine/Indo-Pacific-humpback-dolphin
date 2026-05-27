# 数据描述：
代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data

Ori_file_num(No.)和采样深度的对应关系用[(No.),'depth']表示如下：
1，13.9
2，11.7
3，5.9
4，6.7
5，6.7
6，7.4
7，10.2
8，18
9，5.1
10，3
11，5.2
12，7.2
13，13.3
14，24.3
15，9.6
16，2.8
17，2.1
18，4
19，14.8
20，7.8
21，19.7
22，19.7
23，19.7
24，19.7
25，19.7
26，19.7
27，7.3
28，7
29，8
30，8
31，8
32，8
33，12.5
34,12.5
35,7.9

数据结构：
1.文件00_Data\ClickTrains.csv,文件内容示例如下：
```
Train_num(No.),Ori_file_num(No.),Train_start(s),Train_end(s),Train_duration(ms),Num_of_pulses,Mean_SNR(dB),Mean_IPI(ms),Mean_SPLpp(dB),Mean_Duration(μs),Mean_Fpeak(kHz),Mean_Bandwidth_-3dB(kHz),Mean_Bandwidth_-10dB(kHz)
1,1,278.2543,278.7155,461.2,4,30.2,152.99,156.66,38.19,58.33,57.21,103.42
2,1,296.8332,297.3997,566.51,9,33,70.4,163.5,46.96,98.07,40.33,114.6
3,1,360.1263,360.5878,461.45,3,32.7,230.62,158.07,33.54,118.17,36.26,83.98
4,1,845.5986,845.7011,102.55,5,29.2,25.59,158.59,61.46,120.83,47.14,120.15
5,1,846.5383,846.5898,51.44,3,27.2,25.62,154.28,87.38,38.25,16.88,106.31
6,1,973.9433,974.2491,305.82,6,49.5,61.12,177.29,41.38,86.72,29.16,89.44
7,1,974.6404,975.4858,845.39,23,35.6,38.39,177.67,51.08,73.46,26.78,90.72
```
2.05_DistinguishbyEnergy\reverberation_metadata.csv，内容示例：
```
folder,filename,ratio,label,original_path
PulseTrain_001,Pulse_1.wav,0.9951199281907943,Weak_Reverberation,D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains\PulseTrain_001\Pulse_1.wav
PulseTrain_001,Pulse_2.wav,0.9949265998111061,Weak_Reverberation,D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains\PulseTrain_001\Pulse_2.wav
PulseTrain_001,Pulse_3.wav,0.9947801544598748,Weak_Reverberation,D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains\PulseTrain_001\Pulse_3.wav
```

3.所有正样本位于00_Data\ClickTrains文件夹
├── ClickTrains/
│   ├── PulseTrain_001
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_4.wav
│   	├──PulseTrain.wav
│   	├──PulseParameters.txt
│   ├── PulseTrain_002
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_9.wav     
│   	├──PulseTrain.wav  
│   	├──PulseParameters.txt
│   ├── ……
│   ├── PulseTrain_832
│   	├──Pulse_1.wav
│   	├──Pulse_2.wav
│   	├──...
│   	├──Pulse_21.wav
│   	├──PulseTrain.wav
│   	├──PulseParameters.txt

txt文件示例：
No.Pulse	SNR(dB)	IPI(ms)	SPLpp(dB)	Duration(μs)	Fpeak(kHz)	Bandwidth_-3dB(kHz)	Bandwidth_-10dB(kHz)
1	20.4	154.48	152.08	101.98	76.65	10.14	90.18
2	45.2	102.01	163.91	114.98	114.41	23.11	52.98
3	53.8	142.04	166.38	114.56	114.41	22.54	51.85

# 任务描述与需求
任务1.根据ClickTrains.csv中的数据，计算出每个Ori_file_num(No.)下的Num_of_pulses总数（一共有35个file），并对应到其采样深度。根据深度来画一张图，深度由浅到深，不同的深度的pulses总数。如果不同file的深度相同，则合并，但在图上也要体现这些file的名字（e.g.对应的柱状图或数据点上方标注 File: 29, 30, 31, 32。）。注意在处理数值数据前进行强制数值转换。

任务2：（用ratio表征混响强度）每一个Train_num(No.)下的各个pulse的高低频能量比ratio和文件路径都保存在reverberation_metadata.csv中（'folder'列最后的数字就是Train_num(No.)，但注意是三位数，有832个）。请画出每个深度下的所有pulses高低频能量比ratio的箱线图

任务3：（用duration表征混响强度）读取ClickTrains文件夹下，每一个PulseTrain_(Train_num)下的txt文件（有832个），读取其中的duration列，表示每一个pulse的时序时间。请画出每个深度下的所有pulses的duration的箱线图。

# 代码要求
使用.ipynb代码，不同cell执行不同任务。

