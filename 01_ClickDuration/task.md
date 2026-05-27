# 数据描述：
代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data
数据结构：
1.00_Data\ClickTrains文件夹下的数据
├── ClickTrains/

│   ├── PulseTrain_001/          # 第一个Pulse Train

│   │   └── Pulse_1.wav
│   │   └── Pulse_2.wav
│   │   └── ...
│   │   └── PulseTrain.wav
│   │   └── PulseParameters.txt

│   ├── PulseTrain_002/        # 第二个PulseTrain

│   │   └── ...

│   ├── PulseTrain_003/         # 第三个PulseTrain

│   │   └── ...

│   └── ...      

2.PulseParameters.txt文件内容示例：
No.Pulse	SNR(dB)	IPI(ms)	SPLpp(dB)	Duration(μs)	Fpeak(kHz)	Bandwidth_-3dB(kHz)	Bandwidth_-10dB(kHz)
1	22.7	103.58	157.33	20.83	59.18	68.20	100.32
2	37.7	188.66	160.12	19.09	76.65	90.74	130.76
3	33.8	166.73	157.56	26.48	57.49	60.87	125.12

3.00_Data\ClickTrains.csv
表头与前三行数据示例：
Train_num(No.),Ori_file_num(No.),Train_start(s),Train_end(s),Train_duration(ms),Num_of_pulses,Mean_SNR(dB),Mean_IPI(ms),Mean_SPLpp(dB),Mean_Duration(μs),Mean_Fpeak(kHz),Mean_Bandwidth_-3dB(kHz),Mean_Bandwidth_-10dB(kHz)
1,1,278.2543,278.7155,461.2,4,30.2,152.99,156.66,38.19,58.33,57.21,103.42
2,1,296.8332,297.3997,566.51,9,33,70.4,163.5,46.96,98.07,40.33,114.6
3,1,360.1263,360.5878,461.45,3,32.7,230.62,158.07,33.54,118.17,36.26,83.98

4.Ori_file_num(No.)与深度对应关系
Original audio_file (No.)	Water Depth (m)
1	13.9
2	11.7
3	5.9
4	6.7
5	6.7
6	7.4
7	10.2
8	18
9	5.1
10	3
11	5.2
12	7.2
13	13.3
14	24.3
15	9.6
16	2.8
17	2.1
18	4
19	14.8
20	7.8
21	19.7
22	19.7
23	19.7
24	19.7
25	19.7
26	19.7
27	7.3
28	7
29	8
30	8
31	8
32	8
33	12.5
34	12.5
35	7.9

# 任务描述与需求
用户需要做各深度下海豚click信号的持续时间的统计分析。Ori_file_num(No.)表示不同航次测量的结果，每一个Ori_file_num(No.)中已经检测出了ClickTrain，并且排序为了Train_num(No.)，保存在ClickTrains文件夹下，文件名为PulseTrain_'Train_num(No.)'，'Train_num(No.)'是三位数,需要从ClickTrains.csv文件中查找不同Ori_file_num(No.)对应哪些Train_num(No.)。每一个PulseTrain文件夹下有一个PulseParameters.txt文件用来存放已经测量得出的click信号参数。

请读取每一个PulseParameters.txt中的Duration(μs)和Fpeak(kHz)，作为因变量，深度Water Depth作为自变量（注意，部分不同的Ori_file_num(No.)是在同一深度采集的，将这些都数据作为同一深度的数据），输出一份CLick信号持续持续时间、峰值频率和水深的统计报告，需要做显著性分析，来计算差异性。画出随深度变化的折线箱线图


# 代码要求
使用.ipynb代码，不同cell执行不同功能。
cell1：import需要的包，读取需要的数据；
cell2：计算相关数据
cell3：在输出中展示报告，不需要保存
cell4：保存方式，但将这一部分代码注释掉，用户可以选择保存或者不保存


