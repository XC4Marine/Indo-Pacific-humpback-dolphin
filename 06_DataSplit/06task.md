代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data

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
7,1,974.6
...
```

2.所有正样本位于00_Data\ClickTrains文件夹
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

PulseTrain_001中的数字，其实就是Train_num(No.)，每个Train序列下的所有Pulse是都是该train的正样本，都要提取，都要标注为1.


3.所有负样本的路径信息在04_Classification\filtered_underwater_negatives.csv中
path
D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\02_ClickDetection\FalseClick\Ori_Recording_01\FalseClick_Ori_Recording_01_0001.wav
D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\02_ClickDetection\FalseClick\Ori_Recording_01\FalseClick_Ori_Recording_01_0002.wav
D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\02_ClickDetection\FalseClick\Ori_Recording_01\FalseClick_Ori_Recording_01_0003.wav
D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\02_ClickDetection\FalseClick\Ori_Recording_01\FalseClick_Ori_Recording_01_0004.wav
D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\02_ClickDetection\FalseClick\Ori_Recording_01\FalseClick_Ori_Recording_01_0005.wav
...
文件名FalseClick_Ori_Recording_(Ori_file_num(No.))_(Falseclick num.).wav

4. 10折交叉验证数据划分，正样本分布：
验证集1：Ori_file_num(No.)=10
验证集2：Ori_file_num(No.)=6
验证集3：Ori_file_num(No.)=29/30/31/32
验证集4：Ori_file_num(No.)=2
验证集5：Ori_file_num(No.)=8
验证集6：Ori_file_num(No.)=14
验证集7：Ori_file_num(No.)=21/22/23/24/25/26
验证集8：Ori_file_num(No.)=17/16/3/4/5
验证集9：Ori_file_num(No.)=28/27/20/35/15
验证集10：Ori_file_num(No.)=7/34/13/1/19
每一折中，除了验证集以外的，都作为训练集

测试集：Ori_file_num(No.)=9/11/18

请在06_DataSplit完成以下任务：
1.根据我给你的验证集、训练集、测试集的划分方式，生成不同csv文件，分别保存各数据集位置；
2.04_Classification\filtered_underwater_negatives.csv保存了不同的Ori_file_num(No.)下，检测到的负样本数据的位置。每个Ori_file下有多个负样本，你需要找出数量小于等于该Ori_file下负正样本（如果不够，就小于；如果够，就等于），加入到任务1中的csv文件中，并带上正负样本标签。训练集、验证集、测试中都要有政府样本路径。
3.告诉我每一折中训练集、验证集正负样本的数量，以及测试集中正负样本的数量。

给我.ipynb代码，每一个cell负责一个任务
