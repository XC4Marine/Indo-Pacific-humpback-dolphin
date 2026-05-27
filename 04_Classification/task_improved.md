# 数据描述：
代码仓库位置：D:\Project_Github\Indo-Pacific-humpback-dolphin
数据放置于D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data
数据结构：
1.所有正样本位于03_Distinguish\Output_Classification\Clean和03_Distinguish\Output_Classification\Moderate文件夹
├── Clean/
│   ├── PulseTrain_011_Pulse_9.wav
│   ├── PulseTrain_046_Pulse_36.wav
│   ├── ......
├── Moderate/
│   ├── PulseTrain_001_Pulse_2.wav
│   ├── PulseTrain_005_Pulse_3.wav
│   ├── ......

2.所有负样本位于00_Data\02_ClickDetection\FalseClick文件夹
├── FalseClick/
│   ├── Ori_Recording_01
│   	├──FalseClick_Ori_Recording_01_0001.wav
│   	├──FalseClick_Ori_Recording_01_0002.wav
│   	├──...
│   	├──FalseClick_Ori_Recording_01_0175.wav
│   ├── Ori_Recording_02
│   	├──FalseClick_Ori_Recording_01_0001.wav
│   	├──FalseClick_Ori_Recording_01_0001.wav
│   	├──...
│   	├──FalseClick_Ori_Recording_01_1549.wav
│   ├── ……
│   ├── Ori_Recording_35
│   	├──FalseClick_Ori_Recording_01_0001.wav
│   	├──FalseClick_Ori_Recording_01_0001.wav
│   	├──...
│   	├──FalseClick_Ori_Recording_01_0045.wav

3.难度测试集正样本位于03_Distinguish\Output_Classification\Severe
├── Clean/
│   ├── PulseTrain_011_Pulse_4.wav
│   ├── PulseTrain_013_Pulse_2.wav
│   ├── ......

# 任务描述与需求
1.已知正样本有884个，请根据文件命名，取出所有正样本的音频文件。
2.负样本有非常多，请固定随机种子，从负样本中找到884个音频文件，以及另外100个样本作为难度测试集中的负样本。
3.对所有样本都进行一定程度上的平移，提升模型的泛化能力。
3.直接使用波形文件作为分类模型输入。
4.画出t-sne图。
5.将正负样本按8：2的比例分为训练集和测试集。
6.使用XGboost算法，在训练集中进行5折交叉验证，使用Accuracy作指标，输出统计性Acc结果。
7.在测试集上运行XGBoost分类算法，输出分类报告。
8.从难度测试集中取100个正样本，与步骤2中的100个负样本一起，使用XGboost进行测试，输出分类报告。


# 代码要求
使用.ipynb代码，不同cell执行不同功能。


