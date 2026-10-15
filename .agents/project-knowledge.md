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
10_AutoCorrMultipath/            # 多径检测 (ACF + Teager + Pearson三层过滤, 10%阈值, MAX=3)
├── autocorr_detect.py           # 100样本实验脚本
├── batch_detect.py              # 全量批处理脚本，读取 calibrated_anchors.txt 作为直达锚点
├── calibrated_anchors.txt       # RMS质心法前置校准锚点，18116条 Click 记录
└── all_pulse_times.txt          # 18116 Click 直达/多径时间表
11_ChannelMetrics/               # 信道参数计算与异常校准
├── calc_pulse_params.py         # 基于校准直达时间计算 Delay_us, AmpRatio, Coherence
├── recalibrate_fix_v2.py        # RMS质心法全量前置锚点校准脚本
├── pulse_params.txt             # 33489行信道参数 (18116直达 + 15373多径)
└── anomalous_ampratio_final.txt # AmpRatio<0.9 残留异常 (空)
13_Negative/                     # 长录音 Hard Negatives 自动挖掘
├── masking.py                   # 读取 AllTrain.csv 并构建按原始录音编号分组的保护区间树
├── filtering.py                 # 带通滤波、TEO、频谱质心、ACF次峰与Pearson相干性校验
└── mine_negatives.py            # 多进程文件级调度, 导出 negative_samples_log.csv 与 Negative_Samples/*.wav
15/                              # 单个 click 分段聚类 CSV 可视化
└── plot_csv_parameters_over_time.py # 绘制 RMS/互相关/谱质心/带宽/DBSCAN 标签随时间变化
```

## 数据特点

| 特征 | 描述 |
|------|------|
| 采样格式 | .wav 单通道音频, 576kHz |
| 物种 | 中华白海豚 (Sousa chinensis) |
| 信号类型 | 高频 click 脉冲串 |
| 每条脉冲串 | 含 2~100+ 个脉冲，由 PulseParameters.txt 记录各脉冲参数 |
| 关键参数 | SNR、IPI（脉冲间隔）、SPLpp、Duration、Fpeak、带宽 |
| 混响分类 | 三级：Clean（无混响）/ Moderate（中度）/ Severe（严重） |
| 野外数据 | Results.csv 包含日期/季节/经纬度/水深/海豚群体大小/行为 |
| 负样本挖掘保护索引 | 13_Negative 默认使用 AllTrain.csv，覆盖 click、buzz、burst pulse 所在时段，避免将其误当噪声 |
| 当前负样本预处理 | 20-190 kHz Butterworth 带通；不再使用 HFER 高低频能量比 |
| 当前直达锚点 | 由 `10_AutoCorrMultipath/calibrated_anchors.txt` 提供，基于 ±50μs RMS 质心法全量前置校准 |

## 项目目标
1. 从原始水听器录音中检测出海豚的 click 脉冲串
2. 对每个 click 进行参数提取（持续时间、峰值频率、带宽等）
3. 区分真实 click 与海洋环境噪声（假阳性检测）
4. 对 click 进行混响严重程度分类
5. 最终建立自动化的海豚声学监测与分类系统

## 多径检测流水线 (10_AutoCorrMultipath + 11_ChannelMetrics)

### 当前顺序
1. `09_Locate` 输出每个 Click 的 ±5ms 片段
2. `11_ChannelMetrics/recalibrate_fix_v2.py` 对 18116 个 Click 全量执行 ±50μs RMS 质心法，生成 `10_AutoCorrMultipath/calibrated_anchors.txt`
3. `10_AutoCorrMultipath/batch_detect.py` 读取校准锚点，执行 ACF 次峰初筛与 Pearson 相干性过滤，生成 `all_pulse_times.txt`
4. `11_ChannelMetrics/calc_pulse_params.py` 读取校准直达时间，重算 Delay_us、AmpRatio、Coherence，并输出残留 AmpRatio 异常明细

### 检测与过滤步骤
1. **前置锚点校准**: ±50μs RMS² 滑动窗取 argmax，作为每个 Click 的直达锚点
2. **ACF 次峰初筛**: 归一化自相关, 10% 阈值, |lag|∈[0.1ms, 5ms], 最多 3 个
3. **Teager 交叉验证** (仅实验脚本): Teager 能量 ≥ 全局最大值 0.1%
4. **Pearson 相干性**: 直达窗与多径候窗的 Pearson r, |r| ≥ 0.02

### 锚点校准
- RMS 质心法已作为多径检测前置步骤，而不是只针对异常 AmpRatio 的后处理
- 当前 `calibrated_anchors.txt` 覆盖 18116 个 Click，缺失锚点 0

### 信道三个参数
- Delay_us: 多径相对直达延迟
- AmpRatio: RMS(直达) / RMS(多径)
- Coherence: Pearson r (直达窗, 多径窗)

## 编码环境
- Python 3.11.15 @ D:\Python_env\toothwhale\python.exe
- 依赖：numpy, scipy, matplotlib, soundfile, pandas, tqdm, pathlib, intervaltree

## 当前进度（2026-07-18）
- [x] 01~09 模块：完成
- [x] 10_AutoCorrMultipath: 前置 RMS 质心锚点校准后执行多径检测 (10%阈值, MAX=3), 10510/18116 Click 检出多径, 15373 多径, 缺失锚点 0
- [x] 11_ChannelMetrics: 基于校准直达时间输出 18116 直达 + 15373 多径信道参数, AmpRatio<0.9 残留异常径 0
- [x] 13_Negative: 长录音 Hard Negatives 自动挖掘完成，默认以 AllTrain.csv 构建排他保护带；当前推荐结果为 `13_Negative_Bandpass_20k_190k_NoHFER`，输出 90,580 个负样本，CSV 不含 HFER
- [x] 15: 新增 click 分段聚类 CSV 参数时序绘图脚本，默认读取 `12_ClickSegmentClustering/outputs_100us/Pulse_027_segments_dbscan.csv`

## 定向滤波多径提取（2026-08-06）

- 输入：`D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\Positive_Clicks_3500`，共3500条、576 kHz单通道WAV。
- 可运行Notebook：`D:\Project_Github\Indo-Pacific-humpback-dolphin\17_20260805\multiExtract_tuned_3500.ipynb`；保留原Notebook不覆盖。
- 结果目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin\17_20260805\multipath_results_3500`。
- 检测/评分/识别信号统一使用 `x_detect`：20 kHz Butterworth高通（原型阶数2）串联185–215 kHz Butterworth带阻（原型阶数2），均为SOS与 `sosfiltfilt` 零相位；`x_raw` 仅用于滤波审计。
- 最终检测参数：片段50 μs、中心搜索半宽0.5 ms、主到达保护区100 μs、最大时延1 ms、候选最小间距50 μs、MAD峰高/显著性系数5/3、最低SNR 6 dB、最低NCC 0.55、最低频谱余弦0.70、最低评分0.60、权重0.20/0.50/0.30、最多两条多径。
- 全量结果：3500/3500成功；0/1/2条多径分别为2426/935/139个文件，共导出4713个路径WAV；每条29采样点、576 kHz、单通道PCM16，保持文件内相对幅度，不做逐路径归一化。
- 合成验收：延迟定位误差0采样点，±50 μs外振铃比例0.004419；全量结构和路径WAV一致性验收通过。
- 滤波审计：383条强200 kHz背景样本的185–215 kHz能量衰减中位数约22.78 dB；背景RMS中位下降约4.70 dB；30–180 kHz主体频谱余弦第5百分位0.999145。
- 固定200条审查集生成400张诊断图。当前无逐路径人工真值或双人盲审标签，98%/90%仅作为操作性视觉验收目标，不应作为独立统计准确率引用。

## 召回型多径二次实验（2026-08-06）

- 为回应诊断图中明显延迟到达漏选的问题，保留原1 ms精确率优先结果，新增Notebook：`D:\Project_Github\Indo-Pacific-humpback-dolphin\17_20260805\multiExtract_tuned_3500_recall.ipynb`。
- 新结果目录：`D:\Project_Github\Indo-Pacific-humpback-dolphin\17_20260805\multipath_results_3500_recall`。
- 召回型参数：最大时延2 ms、MAD峰高/显著性4/2、最低SNR 6 dB、NCC 0.50、频谱余弦0.60、评分0.55；片段50 μs、中心窗0.5 ms、保护区100 μs及权重0.20/0.50/0.30保持不变。
- 全量消融：严格门限+1 ms为1213条多径；严格门限+2 ms为2278条；放宽门限+1 ms为2015条；放宽门限+2 ms为3625条。
- 最终召回型结果：3500/3500成功；0/1/2条多径文件数1029/1317/1154；共导出7125个路径WAV，其中3500条主到达、3625条多径。
- 1–2 ms区间选中339条路径；其中335条取得下一次IPI，均早于下一次click，已知最小下一次IPI为8.8 ms，因此当前3500条数据中2 ms搜索窗未与相邻click重叠。
- 召回型结果相对基线有1864个文件增加路径、6个文件减少路径。固定200条审查集的旧ACF一致性代理下降，表明召回提高同时增加伪多径风险；旧ACF不是人工真值。
- 新增 `full_ablation.csv`、`baseline_comparison.csv` 和 `delay_ipi_audit.csv`，用于区分时延范围、门限放宽及相邻click风险。

## BCSC–Transformer 二分类与捷径学习（2026-08-14）

- 可复现实验入口：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\bcsc_transformer_binary_classification.ipynb`；核心流水线：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\bcsc_transformer_pipeline.py`；配置哈希：`33ce6ffa3ab921bc`。
- 数据固定为3500条 `Positive_Clicks_3500` 正样本与3500条来源匹配负样本。负样本从原始长录音重新截取，正负统一预处理，从而避免已有负样本带通和边界截短形成制作流程捷径。
- 数据折分按 `Ori_file_num` 来源组隔离为5折；当前验收来源泄漏为0，测试预测覆盖7000条样本×3个模型，共21000行。
- BCSC完整保留7000条状态，6995条为 `ok`（99.93%），5条负样本为 `unstable_direct`；高置信度次到达候选5237条。7/8扰动复现只代表算法稳定性，不代表真实传播路径的后验概率；负样本结果必须称为主/次到达候选。
- 五折主结果：波形AUROC 0.9803，直达+多径显式特征0.9403，仅直达显式特征0.9301。波形模型仅背景残差AUROC 0.9265，背景交换预测翻转率18.19%，支持存在背景捷径信息。
- 显式多径相对仅直达的配对AUROC差值为+0.0067，录音组Bootstrap 95% CI [-0.0039, 0.0164]；点估计为正但统计证据不明确。保留掩码并清零多径数值后AUROC降至0.9163，说明多径数值属性具有辅助信息，但不等同于证明物理多径真实性。
- 研究报告与全部清单、特征、模型、预测、指标及图表位于 `D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs`；报告主文件为 `research_report.md`。

## 信号—噪声可分性与背景捷径确认（2026-08-18）

- 独立审计入口：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\signal_noise_shortcut_audit.ipynb`；流水线：`D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\shortcut_audit_pipeline.py`；配置哈希：`f047e9f9acace257`。
- 正负样本均从 `00_Data\original_audio` 重截5761点、576 kHz并统一预处理；正样本经 `pulse_precise_positions.txt` 与 `ClickTrains.csv` 映射为原始录音时间，3500/3500覆盖。
- 严格匹配队列从90580条负候选中按录音编号一对一无放回匹配，倾向分逻辑回归+0.2 SD caliper；实际3370对，局部SNR SMD未达标，报告标记严格匹配不可行。
- 七条件使用相同WaveformTransformer/种子/预算/来源隔离五折；简单基线为逻辑回归与浅层梯度提升，仅训练于严格匹配队列。
- 结果：信号区AUROC显著>0.5（形态本身可分）；宽窗口背景AUROC CI下限未>0.5（无背景类别信息）；BCSC残差未显著优于背景（未证实残差泄漏）；严格匹配后全波形显著下降（原任务受SNR/幅度/频谱不平衡影响）。
- 遮罩合成与BCSC合成自检均通过；输出含匹配清单、波形缓存、特征、折分、模型、预测、指标、Bootstrap与PNG图。

## 局部能量块稀疏 Physics 单种子五折实验（2026-08-27）

- 实现入口：`D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\pipeline.py`；新增57点、stride 1、无padding的尾部局部RMS能量损失，并保留旧逐点Physics与历史三条件默认行为。
- 数据、五折和[-15,15] dB混噪与既有低SNR实验一致；数据哈希均为 `70204b5ac4484bb4a08a684e29083390cec974502061c45ba48ea2ef9ac130ed`。四条件配置哈希为 `4aea26d47206f919`。
- 种子20260826共完成20个模型；产物验收通过：20权重、20组epoch日志、20组batch日志、20行折指标、4行种子指标、28000条唯一逐样本预测。
- AUROC：Waveform Transformer 0.84701，Recon+BCE 0.87051，旧逐点Physics 0.87861，Local-Energy Physics 0.87614。
- Local-Energy−旧Physics AUROC差为-0.00247，录音组Bootstrap 95% CI [-0.00883, 0.00478]，不能声明分类改善；Local-Energy−Recon+BCE为+0.00563，CI [-0.00487, 0.01651]；Local-Energy−Transformer为+0.02913，CI [0.01161, 0.04994]。
- Local-Energy−旧Physics的局部能量Hoyer差为+0.03034，95% CI [-0.09475, 0.14719]；活动块比例差为+0.03925，CI [-0.09437, 0.17828]，均不能声明形成更明显块稀疏。
- Local-Energy原始Physics均值为0.0008593，加权贡献均值为8.593e-7，仅占总损失约8.63e-7；当前β=0.001下约束量级偏小，是后续权重敏感性实验的重点。
- 输出与中文报告位于 `D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_local_energy_seed_20260826`；运行环境为 `D:\Python_env\toothwhale\python.exe`。所有置信区间仅反映录音来源不确定性，不能证明跨初始化稳定性。

## 第22个倒谱–CLUB物理解耦二分类初版（2026-09-01）

- 独立入口为 `D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\experiment.py`，固定环境为 `D:\Python_env\toothwhale\python.exe`。
- 任务边界仅为中华白海豚click（1）与困难噪声（0）；倒谱来自接收信号，只是弱信道结构提示，不是真实信道标签。
- 数据只读第21个固定低SNR清单并复制其7000条样本、噪声配对、目标SNR与来源隔离折；第22个缓存和产物完全独立。数据哈希为 `224ec22c07267873dc7a045fd4a339bfc2164f129d144acc702828ac61506615`。
- 统一滤波后的clean click以零填充平移对齐绝对峰至2880点：原本1940/3500条已对齐，实际平移1560条，最终3500/3500条精确对齐；最大SNR定标误差3.553e-15 dB，来源泄漏0。
- 网络使用共享可学习前端及click/channel独立编码器，输出64维 `Z_c/Z_h`；channel响应为0–5 ms的2881点严格因果 `[1,0.25*tanh(tail)]`，不存在负时延或可学习直达参数。
- 功率倒谱统一零填充至8192点，索引0–46清零、47–2880按逐样本RMS归一化；Channel loss仅对正样本启用。
- CLUB采用先Q后主干的双优化器：Q阶段潜变量截断梯度，主干阶段冻结Q；MIM使用无固定点随机负配对，batch小于2时为可微零。
- 7项unittest及RTX 4060真实16样本GPU smoke全部通过。当前只证明代码、数据契约、张量和梯度路径正确；尚未执行quick或完整五折多种子实验，不支持性能、解耦有效性或论文结论。

## 第22个轻量化完整消融结果（2026-09-01）

- 完整运行入口为`22_CepstrumCLUBPhysicsInformedClassification/experiment.py --mode full`；输出目录为`outputs_full_5seeds_6conditions`，配置哈希`5620bbfb55f100cb`。
- 轻量主模型584,699参数，含Q为609,531；12项测试及单折单epoch集成恢复检查通过。
- 五种子×五折×六条件共150模型全部完成，产物验收通过，逐样本预测共210000条且键唯一、数值有限。
- AUROC：Full 0.89983、w/o Recon 0.89490、w/o Channel 0.89853、w/o MIM 0.90512、BCE-only 0.89781、Direct classifier 0.89781。
- Full减各消融的AUROC两级配对Bootstrap 95% CI全部跨0；当前不能宣称Recon、Channel或MIM带来稳定分类提升，也不能仅凭这些分类结果宣称物理解耦成立。
- w/o MIM点估计高于Full约0.00530，但差值CI跨0；应表述为当前证据不足，而非MIM确定有害。
- BCE-only与Direct classifier因有效训练路径完全相同而得到相同结果；后续若需检验“编解码结构本身”，必须定义会实际改变分类前向或梯度路径的非冗余直接分类基线。
