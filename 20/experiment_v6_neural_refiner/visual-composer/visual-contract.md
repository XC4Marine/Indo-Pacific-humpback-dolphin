# v6 留出集结果与典型样本诊断图：视觉契约

Artifact: `v6_rmse_comparison`、`v6_representative_response`、`v6_click_recovery_comparison` 与 `v6_channel_recovery_comparison`  
Target venue / format: 实验原型报告；PNG（展示）与 SVG（可编辑）  
Core claim: 在严格盲推理后，BCSC 引导的神经残差精修在三种输入条件下都降低了逐事件信道幅度响应 RMSE；典型样本图展示这一校正对 click 重构和各事件信道频响的实际影响。  
Reviewer question: 改进是否在每个未见来源组上保持一致？在中位误差样本中，信号、信道及其频域形状如何变化？  
Evidence layer: 主结果 + 定性案例 + 机制检查  
Source data: `outputs/event_metrics.csv`、`outputs/summary_metrics.csv`、`outputs/paired_deltas.csv`、`outputs/blind_predictions/`、v4 盲预测工件与该留出样本的原始 clicktrain；真值仅用于已完成的事后评估和频响案例图。  
Source method / architecture content: v4 independent BCSC 是基线；v6 neural ensemble 是仅校正 35 点信道幅度形状的 BCSC 引导残差网络。源估计保持为 BCSC 输出；诊断性神经时域信道由 35 点幅度校正与 BCSC 相位结合构造，不能表述为网络的原生时域输出。  
Statistics / uncertainty: 每个条件 16 个测试 clicktrain 的中位数；组级配对 bootstrap 95% CI。  
Figure prototype or table type: 配对 slopegraph、带原始点的箱线图、置信区间点图、代表性频响折线图、时域/频域诊断小倍图。  
Panel map: 量化图：(a) clean 条件下每个 clicktrain 的配对误差；(b) 三条件的误差分布；(c) 神经集成相对 BCSC 的配对中位差及 95% CI。诊断图一：(a) 代表事件的观测 click 与两种重构时域；(b) 对应频谱；(c) 共享源时域；(d) 共享源频谱。诊断图二：10 个事件分别显示 BCSC 与神经诊断信道的时域，以及二者和真值的 35 点相对幅频。  
Exact label inventory: BCSC (v4 independent), Neural residual refiner (ensemble), Neural diagnostic reconstruction, Observed click window, Original source click, Clean, −5 dB AWGN, 5 dB AWGN, Channel amplitude RMSE (dB), Paired change (neural − BCSC, dB), Relative amplitude (dB), Frequency (kHz), Truth, Event channel, Time (ms).  
Icon inventory: 原生 Matplotlib 线、点、箱线、误差线；无图标。  
Reference-layout sources and extracted composition principles: 无外部版式复刻；使用直接配对线、固定方法颜色和零基线提高可读性。  
Caption role: 量化图给出完整的未见来源组分布与配对不确定性；案例图仅用于可检查的频响行为，不替代主统计结论。  
Manuscript placement: 实验结果节。  
Output formats: PNG, SVG.  
PPTX editability target: SVG-convertible.  
Traceability: `plot_results_v6.py`；上游 CSV 与盲预测工件路径见脚本常量。
