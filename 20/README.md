# 单 click 与多 click 联合 BCSC 比较

本目录从 `09_Locate` 中固定随机抽取 50 个至少包含 5 个 click 的 PulseTrain，比较：

- 单 click BCSC：使用该 PulseTrain 中信噪比代理最高的锚点 click 独立学习原子；
- 多 click 联合 BCSC：最多使用同一 PulseTrain 中 10 个 click，共享学习一个原子，各 click 保留独立稀疏多径序列。

两种方法均在同一个锚点 click 上读取多径并执行 50 次 Bootstrap，以保证可比性。原子时长被限制在 20–150 μs，实测 −3 dB 带宽通过硬投影限制在 6–72 kHz。

## 主要文件

- `report.html`：中文综合报告；
- `batch_joint_bcsc_compare.py`：完整批处理、断点续跑、绘图和报告代码；
- `selected_50_pulsetrains.csv`：随机抽样清单；
- `results/pulsetrain_summary.csv`：50 组 PulseTrain 汇总；
- `results/path_confidence_comparison.csv`：所有单/多 click 路径结果；
- `results/paired_path_confidence_comparison.csv`：按 ±0.03 ms 延迟匹配后的同一路径置信度前后比较；
- `results/all_results.json`：完整结果；
- `figures/`：总体图、成对置信度图及 50 组原子时频域对比图；
- `checkpoints/`：逐 PulseTrain 断点，可用于恢复和重新生成报告。

## 复现命令

```powershell
D:\Miniconda\python.exe D:\Project_Github\Indo-Pacific-humpback-dolphin\20\batch_joint_bcsc_compare.py --count 50 --bootstrap 50 --workers 4
```

若断点存在，命令会直接读取断点并重新生成统计、图片和报告。加入 `--force` 才会重新运行全部 BCSC 和 Bootstrap。

## 置信度含义

报告中的置信度是残差块重采样、正则强度扰动和原子时长扰动条件下的路径重现率，不是候选为真实物理多径的后验概率。联合方法新增的候选仍需要通过合成数据、人工标注或跨水听器一致性验证误检率。
