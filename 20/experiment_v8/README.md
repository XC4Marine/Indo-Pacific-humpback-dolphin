# 第20实验 v8：自动多径时延结构提取

本实验比较实倒谱 + CA-CFAR、MED 与共享自监督稀疏门控。盲提取只读取 `clicktrain_raw.npy`；Bellhop 到达数据仅由评估器读取。

```powershell
python 20/experiment_v8/run_v8.py --phase extract --conditions clean,5,-5 --folds all --device auto --resume
python 20/experiment_v8/run_v8.py --phase evaluate --conditions clean,5,-5
```

完整执行可使用 `--phase all`。训练阶段仅使用 clean 的外层训练组；每个留出组在 clean、+5 dB 和−5 dB 各预测一次。产物在 `20/experiment_v8/outputs`：预测、检查点、统一指标、图表和中文报告均可复跑。
