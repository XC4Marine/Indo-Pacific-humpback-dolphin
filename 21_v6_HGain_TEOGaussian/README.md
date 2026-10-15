# 实验21 v6：h 增益与 x 的 TEO 高斯形状约束

执行入口：

```powershell
python .\experiment_v6.py --mode full
```

固定复用 `D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_source_only_5seeds` 的 7000 条低 SNR 数据、原始录音隔离五折、噪声混合和样本顺序。A/B/C/D 的定义和所有输出路径记录在 `outputs_seed20260826\run_config.json`。

先用 `--mode smoke` 执行单折短训练及训练前结构、梯度、TEO 有限性、因果卷积、数据隔离核验；正式结果只能使用 `--mode full` 的 20 次训练。
