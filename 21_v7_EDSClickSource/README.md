# 实验 21 v7：双分量 EDS Click 源分支

运行前先执行：

`python experiment_v7.py --mode smoke`

烟雾测试通过后执行：

`python experiment_v7.py --mode full`

v7 固定复用 v4/v6 的 7000 条混合噪声数据、原始录音隔离五折与种子 `20260826`。A/B 使用自由波形源；C/D 使用双分量 EDS 源。B/D 使用单位 L2 能量源及 `h=g*h_shape`，其中 `h_shape[0]=1`、`h[0]=g`。总损失仅为相对重建 MSE 加 BCE。

逐折 EDS 边界仅由该折训练集加噪前正样本 Click 的累计能量 1%–99% 脉冲窗得到。输出位于 `outputs_seed20260826`。
