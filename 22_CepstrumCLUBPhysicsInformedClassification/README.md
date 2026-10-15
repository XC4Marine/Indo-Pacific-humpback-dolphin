# 第22个实验：正类限定 CLUB 物理解耦二分类（v3）

本目录是独立实验，不修改或覆盖
`D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification`
及其检查点。任务仅定义为中华白海豚 click（1）与困难噪声（0）的二分类。

## 固定环境与运行

Python 环境：`D:\Python_env\toothwhale\python.exe`。

GPU 单批次 smoke：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\experiment.py' `
  --mode smoke `
  --device cuda
```

单折短训练：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\experiment.py' `
  --mode quick `
  --device cuda `
  --epochs 2 `
  --batch-size 16
```

单折单 epoch 集成检查（不计入正式结果）：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\experiment.py' `
  --mode full --device cuda --seeds 20260826 --folds 0 --conditions full `
  --max-epochs 1 --patience 1 --bootstrap-runs 10 --resume `
  --output-dir 'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\outputs_integration'
```

五种子、五折、四条件正式训练：

```powershell
& 'D:\Python_env\toothwhale\python.exe' `
  'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\experiment.py' `
  --mode full --device cuda --seeds 20260826 20260827 20260828 20260829 20260830 `
  --folds 0 1 2 3 4 `
  --conditions full without_recon without_mim bce_only `
  --max-epochs 25 --patience 5 --bootstrap-runs 2000 --batch-size 16 --resume
```

测试：

```powershell
& 'D:\Python_env\toothwhale\python.exe' -m unittest discover `
  -s 'D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\tests' `
  -v
```

CLI 提供 `--mode full`、`--seeds`、`--folds`、`--conditions`、`--max-epochs`、
`--patience`、`--resume` 和 `--bootstrap-runs`。正式训练默认输出到
`outputs_full_no_cepstrum_positive_mim_5seeds_4conditions`；smoke、quick、integration 和 full 目录互不覆盖。

## 数据契约

- 只读第21个固定低SNR清单：
  `D:\Project_Github\Indo-Pacific-humpback-dolphin\21_PhysicsInformedClickClassification\outputs_snr_m15_p15_full_5seeds\dataset_manifest.csv`。
- 沿用其中7000条样本、噪声配对、目标 `[-15,15] dB` SNR及来源隔离五折。
- click与噪声分别去直流、20–190 kHz零相位带通并加Tukey窗。click随后以零填充平移对齐绝对峰至索引2880，再按完整5761点RMS重新定标混噪。
- 正式实验只读复用第22个已验收独立峰对齐缓存，锁定数据哈希为
  `224ec22c07267873dc7a045fd4a339bfc2164f129d144acc702828ac61506615`；不重建缓存，
  不修改第21个缓存或检查点。

## 张量与方法

- 输入 `r:[B,5761]`；共享前端为三层 `16→24→32` 一维卷积、stride `4→4→2`，
  输出 `[B,32,181]`，再分入深度可分离卷积 `E_click/E_channel`。
- 两分支均输出64维潜变量 `Z_c/Z_h` 和独立时序特征。
- `D_click` 仅由 `Z_c` 生成 `x_hat:[B,5761]` 与二分类logit；标签头直接读取 `Z_c`。
- `D_channel` 仅由 `Z_h` 生成 `h_hat:[B,2881]`，参数化严格固定为 `[1,0.25*tanh(tail)]`，没有负时延或可学习直达项。
- 轻量主模型参数584,699，含Q为609,531；正式运行逐条件记录总参数、实际训练参数、
  峰值显存和单批次耗时。
- `r_hat=x_hat*h_hat` 使用FFT线性因果卷积并截取前5761点。
- 总损失为逐样本相对L2重建、logits BCE和CLUB互信息上界的启用项之和；没有倒谱软标签或 Channel loss。
- 分类负类始终参与 BCE，并在启用重建时参与接收信号重建；它们永不作为 CLUB 错配对。
- 每批先仅用真实正类 click 的截断潜变量更新Q，再冻结Q更新主干；正类少于2条时跳过Q更新且MIM为可微零。

## 输出与边界

默认输出：
`D:\Project_Github\Indo-Pacific-humpback-dolphin\22_CepstrumCLUBPhysicsInformedClassification\outputs`。
其中包括配置、独立清单/缓存、数据哈希、数据审计、分项损失、梯度审计、
batch/epoch日志、模型检查点和smoke/quick汇总。

正式模式为每个条件/种子/折建立独立目录，逐 epoch 保存最新训练状态，并仅将验证
AUROC最佳状态写入 `best_checkpoint.pt`。已完成模型由 `COMPLETE.json` 标记，`--resume`
会跳过完成项或从未完成项的最新 epoch 恢复。全部模型完成后生成逐样本预测、主表、
消融表、SNR表、正类MIM解耦诊断表和两级配对Bootstrap；正式模式不生成中文研究报告。
