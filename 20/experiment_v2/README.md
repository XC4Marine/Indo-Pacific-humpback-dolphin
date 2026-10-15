# 第20实验 v2：Bellhop 浅海宽带信道

本目录独立于 `20` 的既有实验结果。运行顺序：

```powershell
Set-Location D:\Project_Github\Indo-Pacific-humpback-dolphin\20\experiment_v2
.\setup_bellhop.ps1
```

```powershell
python .\run_experiment_v2.py --resume --workers 1
```

运行全量任务前，可执行：

```powershell
python .\run_experiment_v2.py --self-test
```

`setup_bellhop.ps1` 下载官方 `oalib-acoustics/Acoustics-Toolbox`，只编译 Bellhop 与必需的 `misc` 库；它不会修改系统 `PATH`。`run_experiment_v2.py` 原生解析 Bellhop ASCII arrivals，不依赖 MATLAB。

每条成功 click 的最终合成信号为 `python_results/clicks/<click_id>/rx_raw.npy`；同目录的 `rx_preview.wav` 只用于预览，缩放因子记录在总清单中。所有传播量均是相对数字幅值。

源码来源、提交、许可证和工具链记录见 `toolchain_versions.json`。Bellhop 输入顶行选项 `PVWT` 对应 PCHIP SSP、真空（压力释放）海面、dB/波长海底衰减和 Thorp 体积吸收。
