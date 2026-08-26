#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成BCSC–Transformer研究Notebook。"""

from __future__ import annotations

import json
from pathlib import Path


MODULE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813")
NOTEBOOK_PATH = MODULE_DIR / "bcsc_transformer_binary_classification.ipynb"


def markdown(source: str) -> dict:
    """构造Markdown单元格。"""
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    """构造代码单元格。"""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


cells = [
    markdown(
        """# BCSC–Transformer 二分类与捷径学习研究

本 Notebook 比较三种来源隔离的二分类方案：

1. 统一预处理波形直接输入 Transformer；
2. BCSC 直达径、多径及关系显式特征输入 Transformer；
3. 仅 BCSC 直达径显式特征输入相同的特征 Transformer。

研究问题是：波形模型是否利用背景或多径形成捷径，以及高置信度多径的显式表征能否辅助仅直达径识别。

> BCSC 的7/8复现率表示算法稳定性，不是路径物理真实性的后验概率。负样本中的结果仅称主/次到达候选。
"""
    ),
    markdown(
        """## 1. 固定环境与配置

所有计算使用项目规定的 `D:\\Python_env\\toothwhale\\python.exe`。随机种子、数据来源、BCSC约束、五折划分和模型训练预算均在流水线配置中固定。
"""
    ),
    code(
        """from pathlib import Path
import json
import sys

MODULE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813")
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from bcsc_transformer_pipeline import CFG, run_pipeline

print(f"Python: {sys.executable}")
print(f"输出目录: {CFG.output_dir}")
print(f"配置哈希: {CFG.config_hash}")
print(f"BCSC原子长度: {CFG.atom_length}点 ({CFG.atom_duration_us:.1f} μs)")
"""
    ),
    markdown(
        """## 2. 运行完整流水线

该单元格按四个总体阶段显示进度和ETA：数据读取与统一预处理、7000条BCSC及稳定性复核、路径特征提取、三模型五折训练。缓存按配置哈希保存，意外中断后可继续。
"""
    ),
    code(
        """results = run_pipeline(CFG, force=False, quick=False)
results
"""
    ),
    markdown(
        """## 3. 主要结果表

下列单元格读取五折汇总，展示三组主要实验结果。阈值只由各折验证集确定，置信区间按原始录音组Bootstrap计算。
"""
    ),
    code(
        """import pandas as pd
from IPython.display import Markdown, display

summary = pd.read_csv(CFG.output_dir / "model_summary.csv")
display(summary)
display(Markdown((CFG.output_dir / "research_report.md").read_text(encoding="utf-8")))
"""
    ),
    markdown(
        """## 4. 输出索引

- `dataset_manifest.csv`：7000条样本及来源隔离折分；
- `bcsc_paths.csv`：直达径、最多1条高置信度多径和失败状态；
- `explicit_features.csv`：路径及关系特征；
- `fold_metrics.csv` / `test_predictions.csv`：三模型五折指标与样本级预测；
- `counterfactual_metrics.csv`：波形捷径反事实；
- `feature_ablation_metrics.csv`：多径数值特征和检出子集消融；
- `research_report.md`：独立中文研究报告。
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "toothwhale", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK_PATH)
