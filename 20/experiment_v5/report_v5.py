#!/usr/bin/env python3
"""Render the evidence-backed Chinese report from completed v5 evaluation artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from protocol_v5 import OUTPUT_ROOT


METRICS = ("source_correlation", "source_si_nrmse", "channel_amplitude_rmse_db",
           "channel_phase_circular_mae_rad", "path_f1", "event_reconstruction_nrmse",
           "event_explained_variance")
SHORT = {"source_correlation": "源相关↑", "source_si_nrmse": "SI-NRMSE↓",
         "channel_amplitude_rmse_db": "信道幅度RMSE↓", "channel_phase_circular_mae_rad": "相位圆MAE↓",
         "path_f1": "路径F1↑", "event_reconstruction_nrmse": "重构NRMSE↓",
         "event_explained_variance": "解释方差↑"}
CONDITIONS = (("clean", "无噪声"), ("snr_m05", "−5 dB"), ("snr_p05", "5 dB"))


def value(number: float) -> str:
    return f"{number:.4f}"


def checkpoint_rows() -> list[str]:
    rows = []
    for path in sorted((OUTPUT_ROOT / "checkpoints").glob("learned_unroll_seed*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        rows.append(f"| {saved['seed']} | {saved['epoch']} | {saved['validation_loss']:.5f} |")
    return rows


def main() -> None:
    evaluation = OUTPUT_ROOT / "evaluation"
    events = pd.read_csv(evaluation / "event_metrics.csv")
    paired = pd.read_csv(evaluation / "paired_deltas.csv")
    per_train = events.groupby(["condition", "method", "train_id"], as_index=False)[list(METRICS)].median()
    learned_seed = per_train[per_train.method.str.startswith("unroll_seed")]
    learned = learned_seed.groupby(["condition", "train_id"], as_index=False)[list(METRICS)].mean()
    learned_median = learned.groupby("condition")[list(METRICS)].median()
    learned_seed_medians = learned_seed.groupby(["condition", "method"])[list(METRICS)].median().groupby("condition")[list(METRICS)].agg(["mean", "std"])
    baseline = per_train[per_train.method.isin(("v4_independent", "fixed_unroll"))].groupby(["condition", "method"])[list(METRICS)].median()

    lines = ["# 第20实验 v5：深度展开式 BCSC 原型实验报告", "",
             "## 1. 问题与范围", "",
             "本实验检验六阶段可训练展开式 BCSC 能否在严格来源隔离下，改善共享源与逐事件信道恢复。"
             "可训练模型只在干净合成 clicktrain 上训练；测试输入为无噪声、−5 dB 和5 dB。"
             "模型推理阶段只读取 `clicktrain_raw.npy`，真值仅在推理完成后的独立评估阶段读取。", "",
             "数据按69个原始来源组固定划分为48个训练组、10个验证组、11个测试组；测试集含16个 clicktrain。"
             "结果按每个 clicktrain 的10个事件取中位数，再在16个 clicktrain 上汇总。", "",
             "## 2. 训练完成状态", "", "| 随机种子 | 验证最优epoch | 最低验证总损失 |", "| ---: | ---: | ---: |",
             *checkpoint_rows(), "",
             "## 3. 主结果", "",
             "下表中的可训练展开为：对每个测试 clicktrain 先在三个种子间平均，再取16个 clicktrain 的中位数。"
             "括号内为三个种子各自全体中位数的标准差；v4 与固定展开为单次确定性运行。", ""]

    for condition, title in CONDITIONS:
        lines += [f"### {title}", "", "| 方法 | 源相关↑ | 信道幅度RMSE (dB)↓ | 路径F1↑ | 重构NRMSE↓ | 解释方差↑ |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for method, label in (("v4_independent", "v4 independent"), ("fixed_unroll", "固定六阶段展开")):
            row = baseline.loc[(condition, method)]
            lines.append("| " + label + " | " + " | ".join(value(float(row[key])) for key in
                         ("source_correlation", "channel_amplitude_rmse_db", "path_f1", "event_reconstruction_nrmse", "event_explained_variance")) + " |")
        row = learned_median.loc[condition]; std = learned_seed_medians.loc[condition]
        keys = ("source_correlation", "channel_amplitude_rmse_db", "path_f1", "event_reconstruction_nrmse", "event_explained_variance")
        cells = [f"{value(float(row[key]))} ± {value(float(std[(key, 'std')]))}" for key in keys]
        lines.append("| 可训练六阶段展开（3种子） | " + " | ".join(cells) + " |")
        lines += ["", "| 方法 | SI-NRMSE↓ | 信道相位圆MAE (rad)↓ |", "| --- | ---: | ---: |"]
        for method, label in (("v4_independent", "v4 independent"), ("fixed_unroll", "固定六阶段展开")):
            row = baseline.loc[(condition, method)]
            lines.append(f"| {label} | {value(float(row['source_si_nrmse']))} | {value(float(row['channel_phase_circular_mae_rad']))} |")
        row = learned_median.loc[condition]; std = learned_seed_medians.loc[condition]
        lines.append(f"| 可训练六阶段展开（3种子） | {value(float(row['source_si_nrmse']))} ± {value(float(std[('source_si_nrmse', 'std')]))} | "
                     f"{value(float(row['channel_phase_circular_mae_rad']))} ± {value(float(std[('channel_phase_circular_mae_rad', 'std')]))} |")
        lines.append("")

    lines += ["## 4. 可训练展开相对 v4 的配对结果", "",
              "差值定义为“每个 clicktrain 的三种子平均可训练展开 − v4 independent”；95%区间为按 clicktrain 重采样的bootstrap中位数区间。", "",
              "| 条件 | 指标 | 中位数差 | 95% bootstrap CI |", "| --- | --- | ---: | --- |"]
    for condition, title in CONDITIONS:
        for metric in ("source_correlation", "channel_amplitude_rmse_db", "path_f1"):
            row = paired[(paired.condition == condition) & (paired.metric == metric)].iloc[0]
            lines.append(f"| {title} | {SHORT[metric]} | {value(float(row.median_delta))} | "
                         f"[{value(float(row.bootstrap_median_ci_low))}, {value(float(row.bootstrap_median_ci_high))}] |")

    lines += ["", "## 5. 结果解释", "",
              "1. 三个条件下，源相关的配对95%区间均低于零，当前可训练六阶段展开未达到 v4 independent 的源恢复水平。",
              "2. 路径F1的配对95%区间也均低于零，说明当前可学习步长和阈值不足以保持 v4 的逐事件路径分辨能力。",
              "3. 信道幅度RMSE的点估计对可训练展开更低，但三种条件的配对95%区间均包含零；因此不能据此主张其具有稳定总体改进。",
              "4. 固定六阶段展开也低于完整迭代 v4，表明仅保留六个更新阶段会形成明显的优化预算瓶颈；当前实验不支持将性能差异单独归因于参数是否可学习。", "",
              "## 6. 结论与下一步", "",
              "该原型完成了严格来源隔离、盲推理和三种子评估链路，但当前六阶段配置没有优于完整 v4 independent BCSC。"
              "下一轮应优先增加展开阶段数或引入与 v4 收敛步骤匹配的初始化，再在相同留出划分上进行配对消融；在此之前，不应以该原型结果主张深度展开带来恢复性能提升。", "",
              "## 7. 可复现产物", "",
              "- `group_split.csv`：锁定来源组划分。", "- `train_config.json`、`training_history.csv`、`checkpoints/`：训练配置、轨迹和验证选中的模型。",
              "- `predictions/`：三种输入条件下的盲推理结果。", "- `event_metrics.csv`、`summary_metrics.csv`、`paired_deltas.csv`：事件级、汇总与配对统计结果。"]
    (evaluation / "prototype_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已生成：{evaluation / 'prototype_report_zh.md'}")


if __name__ == "__main__":
    main()
