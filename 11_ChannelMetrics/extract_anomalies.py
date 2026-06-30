#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多径 AmpRatio 异常样本提取脚本

从 pulse_params.txt 中筛选 AmpRatio < 1.0 的多径记录（多径反射能量大于直达声能量），
按 AmpRatio 升序排列，输出 Tab 分隔的异常样本清单至 anomalous_ampratio.txt。

Author: Codex
Date: 2026-06-27
"""

from pathlib import Path
import pandas as pd

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
METRICS_DIR = BASE_DIR / "11_ChannelMetrics"
INPUT_FILE = METRICS_DIR / "pulse_params.txt"
OUTPUT_FILE = METRICS_DIR / "anomalous_ampratio.txt"

# ── 读取数据 ──────────────────────────────────────────────────
print("读取 pulse_params.txt ...")
df = pd.read_csv(INPUT_FILE, sep="\t")

# 过滤多径 + AmpRatio < 1.0
mp = df[df["PathType"] != "Direct"].copy()
anomalies = mp[mp["AmpRatio"] < 1.0].copy()

# 提取 MP 阶数
anomalies["MP_Order"] = anomalies["PathType"].str.extract(r"Multipath_(\d+)").astype(int)

# 按 AmpRatio 升序排列
anomalies = anomalies.sort_values("AmpRatio", ascending=True)

print(f"多径总数: {len(mp)}")
print(f"AmpRatio < 1 异常样本数: {len(anomalies)}")
print(f"AmpRatio 范围: {anomalies['AmpRatio'].min():.6f} ~ {anomalies['AmpRatio'].max():.6f}")

# ── 输出 ──────────────────────────────────────────────────────
# 选择输出列，重命名使表头更清晰
output_cols = {
    "PulseTrain":   "PulseTrain",
    "ClickName":    "ClickName",
    "MP_Order":     "MP_Order",
    "Delay_us":  "Delay_us",
    "AmpRatio":     "AmpRatio",
    "Coherence":    "Coherence",
}
out = anomalies[list(output_cols.keys())].rename(columns=output_cols)

# 写入 Tab 分隔文件
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"# AmpRatio < 1.0 异常多径样本 (共 {len(anomalies)} 条)\n")
    f.write(f"# 多径反射能量大于直达声能量 | AmpRatio 范围: {anomalies['AmpRatio'].min():.6f} ~ {anomalies['AmpRatio'].max():.6f}\n")
    f.write("#")
    for col in out.columns:
        f.write(f"\t{col}")
    f.write("\n")
    for _, row in out.iterrows():
        f.write(f"{row['PulseTrain']}")
        f.write(f"\t{row['ClickName']}")
        f.write(f"\tMP{int(row['MP_Order'])}")
        f.write(f"\t{row['Delay_us']:.2f}")
        f.write(f"\t{row['AmpRatio']:.6f}")
        f.write(f"\t{row['Coherence']:.6f}")
        f.write("\n")

print(f"异常样本清单已保存: {OUTPUT_FILE}")

# ── 分段统计 ──────────────────────────────────────────────────
print("\n── 按 MP 阶数统计 ──")
for order in [1, 2, 3]:
    subset = anomalies[anomalies["MP_Order"] == order]
    print(f"  MP{order}: {len(subset)} 条")

print("\n── AmpRatio 分布区间 ──")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0)"]
anomalies["AmpRatio_bin"] = pd.cut(anomalies["AmpRatio"], bins=bins, labels=labels, include_lowest=True)
for label in labels:
    count = (anomalies["AmpRatio_bin"] == label).sum()
    print(f"  {label}: {count} 条")

print("\n完成。")
