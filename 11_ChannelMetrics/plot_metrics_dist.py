#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多径信道参数分布可视化脚本

读取 pulse_params.txt 中的多径数据，绘制 Delay_us、AmpRatio、Coherence 三个指标的
分布图（3x1 布局：直方图），其中 AmpRatio 轴使用对数刻度以展示长尾分布。
输出 11_ChannelMetrics/metrics_distribution.png。

Author: Codex
Date: 2026-06-27
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── 路径配置 ──────────────────────────────────────────────────
BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
METRICS_DIR = BASE_DIR / "11_ChannelMetrics"
INPUT_FILE = METRICS_DIR / "pulse_params.txt"
OUTPUT_PNG = METRICS_DIR / "metrics_distribution.png"

# ── 读取数据 ──────────────────────────────────────────────────
print("读取 pulse_params.txt ...")
df = pd.read_csv(INPUT_FILE, sep="\t")
# 过滤掉 Direct 行，只保留多径
mp = df[df["PathType"] != "Direct"].copy()
print(f"多径记录数: {len(mp)}")

# ── 统计摘要 ──────────────────────────────────────────────────
for col in ["Delay_us", "AmpRatio", "Coherence"]:
    s = mp[col]
    print(f"\n{col}:")
    print(f"  Mean ± Std: {s.mean():.4f} ± {s.std():.4f}")
    print(f"  Median: {s.median():.4f}")
    print(f"  Min / Max: {s.min():.4f} / {s.max():.4f}")

# ── 绘图 ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

# ─ 子图1: Delay_us ─
ax1 = axes[0]
ax1.hist(mp["Delay_us"], bins=80, color="steelblue", edgecolor="white",
         alpha=0.85, density=True)
ax1.set_xlabel("Delay (μs)")
ax1.set_ylabel("Density")
ax1.set_title("(a) Delay_us Distribution")
median_delay = mp["Delay_us"].median()
ax1.axvline(median_delay, color="red", linestyle="--", linewidth=1.2,
            label=f"Median = {median_delay:.1f} μs")
ax1.legend(fontsize=9)

# ─ 子图2: AmpRatio (log scale) ─
ax2 = axes[1]
# 对数均匀分箱
bins = np.logspace(np.log10(max(mp["AmpRatio"].min(), 0.01)),
                   np.log10(mp["AmpRatio"].max()), 80)
ax2.hist(mp["AmpRatio"], bins=bins, color="darkorange", edgecolor="white",
         alpha=0.85, density=True)
ax2.set_xscale("log")
ax2.set_xlabel("AmpRatio (log scale)")
ax2.set_ylabel("Density")
ax2.set_title("(b) AmpRatio Distribution (Log Scale)")
median_amp = mp["AmpRatio"].median()
ax2.axvline(median_amp, color="red", linestyle="--", linewidth=1.2,
            label=f"Median = {median_amp:.2f}")
# AmpRatio=1 参考线
ax2.axvline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ylim = ax2.get_ylim()
ax2.text(1.0, ylim[1] * 0.95, " AmpRatio=1", fontsize=8, color="gray", va="top")
ax2.legend(fontsize=9)

# ─ 子图3: Coherence ─
ax3 = axes[2]
ax3.hist(mp["Coherence"], bins=60, color="seagreen", edgecolor="white",
         alpha=0.85, density=True)
ax3.set_xlabel("Coherence")
ax3.set_ylabel("Density")
ax3.set_title("(c) Coherence Distribution")
median_coh = mp["Coherence"].median()
ax3.axvline(median_coh, color="red", linestyle="--", linewidth=1.2,
            label=f"Median = {median_coh:.3f}")
ax3.legend(fontsize=9)

# ── 保存 ──────────────────────────────────────────────────────
fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\n分布图已保存: {OUTPUT_PNG}")
print("完成。")
