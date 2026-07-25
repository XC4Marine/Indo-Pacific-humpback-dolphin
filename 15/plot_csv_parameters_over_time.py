#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 click 分段聚类 CSV 中各参数随时间变化的图。

默认读取 100 us 分段分析结果：
D:\Project_Github\Indo-Pacific-humpback-dolphin\12_ClickSegmentClustering\outputs_100us\Pulse_027_segments_dbscan.csv

运行环境参照 .agents/skills/toothwhale-project/SKILL.md：
& "D:\Python_env\toothwhale\python.exe" script.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib


BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
MODULE_DIR = BASE_DIR / "15"
DEFAULT_CSV_PATH = (
    BASE_DIR
    / "12_ClickSegmentClustering"
    / "outputs_100us"
    / "Pulse_027_segments_dbscan.csv"
)
DEFAULT_OUTPUT_DIR = MODULE_DIR / "outputs"

# 将 Matplotlib 缓存放到项目可写目录，避免用户目录权限导致缓存写入失败。
os.environ.setdefault("MPLCONFIGDIR", str(MODULE_DIR / ".matplotlib"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "segment_index",
    "start_time_s",
    "end_time_s",
    "center_time_s",
    "rms_energy",
    "norm_xcorr_peak",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "dbscan_label",
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Plot segment-level click clustering parameters over time."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="输入 CSV 文件的绝对路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出图片所在文件夹的绝对路径。",
    )
    parser.add_argument(
        "--output-name",
        default="Pulse_027_parameters_over_time.png",
        help="输出 PNG 文件名。",
    )
    parser.add_argument(
        "--title",
        default="Pulse_027 segment parameters over time",
        help="图像标题。",
    )
    return parser.parse_args()


def load_segment_csv(csv_path: Path) -> pd.DataFrame:
    """读取并检查 click 分段聚类 CSV。"""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要字段: {missing}")

    if df.empty:
        raise ValueError(f"CSV 文件为空: {csv_path}")

    # 按片段中心时刻排序，确保横轴时间单调递增。
    return df.sort_values("center_time_s").reset_index(drop=True)


def build_label_colors(labels: np.ndarray) -> tuple[list, dict[int, tuple]]:
    """按 DBSCAN 原始标签生成颜色映射。"""

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_by_label = {int(label): cmap(i) for i, label in enumerate(unique_labels)}
    colors = [color_by_label[int(label)] for label in labels]
    return colors, color_by_label


def plot_parameters_over_time(df: pd.DataFrame, output_path: Path, title: str) -> None:
    """绘制 RMS、互相关、谱质心、频谱带宽和 DBSCAN 标签随时间变化。"""

    time_ms = df["center_time_s"].to_numpy(dtype=float) * 1000.0
    labels = df["dbscan_label"].to_numpy(dtype=int)
    colors, color_by_label = build_label_colors(labels)

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1. RMS 能量
    axes[0].plot(time_ms, df["rms_energy"], color="#0072B2", linewidth=1.1)
    axes[0].scatter(time_ms, df["rms_energy"], c=colors, s=18)
    axes[0].set_ylabel("RMS")
    axes[0].grid(True, alpha=0.3)

    # 2. 与参考片段的归一化互相关峰值
    axes[1].plot(time_ms, df["norm_xcorr_peak"], color="#D55E00", linewidth=1.1)
    axes[1].scatter(time_ms, df["norm_xcorr_peak"], c=colors, s=18)
    axes[1].set_ylabel("Norm xcorr peak")
    axes[1].grid(True, alpha=0.3)

    # 3. 谱质心，单位换算为 kHz 便于阅读
    axes[2].plot(time_ms, df["spectral_centroid_hz"] / 1000.0, color="#009E73", linewidth=1.1)
    axes[2].scatter(time_ms, df["spectral_centroid_hz"] / 1000.0, c=colors, s=18)
    axes[2].set_ylabel("Centroid (kHz)")
    axes[2].grid(True, alpha=0.3)

    # 4. 频谱带宽，单位换算为 kHz
    axes[3].plot(time_ms, df["spectral_bandwidth_hz"] / 1000.0, color="#CC79A7", linewidth=1.1)
    axes[3].scatter(time_ms, df["spectral_bandwidth_hz"] / 1000.0, c=colors, s=18)
    axes[3].set_ylabel("Bandwidth (kHz)")
    axes[3].grid(True, alpha=0.3)

    # 5. DBSCAN 原始标签，不做人工重命名或类别修正
    axes[4].plot(time_ms, labels, color="#555555", linewidth=0.5, alpha=0.5)
    axes[4].scatter(time_ms, labels, c=colors, s=28, edgecolor="black", linewidth=0.25)
    axes[4].set_ylabel("DBSCAN label")
    axes[4].set_xlabel("Segment center time (ms)")
    axes[4].grid(True, alpha=0.3)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=7,
            label=f"Label {label}",
        )
        for label, color in sorted(color_by_label.items())
    ]
    axes[4].legend(handles=legend_handles, loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary(df: pd.DataFrame, csv_path: Path, output_path: Path) -> None:
    """输出绘图输入和标签统计。"""

    label_counts = df["dbscan_label"].value_counts().sort_index()
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(
        "Time range ms: "
        f"{df['center_time_s'].min() * 1000.0:.3f} - {df['center_time_s'].max() * 1000.0:.3f}"
    )
    print("DBSCAN label counts:")
    for label, count in label_counts.items():
        print(f"  label {int(label)}: {int(count)}")
    print(f"Figure: {output_path}")


def main() -> None:
    """主入口：读取 CSV 并绘制参数随时间变化图。"""

    args = parse_args()
    output_path = args.output_dir / args.output_name
    df = load_segment_csv(args.csv_path)
    plot_parameters_over_time(df, output_path, args.title)
    print_summary(df, args.csv_path, output_path)


if __name__ == "__main__":
    main()
