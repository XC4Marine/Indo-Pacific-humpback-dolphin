"""Generate the full-dataset single-click versus joint-click comparison table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\20\full_dataset_experiment")
CHANNELS_CSV = ROOT / "all_channels.csv"
SUMMARY_CSV = ROOT / "pulsetrain_summary.csv"
SOURCE_CSV = ROOT / "single_vs_joint_full_718_table.csv"
FIGURE_DIR = ROOT / "figures"
SVG_PATH = FIGURE_DIR / "single_vs_joint_full_718_table.svg"
PNG_PATH = FIGURE_DIR / "single_vs_joint_full_718_table.png"


def progress(percent: int, message: str) -> None:
    width = 30
    done = round(width * percent / 100)
    bar = "#" * done + "-" * (width - done)
    print(f"\rOverall progress [{bar}] {percent:3d}%  {message}", end="", flush=True)
    if percent == 100:
        print()


def summarize(frame: pd.DataFrame, n_groups: int) -> dict[str, float | int]:
    non_direct = frame.loc[frame["path"].ne("direct")]
    return {
        "non_direct_candidates": int(len(non_direct)),
        "mean_confidence": float(non_direct["confidence"].mean()),
        "median_confidence": float(non_direct["confidence"].median()),
        "high_confidence_candidates": int(non_direct["confidence"].ge(0.90).sum()),
        "mean_paths_per_group": float(len(frame) / n_groups),
        "direct_paths": int(frame["path"].eq("direct").sum()),
    }


def main() -> None:
    progress(0, "Loading full experiment data")
    columns = ["method", "voyage", "pulse_train", "click", "path", "confidence"]
    channels = pd.read_csv(CHANNELS_CSV, usecols=columns)
    summary = pd.read_csv(
        SUMMARY_CSV, usecols=["voyage", "pulse_train", "anchor_click"]
    )
    n_groups = len(summary)

    progress(25, "Building paired-anchor comparison")
    single = channels.loc[channels["method"].eq("single")].copy()
    joint_anchor = channels.loc[channels["method"].eq("joint")].merge(
        summary,
        left_on=["voyage", "pulse_train", "click"],
        right_on=["voyage", "pulse_train", "anchor_click"],
        how="inner",
        validate="many_to_one",
    )

    single_stats = summarize(single, n_groups)
    joint_stats = summarize(joint_anchor, n_groups)
    if single_stats["direct_paths"] != n_groups or joint_stats["direct_paths"] != n_groups:
        raise RuntimeError("每种方法应当在每个 PulseTrain 中恰好包含一条直达径。")

    rows = [
        ("非直达径候选数", "non_direct_candidates", "条"),
        ("平均置信度", "mean_confidence", "%"),
        ("置信度中位数", "median_confidence", "%"),
        ("高置信候选数（≥90%）", "high_confidence_candidates", "条"),
        ("平均路径数/组", "mean_paths_per_group", "条/组"),
    ]

    def display_value(key: str, value: float | int) -> str:
        if key in {"mean_confidence", "median_confidence"}:
            return f"{100 * float(value):.1f}%"
        if key == "mean_paths_per_group":
            return f"{float(value):.2f}"
        return f"{int(value):,}"

    progress(45, "Writing reproducible table data")
    with SOURCE_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "single_click",
                "multi_click_joint",
                "unit",
                "definition",
            ],
        )
        writer.writeheader()
        definitions = {
            "non_direct_candidates": "同一锚点click口径下，path不等于direct的候选路径行数",
            "mean_confidence": "所有非直达径候选的confidence算术平均值",
            "median_confidence": "所有非直达径候选的confidence中位数",
            "high_confidence_candidates": "非直达径候选中confidence大于或等于0.90的行数",
            "mean_paths_per_group": "含直达径的候选路径总行数除以718个PulseTrain",
        }
        for label, key, unit in rows:
            single_value = single_stats[key]
            joint_value = joint_stats[key]
            if key in {"mean_confidence", "median_confidence"}:
                single_value = 100 * float(single_value)
                joint_value = 100 * float(joint_value)
            writer.writerow(
                {
                    "metric": label,
                    "single_click": single_value,
                    "multi_click_joint": joint_value,
                    "unit": unit,
                    "definition": definitions[key],
                }
            )

    progress(60, "Rendering publication table")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(9.2, 5.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#222222"
    muted = "#666666"
    grid = "#A6A6A6"
    accent = "#D55E00"
    light_accent = "#FBE9E2"
    x_edges = [0.04, 0.43, 0.715, 0.96]
    x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(3)]
    y_top, y_bottom = 0.82, 0.19
    row_h = (y_top - y_bottom) / 6
    y_centers = [y_top - row_h * (i + 0.5) for i in range(6)]

    ax.text(
        0.04,
        0.935,
        "全量 718 组：单 click 与多 click 联合对比",
        fontsize=20,
        fontweight="bold",
        color=ink,
        va="center",
    )
    ax.text(
        0.04,
        0.875,
        "相同 SNR 最高锚点 click 的配对比较",
        fontsize=10.5,
        color=muted,
        va="center",
    )

    ax.add_patch(Rectangle((x_edges[0], y_top - row_h), x_edges[-1] - x_edges[0], row_h,
                           facecolor="#F2F2F2", edgecolor="none"))
    for x in x_edges[1:-1]:
        ax.plot([x, x], [y_bottom, y_top], color=grid, linewidth=0.9, dashes=(1.5, 4))
    ax.plot([x_edges[0], x_edges[-1]], [y_top, y_top], color=ink, linewidth=1.5)
    for i in range(1, 6):
        y = y_top - i * row_h
        ax.plot([x_edges[0], x_edges[-1]], [y, y], color=grid, linewidth=0.8, dashes=(1.5, 4))
    ax.plot([x_edges[0], x_edges[-1]], [y_bottom, y_bottom], color=ink, linewidth=1.5)

    headers = ["指标", "单 click", "多 click 联合"]
    for x, label in zip(x_centers, headers):
        ax.text(x, y_centers[0], label, ha="center", va="center", fontsize=15.5,
                fontweight="bold", color=ink)

    display_rows = []
    for label, key, _ in rows:
        display_rows.append(
            (label, display_value(key, single_stats[key]), display_value(key, joint_stats[key]), key)
        )

    for row_index, (label, left, right, key) in enumerate(display_rows, start=1):
        y = y_centers[row_index]
        label_color = accent if key == "high_confidence_candidates" else ink
        label_weight = "bold" if key in {"mean_confidence", "high_confidence_candidates"} else "normal"
        label_bbox = dict(boxstyle="round,pad=0.18", facecolor=light_accent, edgecolor="none") \
            if key == "mean_confidence" else None
        ax.text(x_centers[0], y, label, ha="center", va="center", fontsize=13.2,
                fontweight=label_weight, color=label_color, bbox=label_bbox)
        for x, value in zip(x_centers[1:], [left, right]):
            if key == "mean_confidence":
                ax.text(x, y, value, ha="center", va="center", fontsize=14.2,
                        fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor=accent, edgecolor="none"))
            else:
                value_color = accent if key == "high_confidence_candidates" else ink
                value_weight = "bold" if key == "high_confidence_candidates" else "normal"
                ax.text(x, y, value, ha="center", va="center", fontsize=14,
                        fontweight=value_weight, color=value_color)

    ax.text(
        0.04,
        0.105,
        "注：联合方法仅统计与单 click 相同的锚点；高置信度 = 检出率 ≥90%；平均路径数包含直达径。",
        fontsize=9.5,
        color=muted,
        va="center",
    )
    ax.text(
        0.04,
        0.058,
        "数据源：all_channels.csv、pulsetrain_summary.csv",
        fontsize=8.6,
        color=muted,
        va="center",
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    progress(100, "Complete")
    print(f"CSV: {SOURCE_CSV}")
    print(f"SVG: {SVG_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()
