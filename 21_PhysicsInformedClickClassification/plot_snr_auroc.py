#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从分SNR汇总结果生成三种方法的AUROC折线图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs_snr_m15_p15_full_5seeds"
SOURCE_CSV = OUTPUT_ROOT / "metrics" / "snr_bin_summary.csv"
FIGURE_DIR = OUTPUT_ROOT / "figures"

METHODS = (
    ("waveform_transformer", "Waveform Transformer", "#0072B2", "o", "-"),
    ("recon_bce", "Recon + BCE", "#E69F00", "s", "--"),
    ("recon_physics_bce", "Recon + Physics + BCE", "#009E73", "^", "-."),
)


def configure_fonts() -> None:
    available = {font.name for font in fm.fontManager.ttflist}
    preferred = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS")
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": selected,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制不同方法的分SNR AUROC折线图")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-stem", default="snr_auroc_lines")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    configure_fonts()
    source_csv = arguments.source_csv.resolve()
    data = pd.read_csv(source_csv)
    required = {"condition", "snr_lower_db", "snr_upper_db", "auroc_mean"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"分SNR汇总缺少列：{sorted(missing)}")

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for condition, label, color, marker, linestyle in METHODS:
        frame = data[data["condition"] == condition].sort_values("snr_lower_db")
        if len(frame) != 6:
            raise ValueError(f"{condition} 应包含6个SNR区间，实际为 {len(frame)}")
        centers = (frame["snr_lower_db"] + frame["snr_upper_db"]) / 2.0
        axis.plot(
            centers,
            frame["auroc_mean"],
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.3,
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ordered = data[data["condition"] == METHODS[0][0]].sort_values("snr_lower_db")
    centers = (ordered["snr_lower_db"] + ordered["snr_upper_db"]) / 2.0
    def signed_tick(value: float) -> str:
        return f"{value:g}".replace("-", "−")

    tick_labels = [
        f"{signed_tick(lower)}–{signed_tick(upper)}"
        for lower, upper in zip(ordered["snr_lower_db"], ordered["snr_upper_db"])
    ]
    axis.set_xticks(centers, tick_labels)
    axis.set_xlabel("目标SNR区间 (dB)", fontsize=12)
    axis.set_ylabel("AUROC", fontsize=12)
    axis.set_ylim(0.5, 1.0)
    axis.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.axhline(0.5, color="#A6A6A6", linewidth=1.0, linestyle=":", zorder=1)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75, zorder=0)
    axis.tick_params(axis="both", labelsize=10.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D9D9D9",
        fontsize=10,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{arguments.output_stem}.png"
    svg_path = FIGURE_DIR / f"{arguments.output_stem}.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Source: {source_csv}")
    print(f"PNG: {png_path.resolve()}")
    print(f"SVG: {svg_path.resolve()}")


if __name__ == "__main__":
    main()
