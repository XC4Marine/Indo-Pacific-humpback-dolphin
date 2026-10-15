#!/usr/bin/env python3
"""Plot the completed v2 SNR-stratified AUROC and Recall results."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "metrics" / "snr_results.csv"
FIGURES = ROOT / "figures"

METHODS = [
    ("teager", "Teager", "#666666", ":", "X", 1.5, 5.5),
    ("bcsc_summary", "BCSC-Summary", "#009E73", "--", "s", 1.7, 5.0),
    ("structure_only", "Structure-only Path-CNN", "#CC79A7", "-.", "^", 1.7, 5.5),
    ("waveform", "Waveform Transformer", "#222222", "-", "o", 2.2, 5.3),
    ("direct_fusion", "v1 direct fusion", "#E69F00", "--", "D", 2.0, 5.0),
    ("gated_zero_structure", "v2 gated + zero structure", "#D55E00", ":", "P", 2.0, 5.2),
    ("gated_fusion", "v2 gated fusion", "#0072B2", "-", "o", 3.0, 6.0),
]


def draw_panel(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str) -> None:
    for key, label, color, style, marker, width, size in METHODS:
        item = frame.loc[frame["condition"].eq(key)].sort_values("snr_low_db")
        axis.plot(
            item["snr_center"], item[column], label=label, color=color, linestyle=style,
            marker=marker, linewidth=width, markersize=size, markeredgecolor="white",
            markeredgewidth=0.65, zorder=4 if key == "gated_fusion" else 3,
        )
    axis.axvspan(-15, 0, color="#EAF2F8", zorder=0)
    axis.axvline(0, color="#9A9A9A", linewidth=0.8, zorder=1)
    axis.text(-14.75, 0.985, "Low-SNR primary range", va="top", ha="left", fontsize=8.3, color="#555555")
    axis.set_title(title, loc="left", fontweight="bold", pad=10)
    axis.set_xlabel("Positive target SNR (dB)")
    axis.set_xlim(-15.5, 15.5)
    axis.set_xticks([-12.5, -7.5, -2.5, 2.5, 7.5, 12.5])
    axis.set_xticklabels(["−15 to −10", "−10 to −5", "−5 to 0", "0 to 5", "5 to 10", "10 to 15"], rotation=24, ha="right")
    axis.set_ylim(0.35, 1.01)
    axis.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8.8)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing completed SNR result table: {SOURCE}")
    frame = pd.read_csv(SOURCE)
    expected = {key for key, *_ in METHODS}
    if set(frame["condition"].unique()) != expected:
        raise RuntimeError("The SNR result table does not contain the expected seven conditions.")
    if frame.groupby("condition").size().nunique() != 1 or frame.groupby("condition").size().iloc[0] != 6:
        raise RuntimeError("Each condition must have exactly six SNR bins.")
    frame["snr_center"] = (frame["snr_low_db"] + frame["snr_high_db"]) / 2

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#222222", "axes.titlecolor": "#222222",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), sharex=True)
    draw_panel(axes[0], frame, "auroc", "A. Discrimination: AUROC")
    draw_panel(axes[1], frame, "recall", "B. Thresholded detection: Recall")
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("Recall")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.6,
                  columnspacing=1.4, handlelength=2.6, bbox_to_anchor=(0.5, 0.085))
    figure.suptitle("Experiment 23-v2: performance across target-SNR bins", x=0.06, ha="left", y=0.98,
                    fontsize=14, fontweight="bold")
    figure.text(0.06, 0.012,
                "Source: completed five-fold OOF predictions (one seed). Each bin compares its click positives with all 3,500 negatives."
                " Recall uses the validation-derived threshold for each test fold.",
                fontsize=8.2, color="#555555")
    figure.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.26, wspace=0.20)

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "snr_auroc_recall.png", dpi=240, bbox_inches="tight")
    figure.savefig(FIGURES / "snr_auroc_recall.svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
