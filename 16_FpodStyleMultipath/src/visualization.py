from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .multipath_detection import MultipathFeature


FREQUENCY_COLORS = {
    "low": "#0072FF",
    "mid_low": "#00D0A8",
    "mid_high": "#FFD21A",
    "high": "#FF3B1F",
}


def write_features_csv(features: list[MultipathFeature], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "click_id",
        "cluster_id",
        "arrival_time_us",
        "delay_us",
        "frequency_khz",
        "amplitude_db",
        "n_wave_peaks",
        "dominant_ipi_us",
        "click_duration_us",
        "is_direct_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in features:
            writer.writerow(
                {
                    "click_id": item.click_id,
                    "cluster_id": item.cluster_id,
                    "arrival_time_us": f"{item.arrival_time_us:.3f}",
                    "delay_us": f"{item.delay_us:.3f}",
                    "frequency_khz": f"{item.frequency_khz:.3f}",
                    "amplitude_db": f"{item.amplitude_db:.3f}",
                    "n_wave_peaks": item.n_wave_peaks,
                    "dominant_ipi_us": f"{item.dominant_ipi_us:.3f}",
                    "click_duration_us": f"{item.click_duration_us:.3f}",
                    "is_direct_path": int(item.is_direct_path),
                }
            )


def _frequency_color(frequency_khz: float) -> str:
    if not np.isfinite(frequency_khz):
        return "#FFFFFF"
    if frequency_khz < 40.0:
        return FREQUENCY_COLORS["low"]
    if frequency_khz < 80.0:
        return FREQUENCY_COLORS["mid_low"]
    if frequency_khz < 120.0:
        return FREQUENCY_COLORS["mid_high"]
    return FREQUENCY_COLORS["high"]


def _visible_frequency_points(item: MultipathFeature) -> list[float]:
    if np.isfinite(item.frequency_khz):
        return [float(np.clip(item.frequency_khz, 10.0, 145.0))]
    return []


def _pseudo_spl(item: MultipathFeature) -> float:
    if item.is_direct_path or item.delay_us <= 5.0:
        return 640.0
    return float(np.clip(640.0 * (10.0 ** (item.amplitude_db / 20.0)), 25.0, 640.0))


def _configure_fpod_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black", labelsize=8, length=3)
    ax.grid(False)


def plot_multipath_cluster(
    features: list[MultipathFeature],
    output_path: Path,
    title: str,
    time_start_ms: float | None = None,
    time_end_ms: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stable = sorted(features, key=lambda item: (item.arrival_time_us, item.delay_us))
    if time_start_ms is not None and time_end_ms is not None:
        stable = [
            item
            for item in stable
            if time_start_ms <= item.arrival_time_us / 1000.0 <= time_end_ms
        ]

    max_time_ms = max([item.arrival_time_us / 1000.0 for item in stable] + [400.0])
    x_limit_ms = max(400.0, np.ceil(max_time_ms / 10.0) * 10.0)
    if time_start_ms is not None and time_end_ms is not None:
        x_min_ms = float(time_start_ms)
        x_max_ms = float(time_end_ms)
    else:
        x_min_ms = 0.0
        x_max_ms = x_limit_ms

    fig, (ax_freq, ax_amp) = plt.subplots(
        2,
        1,
        figsize=(15.5, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.25], "hspace": 0.0},
    )
    fig.patch.set_facecolor("white")
    _configure_fpod_axis(ax_freq)
    _configure_fpod_axis(ax_amp)

    for item in stable:
        x_ms = item.arrival_time_us / 1000.0
        freqs = _visible_frequency_points(item)
        color = _frequency_color(item.frequency_khz)
        if freqs:
            ax_freq.scatter(
                [x_ms],
                freqs,
                s=7.0 if item.is_direct_path else 5.0,
                c=color,
                marker="s",
                linewidths=0,
                alpha=0.95 if item.is_direct_path else 0.82,
            )

        spl = _pseudo_spl(item)
        linewidth = 1.2 if item.is_direct_path else 0.85
        ax_amp.vlines(x_ms, 0.0, spl, colors=color, linewidth=linewidth, alpha=0.96)

    ax_freq.set_ylim(0.0, 150.0)
    ax_freq.set_ylabel("Frequency - kHz", color="black", fontsize=9)
    ax_freq.yaxis.set_label_position("left")
    ax_freq.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax_freq.set_yticks(np.arange(10, 150, 10))
    ax_freq.set_yticklabels([f"{v:d}kHz" for v in range(10, 150, 10)])
    ax_freq.axhline(0.0, color="#404040", linewidth=0.8)

    ax_amp.set_ylim(0.0, 760.0)
    ax_amp.set_ylabel("Amplitude full scale", color="black", fontsize=9)
    ax_amp.yaxis.set_label_position("left")
    ax_amp.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    ax_amp.set_yticks(np.arange(100, 800, 100))
    ax_amp.set_yticklabels([f"{v:d}SPL" for v in range(100, 800, 100)])
    ax_amp.set_xlim(x_min_ms, x_max_ms)

    xticks = np.arange(x_min_ms, x_max_ms + 0.1, 10.0)
    ax_amp.set_xticks(xticks)
    ax_amp.set_xticklabels(["0 msecs" if v == 0 else f"{int(v)}ms" for v in xticks], fontsize=7)
    ax_amp.tick_params(axis="x", colors="black", labelsize=7, length=6)
    ax_amp.set_xlabel("")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.955, top=0.94, bottom=0.085)
    fig.savefig(str(output_path), dpi=180)
    plt.close(fig)
