#!/usr/bin/env python3
"""Create original-click, original-channel, and pulse-train PNGs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "clicktrains_10click_perturbed"
FIGURE_NAME = "pulsetrain_overview.png"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def delay_tap_profile(channel: np.lib.npyio.NpzFile, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float]:
    arrivals = channel["synthesis_arrivals"] if "synthesis_arrivals" in channel.files else channel["gated_arrivals"]
    first_arrival_s = float(channel["first_arrival_s"])
    duration_s = float(channel["arrival_window_s"]) if "arrival_window_s" in channel.files else 0.004
    relative_delay = arrivals["delay_s"] - first_arrival_s
    valid = (relative_delay >= 0) & (relative_delay <= duration_s)
    taps = np.zeros(int(np.ceil(duration_s * sample_rate)) + 1, dtype=float)
    index = np.rint(relative_delay[valid] * sample_rate).astype(int)
    np.add.at(taps, index, arrivals["effective_gain"][valid] ** 2)
    return np.arange(len(taps)) / sample_rate, np.sqrt(taps), duration_s


def envelope_for_plot(signal: np.ndarray, sample_rate: int, max_bins: int = 18000) -> tuple[np.ndarray, np.ndarray]:
    """Return a min/max envelope so long IPI gaps remain fast to render."""
    if len(signal) <= max_bins:
        return np.arange(len(signal)) / sample_rate * 1e3, signal
    edges = np.linspace(0, len(signal), max_bins + 1, dtype=int)
    lower = np.empty(max_bins); upper = np.empty(max_bins)
    for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        block = signal[start:stop]
        lower[i], upper[i] = block.min(), block.max()
    times = (edges[:-1] + edges[1:] - 1) / (2 * sample_rate) * 1e3
    return np.repeat(times, 2), np.column_stack((lower, upper)).ravel()


def plot_train(row: pd.Series, force: bool) -> bool:
    directory = Path(row.output_path)
    figure = directory / FIGURE_NAME
    if figure.is_file() and not force:
        return False
    original_click, sample_rate = sf.read(directory / "original_click.wav", dtype="float64", always_2d=False)
    pulse_train = np.load(directory / "pulsetrain_raw.npy")
    events = pd.read_csv(directory / "event_manifest.csv")
    with np.load(directory / "original_channel.npz", allow_pickle=True) as channel:
        tap_time, tap_gain, channel_window_s = delay_tap_profile(channel, int(sample_rate))
    click_time = np.arange(len(original_click)) / sample_rate * 1e3
    train_time, train_amplitude = envelope_for_plot(pulse_train, int(sample_rate))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(f"{row.click_id} | {len(events)} 个事件 | 时长 {len(pulse_train) / sample_rate:.3f} s", fontsize=12)

    axes[0].plot(click_time, original_click, color="#1f77b4", linewidth=1.0)
    axes[0].set_title("(1) 原始 click")
    axes[0].set_xlabel("时间 (ms)")
    axes[0].set_ylabel("相对数字幅值")
    axes[0].grid(alpha=0.3)

    axes[1].plot(tap_time * 1e3, tap_gain, color="#d62728", linewidth=0.8)
    axes[1].fill_between(tap_time * 1e3, tap_gain, color="#d62728", alpha=0.25)
    axes[1].set_title(f"(2) 原始时域信道（0–{channel_window_s * 1e3:g} ms）")
    axes[1].set_xlim(0, channel_window_s * 1e3)
    axes[1].set_xlabel("相对首达时间 (ms)")
    axes[1].set_ylabel("RMS 抽头增益")
    axes[1].grid(alpha=0.3)

    axes[2].plot(train_time, train_amplitude, color="#2ca02c", linewidth=0.7)
    for start_ms in events["start_time_ms"]:
        axes[2].axvline(start_ms, color="#555555", alpha=0.18, linewidth=0.5)
    axes[2].set_title("(3) 扰动 IPI 驱动的 pulse train")
    axes[2].set_xlabel("时间 (ms)")
    axes[2].set_ylabel("相对数字幅值（min/max 包络）")
    axes[2].grid(alpha=0.3)

    temporary = figure.with_name(figure.stem + ".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    os.replace(temporary, figure)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = pd.read_csv(args.results_root / "clicktrain_manifest.csv")
    successful = manifest[manifest["status"] == "success"].copy()
    if len(successful) != 110:
        raise RuntimeError(f"预期 110 条成功 pulse train，实际为 {len(successful)}。")
    created = 0
    for index, (_, row) in enumerate(successful.iterrows(), start=1):
        created += int(plot_train(row, args.force))
        print(f"[{index:3d}/110] {row.click_id}", flush=True)
    print(f"完成：新增 {created} 张，目录中总计 {len(successful)} 张 {FIGURE_NAME}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
