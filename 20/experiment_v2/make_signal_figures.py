#!/usr/bin/env python3
"""Create one three-panel PNG for every Python Bellhop v2 synthesized click."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


# Use an installed Windows CJK font so figure labels remain readable in PNGs.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "python_results"
FIGURE_NAME = "channel_signal_overview.png"


def delay_tap_profile(channel: np.lib.npyio.NpzFile, sample_rate: int, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the RMS delay-tap profile from all retained frequency-band arrivals.

    A Bellhop arrival is a delayed complex tap.  The wideband channel retains
    a frequency-specific set of such taps, so the squared gains are pooled in
    sample-wide delay bins rather than implying an unsupported phase alignment
    across the 5 kHz-spaced Bellhop runs.
    """
    # Derived results retain raw gated arrivals for provenance and add the
    # attenuated records actually used by synthesis.  Plot the latter so the
    # channel panel matches the received waveform.
    arrivals = channel["synthesis_arrivals"] if "synthesis_arrivals" in channel.files else channel["gated_arrivals"]
    relative_delay = arrivals["delay_s"] - float(channel["first_arrival_s"])
    valid = (relative_delay >= 0) & (relative_delay <= duration_s)
    n = int(np.ceil(duration_s * sample_rate)) + 1
    taps = np.zeros(n, dtype=float)
    sample = np.rint(relative_delay[valid] * sample_rate).astype(int)
    np.add.at(taps, sample, arrivals["effective_gain"][valid] ** 2)
    return np.arange(n) / sample_rate, np.sqrt(taps)


def plot_click(row: pd.Series, force: bool, duration_s: float, channel_annotation: str) -> bool:
    click_dir = Path(row.output_path)
    figure_path = click_dir / FIGURE_NAME
    if figure_path.is_file() and not force:
        return False
    source, fs = sf.read(row.input_wav, dtype="float64", always_2d=False)
    synthesized = np.load(click_dir / "rx_raw.npy")
    with np.load(click_dir / "channel.npz", allow_pickle=True) as channel:
        tap_time, tap_gain = delay_tap_profile(channel, int(fs), duration_s)
        first_arrival_ms = float(channel["first_arrival_s"]) * 1e3
    source_time = np.arange(source.size) / fs * 1e3
    signal_time = np.arange(synthesized.size) / fs * 1e3
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    identity = f"{row.click_id} | SNR={float(row.snr_db):.1f} dB | distance={float(row.range_m):.1f} m"
    fig.suptitle(identity + ("\n" + channel_annotation.removeprefix(" | ") if channel_annotation else ""), fontsize=11)

    axes[0].plot(source_time, source, color="#1f77b4", linewidth=1.0)
    axes[0].set_title("(1) click 原始信号")
    axes[0].set_xlabel("时间 (ms)")
    axes[0].set_ylabel("相对数字幅值")
    axes[0].grid(alpha=0.3)

    axes[1].plot(tap_time * 1e3, tap_gain, color="#d62728", linewidth=0.8)
    axes[1].fill_between(tap_time * 1e3, tap_gain, color="#d62728", alpha=0.25)
    axes[1].set_title(f"(2) 时域信道：20–190 kHz 有效到达的 RMS 时延抽头（0–{duration_s * 1e3:g} ms）")
    axes[1].set_xlabel("相对首达时间 (ms)")
    axes[1].set_ylabel("相对抽头增益")
    axes[1].set_xlim(0, duration_s * 1e3)
    axes[1].grid(alpha=0.3)

    axes[2].plot(signal_time, synthesized, color="#2ca02c", linewidth=0.8)
    axes[2].set_title(f"(3) 合成多径信号（首达绝对时间 {first_arrival_ms:.3f} ms）")
    axes[2].set_xlabel("相对首达时间 (ms)")
    axes[2].set_ylabel("相对数字幅值")
    axes[2].grid(alpha=0.3)

    temporary = figure_path.with_name(figure_path.stem + ".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    os.replace(temporary, figure_path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="重绘已存在的 PNG。")
    parser.add_argument("--results-root", type=Path, default=RESULTS, help="待绘制的结果根目录。")
    parser.add_argument("--channel-window-ms", type=float, default=None, help="覆盖配置中的时延窗（毫秒）。")
    args = parser.parse_args()
    config_file = args.results_root / "config.json"
    if args.channel_window_ms is None:
        config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
        channel_config = config.get("derivation", config.get("config", {}))
        duration_s = float(channel_config.get("arrival_window_s", 0.010))
    else:
        duration_s = args.channel_window_ms / 1e3
        config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
        channel_config = config.get("derivation", config.get("config", {}))
    if duration_s <= 0:
        raise ValueError("时延窗必须为正数。")
    annotation_parts = []
    max_arrivals = channel_config.get("max_arrivals_per_frequency")
    if max_arrivals is not None and int(max_arrivals) > 0:
        annotation_parts.append(f"Top-{max_arrivals}/5 kHz")
    high_frequency_loss = float(channel_config.get("high_frequency_extra_at_190khz_db", 0.0))
    if high_frequency_loss:
        annotation_parts.append(f"20–190 kHz HF -{high_frequency_loss:g} dB")
    direct_advantage = channel_config.get("minimum_direct_over_reflection_db")
    if direct_advantage is not None:
        annotation_parts.append(f"direct ≥ reflection +{float(direct_advantage):g} dB")
    channel_annotation = " | " + " | ".join(annotation_parts) if annotation_parts else ""
    manifest = pd.read_csv(args.results_root / "click_manifest.csv")
    successful = manifest[manifest.status == "success"].copy()
    if len(successful) != 110:
        raise RuntimeError(f"预期 110 条成功结果，实际为 {len(successful)}。")
    created = 0
    for index, (_, row) in enumerate(successful.iterrows(), start=1):
        created += int(plot_click(row, args.force, duration_s, channel_annotation))
        print(f"[{index:3d}/110] {row.click_id}", flush=True)
    print(f"完成：新增 {created} 张，目录中总计 {len(successful)} 张 {FIGURE_NAME}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
