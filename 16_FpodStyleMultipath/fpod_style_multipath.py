#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create F-POD-style simulated FP1 records and multipath-cluster plots.

This script does not claim to recreate the proprietary F-POD firmware output.
It reconstructs the 16-byte FP1 record shape described in Ivanchikova &
Tregenza (2023), S1 File, from an already extracted WAV click snippet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin") / "16_FpodStyleMultipath" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
DEFAULT_WAV_PATH = (
    BASE_DIR
    / "00_Data"
    / "Positive_Clicks_3500"
    / "Voyage_Ori_15"
    / "PulseTrain_474"
    / "Click_027"
    / "Pulse_027.wav"
)
DEFAULT_ANCHOR_PATH = BASE_DIR / "10_AutoCorrMultipath" / "calibrated_anchors.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "16_FpodStyleMultipath" / "outputs"

TIME_UNIT_S = 5e-6
IPI_UNIT_S = 250e-9
MIN_FREQ_KHZ = 20.0
MAX_FREQ_KHZ = 160.0
MIN_LAG_S = 0.10e-3
MAX_LAG_S = 4.99e-3
MAX_MULTIPATH = 3
ACF_THRESHOLD = 0.10
MARK_HALF_WIDTH_S = 70e-6


@dataclass
class FpodStyleRecord:
    record_role: str
    center_time_ms: float
    relative_delay_us: float
    time_5us_units: int
    n_cycles: int
    loudest_cycle_number: int
    ipi_range_250ns_units: int
    ipi_pk_minus_1: int
    ipi_pk: int
    ipi_pk_plus_1: int
    ipi_pk_plus_2: int
    amp_p_minus_1: int
    amp_pmax: int
    amp_p_plus_1: int
    ipi_before_click_start: int
    amplitude_reversals: int
    duration_250ns_units: int
    last_cycle_ipi_compressed: int
    sonar_flag: int
    frequency_khz: float
    bandwidth_index: float
    cluster_size: int
    cluster_lowest_frequency_khz: float
    cluster_highest_frequency_khz: float
    simulated_fp1_bytes_hex: str
    simulation_note: str


def progress(step: int, total: int, text: str) -> None:
    width = 28
    filled = round(width * step / total)
    bar = "#" * filled + "." * (width - filled)
    print(f"[{bar}] {step}/{total} {text}", flush=True)


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        scale = 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        scale = 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    return sample_rate, data / scale


def load_anchor_sample(anchor_path: Path, wav_path: Path, sample_rate: int, signal: np.ndarray) -> int:
    train = wav_path.parent.parent.name
    click = wav_path.parent.name
    if anchor_path.exists():
        with anchor_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[0] == train and parts[1] == click:
                    return int(parts[3])

    half_window = max(1, int(round(50e-6 * sample_rate)))
    sq = signal * signal
    kernel = np.ones(2 * half_window + 1)
    energy = np.convolve(sq, kernel, mode="same")
    return int(np.argmax(energy))


def autocorr_positive(signal: np.ndarray) -> np.ndarray:
    centered = signal - np.mean(signal)
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if corr[0] <= 0:
        return np.ones_like(corr)
    return corr / corr[0]


def find_local_peaks(values: np.ndarray, min_distance: int, threshold: float) -> list[int]:
    candidates = []
    last_kept = -10**9
    for idx in range(1, len(values) - 1):
        if values[idx] < threshold:
            continue
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            if idx - last_kept >= min_distance:
                candidates.append(idx)
                last_kept = idx
            elif candidates and values[idx] > values[candidates[-1]]:
                candidates[-1] = idx
                last_kept = idx
    return candidates


def detect_multipath_lags(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[int], list[float]]:
    acf = autocorr_positive(signal)
    min_lag = int(round(MIN_LAG_S * sample_rate))
    max_lag = min(len(acf) - 1, int(round(MAX_LAG_S * sample_rate)))
    min_distance = max(1, int(round(50e-6 * sample_rate)))
    valid = acf[min_lag : max_lag + 1]
    peaks = find_local_peaks(valid, min_distance=min_distance, threshold=ACF_THRESHOLD)
    scored = [(min_lag + peak, float(valid[peak])) for peak in peaks]
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:MAX_MULTIPATH]
    selected.sort(key=lambda item: item[0])
    return acf, [item[0] for item in selected], [item[1] for item in selected]


def moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, samples)
    kernel = np.ones(samples) / samples
    return np.convolve(values, kernel, mode="same")


def click_bounds(signal: np.ndarray, sample_rate: int, center_sample: int) -> tuple[int, int]:
    radius = max(4, int(round(0.10e-3 * sample_rate)))
    left = max(0, center_sample - radius)
    right = min(len(signal), center_sample + radius + 1)
    local = signal[left:right]
    envelope = moving_average(np.abs(local), max(1, int(round(8e-6 * sample_rate))))
    peak_local = int(np.argmax(envelope))
    peak_value = float(envelope[peak_local])
    baseline = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - baseline)))
    threshold = max(0.12 * peak_value, baseline + 4.0 * mad)

    quiet_needed = max(2, int(round(18e-6 * sample_rate)))
    start = peak_local
    quiet = 0
    for idx in range(peak_local, -1, -1):
        quiet = quiet + 1 if envelope[idx] < threshold else 0
        if quiet >= quiet_needed:
            start = idx + quiet_needed
            break

    end = peak_local
    quiet = 0
    for idx in range(peak_local, len(envelope)):
        quiet = quiet + 1 if envelope[idx] < threshold else 0
        if quiet >= quiet_needed:
            end = idx - quiet_needed
            break
    if end <= start:
        start = max(0, peak_local - int(round(60e-6 * sample_rate)))
        end = min(len(local) - 1, peak_local + int(round(60e-6 * sample_rate)))
    return left + start, left + end + 1


def positive_cycle_peaks(segment: np.ndarray, sample_rate: int) -> list[int]:
    if len(segment) < 3:
        return []
    working = segment.copy()
    if abs(float(np.min(working))) > abs(float(np.max(working))):
        working *= -1.0
    min_gap = max(1, int(math.floor(sample_rate / 220000.0)))
    peaks = []
    for idx in range(1, len(working) - 1):
        if working[idx] > 0 and working[idx] >= working[idx - 1] and working[idx] > working[idx + 1]:
            if peaks and idx - peaks[-1] < min_gap:
                if working[idx] > working[peaks[-1]]:
                    peaks[-1] = idx
            else:
                peaks.append(idx)
    return peaks


def to_ipi_units(seconds: float) -> int:
    return int(max(0, min(255, round(seconds / IPI_UNIT_S))))


def pack_24bit(value: int) -> tuple[int, int, int]:
    value = max(0, min(0xFFFFFF, int(value)))
    return (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF


def reversal_count(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    diffs = np.diff(np.asarray(values, dtype=np.float64))
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(signs[1:] != signs[:-1]))


def estimate_record(
    signal: np.ndarray,
    sample_rate: int,
    center_sample: int,
    direct_sample: int,
    role: str,
    global_peak_abs: float,
    cluster_size: int,
    cluster_low_freq: float,
    cluster_high_freq: float,
) -> FpodStyleRecord:
    start, end = click_bounds(signal, sample_rate, center_sample)
    segment = signal[start:end]
    peak_indices = positive_cycle_peaks(segment, sample_rate)
    if not peak_indices:
        peak_indices = [int(np.argmax(np.abs(segment)))]

    amps = [abs(float(segment[idx])) for idx in peak_indices]
    pmax_i = int(np.argmax(amps))
    pmax_cycle = pmax_i + 1
    n_cycles = min(255, len(peak_indices))
    peak_times_s = [(start + idx) / sample_rate for idx in peak_indices]
    ipis = np.diff(peak_times_s).tolist()
    ipi_units = [to_ipi_units(ipi) for ipi in ipis]

    def ipi_at(offset_from_pmax: int) -> int:
        ipi_index = pmax_i + offset_from_pmax
        if 0 <= ipi_index < len(ipi_units):
            return int(ipi_units[ipi_index])
        return 0

    def amp_at(offset_from_pmax: int) -> int:
        amp_index = pmax_i + offset_from_pmax
        if 0 <= amp_index < len(amps) and global_peak_abs > 0:
            return int(max(0, min(255, round(255.0 * amps[amp_index] / global_peak_abs))))
        return 0

    duration_s = max(0.0, (end - start) / sample_rate)
    duration_units = int(max(0, min(65535, round(duration_s / IPI_UNIT_S))))
    if n_cycles >= 5:
        cycles_for_freq = min(n_cycles, 16)
        freq_hz = cycles_for_freq / duration_s if duration_s > 0 else 0.0
    elif ipis:
        local_ipis = ipis[max(0, pmax_i - 2) : min(len(ipis), pmax_i + 2)]
        freq_hz = 1.0 / float(np.mean(local_ipis)) if local_ipis else 0.0
    else:
        freq_hz = 0.0
    frequency_khz = max(0.0, freq_hz / 1000.0)

    if ipi_units:
        ipi_range = int(max(ipi_units) - min(ipi_units))
        mean_ipi = float(np.mean(ipi_units))
        bandwidth = ipi_range / mean_ipi if mean_ipi > 0 else 0.0
    else:
        ipi_range = 0
        bandwidth = 0.0

    amp_reversals = min(15, reversal_count(amps))
    time_units = int(round((center_sample / sample_rate) / TIME_UNIT_S))
    b0, b1, b2 = pack_24bit(time_units)
    byte4 = ((min(15, pmax_cycle) & 0x0F) << 4) | (min(15, ipi_range) & 0x0F)
    byte13 = ((amp_reversals & 0x0F) << 4) | ((duration_units >> 8) & 0x0F)
    last_ipi = int(ipi_units[-1]) if ipi_units else 0
    byte15 = min(127, last_ipi) | 0x00
    raw_bytes = bytes(
        [
            b0,
            b1,
            b2,
            n_cycles,
            byte4,
            ipi_at(-1),
            ipi_at(0),
            ipi_at(1),
            ipi_at(2),
            amp_at(-1),
            amp_at(0),
            amp_at(1),
            to_ipi_units((peak_times_s[0] - (start / sample_rate)) if peak_times_s else 0.0),
            byte13,
            duration_units & 0xFF,
            byte15,
        ]
    )

    return FpodStyleRecord(
        record_role=role,
        center_time_ms=1000.0 * center_sample / sample_rate,
        relative_delay_us=1e6 * (center_sample - direct_sample) / sample_rate,
        time_5us_units=time_units,
        n_cycles=n_cycles,
        loudest_cycle_number=pmax_cycle,
        ipi_range_250ns_units=ipi_range,
        ipi_pk_minus_1=ipi_at(-1),
        ipi_pk=ipi_at(0),
        ipi_pk_plus_1=ipi_at(1),
        ipi_pk_plus_2=ipi_at(2),
        amp_p_minus_1=amp_at(-1),
        amp_pmax=amp_at(0),
        amp_p_plus_1=amp_at(1),
        ipi_before_click_start=0,
        amplitude_reversals=amp_reversals,
        duration_250ns_units=duration_units,
        last_cycle_ipi_compressed=byte15,
        sonar_flag=0,
        frequency_khz=frequency_khz,
        bandwidth_index=float(bandwidth),
        cluster_size=cluster_size,
        cluster_lowest_frequency_khz=cluster_low_freq,
        cluster_highest_frequency_khz=cluster_high_freq,
        simulated_fp1_bytes_hex=raw_bytes.hex(" "),
        simulation_note=(
            "FP1-style simulated from WAV. True F-POD gain, thresholds, FPGA peak "
            "selection, and proprietary compression details are not recoverable from this snippet."
        ),
    )


def fpod_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "fpod_frequency",
        [
            (0.00, "#d7191c"),
            (0.18, "#fdae61"),
            (0.35, "#ffff33"),
            (0.52, "#1a9641"),
            (0.68, "#2c7bb6"),
            (0.84, "#313695"),
            (1.00, "#7b3294"),
        ],
    )


def plot_cluster(
    signal: np.ndarray,
    sample_rate: int,
    direct_sample: int,
    records: list[FpodStyleRecord],
    acf: np.ndarray,
    acf_values: list[float],
    output_path: Path,
) -> None:
    cmap = fpod_colormap()
    norm = mcolors.Normalize(vmin=MIN_FREQ_KHZ, vmax=MAX_FREQ_KHZ)
    time_ms = (np.arange(len(signal)) - direct_sample) / sample_rate * 1000.0
    record_times_ms = np.array([record.relative_delay_us / 1000.0 for record in records])
    record_freqs = np.array([record.frequency_khz for record in records])
    record_amps = np.array([record.amp_pmax for record in records])
    colors = [cmap(norm(freq)) for freq in record_freqs]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    fig.suptitle("F-POD-style multipath replicate cluster: PulseTrain_474 / Click_027", fontsize=13)

    axes[0].scatter(record_times_ms, record_freqs, c=colors, s=44, edgecolors="white", linewidths=0.5)
    axes[0].set_ylabel("Frequency (kHz)")
    axes[0].set_ylim(10, 170)
    axes[0].grid(True, alpha=0.25)
    for x, rec in zip(record_times_ms, records):
        axes[0].annotate(rec.record_role, (x, rec.frequency_khz), textcoords="offset points", xytext=(4, 5), fontsize=8)

    axes[1].vlines(record_times_ms, 0, record_amps, colors=colors, linewidth=2.0)
    axes[1].scatter(record_times_ms, record_amps, c=colors, s=34, edgecolors="white", linewidths=0.4)
    axes[1].set_ylabel("Amplitude byte")
    axes[1].set_ylim(0, max(260, float(np.max(record_amps) * 1.15) if len(record_amps) else 260))
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time_ms, signal, color="#303030", linewidth=0.7)
    for rec, color in zip(records, colors):
        x = rec.relative_delay_us / 1000.0
        axes[2].axvline(x, color=color, linewidth=1.4, linestyle="-" if rec.record_role == "Direct" else "--")
        axes[2].axvspan(x - MARK_HALF_WIDTH_S * 1000.0, x + MARK_HALF_WIDTH_S * 1000.0, color=color, alpha=0.13)
    axes[2].set_xlabel("Time relative to direct path (ms)")
    axes[2].set_ylabel("WAV amplitude")
    axes[2].grid(True, alpha=0.25)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.015)
    cbar.set_label("F-POD display frequency color (20 kHz red to 160 kHz violet)")

    lag_text = []
    for idx, value in enumerate(acf_values, start=1):
        lag_text.append(f"MP{idx} ACF={value:.3f}")
    axes[2].text(
        0.01,
        0.95,
        "; ".join(lag_text) if lag_text else "No multipath above threshold",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_outputs(records: list[FpodStyleRecord], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_fp1_style_records.csv"
    json_path = output_dir / f"{stem}_fp1_style_records.json"
    rows = [asdict(record) for record in records]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate FP1-style records and draw a F-POD-style multipath cluster.")
    parser.add_argument("--wav-path", default=str(DEFAULT_WAV_PATH))
    parser.add_argument("--anchor-path", default=str(DEFAULT_ANCHOR_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav_path)
    output_dir = Path(args.output_dir)

    progress(1, 5, "Read WAV")
    sample_rate, signal = read_wav_mono(wav_path)
    direct_sample = load_anchor_sample(Path(args.anchor_path), wav_path, sample_rate, signal)

    progress(2, 5, "Detect multipath lags")
    acf, mp_lags, acf_values = detect_multipath_lags(signal, sample_rate)
    centers = [direct_sample] + [direct_sample + lag for lag in mp_lags if direct_sample + lag < len(signal)]

    progress(3, 5, "Build FP1-style records")
    global_peak_abs = float(np.max(np.abs(signal))) if len(signal) else 1.0
    records: list[FpodStyleRecord] = []
    rough_records = []
    for idx, center in enumerate(centers):
        rough = estimate_record(
            signal=signal,
            sample_rate=sample_rate,
            center_sample=center,
            direct_sample=direct_sample,
            role="Direct" if idx == 0 else f"Multipath_{idx}",
            global_peak_abs=global_peak_abs,
            cluster_size=len(centers),
            cluster_low_freq=0.0,
            cluster_high_freq=0.0,
        )
        rough_records.append(rough)
    freqs = [record.frequency_khz for record in rough_records if record.frequency_khz > 0]
    low_freq = min(freqs) if freqs else 0.0
    high_freq = max(freqs) if freqs else 0.0
    for record in rough_records:
        record.cluster_lowest_frequency_khz = low_freq
        record.cluster_highest_frequency_khz = high_freq
        records.append(record)

    progress(4, 5, "Write tables and figure")
    csv_path, json_path = write_outputs(records, output_dir, wav_path.stem)
    figure_path = output_dir / f"{wav_path.stem}_fpod_style_multipath_cluster.png"
    plot_cluster(signal, sample_rate, direct_sample, records, acf, acf_values, figure_path)

    progress(5, 5, "Done")
    print(f"sample_rate: {sample_rate}")
    print(f"direct_sample: {direct_sample}")
    print(f"direct_time_ms: {1000.0 * direct_sample / sample_rate:.4f}")
    print(f"records: {len(records)}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    print(f"figure: {figure_path}")
    for record in records:
        print(
            f"{record.record_role}: delay_us={record.relative_delay_us:.2f}, "
            f"freq_khz={record.frequency_khz:.2f}, amp={record.amp_pmax}, "
            f"bytes={record.simulated_fp1_bytes_hex}"
        )


if __name__ == "__main__":
    main()
