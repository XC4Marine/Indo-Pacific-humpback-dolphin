#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Click waveform segmentation and DBSCAN clustering.

The script analyzes a single-channel click waveform by splitting it into
continuous short segments, extracting time/frequency/correlation features, and
running DBSCAN on standardized features. It keeps DBSCAN labels unchanged,
including the noise label -1.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    r"D:\Project_Github\Indo-Pacific-humpback-dolphin\12_ClickSegmentClustering\.matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
DEFAULT_OUTPUT_DIR = BASE_DIR / "12_ClickSegmentClustering" / "outputs"


@dataclass(frozen=True)
class ClickClusterConfig:
    """Main analysis parameters.

    segment_us:
        Target segment duration in microseconds. The actual sample count is
        round(segment_us * 1e-6 * sample_rate), so 50 us at 576000 Hz becomes
        29 samples.
    eps:
        DBSCAN radius in standardized feature space.
    min_samples:
        Minimum samples required by DBSCAN for a core point.
    """

    segment_us: float = 100.0
    eps: float = 0.8
    min_samples: int = 5


@dataclass
class ClickClusterResult:
    sample_rate: int
    signal: np.ndarray
    segment_samples: int
    table: pd.DataFrame
    standardized_features: np.ndarray
    pca_features: np.ndarray
    labels: np.ndarray
    reference_segment_index: int


class ClickSegmentClusterer:
    """Reusable click segment feature extractor and DBSCAN clusterer."""

    def __init__(self, config: ClickClusterConfig):
        self.config = config

    def analyze_wav(self, wav_path: Path) -> ClickClusterResult:
        sample_rate, signal = read_mono_wav(wav_path)
        return self.analyze_signal(signal=signal, sample_rate=sample_rate)

    def analyze_signal(self, signal: np.ndarray, sample_rate: int) -> ClickClusterResult:
        signal = np.asarray(signal, dtype=np.float64).reshape(-1)
        segment_samples = segment_sample_count(sample_rate, self.config.segment_us)
        segments = split_segments(signal, segment_samples)
        if len(segments) == 0:
            raise ValueError("Signal is shorter than one complete segment.")

        rows = extract_segment_features(segments, sample_rate, segment_samples)
        reference_index = int(np.argmax(rows["rms_energy"].to_numpy()))
        reference_segment = segments[reference_index]
        rows["norm_xcorr_peak"] = [
            normalized_cross_correlation_peak(segment, reference_segment)
            for segment in segments
        ]

        feature_columns = [
            "rms_energy",
            "center_time_s",
            "norm_xcorr_peak",
            "spectral_centroid_hz",
            "spectral_bandwidth_hz",
        ]
        features = rows[feature_columns].to_numpy(dtype=np.float64)
        standardized = StandardScaler().fit_transform(features)
        labels = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
        ).fit_predict(standardized)
        pca_features = PCA(n_components=2, random_state=0).fit_transform(standardized)

        rows["dbscan_label"] = labels
        return ClickClusterResult(
            sample_rate=sample_rate,
            signal=signal,
            segment_samples=segment_samples,
            table=rows,
            standardized_features=standardized,
            pca_features=pca_features,
            labels=labels,
            reference_segment_index=reference_index,
        )


def read_mono_wav(wav_path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV file and return sample rate plus a normalized mono signal."""

    sample_rate, signal = wavfile.read(str(wav_path))
    signal = np.asarray(signal)
    if signal.ndim == 2:
        if signal.shape[1] != 1:
            raise ValueError(f"Expected mono WAV, got {signal.shape[1]} channels: {wav_path}")
        signal = signal[:, 0]

    if np.issubdtype(signal.dtype, np.integer):
        max_abs = max(abs(np.iinfo(signal.dtype).min), np.iinfo(signal.dtype).max)
        signal = signal.astype(np.float64) / max_abs
    else:
        signal = signal.astype(np.float64)

    return int(sample_rate), signal


def segment_sample_count(sample_rate: int, segment_us: float) -> int:
    samples = int(round(segment_us * 1e-6 * sample_rate))
    if samples < 1:
        raise ValueError("segment_us is too small for the given sample rate.")
    return samples


def split_segments(signal: np.ndarray, segment_samples: int) -> np.ndarray:
    complete_count = len(signal) // segment_samples
    trimmed = signal[: complete_count * segment_samples]
    return trimmed.reshape(complete_count, segment_samples)


def extract_segment_features(
    segments: np.ndarray,
    sample_rate: int,
    segment_samples: int,
) -> pd.DataFrame:
    rows = []
    freqs = np.fft.rfftfreq(segment_samples, d=1.0 / sample_rate)
    for idx, segment in enumerate(segments):
        start_sample = idx * segment_samples
        end_sample = start_sample + segment_samples
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        center_time = (start_sample + 0.5 * segment_samples) / sample_rate
        rms = float(np.sqrt(np.mean(segment**2)))
        centroid, bandwidth = spectral_centroid_and_bandwidth(segment, freqs)
        rows.append(
            {
                "segment_index": idx,
                "start_time_s": start_time,
                "end_time_s": end_time,
                "center_time_s": center_time,
                "rms_energy": rms,
                "spectral_centroid_hz": centroid,
                "spectral_bandwidth_hz": bandwidth,
            }
        )
    return pd.DataFrame(rows)


def spectral_centroid_and_bandwidth(segment: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    windowed = segment * np.hanning(len(segment))
    magnitude = np.abs(np.fft.rfft(windowed))
    weight_sum = float(np.sum(magnitude))
    if weight_sum <= 0:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * magnitude) / weight_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / weight_sum))
    return centroid, bandwidth


def normalized_cross_correlation_peak(segment: np.ndarray, reference: np.ndarray) -> float:
    segment_zm = segment - np.mean(segment)
    reference_zm = reference - np.mean(reference)
    denominator = np.linalg.norm(segment_zm) * np.linalg.norm(reference_zm)
    if denominator <= 0:
        return 0.0
    corr = np.correlate(segment_zm, reference_zm, mode="full") / denominator
    return float(np.max(corr))


def make_demo_click(
    sample_rate: int = 576000,
    duration_s: float = 0.010,
    seed: int = 20260720,
) -> np.ndarray:
    """Generate a simulated click with delayed echoes and background noise."""

    rng = np.random.default_rng(seed)
    t = np.arange(int(round(duration_s * sample_rate))) / sample_rate
    signal = rng.normal(0.0, 0.012, size=len(t))
    signal += gaussian_click(t, center_s=0.0030, freq_hz=95000, width_s=18e-6, amplitude=1.0)
    signal += gaussian_click(t, center_s=0.00345, freq_hz=90000, width_s=22e-6, amplitude=0.42)
    signal += gaussian_click(t, center_s=0.00410, freq_hz=85000, width_s=25e-6, amplitude=0.25)
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs > 0 else signal


def gaussian_click(
    t: np.ndarray,
    center_s: float,
    freq_hz: float,
    width_s: float,
    amplitude: float,
) -> np.ndarray:
    envelope = np.exp(-0.5 * ((t - center_s) / width_s) ** 2)
    carrier = np.sin(2.0 * np.pi * freq_hz * (t - center_s))
    return amplitude * envelope * carrier


def save_outputs(
    result: ClickClusterResult,
    output_dir: Path,
    stem: str,
    config: ClickClusterConfig,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}_segments_dbscan.csv"
    figure_path = output_dir / f"{stem}_segments_dbscan.png"
    output_columns = [
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
    result.table.loc[:, output_columns].to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_result(result, figure_path, config, title=stem)
    return csv_path, figure_path


def plot_result(
    result: ClickClusterResult,
    figure_path: Path,
    config: ClickClusterConfig,
    title: str,
) -> None:
    table = result.table
    time_ms = np.arange(len(result.signal)) / result.sample_rate * 1000.0
    centers_ms = table["center_time_s"].to_numpy() * 1000.0
    labels = result.labels
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
    color_by_label = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = [color_by_label[label] for label in labels]

    fig, axes = plt.subplots(4, 1, figsize=(13, 15))
    fig.suptitle(
        (
            f"{title} | segment_duration_us={config.segment_us:g}, "
            f"segment_samples={result.segment_samples}, eps={config.eps:g}, "
            f"min_samples={config.min_samples}, sample_rate={result.sample_rate} Hz"
        ),
        fontsize=12,
        fontweight="bold",
    )

    axes[0].plot(time_ms, result.signal, color="#2f2f2f", linewidth=0.7)
    axes[0].set_title("Original 10 ms waveform")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(centers_ms, table["rms_energy"], color="#0072B2", linewidth=1.0)
    axes[1].scatter(centers_ms, table["rms_energy"], c=colors, s=18)
    axes[1].axvline(
        centers_ms[result.reference_segment_index],
        color="#D55E00",
        linestyle="--",
        linewidth=1.0,
        label="Max RMS reference segment",
    )
    axes[1].set_title("RMS energy by 50 us segment")
    axes[1].set_xlabel("Segment center time (ms)")
    axes[1].set_ylabel("RMS energy")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(centers_ms, labels, c=colors, s=28, edgecolor="black", linewidth=0.2)
    axes[2].set_title("Raw DBSCAN labels on time axis")
    axes[2].set_xlabel("Segment center time (ms)")
    axes[2].set_ylabel("DBSCAN label")
    axes[2].grid(True, alpha=0.3)

    axes[3].scatter(
        result.pca_features[:, 0],
        result.pca_features[:, 1],
        c=colors,
        s=34,
        edgecolor="black",
        linewidth=0.25,
    )
    axes[3].set_title("PCA projection of standardized features")
    axes[3].set_xlabel("PC1")
    axes[3].set_ylabel("PC2")
    axes[3].grid(True, alpha=0.3)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_by_label[label],
            markeredgecolor="black",
            markersize=7,
            label=f"Label {label}",
        )
        for label in unique_labels
    ]
    axes[2].legend(handles=handles, loc="best", fontsize=8)
    axes[3].legend(handles=handles, loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def run_real_analysis(args: argparse.Namespace, clusterer: ClickSegmentClusterer) -> None:
    wav_path = Path(args.wav_path)
    result = clusterer.analyze_wav(wav_path)
    csv_path, figure_path = save_outputs(
        result=result,
        output_dir=Path(args.output_dir),
        stem=wav_path.stem,
        config=clusterer.config,
    )
    print_summary("real", result, csv_path, figure_path)


def run_demo_analysis(args: argparse.Namespace, clusterer: ClickSegmentClusterer) -> None:
    sample_rate = 576000
    signal = make_demo_click(sample_rate=sample_rate)
    result = clusterer.analyze_signal(signal=signal, sample_rate=sample_rate)
    csv_path, figure_path = save_outputs(
        result=result,
        output_dir=Path(args.output_dir),
        stem="demo_click",
        config=clusterer.config,
    )
    print_summary("demo", result, csv_path, figure_path)


def print_summary(kind: str, result: ClickClusterResult, csv_path: Path, figure_path: Path) -> None:
    label_counts = result.table["dbscan_label"].value_counts().sort_index()
    print(f"[{kind}] sample_rate: {result.sample_rate}")
    print(f"[{kind}] segment_samples: {result.segment_samples}")
    print(f"[{kind}] segment_count: {len(result.table)}")
    print(f"[{kind}] reference_segment_index: {result.reference_segment_index}")
    print(f"[{kind}] label_counts:")
    for label, count in label_counts.items():
        print(f"  label {label}: {count}")
    print(f"[{kind}] csv: {csv_path}")
    print(f"[{kind}] figure: {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a click waveform into short windows and cluster with DBSCAN."
    )
    parser.add_argument(
        "--wav-path",
        default=str(DEFAULT_WAV_PATH),
        help="Input single-channel WAV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.8,
        help="DBSCAN eps in standardized feature space.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--segment-us",
        type=float,
        default=50.0,
        help="Segment duration in microseconds before sample-count rounding.",
    )
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Also run the simulated click, delayed echo, and noise example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ClickClusterConfig(
        segment_us=args.segment_us,
        eps=args.eps,
        min_samples=args.min_samples,
    )
    clusterer = ClickSegmentClusterer(config)
    run_real_analysis(args, clusterer)
    if args.run_demo:
        run_demo_analysis(args, clusterer)


if __name__ == "__main__":
    main()
