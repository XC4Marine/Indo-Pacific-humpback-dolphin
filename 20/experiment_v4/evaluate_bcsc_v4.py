#!/usr/bin/env python3
"""Truth-only evaluator for Experiment 20 v4. Never import this from extraction."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile

from bcsc_v4 import FS, GROUP_BLOCK_SAMPLES, bandpass, canonicalise, measure_3db_bandwidth, path_candidates


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
OUTPUT_ROOT = ROOT / "outputs"
EVALUATION_ROOT = ROOT / "evaluation"
FREQUENCIES_HZ = np.arange(20_000, 190_001, 5_000, dtype=float)
METHODS = ("single_click", "independent", "shared_support")
METHOD_LABELS = {"single_click": "single click", "independent": "independent", "shared_support": "shared support"}
METHOD_COLOURS = {"single_click": "#E69F00", "independent": "#0072B2", "shared_support": "#009E73"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def normalise_truth_source(path: Path) -> np.ndarray:
    _, source = wavfile.read(path)
    source = bandpass(source.astype(np.float64))
    # canonicalise needs a channel matrix only to carry the inverse source shift.
    source, _ = canonicalise(source, np.zeros((1, 2304)))
    return source


def align_source(estimate: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    correlation = np.correlate(estimate, truth, mode="full")
    shift = int(np.argmax(np.abs(correlation)) - (len(truth) - 1))
    shifted = np.roll(estimate, -shift)
    if shift > 0:
        shifted[-shift:] = 0
    elif shift < 0:
        shifted[:-shift] = 0
    scale = float(np.dot(truth, shifted) / (np.dot(shifted, shifted) + 1e-18))
    return shifted * scale, float(shift)


def source_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    aligned, shift = align_source(estimate, truth)
    corr = float(np.dot(aligned, truth) / (np.linalg.norm(aligned) * np.linalg.norm(truth) + 1e-18))
    nrmse = float(np.linalg.norm(aligned - truth) / (np.linalg.norm(truth) + 1e-18))
    n = 16384
    e = np.abs(np.fft.rfft(aligned, n=n))
    t = np.abs(np.fft.rfft(truth, n=n))
    f = np.fft.rfftfreq(n, 1 / FS)
    mask = (f >= 20_000) & (f <= 190_000)
    spectral_mae = float(np.mean(np.abs(20 * np.log10(e[mask] + 1e-12) - 20 * np.log10(t[mask] + 1e-12))))
    ebw, epeak, _, _ = measure_3db_bandwidth(estimate)
    tbw, tpeak, _, _ = measure_3db_bandwidth(truth)
    return {"source_correlation": corr, "source_si_nrmse": nrmse, "source_shift_samples": shift, "source_log_spectral_mae_db": spectral_mae, "source_bandwidth_error_hz": float(ebw - tbw), "source_peak_frequency_error_hz": float(epeak - tpeak)}


def transfer_from_arrivals(arrivals: np.ndarray, first_arrival: float) -> np.ndarray:
    frame = pd.DataFrame.from_records(arrivals)
    response = []
    for frequency in FREQUENCIES_HZ:
        selected = frame[np.isclose(frame["frequency_hz"], frequency)]
        response.append(np.sum(selected["complex_gain"].to_numpy() * np.exp(-2j * np.pi * frequency * (selected["delay_s"].to_numpy() - first_arrival))))
    return np.asarray(response, dtype=np.complex128)


def transfer_metrics(channel: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    spectrum = np.fft.rfft(channel, n=16384)
    f = np.fft.rfftfreq(16384, 1 / FS)
    estimate = np.interp(FREQUENCIES_HZ, f, spectrum.real) + 1j * np.interp(FREQUENCIES_HZ, f, spectrum.imag)
    log_ratio = 20 * np.log10(np.abs(estimate) + 1e-15) - 20 * np.log10(np.abs(truth) + 1e-15)
    amplitude_rmse = float(np.sqrt(np.mean((log_ratio - np.median(log_ratio)) ** 2)))
    phase_difference = np.unwrap(np.angle(estimate) - np.angle(truth))
    slope, intercept = np.polyfit(FREQUENCIES_HZ, phase_difference, 1)
    residual = np.angle(np.exp(1j * (phase_difference - (slope * FREQUENCIES_HZ + intercept))))
    phase_mae = float(np.mean(np.abs(residual)))
    return {"channel_amplitude_rmse_db": amplitude_rmse, "channel_phase_circular_mae_rad": phase_mae, "channel_alignment_delay_us": float(-slope / (2 * np.pi) * 1e6)}


def amplitude_difference(first: np.ndarray, second: np.ndarray) -> float:
    """Scale-invariant magnitude difference between two physical responses."""
    difference = 20 * np.log10(np.abs(first) + 1e-15) - 20 * np.log10(np.abs(second) + 1e-15)
    return float(np.sqrt(np.mean((difference - np.median(difference)) ** 2)))


def relative_spectrum_db(signal: np.ndarray, *, n_fft: int = 16_384) -> tuple[np.ndarray, np.ndarray]:
    """Return a peak-normalised spectrum, so blind scale ambiguity is not plotted as error."""
    frequencies = np.fft.rfftfreq(n_fft, 1 / FS)
    magnitude_db = 20 * np.log10(np.abs(np.fft.rfft(signal, n=n_fft)) + 1e-15)
    analysis = (frequencies >= 20_000) & (frequencies <= 190_000)
    return frequencies, magnitude_db - np.max(magnitude_db[analysis])


def relative_channel_response_db(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return channel-response shape after removing the unidentifiable global gain."""
    frequencies, magnitude_db = relative_spectrum_db(channel)
    analysis = (frequencies >= 20_000) & (frequencies <= 190_000)
    return frequencies, magnitude_db - np.median(magnitude_db[analysis])


def relative_transfer_response_db(response: np.ndarray) -> np.ndarray:
    """Apply the same gain convention to a truth transfer response on the 35-band grid."""
    magnitude_db = 20 * np.log10(np.abs(response) + 1e-15)
    return magnitude_db - np.median(magnitude_db)


def truth_paths(arrivals: np.ndarray, first_arrival: float) -> list[dict[str, float | str]]:
    frame = pd.DataFrame.from_records(arrivals)
    result = []
    for (top, bottom), group in frame.groupby(["top_bounces", "bottom_bounces"]):
        delay = float(np.median(group["delay_s"]) - first_arrival)
        gain = float(np.median(group["effective_gain"]))
        name = "direct" if int(top) == 0 and int(bottom) == 0 else f"{int(top)}_{int(bottom)}"
        result.append({"name": name, "delay_ms": delay * 1e3, "gain": gain})
    direct = next(path for path in result if path["name"] == "direct")
    for path in result:
        path["relative_delay_ms"] = float(path["delay_ms"] - direct["delay_ms"])
        path["relative_amplitude"] = float(path["gain"] / (direct["gain"] + 1e-18))
    return result


def path_metrics(channel: np.ndarray, arrivals: np.ndarray, first_arrival: float) -> dict[str, float]:
    estimated = path_candidates(channel)
    true = truth_paths(arrivals, first_arrival)
    tolerance = 0.03
    direct_true = next(row for row in true if row["name"] == "direct")
    direct_estimated = min(estimated, key=lambda row: abs(row["delay_ms"])) if estimated else None
    matched = 0
    errors = []
    amplitude_errors = []
    for target in true:
        candidates = [row for row in estimated if abs(row["delay_ms"] - target["relative_delay_ms"]) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda row: abs(row["delay_ms"] - target["relative_delay_ms"]))
            matched += 1
            errors.append(abs(best["delay_ms"] - target["relative_delay_ms"]))
            amplitude_errors.append(abs(20 * math.log10(best["relative_amplitude"] + 1e-12) - 20 * math.log10(target["relative_amplitude"] + 1e-12)))
    precision = matched / len(estimated) if estimated else 0.0
    recall = matched / len(true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    direct_mae = abs(direct_estimated["delay_ms"] - direct_true["relative_delay_ms"]) if direct_estimated else float("nan")
    return {"path_precision": precision, "path_recall": recall, "path_f1": f1, "path_delay_mae_ms": float(np.mean(errors)) if errors else float("nan"), "path_relative_amplitude_mae_db": float(np.mean(amplitude_errors)) if amplitude_errors else float("nan"), "direct_delay_mae_ms": direct_mae, "estimated_path_count": len(estimated), "true_path_count": len(true)}


def evaluate_train(method: str, artifact: Path) -> list[dict[str, Any]]:
    train_id = artifact.stem
    train_dir = DATA_ROOT / train_id
    estimate = np.load(artifact)
    source_truth = normalise_truth_source(train_dir / "original_click.wav")
    source = np.asarray(estimate["source"])
    channels = np.asarray(estimate["channels"])
    reconstruction = np.asarray(estimate["reconstructions"])
    with np.load(train_dir / "event_channels.npz", allow_pickle=True) as truth:
        arrivals = truth["event_synthesis_arrivals"]
        event_rx = truth["event_rx_raw"]
        lengths = truth["event_rx_lengths"]
        first_arrival = float(truth["first_arrival_s"])
    with np.load(train_dir / "original_channel.npz", allow_pickle=True) as base_channel:
        base_truth = transfer_from_arrivals(base_channel["synthesis_arrivals"], float(base_channel["first_arrival_s"]))
    clicktrain = bandpass(np.load(train_dir / "clicktrain_raw.npy", allow_pickle=False))
    starts = np.asarray(estimate["detected_window_starts"], dtype=int)
    rows = []
    for event in range(10):
        target = transfer_from_arrivals(arrivals[event], first_arrival)
        event_source = source[event] if source.ndim == 2 else source
        row = {"train_id": train_id, "method": method, "event_index": event + 1, **source_metrics(event_source, source_truth), **transfer_metrics(channels[event], target), **path_metrics(channels[event], arrivals[event], first_arrival)}
        row["base_channel_amplitude_rmse_db"] = transfer_metrics(channels[event], base_truth)["channel_amplitude_rmse_db"]
        row["event_to_base_amplitude_difference_db"] = amplitude_difference(target, base_truth)
        start = int(starts[event]); y = clicktrain[start:start + len(reconstruction[event])]
        if len(y) < len(reconstruction[event]):
            y = np.pad(y, (0, len(reconstruction[event]) - len(y)))
        fitted = reconstruction[event]
        row["event_reconstruction_nrmse"] = float(np.linalg.norm(fitted - y) / (np.linalg.norm(y) + 1e-18))
        row["event_explained_variance"] = float(1 - np.sum((fitted - y) ** 2) / (np.sum(y ** 2) + 1e-18))
        rows.append(row)
    return rows


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    estimates = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(1000)]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metrics = [column for column in frame if column not in {"train_id", "method", "event_index"}]
    summary = []
    for method, group in frame.groupby("method"):
        train_group = group.groupby("train_id", as_index=False)[metrics].median(numeric_only=True)
        for metric in metrics:
            values = train_group[metric].to_numpy(dtype=float)
            low, high = bootstrap_interval(values, abs(hash((method, metric))) % (2**32))
            summary.append({"method": method, "metric": metric, "n_clicktrains": int(np.isfinite(values).sum()), "median": float(np.nanmedian(values)), "q25": float(np.nanquantile(values, .25)), "q75": float(np.nanquantile(values, .75)), "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
    return summary


def make_figures(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    figures = EVALUATION_ROOT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, metric, title in zip(axes, ("source_correlation", "channel_amplitude_rmse_db", "path_f1"), ("Source correlation", "Channel magnitude RMSE (dB)", "Path F1")):
        methods = METHODS
        data = [frame.loc[frame.method == method, metric].dropna() for method in methods]
        axis.boxplot(data, tick_labels=["single click", "independent", "shared support"])
        axis.set_title(title); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "method_comparison.png", dpi=170); plt.close(fig)

    paired = frame.groupby(["train_id", "method"], as_index=False).median(numeric_only=True).pivot(index="train_id", columns="method", values="channel_amplitude_rmse_db")
    choices = [paired["independent"].sub(paired["independent"].median()).abs().idxmin(), (paired["independent"] - paired["single_click"]).idxmax(), (paired["independent"] - paired["shared_support"]).idxmax()]
    labels = ("typical", "single_click_degradation", "shared_support_gain")
    for label, train_id in zip(labels, choices):
        directory = DATA_ROOT / train_id
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        truth = normalise_truth_source(directory / "original_click.wav")
        axes[0, 0].plot(np.arange(len(truth)) / FS * 1e3, truth, label="original click", lw=2, color="#222222")
        truth_frequency, truth_spectrum = relative_spectrum_db(truth)
        axes[0, 1].plot(truth_frequency / 1e3, truth_spectrum, label="original click", lw=2, color="#222222")
        for method in METHODS:
            source = np.load(OUTPUT_ROOT / method / f"{train_id}.npz")["source"]
            source = source[0] if source.ndim == 2 else source
            aligned, _ = align_source(source, truth)
            axes[0, 0].plot(np.arange(len(aligned)) / FS * 1e3, aligned, label=METHOD_LABELS[method], color=METHOD_COLOURS[method], alpha=.9)
            frequency, spectrum = relative_spectrum_db(aligned)
            axes[0, 1].plot(frequency / 1e3, spectrum, label=METHOD_LABELS[method], color=METHOD_COLOURS[method], alpha=.9)
        axes[0, 0].set(title="Original / extracted click", xlabel="ms", ylabel="normalised amplitude")
        axes[0, 1].set(xlim=(20, 190), ylim=(-60, 2), title="Click magnitude spectrum", xlabel="kHz", ylabel="relative magnitude (dB)")
        for axis in axes[0, :2]:
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)

        with np.load(directory / "event_channels.npz", allow_pickle=True) as event_channel:
            event_truth = transfer_from_arrivals(event_channel["event_synthesis_arrivals"][0], float(event_channel["first_arrival_s"]))
        with np.load(directory / "original_channel.npz", allow_pickle=True) as original_channel:
            original_truth = transfer_from_arrivals(original_channel["synthesis_arrivals"], float(original_channel["first_arrival_s"]))
        for method in METHODS:
            channel = np.load(OUTPUT_ROOT / method / f"{train_id}.npz")["channels"][0]
            axes[0, 2].plot(np.arange(len(channel)) / FS * 1e3, channel, label=METHOD_LABELS[method], color=METHOD_COLOURS[method])
            frequency, response = relative_channel_response_db(channel)
            axes[1, 0].plot(frequency / 1e3, response, label=METHOD_LABELS[method], color=METHOD_COLOURS[method])
        axes[0, 2].set(title="Estimated event-1 channel impulse", xlabel="ms", ylabel="relative amplitude")
        axes[0, 2].grid(alpha=.25); axes[0, 2].legend(fontsize=8)
        axes[1, 0].plot(FREQUENCIES_HZ / 1e3, relative_transfer_response_db(event_truth), label="event-1 channel truth", color="#A6A6A6", ls=":", lw=2)
        axes[1, 0].plot(FREQUENCIES_HZ / 1e3, relative_transfer_response_db(original_truth), label="original channel (base)", color="#222222", ls="--", lw=2)
        axes[1, 0].set(xlim=(20, 190), title="Channel response: estimate and truth", xlabel="kHz", ylabel="relative magnitude (dB)")
        axes[1, 0].grid(alpha=.25); axes[1, 0].legend(fontsize=8)
        subset = frame[(frame.train_id == train_id) & (frame.event_index == 1)]
        axes[1, 1].axis("off")
        table_rows = [[METHOD_LABELS[row.method], f"{row.source_correlation:.3f}", f"{row.channel_amplitude_rmse_db:.2f}", f"{row.path_f1:.3f}"] for row in subset.itertuples()]
        metric_table = axes[1, 1].table(cellText=table_rows, colLabels=["method", "source r", "channel RMSE", "path F1"], cellLoc="center", colLoc="center", loc="center")
        metric_table.auto_set_font_size(False); metric_table.set_fontsize(8); metric_table.scale(1.05, 1.4)
        axes[1, 1].set_title("Event-1 evaluation", pad=10)
        axes[1, 2].axis("off")
        axes[1, 2].text(.02, .95, "Response convention\n\nRelative magnitude removes the analysis-band\nmedian gain; blind deconvolution cannot recover\nabsolute channel gain.\n\nBlack dashed: original channel base\nGray dotted: event-1 channel truth", va="top", fontsize=10, linespacing=1.5)
        fig.suptitle(f"{label}: {train_id}"); fig.tight_layout(); fig.savefig(figures / f"case_{label}_{train_id}.png", dpi=170); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--limit", type=int, default=None); args = parser.parse_args()
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ("single_click", "independent", "shared_support"):
        artifacts = sorted((OUTPUT_ROOT / method).glob("*.npz"))
        if args.limit is not None: artifacts = artifacts[:args.limit]
        for index, artifact in enumerate(artifacts, 1):
            rows.extend(evaluate_train(method, artifact))
            print(f"[总体进度 3/3：真值验证] {method} {index}/{len(artifacts)}", flush=True)
    write_csv(EVALUATION_ROOT / "event_metrics.csv", rows)
    summary = summarise(rows); write_csv(EVALUATION_ROOT / "summary_metrics.csv", summary)
    atomic_json(EVALUATION_ROOT / "summary_metrics.json", summary)
    make_figures(rows)
    print(f"验证完成：{EVALUATION_ROOT}", flush=True)


if __name__ == "__main__":
    main()
