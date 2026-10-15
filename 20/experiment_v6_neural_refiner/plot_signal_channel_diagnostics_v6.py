#!/usr/bin/env python3
"""Render transparent time/frequency diagnostics for a median-error hold-out case.

The v6 network natively predicts 35-point channel-amplitude corrections, rather
than a time-domain channel or a new source.  To visualise its waveform effect,
this script applies that amplitude correction to the BCSC channel spectrum while
preserving the BCSC phase, then inverse-transforms it.  These waveforms and the
derived click reconstructions are explicitly diagnostic, not native network
outputs.  Truth is accessed only after blind v6 inference has finished.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "experiment_v5"
V4 = HERE.parent / "experiment_v4"
sys.path[:0] = [str(V5), str(V4)]
from protocol_v5 import DATA_ROOT  # noqa: E402
from bcsc_v4 import FS, bandpass, canonicalise  # noqa: E402
from evaluate_bcsc_v4 import FREQUENCIES_HZ, align_source, transfer_from_arrivals  # noqa: E402
from experiment_v6 import FREQ, OUT  # noqa: E402

FIGURES = OUT / "figures"
BLUE, ORANGE, BLACK, GREY = "#0072B2", "#D55E00", "#222222", "#8C8C8C"
N_FFT = 16_384


def configure() -> None:
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                         "svg.fonttype": "none", "figure.dpi": 160, "savefig.dpi": 300})


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def representative() -> tuple[str, int]:
    metrics = pd.read_csv(OUT / "event_metrics.csv")
    rows = metrics[(metrics.condition == "clean") & (metrics.method == "neural_ensemble")].copy()
    target = rows.channel_amplitude_rmse_db.median()
    item = rows.assign(distance=(rows.channel_amplitude_rmse_db - target).abs()).sort_values(["distance", "train_id", "event_index"]).iloc[0]
    return str(item.train_id), int(item.event_index) - 1


def relative_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.fft.rfftfreq(N_FFT, 1 / FS)
    values = 20 * np.log10(np.abs(np.fft.rfft(signal, N_FFT)) + 1e-15)
    region = (frequency >= 20_000) & (frequency <= 190_000)
    return frequency, values - np.max(values[region])


def source_truth(directory: Path) -> np.ndarray:
    _, values = wavfile.read(directory / "original_click.wav")
    source, _ = canonicalise(bandpass(values.astype(float)), np.zeros((1, 2304)))
    return source


def truth_shapes(directory: Path) -> np.ndarray:
    with np.load(directory / "event_channels.npz", allow_pickle=True) as item:
        arrivals, first = item["event_synthesis_arrivals"], float(item["first_arrival_s"])
    result = []
    for event in arrivals:
        response = transfer_from_arrivals(event, first)
        amplitude = 20 * np.log10(np.abs(response) + 1e-15)
        result.append(amplitude - np.median(amplitude))
    return np.asarray(result, dtype=float)


def diagnostic_channel(channel: np.ndarray, base_shape: np.ndarray, corrected_shape: np.ndarray) -> np.ndarray:
    """Apply the learned 35-point magnitude correction without changing phase."""
    spectrum = np.fft.rfft(channel, N_FFT); frequency = np.fft.rfftfreq(N_FFT, 1 / FS)
    delta_db = np.interp(frequency, FREQ, corrected_shape - base_shape, left=0.0, right=0.0)
    corrected = np.fft.irfft(spectrum * 10 ** (delta_db / 20), N_FFT)
    return corrected[:len(channel)]


def normalise(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return values / (np.max(np.abs(reference)) + 1e-15)


def plot_click_diagnostic(train_id: str, event: int, artifact: np.lib.npyio.NpzFile, neural: np.lib.npyio.NpzFile) -> None:
    directory = DATA_ROOT / train_id
    source = np.asarray(artifact["source"], dtype=float); source_ref = source_truth(directory); source_aligned, _ = align_source(source, source_ref)
    h_bcsc = np.asarray(artifact["channels"][event], dtype=float)
    h_neural = diagnostic_channel(h_bcsc, neural["baseline_shape"][event], neural["corrected_shape"][event])
    y_bcsc = np.asarray(artifact["reconstructions"][event], dtype=float)
    y_neural = fftconvolve(source, h_neural, mode="full")[:len(y_bcsc)]
    raw = bandpass(np.load(directory / "clicktrain_raw.npy", allow_pickle=False).astype(float))
    start = int(artifact["detected_window_starts"][event]); observed = raw[start:start + len(y_bcsc)]
    if len(observed) < len(y_bcsc): observed = np.pad(observed, (0, len(y_bcsc) - len(observed)))
    time = np.arange(len(y_bcsc)) / FS * 1e3; source_time = np.arange(len(source_ref)) / FS * 1e3
    nrmse = lambda estimate: np.linalg.norm(estimate - observed) / (np.linalg.norm(observed) + 1e-15)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.0))
    ax = axes[0, 0]
    ax.plot(time, normalise(observed, observed), color=GREY, lw=1.05, label="Observed click window")
    ax.plot(time, normalise(y_bcsc, observed), color=BLUE, lw=1.25, label=f"BCSC reconstruction (NRMSE {nrmse(y_bcsc):.3f})")
    ax.plot(time, normalise(y_neural, observed), color=ORANGE, lw=1.25, label=f"Neural diagnostic reconstruction (NRMSE {nrmse(y_neural):.3f})")
    ax.set(title="(a) Received click: time domain", xlabel="Time from detected window start (ms)", ylabel="Amplitude / observed peak")
    ax.legend(loc="upper right", frameon=False, fontsize=8); ax.grid(color="#E6E6E6", lw=.8)

    ax = axes[0, 1]
    for values, colour, label in ((observed, GREY, "Observed click window"), (y_bcsc, BLUE, "BCSC reconstruction"), (y_neural, ORANGE, "Neural diagnostic reconstruction")):
        frequency, spectrum = relative_spectrum(values); band = (frequency >= 20_000) & (frequency <= 190_000)
        ax.plot(frequency[band] / 1000, spectrum[band], color=colour, lw=1.3, label=label)
    ax.set(title="(b) Received click: relative spectrum", xlabel="Frequency (kHz)", ylabel="Magnitude (dB, peak-normalised)", ylim=(-60, 3))
    ax.legend(loc="lower left", frameon=False, fontsize=8); ax.grid(color="#E6E6E6", lw=.8)

    ax = axes[1, 0]
    ax.plot(source_time, source_ref, color=BLACK, lw=2.0, label="Original source click")
    ax.plot(source_time, source_aligned, color=BLUE, lw=1.45, label="BCSC recovered source")
    ax.plot(source_time, source_aligned, color=ORANGE, lw=1.2, ls="--", label="Neural source (= BCSC source)")
    ax.set(title="(c) Shared source click: time domain", xlabel="Time (ms)", ylabel="Normalised amplitude")
    ax.legend(loc="upper right", frameon=False, fontsize=8); ax.grid(color="#E6E6E6", lw=.8)

    ax = axes[1, 1]
    for values, colour, style, label in ((source_ref, BLACK, "-", "Original source click"), (source_aligned, BLUE, "-", "BCSC recovered source"), (source_aligned, ORANGE, "--", "Neural source (= BCSC source)")):
        frequency, spectrum = relative_spectrum(values); band = (frequency >= 20_000) & (frequency <= 190_000)
        ax.plot(frequency[band] / 1000, spectrum[band], color=colour, ls=style, lw=1.5, label=label)
    ax.set(title="(d) Shared source click: relative spectrum", xlabel="Frequency (kHz)", ylabel="Magnitude (dB, peak-normalised)", ylim=(-70, 3))
    ax.legend(loc="lower left", frameon=False, fontsize=8); ax.grid(color="#E6E6E6", lw=.8)
    fig.suptitle(f"Typical hold-out sample — click reconstruction comparison (event {event + 1})", fontsize=14, fontweight="bold", y=.99)
    fig.text(.5, .005, "The neural model refines only 35-point channel magnitude. Orange click reconstruction is a phase-preserving diagnostic derived from that refinement; it is not a separately trained waveform output.", ha="center", fontsize=8.3, color="#444444")
    fig.tight_layout(rect=(0, .04, 1, .96))
    save(fig, "v6_click_recovery_comparison")


def plot_channel_diagnostic(train_id: str, artifact: np.lib.npyio.NpzFile, neural: np.lib.npyio.NpzFile) -> None:
    source = np.asarray(artifact["source"], dtype=float); channels = np.asarray(artifact["channels"], dtype=float)
    _ = source  # source is intentionally shared by both methods.
    predicted = np.stack([diagnostic_channel(channels[i], neural["baseline_shape"][i], neural["corrected_shape"][i]) for i in range(10)])
    truth = truth_shapes(DATA_ROOT / train_id); time = np.arange(channels.shape[1]) / FS * 1e3
    fig, axes = plt.subplots(4, 5, figsize=(17.2, 11.5), sharex="col")
    for event in range(10):
        row, col = divmod(event, 5); time_ax = axes[row, col]; freq_ax = axes[row + 2, col]
        scale = max(np.max(np.abs(channels[event])), np.max(np.abs(predicted[event])), 1e-15)
        time_ax.plot(time, channels[event] / scale, color=BLUE, lw=1.05, label="BCSC channel")
        time_ax.plot(time, predicted[event] / scale, color=ORANGE, lw=1.05, label="Neural diagnostic channel")
        time_ax.axhline(0, color="#CFCFCF", lw=.7)
        time_ax.set(title=f"Event {event + 1}", ylabel="Relative amplitude")
        time_ax.grid(color="#EEEEEE", lw=.65)
        if row == 0: time_ax.legend(loc="upper right", fontsize=6.8, frameon=False)
        freq_ax.plot(FREQUENCIES_HZ / 1000, truth[event], color=BLACK, lw=1.8, label="Truth")
        freq_ax.plot(FREQ / 1000, neural["baseline_shape"][event], color=BLUE, marker="o", ms=2.2, lw=1.0, label="BCSC")
        freq_ax.plot(FREQ / 1000, neural["corrected_shape"][event], color=ORANGE, marker="D", ms=2.1, lw=1.1, label="Neural")
        freq_ax.axhline(0, color="#CFCFCF", lw=.7, ls=":")
        freq_ax.set(xlabel="Frequency (kHz)", ylabel="Relative amplitude (dB)", ylim=(-28, 28))
        freq_ax.grid(color="#EEEEEE", lw=.65)
        if event == 0: freq_ax.legend(loc="lower left", fontsize=6.8, frameon=False)
    fig.suptitle("Typical hold-out sample — all 10 event-channel recoveries", fontsize=14, fontweight="bold", y=.995)
    fig.text(.5, .012, "Top two rows: time-domain BCSC channels and phase-preserving neural diagnostic channels. Bottom two rows: the 35-point relative magnitude responses used for evaluation (black truth, blue BCSC, orange neural).", ha="center", fontsize=8.4, color="#444444")
    fig.tight_layout(rect=(0, .035, 1, .97))
    save(fig, "v6_channel_recovery_comparison")


def main() -> None:
    configure(); train_id, event = representative()
    artifact = np.load(V5 / "outputs" / "predictions" / "clean" / "v4_independent" / f"{train_id}.npz")
    neural = np.load(OUT / "blind_predictions" / "clean" / f"{train_id}.npz")
    plot_click_diagnostic(train_id, event, artifact, neural); plot_channel_diagnostic(train_id, artifact, neural)
    print(f"Generated diagnostic figures for {train_id}, event {event + 1}: {FIGURES}")


if __name__ == "__main__":
    main()
