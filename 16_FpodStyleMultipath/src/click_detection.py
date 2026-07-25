from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt


@dataclass(frozen=True)
class ClickDetection:
    click_id: int
    sample_index: int
    time_s: float
    peak_amplitude: float
    envelope_value: float


@dataclass(frozen=True)
class DetectionSettings:
    lowcut_hz: float = 20_000.0
    highcut_hz: float = 220_000.0
    order: int = 4
    prominence_percentiles: tuple[float, ...] = (60.0, 70.0, 80.0, 90.0)
    distance_us_options: tuple[float, ...] = (20.0, 30.0, 50.0)
    min_clicks: int = 10
    max_clicks: int = 500
    preferred_clicks: int = 13
    event_merge_us: float = 22_000.0


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    signal, sample_rate = sf.read(str(path), always_2d=False)
    if signal.ndim > 1:
        signal = signal[:, 0]
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return int(sample_rate), signal


def bandpass_filter(signal: np.ndarray, sample_rate: int, settings: DetectionSettings) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    highcut = min(settings.highcut_hz, nyquist * 0.95)
    lowcut = max(settings.lowcut_hz, 1.0)
    if lowcut >= highcut:
        raise ValueError(
            f"Invalid bandpass for fs={sample_rate}: {lowcut:.1f}-{highcut:.1f} Hz"
        )
    sos = butter(
        settings.order,
        [lowcut / nyquist, highcut / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def hilbert_envelope(filtered: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(filtered))


def detect_clicks(
    filtered: np.ndarray,
    sample_rate: int,
    settings: DetectionSettings | None = None,
) -> tuple[list[ClickDetection], np.ndarray, dict[str, float]]:
    settings = settings or DetectionSettings()
    envelope = hilbert_envelope(filtered)
    best: tuple[float, list[int], dict[str, float]] | None = None
    merge_samples = max(1, int(round(settings.event_merge_us * 1e-6 * sample_rate)))

    for percentile in settings.prominence_percentiles:
        prominence = float(np.percentile(envelope, percentile))
        if prominence <= 0:
            continue
        for distance_us in settings.distance_us_options:
            distance_samples = max(1, int(round(distance_us * 1e-6 * sample_rate)))
            peaks, props = find_peaks(
                envelope,
                prominence=prominence,
                distance=distance_samples,
            )
            peaks = _merge_event_peaks(peaks, envelope, merge_samples)
            count = len(peaks)
            in_range = settings.min_clicks <= count <= settings.max_clicks
            range_penalty = 0.0 if in_range else min(abs(count - settings.min_clicks), abs(count - settings.max_clicks)) * 20.0
            count_penalty = abs(count - settings.preferred_clicks)
            threshold_penalty = (90.0 - percentile) * 0.05
            score = range_penalty + count_penalty + threshold_penalty
            if best is None or score < best[0]:
                best = (
                    score,
                    list(map(int, peaks)),
                    {
                        "prominence_percentile": percentile,
                        "prominence": prominence,
                        "distance_us": distance_us,
                        "detected_clicks": float(count),
                    },
                )

    if best is None:
        return [], envelope, {
            "prominence_percentile": np.nan,
            "prominence": np.nan,
            "distance_us": np.nan,
            "detected_clicks": 0.0,
        }

    _, peaks, chosen = best
    detections = [
        ClickDetection(
            click_id=i + 1,
            sample_index=peak,
            time_s=peak / sample_rate,
            peak_amplitude=float(np.max(np.abs(filtered[max(0, peak - 1) : peak + 2]))),
            envelope_value=float(envelope[peak]),
        )
        for i, peak in enumerate(peaks)
    ]
    return detections, envelope, chosen


def _merge_event_peaks(peaks: np.ndarray, envelope: np.ndarray, merge_samples: int) -> list[int]:
    if len(peaks) == 0:
        return []
    ordered = sorted(map(int, peaks))
    groups: list[list[int]] = [[ordered[0]]]
    group_start = ordered[0]
    for peak in ordered[1:]:
        if peak - group_start <= merge_samples:
            groups[-1].append(peak)
        else:
            groups.append([peak])
            group_start = peak
    merged = [max(group, key=lambda idx: envelope[idx]) for group in groups]
    return merged
