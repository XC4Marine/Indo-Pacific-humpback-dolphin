from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .click_detection import ClickDetection


@dataclass
class MultipathFeature:
    click_id: int
    event_role: str
    direct_sample: int
    sample_index: int
    arrival_time_us: float
    delay_us: float
    frequency_khz: float
    amplitude_db: float
    linear_amplitude: float
    rms_energy: float
    n_wave_peaks: int = 0
    dominant_ipi_us: float = np.nan
    click_duration_us: float = np.nan
    frequency_points_khz: list[float] = field(default_factory=list)
    frequency_strength: list[float] = field(default_factory=list)
    cluster_id: int = -1
    is_direct_path: bool = False


@dataclass(frozen=True)
class MultipathSettings:
    pre_us: float = 50.0
    post_us: float = 900.0
    min_delay_us: float = 70.0
    max_delay_us: float = 900.0
    replica_distance_us: float = 12.0
    replica_prominence_percentile: float = 50.0
    max_replicas: int = 20
    fft_half_window_us: float = 70.0
    fft_min_khz: float = 20.0
    fft_max_khz: float = 220.0
    spectral_floor_db: float = -12.0


def _event_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    center_sample: int,
    settings: MultipathSettings,
) -> tuple[float, list[float], list[float]]:
    ipi_freq, n_peaks, dominant_ipi_us, duration_us = _ipi_frequency(
        signal, sample_rate, center_sample, settings
    )
    if np.isfinite(ipi_freq):
        return ipi_freq, [ipi_freq], [0.0]

    half = max(4, int(round(settings.fft_half_window_us * 1e-6 * sample_rate)))
    start = max(0, center_sample - half)
    end = min(len(signal), center_sample + half + 1)
    segment = signal[start:end]
    if len(segment) < 4:
        return np.nan, [], []

    segment = segment - np.mean(segment)
    windowed = segment * np.hanning(len(segment))
    n_fft = 1
    while n_fft < len(windowed) * 8:
        n_fft *= 2
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs_khz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate) / 1000.0
    band = (freqs_khz >= settings.fft_min_khz) & (freqs_khz <= settings.fft_max_khz)
    if not np.any(band):
        return np.nan, [], []

    f = freqs_khz[band]
    s = spectrum[band]
    max_s = float(np.max(s))
    if max_s <= 0:
        return np.nan, [], []

    weights = s ** 2
    center = float(np.sum(f * weights) / np.sum(weights))
    db = 20.0 * np.log10(s / max_s + 1e-12)
    keep = db >= settings.spectral_floor_db
    if np.count_nonzero(keep) < 3:
        strongest = np.argsort(s)[-5:]
        keep = np.zeros_like(s, dtype=bool)
        keep[strongest] = True

    # Down-sample dense FFT bins so each click remains a compact vertical trace.
    kept_freqs = f[keep]
    kept_strength = db[keep]
    main_band = np.abs(kept_freqs - center) <= 45.0
    main_band &= kept_freqs <= 160.0
    if np.count_nonzero(main_band) >= 3:
        kept_freqs = kept_freqs[main_band]
        kept_strength = kept_strength[main_band]
    if len(kept_freqs) > 7:
        idx = np.linspace(0, len(kept_freqs) - 1, 7).astype(int)
        kept_freqs = kept_freqs[idx]
        kept_strength = kept_strength[idx]
    return center, kept_freqs.tolist(), kept_strength.tolist()


def _ipi_frequency(
    signal: np.ndarray,
    sample_rate: int,
    center_sample: int,
    settings: MultipathSettings,
) -> tuple[float, int, float, float]:
    half = max(4, int(round(settings.fft_half_window_us * 1e-6 * sample_rate)))
    start = max(0, center_sample - half)
    end = min(len(signal), center_sample + half + 1)
    segment = signal[start:end]
    if len(segment) < 5:
        return np.nan, 0, np.nan, np.nan

    local_center = center_sample - start
    main_idx = int(np.argmax(np.abs(segment)))
    polarity = 1.0 if segment[main_idx] >= 0 else -1.0
    oriented = polarity * (segment - np.median(segment))
    max_val = float(np.max(oriented))
    if max_val <= 0:
        return np.nan, 0, np.nan, np.nan

    min_distance = max(1, int(round(2.0e-6 * sample_rate)))
    peaks, _ = find_peaks(oriented, height=max_val * 0.12, distance=min_distance)
    if len(peaks) < 3:
        return np.nan, len(peaks), np.nan, np.nan

    close = peaks[np.abs(peaks - local_center) <= max(3, int(round(65e-6 * sample_rate)))]
    if len(close) >= 3:
        peaks = close

    ipi_us = np.diff(peaks) / sample_rate * 1e6
    valid = ipi_us[(ipi_us >= 4.0) & (ipi_us <= 60.0)]
    if len(valid) < 2:
        return np.nan, len(peaks), np.nan, np.nan

    dominant_ipi_us = float(np.median(valid))
    frequency_khz = float(1000.0 / dominant_ipi_us)
    duration_us = float((peaks[-1] - peaks[0]) / sample_rate * 1e6)
    if not 17.0 <= frequency_khz <= 220.0:
        return np.nan, len(peaks), dominant_ipi_us, duration_us
    return frequency_khz, len(peaks), dominant_ipi_us, duration_us


def _click_wave_metrics(
    signal: np.ndarray,
    sample_rate: int,
    center_sample: int,
    settings: MultipathSettings,
) -> tuple[int, float, float]:
    _, n_peaks, dominant_ipi_us, duration_us = _ipi_frequency(
        signal, sample_rate, center_sample, settings
    )
    return n_peaks, dominant_ipi_us, duration_us


def _local_energy(signal: np.ndarray, sample_rate: int, center_sample: int, half_us: float = 50.0) -> float:
    half = max(1, int(round(half_us * 1e-6 * sample_rate)))
    start = max(0, center_sample - half)
    end = min(len(signal), center_sample + half + 1)
    segment = signal[start:end]
    if len(segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment * segment)))


def extract_multipath_features(
    filtered: np.ndarray,
    envelope: np.ndarray,
    detections: list[ClickDetection],
    sample_rate: int,
    settings: MultipathSettings,
) -> list[MultipathFeature]:
    features: list[MultipathFeature] = []

    for detection in detections:
        direct_sample = detection.sample_index
        pre = int(round(settings.pre_us * 1e-6 * sample_rate))
        post = int(round(settings.post_us * 1e-6 * sample_rate))
        start = max(0, direct_sample - pre)
        end = min(len(filtered), direct_sample + post + 1)
        local_env = envelope[start:end]
        if len(local_env) < 3:
            continue

        local_direct = direct_sample - start
        direct_amp = max(float(envelope[direct_sample]), 1e-12)
        direct_freq, direct_f_points, direct_f_strength = _event_spectrum(
            filtered, sample_rate, direct_sample, settings
        )
        direct_n_peaks, direct_ipi_us, direct_duration_us = _click_wave_metrics(
            filtered, sample_rate, direct_sample, settings
        )
        features.append(
            MultipathFeature(
                click_id=detection.click_id,
                event_role="direct",
                direct_sample=direct_sample,
                sample_index=direct_sample,
                arrival_time_us=detection.time_s * 1e6,
                delay_us=0.0,
                frequency_khz=direct_freq,
                amplitude_db=0.0,
                linear_amplitude=direct_amp,
                rms_energy=_local_energy(filtered, sample_rate, direct_sample),
                n_wave_peaks=direct_n_peaks,
                dominant_ipi_us=direct_ipi_us,
                click_duration_us=direct_duration_us,
                frequency_points_khz=direct_f_points,
                frequency_strength=direct_f_strength,
            )
        )

        min_delay_us = settings.min_delay_us
        if np.isfinite(direct_duration_us):
            min_delay_us = max(min_delay_us, direct_duration_us + 20.0)
        min_lag = max(1, int(round(min_delay_us * 1e-6 * sample_rate)))
        max_lag = min(
            len(local_env) - local_direct - 1,
            int(round(settings.max_delay_us * 1e-6 * sample_rate)),
        )
        if max_lag <= min_lag:
            continue

        search = local_env[local_direct + min_lag : local_direct + max_lag + 1]
        prominence = float(np.percentile(search, settings.replica_prominence_percentile))
        distance = max(1, int(round(settings.replica_distance_us * 1e-6 * sample_rate)))
        peaks, props = find_peaks(search, prominence=prominence, distance=distance)
        if len(peaks) == 0:
            continue

        peak_prom = props.get("prominences", np.zeros_like(peaks, dtype=float))
        order = np.argsort(peak_prom)[::-1]
        selected = sorted(peaks[order[: settings.max_replicas]])

        for rank, local_peak in enumerate(selected, start=1):
            sample_index = start + local_direct + min_lag + int(local_peak)
            amp = max(float(envelope[sample_index]), 1e-12)
            delay_us = (sample_index - direct_sample) / sample_rate * 1e6
            freq, f_points, f_strength = _event_spectrum(
                filtered, sample_rate, sample_index, settings
            )
            n_peaks, dominant_ipi_us, duration_us = _click_wave_metrics(
                filtered, sample_rate, sample_index, settings
            )
            features.append(
                MultipathFeature(
                    click_id=detection.click_id,
                    event_role=f"multipath_{rank}",
                    direct_sample=direct_sample,
                    sample_index=sample_index,
                    arrival_time_us=sample_index / sample_rate * 1e6,
                    delay_us=delay_us,
                    frequency_khz=freq,
                    amplitude_db=float(20.0 * np.log10(amp / direct_amp + 1e-12)),
                    linear_amplitude=amp,
                    rms_energy=_local_energy(filtered, sample_rate, sample_index),
                    n_wave_peaks=n_peaks,
                    dominant_ipi_us=dominant_ipi_us,
                    click_duration_us=duration_us,
                    frequency_points_khz=f_points,
                    frequency_strength=f_strength,
                )
            )

    return features


def auto_extract_multipath_features(
    filtered: np.ndarray,
    envelope: np.ndarray,
    detections: list[ClickDetection],
    sample_rate: int,
) -> tuple[list[MultipathFeature], MultipathSettings]:
    trials = [
        MultipathSettings(post_us=1200.0, max_delay_us=1200.0, replica_distance_us=6.0, replica_prominence_percentile=45.0, max_replicas=10),
        MultipathSettings(post_us=1800.0, max_delay_us=1800.0, replica_distance_us=5.0, replica_prominence_percentile=35.0, max_replicas=14),
        MultipathSettings(post_us=2400.0, max_delay_us=2400.0, replica_distance_us=4.0, replica_prominence_percentile=30.0, max_replicas=18),
        MultipathSettings(post_us=3000.0, max_delay_us=3000.0, replica_distance_us=3.0, replica_prominence_percentile=25.0, max_replicas=20),
    ]
    best_features: list[MultipathFeature] = []
    best_settings = trials[0]
    best_score = float("inf")
    direct_count = max(1, len(detections))

    for settings in trials:
        features = extract_multipath_features(filtered, envelope, detections, sample_rate, settings)
        replicas = [f for f in features if f.delay_us > 0]
        replica_ratio = len(replicas) / direct_count
        no_mp_penalty = 50.0 if len(replicas) == 0 else 0.0
        random_penalty = max(0.0, replica_ratio - 20.0) * 8.0
        target_penalty = abs(replica_ratio - 14.0)
        score = no_mp_penalty + random_penalty + target_penalty
        if score < best_score:
            best_score = score
            best_features = features
            best_settings = settings

    return best_features, best_settings
