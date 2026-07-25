from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import find_peaks, resample_poly

from .multipath_detection import MultipathFeature


@dataclass(frozen=True)
class IpiFpodSettings:
    target_sample_rate: int = 4_000_000
    min_frequency_khz: float = 17.0
    max_frequency_khz: float = 220.0
    amplitude_percentiles: tuple[float, ...] = (80.0, 85.0, 90.0, 95.0, 97.0)
    min_wave_peaks: int = 3
    max_wave_peaks: int = 255
    ipi_change_tolerance: float = 0.50
    min_clicks: int = 10
    max_clicks: int = 500
    preferred_clicks: int = 13
    min_direct_gap_us: float = 500.0
    multipath_min_delay_us: float = 5.0
    multipath_max_delay_us: float = 3_000.0
    multipath_max_members: int = 20
    multipath_frequency_tolerance_khz: float = 85.0


@dataclass(frozen=True)
class IpiClick:
    click_id: int
    sample_index: int
    time_s: float
    peak_amplitude: float
    frequency_khz: float
    n_wave_peaks: int
    dominant_ipi_us: float
    duration_us: float
    wave_peak_samples: tuple[int, ...]
    wave_peak_amplitudes: tuple[float, ...]


def resample_to_fpod_rate(
    signal: np.ndarray,
    sample_rate: int,
    settings: IpiFpodSettings,
) -> tuple[int, np.ndarray]:
    if sample_rate == settings.target_sample_rate:
        return sample_rate, signal
    ratio = Fraction(settings.target_sample_rate, sample_rate).limit_denominator(1000)
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    peak = float(np.max(np.abs(resampled))) if len(resampled) else 0.0
    if peak > 0.0:
        resampled = resampled / peak
    return settings.target_sample_rate, np.asarray(resampled, dtype=np.float64)


def extract_ipi_clicks(
    signal: np.ndarray,
    sample_rate: int,
    settings: IpiFpodSettings | None = None,
) -> tuple[list[IpiClick], dict[str, float]]:
    settings = settings or IpiFpodSettings()
    rectified = np.abs(signal)
    best: tuple[float, list[IpiClick], dict[str, float]] | None = None

    for percentile in settings.amplitude_percentiles:
        threshold = float(np.percentile(rectified, percentile))
        if threshold <= 0.0:
            continue
        clicks = _extract_clicks_at_threshold(signal, rectified, sample_rate, threshold, settings)
        count = len(clicks)
        in_range = settings.min_clicks <= count <= settings.max_clicks
        range_penalty = (
            0.0
            if in_range
            else min(abs(count - settings.min_clicks), abs(count - settings.max_clicks)) * 20.0
        )
        count_penalty = abs(count - settings.preferred_clicks)
        threshold_penalty = (97.0 - percentile) * 0.03
        score = range_penalty + count_penalty + threshold_penalty
        if best is None or score < best[0]:
            best = (
                score,
                clicks,
                {
                    "amplitude_percentile": percentile,
                    "amplitude_threshold": threshold,
                    "detected_clicks": float(count),
                    "target_sample_rate": float(settings.target_sample_rate),
                    "ipi_change_tolerance": settings.ipi_change_tolerance,
                },
            )

    if best is None:
        return [], {
            "amplitude_percentile": np.nan,
            "amplitude_threshold": np.nan,
            "detected_clicks": 0.0,
            "target_sample_rate": float(settings.target_sample_rate),
            "ipi_change_tolerance": settings.ipi_change_tolerance,
        }

    _, clicks, choice = best
    return clicks, choice


def _extract_clicks_at_threshold(
    signal: np.ndarray,
    rectified: np.ndarray,
    sample_rate: int,
    threshold: float,
    settings: IpiFpodSettings,
) -> list[IpiClick]:
    min_peak_distance = max(1, int(round(2.0e-6 * sample_rate)))
    peaks, props = find_peaks(rectified, height=threshold, distance=min_peak_distance)
    if len(peaks) < settings.min_wave_peaks:
        return []

    groups: list[list[int]] = []
    current: list[int] = [int(peaks[0])]
    previous_ipi_us: float | None = None

    for prev_peak, peak in zip(peaks[:-1], peaks[1:]):
        ipi_us = (int(peak) - int(prev_peak)) / sample_rate * 1e6
        valid_frequency = _is_valid_ipi(ipi_us, settings)
        valid_change = (
            previous_ipi_us is None
            or abs(ipi_us - previous_ipi_us) / max(previous_ipi_us, 1e-12)
            <= settings.ipi_change_tolerance
        )
        if valid_frequency and valid_change:
            current.append(int(peak))
            previous_ipi_us = ipi_us
        else:
            groups.append(current)
            current = [int(peak)]
            previous_ipi_us = None
    groups.append(current)

    clicks: list[IpiClick] = []
    for group in groups:
        if not settings.min_wave_peaks <= len(group) <= settings.max_wave_peaks:
            continue
        click = _make_ipi_click(len(clicks) + 1, group, signal, rectified, sample_rate)
        if np.isfinite(click.frequency_khz):
            clicks.append(click)
    return clicks


def _is_valid_ipi(ipi_us: float, settings: IpiFpodSettings) -> bool:
    if ipi_us <= 0.0:
        return False
    frequency_khz = 1000.0 / ipi_us
    return settings.min_frequency_khz <= frequency_khz <= settings.max_frequency_khz


def _make_ipi_click(
    click_id: int,
    group: list[int],
    signal: np.ndarray,
    rectified: np.ndarray,
    sample_rate: int,
) -> IpiClick:
    amplitudes = rectified[group]
    loudest_pos = int(np.argmax(amplitudes))
    loudest_sample = int(group[loudest_pos])
    ipi_us = np.diff(group) / sample_rate * 1e6
    dominant_ipi_us = float(np.median(ipi_us)) if len(ipi_us) else np.nan
    duration_us = float((group[-1] - group[0]) / sample_rate * 1e6)
    frequency_khz = float(1000.0 / dominant_ipi_us) if dominant_ipi_us > 0.0 else np.nan
    return IpiClick(
        click_id=click_id,
        sample_index=loudest_sample,
        time_s=loudest_sample / sample_rate,
        peak_amplitude=float(signal[loudest_sample]),
        frequency_khz=frequency_khz,
        n_wave_peaks=len(group),
        dominant_ipi_us=dominant_ipi_us,
        duration_us=duration_us,
        wave_peak_samples=tuple(map(int, group)),
        wave_peak_amplitudes=tuple(map(float, amplitudes)),
    )


def extract_ipi_multipath_features(
    clicks: list[IpiClick],
    signal: np.ndarray,
    sample_rate: int,
    settings: IpiFpodSettings | None = None,
) -> tuple[list[MultipathFeature], dict[str, float]]:
    settings = settings or IpiFpodSettings()
    features: list[MultipathFeature] = []
    sorted_clicks = sorted(clicks, key=lambda item: item.sample_index)
    used_as_replica: set[int] = set()
    direct_count = 0
    replica_count = 0
    last_direct_sample = -10**18
    direct_gap_samples = int(round(settings.min_direct_gap_us * 1e-6 * sample_rate))

    for direct in sorted_clicks:
        if direct.click_id in used_as_replica:
            continue
        if direct.sample_index - last_direct_sample < direct_gap_samples:
            continue

        direct_count += 1
        last_direct_sample = direct.sample_index
        direct_amp = max(abs(direct.peak_amplitude), 1e-12)
        features.append(_click_to_feature(direct, direct, direct_amp, signal, sample_rate, "direct"))

        members = _find_multipath_members(direct, sorted_clicks, settings)
        for rank, member in enumerate(members[: settings.multipath_max_members], start=1):
            used_as_replica.add(member.click_id)
            replica_count += 1
            features.append(
                _click_to_feature(
                    member,
                    direct,
                    direct_amp,
                    signal,
                    sample_rate,
                    f"multipath_{rank}",
                )
            )

    return features, {
        "direct_clicks": float(direct_count),
        "replica_clicks": float(replica_count),
        "min_delay_us": settings.multipath_min_delay_us,
        "max_delay_us": settings.multipath_max_delay_us,
        "max_members": float(settings.multipath_max_members),
        "frequency_tolerance_khz": settings.multipath_frequency_tolerance_khz,
    }


def _find_multipath_members(
    direct: IpiClick,
    clicks: list[IpiClick],
    settings: IpiFpodSettings,
) -> list[IpiClick]:
    members: list[IpiClick] = []
    for candidate in clicks:
        delay_us = (candidate.sample_index - direct.sample_index) / settings.target_sample_rate * 1e6
        if delay_us <= settings.multipath_min_delay_us:
            continue
        if delay_us > settings.multipath_max_delay_us:
            break
        if abs(candidate.frequency_khz - direct.frequency_khz) > settings.multipath_frequency_tolerance_khz:
            continue
        members.append(candidate)
    return members


def _click_to_feature(
    click: IpiClick,
    direct: IpiClick,
    direct_amp: float,
    signal: np.ndarray,
    sample_rate: int,
    role: str,
) -> MultipathFeature:
    amp = max(abs(click.peak_amplitude), 1e-12)
    delay_us = (click.sample_index - direct.sample_index) / sample_rate * 1e6
    return MultipathFeature(
        click_id=direct.click_id,
        event_role=role,
        direct_sample=direct.sample_index,
        sample_index=click.sample_index,
        arrival_time_us=click.time_s * 1e6,
        delay_us=delay_us,
        frequency_khz=click.frequency_khz,
        amplitude_db=float(20.0 * np.log10(amp / direct_amp + 1e-12)),
        linear_amplitude=amp,
        rms_energy=_local_energy(signal, sample_rate, click.sample_index),
        n_wave_peaks=click.n_wave_peaks,
        dominant_ipi_us=click.dominant_ipi_us,
        click_duration_us=click.duration_us,
        frequency_points_khz=[click.frequency_khz],
        frequency_strength=[0.0],
    )


def _local_energy(signal: np.ndarray, sample_rate: int, center_sample: int, half_us: float = 50.0) -> float:
    half = max(1, int(round(half_us * 1e-6 * sample_rate)))
    start = max(0, center_sample - half)
    end = min(len(signal), center_sample + half + 1)
    segment = signal[start:end]
    if len(segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment * segment)))
