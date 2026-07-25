from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, resample_poly

from .multipath_detection import MultipathFeature


@dataclass(frozen=True)
class FpodPdfSettings:
    target_sample_rate: int = 4_000_000
    min_frequency_khz: float = 17.0
    max_frequency_khz: float = 220.0
    amplitude_percentiles: tuple[float, ...] = (85.0, 90.0, 95.0, 97.0)
    min_cycles: int = 2
    max_cycles: int = 255
    ipi_change_tolerance: float = 0.50
    half_peak_drop_fraction: float = 0.50
    low_cycle_threshold_steps: tuple[float, ...] = (2.2, 1.8, 1.45, 1.2)
    train_min_clicks: int = 3
    train_min_ici_us: float = 3_000.0
    train_max_ici_us: float = 250_000.0
    train_ici_change_tolerance: float = 0.75
    split_window_us: float = 400.0
    multipath_window_us: float = 3_000.0
    multipath_frequency_tolerance_khz: float = 90.0


@dataclass(frozen=True)
class FpodClickRecord:
    record_id: int
    sample_index: int
    time_5us: int
    time_us: float
    n_cycles: int
    loudest_wave_number: int
    ipi_range_250ns: int
    ipi_pk_minus1_250ns: int
    ipi_pk_250ns: int
    ipi_pk_plus1_250ns: int
    ipi_pk_plus2_250ns: int
    amp_pk_minus1: int
    amp_pmax: int
    amp_pk_plus1: int
    ipi_before_click_250ns: int
    amplitude_reversals: int
    duration_5us: int
    last_ipi_250ns: int
    sonar_flag: int
    frequency_khz: float
    bandwidth_index: float
    polarity: int
    train_id: int = 0
    multipath_cluster_id: int = 0
    multipath_cluster_size: int = 1
    cluster_low_frequency_khz: float = np.nan
    cluster_high_frequency_khz: float = np.nan


@dataclass(frozen=True)
class FpodPipelineResult:
    features: list[MultipathFeature]
    click_records: list[FpodClickRecord]
    sample_rate: int
    detection_choice: dict[str, float]
    train_count: int
    multipath_cluster_count: int


def run_fpod_pdf_pipeline(
    signal: np.ndarray,
    sample_rate: int,
    settings: FpodPdfSettings | None = None,
) -> FpodPipelineResult:
    settings = settings or FpodPdfSettings()
    fpod_sample_rate, fpod_signal = resample_to_fpod_rate(signal, sample_rate, settings)
    records, choice = extract_fpod_click_records(fpod_signal, fpod_sample_rate, settings)
    records = assign_multipath_clusters(records, settings)
    records = assign_trains(records, settings)
    features = records_to_multipath_features(records, fpod_signal, fpod_sample_rate)
    train_count = len({item.train_id for item in records if item.train_id > 0})
    cluster_count = len({item.multipath_cluster_id for item in records if item.multipath_cluster_id > 0})
    return FpodPipelineResult(
        features=features,
        click_records=records,
        sample_rate=fpod_sample_rate,
        detection_choice=choice,
        train_count=train_count,
        multipath_cluster_count=cluster_count,
    )


def resample_to_fpod_rate(
    signal: np.ndarray,
    sample_rate: int,
    settings: FpodPdfSettings,
) -> tuple[int, np.ndarray]:
    if sample_rate == settings.target_sample_rate:
        return sample_rate, signal
    ratio = Fraction(settings.target_sample_rate, sample_rate).limit_denominator(1000)
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    peak = float(np.max(np.abs(resampled))) if len(resampled) else 0.0
    if peak > 0.0:
        resampled = resampled / peak
    return settings.target_sample_rate, np.asarray(resampled, dtype=np.float64)


def extract_fpod_click_records(
    signal: np.ndarray,
    sample_rate: int,
    settings: FpodPdfSettings,
) -> tuple[list[FpodClickRecord], dict[str, float]]:
    rectified = np.abs(signal)
    best: tuple[float, list[FpodClickRecord], dict[str, float]] | None = None
    for percentile in settings.amplitude_percentiles:
        threshold = float(np.percentile(rectified, percentile))
        if threshold <= 0.0:
            continue
        records = _records_at_threshold(signal, sample_rate, threshold, settings)
        score = abs(len(records) - 120) + (97.0 - percentile) * 0.2
        if best is None or score < best[0]:
            best = (
                score,
                records,
                {
                    "amplitude_percentile": percentile,
                    "amplitude_threshold": threshold,
                    "click_records": float(len(records)),
                    "ipi_change_tolerance": settings.ipi_change_tolerance,
                },
            )
    if best is None:
        return [], {
            "amplitude_percentile": np.nan,
            "amplitude_threshold": np.nan,
            "click_records": 0.0,
            "ipi_change_tolerance": settings.ipi_change_tolerance,
        }
    _, records, choice = best
    return records, choice


def _records_at_threshold(
    signal: np.ndarray,
    sample_rate: int,
    threshold: float,
    settings: FpodPdfSettings,
) -> list[FpodClickRecord]:
    all_records: list[FpodClickRecord] = []
    for polarity, oriented in ((1, signal), (-1, -signal)):
        peaks = _confirmed_wave_peaks(oriented, sample_rate, threshold, settings)
        groups = _group_wave_peaks(peaks, sample_rate, settings)
        for group in groups:
            record = _make_click_record(
                len(all_records) + 1,
                group,
                signal,
                sample_rate,
                polarity,
                threshold,
                settings,
            )
            if record is not None:
                all_records.append(record)

    all_records.sort(key=lambda item: item.sample_index)
    suppressed: list[FpodClickRecord] = []
    suppress_samples = int(round(5.0e-6 * sample_rate))
    for record in all_records:
        if suppressed and record.sample_index - suppressed[-1].sample_index <= suppress_samples:
            if record.amp_pmax > suppressed[-1].amp_pmax:
                suppressed[-1] = record
            continue
        suppressed.append(record)

    return [
        _renumber_record(record, i + 1)
        for i, record in enumerate(suppressed)
    ]


def _confirmed_wave_peaks(
    oriented: np.ndarray,
    sample_rate: int,
    threshold: float,
    settings: FpodPdfSettings,
) -> list[int]:
    min_distance = max(1, int(round((1_000.0 / settings.max_frequency_khz) * 1e-6 * sample_rate)))
    peaks, props = find_peaks(oriented, height=threshold, distance=min_distance)
    confirmed: list[int] = []
    for i, peak in enumerate(map(int, peaks)):
        next_peak = int(peaks[i + 1]) if i + 1 < len(peaks) else min(len(oriented), peak + min_distance * 3)
        valley = float(np.min(oriented[peak:next_peak])) if next_peak > peak else float(oriented[peak])
        if valley <= oriented[peak] * settings.half_peak_drop_fraction:
            confirmed.append(peak)
    return confirmed


def _group_wave_peaks(
    peaks: list[int],
    sample_rate: int,
    settings: FpodPdfSettings,
) -> list[list[int]]:
    if not peaks:
        return []
    groups: list[list[int]] = []
    current = [peaks[0]]
    previous_ipi_us: float | None = None
    for previous, peak in zip(peaks[:-1], peaks[1:]):
        ipi_us = (peak - previous) / sample_rate * 1e6
        frequency_khz = 1000.0 / ipi_us if ipi_us > 0.0 else np.nan
        valid_frequency = settings.min_frequency_khz <= frequency_khz <= settings.max_frequency_khz
        valid_change = (
            previous_ipi_us is None
            or abs(ipi_us - previous_ipi_us) / max(previous_ipi_us, 1e-12)
            <= settings.ipi_change_tolerance
        )
        if valid_frequency and valid_change:
            current.append(peak)
            previous_ipi_us = ipi_us
        else:
            groups.append(current)
            current = [peak]
            previous_ipi_us = None
    groups.append(current)
    return groups


def _make_click_record(
    record_id: int,
    group: list[int],
    signal: np.ndarray,
    sample_rate: int,
    polarity: int,
    threshold: float,
    settings: FpodPdfSettings,
) -> FpodClickRecord | None:
    n_cycles = len(group)
    if not settings.min_cycles <= n_cycles <= settings.max_cycles:
        return None
    amplitudes = np.abs(signal[group])
    loudest_idx = int(np.argmax(amplitudes))
    low_cycle_multiplier = _threshold_multiplier(n_cycles, settings)
    if float(amplitudes[loudest_idx]) < threshold * low_cycle_multiplier:
        return None

    ipi_us = np.diff(group) / sample_rate * 1e6
    if len(ipi_us) == 0:
        return None
    ipi_250ns = np.rint(ipi_us / 0.25).astype(int)
    frequency_khz = _fpod_frequency_khz(ipi_us, n_cycles, loudest_idx)
    if not np.isfinite(frequency_khz):
        return None

    pmax_sample = group[loudest_idx]
    duration_us = (group[-1] - group[0]) / sample_rate * 1e6
    amp_u8 = np.clip(np.rint(amplitudes * 255.0), 0, 255).astype(int)
    return FpodClickRecord(
        record_id=record_id,
        sample_index=pmax_sample,
        time_5us=int(round((pmax_sample / sample_rate * 1e6) / 5.0)),
        time_us=pmax_sample / sample_rate * 1e6,
        n_cycles=n_cycles,
        loudest_wave_number=loudest_idx + 1,
        ipi_range_250ns=int(np.max(ipi_250ns) - np.min(ipi_250ns)) if len(ipi_250ns) else 0,
        ipi_pk_minus1_250ns=_ipi_at(ipi_250ns, loudest_idx - 1),
        ipi_pk_250ns=_ipi_at(ipi_250ns, loudest_idx),
        ipi_pk_plus1_250ns=_ipi_at(ipi_250ns, loudest_idx + 1),
        ipi_pk_plus2_250ns=_ipi_at(ipi_250ns, loudest_idx + 2),
        amp_pk_minus1=_amp_at(amp_u8, loudest_idx - 1),
        amp_pmax=int(amp_u8[loudest_idx]),
        amp_pk_plus1=_amp_at(amp_u8, loudest_idx + 1),
        ipi_before_click_250ns=0,
        amplitude_reversals=_amplitude_reversals(amp_u8),
        duration_5us=int(round(duration_us / 5.0)),
        last_ipi_250ns=int(ipi_250ns[-1]),
        sonar_flag=0,
        frequency_khz=frequency_khz,
        bandwidth_index=_bandwidth_index(ipi_250ns, amp_u8, loudest_idx),
        polarity=polarity,
    )


def _threshold_multiplier(n_cycles: int, settings: FpodPdfSettings) -> float:
    if n_cycles >= 5:
        return 1.0
    return settings.low_cycle_threshold_steps[max(0, min(3, 4 - n_cycles))]


def _fpod_frequency_khz(ipi_us: np.ndarray, n_cycles: int, loudest_idx: int) -> float:
    if n_cycles <= 4:
        near = [
            ipi_us[i]
            for i in (loudest_idx - 1, loudest_idx, loudest_idx + 1, loudest_idx + 2)
            if 0 <= i < len(ipi_us)
        ]
        return float(1000.0 / np.median(near)) if near else np.nan
    duration_us = float(np.sum(ipi_us[: min(16, len(ipi_us))]))
    cycles = min(16, len(ipi_us))
    return float(1000.0 * cycles / duration_us) if duration_us > 0.0 else np.nan


def _ipi_at(values: np.ndarray, index: int) -> int:
    return int(values[index]) if 0 <= index < len(values) else 0


def _amp_at(values: np.ndarray, index: int) -> int:
    return int(values[index]) if 0 <= index < len(values) else 0


def _amplitude_reversals(amplitudes: np.ndarray) -> int:
    if len(amplitudes) < 3:
        return 0
    diffs = np.diff(amplitudes.astype(float))
    signs = np.sign(diffs[diffs != 0])
    if len(signs) < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _bandwidth_index(ipi_250ns: np.ndarray, amplitudes: np.ndarray, loudest_idx: int) -> float:
    if len(ipi_250ns) < 2:
        return 999.0
    center_ipi = max(float(_ipi_at(ipi_250ns, min(loudest_idx, len(ipi_250ns) - 1))), 1.0)
    center_amp = max(float(amplitudes[loudest_idx]), 1.0)
    ipi_change = float(np.sum(np.abs(ipi_250ns - center_ipi) / center_ipi))
    amp_change = float(np.sum(np.abs(amplitudes - center_amp) / center_amp))
    return ipi_change + amp_change


def _renumber_record(record: FpodClickRecord, record_id: int) -> FpodClickRecord:
    return FpodClickRecord(**{**record.__dict__, "record_id": record_id})


def assign_multipath_clusters(
    records: list[FpodClickRecord],
    settings: FpodPdfSettings,
) -> list[FpodClickRecord]:
    output: list[FpodClickRecord] = []
    used: set[int] = set()
    cluster_id = 0
    for record in records:
        if record.record_id in used:
            continue
        members = [
            candidate
            for candidate in records
            if candidate.record_id not in used
            and 0.0 <= candidate.time_us - record.time_us <= settings.multipath_window_us
            and abs(candidate.frequency_khz - record.frequency_khz)
            <= settings.multipath_frequency_tolerance_khz
        ]
        if len(members) > 1:
            cluster_id += 1
        current_cluster = cluster_id if len(members) > 1 else 0
        low = min(item.frequency_khz for item in members)
        high = max(item.frequency_khz for item in members)
        for member in members:
            used.add(member.record_id)
            output.append(
                FpodClickRecord(
                    **{
                        **member.__dict__,
                        "multipath_cluster_id": current_cluster,
                        "multipath_cluster_size": len(members),
                        "cluster_low_frequency_khz": low,
                        "cluster_high_frequency_khz": high,
                    }
                )
            )
    return sorted(output, key=lambda item: item.sample_index)


def assign_trains(records: list[FpodClickRecord], settings: FpodPdfSettings) -> list[FpodClickRecord]:
    representatives = [
        item
        for item in records
        if item.multipath_cluster_id == 0
        or _is_cluster_representative(item, records)
    ]
    trains: list[list[FpodClickRecord]] = []
    current: list[FpodClickRecord] = []
    previous_ici_us: float | None = None
    for record in representatives:
        if not current:
            current = [record]
            previous_ici_us = None
            continue
        ici_us = record.time_us - current[-1].time_us
        valid_ici = settings.train_min_ici_us <= ici_us <= settings.train_max_ici_us
        valid_change = (
            previous_ici_us is None
            or abs(ici_us - previous_ici_us) / max(previous_ici_us, 1e-12)
            <= settings.train_ici_change_tolerance
        )
        if valid_ici and valid_change:
            current.append(record)
            previous_ici_us = ici_us
        else:
            if len(current) >= settings.train_min_clicks:
                trains.append(current)
            current = [record]
            previous_ici_us = None
    if len(current) >= settings.train_min_clicks:
        trains.append(current)

    train_by_cluster_or_record: dict[int, int] = {}
    for train_id, train in enumerate(trains, start=1):
        for item in train:
            key = item.multipath_cluster_id if item.multipath_cluster_id > 0 else -item.record_id
            train_by_cluster_or_record[key] = train_id

    assigned: list[FpodClickRecord] = []
    for record in records:
        key = record.multipath_cluster_id if record.multipath_cluster_id > 0 else -record.record_id
        assigned.append(FpodClickRecord(**{**record.__dict__, "train_id": train_by_cluster_or_record.get(key, 0)}))
    return assigned


def _is_cluster_representative(record: FpodClickRecord, records: list[FpodClickRecord]) -> bool:
    if record.multipath_cluster_id == 0:
        return True
    first = min(
        item.sample_index
        for item in records
        if item.multipath_cluster_id == record.multipath_cluster_id
    )
    return record.sample_index == first


def records_to_multipath_features(
    records: list[FpodClickRecord],
    signal: np.ndarray,
    sample_rate: int,
) -> list[MultipathFeature]:
    features: list[MultipathFeature] = []
    by_cluster: dict[int, list[FpodClickRecord]] = {}
    singles: list[FpodClickRecord] = []
    for record in records:
        if record.multipath_cluster_id > 0:
            by_cluster.setdefault(record.multipath_cluster_id, []).append(record)
        else:
            singles.append(record)

    for record in singles:
        feature = _record_to_feature(record, record, signal, sample_rate, "single")
        feature.cluster_id = record.train_id
        feature.is_direct_path = True
        features.append(feature)

    for cluster_id, members in sorted(by_cluster.items()):
        members = sorted(members, key=lambda item: item.sample_index)
        direct = members[0]
        for rank, member in enumerate(members):
            role = "direct" if rank == 0 else f"multipath_{rank}"
            feature = _record_to_feature(member, direct, signal, sample_rate, role)
            feature.cluster_id = cluster_id
            feature.is_direct_path = rank == 0
            features.append(feature)
    return sorted(features, key=lambda item: item.arrival_time_us)


def _record_to_feature(
    record: FpodClickRecord,
    direct: FpodClickRecord,
    signal: np.ndarray,
    sample_rate: int,
    role: str,
) -> MultipathFeature:
    amp = max(record.amp_pmax / 255.0, 1e-12)
    direct_amp = max(direct.amp_pmax / 255.0, 1e-12)
    return MultipathFeature(
        click_id=direct.record_id,
        event_role=role,
        direct_sample=direct.sample_index,
        sample_index=record.sample_index,
        arrival_time_us=record.time_us,
        delay_us=record.time_us - direct.time_us,
        frequency_khz=record.frequency_khz,
        amplitude_db=float(20.0 * np.log10(amp / direct_amp + 1e-12)),
        linear_amplitude=amp,
        rms_energy=_local_energy(signal, sample_rate, record.sample_index),
        n_wave_peaks=record.n_cycles,
        dominant_ipi_us=_dominant_ipi_us(record),
        click_duration_us=record.duration_5us * 5.0,
        frequency_points_khz=[record.frequency_khz],
        frequency_strength=[0.0],
    )


def _dominant_ipi_us(record: FpodClickRecord) -> float:
    values = [
        record.ipi_pk_minus1_250ns,
        record.ipi_pk_250ns,
        record.ipi_pk_plus1_250ns,
        record.ipi_pk_plus2_250ns,
    ]
    valid = [item for item in values if item > 0]
    return float(np.median(valid) * 0.25) if valid else np.nan


def _local_energy(signal: np.ndarray, sample_rate: int, center_sample: int, half_us: float = 50.0) -> float:
    half = max(1, int(round(half_us * 1e-6 * sample_rate)))
    start = max(0, center_sample - half)
    end = min(len(signal), center_sample + half + 1)
    segment = signal[start:end]
    if len(segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment * segment)))


def write_fpod_records_csv(records: list[FpodClickRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FpodClickRecord.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
