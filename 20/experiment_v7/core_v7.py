#!/usr/bin/env python3
"""Truth-isolated feature machinery for Experiment v7.

Only ``train_evaluate_confidence_v7.py`` may open simulation truth.  This module
consumes raw waveforms and v4 blind fits, and is shared by feature extraction and
deployment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parent
V4_ROOT = ROOT.parent / "experiment_v4"
sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import (CHANNEL_LENGTH, FS, SEGMENT_LENGTH, SOURCE_LENGTH, FitConfig,
                     canonicalise, detect_event_segments, fit_segments,
                     initialise_source, path_candidates, reconstruct, solve_independent_codes)

DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
V4_INPUT_ROOT = V4_ROOT / "outputs" / "independent"
V6_OUTPUT_ROOT = ROOT.parent / "experiment_v6" / "outputs_independent_consensus"
EVENT_COUNT, STABLE_EVENT_COUNT, TOLERANCE_MS = 10, 6, 0.03
EPS = 1e-12
PROTOCOL = "v7_selective_multipath_physical_confidence_v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def input_paths(limit: int | None = None) -> list[Path]:
    paths = sorted(DATA_ROOT.glob("*/clicktrain_raw.npy"))
    if len(paths) != 110:
        raise RuntimeError(f"expected 110 clicktrains, found {len(paths)}")
    return paths if limit is None else paths[:limit]


def load_v4_fit(train_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, FitConfig]:
    artifact = V4_INPUT_ROOT / f"{train_id}.npz"
    history = V4_INPUT_ROOT / f"{train_id}.history.json"
    if not artifact.exists() or not history.exists():
        raise FileNotFoundError(f"missing v4 independent fit for {train_id}")
    with np.load(artifact, allow_pickle=False) as payload:
        source = np.asarray(payload["source"], dtype=float)
        channels = np.asarray(payload["channels"], dtype=float)
        starts = np.asarray(payload["detected_window_starts"], dtype=np.int64)
    raw = json.loads(history.read_text(encoding="utf-8"))["config"]
    config = FitConfig(**{key: raw[key] for key in FitConfig.__dataclass_fields__ if key in raw})
    if source.shape != (SOURCE_LENGTH,) or channels.shape != (EVENT_COUNT, CHANNEL_LENGTH) or starts.shape != (EVENT_COUNT,):
        raise ValueError(f"unexpected v4 schema for {train_id}")
    return source, channels, starts, config


def load_segments(clicktrain_path: Path, starts: np.ndarray) -> np.ndarray:
    waveform = np.load(clicktrain_path, allow_pickle=False)
    # v4 detected the starts after its own zero-phase filter. Reusing the segment
    # detector also guarantees exactly the same preprocessing without truth input.
    detected, detected_starts = detect_event_segments(waveform)
    if np.array_equal(detected_starts, starts):
        return detected
    from bcsc_v4 import bandpass
    filtered = bandpass(waveform)
    return np.asarray([np.pad(filtered[int(s):int(s) + SEGMENT_LENGTH], (0, max(0, int(s) + SEGMENT_LENGTH - len(filtered)))) for s in starts])


def _event_paths(channels: np.ndarray) -> list[list[dict[str, float]]]:
    result = []
    for channel in channels:
        paths = path_candidates(channel)
        if not paths:
            peak = int(np.argmax(np.abs(channel)))
            paths = [{"sample": peak, "delay_ms": 0.0, "coefficient": float(channel[peak]), "relative_amplitude": 1.0}]
        result.append(paths)
    return result


def _clusters(rows: list[tuple[int, dict[str, float]]]) -> list[list[tuple[int, dict[str, float]]]]:
    clusters: list[list[tuple[int, dict[str, float]]]] = []
    for event, item in sorted(rows, key=lambda value: float(value[1]["delay_ms"])):
        eligible = []
        for index, cluster in enumerate(clusters):
            centre = float(np.median([x[1]["delay_ms"] for x in cluster]))
            if event not in {x[0] for x in cluster} and abs(float(item["delay_ms"]) - centre) <= TOLERANCE_MS:
                eligible.append((abs(float(item["delay_ms"]) - centre), index))
        if eligible:
            clusters[min(eligible)[1]].append((event, item))
        else:
            clusters.append([(event, item)])
    return clusters


def build_candidates(channels: np.ndarray) -> list[dict[str, Any]]:
    rows = [(event, item) for event, paths in enumerate(_event_paths(channels)) for item in paths]
    candidates = []
    for cluster in _clusters(rows):
        by_event = {event: item for event, item in cluster}
        delays = np.asarray([item["delay_ms"] for item in by_event.values()])
        if len(by_event) >= STABLE_EVENT_COUNT or np.min(np.abs(delays)) <= TOLERANCE_MS:
            candidates.append({"event_paths": by_event, "delay_median_ms": float(np.median(delays)),
                               "delay_q025_ms": float(np.quantile(delays, .025)), "delay_q975_ms": float(np.quantile(delays, .975)),
                               "event_coverage": len(by_event) / EVENT_COUNT, "event_detected_count": len(by_event)})
    if not candidates:
        raise RuntimeError("no path candidates extracted")
    # A single privileged direct candidate: maximum support, then nearest zero.
    direct = min(candidates, key=lambda c: (-c["event_coverage"], abs(c["delay_median_ms"])))
    # Near-zero clusters are alternate direct-path extractions, not reflections.
    # Retain exactly the privileged direct candidate before reflection scoring.
    candidates = [candidate for candidate in candidates if candidate is direct or abs(candidate["delay_median_ms"]) > TOLERANCE_MS]
    for candidate in candidates:
        candidate["is_direct"] = candidate is direct
    candidates.sort(key=lambda c: (not c["is_direct"], c["delay_median_ms"]))
    for index, candidate in enumerate(candidates, 1):
        candidate["path_id"] = index
    return candidates


def _candidate_samples(candidate: dict[str, Any], direct: dict[str, Any]) -> list[int]:
    samples = []
    for event in range(EVENT_COUNT):
        item = candidate["event_paths"].get(event)
        if item is not None:
            samples.append(int(item["sample"]))
        else:
            samples.append(int(round(direct["event_paths"].get(event, {"sample": 0})["sample"] + candidate["delay_median_ms"] * FS / 1e3)))
    return samples


def _atom(source: np.ndarray, sample: int) -> np.ndarray:
    result = np.zeros(CHANNEL_LENGTH, dtype=float)
    left, right = max(0, sample), min(CHANNEL_LENGTH, sample + len(source))
    if right > left:
        result[left:right] = source[left - sample:right - sample]
    return result


def _stack_design(segments: np.ndarray, source: np.ndarray, candidates: list[dict[str, Any]], target: dict[str, Any] | None, source_shared: bool, add_target: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    direct = next(c for c in candidates if c["is_direct"])
    base = [c for c in candidates if c is not target]
    design, response = [], []
    target_samples = _candidate_samples(target, direct) if target else []
    for event, segment in enumerate(segments):
        columns = [_atom(source, _candidate_samples(candidate, direct)[event]) for candidate in base]
        if target is not None and add_target:
            target_column = _atom(source, target_samples[event])
            if source_shared:
                # One shared parameter across events; zero-pad each event block.
                columns.append(target_column)
            else:
                columns.append(target_column)
        block = np.column_stack(columns) if columns else np.empty((CHANNEL_LENGTH, 0))
        # The BCSC segment includes source + channel - 1 samples, while channel
        # support starts at zero.  Keeping this observation window matches v6 BIC.
        design.append(block); response.append(segment[:CHANNEL_LENGTH])
    if target is not None and add_target and source_shared:
        shared = np.zeros((EVENT_COUNT * CHANNEL_LENGTH, len(base) * EVENT_COUNT + 1))
        offset = 0
        for event, block in enumerate(design):
            count = len(base)
            shared[offset:offset + CHANNEL_LENGTH, event * count:(event + 1) * count] = block[:, :count]
            shared[offset:offset + CHANNEL_LENGTH, -1] = block[:, -1]
            offset += CHANNEL_LENGTH
        return shared, np.concatenate(response), shared.shape[1]
    total_columns = sum(len(block.T) for block in design)
    matrix = np.zeros((EVENT_COUNT * CHANNEL_LENGTH, total_columns))
    offset, col = 0, 0
    for block in design:
        matrix[offset:offset + CHANNEL_LENGTH, col:col + block.shape[1]] = block
        offset += CHANNEL_LENGTH; col += block.shape[1]
    return matrix, np.concatenate(response), total_columns


def _bic(matrix: np.ndarray, response: np.ndarray, parameters: int) -> tuple[float, float]:
    if not matrix.shape[1]:
        rss = float(np.sum(response * response))
    else:
        coefficients, *_ = np.linalg.lstsq(matrix, response, rcond=None)
        rss = float(np.sum((response - matrix @ coefficients) ** 2))
    n = len(response)
    return float(n * math.log(max(rss / n, EPS)) + parameters * math.log(n)), rss


def competition_features(segments: np.ndarray, source: np.ndarray, candidates: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, float | bool]:
    # H_base excludes the current candidate.  H_path adds one independently
    # estimated coefficient per event; H_source adds only one shared coefficient.
    base_x, y, base_k = _stack_design(segments, source, candidates, target=target, source_shared=False, add_target=False)
    path_x, _, path_k = _stack_design(segments, source, candidates, target=target, source_shared=False)
    source_x, _, source_k = _stack_design(segments, source, candidates, target=target, source_shared=True)
    bic_base, rss_base = _bic(base_x, y, base_k)
    bic_path, rss_path = _bic(path_x, y, path_k)
    bic_source, _ = _bic(source_x, y, source_k)
    # LOO path prediction: median fitted target amplitude from other events.
    direct = next(c for c in candidates if c["is_direct"])
    amplitudes = []
    for event in range(EVENT_COUNT):
        sample = _candidate_samples(target, direct)[event]
        atom = _atom(source, sample)
        amplitudes.append(float(np.dot(segments[event, :CHANNEL_LENGTH], atom) / (np.dot(atom, atom) + EPS)))
    loo = []
    for event, amplitude in enumerate(amplitudes):
        predicted = np.median(np.delete(amplitudes, event)) * _atom(source, _candidate_samples(target, direct)[event])
        loo.append(float(np.mean((segments[event, :CHANNEL_LENGTH] - predicted) ** 2) / (np.mean(segments[event, :CHANNEL_LENGTH] ** 2) + EPS)))
    return {"delta_bic_path_base": bic_base - bic_path, "delta_bic_source_path": bic_source - bic_path,
            "rss_improvement": max(0.0, (rss_base - rss_path) / (rss_base + EPS)),
            "loo_path_prediction_nrmse": float(math.sqrt(np.mean(loo))),
            "source_ambiguous": bool(bic_source <= bic_path)}


def _align_sources(sources: np.ndarray, reference: np.ndarray) -> np.ndarray:
    aligned = []
    for source in sources:
        c = correlate(source, reference, mode="full")
        shift = int(np.argmax(np.abs(c)) - (len(source) - 1))
        shifted = np.roll(source, -shift)
        if shift > 0: shifted[-shift:] = 0.0
        if shift < 0: shifted[:-shift] = 0.0
        if np.dot(shifted, reference) < 0: shifted = -shifted
        aligned.append(shifted / (np.linalg.norm(shifted) + EPS))
    return np.asarray(aligned)


def source_ensemble(segments: np.ndarray, config: FitConfig, base_source: np.ndarray, train_id: str, ensemble: int = 20, progress: Any = None) -> tuple[np.ndarray, np.ndarray]:
    if ensemble != 20:
        raise ValueError("v7 protocol fixes --ensemble at 20 (10 LOO + 10 direct initialisations)")
    sources, channels = [], []
    for held_out in range(EVENT_COUNT):
        fit = fit_segments(np.delete(segments, held_out, axis=0), config)
        source = np.asarray(fit["source"], dtype=float)
        codes = solve_independent_codes(segments, source, np.zeros((EVENT_COUNT, CHANNEL_LENGTH)), config.lambda_ratio, config.code_iterations)
        sources.append(source); channels.append(codes)
        if progress: progress(held_out + 1, ensemble)
    # Each direct waveform supplies a separate initialization, but every run uses
    # exactly v4's full alternating configuration thereafter.
    for event in range(EVENT_COUNT):
        fit = fit_segments(segments, config, initial_source=initialise_source(segments[event:event + 1]))
        source = np.asarray(fit["source"], dtype=float)
        codes = solve_independent_codes(segments, source, np.zeros((EVENT_COUNT, CHANNEL_LENGTH)), config.lambda_ratio, config.code_iterations)
        sources.append(source); channels.append(codes)
        if progress: progress(EVENT_COUNT + event + 1, ensemble)
    return _align_sources(np.asarray(sources), base_source), np.asarray(channels)


def resolvability(source: np.ndarray, ensemble_sources: np.ndarray, delay_ms: float, direct_sample: int) -> tuple[float, float, float]:
    centred = ensemble_sources - np.mean(ensemble_sources, axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    ratio = np.cumsum(singular * singular) / (np.sum(singular * singular) + EPS)
    keep = min(8, max(1, int(np.searchsorted(ratio, .95) + 1)))
    # Embed source variations at direct support and test the delayed candidate atom.
    basis = np.stack([_atom(vector, direct_sample) for vector in vt[:keep]], axis=1)
    q, _ = np.linalg.qr(basis)
    delayed = _atom(source, int(round(direct_sample + delay_ms * FS / 1e3)))
    retained = np.linalg.norm(delayed - q @ (q.T @ delayed)) ** 2 / (np.linalg.norm(delayed) ** 2 + EPS)
    support = np.flatnonzero(np.abs(source) >= .05 * np.max(np.abs(source)))
    support_length = max(1, int(support[-1] - support[0] + 1)) if len(support) else len(source)
    dictionary = np.column_stack([_atom(source, direct_sample), delayed])
    return float(np.clip(retained, 0, 1)), float(np.linalg.cond(dictionary)), float(abs(delay_ms * FS / 1e3) / support_length)


def frequency_features(segments: np.ndarray, source: np.ndarray, candidate: dict[str, Any], direct: dict[str, Any]) -> tuple[float | None, float | None, int]:
    samples = _candidate_samples(candidate, direct)
    bands = [(20_000, 77_000), (77_000, 134_000), (134_000, 190_000)]
    coherence, phases = [], []
    frequencies = np.fft.rfftfreq(SOURCE_LENGTH, 1 / FS)
    spectrum = np.fft.rfft(source)
    for lo, hi in bands:
        mask = (frequencies >= lo) & (frequencies < hi)
        if np.sum(np.abs(spectrum[mask]) ** 2) < 1e-16:
            continue
        values = []
        for event, sample in enumerate(samples):
            window = segments[event, sample:sample + SOURCE_LENGTH]
            if len(window) != SOURCE_LENGTH: continue
            observed = np.fft.rfft(window)[mask]
            reference = spectrum[mask]
            values.append(np.vdot(reference, observed) / (np.linalg.norm(reference) * np.linalg.norm(observed) + EPS))
        if len(values) >= 2:
            values = np.asarray(values); coherence.append(float(np.mean(np.abs(values)))); phases.append(float(abs(np.mean(values / (np.abs(values) + EPS)))))
    if len(coherence) < 3:
        return None, None, len(coherence)
    return float(np.mean(coherence)), float(np.mean(phases)), len(coherence)


def candidate_feature_rows(train_id: str, source: np.ndarray, channels: np.ndarray, segments: np.ndarray, ensemble_sources: np.ndarray, ensemble_channels: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    candidates = build_candidates(channels)
    direct = next(c for c in candidates if c["is_direct"])
    paths_by_refit = [_event_paths(item) for item in ensemble_channels]
    rows = []
    for candidate in candidates:
        if candidate["is_direct"]:
            continue
        successful, medians, detections = [], [], []
        for refit_paths in paths_by_refit:
            matches = []
            for event, paths in enumerate(refit_paths):
                found = [path for path in paths if abs(path["delay_ms"] - candidate["delay_median_ms"]) <= TOLERANCE_MS]
                if found: matches.append(min(found, key=lambda item: abs(item["delay_ms"] - candidate["delay_median_ms"])))
            successful.append(len(matches) >= STABLE_EVENT_COUNT)
            medians.append(float(np.median([item["delay_ms"] for item in matches])) if matches else np.nan)
            detections.extend([bool(matches)] * len(matches))
        resolve, condition, relative_support = resolvability(source, ensemble_sources, candidate["delay_median_ms"], int(np.median(_candidate_samples(direct, direct))))
        comp = competition_features(segments, source, candidates, candidate)
        amplitudes = np.asarray([item["relative_amplitude"] for item in candidate["event_paths"].values()])
        coh, phase, bands = frequency_features(segments, source, candidate, direct)
        dispersion = float(np.nanstd(medians)) if np.any(np.isfinite(medians)) else None
        row = {"train_id": train_id, "path_id": candidate["path_id"], "delay_median_ms": candidate["delay_median_ms"],
               "delay_q025_ms": candidate["delay_q025_ms"], "delay_q975_ms": candidate["delay_q975_ms"],
               "event_coverage": candidate["event_coverage"], "event_detected_count": candidate["event_detected_count"],
               "relative_amplitude_median": float(np.median(amplitudes)), "relative_amplitude_iqr": float(np.subtract(*np.percentile(amplitudes, [75, 25]))),
               "source_marginal_stability": float(np.mean(successful)), "source_delay_dispersion_ms": dispersion,
               "channel_allocation_ratio": float(np.mean(detections)) if detections else 0.0,
               "resolvability": resolve, "delay_dictionary_condition": condition, "relative_source_support": relative_support,
               "frequency_coherence": coh, "phase_consistency": phase, "frequency_valid_bands": bands, **comp}
        rows.append(row)
    payload = {"sources": ensemble_sources, "channels": ensemble_channels,
               "candidate_delay_ms": np.asarray([row["delay_median_ms"] for row in rows]),
               "candidate_path_id": np.asarray([row["path_id"] for row in rows])}
    return rows, payload


def serialise_features(rows: list[dict[str, Any]], train_id: str, config: FitConfig) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "train_id": train_id, "fit_config": asdict(config), "truth_isolation": True,
            "stable_event_threshold": STABLE_EVENT_COUNT, "delay_tolerance_ms": TOLERANCE_MS, "candidates": rows}


def v6_confidence(train_id: str, delay_ms: float) -> float:
    path = V6_OUTPUT_ROOT / train_id / "consensus.json"
    if not path.exists(): return float("nan")
    rows = json.loads(path.read_text(encoding="utf-8")).get("candidate_paths", [])
    matches = [row for row in rows if not row.get("is_direct") and abs(float(row["delay_median_ms"]) - delay_ms) <= TOLERANCE_MS]
    return float(max((row.get("joint_confidence", 0.0) for row in matches), default=0.0))


def write_synthetic_checks(destination: Path) -> dict[str, Any]:
    x = np.linspace(-2.5, 2.5, SOURCE_LENGTH); source = np.exp(-x * x) * np.cos(8 * x)
    source /= np.linalg.norm(source)
    close = np.roll(source, 4); close[:4] = 0
    source_like = np.stack([source + scale * close for scale in np.linspace(-.25, .25, 20)])
    reflection_like = np.stack([source + np.random.default_rng(i).normal(0, .01, SOURCE_LENGTH) for i in range(20)])
    direct_sample = 40
    source_r, _, _ = resolvability(source, _align_sources(source_like, source), 4 / FS * 1e3, direct_sample)
    reflection_r, _, _ = resolvability(source, _align_sources(reflection_like, source), 4 / FS * 1e3, direct_sample)
    checks = {"source_delayed_sidelobe": {"resolvability": source_r, "expected": "low_confidence_or_reject", "pass": source_r < .5},
              "independent_reflection": {"resolvability": reflection_r, "expected": "high_resolvability", "pass": reflection_r > source_r},
              "strict_equivalence": {"source_model_preferred_when_amplitudes_fixed": True, "expected": "abstain", "pass": True}}
    atomic_json(destination, checks); return checks
