from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    r"D:\Project_Github\Indo-Pacific-humpback-dolphin\20\full_dataset_experiment\.matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfiltfilt, welch, windows

import batch_joint_bcsc_compare as base


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE_ROOT = PROJECT_ROOT / "09_Locate"
NOTEBOOK_ROOT = PROJECT_ROOT / "20"
OUTPUT_ROOT = NOTEBOOK_ROOT / "full_dataset_experiment"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
ATOM_FIGURES_DIR = FIGURES_DIR / "atoms"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
SENSITIVITY_DIR = OUTPUT_ROOT / "sensitivity_checkpoints"
MANIFEST_PATH = OUTPUT_ROOT / "full_manifest.csv"
REPORT_PATH = OUTPUT_ROOT / "report.html"

FS = 576_000
MAX_CLICKS = 10
BOOTSTRAP_COUNT = 50
GLOBAL_SEED = 20260831
HIGH_CONFIDENCE = 0.90
STRICT_WILSON_LOW = 0.80
SENSITIVITY_COUNT = 100
PSD_NPERSEG = 256

DEFAULT_CONFIG: dict[str, float | int | str] = {
    "name": "default",
    "lambda_ratio": 0.075,
    "reference_atom_us": 85.0,
    "peak_ratio": 0.015,
    "residual_block_ms": 0.10,
    "match_tolerance_ms": 0.03,
    "max_clicks": 10,
}

SENSITIVITY_CONFIGS = [
    DEFAULT_CONFIG,
    {**DEFAULT_CONFIG, "name": "lambda_0.060", "lambda_ratio": 0.060},
    {**DEFAULT_CONFIG, "name": "lambda_0.090", "lambda_ratio": 0.090},
    {**DEFAULT_CONFIG, "name": "atom_50us", "reference_atom_us": 50.0},
    {**DEFAULT_CONFIG, "name": "atom_120us", "reference_atom_us": 120.0},
    {**DEFAULT_CONFIG, "name": "peak_0.010", "peak_ratio": 0.010},
    {**DEFAULT_CONFIG, "name": "peak_0.030", "peak_ratio": 0.030},
    {**DEFAULT_CONFIG, "name": "block_0.05ms", "residual_block_ms": 0.05},
    {**DEFAULT_CONFIG, "name": "block_0.20ms", "residual_block_ms": 0.20},
    {**DEFAULT_CONFIG, "name": "match_0.02ms", "match_tolerance_ms": 0.02},
    {**DEFAULT_CONFIG, "name": "match_0.05ms", "match_tolerance_ms": 0.05},
    {**DEFAULT_CONFIG, "name": "max_clicks_5", "max_clicks": 5},
]


@dataclass(frozen=True)
class TrainSpec:
    index: int
    voyage: str
    pulse_train: str
    path: str
    wav_count: int

    @property
    def key(self) -> str:
        return f"{self.index:03d}_{self.voyage}_{self.pulse_train}"


def config_hash() -> str:
    payload = {
        "default": DEFAULT_CONFIG,
        "bootstrap": BOOTSTRAP_COUNT,
        "atom_range_us": base.ATOM_RANGE_US,
        "bandwidth_hz": [base.MIN_BANDWIDTH_HZ, base.MAX_BANDWIDTH_HZ],
        "lambda_perturbation": base.LAMBDA_PERTURBATION,
        "min_path_distance_ms": base.MIN_PATH_DISTANCE_MS,
        "algorithm_budget": [
            base.REFERENCE_OUTER,
            base.REFERENCE_FISTA,
            base.REFERENCE_DICTIONARY,
            base.BOOTSTRAP_OUTER,
            base.BOOTSTRAP_FISTA,
            base.BOOTSTRAP_DICTIONARY,
        ],
        "schema": 3,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def ensure_directories() -> None:
    for directory in (OUTPUT_ROOT, RESULTS_DIR, FIGURES_DIR, ATOM_FIGURES_DIR, CHECKPOINT_DIR, SENSITIVITY_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_eligible() -> list[TrainSpec]:
    specs: list[TrainSpec] = []
    for voyage_dir in sorted(SOURCE_ROOT.glob("Voyage_Ori_*")):
        if not voyage_dir.is_dir():
            continue
        for train_dir in sorted(voyage_dir.glob("PulseTrain_*")):
            count = len(list(train_dir.rglob("*.wav")))
            if count >= 5:
                specs.append(TrainSpec(len(specs) + 1, voyage_dir.name, train_dir.name, str(train_dir), count))
    return specs


def save_and_validate_manifest(specs: list[TrainSpec]) -> None:
    rows = [asdict(spec) for spec in specs]
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(encoding="utf-8-sig") as handle:
            old = list(csv.DictReader(handle))
        old_keys = [(row["voyage"], row["pulse_train"], int(row["wav_count"])) for row in old]
        new_keys = [(row["voyage"], row["pulse_train"], int(row["wav_count"])) for row in rows]
        if old_keys != new_keys:
            raise RuntimeError("09_Locate数据清单已变化；请检查full_manifest.csv后使用--force-manifest更新。")
    else:
        write_csv(MANIFEST_PATH, rows)


def to_float_audio(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data.astype(np.float64) / max(abs(info.min), info.max)
    return data.astype(np.float64)


def preprocess_click(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    sample_rate, raw_data = wavfile.read(path)
    if sample_rate != FS:
        raise ValueError(f"采样率不是{FS}: {path}")
    raw = to_float_audio(raw_data)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    raw_mean = float(np.mean(raw))
    raw = raw - raw_mean
    raw_peak = float(np.max(np.abs(raw)))
    raw_rms = float(np.sqrt(np.mean(raw * raw)))
    nyquist = sample_rate / 2
    sos = butter(5, [base.LOWCUT_HZ / nyquist, base.HIGHCUT_HZ / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, raw)
    filtered *= windows.tukey(len(filtered), alpha=0.04)
    filtered_peak = float(np.max(np.abs(filtered)))
    filtered_rms = float(np.sqrt(np.mean(filtered * filtered)))
    normalized = filtered / (filtered_peak + np.finfo(float).eps)
    meta = {
        "sample_rate": sample_rate,
        "signal_length": len(normalized),
        "raw_mean": raw_mean,
        "raw_peak": raw_peak,
        "raw_rms": raw_rms,
        "filtered_peak_before_normalization": filtered_peak,
        "filtered_rms_before_normalization": filtered_rms,
        "snr_proxy": base.snr_proxy(normalized),
    }
    return normalized, meta


def click_number(path: Path) -> int:
    digits = "".join(character for character in path.parent.name if character.isdigit())
    return int(digits or 0)


def load_selected_clicks(train_path: str | Path, max_clicks: int = MAX_CLICKS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wav_path in sorted(Path(train_path).rglob("*.wav"), key=click_number):
        signal, meta = preprocess_click(wav_path)
        rows.append({"wav_path": str(wav_path), "click": wav_path.parent.name, "signal": signal, "meta": meta})
    rows.sort(key=lambda row: row["meta"]["snr_proxy"], reverse=True)
    return rows[:max_clicks]


def extract_candidates(
    code: np.ndarray,
    atom_length: int,
    peak_ratio: float,
    min_distance_ms: float = base.MIN_PATH_DISTANCE_MS,
) -> list[dict[str, Any]]:
    magnitude = np.abs(code)
    if magnitude.size == 0 or float(magnitude.max()) <= 1e-14:
        return []
    distance = max(1, int(round(min_distance_ms * 1e-3 * FS)))
    peaks, _ = find_peaks(magnitude, height=peak_ratio * magnitude.max(), distance=distance)
    strongest = int(np.argmax(magnitude))
    if strongest not in peaks:
        peaks = np.sort(np.append(peaks, strongest))
    coefficients = code[peaks]
    direct_local = int(np.argmax(np.abs(coefficients)))
    direct_code_index = int(peaks[direct_local])
    direct_coefficient = float(coefficients[direct_local])
    denominator = abs(direct_coefficient) + 1e-12
    rows = []
    for code_index, coefficient in zip(peaks, coefficients):
        relative_delay_ms = (int(code_index) - direct_code_index) / FS * 1e3
        relative_amplitude = abs(float(coefficient)) / denominator
        relative_signed_amplitude = float(coefficient) / (direct_coefficient + np.sign(direct_coefficient) * 1e-12)
        rows.append({
            "code_index": int(code_index),
            "arrival_sample": int(code_index + atom_length // 2),
            "coefficient": float(coefficient),
            "direct_coefficient": direct_coefficient,
            "relative_delay_ms": float(relative_delay_ms),
            "relative_amplitude": float(relative_amplitude),
            "relative_signed_amplitude": float(relative_signed_amplitude),
            "relative_level_db": float(20 * np.log10(relative_amplitude + 1e-12)),
            "phase_flip_deg": 0.0 if relative_signed_amplitude >= 0 else 180.0,
        })
    return sorted(rows, key=lambda row: row["relative_delay_ms"])


def match_paths(reference: list[dict[str, Any]], candidates: list[dict[str, Any]], tolerance_ms: float) -> list[dict[str, Any] | None]:
    matches: list[dict[str, Any] | None] = [None] * len(reference)
    pairs: list[tuple[float, int, int]] = []
    for i, ref in enumerate(reference):
        for j, candidate in enumerate(candidates):
            difference = abs(float(candidate["relative_delay_ms"]) - float(ref["relative_delay_ms"]))
            if difference <= tolerance_ms:
                pairs.append((difference, i, j))
    used: set[int] = set()
    for _, i, j in sorted(pairs):
        if matches[i] is None and j not in used:
            matches[i] = candidates[j]
            used.add(j)
    return matches


def percentile(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return float(median), float(low), float(high)


def atom_features(atom: np.ndarray) -> dict[str, float]:
    atom = np.asarray(atom, dtype=float)
    peak_index = int(np.argmax(np.abs(atom)))
    n_fft = base.BANDWIDTH_NFFT
    spectrum = np.fft.rfft(atom, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, 1 / FS)
    magnitude = np.abs(spectrum)
    weights = magnitude * magnitude
    weight_sum = float(weights.sum()) + 1e-15
    centroid = float(np.sum(frequencies * weights) / weight_sum)
    rms_bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * weights) / weight_sum))
    bandwidth, center, left, right = base.measure_bandwidth(atom, FS)
    centered_spectrum = spectrum * np.exp(1j * 2 * np.pi * frequencies * peak_index / FS)
    center_index = int(np.argmax(magnitude))
    peak_phase = float(np.angle(centered_spectrum[center_index]))
    mask = magnitude >= magnitude.max() * 0.1
    phase = np.unwrap(np.angle(centered_spectrum[mask]))
    frequency_masked = frequencies[mask]
    if len(phase) >= 3:
        design = np.column_stack([frequency_masked - center, np.ones_like(frequency_masked)])
        coefficient, *_ = np.linalg.lstsq(design, phase, rcond=None)
        residual = phase - design @ coefficient
        phase_residual_rms = float(np.sqrt(np.mean(residual * residual)))
    else:
        phase_residual_rms = math.nan
    rms = float(np.sqrt(np.mean(atom * atom)))
    normalized_peak = float(np.max(np.abs(atom)))
    return {
        "atom_length_us": len(atom) / FS * 1e6,
        "peak_frequency_khz": center / 1e3,
        "spectral_centroid_khz": centroid / 1e3,
        "rms_bandwidth_khz": rms_bandwidth / 1e3,
        "bandwidth_3db_khz": bandwidth / 1e3,
        "left_3db_khz": left / 1e3,
        "right_3db_khz": right / 1e3,
        "normalized_peak_amplitude": normalized_peak,
        "normalized_rms_amplitude": rms,
        "crest_factor": normalized_peak / (rms + 1e-15),
        "peak_phase_deg_centered": float(np.degrees(peak_phase)),
        "phase_residual_rms_rad": phase_residual_rms,
    }


def residual_profile(residual: np.ndarray) -> dict[str, Any]:
    frequencies, psd = welch(residual, fs=FS, nperseg=min(PSD_NPERSEG, len(residual)))
    return {
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "residual_psd_frequency_hz": frequencies.tolist(),
        "residual_psd": psd.tolist(),
    }


def method_analysis(
    signals: list[np.ndarray],
    click_names: list[str],
    config: dict[str, float | int | str],
    bootstrap_count: int,
    rng: np.random.Generator,
    all_clicks: bool,
) -> dict[str, Any]:
    reference_length = base.atom_length_from_us(float(config["reference_atom_us"]), FS)
    atom, codes, reconstructions = base.bcsc(
        signals,
        reference_length,
        float(config["lambda_ratio"]),
        FS,
        base.REFERENCE_OUTER,
        base.REFERENCE_FISTA,
        base.REFERENCE_DICTIONARY,
    )
    residuals = [signal - reconstruction for signal, reconstruction in zip(signals, reconstructions)]
    explained = [1 - np.sum(residual * residual) / (np.sum(signal * signal) + 1e-12) for signal, residual in zip(signals, residuals)]
    indices = list(range(len(signals))) if all_clicks else [0]
    references: dict[int, list[dict[str, Any]]] = {}
    records: dict[int, list[list[dict[str, Any]]]] = {}
    tolerance = float(config["match_tolerance_ms"])
    peak_ratio = float(config["peak_ratio"])
    for i in indices:
        candidates = extract_candidates(codes[i], reference_length, peak_ratio)
        paths = [row for row in candidates if row["relative_delay_ms"] >= -tolerance]
        for path_index, row in enumerate(paths):
            row["path"] = "direct" if abs(row["relative_delay_ms"]) < 1e-12 else f"multipath_{path_index}"
        references[i] = paths
        records[i] = [[] for _ in paths]

    block_length = max(2, int(round(float(config["residual_block_ms"]) * 1e-3 * FS)))
    bootstrap_bandwidths: list[float] = []
    bootstrap_lengths: list[float] = []
    for _ in range(bootstrap_count):
        length_us = float(rng.uniform(*base.ATOM_RANGE_US))
        length = base.atom_length_from_us(length_us, FS)
        lambda_ratio = float(config["lambda_ratio"]) * rng.uniform(1 - base.LAMBDA_PERTURBATION, 1 + base.LAMBDA_PERTURBATION)
        synthetic = [
            reconstruction + base.circular_block_bootstrap(residual, block_length, rng)
            for reconstruction, residual in zip(reconstructions, residuals)
        ]
        atom_b, codes_b, _ = base.bcsc(
            synthetic,
            length,
            lambda_ratio,
            FS,
            base.BOOTSTRAP_OUTER,
            base.BOOTSTRAP_FISTA,
            base.BOOTSTRAP_DICTIONARY,
        )
        bandwidth_b = base.measure_bandwidth(atom_b, FS)[0]
        if not base.MIN_BANDWIDTH_HZ <= bandwidth_b <= base.MAX_BANDWIDTH_HZ:
            raise RuntimeError(f"Bootstrap原子带宽越界: {bandwidth_b}")
        bootstrap_bandwidths.append(float(bandwidth_b / 1e3))
        bootstrap_lengths.append(float(length / FS * 1e6))
        for i in indices:
            candidates_b = extract_candidates(codes_b[i], length, peak_ratio)
            for path_index, match in enumerate(match_paths(references[i], candidates_b, tolerance)):
                if match is not None:
                    records[i][path_index].append(match)

    click_results: list[dict[str, Any]] = []
    for i in indices:
        paths: list[dict[str, Any]] = []
        for reference, repetitions in zip(references[i], records[i]):
            detected = len(repetitions)
            low, high = base.wilson(detected, bootstrap_count)
            delay_median, delay_low, delay_high = percentile([row["relative_delay_ms"] for row in repetitions])
            amplitude_median, amplitude_low, amplitude_high = percentile([row["relative_amplitude"] for row in repetitions])
            signed_median, signed_low, signed_high = percentile([row["relative_signed_amplitude"] for row in repetitions])
            paths.append({
                **reference,
                "detected_runs": detected,
                "bootstrap_total": bootstrap_count,
                "confidence": detected / bootstrap_count,
                "confidence_low": low,
                "confidence_high": high,
                "confidence_class": "strict_high" if low >= STRICT_WILSON_LOW else ("high" if detected / bootstrap_count >= HIGH_CONFIDENCE else "ordinary"),
                "delay_median_ms": delay_median,
                "delay_low_ms": delay_low,
                "delay_high_ms": delay_high,
                "amplitude_median": amplitude_median,
                "amplitude_low": amplitude_low,
                "amplitude_high": amplitude_high,
                "signed_amplitude_median": signed_median,
                "signed_amplitude_low": signed_low,
                "signed_amplitude_high": signed_high,
            })
        profile = residual_profile(residuals[i])
        click_results.append({
            "click": click_names[i],
            "click_position": i,
            "explained_variance": float(explained[i]),
            **profile,
            "paths": paths,
        })
    features = atom_features(atom)
    return {
        "config": config,
        "atom": atom.tolist(),
        **features,
        "mean_explained_variance": float(np.mean(explained)),
        "bootstrap_atom_length_min_us": float(np.min(bootstrap_lengths)),
        "bootstrap_atom_length_max_us": float(np.max(bootstrap_lengths)),
        "bootstrap_bandwidth_min_khz": float(np.min(bootstrap_bandwidths)),
        "bootstrap_bandwidth_max_khz": float(np.max(bootstrap_bandwidths)),
        "click_results": click_results,
    }


def atom_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return base.atom_similarity(np.asarray(a), np.asarray(b))


def process_train(spec_dict: dict[str, Any], bootstrap_count: int, expected_hash: str) -> dict[str, Any]:
    spec = TrainSpec(**spec_dict)
    started = time.perf_counter()
    selected = load_selected_clicks(spec.path, MAX_CLICKS)
    signals = [row["signal"] for row in selected]
    names = [row["click"] for row in selected]
    seed = GLOBAL_SEED + spec.index * 100_003
    joint = method_analysis(signals, names, DEFAULT_CONFIG, bootstrap_count, np.random.default_rng(seed + 2), True)
    single = method_analysis([signals[0]], [names[0]], DEFAULT_CONFIG, bootstrap_count, np.random.default_rng(seed + 1), True)
    similarity = atom_similarity(np.asarray(single["atom"]), np.asarray(joint["atom"]))
    single["atom_joint_similarity"] = similarity
    joint["atom_single_similarity"] = similarity
    click_metadata = []
    for position, row in enumerate(selected):
        click_metadata.append({
            "click": row["click"],
            "click_position": position,
            "wav_path": row["wav_path"],
            **row["meta"],
        })
    return {
        "config_hash": expected_hash,
        "index": spec.index,
        "key": spec.key,
        "voyage": spec.voyage,
        "pulse_train": spec.pulse_train,
        "path": spec.path,
        "available_clicks": spec.wav_count,
        "used_clicks": len(selected),
        "anchor_click": names[0],
        "click_metadata": click_metadata,
        "single": single,
        "joint": joint,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_main(specs: list[TrainSpec], bootstrap_count: int, workers: int, force: bool) -> list[dict[str, Any]]:
    expected_hash = config_hash()
    completed: dict[int, dict[str, Any]] = {}
    pending: list[TrainSpec] = []
    for spec in specs:
        checkpoint = CHECKPOINT_DIR / f"{spec.key}.json"
        if checkpoint.exists() and not force:
            result = json.loads(checkpoint.read_text(encoding="utf-8"))
            if result.get("config_hash") == expected_hash:
                completed[spec.index] = result
                continue
        pending.append(spec)
    print(f"[总体进度] 已完成{len(completed)}/{len(specs)}；待运行{len(pending)}；每组单/联合各{bootstrap_count}次Bootstrap", flush=True)
    failures: list[dict[str, Any]] = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_train, asdict(spec), bootstrap_count, expected_hash): spec for spec in pending}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                    completed[spec.index] = result
                    atomic_json(CHECKPOINT_DIR / f"{spec.key}.json", result)
                    status = f"{result['elapsed_seconds']:.1f}s"
                except Exception as error:
                    failures.append({**asdict(spec), "error": repr(error)})
                    status = f"失败: {error}"
                done = len(completed) + len(failures)
                print(f"[总体进度] {done}/{len(specs)} ({100*done/len(specs):.1f}%) {spec.voyage}/{spec.pulse_train} {status}", flush=True)
    write_csv(RESULTS_DIR / "failures.csv", failures)
    results = [completed[spec.index] for spec in specs if spec.index in completed]
    atomic_json(RESULTS_DIR / "all_results.json", results)
    if failures:
        print(f"[警告] {len(failures)}组失败，已写入failures.csv；再次运行会自动重试。", flush=True)
    return results


def stratified_sensitivity_specs(specs: list[TrainSpec], count: int) -> list[TrainSpec]:
    rng = np.random.default_rng(GLOBAL_SEED + 77)
    strata: dict[tuple[str, str], list[TrainSpec]] = {}
    for spec in specs:
        click_bin = "5-10" if spec.wav_count <= 10 else ("11-30" if spec.wav_count <= 30 else ">30")
        strata.setdefault((spec.voyage, click_bin), []).append(spec)
    selected: list[TrainSpec] = []
    keys = sorted(strata)
    ideal = {key: count * len(strata[key]) / len(specs) for key in keys}
    allocation = {key: min(len(strata[key]), int(math.floor(ideal[key]))) for key in keys}
    remaining = count - sum(allocation.values())
    for key in sorted(keys, key=lambda item: ideal[item] - allocation[item], reverse=True):
        if remaining <= 0:
            break
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            remaining -= 1
    for key in keys:
        group = strata[key]
        chosen = rng.choice(len(group), size=allocation[key], replace=False)
        selected.extend(group[int(i)] for i in chosen)
    return sorted(selected, key=lambda spec: spec.index)


def high_path_signature(method: dict[str, Any]) -> list[tuple[str, float]]:
    signature = []
    for click in method["click_results"]:
        for path in click["paths"]:
            if path["path"] != "direct" and path["confidence"] >= HIGH_CONFIDENCE:
                signature.append((click["click"], float(path["relative_delay_ms"])))
    return signature


def signature_jaccard(reference: list[tuple[str, float]], candidate: list[tuple[str, float]], tolerance: float = 0.03) -> float:
    pairs: list[tuple[float, int, int]] = []
    for i, (click_a, delay_a) in enumerate(reference):
        for j, (click_b, delay_b) in enumerate(candidate):
            if click_a == click_b and abs(delay_a - delay_b) <= tolerance:
                pairs.append((abs(delay_a - delay_b), i, j))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = 0
    for _, i, j in sorted(pairs):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched += 1
    union = len(reference) + len(candidate) - matched
    return 1.0 if union == 0 else matched / union


def sensitivity_train(spec_dict: dict[str, Any], bootstrap_count: int) -> dict[str, Any]:
    spec = TrainSpec(**spec_dict)
    selected_all = load_selected_clicks(spec.path, MAX_CLICKS)
    outputs = []
    for config_index, config in enumerate(SENSITIVITY_CONFIGS):
        selected = selected_all[: int(config["max_clicks"])]
        signals = [row["signal"] for row in selected]
        names = [row["click"] for row in selected]
        started = time.perf_counter()
        method = method_analysis(
            signals,
            names,
            config,
            bootstrap_count,
            np.random.default_rng(GLOBAL_SEED + spec.index * 100_003 + 10_000 + config_index),
            True,
        )
        confidences = [
            path["confidence"]
            for click in method["click_results"]
            for path in click["paths"]
            if path["path"] != "direct"
        ]
        strict_count = sum(
            path["confidence_low"] >= STRICT_WILSON_LOW
            for click in method["click_results"]
            for path in click["paths"]
            if path["path"] != "direct"
        )
        outputs.append({
            "config": config,
            "atom": method["atom"],
            "atom_features": {key: method[key] for key in atom_features(np.asarray(method["atom"]))},
            "mean_explained_variance": method["mean_explained_variance"],
            "high_count": len(high_path_signature(method)),
            "strict_count": int(strict_count),
            "mean_path_confidence": float(np.mean(confidences)) if confidences else math.nan,
            "high_signature": high_path_signature(method),
            "elapsed_seconds": time.perf_counter() - started,
        })
    baseline = outputs[0]
    baseline_atom = np.asarray(baseline["atom"])
    for row in outputs:
        row["path_jaccard_vs_default"] = signature_jaccard(baseline["high_signature"], row["high_signature"])
        row["atom_similarity_vs_default"] = atom_similarity(baseline_atom, np.asarray(row["atom"]))
        row["high_count_relative_change"] = (row["high_count"] - baseline["high_count"]) / max(1, baseline["high_count"])
        row.pop("atom")
        row.pop("high_signature")
    return {"index": spec.index, "voyage": spec.voyage, "pulse_train": spec.pulse_train, "configs": outputs}


def run_sensitivity(specs: list[TrainSpec], bootstrap_count: int, workers: int, force: bool) -> list[dict[str, Any]]:
    subset = stratified_sensitivity_specs(specs, SENSITIVITY_COUNT)
    write_csv(OUTPUT_ROOT / "sensitivity_manifest.csv", [asdict(spec) for spec in subset])
    completed: dict[int, dict[str, Any]] = {}
    pending = []
    for spec in subset:
        path = SENSITIVITY_DIR / f"{spec.key}.json"
        if path.exists() and not force:
            completed[spec.index] = json.loads(path.read_text(encoding="utf-8"))
        else:
            pending.append(spec)
    print(f"[敏感性进度] 已完成{len(completed)}/{len(subset)}；每组{len(SENSITIVITY_CONFIGS)}种配置×{bootstrap_count}次Bootstrap", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(sensitivity_train, asdict(spec), bootstrap_count): spec for spec in pending}
        for future in as_completed(futures):
            spec = futures[future]
            result = future.result()
            completed[spec.index] = result
            atomic_json(SENSITIVITY_DIR / f"{spec.key}.json", result)
            print(f"[敏感性进度] {len(completed)}/{len(subset)} ({100*len(completed)/len(subset):.1f}%) {spec.voyage}/{spec.pulse_train}", flush=True)
    results = [completed[spec.index] for spec in subset]
    atomic_json(RESULTS_DIR / "sensitivity_all_results.json", results)
    return results


def wrap_degrees(values: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def quantiles(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {key: math.nan for key in ("q2_5", "q25", "median", "q75", "q97_5", "mean", "std")}
    q = np.percentile(finite, [2.5, 25, 50, 75, 97.5])
    return {
        "q2_5": float(q[0]), "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "q97_5": float(q[4]),
        "mean": float(np.mean(finite)), "std": float(np.std(finite)),
    }


def circular_summary(degrees: list[float]) -> dict[str, float]:
    radians = np.radians(np.asarray(degrees, dtype=float))
    if not len(radians):
        return {"circular_mean_deg": math.nan, "resultant_length": math.nan, "circular_std_deg": math.nan}
    resultant = np.mean(np.exp(1j * radians))
    length = float(abs(resultant))
    circular_std = float(np.degrees(np.sqrt(max(0.0, -2 * np.log(max(length, 1e-12))))))
    return {"circular_mean_deg": float(np.degrees(np.angle(resultant))), "resultant_length": length, "circular_std_deg": circular_std}


def aggregate(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    atom_arrays: list[np.ndarray] = []
    atom_ids: list[str] = []
    residual_psds: list[np.ndarray] = []
    residual_ids: list[str] = []
    residual_freq = None
    for result in results:
        metadata = {row["click"]: row for row in result["click_metadata"]}
        single = result["single"]
        joint = result["joint"]
        train_rows.append({
            "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
            "available_clicks": result["available_clicks"], "used_clicks": result["used_clicks"], "anchor_click": result["anchor_click"],
            "single_explained_variance": single["mean_explained_variance"], "joint_explained_variance": joint["mean_explained_variance"],
            "atom_similarity": joint["atom_single_similarity"], "elapsed_seconds": result["elapsed_seconds"],
        })
        for method_name, method in (("single", single), ("joint", joint)):
            atom_id = f"{result['voyage']}__{result['pulse_train']}__{method_name}"
            atom_ids.append(atom_id)
            atom_arrays.append(np.asarray(method["atom"], dtype=float))
            atom_rows.append({
                "atom_id": atom_id, "method": method_name, "voyage": result["voyage"], "pulse_train": result["pulse_train"],
                **{key: method[key] for key in atom_features(np.asarray(method["atom"]))},
                "mean_explained_variance": method["mean_explained_variance"],
                "atom_similarity_single_joint": joint["atom_single_similarity"],
            })
            for click in method["click_results"]:
                click_meta = metadata[click["click"]]
                residual_id = f"{atom_id}__{click['click']}"
                if method_name == "joint":
                    residual_ids.append(residual_id)
                    residual_psds.append(np.asarray(click["residual_psd"], dtype=float))
                    residual_freq = np.asarray(click["residual_psd_frequency_hz"], dtype=float)
                for path in click["paths"]:
                    frequency_hz = method["peak_frequency_khz"] * 1e3
                    effective_phase = float(wrap_degrees(
                        method["peak_phase_deg_centered"]
                        + path["phase_flip_deg"]
                        - 360.0 * frequency_hz * path["relative_delay_ms"] * 1e-3
                    ))
                    row = {
                        "method": method_name,
                        "voyage": result["voyage"], "pulse_train": result["pulse_train"], "click": click["click"],
                        "wav_path": click_meta["wav_path"], "path": path["path"], "atom_id": atom_id,
                        "relative_delay_ms": path["relative_delay_ms"], "relative_amplitude": path["relative_amplitude"],
                        "relative_signed_amplitude": path["relative_signed_amplitude"], "relative_level_db": path["relative_level_db"],
                        "coefficient": path["coefficient"], "direct_coefficient": path["direct_coefficient"],
                        "phase_flip_deg": path["phase_flip_deg"], "effective_phase_at_atom_peak_deg": effective_phase,
                        "detected_runs": path["detected_runs"], "bootstrap_total": path["bootstrap_total"],
                        "confidence": path["confidence"], "confidence_low": path["confidence_low"], "confidence_high": path["confidence_high"],
                        "confidence_class": path["confidence_class"],
                        "delay_median_ms": path["delay_median_ms"], "delay_low_ms": path["delay_low_ms"], "delay_high_ms": path["delay_high_ms"],
                        "amplitude_median": path["amplitude_median"], "amplitude_low": path["amplitude_low"], "amplitude_high": path["amplitude_high"],
                        "signed_amplitude_median": path["signed_amplitude_median"],
                        "atom_peak_frequency_khz": method["peak_frequency_khz"],
                        "atom_spectral_centroid_khz": method["spectral_centroid_khz"],
                        "atom_bandwidth_3db_khz": method["bandwidth_3db_khz"],
                        "atom_normalized_peak_amplitude": method["normalized_peak_amplitude"],
                        "atom_crest_factor": method["crest_factor"],
                        "atom_peak_phase_deg_centered": method["peak_phase_deg_centered"],
                        "filtered_peak_before_normalization": click_meta["filtered_peak_before_normalization"],
                        "raw_peak": click_meta["raw_peak"], "signal_length": click_meta["signal_length"],
                        "explained_variance": click["explained_variance"], "residual_rms": click["residual_rms"],
                    }
                    channel_rows.append(row)
                    if method_name == "single" or click["click"] == result["anchor_click"]:
                        anchor_rows.append(row)
    max_length = max(len(atom) for atom in atom_arrays)
    atom_matrix = np.full((len(atom_arrays), max_length), np.nan)
    for i, atom in enumerate(atom_arrays):
        atom_matrix[i, : len(atom)] = atom
    np.savez_compressed(RESULTS_DIR / "atoms.npz", atom_ids=np.asarray(atom_ids), atoms=atom_matrix, sample_rate=FS)
    if residual_psds:
        np.savez_compressed(
            RESULTS_DIR / "joint_residual_psd.npz",
            residual_ids=np.asarray(residual_ids),
            frequencies_hz=residual_freq,
            psd=np.vstack(residual_psds),
        )
    high_rows = [row for row in channel_rows if row["method"] == "joint" and row["path"] != "direct" and row["confidence"] >= HIGH_CONFIDENCE]
    return train_rows, channel_rows, atom_rows, high_rows


def synthesis_statistics(high_rows: list[dict[str, Any]], atom_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unique_atoms = {row["atom_id"]: row for row in atom_rows if row["method"] == "joint" and any(path["atom_id"] == row["atom_id"] for path in high_rows)}
    phase_values = [float(row["effective_phase_at_atom_peak_deg"]) for row in high_rows]
    delays = [float(row["relative_delay_ms"]) for row in high_rows]
    levels = [float(row["relative_level_db"]) for row in high_rows]
    correlation = math.nan
    if len(delays) >= 2 and np.std(delays) > 0 and np.std(levels) > 0:
        correlation = float(np.corrcoef(delays, levels)[0, 1])
    return {
        "high_confidence_channel_count": len(high_rows),
        "strict_high_confidence_channel_count": sum(row["confidence_low"] >= STRICT_WILSON_LOW for row in high_rows),
        "unique_joint_atom_count": len(unique_atoms),
        "frequency": {
            "atom_peak_frequency_khz": quantiles([float(row["peak_frequency_khz"]) for row in unique_atoms.values()]),
            "atom_spectral_centroid_khz": quantiles([float(row["spectral_centroid_khz"]) for row in unique_atoms.values()]),
            "atom_bandwidth_3db_khz": quantiles([float(row["bandwidth_3db_khz"]) for row in unique_atoms.values()]),
        },
        "amplitude": {
            "atom_normalized_peak": quantiles([float(row["normalized_peak_amplitude"]) for row in unique_atoms.values()]),
            "atom_crest_factor": quantiles([float(row["crest_factor"]) for row in unique_atoms.values()]),
            "channel_relative_amplitude": quantiles([float(row["relative_amplitude"]) for row in high_rows]),
            "channel_relative_level_db": quantiles(levels),
            "filtered_digital_peak": quantiles([float(row["filtered_peak_before_normalization"]) for row in high_rows]),
            "delay_level_correlation": correlation,
        },
        "phase": {
            "atom_peak_phase_centered_deg": circular_summary([float(row["peak_phase_deg_centered"]) for row in unique_atoms.values()]),
            "effective_path_phase_at_atom_peak_deg": circular_summary(phase_values),
            "polarity_inversion_fraction": float(np.mean([row["relative_signed_amplitude"] < 0 for row in high_rows])) if high_rows else math.nan,
            "phase_identifiability": "当前实值BCSC只能可靠区分路径系数的0/180度极性；任意频率相关相位不可与原子形状唯一分离。",
        },
    }


def aggregate_sensitivity(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for config in result["configs"]:
            rows.append({
                "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
                "config": config["config"]["name"],
                "lambda_ratio": config["config"]["lambda_ratio"], "reference_atom_us": config["config"]["reference_atom_us"],
                "peak_ratio": config["config"]["peak_ratio"], "residual_block_ms": config["config"]["residual_block_ms"],
                "match_tolerance_ms": config["config"]["match_tolerance_ms"], "max_clicks": config["config"]["max_clicks"],
                "high_count": config["high_count"], "strict_count": config["strict_count"],
                "mean_path_confidence": config["mean_path_confidence"], "mean_explained_variance": config["mean_explained_variance"],
                "path_jaccard_vs_default": config["path_jaccard_vs_default"],
                "atom_similarity_vs_default": config["atom_similarity_vs_default"],
                "high_count_relative_change": config["high_count_relative_change"], "elapsed_seconds": config["elapsed_seconds"],
                **{f"atom_{key}": value for key, value in config["atom_features"].items()},
            })
    return rows


def parameter_decisions(sensitivity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name in [config["name"] for config in SENSITIVITY_CONFIGS if config["name"] != "default"]:
        subset = [row for row in sensitivity_rows if row["config"] == name]
        jaccard = float(np.nanmedian([row["path_jaccard_vs_default"] for row in subset]))
        atom_similarity_value = float(np.nanmedian([row["atom_similarity_vs_default"] for row in subset]))
        count_change = float(np.nanmedian(np.abs([row["high_count_relative_change"] for row in subset])))
        explained_delta = []
        baseline_map = {
            (row["index"], row["pulse_train"]): row["mean_explained_variance"]
            for row in sensitivity_rows if row["config"] == "default"
        }
        for row in subset:
            explained_delta.append(abs(row["mean_explained_variance"] - baseline_map[(row["index"], row["pulse_train"])]))
        explained_median = float(np.nanmedian(explained_delta))
        if jaccard < 0.60 or count_change > 0.25:
            recommendation = "需要调整或检查模型假设"
        elif jaccard < 0.80 or count_change > 0.10 or atom_similarity_value < 0.90 or explained_median > 0.02:
            recommendation = "建议对敏感样本人工复核"
        else:
            recommendation = "无需常规人工调整"
        rows.append({
            "variant": name, "median_path_jaccard": jaccard, "median_abs_high_count_change": count_change,
            "median_atom_similarity": atom_similarity_value, "median_abs_explained_variance_change": explained_median,
            "recommendation": recommendation,
        })
    return rows


def plot_atom_comparison(result: dict[str, Any]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    for column, (name, method, color) in enumerate((("单click", result["single"], "#A23B72"), ("多click联合", result["joint"], "#188977"))):
        atom = np.asarray(method["atom"])
        peak = int(np.argmax(np.abs(atom)))
        time_us = (np.arange(len(atom)) - peak) / FS * 1e6
        axes[0, column].plot(time_us, atom, color=color, lw=1.3)
        axes[0, column].axvline(0, color="#777", lw=0.7, ls="--")
        axes[0, column].set(title=f"{name}原子", xlabel="相对主峰时间 / μs", ylabel="单位能量幅值")
        spectrum = np.abs(np.fft.rfft(atom, n=base.BANDWIDTH_NFFT))
        frequency = np.fft.rfftfreq(base.BANDWIDTH_NFFT, 1 / FS) / 1e3
        db = 20 * np.log10(spectrum / (spectrum.max() + 1e-15) + 1e-12)
        axes[1, column].plot(frequency, db, color=color, lw=1.3)
        axes[1, column].axhline(-3, color="#777", lw=0.7, ls="--")
        axes[1, column].set(xlim=(0, 200), ylim=(-60, 2), xlabel="频率 / kHz", ylabel="归一化幅度 / dB")
    fig.suptitle(f"{result['voyage']} / {result['pulse_train']} | 原子相似度={result['joint']['atom_single_similarity']:.3f}")
    filename = f"{result['key']}_atom_comparison.png"
    fig.savefig(ATOM_FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
    return filename


def make_summary_figures(train_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]], atom_rows: list[dict[str, Any]], sensitivity_rows: list[dict[str, Any]]) -> None:
    joint_atoms = [row for row in atom_rows if row["method"] == "joint"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist([row["peak_frequency_khz"] for row in joint_atoms], bins=30, color="#188977", alpha=0.85)
    axes[0, 0].set(title="联合原子峰值频率", xlabel="kHz", ylabel="原子数")
    axes[0, 1].hist([row["bandwidth_3db_khz"] for row in joint_atoms], bins=30, color="#427AA1", alpha=0.85)
    axes[0, 1].set(title="联合原子−3 dB带宽", xlabel="kHz", ylabel="原子数")
    axes[1, 0].scatter([row["relative_delay_ms"] for row in high_rows], [row["relative_level_db"] for row in high_rows], s=9, alpha=0.35, color="#A23B72")
    axes[1, 0].set(title="高置信度路径：时延—相对幅值", xlabel="相对时延 / ms", ylabel="相对声级 / dB")
    axes[1, 1].hist([row["effective_phase_at_atom_peak_deg"] for row in high_rows], bins=np.linspace(-180, 180, 37), color="#D98E04", alpha=0.85)
    axes[1, 1].set(title="高置信度路径有效相位", xlabel="相位 / °", ylabel="路径数")
    fig.savefig(FIGURES_DIR / "atom_channel_variation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].scatter([row["single_explained_variance"] for row in train_rows], [row["joint_explained_variance"] for row in train_rows], s=13, alpha=0.5)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="单click解释率", ylabel="联合解释率", title="锚点重建解释率")
    axes[1].hist([row["atom_similarity"] for row in train_rows], bins=30, color="#188977")
    axes[1].set(xlabel="余弦相似度", ylabel="PulseTrain数", title="单/联合原子相似度")
    by_voyage: dict[str, int] = {}
    for row in high_rows:
        by_voyage[row["voyage"]] = by_voyage.get(row["voyage"], 0) + 1
    axes[2].bar(range(len(by_voyage)), list(by_voyage.values()), color="#427AA1")
    axes[2].set_xticks(range(len(by_voyage)), [key.replace("Voyage_Ori_", "") for key in by_voyage], rotation=70)
    axes[2].set(title="各Voyage高置信度多径", ylabel="路径数")
    fig.savefig(FIGURES_DIR / "full_dataset_summary.png", dpi=160)
    plt.close(fig)

    if sensitivity_rows:
        decisions = parameter_decisions(sensitivity_rows)
        labels = [row["variant"] for row in decisions]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        axes[0].barh(labels, [row["median_path_jaccard"] for row in decisions], color="#188977")
        axes[0].axvline(0.8, color="#A23B72", ls="--")
        axes[0].set(xlim=(0, 1), title="路径Jaccard（越高越稳定）")
        axes[1].barh(labels, [row["median_abs_high_count_change"] for row in decisions], color="#D98E04")
        axes[1].axvline(0.1, color="#A23B72", ls="--")
        axes[1].set(title="高置信度数量相对变化")
        axes[2].barh(labels, [row["median_atom_similarity"] for row in decisions], color="#427AA1")
        axes[2].axvline(0.9, color="#A23B72", ls="--")
        axes[2].set(xlim=(0, 1), title="原子相似度")
        fig.savefig(FIGURES_DIR / "parameter_sensitivity.png", dpi=160)
        plt.close(fig)


def format_quantile(summary: dict[str, float], unit: str = "") -> str:
    return f"中位数 {summary['median']:.3g}{unit}，IQR {summary['q25']:.3g}–{summary['q75']:.3g}{unit}，95%范围 {summary['q2_5']:.3g}–{summary['q97_5']:.3g}{unit}"


def write_synthesis_recipe(stats: dict[str, Any]) -> None:
    text = f"""# 基于原子—信道对的接收click合成措施

## 生成公式

对一个经验原子—信道对，合成预处理域接收信号：

`y_syn[n] = A_click * sum_k a_k d[n - tau_k] + r[n]`

- `d`从`atoms.npz`读取，保持与对应`atom_id`绑定；
- `tau_k`由`relative_delay_ms`换算为采样点；
- `a_k`使用`coefficient`，或将直达径设为1并使用`relative_signed_amplitude`；
- `A_click`使用`filtered_peak_before_normalization`恢复数字幅值；没有水听器灵敏度与增益标定时不能换算成Pa或dB re 1 μPa；
- `r[n]`按`joint_residual_psd.npz`中的同类残差PSD生成随机相位有色噪声，并按记录的`residual_rms`缩放。

## 具体措施

1. 优先整组重采样真实的`atom_id + 同一click全部路径`，不要独立抽取频率、幅值、相位和时延；这样能保留它们的相关性。
2. 只使用`high_confidence_channels.csv`中的高置信度非直达径，直达径始终保留；严格仿真可只用`confidence_class=strict_high`。
3. 原子频率变化通过直接抽取经验原子实现；需要连续生成时，对峰值对齐后的原子做PCA，再对生成原子重新执行零均值、单位L2范数和6–72 kHz硬投影。
4. 原子自身幅值是单位能量形状量，真实幅值由路径系数和`filtered_peak_before_normalization`共同给出，不能只缩放原子峰值。
5. 当前实值BCSC的路径相位只支持由系数符号表达的0°/180°极性。有效相位还必须加入传播时延项`-2*pi*f*tau`；若需要任意频率相关相位，应升级为复值/解析信号BCSC后重新估计，不能从当前结果中臆造。
6. 合成后重新计算峰值频率、−3 dB带宽、路径时延、相对幅值和SNR，并与报告中的2.5%–97.5%经验范围比较；越界样本拒绝或重新采样。

当前高置信度多径数：{stats['high_confidence_channel_count']}；严格高置信度多径数：{stats['strict_high_confidence_channel_count']}。
"""
    (OUTPUT_ROOT / "synthesis_recipe.md").write_text(text, encoding="utf-8")


def make_report(results: list[dict[str, Any]], sensitivity_results: list[dict[str, Any]]) -> None:
    train_rows, channel_rows, atom_rows, high_rows = aggregate(results)
    write_csv(RESULTS_DIR / "pulsetrain_summary.csv", train_rows)
    write_csv(RESULTS_DIR / "all_channels.csv", channel_rows)
    write_csv(RESULTS_DIR / "high_confidence_channels.csv", high_rows)
    write_csv(RESULTS_DIR / "atoms.csv", atom_rows)
    sensitivity_rows = aggregate_sensitivity(sensitivity_results) if sensitivity_results else []
    write_csv(RESULTS_DIR / "sensitivity_results.csv", sensitivity_rows)
    decisions = parameter_decisions(sensitivity_rows) if sensitivity_rows else []
    write_csv(RESULTS_DIR / "parameter_adjustment_decisions.csv", decisions)
    stats = synthesis_statistics(high_rows, atom_rows)
    atomic_json(RESULTS_DIR / "atom_channel_synthesis_statistics.json", stats)
    write_synthesis_recipe(stats)
    print("[报告进度] 生成总体图", flush=True)
    make_summary_figures(train_rows, high_rows, atom_rows, sensitivity_rows)
    print(f"[报告进度] 生成{len(results)}组原子时频图", flush=True)
    figures = []
    for index, result in enumerate(results, 1):
        figures.append((result, plot_atom_comparison(result)))
        if index % 25 == 0 or index == len(results):
            print(f"[报告进度] 原子图 {index}/{len(results)} ({100*index/len(results):.1f}%)", flush=True)

    frequency = stats["frequency"]
    amplitude = stats["amplitude"]
    phase = stats["phase"]
    decision_html = "<p>敏感性实验尚未完成。</p>" if not decisions else "<table><thead><tr><th>参数变体</th><th>路径Jaccard</th><th>高置信度数量变化</th><th>原子相似度</th><th>建议</th></tr></thead><tbody>" + "".join(
        f"<tr><td>{html.escape(row['variant'])}</td><td>{row['median_path_jaccard']:.3f}</td><td>{row['median_abs_high_count_change']:.1%}</td><td>{row['median_atom_similarity']:.3f}</td><td>{row['recommendation']}</td></tr>"
        for row in decisions
    ) + "</tbody></table>"
    gallery = "".join(
        f"<details><summary>{html.escape(result['voyage'])} / {html.escape(result['pulse_train'])}</summary><img loading='lazy' src='figures/atoms/{filename}'></details>"
        for result, filename in figures
    )
    complete = len(results) == 718
    report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>全量BCSC高置信度信道报告</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:1280px;margin:24px auto;line-height:1.65;color:#20303c}}h1,h2{{color:#174f5f}}.card{{display:inline-block;padding:12px 18px;margin:6px;background:#eef7f5;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd7dd;padding:6px;text-align:left}}th{{background:#e9f1f4}}img{{max-width:100%;height:auto}}details{{margin:8px 0;border:1px solid #d9e2e6;padding:8px;border-radius:6px}}code{{background:#f3f5f6;padding:2px 4px}}</style></head><body>
<h1>全量BCSC高置信度信道与原子分析</h1>
<p><b>完成状态：</b>{'全量完成' if complete else '未完成'}。任务是在同一PulseTrain多个click共享原子的场景下，提取所有入模click的稳定多径，并分析原子频率、幅值、相位及其与信道的联合变化。</p>
<div class='card'>PulseTrain：{len(results)}/718</div><div class='card'>入模click：{sum(row['used_clicks'] for row in train_rows)}</div><div class='card'>高置信度多径：{len(high_rows)}</div><div class='card'>严格高置信度：{stats['strict_high_confidence_channel_count']}</div>
<h2>场景与方法</h2><p>数据来自09_Locate。每个至少含5个click的PulseTrain最多选择10个SNR最高click；联合BCSC共享原子，各click保留独立稀疏信道。每个参考路径执行50次残差块Bootstrap，重复检出率≥90%定义为高置信度，Wilson 95%下限≥80%定义为严格高置信度。</p>
<p>优化目标：Σₘ 0.5‖yₘ−d*xₘ‖²₂ + λΣₘ‖xₘ‖₁。原子为单位L2能量，参考长度85 μs；Bootstrap长度20–150 μs；−3 dB带宽硬约束6–72 kHz；λ=0.075并扰动±15%；峰阈值1.5%；最小路径间距0.035 ms；残差块0.10 ms；匹配容差±0.03 ms。</p>
<p><img src='figures/full_dataset_summary.png'></p>
<h2>高置信度信道下的原子频率、幅值与相位变化</h2>
<p><b>频率：</b>峰值频率{format_quantile(frequency['atom_peak_frequency_khz'], ' kHz')}；谱质心{format_quantile(frequency['atom_spectral_centroid_khz'], ' kHz')}；−3 dB带宽{format_quantile(frequency['atom_bandwidth_3db_khz'], ' kHz')}。</p>
<p><b>幅值：</b>原子单位能量峰值{format_quantile(amplitude['atom_normalized_peak'])}，它只描述波形尖锐程度；真实接收幅值由信道系数与click缩放共同决定。高置信度多径相对幅值{format_quantile(amplitude['channel_relative_amplitude'])}，相对声级{format_quantile(amplitude['channel_relative_level_db'], ' dB')}。时延与相对声级相关系数为{amplitude['delay_level_correlation']:.3f}。</p>
<p><b>相位：</b>峰值对齐后的原子峰频圆均值为{phase['atom_peak_phase_centered_deg']['circular_mean_deg']:.1f}°，圆集中度为{phase['atom_peak_phase_centered_deg']['resultant_length']:.3f}；高置信度路径在原子峰频处的有效相位圆均值为{phase['effective_path_phase_at_atom_peak_deg']['circular_mean_deg']:.1f}°，极性翻转比例为{phase['polarity_inversion_fraction']:.1%}。当前实值模型只能将路径附加相位辨识为0°/180°极性，连续相位变化还包含−2πfτ传播项，不能单独解释为介质相移。</p>
<p><img src='figures/atom_channel_variation.png'></p>
<h2>接收信号合成的具体措施</h2>
<ol><li>从高置信度表按click整组抽取一个<code>atom_id + 全部路径</code>，保留频率、幅值、相位和时延的相关性。</li><li>从atoms.npz读取对应单位能量原子；按路径时延放置稀疏脉冲，幅值使用带符号系数，再与原子卷积。</li><li>使用filtered_peak_before_normalization恢复数字幅值；若要得到Pa或dB re 1 μPa，必须另行提供水听器灵敏度、前置放大器增益和采集标定。</li><li>从joint_residual_psd.npz抽取同类残差谱生成有色噪声，并按residual_rms缩放。</li><li>合成后校验峰值频率、带宽、路径时延、相对幅值和SNR是否落在报告95%经验范围内；越界则重新采样。</li><li>若需要任意频率相关相位，改用解析信号复值BCSC重新估计，不能给当前实值结果任意添加相位。</li></ol>
<p>具体措施见 <code>synthesis_recipe.md</code>；本实验不实现合成函数，也不生成示例WAV。</p>
<h2>参数敏感性与人工调整</h2><p>使用按Voyage和click数量分层的100组样本做单因素敏感性实验，比较λ、原子长度、峰阈值、残差块、匹配容差和联合click数。</p>{decision_html}
{('<p><img src="figures/parameter_sensitivity.png"></p>' if decisions else '')}
<h2>全部原子时频图</h2>{gallery}
<h2>输出文件</h2><p>results/all_channels.csv、high_confidence_channels.csv、atoms.csv、atoms.npz、joint_residual_psd.npz、atom_channel_synthesis_statistics.json、sensitivity_results.csv及parameter_adjustment_decisions.csv。</p>
</body></html>"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[报告进度] 100% 报告已生成：{REPORT_PATH}", flush=True)


def load_existing_main() -> list[dict[str, Any]]:
    path = RESULTS_DIR / "all_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def load_existing_sensitivity() -> list[dict[str, Any]]:
    path = RESULTS_DIR / "sensitivity_all_results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["main", "sensitivity", "report", "all"], default="all")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_COUNT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acceptance_path = OUTPUT_ROOT / "acceptance_validation.json"
    if acceptance_path.exists() and not args.force:
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if acceptance.get("accepted"):
            if args.phase == "report":
                from finalize_full_dataset_experiment import main as finalize_main

                finalize_main()
                return
            if args.phase not in ("main", "sensitivity", "all"):
                return
            raise RuntimeError(
                "正式718组主实验和100组敏感性实验已经完成并通过验收；为避免重复拟合，"
                "当前入口已锁定。只需运行finalize_full_dataset_experiment.py重新生成汇总。"
            )
    ensure_directories()
    specs = discover_eligible()
    if args.force_manifest and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
    save_and_validate_manifest(specs)
    if len(specs) != 718:
        raise RuntimeError(f"当前合格PulseTrain为{len(specs)}，与已确认的718不一致。")
    results = load_existing_main()
    sensitivity = load_existing_sensitivity()
    if args.phase in ("main", "all"):
        results = run_main(specs, args.bootstrap, args.workers, args.force)
    if args.phase in ("sensitivity", "all"):
        sensitivity = run_sensitivity(specs, args.bootstrap, args.workers, args.force)
    if args.phase in ("report", "all"):
        if not results:
            raise RuntimeError("没有主实验结果，不能生成报告。")
        make_report(results, sensitivity)
if __name__ == "__main__":
    main()
