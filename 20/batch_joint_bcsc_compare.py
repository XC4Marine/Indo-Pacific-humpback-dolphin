from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, fftconvolve, find_peaks, sosfiltfilt, windows

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)


PROJECT_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE_ROOT = PROJECT_ROOT / "09_Locate"
OUTPUT_ROOT = PROJECT_ROOT / "20"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
SELECTION_CSV = OUTPUT_ROOT / "selected_50_pulsetrains.csv"

FS_EXPECTED = 576_000
LOWCUT_HZ = 20_000.0
HIGHCUT_HZ = 190_000.0
REFERENCE_ATOM_US = 85.0
ATOM_RANGE_US = (20.0, 150.0)
MIN_BANDWIDTH_HZ = 6_000.0
MAX_BANDWIDTH_HZ = 72_000.0
BANDWIDTH_NFFT = 16_384
LAMBDA_RATIO = 0.075
LAMBDA_PERTURBATION = 0.15
PEAK_RATIO = 0.015
MIN_PATH_DISTANCE_MS = 0.035
MATCH_TOLERANCE_MS = 0.03
RESIDUAL_BLOCK_MS = 0.10
MAX_CLICKS = 10
GLOBAL_SEED = 20260813

# 参考分解和Bootstrap使用不同预算；单/多click始终使用相同预算。
REFERENCE_OUTER = 20
REFERENCE_FISTA = 70
REFERENCE_DICTIONARY = 15
BOOTSTRAP_OUTER = 10
BOOTSTRAP_FISTA = 35
BOOTSTRAP_DICTIONARY = 8


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


def to_float_audio(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data.astype(np.float64) / max(abs(info.min), info.max)
    return data.astype(np.float64)


def preprocess_click(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    if sample_rate != FS_EXPECTED:
        raise ValueError(f"Unexpected sample rate {sample_rate}: {path}")
    raw = to_float_audio(raw)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    raw -= np.mean(raw)
    nyquist = sample_rate / 2.0
    sos = butter(5, [LOWCUT_HZ / nyquist, HIGHCUT_HZ / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, raw)
    filtered *= windows.tukey(len(filtered), alpha=0.04)
    filtered /= np.max(np.abs(filtered)) + np.finfo(float).eps
    return sample_rate, filtered


def snr_proxy(signal: np.ndarray) -> float:
    peak = int(np.argmax(np.abs(signal)))
    guard = max(10, int(round(0.35e-3 * FS_EXPECTED)))
    mask = np.ones(len(signal), dtype=bool)
    mask[max(0, peak - guard):min(len(signal), peak + guard + 1)] = False
    noise = signal[mask]
    median = np.median(noise)
    robust_sigma = 1.4826 * np.median(np.abs(noise - median)) + 1e-12
    return float(np.max(np.abs(signal)) / robust_sigma)


def soft_threshold(z: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)


def reconstruct(atom: np.ndarray, activation: np.ndarray, signal_len: int) -> np.ndarray:
    return fftconvolve(activation, atom, mode="full")[:signal_len]


def adjoint_atom(residual: np.ndarray, atom: np.ndarray, activation_len: int) -> np.ndarray:
    return correlate(residual, atom, mode="valid", method="fft")[:activation_len]


def adjoint_code(residual: np.ndarray, activation: np.ndarray, atom_length: int) -> np.ndarray:
    return correlate(residual, activation, mode="valid", method="fft")[:atom_length]


def spectral_lipschitz(kernel: np.ndarray, output_len: int) -> float:
    n_fft = 1 << int(np.ceil(np.log2(output_len)))
    return float(np.max(np.abs(np.fft.rfft(kernel, n=n_fft)) ** 2) + 1e-10)


def normalize_atom(atom: np.ndarray) -> np.ndarray:
    atom = np.asarray(atom, dtype=float) - np.mean(atom)
    norm = np.linalg.norm(atom)
    if norm < 1e-12:
        raise RuntimeError("Atom norm is nearly zero")
    atom /= norm
    if atom[np.argmax(np.abs(atom))] < 0:
        atom = -atom
    return atom


def measure_bandwidth(atom: np.ndarray, sample_rate: int, n_fft: int = BANDWIDTH_NFFT) -> tuple[float, float, float, float]:
    spectrum = np.abs(np.fft.rfft(atom, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    peak_index = int(np.argmax(spectrum))
    threshold = spectrum[peak_index] / np.sqrt(2.0)
    left = peak_index
    while left > 0 and spectrum[left - 1] >= threshold:
        left -= 1
    right = peak_index
    while right < len(spectrum) - 1 and spectrum[right + 1] >= threshold:
        right += 1

    def crossing(i_below: int, i_above: int) -> float:
        f0, f1 = frequencies[i_below], frequencies[i_above]
        m0, m1 = spectrum[i_below], spectrum[i_above]
        return float(f0 + (threshold - m0) * (f1 - f0) / (m1 - m0 + 1e-15))

    left_hz = 0.0 if left == 0 else crossing(left - 1, left)
    right_hz = sample_rate / 2.0 if right == len(spectrum) - 1 else crossing(right + 1, right)
    return float(right_hz - left_hz), float(frequencies[peak_index]), left_hz, right_hz


def hard_project_bandwidth(atom: np.ndarray, sample_rate: int) -> np.ndarray:
    projected = normalize_atom(atom)
    bandwidth, _, _, _ = measure_bandwidth(projected, sample_rate)
    if MIN_BANDWIDTH_HZ <= bandwidth <= MAX_BANDWIDTH_HZ:
        return projected
    target = float(np.clip(bandwidth, MIN_BANDWIDTH_HZ, MAX_BANDWIDTH_HZ))
    frequencies = np.fft.rfftfreq(BANDWIDTH_NFFT, d=1.0 / sample_rate)
    taper = windows.tukey(len(projected), alpha=0.20)
    for _ in range(12):
        spectrum = np.fft.rfft(projected, n=BANDWIDTH_NFFT)
        peak_index = int(np.argmax(np.abs(spectrum)))
        center = float(frequencies[peak_index])
        sigma = max(target / (2.0 * np.sqrt(np.log(2.0))), frequencies[1])
        envelope = np.exp(-0.5 * ((frequencies - center) / sigma) ** 2)
        candidate = np.fft.irfft(envelope * np.exp(1j * np.angle(spectrum)), n=BANDWIDTH_NFFT)[:len(projected)]
        projected = normalize_atom(candidate * taper)
        bandwidth, _, _, _ = measure_bandwidth(projected, sample_rate)
        if MIN_BANDWIDTH_HZ <= bandwidth <= MAX_BANDWIDTH_HZ:
            return projected
        target *= (1.05 * MIN_BANDWIDTH_HZ / max(bandwidth, 1.0)) if bandwidth < MIN_BANDWIDTH_HZ else (0.95 * MAX_BANDWIDTH_HZ / bandwidth)
        target = float(np.clip(target, 1_000.0, sample_rate * 0.80))
    raise RuntimeError(f"Bandwidth projection failed: {bandwidth / 1e3:.2f} kHz")


def atom_length_from_us(duration_us: float, sample_rate: int) -> int:
    length = int(round(duration_us * 1e-6 * sample_rate))
    length += length % 2 == 0
    return max(5, length)


def initialize_atom(signals: list[np.ndarray], atom_length: int, sample_rate: int) -> np.ndarray:
    segments = []
    half = atom_length // 2
    for signal in signals:
        peak = int(np.argmax(np.abs(signal)))
        padded = np.pad(signal, (half, half))
        segment = padded[peak:peak + atom_length].copy()
        if segment[np.argmax(np.abs(segment))] < 0:
            segment = -segment
        segments.append(segment)
    atom = np.median(np.stack(segments), axis=0) * windows.tukey(atom_length, alpha=0.35)
    return hard_project_bandwidth(atom, sample_rate)


def fista(signal: np.ndarray, atom: np.ndarray, lam: float, x0: np.ndarray | None, iterations: int) -> np.ndarray:
    activation_len = len(signal) - len(atom) + 1
    x = np.zeros(activation_len) if x0 is None else x0.copy()
    z = x.copy()
    momentum = 1.0
    lipschitz = spectral_lipschitz(atom, len(signal))
    for _ in range(iterations):
        residual = reconstruct(atom, z, len(signal)) - signal
        gradient = adjoint_atom(residual, atom, activation_len)
        x_new = soft_threshold(z - gradient / lipschitz, lam / lipschitz)
        momentum_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        z = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
        x, momentum = x_new, momentum_new
    return x


def shift_atom_codes(atom: np.ndarray, codes: list[np.ndarray], sample_rate: int) -> tuple[np.ndarray, list[np.ndarray]]:
    center = len(atom) // 2
    shift = center - int(np.argmax(np.abs(atom)))
    shifted_atom = np.roll(atom, shift)
    if shift > 0:
        shifted_atom[:shift] = 0.0
    elif shift < 0:
        shifted_atom[shift:] = 0.0
    shifted_atom = hard_project_bandwidth(shifted_atom, sample_rate)
    shifted_codes = []
    for code in codes:
        shifted = np.roll(code, -shift)
        if shift > 0:
            shifted[-shift:] = 0.0
        elif shift < 0:
            shifted[:-shift] = 0.0
        shifted_codes.append(shifted)
    return shifted_atom, shifted_codes


def bcsc(signals: list[np.ndarray], atom_length: int, lambda_ratio: float, sample_rate: int,
         outer_iterations: int, fista_iterations: int, dictionary_iterations: int) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    atom = initialize_atom(signals, atom_length, sample_rate)
    codes = [np.zeros(len(signal) - atom_length + 1) for signal in signals]
    taper = windows.tukey(atom_length, alpha=0.20)
    for iteration in range(outer_iterations):
        anneal = 1.8 - 0.8 * min(iteration / max(outer_iterations - 1, 1), 1.0)
        new_codes = []
        for signal, code in zip(signals, codes):
            max_corr = np.max(np.abs(adjoint_atom(signal, atom, len(code))))
            new_codes.append(fista(signal, atom, lambda_ratio * anneal * max_corr, code, fista_iterations))
        codes = new_codes
        lipschitz = sum(spectral_lipschitz(code, len(signal)) for signal, code in zip(signals, codes)) + 1e-10
        step = 0.8 / lipschitz
        for _ in range(dictionary_iterations):
            gradient = np.zeros_like(atom)
            for signal, code in zip(signals, codes):
                residual = reconstruct(atom, code, len(signal)) - signal
                gradient += adjoint_code(residual, code, len(atom))
            atom = hard_project_bandwidth(normalize_atom((atom - step * gradient) * taper), sample_rate)
        atom, codes = shift_atom_codes(atom, codes, sample_rate)
    final_codes = []
    for signal, code in zip(signals, codes):
        max_corr = np.max(np.abs(adjoint_atom(signal, atom, len(code))))
        final_codes.append(fista(signal, atom, lambda_ratio * max_corr, code, 2 * fista_iterations))
    reconstructions = [reconstruct(atom, code, len(signal)) for signal, code in zip(signals, final_codes)]
    return atom, final_codes, reconstructions


def extract_candidates(code: np.ndarray, sample_rate: int, atom_length: int) -> list[dict]:
    magnitude = np.abs(code)
    distance = max(1, int(round(MIN_PATH_DISTANCE_MS * 1e-3 * sample_rate)))
    peaks, _ = find_peaks(magnitude, height=PEAK_RATIO * magnitude.max(), distance=distance)
    strongest = int(np.argmax(magnitude))
    if strongest not in peaks:
        peaks = np.sort(np.append(peaks, strongest))
    arrivals = peaks + atom_length // 2
    coefficients = code[peaks]
    direct_local = int(np.argmax(np.abs(coefficients)))
    direct_sample = arrivals[direct_local]
    direct_amplitude = np.abs(coefficients[direct_local]) + 1e-12
    candidates = []
    for sample, coefficient in zip(arrivals, coefficients):
        relative_delay = (sample - direct_sample) / sample_rate * 1e3
        relative_amplitude = np.abs(coefficient) / direct_amplitude
        candidates.append({
            "relative_delay_ms": float(relative_delay),
            "relative_amplitude": float(relative_amplitude),
            "relative_level_db": float(20 * np.log10(relative_amplitude + 1e-12)),
        })
    return sorted(candidates, key=lambda row: row["relative_delay_ms"])


def circular_block_bootstrap(residual: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    centered = residual - np.mean(residual)
    starts = rng.integers(0, len(centered), size=int(np.ceil(len(centered) / block_length)))
    indices = (starts[:, None] + np.arange(block_length)[None, :]) % len(centered)
    return centered[indices.ravel()[:len(centered)]]


def match_paths(reference_paths: list[dict], candidates: list[dict]) -> list[dict | None]:
    matches: list[dict | None] = [None] * len(reference_paths)
    pairs = []
    for i, reference in enumerate(reference_paths):
        for j, candidate in enumerate(candidates):
            difference = abs(candidate["relative_delay_ms"] - reference["relative_delay_ms"])
            if difference <= MATCH_TOLERANCE_MS:
                pairs.append((difference, i, j))
    used = set()
    for _, i, j in sorted(pairs):
        if matches[i] is None and j not in used:
            matches[i] = candidates[j]
            used.add(j)
    return matches


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def percentile(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    low, median, high = np.percentile(values, [2.5, 50, 97.5])
    return float(median), float(low), float(high)


def atom_similarity(a: np.ndarray, b: np.ndarray) -> float:
    grid = np.linspace(-1.0, 1.0, 201)
    ga = np.linspace(-1.0, 1.0, len(a))
    gb = np.linspace(-1.0, 1.0, len(b))
    ai = np.interp(grid, ga, a)
    bi = np.interp(grid, gb, b)
    return float(abs(np.dot(ai, bi)) / (np.linalg.norm(ai) * np.linalg.norm(bi) + 1e-12))


def list_wavs(train_path: str | Path) -> list[Path]:
    def click_number(path: Path) -> int:
        digits = "".join(character for character in path.parent.name if character.isdigit())
        return int(digits or 0)
    return sorted(Path(train_path).rglob("*.wav"), key=click_number)


def select_training_clicks(train_path: str | Path) -> tuple[list[str], int, list[float]]:
    candidates = []
    for wav_path in list_wavs(train_path):
        _, signal = preprocess_click(wav_path)
        candidates.append((snr_proxy(signal), str(wav_path), signal))
    candidates.sort(key=lambda row: row[0], reverse=True)
    selected = candidates[:MAX_CLICKS]
    anchor_index = 0
    return [row[1] for row in selected], anchor_index, [float(row[0]) for row in selected]


def method_analysis(signals: list[np.ndarray], anchor_index: int, sample_rate: int, bootstrap_count: int,
                    rng: np.random.Generator) -> dict:
    reference_length = atom_length_from_us(REFERENCE_ATOM_US, sample_rate)
    atom, codes, reconstructions = bcsc(
        signals, reference_length, LAMBDA_RATIO, sample_rate,
        REFERENCE_OUTER, REFERENCE_FISTA, REFERENCE_DICTIONARY,
    )
    residuals = [signal - reconstruction for signal, reconstruction in zip(signals, reconstructions)]
    explained_each = [1.0 - np.sum(residual ** 2) / (np.sum(signal ** 2) + 1e-12) for signal, residual in zip(signals, residuals)]
    reference_candidates = extract_candidates(codes[anchor_index], sample_rate, reference_length)
    reference_paths = [row for row in reference_candidates if row["relative_delay_ms"] >= -MATCH_TOLERANCE_MS]
    for index, row in enumerate(reference_paths):
        row["path"] = "direct" if abs(row["relative_delay_ms"]) < 1e-12 else f"multipath_{index}"
    records: list[list[dict]] = [[] for _ in reference_paths]
    block_length = max(2, int(round(RESIDUAL_BLOCK_MS * 1e-3 * sample_rate)))
    bandwidths = []
    atom_lengths_us = []
    for _ in range(bootstrap_count):
        length_us = float(rng.uniform(*ATOM_RANGE_US))
        length = atom_length_from_us(length_us, sample_rate)
        lambda_ratio = LAMBDA_RATIO * rng.uniform(1 - LAMBDA_PERTURBATION, 1 + LAMBDA_PERTURBATION)
        synthetic = [reconstruction + circular_block_bootstrap(residual, block_length, rng)
                     for reconstruction, residual in zip(reconstructions, residuals)]
        atom_b, codes_b, _ = bcsc(
            synthetic, length, lambda_ratio, sample_rate,
            BOOTSTRAP_OUTER, BOOTSTRAP_FISTA, BOOTSTRAP_DICTIONARY,
        )
        bandwidth_b, _, _, _ = measure_bandwidth(atom_b, sample_rate)
        if not MIN_BANDWIDTH_HZ <= bandwidth_b <= MAX_BANDWIDTH_HZ:
            raise RuntimeError(f"Bootstrap bandwidth out of constraint: {bandwidth_b}")
        bandwidths.append(bandwidth_b)
        atom_lengths_us.append(length / sample_rate * 1e6)
        candidates_b = extract_candidates(codes_b[anchor_index], sample_rate, length)
        for path_index, match in enumerate(match_paths(reference_paths, candidates_b)):
            if match is not None:
                records[path_index].append(match)
    bandwidth, center_frequency, left_hz, right_hz = measure_bandwidth(atom, sample_rate)
    paths = []
    for reference, path_records in zip(reference_paths, records):
        detected = len(path_records)
        low, high = wilson(detected, bootstrap_count)
        delay_median, delay_low, delay_high = percentile([row["relative_delay_ms"] for row in path_records])
        amplitude_median, amplitude_low, amplitude_high = percentile([row["relative_amplitude"] for row in path_records])
        paths.append({
            **reference,
            "detected_runs": detected,
            "confidence": detected / bootstrap_count,
            "confidence_low": low,
            "confidence_high": high,
            "delay_median_ms": delay_median,
            "delay_low_ms": delay_low,
            "delay_high_ms": delay_high,
            "amplitude_median": amplitude_median,
            "amplitude_low": amplitude_low,
            "amplitude_high": amplitude_high,
        })
    return {
        "atom": atom.tolist(),
        "reference_atom_length_us": len(atom) / sample_rate * 1e6,
        "reference_bandwidth_khz": bandwidth / 1e3,
        "reference_center_frequency_khz": center_frequency / 1e3,
        "reference_left_3db_khz": left_hz / 1e3,
        "reference_right_3db_khz": right_hz / 1e3,
        "mean_explained_variance": float(np.mean(explained_each)),
        "anchor_explained_variance": float(explained_each[anchor_index]),
        "bootstrap_atom_length_min_us": float(np.min(atom_lengths_us)),
        "bootstrap_atom_length_max_us": float(np.max(atom_lengths_us)),
        "bootstrap_bandwidth_min_khz": float(np.min(bandwidths) / 1e3),
        "bootstrap_bandwidth_max_khz": float(np.max(bandwidths) / 1e3),
        "paths": paths,
    }


def process_train(spec_dict: dict, bootstrap_count: int) -> dict:
    spec = TrainSpec(**spec_dict)
    started = time.perf_counter()
    selected_paths, anchor_index, snrs = select_training_clicks(spec.path)
    signals = [preprocess_click(path)[1] for path in selected_paths]
    seed = GLOBAL_SEED + spec.index * 100_003
    single = method_analysis([signals[anchor_index]], 0, FS_EXPECTED, bootstrap_count, np.random.default_rng(seed + 1))
    joint = method_analysis(signals, anchor_index, FS_EXPECTED, bootstrap_count, np.random.default_rng(seed + 2))
    single["atom_joint_similarity"] = atom_similarity(np.asarray(single["atom"]), np.asarray(joint["atom"]))
    joint["atom_single_similarity"] = single["atom_joint_similarity"]
    return {
        "index": spec.index,
        "key": spec.key,
        "voyage": spec.voyage,
        "pulse_train": spec.pulse_train,
        "path": spec.path,
        "available_clicks": spec.wav_count,
        "used_clicks": len(signals),
        "selected_wavs": selected_paths,
        "anchor_wav": selected_paths[anchor_index],
        "snr_proxies": snrs,
        "single": single,
        "joint": joint,
        "elapsed_seconds": time.perf_counter() - started,
    }


def discover_eligible() -> list[TrainSpec]:
    eligible = []
    for voyage_dir in sorted(SOURCE_ROOT.iterdir()):
        if not voyage_dir.is_dir():
            continue
        for train_dir in sorted(voyage_dir.glob("PulseTrain_*")):
            wav_count = len(list(train_dir.rglob("*.wav")))
            if wav_count >= 5:
                eligible.append(TrainSpec(0, voyage_dir.name, train_dir.name, str(train_dir), wav_count))
    return eligible


def select_specs(count: int = 50) -> list[TrainSpec]:
    if SELECTION_CSV.exists():
        with SELECTION_CSV.open(encoding="utf-8-sig") as file:
            return [TrainSpec(int(row["index"]), row["voyage"], row["pulse_train"], row["path"], int(row["wav_count"]))
                    for row in csv.DictReader(file)][:count]
    eligible = discover_eligible()
    rng = np.random.default_rng(GLOBAL_SEED)
    chosen_indices = rng.choice(len(eligible), size=count, replace=False)
    specs = []
    for index, pool_index in enumerate(chosen_indices, start=1):
        row = eligible[int(pool_index)]
        specs.append(TrainSpec(index, row.voyage, row.pulse_train, row.path, row.wav_count))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with SELECTION_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "voyage", "pulse_train", "wav_count", "path"])
        writer.writeheader()
        for spec in specs:
            writer.writerow(spec.__dict__)
    return specs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_atom_comparison(result: dict) -> str:
    single = result["single"]
    joint = result["joint"]
    atoms = [("单 click", single, "#A23B72"), ("多 click联合", joint, "#188977")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, (label, method, color) in enumerate(atoms):
        atom = np.asarray(method["atom"])
        peak = int(np.argmax(np.abs(atom)))
        time_us = (np.arange(len(atom)) - peak) / FS_EXPECTED * 1e6
        axes[0, column].plot(time_us, atom, color=color, marker="o", ms=2.2, lw=1.25)
        axes[0, column].axvline(0, color="0.35", ls=":", lw=0.8)
        axes[0, column].set(title=f"{label}原子：{method['reference_atom_length_us']:.1f} μs",
                            xlabel="相对主峰时间 (μs)", ylabel="单位范数幅度")
        magnitude = np.abs(np.fft.rfft(atom, n=BANDWIDTH_NFFT))
        frequency = np.fft.rfftfreq(BANDWIDTH_NFFT, d=1 / FS_EXPECTED) / 1e3
        spectrum_db = 20 * np.log10(np.maximum(magnitude / magnitude.max(), 1e-8))
        axes[1, column].plot(frequency, spectrum_db, color=color, lw=1.3)
        axes[1, column].axhline(-3, color="0.35", ls="--", lw=0.8)
        axes[1, column].axvline(method["reference_left_3db_khz"], color="#B32134", ls="--", lw=0.8)
        axes[1, column].axvline(method["reference_right_3db_khz"], color="#B32134", ls="--", lw=0.8)
        axes[1, column].set(title=f"主频 {method['reference_center_frequency_khz']:.1f} kHz；带宽 {method['reference_bandwidth_khz']:.1f} kHz",
                            xlabel="频率 (kHz)", ylabel="归一化幅度 (dB)", xlim=(0, 150), ylim=(-60, 3))
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle(f"{result['index']:03d} | {result['voyage']} / {result['pulse_train']}\n原子相似度={single['atom_joint_similarity']:.3f}", fontsize=13)
    filename = f"{result['key']}_atom_comparison.png"
    fig.savefig(FIGURES_DIR / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return filename


def pair_method_paths(single_paths: list[dict], joint_paths: list[dict]) -> tuple[list[tuple[dict, dict]], int, int]:
    single_candidates = [row for row in single_paths if row["path"] != "direct"]
    joint_candidates = [row for row in joint_paths if row["path"] != "direct"]
    pairs = []
    for i, single in enumerate(single_candidates):
        for j, joint in enumerate(joint_candidates):
            difference = abs(single["relative_delay_ms"] - joint["relative_delay_ms"])
            if difference <= MATCH_TOLERANCE_MS:
                pairs.append((difference, i, j))
    matched = []
    used_single = set()
    used_joint = set()
    for _, i, j in sorted(pairs):
        if i not in used_single and j not in used_joint:
            matched.append((single_candidates[i], joint_candidates[j]))
            used_single.add(i)
            used_joint.add(j)
    return matched, len(single_candidates) - len(used_single), len(joint_candidates) - len(used_joint)


def aggregate(results: list[dict], bootstrap_count: int) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = []
    path_rows = []
    paired_rows = []
    for result in results:
        figure = plot_atom_comparison(result)
        single = result["single"]
        joint = result["joint"]
        matched, single_only, joint_only = pair_method_paths(single["paths"], joint["paths"])
        train_rows.append({
            "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
            "available_clicks": result["available_clicks"], "used_clicks": result["used_clicks"],
            "anchor_wav": result["anchor_wav"], "single_explained": single["anchor_explained_variance"],
            "joint_anchor_explained": joint["anchor_explained_variance"], "joint_mean_explained": joint["mean_explained_variance"],
            "single_bandwidth_khz": single["reference_bandwidth_khz"], "joint_bandwidth_khz": joint["reference_bandwidth_khz"],
            "single_center_khz": single["reference_center_frequency_khz"], "joint_center_khz": joint["reference_center_frequency_khz"],
            "atom_similarity": single["atom_joint_similarity"], "single_path_count": len(single["paths"]),
            "joint_path_count": len(joint["paths"]), "elapsed_seconds": result["elapsed_seconds"], "figure": figure,
            "matched_non_direct_paths": len(matched), "single_only_candidates": single_only,
            "joint_only_candidates": joint_only,
        })
        for method_name, method in (("single", single), ("joint", joint)):
            for path in method["paths"]:
                path_rows.append({
                    "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
                    "method": method_name, "path": path["path"], "reference_delay_ms": path["relative_delay_ms"],
                    "reference_amplitude": path["relative_amplitude"], "reference_level_db": path["relative_level_db"],
                    "detected_runs": path["detected_runs"], "bootstrap_runs": bootstrap_count,
                    "confidence": path["confidence"], "confidence_low": path["confidence_low"], "confidence_high": path["confidence_high"],
                    "delay_median_ms": path["delay_median_ms"], "delay_low_ms": path["delay_low_ms"], "delay_high_ms": path["delay_high_ms"],
                    "amplitude_median": path["amplitude_median"], "amplitude_low": path["amplitude_low"], "amplitude_high": path["amplitude_high"],
                })
        for pair_index, (single_path, joint_path) in enumerate(matched, start=1):
            paired_rows.append({
                "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
                "pair_index": pair_index, "single_delay_ms": single_path["relative_delay_ms"],
                "joint_delay_ms": joint_path["relative_delay_ms"],
                "delay_difference_ms": joint_path["relative_delay_ms"] - single_path["relative_delay_ms"],
                "single_confidence": single_path["confidence"], "joint_confidence": joint_path["confidence"],
                "confidence_difference": joint_path["confidence"] - single_path["confidence"],
                "single_reference_level_db": single_path["relative_level_db"],
                "joint_reference_level_db": joint_path["relative_level_db"],
                "single_only_candidates_in_train": single_only,
                "joint_only_candidates_in_train": joint_only,
            })
    return train_rows, path_rows, paired_rows


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    half = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return mean, mean - half, mean + half


def make_summary_figure(train_rows: list[dict], path_rows: list[dict]) -> str:
    single_conf = [row["confidence"] for row in path_rows if row["method"] == "single" and row["path"] != "direct"]
    joint_conf = [row["confidence"] for row in path_rows if row["method"] == "joint" and row["path"] != "direct"]
    single_explained = [row["single_explained"] for row in train_rows]
    joint_explained = [row["joint_anchor_explained"] for row in train_rows]
    similarity = [row["atom_similarity"] for row in train_rows]
    single_bw = [row["single_bandwidth_khz"] for row in train_rows]
    joint_bw = [row["joint_bandwidth_khz"] for row in train_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].boxplot([single_conf, joint_conf], tick_labels=["单 click", "多 click联合"], showmeans=True)
    axes[0, 0].set(title="全部非直达径候选的稳定性置信度", ylabel="检出率")
    axes[0, 1].scatter(single_explained, joint_explained, color="#2457A6", alpha=0.8)
    bounds = [min(single_explained + joint_explained), max(single_explained + joint_explained)]
    axes[0, 1].plot(bounds, bounds, color="0.4", ls="--")
    axes[0, 1].set(title="锚点 click 重构解释率", xlabel="单 click", ylabel="多 click联合")
    axes[1, 0].hist(similarity, bins=np.linspace(0, 1, 16), color="#5B3A8C", alpha=0.8)
    axes[1, 0].set(title="单/多 click原子相似度分布", xlabel="绝对相关系数", ylabel="PulseTrain数")
    axes[1, 1].scatter(single_bw, joint_bw, color="#188977", alpha=0.8)
    axes[1, 1].plot([6, 72], [6, 72], color="0.4", ls="--")
    axes[1, 1].set(title="原子 −3 dB带宽", xlabel="单 click (kHz)", ylabel="多 click联合 (kHz)")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    path = FIGURES_DIR / "overall_single_vs_joint_summary.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path.name


def make_paired_figure(paired_rows: list[dict]) -> str:
    single = np.array([row["single_confidence"] for row in paired_rows])
    joint = np.array([row["joint_confidence"] for row in paired_rows])
    difference = joint - single
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].scatter(single, joint, color="#2457A6", alpha=0.72)
    axes[0].plot([0, 1], [0, 1], color="0.4", ls="--")
    axes[0].set(title=f"延迟匹配的同一路径（n={len(paired_rows)}）", xlabel="单 click置信度", ylabel="多 click联合置信度", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axes[1].hist(difference, bins=np.linspace(-1, 1, 21), color="#188977", alpha=0.82)
    axes[1].axvline(0, color="0.4", ls="--")
    axes[1].axvline(float(np.mean(difference)), color="#B32134", lw=1.3, label=f"平均变化 {np.mean(difference):+.3f}")
    axes[1].set(title="同一路径置信度变化", xlabel="多 click − 单 click", ylabel="路径数")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    path = FIGURES_DIR / "paired_path_confidence_comparison.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path.name


def make_report(train_rows: list[dict], path_rows: list[dict], paired_rows: list[dict], bootstrap_count: int) -> None:
    summary_figure = make_summary_figure(train_rows, path_rows)
    paired_figure = make_paired_figure(paired_rows)
    single_non_direct = [row for row in path_rows if row["method"] == "single" and row["path"] != "direct"]
    joint_non_direct = [row for row in path_rows if row["method"] == "joint" and row["path"] != "direct"]
    single_conf_mean = float(np.mean([row["confidence"] for row in single_non_direct])) if single_non_direct else math.nan
    joint_conf_mean = float(np.mean([row["confidence"] for row in joint_non_direct])) if joint_non_direct else math.nan
    single_high = sum(row["confidence"] >= 0.9 for row in single_non_direct)
    joint_high = sum(row["confidence"] >= 0.9 for row in joint_non_direct)
    similarity_mean = float(np.mean([row["atom_similarity"] for row in train_rows]))
    explained_delta = [row["joint_anchor_explained"] - row["single_explained"] for row in train_rows]
    delta_mean, delta_low, delta_high = mean_ci(explained_delta)
    bandwidth_valid = all(6 <= row["single_bandwidth_khz"] <= 72 and 6 <= row["joint_bandwidth_khz"] <= 72 for row in train_rows)
    paired_difference = np.array([row["confidence_difference"] for row in paired_rows])
    paired_mean = float(np.mean(paired_difference))
    paired_median = float(np.median(paired_difference))
    paired_improved = float(np.mean(paired_difference > 0))
    single_only = sum(int(row["single_only_candidates"]) for row in train_rows)
    joint_only = sum(int(row["joint_only_candidates"]) for row in train_rows)

    rows_html = []
    for row in train_rows:
        rows_html.append(
            "<tr>" + "".join([
                f"<td>{row['index']:03d}</td>", f"<td>{html.escape(row['voyage'])}</td>",
                f"<td>{html.escape(row['pulse_train'])}</td>", f"<td>{row['used_clicks']}</td>",
                f"<td>{row['single_explained']:.3f}</td>", f"<td>{row['joint_anchor_explained']:.3f}</td>",
                f"<td>{row['atom_similarity']:.3f}</td>", f"<td>{row['single_bandwidth_khz']:.1f}</td>",
                f"<td>{row['joint_bandwidth_khz']:.1f}</td>", f"<td><a href='figures/{html.escape(row['figure'])}'>图</a></td>",
            ]) + "</tr>"
        )
    gallery = "".join(
        f"<figure><a href='figures/{html.escape(row['figure'])}'><img src='figures/{html.escape(row['figure'])}' loading='lazy'></a>"
        f"<figcaption>{row['index']:03d} | {html.escape(row['voyage'])} / {html.escape(row['pulse_train'])}</figcaption></figure>"
        for row in train_rows
    )
    report = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>单click与多click联合BCSC比较报告</title><style>
body{{font-family:'Microsoft YaHei',Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#20242a;line-height:1.55}}
h1,h2{{color:#17324d}} .note{{background:#eef5fb;border-left:4px solid #2457a6;padding:12px 16px}}
.metric{{display:inline-block;background:#f4f6f8;margin:5px;padding:10px 14px;border-radius:7px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd3da;padding:6px;text-align:right}}th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:15px}}figure{{margin:0;border:1px solid #d7dce1;padding:8px}}img{{width:100%;height:auto}}figcaption{{font-size:13px}}
</style></head><body><h1>单 click 与同 PulseTrain 多 click联合 BCSC 比较报告</h1>
<p>随机种子 {GLOBAL_SEED}；从至少5个click的PulseTrain中固定随机抽取{len(train_rows)}组；每组最多使用{MAX_CLICKS}个click；每种方法{bootstrap_count}次扰动Bootstrap。</p>
<div class='note'><b>可比性定义：</b>两种方法都在同一高信噪比锚点click上提取和匹配多径；区别仅在于单click方法独立学习原子，多click方法由同一PulseTrain最多10个click共享学习原子。置信度是扰动重现率，不是物理真实性概率。</div>
<h2>总体结果</h2>
<div class='metric'>单click非直达径平均置信度：<b>{single_conf_mean:.1%}</b></div>
<div class='metric'>多click联合非直达径平均置信度：<b>{joint_conf_mean:.1%}</b></div>
<div class='metric'>高置信路径数：<b>{single_high} → {joint_high}</b></div>
<div class='metric'>原子平均相似度：<b>{similarity_mean:.3f}</b></div>
<div class='metric'>锚点解释率变化：<b>{delta_mean:+.3f}</b>（近似95%区间 {delta_low:+.3f}–{delta_high:+.3f}）</div>
<div class='metric'>时长/带宽约束：<b>{'全部通过' if bandwidth_valid else '存在越界'}</b></div>
<p><img src='figures/{summary_figure}' alt='总体比较图'></p>
<h2>同一路径的成对置信度比较</h2>
<p>按相对延迟差≤{MATCH_TOLERANCE_MS:.2f} ms进行一对一匹配，共匹配<b>{len(paired_rows)}</b>条非直达径。多click联合相对单click的置信度平均变化为<b>{paired_mean:+.3f}</b>，中位数变化为<b>{paired_median:+.3f}</b>；有<b>{paired_improved:.1%}</b>的匹配路径得到提升。另有单click独有候选<b>{single_only}</b>条、多click联合独有候选<b>{joint_only}</b>条。</p>
<p><img src='figures/{paired_figure}' alt='成对路径置信度比较'></p>
<h2>方法与约束</h2><ul>
<li>优化目标：Σₘ 0.5‖yₘ−d*xₘ‖²₂ + λΣₘ‖xₘ‖₁；单click是M=1的特例。</li>
<li>原子参考时长85 μs；Bootstrap时长在20–150 μs内随机扰动。</li>
<li>每次原子更新后执行硬频谱投影，要求实测−3 dB带宽为6–72 kHz。</li>
<li>正则强度±15%；残差0.10 ms循环移动块Bootstrap；路径匹配容差±0.03 ms。</li>
</ul>
<h2>50组统计</h2><table><thead><tr><th>编号</th><th>Voyage</th><th>PulseTrain</th><th>联合click数</th><th>单解释率</th><th>联合解释率</th><th>原子相似度</th><th>单带宽/kHz</th><th>联合带宽/kHz</th><th>时频图</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>
<h2>各组原子时频域前后比较</h2><div class='gallery'>{gallery}</div>
<h2>结果文件</h2><ul><li>results/pulsetrain_summary.csv</li><li>results/path_confidence_comparison.csv</li><li>results/paired_path_confidence_comparison.csv</li><li>results/all_results.json</li><li>selected_50_pulsetrains.csv</li></ul>
</body></html>"""
    (OUTPUT_ROOT / "report.html").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for directory in (OUTPUT_ROOT, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    specs = select_specs(args.count)
    print(f"[总体进度] 0/{len(specs)} PulseTrain；每组2种方法×{args.bootstrap}次Bootstrap", flush=True)
    completed: dict[int, dict] = {}
    pending = []
    for spec in specs:
        checkpoint = CHECKPOINT_DIR / f"{spec.key}.json"
        if checkpoint.exists() and not args.force:
            completed[spec.index] = json.loads(checkpoint.read_text(encoding="utf-8"))
        else:
            pending.append(spec)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_train, spec.__dict__, args.bootstrap): spec for spec in pending}
            progress = tqdm(total=len(specs), initial=len(completed), desc="总体进度：PulseTrain")
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    progress.close()
                    raise RuntimeError(f"Failed at {spec.key}: {error}") from error
                completed[spec.index] = result
                (CHECKPOINT_DIR / f"{spec.key}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                progress.update(1)
                print(f"[完成] {len(completed)}/{len(specs)} {spec.voyage}/{spec.pulse_train} {result['elapsed_seconds']:.1f}s", flush=True)
            progress.close()
    results = [completed[spec.index] for spec in specs]
    train_rows, path_rows, paired_rows = aggregate(results, args.bootstrap)
    write_csv(RESULTS_DIR / "pulsetrain_summary.csv", train_rows)
    write_csv(RESULTS_DIR / "path_confidence_comparison.csv", path_rows)
    write_csv(RESULTS_DIR / "paired_path_confidence_comparison.csv", paired_rows)
    (RESULTS_DIR / "all_results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    make_report(train_rows, path_rows, paired_rows, args.bootstrap)
    print(f"[总体进度] 100% 完成。报告：{OUTPUT_ROOT / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
