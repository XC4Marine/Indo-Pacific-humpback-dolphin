"""Blind constrained convolutional sparse coding for Experiment 20 v4.

The extraction code in this module intentionally accepts only ``clicktrain_raw.npy``.
Truth assets are handled by the separate evaluator and must never be imported here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.signal import butter, correlate, fftconvolve, find_peaks, sosfiltfilt, windows


FS = 576_000
SOURCE_LENGTH = 116
CHANNEL_WINDOW_S = 0.004
CHANNEL_LENGTH = int(round(CHANNEL_WINDOW_S * FS))
SEGMENT_LENGTH = SOURCE_LENGTH + CHANNEL_LENGTH - 1
LOWCUT_HZ = 20_000.0
HIGHCUT_HZ = 190_000.0
MIN_BANDWIDTH_HZ = 6_000.0
MAX_BANDWIDTH_HZ = 72_000.0
BANDWIDTH_NFFT = 16_384
GROUP_BLOCK_SAMPLES = 13  # 22.6 us, covering the configured +/-10 us perturbation.


@dataclass(frozen=True)
class FitConfig:
    lambda_ratio: float = 0.075
    group_ratio: float = 0.0
    outer_iterations: int = 28
    code_iterations: int = 70
    source_iterations: int = 6
    admm_iterations: int = 10
    admm_rho: float = 0.25
    convergence_tolerance: float = 1e-6
    peak_ratio: float = 0.015
    min_path_distance_ms: float = 0.035

    @property
    def method(self) -> Literal["independent", "shared_support"]:
        return "shared_support" if self.group_ratio > 0 else "independent"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bandpass(signal: np.ndarray, sample_rate: int = FS) -> np.ndarray:
    """Apply the same 20--190 kHz zero-phase preprocessing used by experiment 19."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    sos = butter(5, [LOWCUT_HZ, HIGHCUT_HZ], btype="bandpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal)


def reconstruct(source: np.ndarray, channel: np.ndarray, length: int) -> np.ndarray:
    return fftconvolve(source, channel, mode="full")[:length]


def source_adjoint(residual: np.ndarray, channel: np.ndarray) -> np.ndarray:
    return correlate(residual, channel, mode="valid")


def channel_adjoint(residual: np.ndarray, source: np.ndarray) -> np.ndarray:
    return correlate(residual, source, mode="valid")


def soft_threshold(values: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _normalise_source(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    if len(source) != SOURCE_LENGTH:
        raise ValueError(f"source must have exactly {SOURCE_LENGTH} samples")
    norm = np.linalg.norm(source)
    if norm < 1e-14:
        raise RuntimeError("source norm collapsed to zero")
    source = source / norm
    if source[int(np.argmax(np.abs(source)))] < 0:
        source = -source
    return source


def measure_3db_bandwidth(source: np.ndarray, sample_rate: int = FS, n_fft: int = BANDWIDTH_NFFT) -> tuple[float, float, float, float]:
    """Return -3 dB width, peak, left and right crossings with linear interpolation."""
    spectrum = np.abs(np.fft.rfft(source, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    peak_index = int(np.argmax(spectrum))
    threshold = spectrum[peak_index] / np.sqrt(2.0)
    left = peak_index
    right = peak_index
    while left > 0 and spectrum[left - 1] >= threshold:
        left -= 1
    while right < len(spectrum) - 1 and spectrum[right + 1] >= threshold:
        right += 1

    def crossing(below: int, above: int) -> float:
        f0, f1 = frequencies[below], frequencies[above]
        m0, m1 = spectrum[below], spectrum[above]
        return float(f0 + (threshold - m0) * (f1 - f0) / (m1 - m0 + 1e-15))

    left_hz = 0.0 if left == 0 else crossing(left - 1, left)
    right_hz = sample_rate / 2.0 if right == len(spectrum) - 1 else crossing(right + 1, right)
    return float(right_hz - left_hz), float(frequencies[peak_index]), float(left_hz), float(right_hz)


def project_bandwidth(source: np.ndarray) -> np.ndarray:
    """Project a 116-point source onto the experiment-19 6--72 kHz hard constraint."""
    projected = _normalise_source(source)
    bandwidth, _, _, _ = measure_3db_bandwidth(projected)
    if MIN_BANDWIDTH_HZ <= bandwidth <= MAX_BANDWIDTH_HZ:
        return projected
    target = float(np.clip(bandwidth, MIN_BANDWIDTH_HZ, MAX_BANDWIDTH_HZ))
    frequencies = np.fft.rfftfreq(BANDWIDTH_NFFT, d=1.0 / FS)
    taper = windows.tukey(SOURCE_LENGTH, alpha=0.20)
    for _ in range(16):
        spectrum = np.fft.rfft(projected, n=BANDWIDTH_NFFT)
        centre = float(frequencies[int(np.argmax(np.abs(spectrum)))])
        sigma = max(target / (2.0 * np.sqrt(np.log(2.0))), frequencies[1])
        envelope = np.exp(-0.5 * ((frequencies - centre) / sigma) ** 2)
        candidate = np.fft.irfft(envelope * np.exp(1j * np.angle(spectrum)), n=BANDWIDTH_NFFT)[:SOURCE_LENGTH]
        projected = _normalise_source(candidate * taper)
        bandwidth, _, _, _ = measure_3db_bandwidth(projected)
        if MIN_BANDWIDTH_HZ <= bandwidth <= MAX_BANDWIDTH_HZ:
            return projected
        if bandwidth < MIN_BANDWIDTH_HZ:
            target *= 1.05 * MIN_BANDWIDTH_HZ / max(bandwidth, 1.0)
        else:
            target *= 0.95 * MAX_BANDWIDTH_HZ / bandwidth
        target = float(np.clip(target, 1_000.0, FS * 0.80))
    raise RuntimeError(f"-3 dB bandwidth projection failed ({bandwidth / 1e3:.2f} kHz)")


def canonicalise(source: np.ndarray, channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix blind scale/sign/shift ambiguity without consulting a truth waveform."""
    source = _normalise_source(source)
    channels = np.asarray(channels, dtype=np.float64).copy()
    peak = int(np.argmax(np.abs(source)))
    target = (SOURCE_LENGTH - 1) // 2
    shift = target - peak
    if shift:
        shifted_source = np.roll(source, shift)
        if shift > 0:
            shifted_source[:shift] = 0.0
        else:
            shifted_source[shift:] = 0.0
        shifted_channels = np.roll(channels, -shift, axis=1)
        if shift > 0:
            shifted_channels[:, -shift:] = 0.0
        else:
            shifted_channels[:, :-shift] = 0.0
        source, channels = shifted_source, shifted_channels
    source = project_bandwidth(source)
    if source[int(np.argmax(np.abs(source)))] < 0:
        source, channels = -source, -channels
    return source, channels


def detect_event_segments(clicktrain: np.ndarray, expected_events: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Detect direct-path bursts from the waveform alone and return equal-length windows."""
    filtered = bandpass(clicktrain)
    envelope = np.abs(filtered)
    smoothing = max(3, int(round(0.08e-3 * FS)))
    envelope = np.convolve(envelope, np.ones(smoothing) / smoothing, mode="same")
    distance = int(round(5e-3 * FS))
    prominence = max(float(np.max(envelope)) * 0.025, np.finfo(float).eps)
    peaks, properties = find_peaks(envelope, distance=distance, prominence=prominence)
    if len(peaks) < expected_events:
        raise RuntimeError(f"only {len(peaks)} event peaks detected; expected {expected_events}")
    heights = properties.get("prominences", envelope[peaks])
    chosen = peaks[np.argsort(heights)[-expected_events:]]
    chosen.sort()
    # The window starts before the observed burst; the source/channel ambiguity is
    # subsequently canonicalised, so no hidden emission timestamp is required.
    lead = int(round(0.20e-3 * FS))
    segments: list[np.ndarray] = []
    starts: list[int] = []
    for peak in chosen:
        start = max(0, int(peak) - lead)
        stop = start + SEGMENT_LENGTH
        segment = filtered[start:stop]
        if len(segment) < SEGMENT_LENGTH:
            segment = np.pad(segment, (0, SEGMENT_LENGTH - len(segment)))
        segments.append(segment)
        starts.append(start)
    return np.asarray(segments), np.asarray(starts, dtype=np.int64)


def initialise_source(segments: np.ndarray) -> np.ndarray:
    centre = int(np.median(np.argmax(np.abs(segments), axis=1)))
    half = SOURCE_LENGTH // 2
    atoms = []
    for segment in segments:
        padded = np.pad(segment, (half, half))
        atom = padded[centre:centre + SOURCE_LENGTH].copy()
        if atom[np.argmax(np.abs(atom))] < 0:
            atom = -atom
        atoms.append(atom)
    return project_bandwidth(np.median(np.stack(atoms), axis=0) * windows.tukey(SOURCE_LENGTH, alpha=0.35))


def _code_lipschitz(source: np.ndarray) -> float:
    n = 1 << int(np.ceil(np.log2(SEGMENT_LENGTH + SOURCE_LENGTH)))
    return float(np.max(np.abs(np.fft.rfft(source, n=n)) ** 2) + 1e-10)


def _source_lipschitz(channels: np.ndarray, weights: np.ndarray) -> float:
    n = 1 << int(np.ceil(np.log2(SEGMENT_LENGTH + CHANNEL_LENGTH)))
    total = 0.0
    for channel, weight in zip(channels, weights):
        total += float(weight) * float(np.max(np.abs(np.fft.rfft(channel, n=n)) ** 2))
    return total + 1e-10


def solve_independent_codes(segments: np.ndarray, source: np.ndarray, channels: np.ndarray, lambda_ratio: float, iterations: int) -> np.ndarray:
    """FISTA solve for the independent sparse channels at fixed source."""
    result = channels.copy()
    lipschitz = _code_lipschitz(source)
    for event, segment in enumerate(segments):
        scale = np.max(np.abs(channel_adjoint(segment, source))) + 1e-12
        threshold = lambda_ratio * scale / lipschitz
        x = result[event].copy()
        z = x.copy()
        momentum = 1.0
        for _ in range(iterations):
            residual = reconstruct(source, z, len(segment)) - segment
            gradient = channel_adjoint(residual, source)
            x_new = soft_threshold(z - gradient / lipschitz, threshold)
            momentum_new = 0.5 * (1 + np.sqrt(1 + 4 * momentum * momentum))
            z = x_new + ((momentum - 1) / momentum_new) * (x_new - x)
            x, momentum = x_new, momentum_new
        result[event] = x
    return result


def _block_shrink(values: np.ndarray, threshold: float) -> np.ndarray:
    result = np.zeros_like(values)
    for start in range(0, values.shape[1], GROUP_BLOCK_SAMPLES):
        stop = min(values.shape[1], start + GROUP_BLOCK_SAMPLES)
        block = values[:, start:stop]
        norm = float(np.linalg.norm(block))
        if norm > threshold:
            result[:, start:stop] = block * (1.0 - threshold / norm)
    return result


def solve_shared_support_admm(segments: np.ndarray, source: np.ndarray, channels: np.ndarray, lambda_ratio: float, group_ratio: float, iterations: int, rho: float) -> np.ndarray:
    """ADMM update for L1 channels plus 13-sample cross-event group support."""
    scales = np.asarray([np.max(np.abs(channel_adjoint(y, source))) + 1e-12 for y in segments])
    lam = float(lambda_ratio * np.median(scales))
    group_lam = float(group_ratio * lam)
    h = channels.copy()
    z_l1 = h.copy()
    z_group = h.copy()
    u_l1 = np.zeros_like(h)
    u_group = np.zeros_like(h)
    lipschitz = _code_lipschitz(source) + 2.0 * rho
    for _ in range(iterations):
        for event, segment in enumerate(segments):
            residual = reconstruct(source, h[event], len(segment)) - segment
            gradient = channel_adjoint(residual, source) + rho * (h[event] - z_l1[event] + u_l1[event]) + rho * (h[event] - z_group[event] + u_group[event])
            h[event] -= gradient / lipschitz
        z_l1 = soft_threshold(h + u_l1, lam / rho)
        z_group = _block_shrink(h + u_group, group_lam / rho)
        u_l1 += h - z_l1
        u_group += h - z_group
    return h


def objective(segments: np.ndarray, source: np.ndarray, channels: np.ndarray, config: FitConfig) -> dict[str, float]:
    energies = np.sum(segments * segments, axis=1) + 1e-18
    residual_energy = 0.0
    for segment, channel, energy in zip(segments, channels, energies):
        residual = reconstruct(source, channel, len(segment)) - segment
        residual_energy += float(np.sum(residual * residual) / energy)
    l1 = float(config.lambda_ratio * np.sum(np.abs(channels)))
    group = 0.0
    if config.group_ratio > 0:
        for start in range(0, channels.shape[1], GROUP_BLOCK_SAMPLES):
            group += float(np.linalg.norm(channels[:, start:start + GROUP_BLOCK_SAMPLES]))
        group *= config.lambda_ratio * config.group_ratio
    return {"data": residual_energy, "l1": l1, "group": group, "total": residual_energy + l1 + group}


def fit_segments(segments: np.ndarray, config: FitConfig, initial_source: np.ndarray | None = None) -> dict[str, Any]:
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 2 or not len(segments) or segments.shape[1] != SEGMENT_LENGTH:
        raise ValueError(f"expected (event_count, {SEGMENT_LENGTH}) segments, got {segments.shape}")
    # ``initial_source`` is an opt-in initialization hook.  The default remains
    # bit-for-bit the historical v4 behavior; v7 uses it only for its explicitly
    # specified, full-iteration direct-waveform multi-start source refits.
    source = initialise_source(segments) if initial_source is None else project_bandwidth(initial_source)
    channels = np.zeros((len(segments), CHANNEL_LENGTH), dtype=np.float64)
    weights = 1.0 / (np.sum(segments * segments, axis=1) + 1e-18)
    history: list[dict[str, float]] = []
    converged = False
    previous = np.inf
    for iteration in range(config.outer_iterations):
        anneal = 1.8 - 0.8 * iteration / max(config.outer_iterations - 1, 1)
        if config.method == "independent":
            channels = solve_independent_codes(segments, source, channels, config.lambda_ratio * anneal, config.code_iterations)
        else:
            channels = solve_shared_support_admm(segments, source, channels, config.lambda_ratio * anneal, config.group_ratio, config.admm_iterations, config.admm_rho)
        step = 0.8 / _source_lipschitz(channels, weights)
        taper = windows.tukey(SOURCE_LENGTH, alpha=0.20)
        for _ in range(config.source_iterations):
            gradient = np.zeros(SOURCE_LENGTH)
            for segment, channel, weight in zip(segments, channels, weights):
                residual = reconstruct(source, channel, len(segment)) - segment
                gradient += weight * source_adjoint(residual, channel)
            source = project_bandwidth((source - step * gradient) * taper)
        source, channels = canonicalise(source, channels)
        terms = objective(segments, source, channels, config)
        terms["iteration"] = float(iteration + 1)
        bandwidth, _, _, _ = measure_3db_bandwidth(source)
        terms["bandwidth_3db_hz"] = bandwidth
        terms["bandwidth_pass"] = float(MIN_BANDWIDTH_HZ <= bandwidth <= MAX_BANDWIDTH_HZ)
        history.append(terms)
        if previous - terms["total"] <= config.convergence_tolerance * max(1.0, previous):
            converged = iteration >= 3
            if converged:
                break
        previous = terms["total"]
    reconstructions = np.stack([reconstruct(source, channel, SEGMENT_LENGTH) for channel in channels])
    explained = 1.0 - np.sum((segments - reconstructions) ** 2, axis=1) / (np.sum(segments ** 2, axis=1) + 1e-18)
    bandwidth, peak, left, right = measure_3db_bandwidth(source)
    return {
        "source": source,
        "channels": channels,
        "reconstructions": reconstructions,
        "history": history,
        "converged": converged,
        "iterations": len(history),
        "explained_variance": explained,
        "constraint": {
            "source_length": int(len(source)), "source_l2_norm": float(np.linalg.norm(source)),
            "bandwidth_3db_hz": bandwidth, "peak_frequency_hz": peak, "left_3db_hz": left, "right_3db_hz": right,
            "bandwidth_pass": bool(MIN_BANDWIDTH_HZ <= bandwidth <= MAX_BANDWIDTH_HZ),
        },
    }


def fit_clicktrain(path: str | Path, config: FitConfig) -> dict[str, Any]:
    """Extraction entry point. It reads one and only one clicktrain waveform."""
    path = Path(path)
    if path.name != "clicktrain_raw.npy":
        raise ValueError("extraction accepts only clicktrain_raw.npy")
    segments, starts = detect_event_segments(np.load(path, allow_pickle=False))
    result = fit_segments(segments, config)
    result["detected_window_starts"] = starts
    result["input_path"] = str(path)
    return result


def fit_single_segment(segment: np.ndarray, config: FitConfig) -> dict[str, Any]:
    """Fit one click only; no other click or clicktrain-level state is supplied."""
    result = fit_segments(np.asarray(segment, dtype=np.float64).reshape(1, -1), FitConfig(**{**config.as_dict(), "group_ratio": 0.0}))
    return result


def path_candidates(channel: np.ndarray, peak_ratio: float = 0.015, min_distance_ms: float = 0.035) -> list[dict[str, float]]:
    magnitude = np.abs(channel)
    if not np.any(magnitude):
        return []
    distance = max(1, int(round(min_distance_ms * 1e-3 * FS)))
    peaks, _ = find_peaks(magnitude, height=peak_ratio * magnitude.max(), distance=distance)
    strongest = int(np.argmax(magnitude))
    if strongest not in peaks:
        peaks = np.sort(np.append(peaks, strongest))
    reference = int(peaks[np.argmax(magnitude[peaks])])
    return [{"sample": int(index), "delay_ms": float((index - reference) / FS * 1e3), "coefficient": float(channel[index]), "relative_amplitude": float(magnitude[index] / (magnitude[reference] + 1e-18))} for index in peaks]
