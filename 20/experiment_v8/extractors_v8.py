"""Truth-free Cepstrum/CA-CFAR, MED, and shared-topology fusion routines."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.stats import norm

from protocol_v8 import CHANNEL_SAMPLES, EVENT_COUNT, FS, SEGMENT_SAMPLES

V4_ROOT = Path(__file__).resolve().parent.parent / "experiment_v4"
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import detect_event_segments  # noqa: E402

EPS = 1e-12
MIN_DELAY_SAMPLES = int(round(30e-6 * FS))
TOLERANCE_MS = 0.03
STABLE_EVENT_COUNT = 6


@dataclass(frozen=True)
class Prepared:
    segments: np.ndarray
    support_samples: np.ndarray


def prepare_segments(clicktrain: np.ndarray) -> Prepared:
    segments, _ = detect_event_segments(np.asarray(clicktrain, dtype=np.float64))
    if segments.shape != (EVENT_COUNT, SEGMENT_SAMPLES):
        raise ValueError(f"unexpected event matrix shape {segments.shape}")
    normalised, supports = [], []
    for segment in segments:
        peak = int(np.argmax(np.abs(segment)))
        threshold = .05 * float(np.max(np.abs(segment)))
        left = peak
        while left > 0 and abs(segment[left - 1]) >= threshold: left -= 1
        right = peak
        while right + 1 < len(segment) and abs(segment[right + 1]) >= threshold: right += 1
        width = right - left + 1
        supports.append(int(np.clip(width, 8, 256)))
        normalised.append(segment / (float(np.max(np.abs(segment))) + EPS))
    return Prepared(np.asarray(normalised, dtype=np.float64), np.asarray(supports, dtype=int))


def _direct_index(segment: np.ndarray) -> int:
    return int(np.argmax(np.abs(segment)))


def _ca_cfar(values: np.ndarray, start: int, stop: int, support: int, pfa: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Cell-averaging CFAR with data-adaptive reference and guard spans."""
    reference = int(np.clip(4 * support, 16, 256)); guard = int(np.clip(support, 4, 128))
    detected = np.zeros(len(values), dtype=bool); thresholds = np.full(len(values), np.nan)
    for index in range(start, stop):
        left = values[max(0, index - guard - reference):max(0, index - guard)]
        right = values[min(len(values), index + guard + 1):min(len(values), index + guard + reference + 1)]
        cells = np.concatenate((left, right))
        if len(cells) < max(8, reference // 2):
            continue
        alpha = len(cells) * (pfa ** (-1.0 / len(cells)) - 1.0)
        thresholds[index] = float(alpha * np.mean(cells))
        detected[index] = values[index] > thresholds[index]
    return detected, thresholds


def cepstrum_cfar_event(segment: np.ndarray, support: int) -> list[float]:
    # Cepstral quefrency is already a relative delay; unlike waveform peaks it
    # must not be offset by the absolute direct-arrival index in the window.
    start = max(MIN_DELAY_SAMPLES, support); stop = min(CHANNEL_SAMPLES, len(segment) - 1)
    if stop <= start + 4:
        return []
    nfft = 1 << int(np.ceil(np.log2(len(segment))))
    spectrum = np.fft.rfft(segment * np.hanning(len(segment)), n=nfft)
    floor = max(float(np.median(np.abs(spectrum) ** 2)) * 1e-8, EPS)
    cepstrum = np.fft.irfft(np.log(np.maximum(np.abs(spectrum) ** 2, floor)), n=nfft)[:len(segment)]
    energy = cepstrum * cepstrum
    detected, _ = _ca_cfar(energy, start, stop, support)
    peaks, _ = find_peaks(energy[start:stop], distance=max(1, support))
    return [float((start + int(peak)) / FS * 1e3) for peak in peaks if detected[start + int(peak)]]


def _med_output(signal: np.ndarray, filter_length: int, max_iterations: int = 25, tolerance: float = 1e-6) -> np.ndarray:
    """A regularised fixed-point minimum-entropy deconvolution solver."""
    padded = np.pad(np.asarray(signal, dtype=float), (filter_length - 1, 0))
    matrix = np.lib.stride_tricks.sliding_window_view(padded, filter_length)[:, ::-1].copy()
    matrix /= np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True) + EPS)
    filt = np.zeros(filter_length); filt[filter_length // 2] = 1.0
    covariance = matrix.T @ matrix / len(matrix)
    ridge = 1e-5 * float(np.trace(covariance) / filter_length + EPS)
    for _ in range(max_iterations):
        output = matrix @ filt
        target = matrix.T @ (output ** 3) / len(output)
        updated = np.linalg.solve(covariance + ridge * np.eye(filter_length), target)
        updated /= np.linalg.norm(updated) + EPS
        if np.linalg.norm(updated - filt) < tolerance or np.linalg.norm(updated + filt) < tolerance:
            filt = updated; break
        filt = updated
    return matrix @ filt


def _bh_fdr(values: np.ndarray, q: float = .01) -> np.ndarray:
    if not len(values):
        return np.zeros(0, dtype=bool)
    center = float(np.median(values)); scale = max(1.4826 * float(np.median(np.abs(values - center))), EPS)
    pvalues = norm.sf((values - center) / scale)
    order = np.argsort(pvalues); ranked = pvalues[order]
    accepted = np.flatnonzero(ranked <= q * (np.arange(1, len(values) + 1) / len(values)))
    result = np.zeros(len(values), dtype=bool)
    if len(accepted):
        result[order[:accepted[-1] + 1]] = True
    return result


def med_event(segment: np.ndarray, support: int) -> list[float]:
    direct = _direct_index(segment); filter_length = int(np.clip(support, 8, 64))
    output = _med_output(segment, filter_length)
    envelope = np.abs(hilbert(output))
    start = direct + max(MIN_DELAY_SAMPLES, support); stop = min(len(segment), direct + CHANNEL_SAMPLES)
    if stop <= start + 4:
        return []
    peaks, _ = find_peaks(envelope[start:stop], distance=max(1, support))
    indices = start + peaks
    keep = _bh_fdr(envelope[indices])
    return [float((index - direct) / FS * 1e3) for index in indices[keep]]


def fuse_event_candidates(event_delays: list[list[float]], *, mask_scores: np.ndarray | None = None) -> tuple[list[float], np.ndarray]:
    """Cluster event candidates without a path-count cap, retaining 6/10 support."""
    rows = [(event, delay) for event, values in enumerate(event_delays) for delay in values if delay > 0]
    clusters: list[list[tuple[int, float]]] = []
    for event, delay in sorted(rows, key=lambda item: item[1]):
        eligible = [(abs(delay - float(np.median([value for _, value in cluster]))), index)
                    for index, cluster in enumerate(clusters) if event not in {item[0] for item in cluster}
                    and abs(delay - float(np.median([value for _, value in cluster]))) <= TOLERANCE_MS]
        if eligible:
            clusters[min(eligible)[1]].append((event, delay))
        else:
            clusters.append([(event, delay)])
    stable = [float(np.median([delay for _, delay in cluster])) for cluster in clusters
              if len({event for event, _ in cluster}) >= STABLE_EVENT_COUNT]
    mask = np.zeros(CHANNEL_SAMPLES, dtype=np.uint8)
    for delay in stable:
        index = int(round(delay * 1e-3 * FS))
        if 0 < index < len(mask):
            mask[index] = 1
    if mask_scores is not None:
        mask = np.maximum(mask, np.asarray(mask_scores, dtype=np.uint8))
    return sorted(stable), mask


def classical_prediction(prepared: Prepared, method: str) -> tuple[list[list[float]], list[float], np.ndarray]:
    if method not in {"cepstrum_cfar", "med"}:
        raise ValueError(f"unknown classical method {method}")
    function = cepstrum_cfar_event if method == "cepstrum_cfar" else med_event
    event_delays = [function(segment, int(support)) for segment, support in zip(prepared.segments, prepared.support_samples)]
    stable, mask = fuse_event_candidates(event_delays)
    return event_delays, stable, mask
