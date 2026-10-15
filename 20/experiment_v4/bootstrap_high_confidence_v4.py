#!/usr/bin/env python3
"""Truth-isolated conditional residual-bootstrap path stability for v4 BCSC.

Only raw clicktrains and blind BCSC outputs enter here.  The source is held fixed
and sparse channels are refit after residual-block resampling, so confidence is
conditional on the extracted source rather than a claim of full BCSC uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from bcsc_v4 import CHANNEL_LENGTH, SEGMENT_LENGTH, FitConfig, bandpass, path_candidates, solve_independent_codes, solve_shared_support_admm

ROOT = Path(__file__).resolve().parent
INPUT_ROOT = ROOT / "outputs"
OUTPUT_ROOT = ROOT / "outputs_bootstrap_path_pilot"
DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
METHODS = ("single_click", "independent", "shared_support")
PROTOCOL = "v4_conditional_residual_bootstrap_v1"
DEFAULT_BOOTSTRAPS = 50
DEFAULT_LIMIT = 12
BLOCK_MS = .10
MATCH_TOLERANCE_MS = .03
HIGH_CONFIDENCE = .90
STRICT_WILSON_LOW = .80
ADAPTIVE_MAX_CANDIDATES = 6


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(*items: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, items)).encode()).digest()[:8], "little")


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    return float((p + z * z / (2 * total) - z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denominator)


def circular_block_bootstrap(residual: np.ndarray, block_samples: int, rng: np.random.Generator) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    blocks: list[np.ndarray] = []
    while sum(map(len, blocks)) < len(residual):
        start = int(rng.integers(len(residual)))
        blocks.append(residual[(start + np.arange(block_samples)) % len(residual)])
    return np.concatenate(blocks)[:len(residual)]


def match_paths(reference: list[dict[str, float]], candidates: list[dict[str, float]]) -> list[dict[str, float] | None]:
    matches: list[dict[str, float] | None] = [None] * len(reference)
    pairs = [(abs(float(candidate["delay_ms"]) - float(ref["delay_ms"])), ref_i, candidate_i)
             for ref_i, ref in enumerate(reference) for candidate_i, candidate in enumerate(candidates)
             if abs(float(candidate["delay_ms"]) - float(ref["delay_ms"])) <= MATCH_TOLERANCE_MS]
    used: set[int] = set()
    for _, ref_i, candidate_i in sorted(pairs):
        if matches[ref_i] is None and candidate_i not in used:
            matches[ref_i] = candidates[candidate_i]
            used.add(candidate_i)
    return matches


def load_config(method: str) -> FitConfig:
    selected = json.loads((INPUT_ROOT / "calibration" / f"{method}.json").read_text(encoding="utf-8"))["selected"]
    return FitConfig(lambda_ratio=float(selected["lambda_ratio"]), group_ratio=float(selected["group_ratio"]))


def input_paths(limit: int | None) -> list[Path]:
    paths = sorted(DATA_ROOT.glob("*/clicktrain_raw.npy"), key=lambda p: (stable_seed(PROTOCOL, p.parent.name), p.parent.name))
    return paths if limit is None else paths[:limit]


def load_segments(path: Path, starts: np.ndarray) -> np.ndarray:
    waveform = bandpass(np.load(path, allow_pickle=False))
    windows = []
    for start in starts:
        segment = waveform[int(start):int(start) + SEGMENT_LENGTH]
        windows.append(np.pad(segment, (0, max(0, SEGMENT_LENGTH - len(segment)))))
    return np.asarray(windows, dtype=np.float64)


def bootstrap_channels(method: str, synthetic: np.ndarray, source: np.ndarray, config: FitConfig) -> np.ndarray:
    if method == "single_click":
        return np.stack([solve_independent_codes(synthetic[i:i + 1], source[i], np.zeros((1, CHANNEL_LENGTH)), config.lambda_ratio, 35)[0] for i in range(len(synthetic))])
    if method == "shared_support":
        return solve_shared_support_admm(synthetic, source, np.zeros((len(synthetic), CHANNEL_LENGTH)), config.lambda_ratio, config.group_ratio, 25, config.admm_rho)
    return solve_independent_codes(synthetic, source, np.zeros((len(synthetic), CHANNEL_LENGTH)), config.lambda_ratio, 35)


def candidate_summaries(reference: list[dict[str, float]], records: list[list[dict[str, float]]], count: int) -> list[dict[str, Any]]:
    """Attach bootstrap stability summaries to every blind primary-fit candidate."""
    summaries: list[dict[str, Any]] = []
    for index, (candidate, repetitions) in enumerate(zip(reference, records)):
        detected = len(repetitions); confidence = detected / count; lower = wilson_lower(detected, count)
        direct = abs(float(candidate["delay_ms"])) < 1e-12
        rows = repetitions or [candidate]
        summaries.append({"path_index": index, "sample": int(candidate["sample"]), "delay_ms": float(candidate["delay_ms"]),
                          "delay_median_ms": float(np.median([row["delay_ms"] for row in rows])),
                          "relative_amplitude": float(candidate["relative_amplitude"]),
                          "relative_amplitude_median": float(np.median([row["relative_amplitude"] for row in rows])),
                          "coefficient": float(candidate["coefficient"]), "coefficient_median": float(np.median([row["coefficient"] for row in rows])),
                          "detected_runs": detected, "bootstrap_total": count, "confidence": confidence,
                          "wilson_lower_95": lower, "is_direct": direct})
    return summaries


def strict_paths(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if candidate["is_direct"] or (candidate["confidence"] >= HIGH_CONFIDENCE and candidate["wilson_lower_95"] >= STRICT_WILSON_LOW)]


def _support_design(source: np.ndarray, samples: list[int], length: int) -> np.ndarray:
    design = np.zeros((length, len(samples)), dtype=np.float64)
    for column, sample in enumerate(samples):
        start, stop = max(0, sample), min(length, sample + len(source))
        if stop > start:
            design[start:stop, column] = source[start - sample:stop - sample]
    return design


def adaptive_bic_paths(candidates: list[dict[str, Any]], source: np.ndarray, segment: np.ndarray) -> tuple[list[dict[str, Any]], float]:
    """Choose a bootstrap-ranked sparse support by BIC without truth or path count.

    Candidate timing is supplied by blind BCSC/bootstrapping.  For every support
    subset, amplitudes are refit to the observed window, which avoids inheriting
    the severe amplitude shrinkage of the FISTA coefficients.
    """
    direct = next((candidate for candidate in candidates if candidate["is_direct"]), None)
    if direct is None:
        return [], float("inf")
    remaining = sorted((candidate for candidate in candidates if not candidate["is_direct"]), key=lambda candidate: (-float(candidate["confidence"]), -float(candidate["relative_amplitude"]), abs(float(candidate["delay_median_ms"]))))
    pool = [direct, *remaining[:ADAPTIVE_MAX_CANDIDATES - 1]]
    best_bic, best_support, best_coefficients = float("inf"), [direct], np.asarray([float(direct["coefficient_median"])])
    for count in range(1, len(pool) + 1):
        for subset in itertools.combinations(pool[1:], count - 1):
            support = [direct, *subset]
            design = _support_design(source, [int(candidate["sample"]) for candidate in support], len(segment))
            coefficients, *_ = np.linalg.lstsq(design, segment, rcond=None)
            residual = segment - design @ coefficients
            bic = len(segment) * np.log(float(np.mean(residual * residual)) + 1e-18) + len(support) * np.log(len(segment))
            if bic < best_bic:
                best_bic, best_support, best_coefficients = float(bic), support, coefficients
    direct_coefficient = float(best_coefficients[0])
    selected = []
    for candidate, coefficient in zip(best_support, best_coefficients):
        updated = dict(candidate)
        updated["coefficient_bic_refit"] = float(coefficient)
        updated["coefficient_median"] = float(coefficient)
        updated["relative_amplitude_median"] = float(abs(coefficient) / (abs(direct_coefficient) + 1e-18))
        selected.append(updated)
    return selected, best_bic


def run_one(path: Path, method: str, count: int) -> dict[str, Any]:
    train_id = path.parent.name; artifact = INPUT_ROOT / method / f"{train_id}.npz"
    with np.load(artifact, allow_pickle=False) as item:
        source, channels = np.asarray(item["source"], float), np.asarray(item["channels"], float)
        reconstructions, starts = np.asarray(item["reconstructions"], float), np.asarray(item["detected_window_starts"], int)
    segments = load_segments(path, starts)
    if channels.shape != (10, CHANNEL_LENGTH) or reconstructions.shape != (10, SEGMENT_LENGTH):
        raise ValueError(f"unexpected BCSC artifact dimensions: {artifact}")
    if (method == "single_click" and source.shape != (10, 116)) or (method != "single_click" and source.shape != (116,)):
        raise ValueError(f"unexpected source dimensions: {artifact}")
    reference = [path_candidates(channel) for channel in channels]
    records: list[list[list[dict[str, float]]]] = [[[] for _ in values] for values in reference]
    residuals = segments - reconstructions
    rng = np.random.default_rng(stable_seed(PROTOCOL, train_id, method)); block = max(2, int(round(BLOCK_MS * 1e-3 * 576_000))); config = load_config(method)
    for replicate in range(count):
        synthetic = np.stack([fitted + circular_block_bootstrap(residual, block, rng) for fitted, residual in zip(reconstructions, residuals)])
        boot = bootstrap_channels(method, synthetic, source, config)
        for event, (ref, channel) in enumerate(zip(reference, boot)):
            for path_index, match in enumerate(match_paths(ref, path_candidates(channel))):
                if match is not None:
                    records[event][path_index].append(match)
        print(f"[总体进度：路径 bootstrap] {train_id} | {method} | {replicate + 1:02d}/{count}", flush=True)
    events = []
    for index, ref in enumerate(reference):
        candidates = candidate_summaries(ref, records[index], count)
        event_source = source[index] if method == "single_click" else source
        adaptive, adaptive_bic = adaptive_bic_paths(candidates, event_source, segments[index])
        strict = strict_paths(candidates)
        events.append({"event_index": index + 1, "reference_path_count": len(ref), "candidate_path_count": len(candidates),
                       "stable_path_count": len(strict), "stable_paths": strict, "candidate_paths": candidates,
                       "adaptive_bic_path_count": len(adaptive), "adaptive_bic": adaptive_bic, "adaptive_bic_paths": adaptive})
    return {"protocol": PROTOCOL, "scope": "conditional_residual_bootstrap_fixed_source", "method": method, "train_id": train_id,
            "bootstrap_count": count, "block_ms": BLOCK_MS, "match_tolerance_ms": MATCH_TOLERANCE_MS,
            "high_confidence_threshold": HIGH_CONFIDENCE, "strict_wilson_lower_95": STRICT_WILSON_LOW, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="deterministic pilot count; use 110 for a full run")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=METHODS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="separate output root; preserves existing pilot results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 2: raise ValueError("bootstrap count must be at least two")
    paths = input_paths(args.limit)
    if not paths: raise RuntimeError("no clicktrain inputs")
    output_root = args.output_root
    atomic_json(output_root / "protocol.json", {"protocol": PROTOCOL, "input_count": len(paths), "train_ids": [p.parent.name for p in paths], "methods": args.methods, "bootstrap_count": args.bootstrap, "truth_isolation": True, "scope": "fixed-source conditional residual bootstrap plus blind BIC support selection", "selection": "adaptive_bic_consensus"})
    total = len(paths) * len(args.methods); completed = 0
    for method in args.methods:
        for path in paths:
            completed += 1; output = output_root / method / f"{path.parent.name}.json"
            if output.exists() and not args.force: status = "resumed"
            else: atomic_json(output, run_one(path, method, args.bootstrap)); status = "success"
            print(f"[总体进度：高置信多径试验] {completed:02d}/{total} | {method} | {path.parent.name} | {status}", flush=True)


if __name__ == "__main__": main()
