#!/usr/bin/env python3
"""Truth-isolated consensus support extraction from v4 independent BCSC fits.

This entry point deliberately consumes only the v4 blind fit, the raw clicktrain,
and its detected windows.  It holds the extracted source fixed and therefore
reports conditional residual-bootstrap stability, not full BCSC uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
V4_ROOT = ROOT.parent / "experiment_v4"
sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import CHANNEL_LENGTH, FS, SEGMENT_LENGTH, FitConfig, bandpass, path_candidates, reconstruct, solve_independent_codes  # noqa: E402

DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
V4_INPUT_ROOT = V4_ROOT / "outputs" / "independent"
OUTPUT_ROOT = ROOT / "outputs_independent_consensus"
PROTOCOL = "v6_independent_consensus_conditional_residual_bootstrap_v1"
EVENT_COUNT = 10
STABLE_EVENT_COUNT = 6
TOLERANCE_MS = 0.03
BLOCK_MS = 0.10
EPSILON = 1e-18
MAX_PATHS = 6


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
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "little")


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    return float((p + z * z / (2 * total) - z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denominator)


def circular_block_bootstrap(residual: np.ndarray, block_samples: int, rng: np.random.Generator) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    starts = rng.integers(0, len(residual), size=int(np.ceil(len(residual) / block_samples)))
    indices = np.concatenate([(start + np.arange(block_samples)) % len(residual) for start in starts])[:len(residual)]
    return residual[indices]


def input_paths(limit: int | None) -> list[Path]:
    paths = sorted(DATA_ROOT.glob("*/clicktrain_raw.npy"))
    if len(paths) != 110:
        raise RuntimeError(f"expected 110 input clicktrains, found {len(paths)}")
    return paths if limit is None else paths[:limit]


def load_v4_fit(train_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FitConfig]:
    artifact = V4_INPUT_ROOT / f"{train_id}.npz"
    history = V4_INPUT_ROOT / f"{train_id}.history.json"
    if not artifact.exists() or not history.exists():
        raise FileNotFoundError(f"missing v4 independent fit for {train_id}")
    with np.load(artifact, allow_pickle=False) as payload:
        source = np.asarray(payload["source"], dtype=float)
        channels = np.asarray(payload["channels"], dtype=float)
        fitted = np.asarray(payload["reconstructions"], dtype=float)
        starts = np.asarray(payload["detected_window_starts"], dtype=np.int64)
    config_row = json.loads(history.read_text(encoding="utf-8"))["config"]
    config = FitConfig(**{key: config_row[key] for key in FitConfig.__dataclass_fields__ if key in config_row})
    if source.ndim != 1 or channels.shape != (EVENT_COUNT, CHANNEL_LENGTH) or fitted.shape != (EVENT_COUNT, SEGMENT_LENGTH) or starts.shape != (EVENT_COUNT,):
        raise ValueError(f"unexpected v4 independent schema for {train_id}")
    return source, channels, fitted, starts, config


def load_segments(clicktrain_path: Path, starts: np.ndarray) -> np.ndarray:
    waveform = bandpass(np.load(clicktrain_path, allow_pickle=False))
    windows = []
    for start in starts:
        segment = waveform[int(start):int(start) + SEGMENT_LENGTH]
        windows.append(np.pad(segment, (0, max(0, SEGMENT_LENGTH - len(segment)))))
    return np.asarray(windows, dtype=float)


def event_paths(channels: np.ndarray) -> list[list[dict[str, float]]]:
    """Return path candidates in timing relative to each event's direct peak."""
    result: list[list[dict[str, float]]] = []
    for channel in channels:
        paths = path_candidates(channel)
        if not paths:
            peak = int(np.argmax(np.abs(channel)))
            paths = [{"sample": peak, "delay_ms": 0.0, "coefficient": float(channel[peak]), "relative_amplitude": 1.0}]
        result.append(paths)
    return result


def _cluster_rows(rows: list[tuple[int, dict[str, float]]]) -> list[list[tuple[int, dict[str, float]]]]:
    """Greedy one-observation-per-event delay clusters, constrained by tolerance."""
    clusters: list[list[tuple[int, dict[str, float]]]] = []
    for event, path in sorted(rows, key=lambda row: float(row[1]["delay_ms"])):
        choices = []
        for index, cluster in enumerate(clusters):
            median = float(np.median([item[1]["delay_ms"] for item in cluster]))
            if event not in {item[0] for item in cluster} and abs(float(path["delay_ms"]) - median) <= TOLERANCE_MS:
                choices.append((abs(float(path["delay_ms"]) - median), index))
        if choices:
            clusters[min(choices)[1]].append((event, path))
        else:
            clusters.append([(event, path)])
    return clusters


def build_candidates(paths_by_event: list[list[dict[str, float]]]) -> list[dict[str, Any]]:
    rows = [(event, path) for event, paths in enumerate(paths_by_event) for path in paths]
    clusters = _cluster_rows(rows)
    candidates = []
    for cluster in clusters:
        delays = np.asarray([row[1]["delay_ms"] for row in cluster], dtype=float)
        is_direct = bool(np.all(np.abs(delays) <= TOLERANCE_MS))
        if len(cluster) >= STABLE_EVENT_COUNT or is_direct:
            by_event = {event: path for event, path in cluster}
            candidates.append({"is_direct": is_direct, "event_paths": by_event,
                               "delay_median_ms": float(np.median(delays)),
                               "delay_q025_ms": float(np.quantile(delays, .025)),
                               "delay_q975_ms": float(np.quantile(delays, .975)),
                               "event_coverage": len(by_event) / EVENT_COUNT,
                               "event_detected_count": len(by_event)})
    # Direct path has a privileged role even if numerical candidate extraction makes it absent.
    if not any(row["is_direct"] for row in candidates):
        direct = [{"sample": paths_by_event[event][0]["sample"],
                   "delay_ms": 0.0, "coefficient": paths_by_event[event][0]["coefficient"], "relative_amplitude": 1.0}
                  for event in range(EVENT_COUNT)]
        candidates.append({"is_direct": True, "event_paths": dict(enumerate(direct)), "delay_median_ms": 0.0,
                           "delay_q025_ms": 0.0, "delay_q975_ms": 0.0, "event_coverage": 1.0, "event_detected_count": EVENT_COUNT})
    candidates.sort(key=lambda row: (not row["is_direct"], row["delay_median_ms"]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["path_id"] = index
    return candidates


def match_candidate(candidate: dict[str, Any], paths: list[dict[str, float]]) -> dict[str, float] | None:
    matches = [path for path in paths if abs(float(path["delay_ms"]) - float(candidate["delay_median_ms"])) <= TOLERANCE_MS]
    return min(matches, key=lambda path: abs(float(path["delay_ms"]) - float(candidate["delay_median_ms"]))) if matches else None


def bootstrap_fixed_source(source: np.ndarray, fitted: np.ndarray, segments: np.ndarray, config: FitConfig,
                           candidates: list[dict[str, Any]], count: int, train_id: str,
                           progress: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
    path_count = len(candidates)
    detected = np.zeros((count, path_count, EVENT_COUNT), dtype=bool)
    delays = np.full((count, path_count, EVENT_COUNT), np.nan, dtype=float)
    amplitudes = np.full((count, path_count, EVENT_COUNT), np.nan, dtype=float)
    residuals = segments - fitted
    block = max(2, int(round(BLOCK_MS * 1e-3 * FS)))
    rng = np.random.default_rng(stable_seed(PROTOCOL, train_id, count))
    for replicate in range(count):
        synthetic = np.stack([fit + circular_block_bootstrap(residual, block, rng) for fit, residual in zip(fitted, residuals)])
        codes = solve_independent_codes(synthetic, source, np.zeros((EVENT_COUNT, CHANNEL_LENGTH)), config.lambda_ratio, config.code_iterations)
        recovered = event_paths(codes)
        for candidate_index, candidate in enumerate(candidates):
            hits = []
            for event, event_candidates in enumerate(recovered):
                match = match_candidate(candidate, event_candidates)
                if match is not None:
                    hits.append(match)
                    delays[replicate, candidate_index, event] = float(match["delay_ms"])
                    amplitudes[replicate, candidate_index, event] = float(match["relative_amplitude"])
            # A replicate supports the common structure only if six simultaneous event detections occur.
            if len(hits) >= STABLE_EVENT_COUNT:
                for event, event_candidates in enumerate(recovered):
                    match = match_candidate(candidate, event_candidates)
                    detected[replicate, candidate_index, event] = match is not None
        if progress:
            progress(replicate + 1, count)
    return {"detected": detected, "delay_ms": delays, "relative_amplitude": amplitudes}


def support_design(source: np.ndarray, samples: list[int], length: int) -> np.ndarray:
    design = np.zeros((length, len(samples)), dtype=float)
    for column, sample in enumerate(samples):
        start, stop = max(0, sample), min(length, sample + len(source))
        if stop > start:
            design[start:stop, column] = source[start - sample:stop - sample]
    return design


def event_samples_for_candidate(candidate: dict[str, Any], direct: dict[str, Any]) -> list[int]:
    values = []
    for event in range(EVENT_COUNT):
        observed = candidate["event_paths"].get(event)
        direct_path = direct["event_paths"].get(event)
        if observed is not None:
            values.append(int(observed["sample"]))
        elif direct_path is not None:
            values.append(int(round(float(direct_path["sample"]) + float(candidate["delay_median_ms"]) * FS / 1e3)))
        else:
            values.append(int(round(float(candidate["delay_median_ms"]) * FS / 1e3)))
    return values


def bic_select(source: np.ndarray, segments: np.ndarray, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, np.ndarray, np.ndarray]:
    direct = next(row for row in candidates if row["is_direct"])
    reflections = sorted((row for row in candidates if not row["is_direct"]), key=lambda row: (-float(row["joint_confidence"]), -float(row["event_coverage"])))[:MAX_PATHS - 1]
    pool = [direct, *reflections]
    best: tuple[float, list[dict[str, Any]], np.ndarray, np.ndarray] | None = None
    n = int(np.prod(segments.shape))
    for size in range(1, len(pool) + 1):
        for subset in itertools.combinations(pool[1:], size - 1):
            support = [direct, *subset]
            coefficients = np.zeros((EVENT_COUNT, len(support)), dtype=float)
            reconstructed = np.zeros_like(segments)
            for event in range(EVENT_COUNT):
                samples = [event_samples_for_candidate(candidate, direct)[event] for candidate in support]
                design = support_design(source, samples, segments.shape[1])
                coefficients[event], *_ = np.linalg.lstsq(design, segments[event], rcond=None)
                reconstructed[event] = design @ coefficients[event]
            rss = float(np.sum((segments - reconstructed) ** 2))
            bic = n * np.log(rss / n + EPSILON) + EVENT_COUNT * len(support) * np.log(n)
            proposal = (float(bic), support, coefficients, reconstructed)
            if best is None or proposal[0] < best[0]:
                best = proposal
    assert best is not None
    return best[1], best[0], best[2], best[3]


def summarise_candidates(candidates: list[dict[str, Any]], bootstrap: dict[str, np.ndarray], count: int) -> None:
    for index, candidate in enumerate(candidates):
        joint = int(np.sum(np.sum(bootstrap["detected"][:, index, :], axis=1) >= STABLE_EVENT_COUNT))
        candidate["joint_detected_runs"] = joint
        candidate["joint_confidence"] = joint / count
        candidate["wilson_lower_95"] = wilson_lower(joint, count)
        values = bootstrap["delay_ms"][:, index, :][bootstrap["detected"][:, index, :]]
        if len(values):
            candidate["bootstrap_delay_median_ms"] = float(np.median(values))
            candidate["bootstrap_delay_q025_ms"] = float(np.quantile(values, .025))
            candidate["bootstrap_delay_q975_ms"] = float(np.quantile(values, .975))
        else:
            candidate["bootstrap_delay_median_ms"] = candidate["delay_median_ms"]
            candidate["bootstrap_delay_q025_ms"] = candidate["delay_q025_ms"]
            candidate["bootstrap_delay_q975_ms"] = candidate["delay_q975_ms"]


def finalise_paths(selected: list[dict[str, Any]], all_candidates: list[dict[str, Any]], coefficients: np.ndarray) -> list[dict[str, Any]]:
    direct_index = next(index for index, row in enumerate(selected) if row["is_direct"])
    rows = []
    for selected_index, candidate in enumerate(selected, start=1):
        original = next(row for row in all_candidates if row["path_id"] == candidate["path_id"])
        amplitudes = coefficients[:, selected_index - 1]
        rel = np.abs(amplitudes) / (np.abs(coefficients[:, direct_index]) + EPSILON)
        event_delay = [(sample - event_samples_for_candidate(selected[direct_index], selected[direct_index])[event]) / FS * 1e3
                       for event, sample in enumerate(event_samples_for_candidate(candidate, selected[direct_index]))]
        rows.append({"path_id": selected_index, "candidate_path_id": original["path_id"], "is_direct": bool(candidate["is_direct"]),
                     "delay_median_ms": float(np.median(event_delay)), "delay_q025_ms": float(np.quantile(event_delay, .025)), "delay_q975_ms": float(np.quantile(event_delay, .975)),
                     "joint_confidence": float(original["joint_confidence"]), "wilson_lower_95": float(original.get("wilson_lower_95", 0.0)),
                     "event_coverage": float(original["event_coverage"]), "event_detected_count": int(original["event_detected_count"]),
                     "relative_amplitude_median": float(np.median(rel)), "relative_amplitude_q25": float(np.quantile(rel, .25)), "relative_amplitude_q75": float(np.quantile(rel, .75)),
                     "event_delay_ms": [float(value) for value in event_delay], "event_coefficient": [float(value) for value in amplitudes],
                     "event_relative_amplitude": [float(value) for value in rel]})
    return rows


def serialise_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in candidate.items() if key != "event_paths"}
    result["event_paths"] = {str(event + 1): {key: float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value for key, value in path.items()}
                             for event, path in candidate["event_paths"].items()}
    return result


def run_one(clicktrain_path: Path, bootstrap_count: int, progress: Callable[[int, int], None] | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_id = clicktrain_path.parent.name
    source, channels, fitted, starts, config = load_v4_fit(train_id)
    segments = load_segments(clicktrain_path, starts)
    candidates = build_candidates(event_paths(channels))
    samples = bootstrap_fixed_source(source, fitted, segments, config, candidates, bootstrap_count, train_id, progress)
    summarise_candidates(candidates, samples, bootstrap_count)
    selected, bic, coefficients, reconstructed = bic_select(source, segments, candidates)
    final_paths = finalise_paths(selected, candidates, coefficients)
    result = {"protocol": PROTOCOL, "scope": "fixed-source conditional residual bootstrap; not full BCSC parameter uncertainty",
              "train_id": train_id, "method": "independent", "bootstrap_count": bootstrap_count, "block_ms": BLOCK_MS,
              "stable_event_threshold": STABLE_EVENT_COUNT, "delay_tolerance_ms": TOLERANCE_MS, "bic": float(bic),
              "bic_parameter_count": int(EVENT_COUNT * len(selected)), "candidate_paths": [serialise_candidate(row) for row in candidates],
              "final_paths": final_paths, "event_window_starts": [int(value) for value in starts],
              "source_length_samples": int(len(source)), "reconstruction_nrmse": float(np.linalg.norm(segments - reconstructed) / (np.linalg.norm(segments) + EPSILON))}
    samples.update({"source": source, "segments": segments, "reconstruction": reconstructed, "coefficients": coefficients,
                    "candidate_delay_ms": np.asarray([row["delay_median_ms"] for row in candidates]),
                    "candidate_path_ids": np.asarray([row["path_id"] for row in candidates])})
    return result, samples


def save_one(root: Path, train_id: str, result: dict[str, Any], samples: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    directory = root / train_id
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "consensus.json", result)
    np.savez_compressed(directory / "bootstrap_samples.npz", **samples)
    return [{"train_id": train_id, "path_id": path["path_id"], "candidate_path_id": path["candidate_path_id"], "is_direct": path["is_direct"],
             "delay_median_ms": path["delay_median_ms"], "delay_q025_ms": path["delay_q025_ms"], "delay_q975_ms": path["delay_q975_ms"],
             "joint_confidence": path["joint_confidence"], "wilson_lower_95": path["wilson_lower_95"], "event_coverage": path["event_coverage"],
             "relative_amplitude_median": path["relative_amplitude_median"], "relative_amplitude_q25": path["relative_amplitude_q25"], "relative_amplitude_q75": path["relative_amplitude_q75"], "bic": result["bic"]}
            for path in result["final_paths"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 2:
        raise ValueError("--bootstrap must be at least two")
    paths = input_paths(args.limit)
    protocol = {"protocol": PROTOCOL, "input_count": len(paths), "train_ids": [path.parent.name for path in paths], "method": "independent",
                "bootstrap_count": args.bootstrap, "stable_event_threshold": STABLE_EVENT_COUNT, "delay_tolerance_ms": TOLERANCE_MS,
                "truth_isolation": True, "scope": "fixed-source conditional residual bootstrap"}
    atomic_json(args.output_root / "protocol.json", protocol)
    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, clicktrain_path in enumerate(paths, start=1):
        train_id = clicktrain_path.parent.name
        destination = args.output_root / train_id / "consensus.json"
        started = time.monotonic()
        try:
            if destination.exists() and not args.force:
                result = json.loads(destination.read_text(encoding="utf-8"))
                status = "resumed"
                rows.extend([{**row, "train_id": train_id} for row in []])
            else:
                def progress(done: int, total: int) -> None:
                    print(f"[当前 clicktrain bootstrap] {index}/{len(paths)} | {train_id} | {done:03d}/{total}", flush=True)
                result, samples = run_one(clicktrain_path, args.bootstrap, progress)
                rows.extend(save_one(args.output_root, train_id, result, samples)); status = "success"
            manifest.append({"train_id": train_id, "status": status, "elapsed_seconds": time.monotonic() - started, "artifact": str(destination)})
        except Exception as exc:
            manifest.append({"train_id": train_id, "status": "failed", "elapsed_seconds": time.monotonic() - started, "error": repr(exc)})
        atomic_json(args.output_root / "manifest.json", manifest)
        print(f"[总体进度] {index:03d}/{len(paths)} | {train_id} | {manifest[-1]['status']}", flush=True)
    # Rebuild summary from all available result files so interrupted runs remain resumable.
    all_rows = []
    for result_path in sorted(args.output_root.glob("*/consensus.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for path in result["final_paths"]:
            all_rows.append({"train_id": result["train_id"], "path_id": path["path_id"], "candidate_path_id": path["candidate_path_id"], "is_direct": path["is_direct"],
                             "delay_median_ms": path["delay_median_ms"], "delay_q025_ms": path["delay_q025_ms"], "delay_q975_ms": path["delay_q975_ms"], "joint_confidence": path["joint_confidence"], "wilson_lower_95": path["wilson_lower_95"], "event_coverage": path["event_coverage"], "relative_amplitude_median": path["relative_amplitude_median"], "relative_amplitude_q25": path["relative_amplitude_q25"], "relative_amplitude_q75": path["relative_amplitude_q75"], "bic": result["bic"]})
    atomic_csv(args.output_root / "path_confidence.csv", all_rows)


if __name__ == "__main__":
    main()
