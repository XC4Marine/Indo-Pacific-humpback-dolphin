#!/usr/bin/env python3
"""Truth-isolated full AWGN robustness extraction for Experiment 20 v4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from bcsc_v4 import FitConfig, detect_event_segments, fit_segments, fit_single_segment
from run_bcsc_v4 import OUTPUT_ROOT, assert_truth_isolation, atomic_json, discover_inputs


ROOT = Path(__file__).resolve().parent
NOISE_OUTPUT_ROOT = ROOT / "outputs_noise"
METHODS = ("single_click", "independent", "shared_support")
SNR_LEVELS_DB = tuple(range(-15, 16, 2))
NOISE_PROTOCOL_VERSION = "awgn_rms_v1"


def write_manifest(rows: list[dict[str, Any]], snr_db: int) -> None:
    """Persist one SNR's diagnostics, enabling race-free parallel SNR workers."""
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    path = NOISE_OUTPUT_ROOT / snr_tag(snr_db) / "extraction_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def snr_tag(snr_db: int) -> str:
    return f"snr_{'m' if snr_db < 0 else 'p'}{abs(snr_db):02d}"


def noise_seed(train_id: str, snr_db: int) -> int:
    """Stable per-condition seed; no manifest or truth asset is consulted."""
    digest = hashlib.sha256(f"{NOISE_PROTOCOL_VERSION}|{train_id}|{snr_db}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def add_awgn(clicktrain: np.ndarray, snr_db: int, seed: int) -> tuple[np.ndarray, dict[str, float | int]]:
    """Add zero-mean Gaussian noise at an exact whole-clicktrain RMS SNR."""
    signal = np.asarray(clicktrain, dtype=np.float64).reshape(-1)
    signal_rms = float(np.sqrt(np.mean(signal * signal)))
    if signal_rms <= np.finfo(float).eps:
        raise ValueError("cannot assign SNR to a zero-energy clicktrain")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=signal.shape)
    noise -= np.mean(noise)
    target_noise_rms = signal_rms * 10.0 ** (-float(snr_db) / 20.0)
    noise *= target_noise_rms / (np.sqrt(np.mean(noise * noise)) + np.finfo(float).eps)
    noisy = signal + noise
    realised_snr_db = 20.0 * np.log10(signal_rms / (np.sqrt(np.mean(noise * noise)) + np.finfo(float).eps))
    return noisy, {"noise_seed": int(seed), "signal_rms": signal_rms, "noise_rms": float(np.sqrt(np.mean(noise * noise))), "realised_snr_db": float(realised_snr_db)}


def fit_noisy_clicktrain(clicktrain: np.ndarray, method: str, config: FitConfig) -> dict[str, Any]:
    """Fit a noisy waveform through the same blind detector and BCSC solvers."""
    segments, starts = detect_event_segments(clicktrain)
    if method != "single_click":
        result = fit_segments(segments, config)
        result["detected_window_starts"] = starts
        return result
    event_results = [fit_single_segment(segment, config) for segment in segments]
    constraints = [result["constraint"] for result in event_results]
    return {
        "source": np.stack([result["source"] for result in event_results]),
        "channels": np.stack([result["channels"][0] for result in event_results]),
        "reconstructions": np.stack([result["reconstructions"][0] for result in event_results]),
        "detected_window_starts": starts,
        "history": [result["history"] for result in event_results],
        "constraint": {
            "source_length": 116,
            "source_l2_norm_min": float(min(item["source_l2_norm"] for item in constraints)),
            "source_l2_norm_max": float(max(item["source_l2_norm"] for item in constraints)),
            "bandwidth_pass": bool(all(item["bandwidth_pass"] for item in constraints)),
        },
        "converged": bool(all(result["converged"] for result in event_results)),
        "iterations": int(max(result["iterations"] for result in event_results)),
        "explained_variance": np.asarray([result["explained_variance"][0] for result in event_results]),
    }


def load_configs() -> dict[str, FitConfig]:
    configs: dict[str, FitConfig] = {}
    for method in METHODS:
        calibration = OUTPUT_ROOT / "calibration" / f"{method}.json"
        if not calibration.exists():
            raise FileNotFoundError(f"missing clean-run calibration: {calibration}")
        selected = json.loads(calibration.read_text(encoding="utf-8"))["selected"]
        configs[method] = FitConfig(lambda_ratio=float(selected["lambda_ratio"]), group_ratio=float(selected["group_ratio"]))
    return configs


def save_result(directory: Path, train_id: str, method: str, snr_db: int, result: dict[str, Any], config: FitConfig, noise: dict[str, float | int]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{train_id}.npz"
    np.savez_compressed(artifact, source=result["source"], channels=result["channels"], reconstructions=result["reconstructions"], detected_window_starts=result["detected_window_starts"], noise_snr_db=np.asarray(snr_db), noise_seed=np.asarray(noise["noise_seed"]))
    history = directory / f"{train_id}.history.json"
    atomic_json(history, {"method": method, "snr_db": snr_db, "noise": noise, "config": config.as_dict(), "history": result["history"], "constraint": result["constraint"], "converged": result["converged"], "iterations": result["iterations"]})
    return {"train_id": train_id, "method": method, "snr_db": snr_db, "status": "success", "artifact": str(artifact), "history": str(history), "iterations": result["iterations"], "converged": result["converged"], "mean_explained_variance": float(np.mean(result["explained_variance"])), **noise, **result["constraint"]}


def run(paths: list[Path], snrs: tuple[int, ...], configs: dict[str, FitConfig], resume: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(paths) * len(snrs) * len(METHODS)
    completed = 0
    for snr_db in snrs:
        for path in paths:
            train_id = path.parent.name
            seed = noise_seed(train_id, snr_db)
            noisy, noise = add_awgn(np.load(path, allow_pickle=False), snr_db, seed)
            for method in METHODS:
                completed += 1
                directory = NOISE_OUTPUT_ROOT / snr_tag(snr_db) / method
                artifact = directory / f"{train_id}.npz"
                if resume and artifact.exists():
                    row = {"train_id": train_id, "method": method, "snr_db": snr_db, "status": "resumed", "artifact": str(artifact), **noise}
                else:
                    started = time.monotonic()
                    try:
                        result = fit_noisy_clicktrain(noisy, method, configs[method])
                        row = save_result(directory, train_id, method, snr_db, result, configs[method], noise)
                        row["elapsed_seconds"] = time.monotonic() - started
                    except Exception as exc:
                        row = {"train_id": train_id, "method": method, "snr_db": snr_db, "status": "failed", "error": repr(exc), "elapsed_seconds": time.monotonic() - started, **noise}
                rows.append(row)
                write_manifest([item for item in rows if item["snr_db"] == snr_db], snr_db)
                print(f"[总体进度：噪声全量提取] {completed:4d}/{total} | SNR={snr_db:>3} dB | {method} | {train_id} | {row['status']}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test count only; omit for all 110 clicktrains")
    parser.add_argument("--snr", type=int, nargs="*", choices=SNR_LEVELS_DB, default=None, help="optional SNR subset; omit for all 16 levels")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_truth_isolation()
    paths = discover_inputs()
    if args.limit is not None:
        paths = paths[:args.limit]
    snrs = tuple(args.snr) if args.snr is not None else SNR_LEVELS_DB
    configs = load_configs()
    NOISE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(NOISE_OUTPUT_ROOT / "run_metadata.json", {"protocol": NOISE_PROTOCOL_VERSION, "input_count": len(paths), "snr_levels_db": snrs, "methods": METHODS, "noise_definition": "zero-mean AWGN calibrated to the unfiltered whole-clicktrain RMS", "truth_isolation": True, "configs": {method: config.as_dict() for method, config in configs.items()}})
    rows = run(paths, snrs, configs, resume=not args.no_resume)
    for snr_db in snrs:
        write_manifest([item for item in rows if item["snr_db"] == snr_db], snr_db)
    print("[总体进度：噪声全量提取] 完成。请运行 evaluate_noise_robustness_v4.py。", flush=True)


if __name__ == "__main__":
    main()
