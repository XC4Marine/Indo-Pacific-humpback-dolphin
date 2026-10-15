#!/usr/bin/env python3
"""Truth-isolated 1,100-event single-click BCSC ablation for Experiment 20 v4."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bcsc_v4 import FitConfig, detect_event_segments, fit_single_segment
from run_bcsc_v4 import DATA_ROOT, OUTPUT_ROOT, atomic_csv, atomic_json, bic_score, calibration_paths, discover_inputs


METHOD = "single_click"


def load_blind_segments(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only a clicktrain waveform and produce the ten observed event windows."""
    return detect_event_segments(np.load(path, allow_pickle=False))


def tune_single(paths: list[Path]) -> FitConfig:
    choices: list[dict[str, Any]] = []
    for index, lambda_ratio in enumerate((0.04, 0.06, 0.075, 0.09, 0.12), start=1):
        config = FitConfig(lambda_ratio=lambda_ratio, outer_iterations=10, code_iterations=24, source_iterations=3)
        scores: list[float] = []
        failures = 0
        for path in paths:
            segments, _ = load_blind_segments(path)
            for segment in segments:
                try:
                    scores.append(bic_score(fit_single_segment(segment, config)))
                except Exception:
                    failures += 1
        choices.append({"lambda_ratio": lambda_ratio, "group_ratio": 0.0, "mean_bic": float(np.mean(scores)) if scores else float("inf"), "successes": len(scores), "failures": failures})
        print(f"[总体进度 1/3：单click校准] {index}/5 | lambda={lambda_ratio:.3f}", flush=True)
    selected = min(choices, key=lambda row: (row["mean_bic"], row["failures"]))
    directory = OUTPUT_ROOT / "calibration"; directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / f"{METHOD}.json", {"paths": [str(path) for path in paths], "event_count": len(paths) * 10, "choices": choices, "selected": selected})
    return FitConfig(lambda_ratio=float(selected["lambda_ratio"]))


def save_train(train_id: str, starts: np.ndarray, results: list[dict[str, Any]], config: FitConfig) -> dict[str, Any]:
    directory = OUTPUT_ROOT / METHOD; directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{train_id}.npz"
    np.savez_compressed(artifact, source=np.stack([item["source"] for item in results]), channels=np.stack([item["channels"][0] for item in results]), reconstructions=np.stack([item["reconstructions"][0] for item in results]), detected_window_starts=starts)
    event_history = [{"event_index": index + 1, "history": item["history"], "constraint": item["constraint"], "converged": item["converged"], "iterations": item["iterations"]} for index, item in enumerate(results)]
    history = directory / f"{train_id}.history.json"; atomic_json(history, {"config": config.as_dict(), "events": event_history})
    constraints = [item["constraint"] for item in results]
    return {"train_id": train_id, "method": METHOD, "status": "success", "artifact": str(artifact), "history": str(history), "event_count": 10, "mean_iterations": float(np.mean([item["iterations"] for item in results])), "mean_explained_variance": float(np.mean([item["explained_variance"][0] for item in results])), "all_source_length_116": all(item["source_length"] == 116 for item in constraints), "all_bandwidth_pass": all(item["bandwidth_pass"] for item in constraints), "min_source_l2_norm": float(min(item["source_l2_norm"] for item in constraints)), "max_source_l2_norm": float(max(item["source_l2_norm"] for item in constraints))}


def run(paths: list[Path], config: FitConfig, resume: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directory = OUTPUT_ROOT / METHOD; directory.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        train_id = path.parent.name; artifact = directory / f"{train_id}.npz"
        if resume and artifact.exists():
            rows.append({"train_id": train_id, "method": METHOD, "status": "resumed", "artifact": str(artifact)}); continue
        started = time.monotonic()
        try:
            segments, starts = load_blind_segments(path)
            results = [fit_single_segment(segment, config) for segment in segments]
            row = save_train(train_id, starts, results, config)
            row["elapsed_seconds"] = time.monotonic() - started
        except Exception as exc:
            row = {"train_id": train_id, "method": METHOD, "status": "failed", "error": repr(exc), "elapsed_seconds": time.monotonic() - started}
        rows.append(row); atomic_csv(directory / "manifest.csv", rows)
        print(f"[总体进度 2/3：单click全量提取] {index:3d}/{len(paths)} | {train_id} | {row['status']}", flush=True)
    atomic_csv(directory / "manifest.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "extract", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    paths = discover_inputs(); paths = paths if args.limit is None else paths[:args.limit]
    calibration_file = OUTPUT_ROOT / "calibration" / f"{METHOD}.json"
    if args.phase in ("calibrate", "all") or not calibration_file.exists():
        config = tune_single(calibration_paths(paths, min(10, len(paths))))
    else:
        selected = json.loads(calibration_file.read_text(encoding="utf-8"))["selected"]
        config = FitConfig(lambda_ratio=float(selected["lambda_ratio"]))
    if args.phase in ("extract", "all"):
        rows = run(paths, config, resume=not args.no_resume)
        atomic_csv(OUTPUT_ROOT / f"{METHOD}_manifest.csv", rows)
    print("[总体进度 3/3] 单click消融提取完成。请运行 evaluate_bcsc_v4.py。", flush=True)


if __name__ == "__main__":
    main()
