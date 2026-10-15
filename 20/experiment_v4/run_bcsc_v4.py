#!/usr/bin/env python3
"""Run the truth-isolated full Experiment 20 v4 BCSC extraction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from bcsc_v4 import FitConfig, fit_clicktrain


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
OUTPUT_ROOT = ROOT / "outputs"
EXTRACTION_FORBIDDEN_NAMES = {"event_manifest.csv", "original_click.wav", "original_channel.npz", "event_channels.npz"}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def discover_inputs() -> list[Path]:
    paths = sorted(DATA_ROOT.glob("*/clicktrain_raw.npy"))
    if len(paths) != 110:
        raise RuntimeError(f"expected 110 clicktrains, found {len(paths)}")
    return paths


def assert_truth_isolation() -> None:
    """Static guard: extraction package must never open a truth asset by name."""
    # The runner stores the denylist as a policy constant.  Inspect the numerical
    # extraction core, where an actual truth-file read would be a violation.
    sources = (ROOT / "bcsc_v4.py").read_text(encoding="utf-8")
    leaked = [name for name in EXTRACTION_FORBIDDEN_NAMES if name in sources]
    if leaked:
        raise RuntimeError(f"truth-isolation violation in extraction sources: {leaked}")


def calibration_paths(paths: list[Path], count: int = 10) -> list[Path]:
    """A fixed waveform-name hash sample; no truth field is consulted."""
    ranked = sorted(paths, key=lambda path: hashlib.sha256(path.parent.name.encode()).hexdigest())
    return ranked[:count]


def bic_score(result: dict[str, Any]) -> float:
    residual = max(float(result["history"][-1]["data"]), 1e-18)
    active = int(np.count_nonzero(np.abs(result["channels"]) > 1e-10)) + 116
    observations = int(np.prod(result["reconstructions"].shape))
    return float(observations * np.log(residual / observations + 1e-18) + active * np.log(observations))


def tune(paths: list[Path], method: str) -> FitConfig:
    lambda_grid = (0.04, 0.06, 0.075, 0.09, 0.12)
    group_grid = (0.25, 0.50, 1.00) if method == "shared_support" else (0.0,)
    fast = FitConfig(outer_iterations=10, code_iterations=24, source_iterations=3, admm_iterations=5)
    choices: list[dict[str, Any]] = []
    total = len(lambda_grid) * len(group_grid)
    done = 0
    for lambda_ratio in lambda_grid:
        for group_ratio in group_grid:
            config = replace(fast, lambda_ratio=lambda_ratio, group_ratio=group_ratio)
            scores = []
            failures = 0
            for path in paths:
                try:
                    scores.append(bic_score(fit_clicktrain(path, config)))
                except Exception:
                    failures += 1
            choices.append({"lambda_ratio": lambda_ratio, "group_ratio": group_ratio, "mean_bic": float(np.mean(scores)) if scores else float("inf"), "successes": len(scores), "failures": failures})
            done += 1
            print(f"[总体进度 1/3：校准] {method} {done}/{total} | lambda={lambda_ratio:.3f} group={group_ratio:.2f}", flush=True)
    winner = min(choices, key=lambda row: (row["mean_bic"], row["failures"]))
    (OUTPUT_ROOT / "calibration").mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_ROOT / "calibration" / f"{method}.json", {"paths": [str(path) for path in paths], "choices": choices, "selected": winner})
    return FitConfig(lambda_ratio=float(winner["lambda_ratio"]), group_ratio=float(winner["group_ratio"]))


def save_result(output: Path, train_id: str, method: str, result: dict[str, Any], config: FitConfig) -> dict[str, Any]:
    directory = output / method
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{train_id}.npz"
    np.savez_compressed(artifact, source=result["source"], channels=result["channels"], reconstructions=result["reconstructions"], detected_window_starts=result["detected_window_starts"])
    history_path = directory / f"{train_id}.history.json"
    atomic_json(history_path, {"config": config.as_dict(), "history": result["history"], "constraint": result["constraint"], "converged": result["converged"], "iterations": result["iterations"]})
    return {"train_id": train_id, "method": method, "status": "success", "artifact": str(artifact), "history": str(history_path), "iterations": result["iterations"], "converged": result["converged"], "mean_explained_variance": float(np.mean(result["explained_variance"])), **result["constraint"]}


def run_method(paths: list[Path], method: str, config: FitConfig, resume: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directory = OUTPUT_ROOT / method
    directory.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        train_id = path.parent.name
        artifact = directory / f"{train_id}.npz"
        if resume and artifact.exists():
            rows.append({"train_id": train_id, "method": method, "status": "resumed", "artifact": str(artifact)})
            continue
        started = time.monotonic()
        try:
            result = fit_clicktrain(path, config)
            row = save_result(OUTPUT_ROOT, train_id, method, result, config)
            row["elapsed_seconds"] = time.monotonic() - started
        except Exception as exc:
            row = {"train_id": train_id, "method": method, "status": "failed", "error": repr(exc), "elapsed_seconds": time.monotonic() - started}
        rows.append(row)
        atomic_csv(directory / "manifest.csv", rows)
        print(f"[总体进度 2/3：全量提取] {method} {index:3d}/{len(paths)} | {train_id} | {row['status']}", flush=True)
    atomic_csv(directory / "manifest.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "extract", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="only for a smoke run; omit for all 110 inputs")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_truth_isolation()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = discover_inputs()
    if args.limit is not None:
        paths = paths[:args.limit]
    metadata = {"input_count": len(paths), "data_root": str(DATA_ROOT), "truth_isolation": True, "source_length_samples": 116, "channel_window_ms": 4.0}
    atomic_json(OUTPUT_ROOT / "run_metadata.json", metadata)
    calibration = calibration_paths(paths, min(10, len(paths)))
    configs: dict[str, FitConfig] = {}
    for method in ("independent", "shared_support"):
        calibration_file = OUTPUT_ROOT / "calibration" / f"{method}.json"
        if args.phase in ("calibrate", "all") or not calibration_file.exists():
            configs[method] = tune(calibration, method)
        else:
            selected = json.loads(calibration_file.read_text(encoding="utf-8"))["selected"]
            configs[method] = FitConfig(lambda_ratio=float(selected["lambda_ratio"]), group_ratio=float(selected["group_ratio"]))
    if args.phase in ("extract", "all"):
        all_rows = []
        for method, config in configs.items():
            all_rows.extend(run_method(paths, method, config, resume=not args.no_resume))
        atomic_csv(OUTPUT_ROOT / "extraction_manifest.csv", all_rows)
    print("[总体进度 3/3] 提取阶段完成。请运行 evaluate_bcsc_v4.py 生成隔离的真值验证。", flush=True)


if __name__ == "__main__":
    main()
