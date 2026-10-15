#!/usr/bin/env python3
"""Truth-only evaluation for v7 source, time-channel, and click reconstruction."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent; ROOT20 = HERE.parent; V4 = ROOT20 / "experiment_v4"; V5 = ROOT20 / "experiment_v5"
sys.path[:0] = [str(V5), str(V4), str(HERE)]
from protocol_v5 import DATA_ROOT, discover_inputs, group_split  # noqa: E402
from bcsc_v4 import bandpass  # noqa: E402
from evaluate_bcsc_v4 import align_source  # noqa: E402
from training_v7 import effective_channels, truth_source  # noqa: E402

OUT = HERE / "outputs"; SEGMENT_LENGTH = 2419


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def rel_error(estimate: np.ndarray, target: np.ndarray) -> float:
    scale = float(np.dot(target, estimate) / (np.dot(estimate, estimate) + 1e-18))
    return float(np.linalg.norm(scale * estimate - target) / (np.linalg.norm(target) + 1e-18))


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); draws = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(2000)]
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    rows: list[dict[str, object]] = []; split = [row for row in group_split(discover_inputs()) if row["split"] == "test"]
    for index, row in enumerate(split, 1):
        train_id = str(row["train_id"]); directory = DATA_ROOT / train_id
        base = np.load(V5 / "outputs" / "predictions" / "clean" / "v4_independent" / f"{train_id}.npz")
        v7 = np.load(OUT / "blind_predictions" / "clean" / f"{train_id}.npz")
        raw = bandpass(np.load(directory / "clicktrain_raw.npy", allow_pickle=False).astype(float)); starts = np.asarray(base["detected_window_starts"], dtype=int)
        observed = np.stack([np.pad(raw[int(start):int(start) + SEGMENT_LENGTH], (0, max(0, int(start) + SEGMENT_LENGTH - len(raw))))[:SEGMENT_LENGTH] for start in starts])
        target_source = truth_source(directory); target_channels = effective_channels(directory, starts, target_source, observed)
        for method, item in (("v4_independent", base), ("v7_time_refiner", v7)):
            source, _ = align_source(np.asarray(item["source"]), target_source)
            source_error = float(np.linalg.norm(source - target_source) / (np.linalg.norm(target_source) + 1e-18))
            channels, reconstruction = np.asarray(item["channels"]), np.asarray(item["reconstructions"])
            for event in range(10):
                rows.append({"train_id": train_id, "event_index": event + 1, "method": method, "source_si_nrmse": source_error,
                             "channel_time_si_nrmse": rel_error(channels[event], target_channels[event]),
                             "click_reconstruction_nrmse": float(np.linalg.norm(reconstruction[event] - observed[event]) / (np.linalg.norm(observed[event]) + 1e-18))})
        print(f"[总体进度：v7 真值评估] {index}/{len(split)} | {train_id}", flush=True)
    write_csv(OUT / "event_metrics.csv", rows)
    summary = []
    for metric in ("source_si_nrmse", "channel_time_si_nrmse", "click_reconstruction_nrmse"):
        grouped = {}
        for method in ("v4_independent", "v7_time_refiner"):
            values = np.asarray([np.median([float(row[metric]) for row in rows if row["train_id"] == train_id and row["method"] == method]) for train_id in sorted({str(row["train_id"]) for row in rows})])
            lo, hi = bootstrap(values, abs(hash((metric, method))) % (2**32)); grouped[method] = values
            summary.append({"metric": metric, "method": method, "n_clicktrains": len(values), "median": float(np.median(values)), "ci_low": lo, "ci_high": hi})
        delta = grouped["v7_time_refiner"] - grouped["v4_independent"]; lo, hi = bootstrap(delta, abs(hash((metric, "delta"))) % (2**32))
        summary.append({"metric": metric, "method": "v7_minus_v4", "n_clicktrains": len(delta), "median": float(np.median(delta)), "ci_low": lo, "ci_high": hi})
    write_csv(OUT / "summary_metrics.csv", summary); print(f"v7 评估完成：{OUT / 'summary_metrics.csv'}", flush=True)


if __name__ == "__main__": main()
