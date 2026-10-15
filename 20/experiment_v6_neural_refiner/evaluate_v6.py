#!/usr/bin/env python3
"""Truth-only evaluator for blind v6 prediction artifacts."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; V5 = HERE.parent / "experiment_v5"; V4 = HERE.parent / "experiment_v4"
sys.path[:0] = [str(V5), str(V4)]
from protocol_v5 import DATA_ROOT, discover_inputs, group_split  # noqa: E402
from evaluate_bcsc_v4 import transfer_from_arrivals  # noqa: E402
from experiment_v6 import CONDITIONS, OUT, report, write_csv  # noqa: E402


def truth_shape(directory: Path) -> np.ndarray:
    with np.load(directory / "event_channels.npz", allow_pickle=True) as item:
        arrivals = item["event_synthesis_arrivals"]; first = float(item["first_arrival_s"])
    result = []
    for event in arrivals:
        response = transfer_from_arrivals(event, first); values = 20 * np.log10(np.abs(response) + 1e-15); result.append(values - np.median(values))
    return np.asarray(result, dtype=np.float32)


def main() -> None:
    test_ids = [row["train_id"] for row in group_split(discover_inputs()) if row["split"] == "test"]; rows = []
    for condition in CONDITIONS:
        for index, train_id in enumerate(test_ids, 1):
            item = np.load(OUT / "blind_predictions" / condition / f"{train_id}.npz"); target = truth_shape(DATA_ROOT / train_id)
            for method, values in (("neural_ensemble", item["corrected_shape"]), ("v4_independent", item["baseline_shape"])):
                errors = np.sqrt(np.mean((values - target) ** 2, axis=1))
                for event, error in enumerate(errors, 1): rows.append({"condition": condition, "train_id": train_id, "event_index": event, "method": method, "channel_amplitude_rmse_db": float(error)})
            print(f"[总体进度：v6真值评估] {condition} {index}/{len(test_ids)}", flush=True)
    report(rows); print(f"真值评估完成：{OUT / 'experiment_report_zh.md'}", flush=True)


if __name__ == "__main__": main()
