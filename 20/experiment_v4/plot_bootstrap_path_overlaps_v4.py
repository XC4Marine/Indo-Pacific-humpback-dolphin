#!/usr/bin/env python3
"""Render truth-versus-high-confidence path overlap cases for the bootstrap pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_high_confidence_v4 import METHODS, OUTPUT_ROOT
from evaluate_bcsc_v4 import DATA_ROOT, METHOD_COLOURS, METHOD_LABELS, truth_paths


ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = ROOT / "evaluation_bootstrap_path_pilot"
OUTPUT_DIR = EVALUATION_ROOT / "figures" / "path_overlaps"
FLOOR_DB = -46.0


def load_stable_paths(method: str, train_id: str, event_index: int) -> list[dict[str, Any]]:
    record = json.loads((OUTPUT_ROOT / method / f"{train_id}.json").read_text(encoding="utf-8"))
    return next(event["stable_paths"] for event in record["events"] if int(event["event_index"]) == event_index)


def load_truth(train_id: str, event_index: int) -> list[dict[str, Any]]:
    with np.load(DATA_ROOT / train_id / "event_channels.npz", allow_pickle=True) as payload:
        return truth_paths(payload["event_synthesis_arrivals"][event_index - 1], float(payload["first_arrival_s"]))


def select_cases(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    frame = frame.copy()
    frame["f1_loss"] = frame["raw_path_f1"] - frame["high_conf_path_f1"]
    frame["event_key"] = frame["train_id"].astype(str) + "|" + frame["event_index"].astype(str)
    median_f1 = float(frame["high_conf_path_f1"].median())
    ordered = {
        "best recovery": frame.sort_values(["high_conf_path_f1", "high_conf_estimated_path_count", "train_id", "event_index"], ascending=[False, False, True, True]),
        "typical result": frame.assign(distance=np.abs(frame["high_conf_path_f1"] - median_f1)).sort_values(["distance", "train_id", "event_index"]),
        "largest F1 loss": frame.sort_values(["f1_loss", "raw_path_f1", "train_id", "event_index"], ascending=[False, False, True, True]),
    }
    chosen: list[tuple[str, pd.Series]] = []
    used: set[str] = set()
    for label, candidates in ordered.items():
        row = next(candidate for _, candidate in candidates.iterrows() if candidate["event_key"] not in used)
        used.add(str(row["event_key"])); chosen.append((label, row))
    return chosen


def plot_method(method: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    colour = METHOD_COLOURS[method]
    cases = select_cases(frame[frame["method"] == method])
    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10.0), sharex=False)
    records: list[dict[str, Any]] = []
    for axis, (label, row) in zip(axes, cases):
        train_id, event_index = str(row["train_id"]), int(row["event_index"])
        truth, stable = load_truth(train_id, event_index), load_stable_paths(method, train_id, event_index)
        true_delay = np.asarray([float(item["relative_delay_ms"]) for item in truth])
        true_amplitude = 20 * np.log10(np.maximum([float(item["relative_amplitude"]) for item in truth], 1e-12))
        stable_delay = np.asarray([float(item["delay_median_ms"]) for item in stable])
        stable_amplitude = 20 * np.log10(np.maximum([float(item["relative_amplitude_median"]) for item in stable], 1e-12))
        axis.vlines(true_delay, FLOOR_DB, true_amplitude, color="#222222", lw=1.6, zorder=1)
        axis.scatter(true_delay, true_amplitude, color="#222222", marker="o", s=48, label="true physical paths", zorder=3)
        if len(stable_delay):
            axis.vlines(stable_delay, FLOOR_DB, stable_amplitude, color=colour, lw=1.8, zorder=2)
            axis.scatter(stable_delay, stable_amplitude, color=colour, marker="x", s=66, linewidths=2, label="bootstrap high-confidence paths", zorder=4)
        right = max(.12, float(max(np.max(true_delay), np.max(stable_delay) if len(stable_delay) else 0.0)) + .07)
        axis.set(xlim=(-.05, right), ylim=(FLOOR_DB, 3), ylabel="relative path amplitude (dB)")
        axis.grid(alpha=.25); axis.legend(loc="lower right", fontsize=9)
        axis.set_title(f"{label}: {train_id}, event {event_index} | raw F1={row['raw_path_f1']:.3f}, high-confidence F1={row['high_conf_path_f1']:.3f}", fontsize=10)
        records.append({"method": method, "case_type": label, "train_id": train_id, "event_index": event_index,
                        "raw_path_f1": float(row["raw_path_f1"]), "high_conf_path_f1": float(row["high_conf_path_f1"]),
                        "true_path_count": len(truth), "high_conf_path_count": len(stable)})
    axes[-1].set_xlabel("relative delay (ms)")
    figure.suptitle(f"True versus bootstrap high-confidence paths: {METHOD_LABELS[method]}", fontsize=15)
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{method}_truth_high_confidence_overlap.png"
    figure.savefig(output, dpi=300); plt.close(figure)
    return records


def main() -> None:
    metrics = pd.read_csv(EVALUATION_ROOT / "event_metrics.csv")
    all_records: list[dict[str, Any]] = []
    for method in METHODS:
        all_records.extend(plot_method(method, metrics))
    (OUTPUT_DIR / "selected_cases.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"路径重合图完成：{OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
