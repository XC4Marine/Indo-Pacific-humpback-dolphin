#!/usr/bin/env python3
"""Truth-only validation of bootstrap-stable BCSC supports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcsc_v4 import CHANNEL_LENGTH
from bootstrap_high_confidence_v4 import INPUT_ROOT, METHODS, OUTPUT_ROOT, PROTOCOL
from evaluate_bcsc_v4 import DATA_ROOT, METHOD_COLOURS, METHOD_LABELS, path_metrics, transfer_from_arrivals, transfer_metrics, truth_paths

EVALUATION_ROOT = Path(__file__).resolve().parent / "evaluation_bootstrap_path_pilot"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"); os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def stable_path_metrics(stable: list[dict[str, Any]], arrivals: np.ndarray, first_arrival: float) -> dict[str, float]:
    truth = truth_paths(arrivals, first_arrival)
    estimated = [{"delay_ms": float(row["delay_median_ms"]), "relative_amplitude": float(row["relative_amplitude_median"])} for row in stable]
    pairs = [(abs(candidate["delay_ms"] - float(target["relative_delay_ms"])), truth_i, estimate_i)
             for truth_i, target in enumerate(truth) for estimate_i, candidate in enumerate(estimated)
             if abs(candidate["delay_ms"] - float(target["relative_delay_ms"])) <= .03]
    used_truth: set[int] = set(); used_estimate: set[int] = set(); delays = []; amplitudes = []
    for error, truth_i, estimate_i in sorted(pairs):
        if truth_i in used_truth or estimate_i in used_estimate: continue
        used_truth.add(truth_i); used_estimate.add(estimate_i); target, candidate = truth[truth_i], estimated[estimate_i]
        delays.append(error); amplitudes.append(abs(20 * math.log10(candidate["relative_amplitude"] + 1e-12) - 20 * math.log10(float(target["relative_amplitude"]) + 1e-12)))
    hits = len(used_truth); precision = hits / len(estimated) if estimated else 0.; recall = hits / len(truth) if truth else 0.
    return {"path_precision": precision, "path_recall": recall, "path_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "path_delay_mae_ms": float(np.mean(delays)) if delays else float("nan"), "path_relative_amplitude_mae_db": float(np.mean(amplitudes)) if amplitudes else float("nan"),
            "estimated_path_count": len(estimated), "true_path_count": len(truth)}


def stable_channel(stable: list[dict[str, Any]], reference: np.ndarray) -> np.ndarray:
    channel = np.zeros(CHANNEL_LENGTH, dtype=float)
    for path in stable:
        sample = int(path["sample"])
        if 0 <= sample < CHANNEL_LENGTH: channel[sample] = float(path["coefficient_median"])
    if not np.any(channel) and np.any(reference):
        peak = int(np.argmax(np.abs(reference))); channel[peak] = reference[peak]
    return channel


def evaluate_one(method: str, result_path: Path, path_field: str) -> list[dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("protocol") != PROTOCOL: raise ValueError(f"protocol mismatch: {result_path}")
    train_id = str(result["train_id"])
    with np.load(INPUT_ROOT / method / f"{train_id}.npz", allow_pickle=False) as item: channels = np.asarray(item["channels"], float)
    with np.load(DATA_ROOT / train_id / "event_channels.npz", allow_pickle=True) as truth:
        arrivals, first_arrival = truth["event_synthesis_arrivals"], float(truth["first_arrival_s"])
    rows = []
    for event in result["events"]:
        index = int(event["event_index"]) - 1; response = transfer_from_arrivals(arrivals[index], first_arrival)
        selected = event[path_field]
        raw, high = path_metrics(channels[index], arrivals[index], first_arrival), stable_path_metrics(selected, arrivals[index], first_arrival)
        raw_transfer, high_transfer = transfer_metrics(channels[index], response), transfer_metrics(stable_channel(selected, channels[index]), response)
        row = {"train_id": train_id, "method": method, "event_index": index + 1, "bootstrap_count": int(result["bootstrap_count"])}
        for prefix, data in (("raw", raw), ("high_conf", high), ("raw", raw_transfer), ("high_conf", high_transfer)):
            for key, value in data.items(): row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def interval(values: np.ndarray, token: str) -> tuple[float, float]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values): return float("nan"), float("nan")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little"))
    medians = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(2000)]
    return float(np.quantile(medians, .025)), float(np.quantile(medians, .975))


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows); metric_columns = [x for x in frame if x not in {"train_id", "method", "event_index", "bootstrap_count"}]
    summary: list[dict[str, Any]] = []
    for method, group in frame.groupby("method"):
        per_train = group.groupby("train_id", as_index=False)[metric_columns].median(numeric_only=True)
        for metric in metric_columns:
            values = per_train[metric].to_numpy(float); low, high = interval(values, f"level|{method}|{metric}")
            summary.append({"method": method, "comparison": "level", "metric": metric, "n_clicktrains": int(np.isfinite(values).sum()), "median": float(np.nanmedian(values)), "q25": float(np.nanquantile(values, .25)), "q75": float(np.nanquantile(values, .75)), "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
        for metric in ("path_precision", "path_recall", "path_f1", "path_delay_mae_ms", "path_relative_amplitude_mae_db", "channel_amplitude_rmse_db", "channel_phase_circular_mae_rad"):
            raw_key, high_key = f"raw_{metric}", f"high_conf_{metric}"
            delta = per_train[high_key].to_numpy(float) - per_train[raw_key].to_numpy(float); low, high = interval(delta, f"delta|{method}|{metric}")
            summary.append({"method": method, "comparison": "high_conf_minus_raw", "metric": metric, "n_clicktrains": int(np.isfinite(delta).sum()), "median": float(np.nanmedian(delta)), "q25": float(np.nanquantile(delta, .25)), "q75": float(np.nanquantile(delta, .75)), "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
    return summary


def make_figure(summary_rows: list[dict[str, Any]], evaluation_root: Path, selection_label: str) -> Path:
    summary = pd.DataFrame(summary_rows); figures = evaluation_root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    specs = (("path_f1", "Path F1 (higher is better)", "raw_path_f1", "high_conf_path_f1"), ("channel_amplitude_rmse_db", "Channel magnitude RMSE (dB, lower is better)", "raw_channel_amplitude_rmse_db", "high_conf_channel_amplitude_rmse_db"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5)); x = np.arange(len(METHODS)); width = .32
    for axis, (_, title, raw_key, high_key) in zip(axes, specs):
        for offset, metric, hatch, label in ((-width / 2, raw_key, "", "raw support"), (width / 2, high_key, "//", selection_label)):
            group = summary[(summary.comparison == "level") & (summary.metric == metric)].set_index("method").reindex(METHODS)
            values, low, high = (group[key].to_numpy(float) for key in ("median", "bootstrap_median_ci_low", "bootstrap_median_ci_high"))
            axis.bar(x + offset, values, width, color=[METHOD_COLOURS[m] for m in METHODS], hatch=hatch, edgecolor="#222222", linewidth=.45, label=label)
            axis.errorbar(x + offset, values, yerr=np.vstack([values - low, high - values]), fmt="none", color="#222222", capsize=3, lw=.9)
        axis.set_xticks(x, [METHOD_LABELS[m] for m in METHODS], rotation=12, ha="right"); axis.set_title(title); axis.grid(axis="y", alpha=.25); axis.legend(fontsize=8)
    figure.suptitle("Conditional residual-bootstrap pilot: 12 clicktrains × 10 events; 50 resamples"); figure.tight_layout()
    output = figures / "bootstrap_support_selection_pilot.png"; figure.savefig(output, dpi=300); plt.close(figure); return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=METHODS)
    parser.add_argument("--bootstrap-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--evaluation-root", type=Path, default=EVALUATION_ROOT)
    parser.add_argument("--path-field", choices=("stable_paths", "adaptive_bic_paths"), default="stable_paths")
    parser.add_argument("--selection-label", default="high-confidence support")
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for method in args.methods:
        paths = sorted((args.bootstrap_root / method).glob("*.json"))
        if not paths: raise FileNotFoundError(f"no bootstrap artifacts for {method}")
        for index, path in enumerate(paths, 1):
            rows.extend(evaluate_one(method, path, args.path_field)); print(f"[总体进度：路径真值验证] {method} {index}/{len(paths)}", flush=True)
    summary = summarise(rows); write_csv(args.evaluation_root / "event_metrics.csv", rows); write_csv(args.evaluation_root / "summary_metrics.csv", summary); atomic_json(args.evaluation_root / "summary_metrics.json", summary)
    print(f"验证完成：{make_figure(summary, args.evaluation_root, args.selection_label)}", flush=True)


if __name__ == "__main__": main()
