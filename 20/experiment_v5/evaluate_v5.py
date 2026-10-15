#!/usr/bin/env python3
"""Truth-only v5 evaluator. Run only after infer_v5.py has completed."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from infer_v5 import CONDITIONS, add_awgn, condition_name
from protocol_v5 import DATA_ROOT, OUTPUT_ROOT, discover_inputs, group_split, write_csv

V4_ROOT = Path(__file__).resolve().parent.parent / "experiment_v4"
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import bandpass  # noqa: E402
from evaluate_bcsc_v4 import (normalise_truth_source, path_metrics, source_metrics, transfer_from_arrivals, transfer_metrics)  # noqa: E402


def evaluate_artifact(method: str, artifact: Path, snr_db: int | None) -> list[dict[str, Any]]:
    train_id = artifact.stem; directory = DATA_ROOT / train_id; estimate = np.load(artifact)
    source_truth = normalise_truth_source(directory / "original_click.wav")
    with np.load(directory / "event_channels.npz", allow_pickle=True) as truth:
        arrivals = truth["event_synthesis_arrivals"]; first_arrival = float(truth["first_arrival_s"])
    input_waveform = bandpass(add_awgn(np.load(directory / "clicktrain_raw.npy", allow_pickle=False), train_id, snr_db))
    source = np.asarray(estimate["source"]); channels = np.asarray(estimate["channels"]); recon = np.asarray(estimate["reconstructions"])
    starts = np.asarray(estimate["detected_window_starts"], dtype=int); rows: list[dict[str, Any]] = []
    for event in range(10):
        target = transfer_from_arrivals(arrivals[event], first_arrival)
        fitted = recon[event]; observed = input_waveform[starts[event]:starts[event] + len(fitted)]
        if len(observed) < len(fitted): observed = np.pad(observed, (0, len(fitted) - len(observed)))
        error = fitted - observed
        rows.append({"train_id": train_id, "method": method, "event_index": event + 1,
                     **source_metrics(source, source_truth), **transfer_metrics(channels[event], target),
                     **path_metrics(channels[event], arrivals[event], first_arrival),
                     "event_reconstruction_nrmse": float(np.linalg.norm(error) / (np.linalg.norm(observed) + 1e-18)),
                     "event_explained_variance": float(1 - np.sum(error * error) / (np.sum(observed * observed) + 1e-18))})
    return rows


def bootstrap(values: np.ndarray, token: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return float("nan"), float("nan")
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % (2**32)
    rng = np.random.default_rng(seed); samples = np.asarray([np.median(rng.choice(values, len(values), replace=True)) for _ in range(1000)])
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows); excluded = {"train_id", "method", "event_index", "condition", "snr_db"}
    metrics = [name for name in frame.columns if name not in excluded]
    result: list[dict[str, Any]] = []
    for (condition, method), group in frame.groupby(["condition", "method"]):
        per_train = group.groupby("train_id", as_index=False)[metrics].median(numeric_only=True)
        for metric in metrics:
            values = per_train[metric].to_numpy(dtype=float); low, high = bootstrap(values, f"{condition}|{method}|{metric}")
            result.append({"condition": condition, "method": method, "metric": metric, "n_clicktrains": int(len(values)),
                           "median": float(np.nanmedian(values)), "q25": float(np.nanquantile(values, .25)),
                           "q75": float(np.nanquantile(values, .75)), "bootstrap_median_ci_low": low,
                           "bootstrap_median_ci_high": high})
    return result


def paired_learned_vs_v4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows); learned = frame[frame.method.str.startswith("unroll_seed")].copy()
    if learned.empty: return []
    metrics = ("source_correlation", "channel_amplitude_rmse_db", "path_f1")
    values: list[dict[str, Any]] = []
    for condition in sorted(frame.condition.unique()):
        base = frame[(frame.condition == condition) & (frame.method == "v4_independent")].groupby("train_id")[list(metrics)].median()
        aggregate = learned[learned.condition == condition].groupby(["train_id", "method"])[list(metrics)].median().groupby("train_id").mean()
        common = base.index.intersection(aggregate.index)
        for metric in metrics:
            delta = aggregate.loc[common, metric].to_numpy() - base.loc[common, metric].to_numpy()
            low, high = bootstrap(delta, f"paired|{condition}|{metric}")
            values.append({"condition": condition, "comparison": "mean_learned_unroll_minus_v4_independent", "metric": metric,
                           "n_clicktrains": len(common), "median_delta": float(np.median(delta)),
                           "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
    return values


def write_report(summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    report = ["# 第20实验 v5：深度展开式 BCSC 原型报告", "", "本报告仅汇总留出来源组上的实际计算结果；不预设任何方法优劣。", "",
              "## 覆盖", "", "- 测试划分：11个严格来源组、16个 clicktrain。", "- 条件：clean、−5 dB、5 dB。", "- 对照：v4 independent、fixed unroll、三随机种子 learned unroll。", "",
              "## 汇总产物", "", "- `event_metrics.csv`：事件级指标。", "- `summary_metrics.csv`：按 clicktrain 聚类的中位数、四分位数和95% bootstrap区间。", "- `paired_deltas.csv`：三种子平均 learned unroll 相对 v4 independent 的配对差。", "",
              "## 结果状态", "", f"已写入 {len(summary)} 条汇总指标和 {len(paired)} 条配对比较。结论应仅依据这些数值及置信区间解释。"]
    path = OUTPUT_ROOT / "evaluation" / "prototype_report_zh.md"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--limit", type=int, default=None); args = parser.parse_args()
    expected = [row["train_id"] for row in group_split(discover_inputs()) if row["split"] == "test"]
    if args.limit is not None: expected = expected[:args.limit]
    rows: list[dict[str, Any]] = []
    for condition_index, snr_db in enumerate(CONDITIONS, 1):
        name = condition_name(snr_db); root = OUTPUT_ROOT / "predictions" / name
        for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            artifacts = [method_dir / f"{train_id}.npz" for train_id in expected]
            missing = [path for path in artifacts if not path.is_file()]
            if missing: raise FileNotFoundError(f"missing {len(missing)} artifacts for {name}/{method_dir.name}")
            for index, artifact in enumerate(artifacts, 1):
                for row in evaluate_artifact(method_dir.name, artifact, snr_db):
                    row["condition"] = name; row["snr_db"] = "clean" if snr_db is None else snr_db; rows.append(row)
                print(f"[总体进度：真值评估] condition {condition_index}/{len(CONDITIONS)} | {method_dir.name} {index}/{len(artifacts)}", flush=True)
    summary = summarise(rows); paired = paired_learned_vs_v4(rows)
    evaluation = OUTPUT_ROOT / "evaluation"; write_csv(evaluation / "event_metrics.csv", rows)
    write_csv(evaluation / "summary_metrics.csv", summary); write_csv(evaluation / "paired_deltas.csv", paired)
    write_report(summary, paired); print(f"真值评估完成：{evaluation}", flush=True)


if __name__ == "__main__":
    main()
