#!/usr/bin/env python3
"""Truth-only evaluator and figure generator for full Experiment 20 v4 AWGN runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bcsc_v4 import path_candidates
from evaluate_bcsc_v4 import (DATA_ROOT, FS, METHOD_COLOURS, METHOD_LABELS, align_source, evaluate_train, normalise_truth_source, relative_spectrum_db, truth_paths, write_csv)
from run_noise_robustness_v4 import METHODS, NOISE_OUTPUT_ROOT, SNR_LEVELS_DB, snr_tag


ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = ROOT / "evaluation_noise"
CONDITION_ROOT = EVALUATION_ROOT / "conditions"
CASE_SNR_DB = -15


def bootstrap_interval(values: np.ndarray, seed_text: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    medians = np.asarray([np.median(rng.choice(values, len(values), replace=True)) for _ in range(1000)])
    return float(np.quantile(medians, .025)), float(np.quantile(medians, .975))


def evaluate(snrs: tuple[int, ...], limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    expected_ids = sorted(path.parent.name for path in DATA_ROOT.glob("*/clicktrain_raw.npy"))
    if limit is not None:
        expected_ids = expected_ids[:limit]
    for snr_db in snrs:
        for method in METHODS:
            directory = NOISE_OUTPUT_ROOT / snr_tag(snr_db) / method
            artifacts = sorted(directory.glob("*.npz"))
            if limit is not None:
                expected = set(expected_ids)
                artifacts = [artifact for artifact in artifacts if artifact.stem in expected]
            found = {artifact.stem for artifact in artifacts}
            coverage.append({"snr_db": snr_db, "method": method, "expected_clicktrains": len(expected_ids), "available_clicktrains": len(found), "missing_clicktrains": len(set(expected_ids) - found)})
            for index, artifact in enumerate(artifacts, start=1):
                for row in evaluate_train(method, artifact):
                    row["snr_db"] = snr_db
                    rows.append(row)
                print(f"[总体进度：噪声真值验证] SNR={snr_db:>3} dB | {method} {index:3d}/{len(artifacts)}", flush=True)
    return rows, coverage


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metrics = [column for column in frame if column not in {"train_id", "method", "event_index", "snr_db"}]
    result: list[dict[str, Any]] = []
    for (method, snr_db), group in frame.groupby(["method", "snr_db"]):
        per_train = group.groupby("train_id", as_index=False)[metrics].median(numeric_only=True)
        for metric in metrics:
            values = per_train[metric].to_numpy(dtype=float)
            low, high = bootstrap_interval(values, f"{method}|{snr_db}|{metric}")
            result.append({"method": method, "snr_db": int(snr_db), "metric": metric, "n_clicktrains": int(np.isfinite(values).sum()), "median": float(np.nanmedian(values)), "q25": float(np.nanquantile(values, .25)), "q75": float(np.nanquantile(values, .75)), "bootstrap_median_ci_low": low, "bootstrap_median_ci_high": high})
    return result


def select_representative_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select distinct, deterministic −15 dB cases using truth-only evaluation metrics."""
    frame = pd.DataFrame(rows)
    frame = frame[frame["snr_db"] == CASE_SNR_DB]
    if frame.empty:
        raise RuntimeError(f"cannot select representative cases: {CASE_SNR_DB} dB metrics are missing")
    per_method = frame.groupby(["train_id", "method"], as_index=False)[["source_si_nrmse", "channel_amplitude_rmse_db"]].median()
    scores = per_method.groupby("train_id", as_index=False)[["source_si_nrmse", "channel_amplitude_rmse_db"]].mean()
    for column in ("source_si_nrmse", "channel_amplitude_rmse_db"):
        scale = float(scores[column].quantile(.75) - scores[column].quantile(.25))
        scores[f"{column}_robust_z"] = (scores[column] - scores[column].median()) / max(scale, 1e-12)
    scores["typical_distance"] = np.hypot(scores["source_si_nrmse_robust_z"], scores["channel_amplitude_rmse_db_robust_z"])
    ordered = {
        "typical": scores.sort_values(["typical_distance", "train_id"]),
        "click_hard": scores.sort_values(["source_si_nrmse", "train_id"], ascending=[False, True]),
        "channel_hard": scores.sort_values(["channel_amplitude_rmse_db", "train_id"], ascending=[False, True]),
    }
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for case_type, candidates in ordered.items():
        row = next((item for item in candidates.to_dict("records") if str(item["train_id"]) not in used), None)
        if row is None:
            raise RuntimeError(f"could not select a distinct {case_type} representative case")
        used.add(str(row["train_id"]))
        chosen.append({"case_type": case_type, "snr_db": CASE_SNR_DB, **row})
    return chosen


def plot_summary_metric(axis: plt.Axes, summary: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    for method in METHODS:
        group = summary[(summary.method == method) & (summary.metric == metric)].sort_values("snr_db")
        if group.empty:
            continue
        x = group.snr_db.to_numpy(dtype=float)
        y = group["median"].to_numpy(dtype=float)
        low = group.bootstrap_median_ci_low.to_numpy(dtype=float)
        high = group.bootstrap_median_ci_high.to_numpy(dtype=float)
        axis.plot(x, y, marker="o", ms=4, lw=2, label=METHOD_LABELS[method], color=METHOD_COLOURS[method])
        axis.fill_between(x, low, high, color=METHOD_COLOURS[method], alpha=.16, linewidth=0)
    axis.set(title=title, xlabel="input SNR (dB)", ylabel=ylabel, xticks=SNR_LEVELS_DB)
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)


def _case_artifact(method: str, train_id: str) -> Path:
    return NOISE_OUTPUT_ROOT / snr_tag(CASE_SNR_DB) / method / f"{train_id}.npz"


def plot_case(case: dict[str, Any]) -> list[Path]:
    """Create separate method-row views, avoiding three-method trace overlap."""
    train_id = str(case["train_id"])
    directory = DATA_ROOT / train_id
    truth_source = normalise_truth_source(directory / "original_click.wav")
    with np.load(directory / "event_channels.npz", allow_pickle=True) as event_channel:
        event_arrivals = event_channel["event_synthesis_arrivals"][0]
        physical_paths = truth_paths(event_arrivals, float(event_channel["first_arrival_s"]))
    estimates: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        artifact = _case_artifact(method, train_id)
        if not artifact.exists():
            raise FileNotFoundError(f"missing representative artifact: {artifact}")
        with np.load(artifact, allow_pickle=False) as payload:
            source = np.asarray(payload["source"])
            estimates[method] = {"source": source[0] if source.ndim == 2 else source, "channel": np.asarray(payload["channels"])[0]}

    output_dir = EVALUATION_ROOT / "figures" / "representative_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{case['case_type']}_{train_id}"
    time_ms = np.arange(len(truth_source)) / FS * 1e3
    frequency, truth_spectrum = relative_spectrum_db(truth_source)
    source_band = (frequency >= 20_000) & (frequency <= 190_000)
    truth_delay = np.asarray([row["relative_delay_ms"] for row in physical_paths], dtype=float)
    truth_amplitude = 20 * np.log10(np.maximum(np.asarray([row["relative_amplitude"] for row in physical_paths], dtype=float), 1e-12))

    figures: list[Path] = []
    for view in ("waveform", "spectrum", "channel_structure"):
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=view != "waveform")
        for axis, method in zip(axes, METHODS):
            colour = METHOD_COLOURS[method]
            aligned, _ = align_source(estimates[method]["source"], truth_source)
            if view == "waveform":
                axis.plot(time_ms, truth_source, color="#222222", lw=2.0, label="original click")
                axis.plot(time_ms, aligned, color=colour, lw=1.6, label=f"predicted: {METHOD_LABELS[method]}")
                axis.set(ylabel="normalised amplitude")
            elif view == "spectrum":
                _, predicted_spectrum = relative_spectrum_db(aligned)
                axis.plot(frequency[source_band] / 1e3, truth_spectrum[source_band], color="#222222", lw=2.0, label="original click")
                axis.plot(frequency[source_band] / 1e3, predicted_spectrum[source_band], color=colour, lw=1.6, label=f"predicted: {METHOD_LABELS[method]}")
                axis.set(ylabel="relative magnitude (dB)", xlim=(20, 190))
            else:
                # The metric keeps every candidate; this qualitative view retains only the
                # six strongest to make path correspondence visually inspectable.
                predicted_paths = sorted(path_candidates(estimates[method]["channel"]), key=lambda row: abs(row["coefficient"]), reverse=True)[:6]
                predicted_paths.sort(key=lambda row: row["delay_ms"])
                predicted_delay = np.asarray([row["delay_ms"] for row in predicted_paths], dtype=float)
                predicted_amplitude = 20 * np.log10(np.maximum(np.asarray([row["relative_amplitude"] for row in predicted_paths], dtype=float), 1e-12))
                axis.vlines(truth_delay, -45, truth_amplitude, color="#222222", lw=1.4)
                axis.scatter(truth_delay, truth_amplitude, color="#222222", marker="o", s=32, label="original physical paths", zorder=3)
                if len(predicted_delay):
                    axis.vlines(predicted_delay, -45, predicted_amplitude, color=colour, lw=1.4)
                    axis.scatter(predicted_delay, predicted_amplitude, color=colour, marker="x", s=40, label=f"predicted: {METHOD_LABELS[method]}", zorder=3)
                xmax = max(0.1, float(max(np.max(truth_delay), np.max(predicted_delay) if len(predicted_delay) else 0.0)) + .08)
                axis.set(ylabel="relative path amplitude (dB)", xlim=(-.08, xmax), ylim=(-45, 3))
            axis.set_title(METHOD_LABELS[method])
            axis.grid(alpha=.25)
            axis.legend(loc="lower left" if view == "channel_structure" else "upper right", fontsize=8)
        axes[-1].set_xlabel("ms" if view == "waveform" else "kHz" if view == "spectrum" else "relative delay (ms)")
        figure.suptitle(f"{case['case_type'].replace('_', ' ')} case at {CASE_SNR_DB} dB: {train_id}")
        figure.tight_layout()
        path = output_dir / f"{slug}_{view}.png"
        figure.savefig(path, dpi=300)
        plt.close(figure)
        figures.append(path)
    return figures


def make_figure(summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> Path:
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("cannot plot a noise figure without evaluated rows")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    plot_summary_metric(axes[0], summary, "source_si_nrmse", "Click prediction error across AWGN conditions", "source SI-NRMSE")
    plot_summary_metric(axes[1], summary, "source_log_spectral_mae_db", "Click spectral prediction error across AWGN conditions", "log-spectral MAE (dB)")
    plot_summary_metric(axes[2], summary, "channel_amplitude_rmse_db", "Channel prediction error across AWGN conditions", "channel magnitude RMSE (dB)")
    covered = int(summary["n_clicktrains"].max())
    condition_count = int(summary["snr_db"].nunique())
    fig.suptitle(f"AWGN robustness: {covered} clicktrains, {condition_count} SNR conditions")
    fig.tight_layout()
    figures = EVALUATION_ROOT / "figures"; figures.mkdir(parents=True, exist_ok=True)
    output = figures / "noise_robustness.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    cases = select_representative_cases(rows)
    (figures / "representative_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for case in cases:
        plot_case(case)
    return output


def evaluate_conditions(snrs: tuple[int, ...], limit: int | None) -> None:
    """Write condition-local shards, so distinct SNR workers never share a file."""
    for snr_db in snrs:
        rows, coverage = evaluate((snr_db,), limit)
        directory = CONDITION_ROOT / snr_tag(snr_db)
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(directory / "noise_event_metrics.csv", rows)
        write_csv(directory / "noise_coverage.csv", coverage)


def consolidate(snrs: tuple[int, ...]) -> Path:
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for snr_db in snrs:
        directory = CONDITION_ROOT / snr_tag(snr_db)
        metrics_path = directory / "noise_event_metrics.csv"
        coverage_path = directory / "noise_coverage.csv"
        if not metrics_path.exists() or not coverage_path.exists():
            raise FileNotFoundError(f"missing evaluated condition shard for {snr_db} dB")
        rows.extend(pd.read_csv(metrics_path).to_dict("records"))
        coverage.extend(pd.read_csv(coverage_path).to_dict("records"))
    summary = summarise(rows)
    write_csv(EVALUATION_ROOT / "noise_event_metrics.csv", rows)
    write_csv(EVALUATION_ROOT / "noise_coverage.csv", coverage)
    write_csv(EVALUATION_ROOT / "noise_summary_metrics.csv", summary)
    (EVALUATION_ROOT / "noise_summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return make_figure(summary, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--snr", type=int, nargs="*", choices=SNR_LEVELS_DB, default=None)
    parser.add_argument("--mode", choices=("all", "condition", "consolidate"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snrs = tuple(args.snr) if args.snr is not None else SNR_LEVELS_DB
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "condition"):
        evaluate_conditions(snrs, args.limit)
    if args.mode in ("all", "consolidate"):
        figure = consolidate(snrs)
        print(f"噪声验证完成：{figure}", flush=True)


if __name__ == "__main__":
    main()
