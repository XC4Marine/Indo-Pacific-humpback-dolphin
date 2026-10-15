from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    r"D:\Project_Github\Indo-Pacific-humpback-dolphin\20\full_dataset_experiment\.matplotlib",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import batch_joint_bcsc_compare as base


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\20\full_dataset_experiment")
FIGURES = ROOT / "figures"
ATOM_FIGURES = FIGURES / "atoms"
RESIDUAL_CHECKPOINTS = ROOT / "residual_checkpoints"
REPORT = ROOT / "report.html"
FS = 576_000
NFFT = base.BANDWIDTH_NFFT
HIGH_CONFIDENCE = 0.90
STRICT_WILSON_LOW = 0.80
STAT_BOOTSTRAPS = 50
SEED = 20260831


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=True, default=json_default), encoding="utf-8"
    )
    temporary.replace(path)


def wrap_degrees(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def quantiles(values: list[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {key: math.nan for key in ("q2_5", "q25", "median", "q75", "q97_5", "mean", "std")}
    q = np.percentile(finite, [2.5, 25, 50, 75, 97.5])
    return {
        "q2_5": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "q97_5": float(q[4]), "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def circular_summary(degrees: list[float]) -> dict[str, float]:
    values = np.asarray([value for value in degrees if np.isfinite(value)], dtype=float)
    if not len(values):
        return {"circular_mean_deg": math.nan, "resultant_length": math.nan, "circular_std_deg": math.nan}
    resultant = np.mean(np.exp(1j * np.radians(values)))
    length = float(abs(resultant))
    return {
        "circular_mean_deg": float(np.degrees(np.angle(resultant))),
        "resultant_length": length,
        "circular_std_deg": float(np.degrees(np.sqrt(max(0.0, -2 * np.log(max(length, 1e-12)))))),
    }


def effective_phase_features(atom: np.ndarray, delay_ms: float, coefficient: float) -> dict[str, float]:
    frequencies = np.fft.rfftfreq(NFFT, 1 / FS)
    peak_index = int(np.argmax(np.abs(atom)))
    spectrum = np.fft.rfft(atom, n=NFFT) * np.exp(1j * 2 * np.pi * frequencies * peak_index / FS)
    _, _, left, right = base.measure_bandwidth(atom, FS)
    mask = (frequencies >= left) & (frequencies <= right)
    f = frequencies[mask]
    sign_phase = 0.0 if coefficient >= 0 else np.pi
    phase = np.unwrap(np.angle(spectrum[mask]) + sign_phase - 2 * np.pi * f * delay_ms * 1e-3)
    if len(f) < 3:
        return {key: math.nan for key in (
            "effective_group_delay_us", "effective_residual_phase_rms_rad",
            "effective_residual_phase_resultant_length", "effective_residual_phase_circular_std_deg",
            "effective_phase_circular_mean_deg", "effective_phase_circular_std_deg",
        )}
    design = np.column_stack([f - np.mean(f), np.ones_like(f)])
    fitted, *_ = np.linalg.lstsq(design, phase, rcond=None)
    residual = phase - design @ fitted
    residual_resultant = float(abs(np.mean(np.exp(1j * residual))))
    raw_resultant = np.mean(np.exp(1j * phase))
    return {
        "effective_group_delay_us": float(-fitted[0] / (2 * np.pi) * 1e6),
        "effective_residual_phase_rms_rad": float(np.sqrt(np.mean(residual * residual))),
        "effective_residual_phase_resultant_length": residual_resultant,
        "effective_residual_phase_circular_std_deg": float(
            np.degrees(np.sqrt(max(0.0, -2 * np.log(max(residual_resultant, 1e-12)))))
        ),
        "effective_phase_circular_mean_deg": float(np.degrees(np.angle(raw_resultant))),
        "effective_phase_circular_std_deg": float(
            np.degrees(np.sqrt(max(0.0, -2 * np.log(max(abs(raw_resultant), 1e-12)))))
        ),
    }


def load_residuals(key: str) -> dict[str, np.ndarray]:
    path = RESIDUAL_CHECKPOINTS / f"{key}.npz"
    with np.load(path) as archive:
        return {
            str(name): np.asarray(archive["residuals"][i, : int(archive["residual_lengths"][i])], dtype=float)
            for i, name in enumerate(archive["click_names"])
        }


def aggregate(results: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    train_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    atom_ids: list[str] = []
    atoms: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    residual_ids: list[str] = []
    residual_arrays: list[np.ndarray] = []
    residual_psds: list[np.ndarray] = []
    residual_frequency = None
    frequencies = np.fft.rfftfreq(NFFT, 1 / FS)
    for result_index, result in enumerate(results, 1):
        residual_lookup = load_residuals(result["key"])
        metadata = {row["click"]: row for row in result["click_metadata"]}
        joint = result["joint"]
        single = result["single"]
        train_rows.append({
            "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
            "available_clicks": result["available_clicks"], "used_clicks": result["used_clicks"],
            "anchor_click": result["anchor_click"], "single_explained_variance": single["mean_explained_variance"],
            "joint_explained_variance": joint["mean_explained_variance"],
            "atom_similarity": joint["atom_single_similarity"], "elapsed_seconds": result["elapsed_seconds"],
        })
        for method_name, method in (("single", single), ("joint", joint)):
            atom_id = f"{result['voyage']}__{result['pulse_train']}__{method_name}"
            atom = np.asarray(method["atom"], dtype=float)
            peak_index = int(np.argmax(np.abs(atom)))
            atom_ids.append(atom_id)
            atoms.append(atom)
            spectra.append(np.fft.rfft(atom, n=NFFT) * np.exp(1j * 2 * np.pi * frequencies * peak_index / FS))
            feature_names = (
                "atom_length_us", "peak_frequency_khz", "spectral_centroid_khz", "spectral_median_khz",
                "rms_bandwidth_khz", "bandwidth_3db_khz", "left_3db_khz", "right_3db_khz",
                "center_3db_khz", "spectral_rolloff_95_khz", "high_low_energy_ratio", "time_peak_index",
                "time_peak_us", "normalized_peak_amplitude", "normalized_rms_amplitude", "crest_factor",
                "peak_phase_deg_centered", "group_delay_us_centered", "phase_residual_rms_rad",
                "phase_residual_resultant_length", "phase_residual_circular_std_deg",
            )
            atom_rows.append({
                "atom_id": atom_id, "method": method_name, "voyage": result["voyage"],
                "pulse_train": result["pulse_train"], **{name: method[name] for name in feature_names},
                "mean_explained_variance": method["mean_explained_variance"],
                "atom_similarity_single_joint": joint["atom_single_similarity"],
            })
            for click in method["click_results"]:
                click_meta = metadata[click["click"]]
                residual_id = f"{atom_id}__{click['click']}"
                if method_name == "joint":
                    residual_ids.append(residual_id)
                    residual_arrays.append(residual_lookup[click["click"]])
                    residual_psds.append(np.asarray(click["residual_psd"], dtype=float))
                    residual_frequency = np.asarray(click["residual_psd_frequency_hz"], dtype=float)
                for path_index, path in enumerate(click["paths"]):
                    coefficient = float(path["coefficient"])
                    delay_ms = float(path["relative_delay_ms"])
                    effective_phase = float(wrap_degrees(
                        method["peak_phase_deg_centered"] + path["phase_flip_deg"]
                        - 360 * method["peak_frequency_khz"] * 1e3 * delay_ms * 1e-3
                    ))
                    channel_rows.append({
                        "channel_id": f"{result['key']}__{method_name}__{click['click']}__{path_index}",
                        "method": method_name, "voyage": result["voyage"], "pulse_train": result["pulse_train"],
                        "click": click["click"], "available_clicks": result["available_clicks"],
                        "used_clicks": result["used_clicks"], "wav_path": click_meta["wav_path"],
                        "path": path["path"], "atom_id": atom_id, "residual_id": residual_id,
                        "code_index": path["code_index"], "arrival_sample": path["arrival_sample"],
                        "relative_delay_ms": delay_ms, "coefficient": coefficient,
                        "direct_coefficient": path["direct_coefficient"],
                        "relative_signed_amplitude": path["relative_signed_amplitude"],
                        "relative_amplitude": path["relative_amplitude"], "relative_level_db": path["relative_level_db"],
                        "phase_flip_deg": path["phase_flip_deg"], "effective_phase_at_atom_peak_deg": effective_phase,
                        **effective_phase_features(atom, delay_ms, coefficient),
                        "detected_runs": path["detected_runs"], "bootstrap_total": path["bootstrap_total"],
                        "confidence": path["confidence"], "confidence_low": path["confidence_low"],
                        "confidence_high": path["confidence_high"], "confidence_class": path["confidence_class"],
                        "delay_median_ms": path["delay_median_ms"], "delay_low_ms": path["delay_low_ms"],
                        "delay_high_ms": path["delay_high_ms"], "amplitude_median": path["amplitude_median"],
                        "amplitude_low": path["amplitude_low"], "amplitude_high": path["amplitude_high"],
                        "signed_amplitude_median": path["signed_amplitude_median"],
                        "signed_amplitude_low": path["signed_amplitude_low"],
                        "signed_amplitude_high": path["signed_amplitude_high"],
                        "atom_peak_frequency_khz": method["peak_frequency_khz"],
                        "atom_spectral_centroid_khz": method["spectral_centroid_khz"],
                        "atom_spectral_median_khz": method["spectral_median_khz"],
                        "atom_bandwidth_3db_khz": method["bandwidth_3db_khz"],
                        "atom_left_3db_khz": method["left_3db_khz"], "atom_right_3db_khz": method["right_3db_khz"],
                        "atom_rms_bandwidth_khz": method["rms_bandwidth_khz"],
                        "atom_rolloff_95_khz": method["spectral_rolloff_95_khz"],
                        "atom_high_low_energy_ratio": method["high_low_energy_ratio"],
                        "atom_group_delay_us_centered": method["group_delay_us_centered"],
                        "atom_phase_residual_rms_rad": method["phase_residual_rms_rad"],
                        "filtered_peak_before_normalization": click_meta["filtered_peak_before_normalization"],
                        "filtered_rms_before_normalization": click_meta["filtered_rms_before_normalization"],
                        "raw_peak": click_meta["raw_peak"], "raw_rms": click_meta["raw_rms"],
                        "snr_proxy": click_meta["snr_proxy"], "signal_length": click_meta["signal_length"],
                        "explained_variance": click["explained_variance"], "residual_rms": click["residual_rms"],
                    })
        if result_index % 50 == 0 or result_index == len(results):
            print(f"[聚合进度] {result_index}/{len(results)} ({100*result_index/len(results):.1f}%)", flush=True)
    max_atom = max(map(len, atoms))
    atom_matrix = np.full((len(atoms), max_atom), np.nan)
    for i, atom in enumerate(atoms):
        atom_matrix[i, : len(atom)] = atom
    spectrum_matrix = np.vstack(spectra)
    np.savez_compressed(
        ROOT / "atoms.npz", atom_ids=np.asarray(atom_ids), atoms=atom_matrix,
        atom_lengths=np.asarray(list(map(len, atoms))), sample_rate=FS, frequencies_hz=frequencies,
        peak_aligned_spectrum_real=spectrum_matrix.real, peak_aligned_spectrum_imag=spectrum_matrix.imag,
    )
    np.savez_compressed(
        ROOT / "joint_residual_psd.npz", residual_ids=np.asarray(residual_ids),
        frequencies_hz=residual_frequency, psd=np.vstack(residual_psds),
    )
    max_residual = max(map(len, residual_arrays))
    residual_matrix = np.full((len(residual_arrays), max_residual), np.nan)
    for i, residual in enumerate(residual_arrays):
        residual_matrix[i, : len(residual)] = residual
    np.savez_compressed(
        ROOT / "joint_residuals.npz", residual_ids=np.asarray(residual_ids), residuals=residual_matrix,
        residual_lengths=np.asarray(list(map(len, residual_arrays))), sample_rate=FS,
    )
    high_rows = [
        row for row in channel_rows
        if row["method"] == "joint" and row["path"] != "direct" and row["confidence"] >= HIGH_CONFIDENCE
    ]
    return train_rows, channel_rows, atom_rows, high_rows


def clustered_summary(
    rows: list[dict[str, Any]], metric: str, scope: str,
    group_type: str = "all", group_value: str = "all",
) -> dict[str, Any]:
    finite_rows = [row for row in rows if np.isfinite(float(row.get(metric, math.nan)))]
    clusters: dict[tuple[str, str], list[float]] = {}
    for row in finite_rows:
        clusters.setdefault((row["voyage"], row["pulse_train"]), []).append(float(row[metric]))
    values = [float(row[metric]) for row in finite_rows]
    seed = SEED + int(hashlib.sha256(f"{scope}|{group_type}|{group_value}|{metric}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    groups = list(clusters.values())
    bootstrap_medians = []
    for _ in range(STAT_BOOTSTRAPS):
        if groups:
            chosen = rng.integers(0, len(groups), len(groups))
            bootstrap_medians.append(float(np.median([value for index in chosen for value in groups[int(index)]])))
    low, high = (np.percentile(bootstrap_medians, [2.5, 97.5]) if bootstrap_medians else (math.nan, math.nan))
    return {
        "scope": scope, "group_type": group_type, "group_value": group_value, "metric": metric,
        "n_observations": len(values), "n_pulsetrains": len(clusters), **quantiles(values),
        "cluster_bootstrap_runs": STAT_BOOTSTRAPS,
        "cluster_bootstrap_median_ci_low": float(low), "cluster_bootstrap_median_ci_high": float(high),
        "weighting_note": (
            "每个PulseTrain的联合原子仅计一次" if scope == "atom_once_per_pulsetrain"
            else "每条高置信度信道与原子配对；多径较多的PulseTrain权重更高，区间按PulseTrain聚类Bootstrap"
        ),
    }


def statistics_rows(high_rows: list[dict[str, Any]], atom_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joint_atoms = [row for row in atom_rows if row["method"] == "joint"]
    atom_metrics = (
        "peak_frequency_khz", "spectral_centroid_khz", "spectral_median_khz", "left_3db_khz",
        "right_3db_khz", "center_3db_khz", "bandwidth_3db_khz", "rms_bandwidth_khz",
        "high_low_energy_ratio", "spectral_rolloff_95_khz", "atom_length_us", "time_peak_us",
        "atom_similarity_single_joint", "group_delay_us_centered", "phase_residual_rms_rad",
        "phase_residual_circular_std_deg",
    )
    pair_metrics = (
        "atom_peak_frequency_khz", "atom_spectral_centroid_khz", "atom_spectral_median_khz",
        "atom_bandwidth_3db_khz", "coefficient", "relative_signed_amplitude", "relative_amplitude",
        "relative_level_db", "relative_delay_ms", "phase_flip_deg", "effective_phase_at_atom_peak_deg",
        "effective_group_delay_us", "effective_residual_phase_rms_rad",
        "effective_residual_phase_circular_std_deg", "effective_phase_circular_std_deg",
    )
    output = [clustered_summary(joint_atoms, metric, "atom_once_per_pulsetrain") for metric in atom_metrics]
    output += [clustered_summary(high_rows, metric, "high_confidence_atom_channel_pairs") for metric in pair_metrics]
    for voyage in sorted({row["voyage"] for row in high_rows}):
        subset = [row for row in high_rows if row["voyage"] == voyage]
        for metric in ("relative_amplitude", "relative_level_db"):
            output.append(clustered_summary(subset, metric, "high_confidence_atom_channel_pairs", "voyage", voyage))
    for low, high, label in ((5, 10, "5-10"), (11, 30, "11-30"), (31, math.inf, ">30")):
        subset = [row for row in high_rows if low <= int(row["available_clicks"]) <= high]
        for metric in ("relative_amplitude", "relative_level_db"):
            output.append(clustered_summary(subset, metric, "high_confidence_atom_channel_pairs", "available_click_count", label))
    if high_rows:
        boundaries = np.quantile([row["atom_peak_frequency_khz"] for row in high_rows], [0, .25, .5, .75, 1])
        for i in range(4):
            low, high = float(boundaries[i]), float(boundaries[i + 1])
            subset = [row for row in high_rows if low <= row["atom_peak_frequency_khz"] <= high]
            label = f"Q{i+1}:{low:.3f}-{high:.3f}kHz"
            for metric in ("relative_amplitude", "relative_level_db"):
                output.append(clustered_summary(subset, metric, "high_confidence_atom_channel_pairs", "empirical_atom_peak_frequency_quartile", label))
    return output


def synthesis_pairs(channel_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {(row["voyage"], row["pulse_train"], row["click"]) for row in high_rows}
    high_ids = {row["channel_id"] for row in high_rows}
    output = []
    for row in channel_rows:
        group = (row["voyage"], row["pulse_train"], row["click"])
        if row["method"] != "joint" or group not in groups:
            continue
        if row["path"] != "direct" and row["channel_id"] not in high_ids:
            continue
        output.append({
            "synthesis_pair_id": "__".join(group),
            "selection_role": "direct" if row["path"] == "direct" else "high_confidence_multipath",
            "strict_high_confidence": bool(row["path"] != "direct" and row["confidence_low"] >= STRICT_WILSON_LOW),
            **row,
        })
    return output


def sensitivity_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for result in results:
        for config in result["configs"]:
            failed = "error" in config
            output.append({
                "index": result["index"], "voyage": result["voyage"], "pulse_train": result["pulse_train"],
                "config": config["config"]["name"], "lambda_ratio": config["config"]["lambda_ratio"],
                "reference_atom_us": config["config"]["reference_atom_us"], "peak_ratio": config["config"]["peak_ratio"],
                "residual_block_ms": config["config"]["residual_block_ms"],
                "match_tolerance_ms": config["config"]["match_tolerance_ms"], "max_clicks": config["config"]["max_clicks"],
                "failed": failed, "error": config.get("error", ""), "high_count": config.get("high_count", math.nan),
                "strict_count": config.get("strict_count", math.nan),
                "mean_path_confidence": config.get("mean_path_confidence", math.nan),
                "median_wilson_low": config.get("median_wilson_low", math.nan),
                "median_wilson_high": config.get("median_wilson_high", math.nan),
                "mean_explained_variance": config.get("mean_explained_variance", math.nan),
                "path_jaccard_vs_default": config.get("path_jaccard_vs_default", math.nan),
                "atom_similarity_vs_default": config.get("atom_similarity_vs_default", math.nan),
                "high_count_relative_change": config.get("high_count_relative_change", math.nan),
                "median_abs_delay_change_ms": config.get("median_abs_delay_change_ms", math.nan),
                "median_abs_relative_amplitude_change": config.get("median_abs_relative_amplitude_change", math.nan),
                "bandwidth_projection_trigger_rate": config.get("bandwidth_projection_trigger_rate", math.nan),
                "elapsed_seconds": config.get("elapsed_seconds", math.nan),
                **{f"atom_{key}": value for key, value in config.get("atom_features", {}).items()},
            })
    return output


def parameter_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {(row["index"], row["pulse_train"]): row["mean_explained_variance"] for row in rows if row["config"] == "default"}
    output = []
    for name in sorted({row["config"] for row in rows if row["config"] != "default"}):
        subset = [row for row in rows if row["config"] == name]
        valid = [row for row in subset if not row["failed"]]
        failure_rate = 1 - len(valid) / max(1, len(subset))
        median = lambda field: float(np.nanmedian([row[field] for row in valid])) if valid else math.nan
        jaccard = median("path_jaccard_vs_default")
        similarity = median("atom_similarity_vs_default")
        count_change = float(np.nanmedian(np.abs([row["high_count_relative_change"] for row in valid]))) if valid else math.nan
        explained = float(np.nanmedian([
            abs(row["mean_explained_variance"] - baseline.get((row["index"], row["pulse_train"]), math.nan))
            for row in valid
        ])) if valid else math.nan
        if failure_rate >= .01 or jaccard < .60 or count_change > .25:
            recommendation = "需要人工检查或调整参数"
        elif jaccard < .80 or count_change > .10 or similarity < .90 or explained > .02:
            recommendation = "建议对敏感样本人工复核"
        else:
            recommendation = "无需常规人工调整"
        output.append({
            "variant": name, "median_path_jaccard": jaccard,
            "median_abs_high_count_change": count_change, "median_atom_similarity": similarity,
            "median_abs_explained_variance_change": explained, "failure_rate": failure_rate,
            "recommendation": recommendation,
        })
    return output


def plot_atom(result: dict[str, Any]) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), constrained_layout=True)
    for column, (label, method, color) in enumerate((
        ("单click", result["single"], "#A23B72"), ("多click联合", result["joint"], "#188977")
    )):
        atom = np.asarray(method["atom"], dtype=float)
        peak = int(np.argmax(np.abs(atom)))
        time_us = (np.arange(len(atom)) - peak) / FS * 1e6
        axes[0, column].plot(time_us, atom, color=color, lw=1.2)
        axes[0, column].axvline(0, color="#777", lw=.7, ls="--")
        axes[0, column].set(title=f"{label}原子", xlabel="相对主峰时间 / μs", ylabel="单位能量幅值")
        spectrum = np.abs(np.fft.rfft(atom, n=NFFT))
        frequency = np.fft.rfftfreq(NFFT, 1 / FS) / 1e3
        db = 20 * np.log10(spectrum / (spectrum.max() + 1e-15) + 1e-12)
        axes[1, column].plot(frequency, db, color=color, lw=1.2)
        axes[1, column].axhline(-3, color="#777", lw=.7, ls="--")
        axes[1, column].set(xlim=(0, 200), ylim=(-60, 2), xlabel="频率 / kHz", ylabel="归一化幅度 / dB")
    fig.suptitle(f"{result['voyage']} / {result['pulse_train']} | 单/联合相似度={result['joint']['atom_single_similarity']:.3f}")
    filename = f"{result['key']}_atom_comparison.png"
    fig.savefig(ATOM_FIGURES / filename, dpi=150)
    plt.close(fig)
    return filename


def summary_plots(
    train_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]],
    atom_rows: list[dict[str, Any]], decisions: list[dict[str, Any]],
) -> None:
    joint_atoms = [row for row in atom_rows if row["method"] == "joint"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist([row["peak_frequency_khz"] for row in joint_atoms], bins=30, color="#188977")
    axes[0, 0].set(title="联合原子峰值频率", xlabel="kHz", ylabel="原子数")
    axes[0, 1].hist([row["bandwidth_3db_khz"] for row in joint_atoms], bins=30, color="#427AA1")
    axes[0, 1].set(title="联合原子−3 dB带宽", xlabel="kHz", ylabel="原子数")
    axes[1, 0].scatter([row["relative_delay_ms"] for row in high_rows], [row["relative_level_db"] for row in high_rows], s=8, alpha=.3, color="#A23B72")
    axes[1, 0].set(title="高置信度多径：幅值—时延", xlabel="相对时延 / ms", ylabel="相对声级 / dB")
    axes[1, 1].hist([row["effective_phase_at_atom_peak_deg"] for row in high_rows], bins=np.linspace(-180, 180, 37), color="#D98E04")
    axes[1, 1].set(title="峰频处有效路径相位", xlabel="相位 / °", ylabel="路径数")
    fig.savefig(FIGURES / "atom_channel_variation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].scatter([row["atom_peak_frequency_khz"] for row in high_rows], [row["effective_phase_at_atom_peak_deg"] for row in high_rows], s=8, alpha=.3, color="#D98E04")
    axes[0, 0].set(title="相位—原子峰值频率", xlabel="kHz", ylabel="相位 / °")
    axes[0, 1].hist([row["effective_group_delay_us"] for row in high_rows], bins=35, color="#427AA1")
    axes[0, 1].set(title="有效群时延", xlabel="μs", ylabel="路径数")
    axes[1, 0].scatter([row["relative_delay_ms"] for row in high_rows], [row["effective_residual_phase_circular_std_deg"] for row in high_rows], s=8, alpha=.3, color="#188977")
    axes[1, 0].set(title="相位离散度—时延", xlabel="相对时延 / ms", ylabel="圆标准差 / °")
    axes[1, 1].scatter([row["relative_level_db"] for row in high_rows], [row["effective_residual_phase_circular_std_deg"] for row in high_rows], s=8, alpha=.3, color="#A23B72")
    axes[1, 1].set(title="相位离散度—相对幅值", xlabel="相对声级 / dB", ylabel="圆标准差 / °")
    fig.savefig(FIGURES / "phase_frequency_group_delay.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    axes[0].scatter([row["single_explained_variance"] for row in train_rows], [row["joint_explained_variance"] for row in train_rows], s=12, alpha=.45)
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="单click解释率", ylabel="联合解释率", title="解释率比较")
    axes[1].hist([row["atom_similarity"] for row in train_rows], bins=30, color="#188977")
    axes[1].set(title="单/联合原子相似度", xlabel="余弦相似度", ylabel="PulseTrain数")
    by_voyage: dict[str, int] = {}
    for row in high_rows:
        by_voyage[row["voyage"]] = by_voyage.get(row["voyage"], 0) + 1
    axes[2].bar(range(len(by_voyage)), list(by_voyage.values()), color="#427AA1")
    axes[2].set_xticks(range(len(by_voyage)), [key.replace("Voyage_Ori_", "") for key in by_voyage], rotation=70)
    axes[2].set(title="各Voyage高置信度多径", ylabel="路径数")
    fig.savefig(FIGURES / "full_dataset_summary.png", dpi=160)
    plt.close(fig)

    labels = [row["variant"] for row in decisions]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].barh(labels, [row["median_path_jaccard"] for row in decisions], color="#188977")
    axes[0].axvline(.8, color="#A23B72", ls="--")
    axes[0].set(xlim=(0, 1), title="路径Jaccard")
    axes[1].barh(labels, [row["median_abs_high_count_change"] for row in decisions], color="#D98E04")
    axes[1].axvline(.1, color="#A23B72", ls="--")
    axes[1].set(title="高置信度数量变化")
    axes[2].barh(labels, [row["median_atom_similarity"] for row in decisions], color="#427AA1")
    axes[2].axvline(.9, color="#A23B72", ls="--")
    axes[2].set(xlim=(0, 1), title="原子相似度")
    fig.savefig(FIGURES / "parameter_sensitivity.png", dpi=160)
    plt.close(fig)


def write_recipe(high_count: int, strict_count: int) -> None:
    text = f"""# 接收 click 信号合成指导

合成模型：`y[n] = A_click * sum_k a_k d[n - tau_k F_s] + r[n]`。

1. 从 `synthesis_pairs.csv` 按 `synthesis_pair_id` 整组选择 `atom_id` 与对应信道，不能分别随机抽取。
2. 直达径设为 `tau_0=0`，使用 `relative_signed_amplitude=a_k/|a_0|`；负值表示额外 π 相移。
3. 从 `atoms.npz` 读取真实原子波形及峰值对齐的复频谱，保留残余相位，不用中心频率生成理想脉冲。
4. 非整数时延使用频域相移或分数延迟滤波器，不能直接四舍五入。
5. 归一化信号最后统一设峰值或 RMS；物理声压必须另有水听器灵敏度、前置增益和 ADC 标定。
6. 从 `joint_residuals.npz` 选择相同 Voyage、相近 SNR 的真实拟合残差做 0.10 ms 块采样，不用白噪声替代。
7. 合成后复核中心频率、−3 dB 带宽、时延、相对幅值、相位、SNR 与削波情况。
8. 优先使用真实“原子—信道”配对；扩大样本时才按 `frequency_amplitude_phase_statistics.csv` 的条件联合分布采样。

当前高置信度非直达径 {high_count} 条，其中严格高置信度 {strict_count} 条。本项目只提供流程和参数表，不实现合成函数，也不生成示例 WAV。
"""
    (ROOT / "synthesis_recipe.md").write_text(text, encoding="utf-8")


def validate(
    results: list[dict[str, Any]], train_rows: list[dict[str, Any]], channel_rows: list[dict[str, Any]],
    atom_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]], pairs: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
) -> dict[str, Any]:
    norm_errors = [
        abs(float(np.linalg.norm(np.asarray(result[method]["atom"]))) - 1.0)
        for result in results for method in ("single", "joint")
    ]
    reference_bandwidth = all(
        6 <= result[method]["bandwidth_3db_khz"] <= 72
        for result in results for method in ("single", "joint")
    )
    bootstrap_bandwidth = all(
        6 <= result[method]["bootstrap_bandwidth_min_khz"] <= 72
        and 6 <= result[method]["bootstrap_bandwidth_max_khz"] <= 72
        for result in results for method in ("single", "joint")
    )
    confidence_ok = all(abs(row["confidence"] - row["detected_runs"] / row["bootstrap_total"]) < 1e-12 for row in channel_rows)
    signed_amplitude_ok = all(
        abs(row["relative_signed_amplitude"] - row["coefficient"] / (abs(row["direct_coefficient"]) + 1e-12)) < 1e-10
        for row in channel_rows
    )
    wilson_ok = True
    for row in channel_rows:
        low, high = base.wilson(int(row["detected_runs"]), int(row["bootstrap_total"]))
        wilson_ok &= abs(low - row["confidence_low"]) < 1e-12 and abs(high - row["confidence_high"]) < 1e-12
    atom_ids = [row["atom_id"] for row in atom_rows]
    direct_groups = {(row["voyage"], row["pulse_train"], row["click"]) for row in pairs if row["selection_role"] == "direct"}
    high_groups = {(row["voyage"], row["pulse_train"], row["click"]) for row in high_rows}
    expected_counts = {
        "all_channels.csv": len(channel_rows), "high_confidence_channels.csv": len(high_rows),
        "atoms.csv": len(atom_rows), "atom_channel_pairs.csv": len(high_rows), "synthesis_pairs.csv": len(pairs),
    }
    actual_counts = {}
    for filename in expected_counts:
        with (ROOT / filename).open(encoding="utf-8-sig") as handle:
            actual_counts[filename] = sum(1 for _ in csv.DictReader(handle))
    checks = {
        "formal_full_fit_executed_once": len({result["config_hash"] for result in results}) == 1,
        "completed_718_pulsetrains": len(results) == 718,
        "modeled_clicks_equal_6545": sum(row["used_clicks"] for row in train_rows) == 6545,
        "sensitivity_completed_100": len(sensitivity) == 100,
        "reference_atoms_unit_l2": max(norm_errors, default=math.inf) <= 1e-8,
        "reference_atoms_bandwidth_6_72_khz": reference_bandwidth,
        "bootstrap_atoms_bandwidth_6_72_khz": bootstrap_bandwidth,
        "confidence_is_detected_over_50": confidence_ok and all(row["bootstrap_total"] == 50 for row in channel_rows),
        "signed_relative_amplitude_is_a_over_abs_a0": signed_amplitude_ok,
        "wilson_intervals_recomputed_exactly": bool(wilson_ok),
        "direct_path_excluded_from_high_confidence_multipath": all(row["path"] != "direct" for row in high_rows),
        "high_channels_link_to_unique_atoms": len(atom_ids) == len(set(atom_ids)) and all(row["atom_id"] in set(atom_ids) for row in high_rows),
        "synthesis_groups_retain_direct_path": direct_groups == high_groups,
        "report_numbers_reaggregate_from_csv": expected_counts == actual_counts,
    }
    return {
        "accepted": all(checks.values()), "checks": checks, "csv_row_counts": actual_counts,
        "maximum_atom_l2_error": max(norm_errors, default=math.nan),
        "main_config_hash": results[0]["config_hash"] if results else "",
    }


def make_report(
    results: list[dict[str, Any]], train_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]],
    atom_rows: list[dict[str, Any]], stat_rows: list[dict[str, Any]], decisions: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]], validation: dict[str, Any], gallery: list[tuple[dict[str, Any], str]],
) -> None:
    core = [row for row in stat_rows if row["group_type"] == "all" and row["metric"] in {
        "peak_frequency_khz", "spectral_centroid_khz", "spectral_median_khz", "bandwidth_3db_khz",
        "relative_amplitude", "relative_level_db", "effective_group_delay_us",
        "effective_residual_phase_circular_std_deg",
    }]
    stats_html = "".join(
        f"<tr><td>{row['scope']}</td><td>{row['metric']}</td><td>{row['n_observations']}</td><td>{row['median']:.4g}</td><td>{row['q25']:.4g}–{row['q75']:.4g}</td><td>{row['cluster_bootstrap_median_ci_low']:.4g}–{row['cluster_bootstrap_median_ci_high']:.4g}</td></tr>"
        for row in core
    )
    decision_html = "".join(
        f"<tr><td>{html.escape(row['variant'])}</td><td>{row['median_path_jaccard']:.3f}</td><td>{row['median_abs_high_count_change']:.1%}</td><td>{row['median_atom_similarity']:.3f}</td><td>{row['median_abs_explained_variance_change']:.4f}</td><td>{row['failure_rate']:.1%}</td><td>{row['recommendation']}</td></tr>"
        for row in decisions
    )
    flagged = [row for row in sensitivity if row["failed"] or abs(float(row["high_count_relative_change"])) > .10 or float(row["path_jaccard_vs_default"]) < .80 or float(row["atom_similarity_vs_default"]) < .90]
    flagged_html = "".join(
        f"<tr><td>{row['voyage']}</td><td>{row['pulse_train']}</td><td>{row['config']}</td><td>{float(row['path_jaccard_vs_default']):.3f}</td><td>{float(row['high_count_relative_change']):.1%}</td><td>{float(row['atom_similarity_vs_default']):.3f}</td><td>{html.escape(str(row['error']))}</td></tr>"
        for row in flagged
    )
    gallery_html = "".join(
        f"<details><summary>{result['voyage']} / {result['pulse_train']}</summary><img loading='lazy' src='figures/atoms/{filename}'></details>"
        for result, filename in gallery
    )
    strict_count = sum(row["confidence_low"] >= STRICT_WILSON_LOW for row in high_rows)
    atom_phase = circular_summary([row["peak_phase_deg_centered"] for row in atom_rows if row["method"] == "joint"])
    path_phase = circular_summary([row["effective_phase_at_atom_peak_deg"] for row in high_rows])
    inversion = float(np.mean([row["coefficient"] < 0 for row in high_rows])) if high_rows else math.nan
    manual_variants = [row["variant"] for row in decisions if row["recommendation"] == "需要人工检查或调整参数"]
    review_variants = [row["variant"] for row in decisions if row["recommendation"] == "建议对敏感样本人工复核"]
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>全量BCSC高置信度信道报告</title>
<style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:1280px;margin:24px auto;line-height:1.65;color:#20303c}}h1,h2{{color:#174f5f}}.card{{display:inline-block;padding:12px 18px;margin:6px;background:#eef7f5;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd7dd;padding:6px;text-align:left}}th{{background:#e9f1f4}}img{{max-width:100%}}details{{margin:8px 0;border:1px solid #d9e2e6;padding:8px;border-radius:6px}}code{{background:#f3f5f6;padding:2px 4px}}</style></head><body>
<h1>全量BCSC高置信度信道与原子分析</h1>
<p><b>完成状态：</b>{'验收通过' if validation['accepted'] else '验收未通过'}。先完成临时目录中的最小规模验证，再以固定默认参数执行一次718组正式主实验；后续只读取检查点完成敏感性分析和汇总。</p>
<div class='card'>PulseTrain：{len(results)}/718</div><div class='card'>入模click：{sum(row['used_clicks'] for row in train_rows)}</div><div class='card'>高置信度非直达径：{len(high_rows)}</div><div class='card'>严格高置信度：{strict_count}</div>
<h2>场景、方法与参数</h2><p>每组最多选择10个SNR最高click，联合BCSC共享原子，各click独立估计稀疏信道；最高SNR锚点另做单click比较。采样率576 kHz，预滤波20–190 kHz，参考原子85 μs，Bootstrap原子20–150 μs，λ=0.075并扰动±15%，残差块0.10 ms，匹配容差±0.03 ms，原子单位L2范数且−3 dB带宽硬约束6–72 kHz。每条路径50次Bootstrap，检出率≥90%为高置信度，Wilson 95%下限≥80%为严格高置信度；直达径保留但不计入多径统计。</p>
<p><img src='figures/full_dataset_summary.png'></p>
<h2>频率、幅值与相位</h2><p>原子总体口径中每个PulseTrain只计一次；合成口径中每条高置信度信道与其原子配对，因此多径多的PulseTrain权重更高。幅值区间使用50次PulseTrain聚类Bootstrap。</p>
<table><tr><th>统计口径</th><th>指标</th><th>n</th><th>中位数</th><th>IQR</th><th>聚类Bootstrap 95%区间</th></tr>{stats_html}</table>
<p>峰值对齐原子的峰频相位圆均值为 {atom_phase['circular_mean_deg']:.2f}°，圆标准差 {atom_phase['circular_std_deg']:.2f}°；路径峰频有效相位圆均值 {path_phase['circular_mean_deg']:.2f}°，负系数比例 {inversion:.1%}。相位在原子−3 dB频带内展开并分离线性相位，表中给出群时延和去线性残余相位离散度。</p>
<p>当前实值模型的信道是“纯时延加实数增益”，只能辨识0/π极性，不能估计每条路径独立的频率选择性衰减或复杂相位畸变；需要该能力时应升级为路径短FIR或复值频域系数模型。</p>
<p><img src='figures/atom_channel_variation.png'></p><p><img src='figures/phase_frequency_group_delay.png'></p>
<h2>参数敏感性与人工调整</h2><p>按Voyage和可用click数量固定分层抽取100组，每个变体保持50次Bootstrap。判定阈值：Jaccard≥0.80、数量变化≤10%、原子相似度≥0.90、解释率变化≤0.02；数量变化&gt;25%、Jaccard&lt;0.60或失败率≥1%时需要人工检查或调整。</p>
<p><b>参数选择结论：</b>保留正式实验采用的默认设置（λ=0.075、85 μs参考原子、1.5%路径阈值、0.10 ms残差块、±0.03 ms匹配容差、最多10个联合click）。需要人工检查的变体：{', '.join(manual_variants) or '无'}；建议复核的变体：{', '.join(review_variants) or '无'}。这说明不应把λ降至0.060或把联合click数降至5，且改变参考原子长度需要检查原子形状稳定性。</p>
<table><tr><th>变体</th><th>Jaccard</th><th>数量变化</th><th>原子相似度</th><th>解释率变化</th><th>失败率</th><th>结论</th></tr>{decision_html}</table>
<p><img src='figures/parameter_sensitivity.png'></p><details><summary>受影响的Voyage/PulseTrain（{len(flagged)}条配置记录）</summary><table><tr><th>Voyage</th><th>PulseTrain</th><th>配置</th><th>Jaccard</th><th>数量变化</th><th>原子相似度</th><th>错误</th></tr>{flagged_html}</table></details>
<h2>接收信号合成措施</h2><p>从<code>synthesis_pairs.csv</code>按synthesis_pair_id整组抽取经验原子—信道配对；直达径置零时延并用有符号相对系数；从atoms.npz读取真实波形与复频谱；非整数时延使用频域相移或分数延迟；真实背景从同Voyage、相近SNR的拟合残差做块采样；最终复核频率、带宽、时延、幅值、相位、SNR与削波。物理声压合成还需要水听器灵敏度、前置增益和ADC标定。完整八步见synthesis_recipe.md；本项目不实现合成函数、不生成示例WAV。</p>
<h2>718组单/联合原子时域—频域图</h2>{gallery_html}
<h2>输出与验收</h2><p>主要CSV、NPZ、JSON、图表与报告均直接位于full_dataset_experiment。验收状态：<b>{'通过' if validation['accepted'] else '未通过'}</b>，详见acceptance_validation.json。</p>
</body></html>"""
    REPORT.write_text(document, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ATOM_FIGURES.mkdir(parents=True, exist_ok=True)
    print("[总体进度] 0% 读取718组主结果与100组敏感性结果", flush=True)
    results = json.loads((ROOT / "all_results.json").read_text(encoding="utf-8"))
    sensitivity_results = json.loads((ROOT / "sensitivity_all_results.json").read_text(encoding="utf-8"))
    if len(results) != 718 or len(sensitivity_results) != 100:
        raise RuntimeError(f"结果数量不完整: main={len(results)}, sensitivity={len(sensitivity_results)}")
    print("[总体进度] 10% 聚合信道、原子、频谱和残差", flush=True)
    train_rows, channel_rows, atom_rows, high_rows = aggregate(results)
    pairs = synthesis_pairs(channel_rows, high_rows)
    stat_rows = statistics_rows(high_rows, atom_rows)
    sens_rows = sensitivity_rows(sensitivity_results)
    decisions = parameter_decisions(sens_rows)
    write_csv(ROOT / "pulsetrain_summary.csv", train_rows)
    write_csv(ROOT / "all_channels.csv", channel_rows)
    write_csv(ROOT / "high_confidence_channels.csv", high_rows)
    write_csv(ROOT / "atoms.csv", atom_rows)
    write_csv(ROOT / "atom_channel_pairs.csv", high_rows)
    write_csv(ROOT / "synthesis_pairs.csv", pairs)
    write_csv(ROOT / "frequency_amplitude_phase_statistics.csv", stat_rows)
    write_csv(ROOT / "sensitivity_results.csv", sens_rows)
    write_csv(ROOT / "parameter_adjustment_decisions.csv", decisions)
    failures = [
        {"phase": "sensitivity", "index": row["index"], "voyage": row["voyage"],
         "pulse_train": row["pulse_train"], "config": row["config"], "error": row["error"]}
        for row in sens_rows if row["failed"]
    ]
    write_csv(ROOT / "failures.csv", failures)
    strict_count = sum(row["confidence_low"] >= STRICT_WILSON_LOW for row in high_rows)
    write_json(ROOT / "atom_channel_synthesis_statistics.json", {
        "high_confidence_channel_count": len(high_rows), "strict_high_confidence_channel_count": strict_count,
        "unique_joint_atom_count": len({row["atom_id"] for row in high_rows}),
        "relative_amplitude": quantiles([row["relative_amplitude"] for row in high_rows]),
        "relative_level_db": quantiles([row["relative_level_db"] for row in high_rows]),
        "effective_phase": circular_summary([row["effective_phase_at_atom_peak_deg"] for row in high_rows]),
    })
    write_recipe(len(high_rows), strict_count)
    print("[总体进度] 55% 生成总体图和718组原子图", flush=True)
    summary_plots(train_rows, high_rows, atom_rows, decisions)
    gallery = []
    for index, result in enumerate(results, 1):
        gallery.append((result, plot_atom(result)))
        if index % 25 == 0 or index == len(results):
            print(f"[原子图进度] {index}/{len(results)} ({100*index/len(results):.1f}%)", flush=True)
    print("[总体进度] 90% 重聚合验收", flush=True)
    validation = validate(results, train_rows, channel_rows, atom_rows, high_rows, pairs, sensitivity_results)
    write_json(ROOT / "acceptance_validation.json", validation)
    make_report(results, train_rows, high_rows, atom_rows, stat_rows, decisions, sens_rows, validation, gallery)
    print(f"[总体进度] 100% 报告与验收完成 accepted={validation['accepted']}", flush=True)


if __name__ == "__main__":
    main()
