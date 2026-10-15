"""Export 263 PulseTrain-level strong pseudo-channel labels and all model-input clicks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, windows


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\20\full_dataset_experiment")
OUTPUT_ROOT = ROOT / "strong_channel_labels_263"
ALL_CHANNELS_CSV = ROOT / "all_channels.csv"
HIGH_CHANNELS_CSV = ROOT / "high_confidence_channels.csv"
SUMMARY_CSV = ROOT / "pulsetrain_summary.csv"

FS = 576_000
LOWCUT_HZ = 20_000
HIGHCUT_HZ = 190_000
MATCH_TOLERANCE_MS = 0.03
MIN_CLICK_SUPPORT = 0.80
MIN_WILSON_LOW = 0.80

INK = "#222222"
MUTED = "#666666"
GRID = "#D9D9D9"
DIRECT_COLOR = "#D55E00"
POSITIVE_COLOR = "#0072B2"
NEGATIVE_COLOR = "#CC79A7"


def progress(done: int, total: int, message: str) -> None:
    fraction = 0 if total == 0 else done / total
    width = 32
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rOverall progress [{bar}] {100 * fraction:6.2f}%  "
        f"{message} ({done}/{total})",
        end="",
        flush=True,
    )
    if done == total:
        print()


def to_float_audio(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data.astype(np.float64) / max(abs(info.min), info.max)
    return data.astype(np.float64)


def preprocess_click(path: str | Path) -> tuple[np.ndarray, dict[str, float | int]]:
    sample_rate, raw_data = wavfile.read(path)
    if sample_rate != FS:
        raise ValueError(f"Sample rate is not {FS}: {path}")
    raw = to_float_audio(raw_data)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    raw_mean = float(np.mean(raw))
    centered = raw - raw_mean
    raw_peak = float(np.max(np.abs(centered)))
    raw_rms = float(np.sqrt(np.mean(centered * centered)))
    nyquist = sample_rate / 2
    sos = butter(
        5,
        [LOWCUT_HZ / nyquist, HIGHCUT_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )
    filtered = sosfiltfilt(sos, centered)
    filtered *= windows.tukey(len(filtered), alpha=0.04)
    filtered_peak = float(np.max(np.abs(filtered)))
    filtered_rms = float(np.sqrt(np.mean(filtered * filtered)))
    normalized = filtered / (filtered_peak + np.finfo(float).eps)
    return normalized.astype(np.float32), {
        "sample_rate": sample_rate,
        "signal_length": len(normalized),
        "raw_mean": raw_mean,
        "raw_peak": raw_peak,
        "raw_rms": raw_rms,
        "filtered_peak_before_normalization": filtered_peak,
        "filtered_rms_before_normalization": filtered_rms,
    }


def choose_one_per_click(rows: pd.DataFrame, center: float) -> pd.DataFrame:
    candidates = rows.loc[
        (rows["relative_delay_ms"] - center).abs().le(MATCH_TOLERANCE_MS)
    ].copy()
    if candidates.empty:
        return candidates
    candidates["distance_to_center_ms"] = (
        candidates["relative_delay_ms"] - center
    ).abs()
    return (
        candidates.sort_values(
            ["click", "distance_to_center_ms", "confidence_low", "confidence"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates("click", keep="first")
        .copy()
    )


def extract_stable_clusters(
    strict_paths: pd.DataFrame, used_clicks: int
) -> list[dict[str, Any]]:
    required_clicks = math.ceil(used_clicks * MIN_CLICK_SUPPORT - 1e-12)
    remaining = strict_paths.copy()
    raw_clusters: list[dict[str, Any]] = []
    while not remaining.empty:
        best_members: pd.DataFrame | None = None
        best_score: tuple[float, ...] | None = None
        for center in np.sort(remaining["relative_delay_ms"].unique()):
            members = choose_one_per_click(remaining, float(center))
            support = members["click"].nunique()
            if support == 0:
                continue
            score = (
                float(support),
                float(members["confidence_low"].sum()),
                -float(members["distance_to_center_ms"].median()),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_members = members
        if best_members is None or best_members["click"].nunique() < required_clicks:
            break

        refined_center = float(best_members["relative_delay_ms"].median())
        refined_members = choose_one_per_click(remaining, refined_center)
        if refined_members["click"].nunique() >= required_clicks:
            best_members = refined_members

        raw_clusters.append({"members": best_members.copy()})
        remaining = remaining.drop(index=best_members.index)

    raw_clusters.sort(
        key=lambda cluster: float(cluster["members"]["relative_delay_ms"].median())
    )
    clusters: list[dict[str, Any]] = []
    for path_index, cluster in enumerate(raw_clusters, start=1):
        members = cluster["members"].copy()
        signed = members["relative_signed_amplitude"].astype(float)
        absolute = members["relative_amplitude"].astype(float)
        delays = members["relative_delay_ms"].astype(float)
        delay_center = float((delays.min() + delays.max()) / 2)
        clusters.append(
            {
                "path_label": f"stable_multipath_{path_index}",
                "delay_center_ms": delay_center,
                "delay_median_ms": float(delays.median()),
                "delay_q1_ms": float(delays.quantile(0.25)),
                "delay_q3_ms": float(delays.quantile(0.75)),
                "signed_amplitude_median": float(signed.median()),
                "signed_amplitude_q1": float(signed.quantile(0.25)),
                "signed_amplitude_q3": float(signed.quantile(0.75)),
                "relative_amplitude_median": float(absolute.median()),
                "relative_level_db_median": float(
                    members["relative_level_db"].astype(float).median()
                ),
                "support_clicks": int(members["click"].nunique()),
                "used_clicks": used_clicks,
                "support_fraction": float(members["click"].nunique() / used_clicks),
                "confidence_median": float(members["confidence"].median()),
                "wilson_low_median": float(members["confidence_low"].median()),
                "wilson_low_min": float(members["confidence_low"].min()),
                "members": members,
            }
        )
    return clusters


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def render_preview(
    output_path: Path,
    label_id: str,
    voyage: str,
    pulse_train: str,
    click_ids: list[str],
    waveforms: list[np.ndarray],
    direct_samples: list[int],
    clusters: list[dict[str, Any]],
) -> None:
    n_clicks = len(click_ids)
    columns = 2
    waveform_rows = math.ceil(n_clicks / columns)
    figure_height = 3.1 + 2.15 * waveform_rows
    fig = plt.figure(figsize=(14.5, figure_height), facecolor="white")
    grid = fig.add_gridspec(
        waveform_rows + 1,
        columns,
        height_ratios=[1.15] + [1.0] * waveform_rows,
        left=0.07,
        right=0.98,
        bottom=0.07,
        top=0.92,
        hspace=0.48,
        wspace=0.16,
    )
    channel_ax = fig.add_subplot(grid[0, :])

    delays = [0.0] + [float(c["delay_center_ms"]) for c in clusters]
    amplitudes = [1.0] + [float(c["signed_amplitude_median"]) for c in clusters]
    colors = [DIRECT_COLOR] + [
        POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in amplitudes[1:]
    ]
    markers = ["o"] + ["^" if value >= 0 else "v" for value in amplitudes[1:]]
    channel_ax.axhline(0, color=GRID, linewidth=0.8)
    for delay, amplitude, color, marker in zip(delays, amplitudes, colors, markers):
        channel_ax.vlines(delay, 0, amplitude, color=color, linewidth=2)
        channel_ax.scatter(delay, amplitude, color=color, marker=marker, s=55, zorder=3)
    for cluster in clusters:
        channel_ax.text(
            float(cluster["delay_center_ms"]),
            float(cluster["signed_amplitude_median"]),
            (
                f"  {float(cluster['delay_center_ms']):.3f} ms\n"
                f"  {int(cluster['support_clicks'])}/{n_clicks} clicks"
            ),
            fontsize=8,
            color=INK,
            va="center",
        )
    channel_ax.set_title("(a) PulseTrain级强信道伪标签", loc="left", fontsize=11, fontweight="bold")
    channel_ax.set_xlabel("相对直达径时延（ms）", fontsize=9)
    channel_ax.set_ylabel("有符号相对幅值", fontsize=9)
    channel_ax.grid(axis="y", color=GRID, linewidth=0.7)

    max_delay = max(float(c["delay_center_ms"]) for c in clusters)
    left_limit = -0.20
    right_limit = max(0.50, max_delay + 0.25)
    member_lookup: dict[tuple[str, str], float] = {}
    for cluster in clusters:
        for _, member in cluster["members"].iterrows():
            member_lookup[(str(member["click"]), str(cluster["path_label"]))] = float(
                member["relative_delay_ms"]
            )

    for click_index, (click_id, waveform, direct_sample) in enumerate(
        zip(click_ids, waveforms, direct_samples)
    ):
        row = click_index // columns + 1
        column = click_index % columns
        ax = fig.add_subplot(grid[row, column])
        time_ms = (np.arange(len(waveform)) - direct_sample) / FS * 1e3
        ax.plot(time_ms, waveform, color=INK, linewidth=0.65)
        ax.axvline(0, color=DIRECT_COLOR, linewidth=1.35)
        for cluster in clusters:
            key = (click_id, str(cluster["path_label"]))
            if key not in member_lookup:
                continue
            delay = member_lookup[key]
            amplitude = float(cluster["signed_amplitude_median"])
            color = POSITIVE_COLOR if amplitude >= 0 else NEGATIVE_COLOR
            ax.axvline(delay, color=color, linestyle="--", linewidth=1.15)
        ax.set_xlim(left_limit, right_limit)
        ax.set_ylim(-1.08, 1.08)
        ax.set_title(f"{click_id} 输入波形", loc="left", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("相对直达径时间（ms）", fontsize=8)
        ax.set_ylabel("峰值归一化幅值", fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=7.5, colors=MUTED)
    if n_clicks % columns:
        empty_ax = fig.add_subplot(grid[waveform_rows, 1])
        empty_ax.axis("off")

    legend = [
        Line2D([0], [0], color=DIRECT_COLOR, linewidth=1.7, label="直达径"),
        Line2D([0], [0], color=POSITIVE_COLOR, linestyle="--", linewidth=1.4,
               label="正系数稳定多径"),
        Line2D([0], [0], color=NEGATIVE_COLOR, linestyle="--", linewidth=1.4,
               label="负系数稳定多径"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        f"{label_id}｜{voyage} / {pulse_train}｜全部 {n_clicks} 个单click输入波形",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.945,
        "显示窗口覆盖全部稳定多径；NPZ保存每个click的完整576 kHz模型输入数组",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme(label_count: int, click_count: int) -> None:
    text = f"""# PulseTrain-level strong pseudo-channel labels

This directory contains {label_count} strong channel labels and {click_count} complete single-click model-input waveforms.

## Label definition

- Click-level path confidence is at least 90% and its Wilson 95% lower bound is at least 80%.
- Paths from different clicks match when their relative delays differ by no more than +/-0.03 ms.
- A stable path must occur in at least 80% of the modeled clicks in its PulseTrain.
- One directory represents one PulseTrain-level channel label and includes all stable non-direct paths plus the direct path.

## Files

- `index.csv`: one row per PulseTrain-level label.
- `all_strong_channels.csv`: direct and stable multipath summaries for all labels.
- `all_path_members.csv`: click-level paths supporting each stable common path.
- Each label directory contains `channel_label.csv`, `path_members.csv`, `input_waveforms.npz`, `metadata.json`, and `preview.png`.

`input_waveforms.npz` stores the exact BCSC input convention: mean removal, fifth-order 20-190 kHz zero-phase Butterworth filtering, Tukey window with alpha 0.04, and per-click peak normalization. These arrays are not the unprocessed WAV samples.
"""
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_plotting()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    progress(0, 263, "Loading experiment tables")
    all_channels = pd.read_csv(ALL_CHANNELS_CSV)
    high = pd.read_csv(HIGH_CHANNELS_CSV)
    summary = pd.read_csv(SUMMARY_CSV).sort_values("index")
    strict = high.loc[high["confidence_low"].ge(MIN_WILSON_LOW)].copy()
    strict_groups = {
        key: group.copy()
        for key, group in strict.groupby(["voyage", "pulse_train"], sort=False)
    }
    direct_rows = all_channels.loc[
        all_channels["method"].eq("joint") & all_channels["path"].eq("direct")
    ].copy()
    direct_groups = {
        key: group.sort_values("click").copy()
        for key, group in direct_rows.groupby(["voyage", "pulse_train"], sort=False)
    }

    candidates: list[dict[str, Any]] = []
    for summary_row in summary.itertuples(index=False):
        key = (summary_row.voyage, summary_row.pulse_train)
        paths = strict_groups.get(key)
        if paths is None:
            continue
        clusters = extract_stable_clusters(paths, int(summary_row.used_clicks))
        if clusters:
            candidates.append(
                {
                    "summary": summary_row,
                    "clusters": clusters,
                    "direct_rows": direct_groups[key],
                }
            )
    if len(candidates) != 263:
        raise RuntimeError(f"Expected 263 strong labels, found {len(candidates)}")

    index_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    all_member_rows: list[dict[str, Any]] = []
    total_waveforms = 0
    for label_number, candidate in enumerate(candidates, start=1):
        summary_row = candidate["summary"]
        clusters = candidate["clusters"]
        click_rows = candidate["direct_rows"]
        label_id = (
            f"SC_{label_number:04d}_{summary_row.voyage}_"
            f"{summary_row.pulse_train}"
        )
        label_dir = OUTPUT_ROOT / label_id
        label_dir.mkdir(parents=True, exist_ok=True)

        click_ids: list[str] = []
        waveforms: list[np.ndarray] = []
        waveform_lengths: list[int] = []
        direct_samples: list[int] = []
        source_paths: list[str] = []
        preprocessing_rows: list[dict[str, Any]] = []
        for _, click_row in click_rows.iterrows():
            waveform, preprocessing = preprocess_click(str(click_row["wav_path"]))
            click_id = str(click_row["click"])
            click_ids.append(click_id)
            waveforms.append(waveform)
            waveform_lengths.append(len(waveform))
            direct_samples.append(int(click_row["arrival_sample"]))
            source_paths.append(str(click_row["wav_path"]))
            preprocessing_rows.append({"click": click_id, **preprocessing})
        if len(click_ids) != int(summary_row.used_clicks):
            raise RuntimeError(
                f"{label_id}: expected {summary_row.used_clicks} clicks, found {len(click_ids)}"
            )

        max_length = max(waveform_lengths)
        waveform_matrix = np.full((len(waveforms), max_length), np.nan, dtype=np.float32)
        for row_index, waveform in enumerate(waveforms):
            waveform_matrix[row_index, : len(waveform)] = waveform
        np.savez_compressed(
            label_dir / "input_waveforms.npz",
            click_ids=np.asarray(click_ids),
            waveforms=waveform_matrix,
            waveform_lengths=np.asarray(waveform_lengths, dtype=np.int32),
            direct_samples=np.asarray(direct_samples, dtype=np.int32),
            sample_rate=np.asarray(FS, dtype=np.int32),
            source_wav_paths=np.asarray(source_paths),
        )

        label_rows = [
            {
                "label_id": label_id,
                "voyage": summary_row.voyage,
                "pulse_train": summary_row.pulse_train,
                "path_label": "direct",
                "delay_center_ms": 0.0,
                "delay_median_ms": 0.0,
                "delay_q1_ms": 0.0,
                "delay_q3_ms": 0.0,
                "signed_amplitude_median": 1.0,
                "signed_amplitude_q1": 1.0,
                "signed_amplitude_q3": 1.0,
                "relative_amplitude_median": 1.0,
                "relative_level_db_median": 0.0,
                "support_clicks": len(click_ids),
                "used_clicks": len(click_ids),
                "support_fraction": 1.0,
                "confidence_median": 1.0,
                "wilson_low_median": float(click_rows["confidence_low"].median()),
                "wilson_low_min": float(click_rows["confidence_low"].min()),
            }
        ]
        member_rows: list[dict[str, Any]] = []
        for cluster in clusters:
            label_row = {
                "label_id": label_id,
                "voyage": summary_row.voyage,
                "pulse_train": summary_row.pulse_train,
                **{key: value for key, value in cluster.items() if key != "members"},
            }
            label_rows.append(label_row)
            for _, member in cluster["members"].iterrows():
                member_rows.append(
                    {
                        "label_id": label_id,
                        "common_path_label": cluster["path_label"],
                        **member.to_dict(),
                    }
                )
        pd.DataFrame(label_rows).to_csv(
            label_dir / "channel_label.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(member_rows).to_csv(
            label_dir / "path_members.csv", index=False, encoding="utf-8-sig"
        )

        metadata = {
            "label_id": label_id,
            "label_type": "BCSC_strict_cross_click_pseudo_channel",
            "voyage": summary_row.voyage,
            "pulse_train": summary_row.pulse_train,
            "used_clicks": len(click_ids),
            "stable_multipath_count": len(clusters),
            "atom_id": f"{summary_row.voyage}__{summary_row.pulse_train}__joint",
            "criteria": {
                "click_confidence_min": 0.90,
                "wilson_95_lower_min": MIN_WILSON_LOW,
                "cross_click_support_min": MIN_CLICK_SUPPORT,
                "delay_matching_tolerance_ms": MATCH_TOLERANCE_MS,
            },
            "preprocessing": {
                "sample_rate_hz": FS,
                "mean_removed": True,
                "bandpass_hz": [LOWCUT_HZ, HIGHCUT_HZ],
                "butterworth_order": 5,
                "zero_phase_filter": True,
                "tukey_alpha": 0.04,
                "per_click_peak_normalization": True,
            },
            "click_preprocessing_metrics": preprocessing_rows,
            "source_wav_paths": source_paths,
        }
        (label_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        render_preview(
            label_dir / "preview.png",
            label_id,
            str(summary_row.voyage),
            str(summary_row.pulse_train),
            click_ids,
            waveforms,
            direct_samples,
            clusters,
        )

        minimum_support = min(float(cluster["support_fraction"]) for cluster in clusters)
        index_rows.append(
            {
                "label_id": label_id,
                "voyage": summary_row.voyage,
                "pulse_train": summary_row.pulse_train,
                "used_clicks": len(click_ids),
                "stable_multipath_count": len(clusters),
                "minimum_path_support_fraction": minimum_support,
                "maximum_delay_ms": max(float(c["delay_center_ms"]) for c in clusters),
                "atom_id": metadata["atom_id"],
                "relative_folder": label_id,
                "waveform_npz": f"{label_id}/input_waveforms.npz",
                "channel_csv": f"{label_id}/channel_label.csv",
                "preview_png": f"{label_id}/preview.png",
            }
        )
        all_label_rows.extend(label_rows)
        all_member_rows.extend(member_rows)
        total_waveforms += len(click_ids)
        if label_number == 1 or label_number % 5 == 0 or label_number == len(candidates):
            progress(label_number, len(candidates), "Exporting labels and click inputs")

    index_frame = pd.DataFrame(index_rows)
    label_frame = pd.DataFrame(all_label_rows)
    member_frame = pd.DataFrame(all_member_rows)
    index_frame.to_csv(OUTPUT_ROOT / "index.csv", index=False, encoding="utf-8-sig")
    label_frame.to_csv(
        OUTPUT_ROOT / "all_strong_channels.csv", index=False, encoding="utf-8-sig"
    )
    member_frame.to_csv(
        OUTPUT_ROOT / "all_path_members.csv", index=False, encoding="utf-8-sig"
    )
    write_readme(len(index_frame), total_waveforms)

    config_payload = {
        "sample_rate_hz": FS,
        "bandpass_hz": [LOWCUT_HZ, HIGHCUT_HZ],
        "match_tolerance_ms": MATCH_TOLERANCE_MS,
        "minimum_click_support": MIN_CLICK_SUPPORT,
        "minimum_wilson_low": MIN_WILSON_LOW,
        "label_count": len(index_frame),
        "waveform_count": total_waveforms,
    }
    config_payload["config_sha256"] = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (OUTPUT_ROOT / "export_manifest.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Labels: {len(index_frame)}")
    print(f"Input waveforms: {total_waveforms}")


if __name__ == "__main__":
    main()
