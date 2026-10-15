"""Render four traceable strict-high-confidence BCSC channel examples."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, windows


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\20\full_dataset_experiment")
CHANNELS_CSV = ROOT / "all_channels.csv"
HIGH_CSV = ROOT / "high_confidence_channels.csv"
ATOMS_CSV = ROOT / "atoms.csv"
ATOMS_NPZ = ROOT / "atoms.npz"
OUTPUT_DIR = ROOT / "high_confidence_examples"
SELECTION_CSV = OUTPUT_DIR / "selected_high_confidence_examples.csv"
OVERVIEW_PNG = OUTPUT_DIR / "high_confidence_examples_overview.png"
OVERVIEW_SVG = OUTPUT_DIR / "high_confidence_examples_overview.svg"

FS = 576_000
LOWCUT_HZ = 20_000
HIGHCUT_HZ = 190_000
STRICT_WILSON_LOW = 0.80

INK = "#222222"
MUTED = "#666666"
GRID = "#D9D9D9"
DIRECT = "#D55E00"
MULTIPATH = "#0072B2"
ATOM = "#009E73"
SELECTED = "#CC79A7"


def progress(percent: int, message: str) -> None:
    width = 30
    done = round(width * percent / 100)
    bar = "#" * done + "-" * (width - done)
    print(f"\rOverall progress [{bar}] {percent:3d}%  {message}", end="", flush=True)
    if percent == 100:
        print()


def to_float_audio(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data.astype(np.float64) / max(abs(info.min), info.max)
    return data.astype(np.float64)


def load_signal(path: str) -> tuple[np.ndarray, np.ndarray]:
    sample_rate, raw_data = wavfile.read(path)
    if sample_rate != FS:
        raise ValueError(f"Unexpected sample rate {sample_rate}: {path}")
    raw = to_float_audio(raw_data)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    centered = raw - np.mean(raw)
    sos = butter(
        5,
        [LOWCUT_HZ / (sample_rate / 2), HIGHCUT_HZ / (sample_rate / 2)],
        btype="bandpass",
        output="sos",
    )
    filtered = sosfiltfilt(sos, centered) * windows.tukey(len(centered), alpha=0.04)
    filtered /= np.max(np.abs(filtered)) + np.finfo(float).eps
    return raw, filtered


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def select_examples(strict: pd.DataFrame) -> list[dict[str, object]]:
    specifications = [
        ("短时延（P20）", 0.20),
        ("中等时延（P50）", 0.50),
        ("长时延（P80）", 0.80),
        ("最长时延", 1.00),
    ]
    selected: list[dict[str, object]] = []
    used_clicks: set[tuple[str, str, str]] = set()
    for label, quantile in specifications:
        target = (
            float(strict["relative_delay_ms"].max())
            if quantile == 1.0
            else float(strict["relative_delay_ms"].quantile(quantile))
        )
        candidates = strict.copy()
        candidate_keys = list(
            zip(candidates["voyage"], candidates["pulse_train"], candidates["click"])
        )
        candidates = candidates.loc[[key not in used_clicks for key in candidate_keys]].copy()
        candidates["selection_distance"] = np.abs(candidates["relative_delay_ms"] - target)
        nearest_distance = candidates["selection_distance"].min()
        nearest = candidates.loc[candidates["selection_distance"].eq(nearest_distance)].sort_values(
            ["relative_amplitude", "snr_proxy", "confidence_low"], ascending=False
        )
        row = nearest.iloc[0].to_dict()
        used_clicks.add((str(row["voyage"]), str(row["pulse_train"]), str(row["click"])))
        row["case_label"] = label
        row["selection_quantile"] = quantile
        row["selection_target_delay_ms"] = target
        selected.append(row)
    return selected


def mark_paths(
    ax: plt.Axes,
    paths: pd.DataFrame,
    selected_channel_id: str,
    annotate: bool,
) -> None:
    for rank, (_, path) in enumerate(paths.iterrows()):
        is_direct = path["path"] == "direct"
        is_selected = str(path["channel_id"]) == selected_channel_id
        color = DIRECT if is_direct else (SELECTED if is_selected else MULTIPATH)
        linestyle = "-" if is_direct else "--"
        linewidth = 1.8 if is_direct or is_selected else 1.25
        delay = float(path["relative_delay_ms"])
        ax.axvline(delay, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95, zorder=3)
        if annotate:
            if is_direct:
                label = "直达径\n0 ms"
            else:
                label = (
                    f"{path['path']}\nΔt={delay:.3f} ms\n"
                    f"检出={100 * float(path['confidence']):.0f}%"
                )
            y_fraction = 0.94 - 0.22 * (rank % 3)
            ax.text(
                delay,
                y_fraction,
                label,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7.6,
                color=color,
                fontweight="bold" if is_selected else "normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
                zorder=5,
            )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def draw_case_panels(
    axes: list[plt.Axes],
    case: dict[str, object],
    paths: pd.DataFrame,
    atom: np.ndarray,
    atom_features: pd.Series,
    overview: bool = False,
) -> None:
    raw, filtered = load_signal(str(case["wav_path"]))
    direct_row = paths.loc[paths["path"].eq("direct")].iloc[0]
    direct_sample = int(direct_row["arrival_sample"])
    time_ms = (np.arange(len(raw)) - direct_sample) / FS * 1e3
    strict_paths = paths.loc[
        paths["path"].eq("direct")
        | (paths["path"].ne("direct") & paths["confidence_low"].ge(STRICT_WILSON_LOW))
    ].sort_values("relative_delay_ms")
    max_delay = float(strict_paths["relative_delay_ms"].max())
    zoom_right = min(float(time_ms[-1]), max(0.32, max_delay + 0.16))

    raw_ax, filtered_ax, atom_ax, spectrum_ax = axes
    raw_ax.plot(time_ms, raw, color=INK, linewidth=0.65)
    mark_paths(raw_ax, strict_paths, str(case["channel_id"]), annotate=False)
    raw_ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
    raw_ax.set_title("(a) 原始 click（已标记路径）", loc="left", fontsize=10, fontweight="bold")
    raw_ax.set_xlabel("相对直达径时间（ms）", fontsize=8.5)
    raw_ax.set_ylabel("原始数字幅值", fontsize=8.5)
    style_axis(raw_ax)

    filtered_ax.plot(time_ms, filtered, color=INK, linewidth=0.75)
    mark_paths(filtered_ax, strict_paths, str(case["channel_id"]), annotate=not overview)
    filtered_ax.set_xlim(-0.12, zoom_right)
    filtered_ax.set_ylim(-1.08, 1.08)
    filtered_ax.set_title("(b) 20–190 kHz 预处理信号局部", loc="left", fontsize=10, fontweight="bold")
    filtered_ax.set_xlabel("相对直达径时间（ms）", fontsize=8.5)
    filtered_ax.set_ylabel("峰值归一化幅值", fontsize=8.5)
    style_axis(filtered_ax)

    peak_index = int(np.argmax(np.abs(atom)))
    atom_time_us = (np.arange(len(atom)) - peak_index) / FS * 1e6
    atom_ax.plot(atom_time_us, atom, color=ATOM, linewidth=1.35)
    atom_ax.axvline(0, color=MUTED, linestyle=":", linewidth=1)
    atom_ax.set_title("(c) 联合提取原子（主峰对齐）", loc="left", fontsize=10, fontweight="bold")
    atom_ax.set_xlabel("相对原子主峰时间（μs）", fontsize=8.5)
    atom_ax.set_ylabel("单位 L2 范数幅值", fontsize=8.5)
    style_axis(atom_ax)

    spectrum = np.fft.rfft(atom, n=16_384)
    frequency_khz = np.fft.rfftfreq(16_384, 1 / FS) / 1e3
    magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum), np.finfo(float).eps))
    magnitude_db -= np.max(magnitude_db)
    spectrum_ax.plot(frequency_khz, magnitude_db, color=MULTIPATH, linewidth=1.25)
    left = float(atom_features["left_3db_khz"])
    right = float(atom_features["right_3db_khz"])
    peak_frequency = float(atom_features["peak_frequency_khz"])
    spectrum_ax.axvspan(left, right, color="#56B4E9", alpha=0.18, label="−3 dB 频带")
    spectrum_ax.axvline(peak_frequency, color=DIRECT, linestyle="--", linewidth=1.2)
    spectrum_ax.set_xlim(20, 190)
    spectrum_ax.set_ylim(-45, 2)
    spectrum_ax.set_title("(d) 原子幅度谱", loc="left", fontsize=10, fontweight="bold")
    spectrum_ax.set_xlabel("频率（kHz）", fontsize=8.5)
    spectrum_ax.set_ylabel("相对幅度（dB）", fontsize=8.5)
    style_axis(spectrum_ax)


def render_individual(
    index: int,
    case: dict[str, object],
    paths: pd.DataFrame,
    atom: np.ndarray,
    atom_features: pd.Series,
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(12.5, 9.4), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.05, 1.0],
        left=0.07,
        right=0.98,
        bottom=0.10,
        top=0.865,
        hspace=0.36,
        wspace=0.12,
    )
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, :]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    draw_case_panels(axes, case, paths, atom, atom_features)
    signed = float(case["relative_signed_amplitude"])
    fig.suptitle(
        f"严格高置信度信道案例 {index}｜{case['case_label']}",
        fontsize=17,
        fontweight="bold",
        color=INK,
        y=0.982,
    )
    fig.text(
        0.5,
        0.935,
        (
            f"{case['voyage']} / {case['pulse_train']} / {case['click']} / {case['path']}   "
            f"Δt={float(case['relative_delay_ms']):.3f} ms   "
            f"a/|a0|={signed:+.3f} ({float(case['relative_level_db']):.1f} dB)   "
            f"检出率={100 * float(case['confidence']):.0f}%   "
            f"Wilson下限={100 * float(case['confidence_low']):.1f}%"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
    )
    legend = [
        Line2D([0], [0], color=DIRECT, linewidth=1.8, label="直达径"),
        Line2D([0], [0], color=MULTIPATH, linestyle="--", linewidth=1.4, label="其他严格高置信度多径"),
        Line2D([0], [0], color=SELECTED, linestyle="--", linewidth=2.0, label="本案例目标多径"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    stem = f"case_{index:02d}_{case['selection_quantile']:.2f}_{case['voyage']}_{case['pulse_train']}_{case['click']}"
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def render_overview(
    selected: list[dict[str, object]],
    all_channels: pd.DataFrame,
    atom_lookup: dict[str, np.ndarray],
    atom_features: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(23, 15.5), facecolor="white", constrained_layout=True)
    for index, case in enumerate(selected):
        key_mask = (
            all_channels["method"].eq("joint")
            & all_channels["voyage"].eq(case["voyage"])
            & all_channels["pulse_train"].eq(case["pulse_train"])
            & all_channels["click"].eq(case["click"])
        )
        paths = all_channels.loc[key_mask].copy()
        atom_id = str(case["atom_id"])
        draw_case_panels(list(axes[index]), case, paths, atom_lookup[atom_id], atom_features.loc[atom_id], overview=True)
        axes[index, 0].text(
            -0.25,
            0.5,
            f"案例 {index + 1}\n{case['case_label']}\nΔt={float(case['relative_delay_ms']):.3f} ms",
            transform=axes[index, 0].transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=SELECTED,
        )
    fig.suptitle("4 个严格高置信度信道及对应原始 click 与联合原子", fontsize=19, fontweight="bold", color=INK)
    fig.savefig(OVERVIEW_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OVERVIEW_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    configure_plotting()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress(0, "Loading channels and atom metadata")
    all_channels = pd.read_csv(CHANNELS_CSV)
    high = pd.read_csv(HIGH_CSV)
    atoms_meta = pd.read_csv(ATOMS_CSV).set_index("atom_id")
    strict = high.loc[high["confidence_low"].ge(STRICT_WILSON_LOW)].copy()
    selected = select_examples(strict)

    progress(18, "Loading extracted atom waveforms")
    atom_data = np.load(ATOMS_NPZ)
    atom_lookup: dict[str, np.ndarray] = {}
    for atom_id, row, length in zip(
        atom_data["atom_ids"], atom_data["atoms"], atom_data["atom_lengths"]
    ):
        atom_lookup[str(atom_id)] = np.asarray(row[: int(length)], dtype=float)

    progress(30, "Writing traceable case selection")
    selection_rows: list[dict[str, object]] = []
    outputs: list[tuple[Path, Path]] = []
    for index, case in enumerate(selected, start=1):
        key_mask = (
            all_channels["method"].eq("joint")
            & all_channels["voyage"].eq(case["voyage"])
            & all_channels["pulse_train"].eq(case["pulse_train"])
            & all_channels["click"].eq(case["click"])
        )
        paths = all_channels.loc[key_mask].copy()
        strict_paths = paths.loc[
            paths["path"].eq("direct")
            | (paths["path"].ne("direct") & paths["confidence_low"].ge(STRICT_WILSON_LOW))
        ].sort_values("relative_delay_ms")
        for _, path in strict_paths.iterrows():
            selection_rows.append(
                {
                    "case_index": index,
                    "case_label": case["case_label"],
                    "selection_quantile": case["selection_quantile"],
                    "selection_target_delay_ms": case["selection_target_delay_ms"],
                    "selected_target": str(path["channel_id"]) == str(case["channel_id"]),
                    **path.to_dict(),
                }
            )
        outputs.append(
            render_individual(
                index,
                case,
                paths,
                atom_lookup[str(case["atom_id"])],
                atoms_meta.loc[str(case["atom_id"])],
            )
        )
        progress(30 + index * 12, f"Rendered case {index}/4")

    pd.DataFrame(selection_rows).to_csv(SELECTION_CSV, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    progress(82, "Rendering four-case overview")
    render_overview(selected, all_channels, atom_lookup, atoms_meta)
    progress(100, "Complete")
    print(f"Selection CSV: {SELECTION_CSV}")
    print(f"Overview PNG: {OVERVIEW_PNG}")
    for png_path, svg_path in outputs:
        print(f"Case PNG: {png_path}")
        print(f"Case SVG: {svg_path}")


if __name__ == "__main__":
    main()
