"""Generate all-sample OMP reconstruction diagnostics for experiment 23-v7.

The reconstruction is the convolution of a sparse signed tap train (the saved
OMP delays and jointly re-estimated coefficients) with the matching direct-path
template.  The crop follows the experiment's 5,761-sample waveform convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "23_BCSCStructureClickClassification" / "v7_omp_global_softlabel_proxy"
SOURCE = ROOT / "23_BCSCStructureClickClassification" / "v4_rebalanced_snr_m5_p15_step2_3seeds"
ARCHIVE = EXPERIMENT / "labels" / "omp_labels.npz"
OUTPUT = EXPERIMENT / "omp_statistics"
SAMPLE_RATE_HZ = 576_000
WAVE_SAMPLES = 5_761
CENTER = 2_880
EPS = np.finfo(np.float64).eps

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 200,
})


def parse_sample_id(sample_id: str) -> tuple[int, int]:
    """Return the source click-train and pulse numbers embedded in a v7 ID."""
    parts = str(sample_id).split("_")
    return int(parts[2]), int(parts[4])


def template_path(sample_id: str) -> Path:
    train, pulse = parse_sample_id(sample_id)
    return ROOT / "00_Data" / "ClickTrains" / f"PulseTrain_{train:03d}" / f"Pulse_{pulse}.wav"


def read_wave(path: Path, expected_samples: int | None = None) -> np.ndarray:
    rate, wave = wavfile.read(path, mmap=True)
    if rate != SAMPLE_RATE_HZ:
        raise ValueError(f"Unexpected sample rate in {path}: {rate}")
    value = np.asarray(wave, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or (expected_samples is not None and value.size != expected_samples):
        raise ValueError(f"Invalid waveform: {path}")
    return value


def reconstruct_by_convolution(template: np.ndarray, delays: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Convolve the sparse OMP tap train with its template and crop to 5,761 samples."""
    peak = int(np.argmax(np.abs(template)))
    starts = CENTER - peak + delays.astype(int)
    offset = max(0, -int(starts.min()))
    taps = np.zeros(int(starts.max()) + offset + 1, dtype=np.float64)
    taps[starts + offset] = coefficients
    convolved = np.convolve(taps, template, mode="full")
    cropped = convolved[offset : offset + WAVE_SAMPLES]
    if cropped.size < WAVE_SAMPLES:
        cropped = np.pad(cropped, (0, WAVE_SAMPLES - cropped.size))
    return cropped


def pearson_r(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    return float(np.dot(first_centered, second_centered) / (np.linalg.norm(first_centered) * np.linalg.norm(second_centered) + EPS))


def summary_row(name: str, values: np.ndarray, unit: str) -> dict[str, float | str]:
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "metric": name,
        "unit": unit,
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SOURCE / "dataset_manifest.csv").reset_index(drop=True)
    with np.load(ARCHIVE, allow_pickle=True) as archive:
        labels = archive["physical_labels"].astype(np.float64)
        coefficients = archive["coefficients"].astype(np.float64)
        delays = archive["delays"].astype(np.int16)
        valid = archive["aux_valid"].astype(bool)
    indices = np.flatnonzero(valid)
    records: list[dict[str, float | int | str]] = []
    for index in tqdm(indices, desc="总体进度：OMP 卷积重建与统计", unit="样本"):
        row = manifest.iloc[int(index)]
        original = read_wave(Path(row.wav_path), expected_samples=WAVE_SAMPLES)
        template = read_wave(template_path(str(row.sample_id)))
        reconstruction = reconstruct_by_convolution(template, delays[index], coefficients[index])
        residual = original - reconstruction
        records.append({
            "manifest_index": int(index),
            "sample_id": str(row.sample_id),
            "test_fold": int(row.test_fold),
            "target_snr_db": float(row.target_snr_db),
            "pearson_r": pearson_r(original, reconstruction),
            "relative_residual_l2": float(np.linalg.norm(residual) / (np.linalg.norm(original) + EPS)),
            "reconstruction_to_original_l2_gain": float(np.linalg.norm(reconstruction) / (np.linalg.norm(original) + EPS)),
            "mean_delay_us": float(labels[index, 0]),
            "rms_delay_spread_us": float(labels[index, 1]),
            "max_excess_delay_us": float(labels[index, 2]),
            "rician_k_db": float(labels[index, 3]),
            "coherence_bandwidth_hz": float(labels[index, 5]),
            "selected_path_count": int(delays.shape[1]),
        })
    samples = pd.DataFrame(records)
    samples.to_csv(OUTPUT / "omp_per_sample_statistics.csv", index=False, encoding="utf-8-sig")

    metrics = [
        ("Pearson 相关系数", "pearson_r", "", False),
        ("相对残差 L2", "relative_residual_l2", "", True),
        ("恢复/原始 L2 增益", "reconstruction_to_original_l2_gain", "", False),
        ("平均附加时延", "mean_delay_us", "μs", True),
        ("RMS 时延扩展", "rms_delay_spread_us", "μs", True),
        ("最大附加时延", "max_excess_delay_us", "μs", True),
        ("莱斯 K 因子", "rician_k_db", "dB", False),
        ("相干带宽预估", "coherence_bandwidth_hz", "Hz", True),
    ]
    summaries = [summary_row(title, samples[column].to_numpy(), unit) for title, column, unit, _ in metrics]
    summaries.append(summary_row("OMP 选中路径数", samples["selected_path_count"].to_numpy(), "paths"))
    pd.DataFrame(summaries).to_csv(OUTPUT / "omp_summary_statistics.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for axis, (title, column, unit, log_x) in zip(axes.flat, metrics):
        values = samples[column].to_numpy()
        axis.hist(values, bins=50, color="#2878B5", edgecolor="white", linewidth=0.35)
        median = float(np.median(values))
        axis.axvline(median, color="#C82423", linewidth=1.4, label=f"中位数 {median:.4g}")
        if log_x:
            positive = values[values > 0]
            if positive.size:
                axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel(unit)
        axis.set_ylabel("样本数")
        axis.legend(fontsize=8)
    figure.suptitle("实验 23-v7：OMP 卷积重建与信道统计（n=3,500）", fontsize=15)
    figure.savefig(OUTPUT / "omp_reconstruction_and_channel_histograms.png", bbox_inches="tight")
    plt.close(figure)

    delay_us = delays[valid, 1:].astype(np.float64).ravel() / SAMPLE_RATE_HZ * 1e6
    coefficient_abs = np.abs(coefficients[valid]).ravel()
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].hist(delay_us, bins=np.geomspace(delay_us[delay_us > 0].min(), delay_us.max(), 60), color="#5AB4AC", edgecolor="white", linewidth=0.3)
    axes[0].set_xscale("log")
    axes[0].set_title("全部附加 OMP 路径时延")
    axes[0].set_xlabel("附加时延 (μs，对数轴)")
    axes[0].set_ylabel("路径数")
    axes[1].hist(coefficient_abs, bins=np.geomspace(coefficient_abs[coefficient_abs > 0].min(), coefficient_abs.max(), 60), color="#D89216", edgecolor="white", linewidth=0.3)
    axes[1].set_xscale("log")
    axes[1].set_title("全部 OMP 系数绝对值")
    axes[1].set_xlabel("|coefficient|（原始模板尺度，对数轴）")
    axes[1].set_ylabel("原子数")
    figure.suptitle("实验 23-v7：OMP 原子分布（49,000 条附加路径；52,500 个原子）", fontsize=14)
    figure.savefig(OUTPUT / "omp_atom_histograms.png", bbox_inches="tight")
    plt.close(figure)

    report = {
        "sample_count": int(len(samples)),
        "reconstruction": "sparse signed OMP tap train convolved with the matching direct-path template; cropped to 5761 samples",
        "similarity": "Pearson correlation coefficient between original and reconstructed waveform",
        "relative_residual_l2": "||original - reconstruction||_2 / ||original||_2",
        "l2_gain": "||reconstruction||_2 / ||original||_2",
        "path_count": "The OMP protocol fixes this at 15 (one direct plus fourteen additional paths).",
        "caution": "Raw OMP coefficients depend on the template amplitude scale, so their magnitudes are plotted but are not used as a cross-sample channel-gain metric.",
    }
    (OUTPUT / "readme.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summaries).set_index("metric")
    def cell(metric: str, column: str, digits: int = 4) -> str:
        return f"{float(summary.loc[metric, column]):.{digits}g}"
    markdown = [
        "# 实验 23-v7：OMP 卷积重建统计摘要",
        "",
        "- 样本：3,500 条有效正样本。",
        "- 恢复：将保存的 15 个 OMP 延迟/有符号系数组成稀疏抽头序列，与对应直达模板卷积，再按原实验裁剪为 5,761 点。",
        "- 相似度：Pearson 相关系数；相对残差为 `||x-x̂||₂ / ||x||₂`；L2 增益为 `||x̂||₂ / ||x||₂`。",
        "",
        "| 指标 | 单位 | 均值 | 中位数 | P05 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in summary.index:
        markdown.append(f"| {metric} | {summary.loc[metric, 'unit']} | {cell(metric, 'mean')} | {cell(metric, 'median')} | {cell(metric, 'p05')} | {cell(metric, 'p95')} |")
    degenerate = int((samples["rms_delay_spread_us"] < 1e-6).sum())
    markdown.extend([
        "",
        f"> 有 {degenerate} 条样本的 RMS 时延扩展近似为零；相干带宽公式因此被 `EPS` 限制到约 1e12 Hz。该少数退化值显著抬高了相干带宽均值，跨样本比较建议使用中位数和分位数。",
        "> OMP 协议固定选择 15 个原子（1 条直达路径 + 14 条附加路径）；故“选中路径数”没有样本间变异。原始 OMP 系数幅度依赖模板幅值尺度，未将其解释为跨样本可比的总信道增益。",
    ])
    (OUTPUT / "OMP统计摘要.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"已输出：{OUTPUT}")


if __name__ == "__main__":
    main()
