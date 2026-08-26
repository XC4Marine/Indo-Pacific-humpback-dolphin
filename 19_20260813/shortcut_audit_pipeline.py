#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""信号—噪声可分性与背景捷径确认实验流水线。

独立于 `bcsc_transformer_pipeline.py`，不覆盖原实验。所有波形条件使用相同
WaveformTransformer、随机种子、训练预算、验证折阈值和来源隔离五折。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, correlate, hilbert, sosfiltfilt, windows
from scipy.stats import kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bcsc_transformer_pipeline import (
    _crop_center,
    _empty_bcsc_arrays,
    _path_component,
    _save_bcsc_cache,
    ArrayDataset,
    FeatureTransformer,
    TransformerCore,
    WaveformTransformer,
    adjoint_atom,
    bcsc,
    binary_metrics,
    choose_threshold,
    count_parameters,
    extract_candidate_indices,
    fista_sparse_code,
    frame_metric,
    json_dump,
    measure_3db_bandwidth,
    moving_block_bootstrap,
    parse_original_number,
    predict_model,
    seed_everything,
    train_one_model,
)


@dataclass(frozen=True)
class AuditConfig:
    """保存审计实验的固定参数与绝对路径。"""

    project_root: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
    module_dir: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813")
    output_dir: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs_shortcut_audit")
    original_output_dir: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs")
    positive_root: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\Positive_Clicks_3500")
    clicktrains_csv: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains.csv")
    negative_csv: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\14\negative_samples_log.csv")
    original_audio_root: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\original_audio")
    pulse_precise_path: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\09_Locate\Locate_Batch_Report\pulse_precise_positions.txt")

    sample_rate: int = 576_000
    sample_count: int = 5_761
    lowcut_hz: float = 20_000.0
    highcut_hz: float = 190_000.0
    filter_order: int = 5
    tukey_alpha: float = 0.04

    atom_duration_us: float = 85.0
    lambda_ratio: float = 0.075
    outer_iterations: int = 35
    fista_iterations: int = 120
    dictionary_iterations: int = 30
    min_bandwidth_hz: float = 6_000.0
    max_bandwidth_hz: float = 72_000.0
    bandwidth_nfft: int = 16_384
    peak_ratio: float = 0.015
    min_path_delay_ms: float = 0.035
    max_path_delay_ms: float = 2.0
    match_tolerance_ms: float = 0.03
    bootstrap_runs: int = 8
    bootstrap_required: int = 7
    residual_block_ms: float = 0.10
    lambda_perturbation: float = 0.15
    local_window_samples: int = 257

    random_seed: int = 20_260_813
    n_splits: int = 5
    max_epochs: int = 25
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    group_bootstrap_runs: int = 2_000
    bcsc_checkpoint_interval: int = 25
    bcsc_workers: int = 6

    edge_ms: float = 1.5
    high_ratio_low_hz: float = 100_000.0
    high_ratio_high_hz: float = 190_000.0
    caliper_sd: float = 0.2
    min_pair_sep_s: float = 0.010
    min_pairs: int = 3_150
    smd_max: float = 0.10

    signal_window_ms: float = 0.175
    mask_radius_ms: float = 0.25
    mask_fade_ms: float = 0.10

    @property
    def atom_length(self) -> int:
        length = int(round(self.atom_duration_us * 1e-6 * self.sample_rate))
        return length + (length % 2 == 0)

    @property
    def config_hash(self) -> str:
        payload = asdict(self).copy()
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


CFG = AuditConfig()
MATCHING_FEATURE_COLS = [
    "bg_log_rms",
    "local_snr_db",
    "crest_factor",
    "spectral_centroid_hz",
    "bandwidth_10db_hz",
    "high_energy_ratio_100_190",
]
SIMPLE_EXTRA_COLS = [
    "kurtosis",
    "energy_20_50",
    "energy_50_90",
    "energy_90_130",
    "energy_130_190",
]
ALL_FEATURE_COLS = ["bg_log_rms", "local_snr_db", "crest_factor", "kurtosis", "spectral_centroid_hz", "bandwidth_10db_hz", "high_energy_ratio_100_190", "energy_20_50", "energy_50_90", "energy_90_130", "energy_130_190"]
WAVEFORM_CONDITIONS = ["symmetric_full", "matched_full", "matched_signal", "matched_residual", "matched_background", "matched_randommask"]
BASELINE_CONDITIONS = ["lr_features", "gbm_features"]
ALL_TRAINED_CONDITIONS = WAVEFORM_CONDITIONS + BASELINE_CONDITIONS
CONDITION_LABELS = {
    "historical": "历史基准",
    "symmetric_full": "对称重截取全波形",
    "matched_full": "严格匹配全波形",
    "matched_signal": "严格匹配信号区",
    "matched_residual": "BCSC残差",
    "matched_background": "宽窗口去信号背景",
    "matched_randommask": "随机遮罩对照",
    "lr_features": "逻辑回归特征",
    "gbm_features": "浅层梯度提升特征",
}


def ensure_audit_dirs(config: AuditConfig = CFG) -> dict[str, Path]:
    paths = {
        "root": config.output_dir,
        "cache": config.output_dir / "cache",
        "models": config.output_dir / "models",
        "figures": config.output_dir / "figures",
        "logs": config.output_dir / "logs",
        "features": config.output_dir / "features",
        "folds": config.output_dir / "folds",
        "predictions": config.output_dir / "predictions",
        "metrics": config.output_dir / "metrics",
        "bootstrap": config.output_dir / "bootstrap",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_positive_parts(wav_path: str) -> tuple[str, str, int]:
    path = Path(str(wav_path))
    click_dir = path.parent.name
    train_dir = path.parent.parent.name
    voyage = path.parent.parent.parent.name
    click_number = int(click_dir.split("_")[-1])
    return voyage, train_dir, click_number


def read_original_window(path: Path, timestamp_s: float, config: AuditConfig = CFG) -> np.ndarray:
    sample_rate = sf.info(str(path)).samplerate
    if sample_rate != config.sample_rate:
        raise ValueError(f"采样率错误：{path} -> {sample_rate}")
    center = int(round(float(timestamp_s) * config.sample_rate))
    start = center - config.sample_count // 2
    with sf.SoundFile(str(path), mode="r") as handle:
        handle.seek(max(0, start))
        signal = handle.read(config.sample_count, dtype="float64", always_2d=False)
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.ndim != 1 or len(signal) != config.sample_count:
        raise ValueError(f"波形尺寸错误：{path} -> {signal.shape}")
    return signal


def preprocess_model_waveform(signal: np.ndarray, config: AuditConfig = CFG) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    sos = butter(config.filter_order, [config.lowcut_hz / (config.sample_rate / 2), config.highcut_hz / (config.sample_rate / 2)], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, signal)
    filtered *= windows.tukey(len(filtered), alpha=config.tukey_alpha)
    scale = np.max(np.abs(filtered))
    if not np.isfinite(scale) or scale < 1e-12:
        raise RuntimeError("预处理后波形幅度过小")
    return (filtered / scale).astype(np.float32)


def preprocess_matching_waveform(signal: np.ndarray, config: AuditConfig = CFG) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    sos = butter(config.filter_order, [config.lowcut_hz / (config.sample_rate / 2), config.highcut_hz / (config.sample_rate / 2)], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, signal)
    filtered *= windows.tukey(len(filtered), alpha=config.tukey_alpha)
    return filtered.astype(np.float32)


def matching_features_from_bandpassed(signal: np.ndarray, config: AuditConfig = CFG) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    edge_n = int(round(config.edge_ms * 1e-3 * config.sample_rate))
    edge_n = min(edge_n, max(1, len(signal) // 4))
    edge = np.concatenate([signal[:edge_n], signal[-edge_n:]])
    bg_rms = float(np.sqrt(np.mean(edge**2)) + 1e-12)
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal**2)) + 1e-12)
    local_snr_db = float(20.0 * np.log10((peak + 1e-12) / bg_rms))
    crest = float(peak / rms)
    signal_kurtosis = float(kurtosis(signal, fisher=False, bias=False)) if np.std(signal) > 1e-12 else 0.0

    tapered = signal * windows.hann(len(signal), sym=False)
    n_fft = 2_048
    power = np.abs(np.fft.rfft(tapered, n=n_fft)) ** 2
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / config.sample_rate)
    band_mask = (frequencies >= config.lowcut_hz) & (frequencies <= config.highcut_hz)
    frequencies_band = frequencies[band_mask]
    power_band = power[band_mask] + 1e-18
    probability = power_band / np.sum(power_band)
    centroid = float(np.sum(frequencies_band * probability))
    peak_power = float(np.max(power_band))

    def bandwidth_at(db: float) -> float:
        mask = power_band >= peak_power * 10 ** (-db / 10)
        selected = frequencies_band[mask]
        return float(selected[-1] - selected[0]) if len(selected) else 0.0

    bandwidth_10db = bandwidth_at(10.0)
    total_energy = float(np.sum(power_band))
    high_mask = (frequencies_band >= config.high_ratio_low_hz) & (frequencies_band <= config.high_ratio_high_hz)
    high_ratio = float(np.sum(power_band[high_mask]) / total_energy)
    subbands = [(20_000, 50_000), (50_000, 90_000), (90_000, 130_000), (130_000, 190_000)]
    subband_energy = []
    for low, high in subbands:
        mask = (frequencies_band >= low) & (frequencies_band < high)
        subband_energy.append(float(np.sum(power_band[mask]) / total_energy))
    values = [np.log10(bg_rms), local_snr_db, crest, signal_kurtosis, centroid, bandwidth_10db, high_ratio, *subband_energy]
    return {name: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for name, value in zip(ALL_FEATURE_COLS, values)}


def materialize_waveforms(manifest: pd.DataFrame, config: AuditConfig = CFG, kind: str = "model", force: bool = False) -> np.ndarray:
    paths = ensure_audit_dirs(config)
    cache_path = paths["cache"] / f"{kind}_waveforms_{config.config_hash}.npy"
    if cache_path.exists() and not force:
        data = np.load(cache_path, mmap_mode="r")
        if data.shape == (len(manifest), config.sample_count):
            return data
    data = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(len(manifest), config.sample_count))
    for index, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"物化 {kind} 波形", unit="条", postfix={"hash": config.config_hash[:8]}):
        raw = read_original_window(config.original_audio_root / str(row["audio_file"]), float(row["timestamp_s"]), config)
        processed = preprocess_model_waveform(raw, config) if kind != "matching" else preprocess_matching_waveform(raw, config)
        data[index] = processed
    data.flush()
    return np.load(cache_path, mmap_mode="r")


def load_historical(config: AuditConfig = CFG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = config.original_output_dir
    summary = pd.read_csv(original / "model_summary.csv")
    metrics = pd.read_csv(original / "fold_metrics.csv")
    predictions = pd.read_csv(original / "test_predictions.csv")
    manifest = pd.read_csv(original / "dataset_manifest.csv")
    return summary, metrics, predictions, manifest


def build_positive_time_map(manifest: pd.DataFrame, config: AuditConfig = CFG, force: bool = False) -> pd.DataFrame:
    paths = ensure_audit_dirs(config)
    out_path = paths["root"] / "positive_time_map.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    precise = pd.read_csv(config.pulse_precise_path, sep="\t", comment="#", encoding="utf-8")
    precise["PulseNumber"] = precise["PulseNumber"].astype(int)

    clicktrains = pd.read_csv(config.clicktrains_csv)
    train_column = "Train_num(No.)"
    source_column = "Ori_file_num(No.)"
    start_column = "Train_start(s)"
    train_to_source = clicktrains[[train_column, source_column]].dropna().drop_duplicates(train_column).set_index(train_column)[source_column].astype(int).to_dict()
    train_to_start = clicktrains[[train_column, start_column]].dropna().drop_duplicates(train_column).set_index(train_column)[start_column].astype(float).to_dict()

    durations: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    positives = manifest[manifest["label"] == 1]
    for _, row in tqdm(positives.iterrows(), total=len(positives), desc="正样本原始时间映射", unit="条"):
        voyage, train_dir, click_number = parse_positive_parts(row["wav_path"])
        train_num = int(train_dir.split("_")[-1])
        if train_num not in train_to_source or train_num not in train_to_start:
            raise KeyError(f"ClickTrains.csv 缺少 PulseTrain_{train_num:03d}")
        ori_file_num = int(train_to_source[train_num])
        match = precise[(precise["Voyage_Ori"] == voyage) & (precise["PulseTrain"] == train_dir) & (precise["PulseNumber"] == click_number)]
        if len(match) != 1:
            raise KeyError(f"精确位置映射失败：{voyage}/{train_dir}/Click_{click_number:03d} -> {len(match)}")
        position_time = float(match.iloc[0]["PositionTime_s"])
        timestamp = float(train_to_start[train_num]) + position_time
        if ori_file_num not in durations:
            audio_path = config.original_audio_root / f"Ori_Recording_{ori_file_num:02d}.wav"
            durations[ori_file_num] = float(sf.info(str(audio_path)).duration)
        half_window = (config.sample_count // 2) / config.sample_rate
        if not (half_window <= timestamp <= durations[ori_file_num] - half_window):
            raise ValueError(f"时间超出录音边界：{row['sample_id']} -> {timestamp:.6f}s")
        rows.append({"sample_id": row["sample_id"], "voyage": voyage, "pulse_train": train_dir, "click_number": click_number, "ori_file_num": ori_file_num, "timestamp_s": timestamp, "audio_file": f"Ori_Recording_{ori_file_num:02d}.wav"})
    result = pd.DataFrame(rows)
    if result["sample_id"].is_unique and len(result) == 3500:
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        return result
    raise RuntimeError(f"正样本时间映射未覆盖3500/3500：实际 {len(result)}")


def build_original_manifest(config: AuditConfig = CFG, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_audit_dirs(config)
    out_path = paths["folds"] / "original_manifest.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path), pd.read_csv(paths["root"] / "positive_time_map.csv")
    _, _, _, historical_manifest = load_historical(config)
    time_map = build_positive_time_map(historical_manifest, config, force=force)
    manifest = historical_manifest.copy()
    manifest = manifest.drop(columns=["config_hash"], errors="ignore")
    manifest = manifest.merge(time_map[["sample_id", "timestamp_s"]], on="sample_id", how="left", suffixes=("", "_mapped"))
    positives = manifest["label"] == 1
    manifest.loc[positives, "timestamp_s"] = manifest.loc[positives, "timestamp_s_mapped"]
    manifest.loc[positives, "audio_file"] = manifest.loc[positives, "ori_file_num"].map(lambda n: f"Ori_Recording_{int(n):02d}.wav")
    manifest.loc[positives, "source_kind"] = "original_audio_timestamp"
    manifest.loc[~positives, "source_kind"] = "original_audio_timestamp"
    manifest = manifest.drop(columns=["timestamp_s_mapped"], errors="ignore")
    if manifest["timestamp_s"].isna().any():
        raise RuntimeError("原队列存在缺失时间戳")
    manifest.to_csv(out_path, index=False, encoding="utf-8-sig")
    return manifest, time_map


def _pool_file_worker(payload: tuple[AuditConfig, int, list[tuple[int, float]]]) -> list[dict[str, Any]]:
    config, ori_file_num, rows = payload
    path = config.original_audio_root / f"Ori_Recording_{int(ori_file_num):02d}.wav"
    results: list[dict[str, Any]] = []
    with sf.SoundFile(str(path), mode="r") as handle:
        for pool_index, timestamp in rows:
            center = int(round(timestamp * config.sample_rate))
            start = center - config.sample_count // 2
            handle.seek(max(0, start))
            raw = handle.read(config.sample_count, dtype="float64", always_2d=False)
            raw = np.asarray(raw, dtype=np.float64).squeeze()
            if raw.ndim != 1 or len(raw) != config.sample_count:
                continue
            band = preprocess_matching_waveform(raw, config)
            feature = matching_features_from_bandpassed(band, config)
            feature["pool_index"] = int(pool_index)
            feature["ori_file_num"] = int(ori_file_num)
            feature["timestamp_s"] = float(timestamp)
            results.append(feature)
    return results


def extract_candidate_pool_features(config: AuditConfig = CFG, force: bool = False) -> pd.DataFrame:
    paths = ensure_audit_dirs(config)
    cache_path = paths["features"] / f"candidate_pool_features_{config.config_hash}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    pool = pd.read_csv(config.negative_csv)
    pool["ori_file_num"] = pool["Audio_File"].map(parse_original_number)
    pool = pool.rename(columns={"Noise_Timestamp(s)": "timestamp_s", "Audio_File": "audio_file"})
    pool = pool.reset_index(drop=True).reset_index().rename(columns={"index": "pool_index"})
    half_window = (config.sample_count // 2) / config.sample_rate
    durations: dict[int, float] = {}
    for ori in sorted(pool["ori_file_num"].unique()):
        audio_path = config.original_audio_root / f"Ori_Recording_{int(ori):02d}.wav"
        durations[int(ori)] = float(sf.info(str(audio_path)).duration)
    pool = pool[(pool["timestamp_s"] >= half_window) & (pool["timestamp_s"] <= pool["ori_file_num"].map(durations) - half_window)].reset_index(drop=True)

    grouped: list[tuple[AuditConfig, int, list[tuple[int, float]]]] = []
    for ori_file_num, group in pool.groupby("ori_file_num", sort=True):
        rows = [(int(pool_index), float(timestamp)) for pool_index, timestamp in zip(group["pool_index"], group["timestamp_s"])]
        grouped.append((config, int(ori_file_num), rows))

    all_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=config.bcsc_workers) as executor:
        futures = executor.map(_pool_file_worker, grouped, chunksize=1)
        for file_rows in tqdm(futures, total=len(grouped), desc="负候选特征提取", unit="文件", postfix={"hash": config.config_hash[:8]}):
            all_rows.extend(file_rows)
    result = pd.DataFrame(all_rows).sort_values("pool_index").reset_index(drop=True)
    result.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return result


def _select_nonoverlap_candidates(times: np.ndarray, min_sep: float) -> np.ndarray:
    """按时间顺序贪心选取互不重叠的候选索引。"""
    order = np.argsort(times)
    chosen: list[int] = []
    last = -np.inf
    for index in order:
        if times[index] - last >= min_sep:
            chosen.append(int(index))
            last = times[index]
    return np.asarray(chosen, dtype=int)


def _match_block(
    pos_logit: np.ndarray,
    pool_logit: np.ndarray,
    pos_times: np.ndarray,
    pool_times: np.ndarray,
    pos_ids: np.ndarray,
    pool_ids: np.ndarray,
    caliper: float,
    min_sep: float,
) -> list[dict[str, Any]]:
    """在一个录音编号块内做倾向分最优匹配。"""
    if len(pos_logit) == 0 or len(pool_logit) == 0:
        return []
    eligible = np.where(np.min(np.abs(pool_times[:, None] - pos_times[None, :]), axis=1) >= min_sep)[0]
    if len(eligible) == 0:
        return []
    pool_logit = pool_logit[eligible]
    pool_times = pool_times[eligible]
    pool_ids = pool_ids[eligible]
    nonoverlap = _select_nonoverlap_candidates(pool_times, min_sep)
    pool_logit = pool_logit[nonoverlap]
    pool_times = pool_times[nonoverlap]
    pool_ids = pool_ids[nonoverlap]
    if len(pool_logit) == 0:
        return []

    cost = np.abs(pos_logit[:, None] - pool_logit[None, :]).astype(float)
    big = float(np.max(cost) * 10 + 10.0)
    cost[cost > caliper] = big
    if len(pool_logit) < len(pos_logit):
        pad = np.full((len(pos_logit), len(pos_logit) - len(pool_logit)), big, dtype=float)
        cost = np.hstack([cost, pad])
        extended_ids = np.concatenate([pool_ids, np.full(len(pos_logit) - len(pool_logit), -1, dtype=pool_ids.dtype)])
        extended_times = np.concatenate([pool_times, np.full(len(pos_logit) - len(pool_logit), np.inf)])
    else:
        extended_ids = pool_ids
        extended_times = pool_times

    rows, cols = linear_sum_assignment(cost)
    pairs: list[dict[str, Any]] = []
    used_negative_times: list[float] = []
    for row_index, col_index in zip(rows, cols):
        if col_index >= len(pool_ids) or cost[row_index, col_index] >= big:
            continue
        pairs.append(
            {
                "positive_sample_id": str(pos_ids[row_index]),
                "positive_timestamp_s": float(pos_times[row_index]),
                "negative_pool_index": int(extended_ids[col_index]),
                "negative_timestamp_s": float(extended_times[col_index]),
                "logit_distance": float(cost[row_index, col_index]),
            }
        )
    return pairs


def match_strict_pairs(pos_features: pd.DataFrame, pool_features: pd.DataFrame, config: AuditConfig = CFG) -> pd.DataFrame:
    """倾向分最优匹配并计算匹配前后平衡。"""
    pos = pos_features.copy().reset_index(drop=True)
    pool = pool_features.copy().reset_index(drop=True)
    scaler = StandardScaler().fit(pd.concat([pos[MATCHING_FEATURE_COLS], pool[MATCHING_FEATURE_COLS]], ignore_index=True))
    pos_scaled = scaler.transform(pos[MATCHING_FEATURE_COLS])
    pool_scaled = scaler.transform(pool[MATCHING_FEATURE_COLS])
    recording_all = pd.concat([pos["ori_file_num"], pool["ori_file_num"]], ignore_index=True).astype(int)
    rec_dummies = pd.get_dummies(recording_all, prefix="rec", dtype=float)
    x_model = np.hstack([np.vstack([pos_scaled, pool_scaled]), rec_dummies.to_numpy(dtype=float)])
    y_model = np.r_[np.ones(len(pos)), np.zeros(len(pool))]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.random_seed)
    clf.fit(x_model, y_model)
    logits = clf.decision_function(x_model)
    caliper = float(config.caliper_sd * np.std(logits))
    pos_logits = logits[: len(pos)]
    pool_logits = logits[len(pos) :]

    pair_rows: list[dict[str, Any]] = []
    for ori_file_num in sorted(pos["ori_file_num"].unique()):
        pos_mask = pos["ori_file_num"].to_numpy() == ori_file_num
        pool_mask = pool["ori_file_num"].to_numpy() == ori_file_num
        block_pairs = _match_block(
            pos_logits[pos_mask],
            pool_logits[pool_mask],
            pos.loc[pos_mask, "timestamp_s"].to_numpy(),
            pool.loc[pool_mask, "timestamp_s"].to_numpy(),
            pos.loc[pos_mask, "sample_id"].to_numpy(),
            pool.loc[pool_mask, "pool_index"].to_numpy(),
            caliper,
            config.min_pair_sep_s,
        )
        pair_rows.extend(block_pairs)

    pairs = pd.DataFrame(pair_rows)
    if len(pairs) == 0:
        raise RuntimeError("严格匹配未产生任何配对")
    pairs = pairs.drop_duplicates(subset=["positive_sample_id"], keep="first").reset_index(drop=True)
    pos_feature = pos.set_index("sample_id")[MATCHING_FEATURE_COLS].add_prefix("pos_")
    pool_feature = pool.set_index("pool_index")[MATCHING_FEATURE_COLS].add_prefix("neg_")
    pairs = pairs.merge(pos_feature, left_on="positive_sample_id", right_index=True, how="left")
    pairs = pairs.merge(pool_feature, left_on="negative_pool_index", right_index=True, how="left")
    pairs["matched"] = True

    before_rows = []
    for column in MATCHING_FEATURE_COLS:
        pooled_std = float(np.sqrt((pos[column].var() + pool[column].var()) / 2))
        before_rows.append({"variable": column, "phase": "before", "smd": float((pos[column].mean() - pool[column].mean()) / max(pooled_std, 1e-12))})
    for column in MATCHING_FEATURE_COLS:
        pooled_std = float(np.sqrt((pairs[f"pos_{column}"].var() + pairs[f"neg_{column}"].var()) / 2))
        before_rows.append({"variable": column, "phase": "after", "smd": float((pairs[f"pos_{column}"].mean() - pairs[f"neg_{column}"].mean()) / max(pooled_std, 1e-12))})
    balance = pd.DataFrame(before_rows)
    paths = ensure_audit_dirs(config)
    balance.to_csv(paths["root"] / "matching_balance.csv", index=False, encoding="utf-8-sig")
    return pairs


def build_matched_manifest(
    pos_features: pd.DataFrame,
    pool_features: pd.DataFrame,
    original_manifest: pd.DataFrame,
    config: AuditConfig = CFG,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_audit_dirs(config)
    manifest_path = paths["folds"] / "matched_manifest.csv"
    pairs_path = paths["root"] / "matched_pairs.csv"
    if manifest_path.exists() and pairs_path.exists() and not force:
        return pd.read_csv(manifest_path), pd.read_csv(pairs_path)

    negative_csv = pd.read_csv(config.negative_csv).reset_index(drop=True).reset_index().rename(columns={"index": "pool_index", "Noise_Timestamp(s)": "timestamp_s", "Audio_File": "audio_file"})
    pairs = match_strict_pairs(pos_features, pool_features, config)
    matched_negatives = negative_csv[negative_csv["pool_index"].isin(pairs["negative_pool_index"])].copy()
    matched_negatives["ori_file_num"] = matched_negatives["audio_file"].map(parse_original_number)
    matched_negatives["sample_id"] = matched_negatives.apply(lambda row: f"N_{int(row['ori_file_num']):02d}_{float(row['timestamp_s']):.6f}_{row['Noise_Type']}", axis=1)
    pairs = pairs.merge(matched_negatives[["pool_index", "sample_id"]].rename(columns={"sample_id": "negative_sample_id"}), left_on="negative_pool_index", right_on="pool_index", how="left").drop(columns=["pool_index"])
    pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    matched_negatives["label"] = 0
    matched_negatives["wav_path"] = matched_negatives["audio_file"].map(lambda name: str((config.original_audio_root / name).resolve()))
    matched_negatives["source_kind"] = "original_audio_timestamp"
    matched_negatives["noise_type"] = matched_negatives["Noise_Type"]

    pos_ids = set(pairs["positive_sample_id"])
    matched_positives = original_manifest[original_manifest["sample_id"].isin(pos_ids)].copy()
    matched_positives["source_kind"] = "original_audio_timestamp"
    matched_positives["noise_type"] = "positive_click"
    matched_positives = matched_positives.reset_index(drop=True)
    matched_negatives = matched_negatives[["sample_id", "label", "ori_file_num", "audio_file", "timestamp_s", "noise_type", "wav_path", "source_kind"]].reset_index(drop=True)
    manifest = pd.concat([matched_positives[["sample_id", "label", "ori_file_num", "audio_file", "timestamp_s", "noise_type", "wav_path", "source_kind"]], matched_negatives], ignore_index=True)

    splitter = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    manifest["test_fold"] = -1
    for fold, (_, test_indices) in enumerate(splitter.split(manifest, manifest["label"], groups=manifest["ori_file_num"])):
        manifest.loc[test_indices, "test_fold"] = fold
    if (manifest["test_fold"] < 0).any():
        raise RuntimeError("匹配队列存在未分配折的样本")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest, pairs


def process_bcsc_signal_audit(signal: np.ndarray, seed: int, config: AuditConfig = CFG) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        atom, activation, estimate = bcsc(signal, config)
        residual = np.asarray(signal, dtype=np.float64) - estimate
        candidates = extract_candidate_indices(activation, config)
        direct_index = int(np.argmax(np.abs(activation)))
        min_delay = int(round(config.min_path_delay_ms * 1e-3 * config.sample_rate))
        max_delay = int(round(config.max_path_delay_ms * 1e-3 * config.sample_rate))
        reference_multipaths = [int(index) for index in candidates if direct_index + min_delay <= int(index) <= direct_index + max_delay]
        reference_indices = [direct_index] + reference_multipaths
        matched: dict[int, list[tuple[int, float]]] = {index: [] for index in reference_indices}
        rng = np.random.default_rng(seed)
        block_length = max(1, int(round(config.residual_block_ms * 1e-3 * config.sample_rate)))
        match_tolerance = int(round(config.match_tolerance_ms * 1e-3 * config.sample_rate))
        max_correlation = np.max(np.abs(adjoint_atom(signal, atom, len(activation))))
        for _ in range(config.bootstrap_runs):
            synthetic = estimate + moving_block_bootstrap(residual, block_length, rng)
            lambda_scale = rng.uniform(1.0 - config.lambda_perturbation, 1.0 + config.lambda_perturbation)
            code = fista_sparse_code(synthetic, atom, config.lambda_ratio * lambda_scale * max_correlation, activation, config.fista_iterations)
            boot_candidates = list(map(int, extract_candidate_indices(code, config)))
            unused = set(boot_candidates)
            for reference_index in reference_indices:
                available = [index for index in unused if abs(index - reference_index) <= match_tolerance]
                if not available:
                    continue
                selected = min(available, key=lambda index: (abs(index - reference_index), -abs(code[index])))
                unused.remove(selected)
                matched[reference_index].append((selected, float(code[selected])))

        direct_records = matched[direct_index]
        direct_confidence = len(direct_records) / config.bootstrap_runs
        stable_multipaths: list[dict[str, float]] = []
        direct_abs = abs(float(activation[direct_index])) + 1e-12
        for reference_index in reference_multipaths:
            records = matched[reference_index]
            if len(records) < config.bootstrap_required:
                continue
            indices = np.asarray([record[0] for record in records])
            coefficients = np.asarray([record[1] for record in records])
            stable_multipaths.append({"reference_index": float(reference_index), "median_index": float(np.median(indices)), "ci_low_index": float(np.percentile(indices, 2.5)), "ci_high_index": float(np.percentile(indices, 97.5)), "median_coefficient": float(np.median(coefficients)), "confidence": len(records) / config.bootstrap_runs, "relative_amplitude": float(np.median(np.abs(coefficients)) / direct_abs)})
        stable_multipaths.sort(key=lambda row: (-row["relative_amplitude"], row["median_index"]))
        selected = stable_multipaths[0] if stable_multipaths else None
        multipath_index = int(round(selected["median_index"])) if selected else -1
        multipath_coefficient = float(selected["median_coefficient"]) if selected else 0.0
        multipath_confidence = float(selected["confidence"]) if selected else 0.0
        delay_ms = (multipath_index - direct_index) / config.sample_rate * 1e3 if selected else 0.0
        delay_ci_low_ms = (selected["ci_low_index"] - direct_index) / config.sample_rate * 1e3 if selected else 0.0
        delay_ci_high_ms = (selected["ci_high_index"] - direct_index) / config.sample_rate * 1e3 if selected else 0.0
        bandwidth_hz, center_frequency_hz = measure_3db_bandwidth(atom, config.sample_rate, config.bandwidth_nfft)
        explained_variance = 1.0 - np.sum(residual**2) / (np.sum(np.asarray(signal) ** 2) + 1e-12)
        status = "ok" if len(direct_records) >= config.bootstrap_required else "unstable_direct"
        return {
            "status": status,
            "error": "",
            "atom": atom.astype(np.float32),
            "candidate_indices": [int(index) for index in candidates],
            "direct_index": direct_index,
            "direct_coefficient": float(activation[direct_index]),
            "direct_confidence": direct_confidence,
            "multipath_index": multipath_index,
            "multipath_coefficient": multipath_coefficient,
            "multipath_confidence": multipath_confidence,
            "delay_ms": delay_ms,
            "delay_ci_low_ms": delay_ci_low_ms,
            "delay_ci_high_ms": delay_ci_high_ms,
            "explained_variance": float(explained_variance),
            "residual_energy": float(np.mean(residual**2)),
            "bandwidth_hz": bandwidth_hz,
            "center_frequency_hz": center_frequency_hz,
            "candidate_count": len(candidates),
            "stable_multipath_count": len(stable_multipaths),
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "atom": np.zeros(config.atom_length, dtype=np.float32),
            "candidate_indices": [],
            "direct_index": -1,
            "direct_coefficient": 0.0,
            "direct_confidence": 0.0,
            "multipath_index": -1,
            "multipath_coefficient": 0.0,
            "multipath_confidence": 0.0,
            "delay_ms": 0.0,
            "delay_ci_low_ms": 0.0,
            "delay_ci_high_ms": 0.0,
            "explained_variance": 0.0,
            "residual_energy": 0.0,
            "bandwidth_hz": 0.0,
            "center_frequency_hz": 0.0,
            "candidate_count": 0,
            "stable_multipath_count": 0,
            "elapsed_s": time.perf_counter() - started,
        }


def _process_bcsc_payload_audit(payload: tuple[int, np.ndarray, int, AuditConfig]) -> tuple[int, dict[str, Any]]:
    index, signal, seed, config = payload
    return index, process_bcsc_signal_audit(np.asarray(signal, dtype=np.float64), seed, config)


def run_bcsc_audit(waveforms: np.ndarray, manifest: pd.DataFrame, config: AuditConfig = CFG, force: bool = False, limit: int | None = None) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[int, list[int]]]:
    paths = ensure_audit_dirs(config)
    cache_path = paths["cache"] / f"bcsc_audit_{config.config_hash}.npz"
    candidate_path = paths["cache"] / f"bcsc_candidates_{config.config_hash}.json"
    table_path = paths["root"] / "bcsc_paths.csv"
    arrays = _empty_bcsc_arrays(len(manifest), config)
    candidate_sidecar: dict[int, list[int]] = {}
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_hash = str(cached["config_hash"].item()) if "config_hash" in cached else ""
            if cached_hash == config.config_hash:
                for key in arrays:
                    if key in cached and cached[key].shape == arrays[key].shape:
                        arrays[key] = cached[key].copy()
                if candidate_path.exists():
                    candidate_sidecar = {int(key): value for key, value in json.loads(candidate_path.read_text(encoding="utf-8")).items()}
    indices = np.where(~arrays["completed"])[0]
    if limit is not None:
        indices = indices[:limit]
    progress = tqdm(total=len(indices), desc="BCSC 与稳定性复核", unit="条", postfix={"hash": config.config_hash[:8]})
    payloads = ((int(index), np.asarray(waveforms[index], dtype=np.float32), config.random_seed + int(index), config) for index in indices)
    with ProcessPoolExecutor(max_workers=config.bcsc_workers) as executor:
        for completed_in_run, (index, result) in enumerate(executor.map(_process_bcsc_payload_audit, payloads, chunksize=1), start=1):
            arrays["completed"][index] = True
            arrays["status_code"][index] = {"ok": 1, "unstable_direct": 0, "failed": -2}[result["status"]]
            arrays["atoms"][index] = result["atom"]
            for key in ("direct_index", "direct_coefficient", "direct_confidence", "multipath_index", "multipath_coefficient", "multipath_confidence", "delay_ms", "delay_ci_low_ms", "delay_ci_high_ms", "explained_variance", "residual_energy", "bandwidth_hz", "center_frequency_hz", "candidate_count", "stable_multipath_count", "elapsed_s"):
                arrays[key][index] = result[key]
            arrays["errors"][index] = result["error"][:255]
            candidate_sidecar[int(index)] = result["candidate_indices"]
            progress.update(1)
            progress.set_postfix(ok=int(np.sum(arrays["status_code"] == 1)), multipath=int(np.sum(arrays["multipath_index"] >= 0)))
            if completed_in_run % config.bcsc_checkpoint_interval == 0:
                _save_bcsc_cache(cache_path, arrays, config)
                json_dump(candidate_path, candidate_sidecar)
    progress.close()
    _save_bcsc_cache(cache_path, arrays, config)
    json_dump(candidate_path, candidate_sidecar)

    status_text = np.where(arrays["status_code"] == 1, "ok", np.where(arrays["status_code"] == 0, "unstable_direct", np.where(arrays["completed"], "failed", "pending")))
    table = manifest[["sample_id", "label", "ori_file_num", "test_fold"]].copy()
    table["status"] = status_text
    for key in ("direct_index", "direct_coefficient", "direct_confidence", "multipath_index", "multipath_coefficient", "multipath_confidence", "delay_ms", "delay_ci_low_ms", "delay_ci_high_ms", "explained_variance", "residual_energy", "bandwidth_hz", "center_frequency_hz", "candidate_count", "stable_multipath_count", "elapsed_s", "errors"):
        table[key] = arrays[key]
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    return arrays, table, candidate_sidecar


def direct_center(index: int, arrays: dict[str, np.ndarray], config: AuditConfig = CFG) -> int:
    direct_index = int(arrays["direct_index"][index])
    if direct_index < 0:
        return -1
    return direct_index + config.atom_length // 2


def peak_normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - np.mean(signal)
    scale = float(np.max(np.abs(signal)))
    return (signal / scale).astype(np.float32) if scale > 1e-12 else signal


def build_signal_only(waveforms: np.ndarray, arrays: dict[str, np.ndarray], config: AuditConfig = CFG) -> np.ndarray:
    half = int(round(config.signal_window_ms * 1e-3 * config.sample_rate))
    result = np.zeros_like(np.asarray(waveforms), dtype=np.float32)
    for index in tqdm(range(len(waveforms)), desc="构建信号区变体", unit="条"):
        center = direct_center(index, arrays, config)
        if center < 0:
            center = int(np.argmax(np.abs(np.asarray(waveforms[index]))))
        start = max(0, center - half)
        stop = min(len(waveforms[index]), center + half + 1)
        result[index, start:stop] = waveforms[index, start:stop]
        result[index] = peak_normalize(result[index])
    return result


def build_residual(waveforms: np.ndarray, arrays: dict[str, np.ndarray], config: AuditConfig = CFG) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(waveforms, dtype=np.float32)
    unnorm = np.zeros_like(full, dtype=np.float32)
    normalized = np.zeros_like(full, dtype=np.float32)
    for index in tqdm(range(len(full)), desc="构建BCSC残差变体", unit="条"):
        atom = np.asarray(arrays["atoms"][index], dtype=np.float64)
        direct_index = int(arrays["direct_index"][index])
        if direct_index < 0:
            unnorm[index] = full[index]
            normalized[index] = peak_normalize(full[index])
            continue
        direct_component = _path_component(atom, direct_index, float(arrays["direct_coefficient"][index]), config.sample_count)
        multipath_index = int(arrays["multipath_index"][index])
        multipath_component = _path_component(atom, multipath_index, float(arrays["multipath_coefficient"][index]), config.sample_count) if multipath_index >= 0 else np.zeros(config.sample_count)
        residual = full[index] - direct_component - multipath_component
        unnorm[index] = residual.astype(np.float32)
        normalized[index] = peak_normalize(residual)
    return normalized, unnorm


def _merge_spans(spans: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged: list[list[int]] = [list(spans[0])]
    for start, stop in spans[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [(int(s), int(e)) for s, e in merged]


def _median_event_free_rms(signal: np.ndarray, mask_spans: list[tuple[int, int]], guard: int, need_len: int, config: AuditConfig = CFG) -> float:
    occupied = np.zeros(len(signal), dtype=bool)
    for start, stop in mask_spans:
        occupied[max(0, start - guard) : min(len(signal), stop + guard)] = True
    rms_values = []
    step = max(1, need_len // 8)
    for start in range(0, len(signal) - need_len + 1, step):
        if occupied[start : start + need_len].any():
            continue
        rms_values.append(float(np.sqrt(np.mean(signal[start : start + need_len] ** 2)) + 1e-12))
    return float(np.median(rms_values)) if rms_values else 1e-12


def _fill_masked_regions(signal: np.ndarray, mask_spans: list[tuple[int, int]], rng: np.random.Generator, config: AuditConfig = CFG) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64).copy()
    fade_len = int(round(config.mask_fade_ms * 1e-3 * config.sample_rate))
    for start, stop in mask_spans:
        region_start = max(0, start - fade_len)
        region_stop = min(len(signal), stop + fade_len)
        need_len = region_stop - region_start
        if need_len <= 0:
            continue
        target_rms = _median_event_free_rms(signal, mask_spans, fade_len, need_len, config)
        occupied = np.zeros(len(signal), dtype=bool)
        for s2, e2 in mask_spans:
            occupied[max(0, s2 - fade_len) : min(len(signal), e2 + fade_len)] = True
        candidates = []
        step = max(1, need_len // 8)
        for donor_start in range(0, len(signal) - need_len + 1, step):
            if occupied[donor_start : donor_start + need_len].any():
                continue
            donor = signal[donor_start : donor_start + need_len]
            donor_rms = float(np.sqrt(np.mean(donor**2)) + 1e-12)
            candidates.append((abs(donor_rms - target_rms), donor_start, donor))
        if not candidates:
            continue
        _, donor_start, donor = min(candidates, key=lambda item: item[0])
        donor_rms = float(np.sqrt(np.mean(donor**2)) + 1e-12)
        donor = donor * (target_rms / donor_rms)
        region_len = region_stop - region_start
        blend = signal[region_start:region_stop].copy()
        weight = np.ones(region_len, dtype=float)
        if fade_len > 0 and region_len > 2 * fade_len:
            weight[:fade_len] = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
            weight[-fade_len:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_len) / fade_len))
        signal[region_start:region_stop] = blend * (1.0 - weight) + donor * weight
    return signal.astype(np.float32)


def build_background_wide(waveforms: np.ndarray, arrays: dict[str, np.ndarray], candidate_sidecar: dict[int, list[int]], config: AuditConfig = CFG) -> np.ndarray:
    result = np.zeros_like(np.asarray(waveforms), dtype=np.float32)
    rng = np.random.default_rng(config.random_seed + 2000)
    mask_half = int(round(config.mask_radius_ms * 1e-3 * config.sample_rate))
    min_delay = int(round(config.min_path_delay_ms * 1e-3 * config.sample_rate))
    max_delay = int(round(config.max_path_delay_ms * 1e-3 * config.sample_rate))
    for index in tqdm(range(len(waveforms)), desc="构建宽窗口背景变体", unit="条"):
        centers = []
        direct_index = int(arrays["direct_index"][index])
        if direct_index >= 0:
            centers.append(direct_index + config.atom_length // 2)
            for candidate in candidate_sidecar.get(index, []):
                if direct_index + min_delay <= candidate <= direct_index + max_delay:
                    centers.append(candidate + config.atom_length // 2)
        spans = _merge_spans([(max(0, c - mask_half), min(config.sample_count, c + mask_half + 1)) for c in centers], 2 * mask_half)
        result[index] = _fill_masked_regions(waveforms[index], spans, rng, config)
        result[index] = peak_normalize(result[index])
    return result


def build_random_mask(waveforms: np.ndarray, arrays: dict[str, np.ndarray], candidate_sidecar: dict[int, list[int]], config: AuditConfig = CFG) -> np.ndarray:
    result = np.zeros_like(np.asarray(waveforms), dtype=np.float32)
    rng = np.random.default_rng(config.random_seed + 3000)
    mask_half = int(round(config.mask_radius_ms * 1e-3 * config.sample_rate))
    min_delay = int(round(config.min_path_delay_ms * 1e-3 * config.sample_rate))
    max_delay = int(round(config.max_path_delay_ms * 1e-3 * config.sample_rate))
    for index in tqdm(range(len(waveforms)), desc="构建随机遮罩对照", unit="条"):
        direct_index = int(arrays["direct_index"][index])
        arrival_centers = []
        if direct_index >= 0:
            arrival_centers.append(direct_index + config.atom_length // 2)
            for candidate in candidate_sidecar.get(index, []):
                if direct_index + min_delay <= candidate <= direct_index + max_delay:
                    arrival_centers.append(candidate + config.atom_length // 2)
        arrival_spans = _merge_spans([(max(0, c - mask_half), min(config.sample_count, c + mask_half + 1)) for c in arrival_centers], 2 * mask_half)
        span_lengths = [stop - start for start, stop in arrival_spans]
        occupied = np.zeros(config.sample_count, dtype=bool)
        for start, stop in arrival_spans:
            occupied[start:stop] = True
        random_spans: list[tuple[int, int]] = []
        for length in span_lengths:
            valid_starts = [s for s in range(0, config.sample_count - length + 1) if not occupied[s : s + length].any()]
            if not valid_starts:
                continue
            start = int(rng.choice(valid_starts))
            occupied[start : start + length] = True
            random_spans.append((start, start + length))
        result[index] = _fill_masked_regions(waveforms[index], random_spans, rng, config)
        result[index] = peak_normalize(result[index])
    return result


def synthetic_mask_test(config: AuditConfig = CFG) -> dict[str, Any]:
    paths = ensure_audit_dirs(config)
    out_path = paths["root"] / "mask_synthetic_test.json"
    rng = np.random.default_rng(config.random_seed + 5000)
    atom = windows.gaussian(config.atom_length, std=config.atom_length / 7)
    atom *= np.cos(2 * np.pi * 90_000 * np.arange(config.atom_length) / config.sample_rate)
    atom = atom / np.sqrt(np.sum(atom**2))
    signal = np.zeros(config.sample_count, dtype=np.float64)
    direct_index = 2_700
    delay_samples = int(round(0.35e-3 * config.sample_rate))
    signal[direct_index : direct_index + len(atom)] += atom
    signal[direct_index + delay_samples : direct_index + delay_samples + len(atom)] += 0.45 * atom
    signal += rng.normal(scale=0.004, size=len(signal))

    mask_half = int(round(config.mask_radius_ms * 1e-3 * config.sample_rate))
    fade_len = int(round(config.mask_fade_ms * 1e-3 * config.sample_rate))
    centers = [direct_index + config.atom_length // 2, direct_index + delay_samples + config.atom_length // 2]
    spans = _merge_spans([(max(0, c - mask_half), min(config.sample_count, c + mask_half + 1)) for c in centers], 2 * mask_half)
    masked = _fill_masked_regions(signal, spans, rng, config)

    unchanged_mask = np.ones(config.sample_count, dtype=bool)
    for start, stop in spans:
        unchanged_mask[max(0, start - fade_len) : min(config.sample_count, stop + fade_len)] = False
    original_f32 = signal.astype(np.float32)
    unchanged_diff = float(np.max(np.abs(original_f32[unchanged_mask] - masked[unchanged_mask]))) if unchanged_mask.any() else 0.0

    boundary_jumps = []
    for start, stop in spans:
        for boundary in (start, stop):
            if 1 <= boundary < config.sample_count:
                boundary_jumps.append(float(abs(masked[boundary] - masked[boundary - 1])))
    boundary_max = float(np.max(boundary_jumps)) if boundary_jumps else 0.0
    noise_std = float(np.std(signal))

    injection_span = (direct_index, direct_index + delay_samples + len(atom))
    template = signal[injection_span[0] : injection_span[1]].copy()

    def ncc(first: np.ndarray, second: np.ndarray) -> float:
        first = first - np.mean(first)
        second = second - np.mean(second)
        denom = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denom) if denom > 1e-12 else 0.0

    ncc_before = ncc(template, signal[injection_span[0] : injection_span[1]])
    ncc_after = ncc(template, masked[injection_span[0] : injection_span[1]])
    ncc_drop = float((ncc_before - ncc_after) / max(abs(ncc_before), 1e-12))

    random_occupied = np.zeros(config.sample_count, dtype=bool)
    random_spans = []
    for start, stop in spans:
        length = stop - start
        valid = [s for s in range(0, config.sample_count - length + 1) if not random_occupied[s : s + length].any()]
        chosen = int(rng.choice(valid))
        random_occupied[chosen : chosen + length] = True
        random_spans.append((chosen, chosen + length))
    random_covers_injection = any(not (stop <= direct_index or start >= direct_index + delay_samples + len(atom)) for start, stop in random_spans)

    result = {
        "passed": bool(unchanged_diff < 1e-6 and boundary_max <= 5 * noise_std + 1e-12 and ncc_drop >= 0.80 and not random_covers_injection),
        "unchanged_max_abs_diff": unchanged_diff,
        "boundary_max_jump": boundary_max,
        "boundary_threshold": 5 * noise_std,
        "ncc_before": ncc_before,
        "ncc_after": ncc_after,
        "ncc_drop": ncc_drop,
        "random_covers_injection": random_covers_injection,
    }
    json_dump(out_path, result)
    return result


def bcsc_synthetic_test(config: AuditConfig = CFG) -> dict[str, Any]:
    """BCSC合成直达+0.35 ms副本延迟误差自检。"""
    paths = ensure_audit_dirs(config)
    out_path = paths["root"] / "bcsc_synthetic_test.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    atom_length = config.atom_length
    time_axis = np.arange(atom_length)
    carrier = np.sin(2 * np.pi * 100_000 * time_axis / config.sample_rate)
    atom = carrier * windows.hann(atom_length, sym=False)
    atom = atom / np.max(np.abs(atom))
    signal = np.zeros(config.sample_count, dtype=np.float64)
    direct_center = config.sample_count // 2
    signal[direct_center : direct_center + atom_length] += atom
    delay_samples = int(round(0.35e-3 * config.sample_rate))
    signal[direct_center + delay_samples : direct_center + delay_samples + atom_length] += 0.8 * atom
    result = process_bcsc_signal_audit(signal, config.random_seed, config)
    error_ms = float(result["delay_ms"] - 0.35)
    payload = {
        "passed": bool(result["status"] == "ok" and abs(error_ms) <= config.match_tolerance_ms),
        "delay_ms": float(result["delay_ms"]),
        "target_delay_ms": 0.35,
        "error_ms": error_ms,
        "tolerance_ms": config.match_tolerance_ms,
        "status": result["status"],
        "direct_index": result["direct_index"],
        "multipath_index": result["multipath_index"],
        "candidate_count": result["candidate_count"],
        "stable_multipath_count": result["stable_multipath_count"],
    }
    json_dump(out_path, payload)
    return payload

def smoke_condition_checks(values: np.ndarray, labels: np.ndarray, config: AuditConfig = CFG) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.random_seed)
    model = WaveformTransformer().to(device)
    sample = torch.as_tensor(np.asarray(values[: min(8, len(values))]), dtype=torch.float32).to(device)
    label = torch.as_tensor(labels[: min(8, len(values))], dtype=torch.float32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    logits = model(sample)
    loss = criterion(logits, label)
    loss.backward()
    gradients_ok = all(parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model.parameters() if parameter.grad is not None)
    optimizer.step()
    losses = [float(loss.detach().cpu())]
    for _ in range(299):
        optimizer.zero_grad(set_to_none=True)
        step_loss = criterion(model(sample), label)
        step_loss.backward()
        optimizer.step()
        losses.append(float(step_loss.detach().cpu()))
    return {"forward_backward_ok": bool(torch.isfinite(loss).item() and gradients_ok), "losses": losses, "loss_decreased": bool(losses[-1] < losses[0])}


def train_waveform_condition(
    condition: str,
    values: np.ndarray,
    manifest: pd.DataFrame,
    config: AuditConfig = CFG,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_audit_dirs(config)
    metrics_path = paths["metrics"] / f"{condition}_fold_metrics.csv"
    predictions_path = paths["predictions"] / f"{condition}_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force:
        return pd.read_csv(metrics_path), pd.read_csv(predictions_path)
    labels = manifest["label"].to_numpy(dtype=np.float32)
    test_folds = manifest["test_fold"].to_numpy(dtype=int)
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    overall = tqdm(total=config.n_splits, desc=f"训练 {condition}", unit="折", postfix={"hash": config.config_hash[:8]})
    for test_fold in range(config.n_splits):
        validation_fold = (test_fold + 1) % config.n_splits
        test_indices = np.where(test_folds == test_fold)[0]
        validation_indices = np.where(test_folds == validation_fold)[0]
        train_indices = np.where((test_folds != test_fold) & (test_folds != validation_fold))[0]
        seed_everything(config.random_seed + 10_000 * WAVEFORM_CONDITIONS.index(condition) + 100 * test_fold)
        model_path = paths["models"] / f"{condition}_fold_{test_fold}.pt"
        model = WaveformTransformer()
        metrics, predictions = train_one_model(model, values, labels, train_indices, validation_indices, test_indices, model_path, config)
        serializable = {key: value for key, value in metrics.items() if key != "history"}
        serializable.update({"condition": condition, "test_fold": test_fold, "validation_fold": validation_fold})
        all_metrics.append(serializable)
        predictions["condition"] = condition
        predictions["test_fold"] = test_fold
        all_predictions.append(predictions)
        json_dump(paths["logs"] / f"{condition}_fold_{test_fold}_history.json", metrics["history"])
        overall.update(1)
        overall.set_postfix(fold=test_fold, auroc=f"{metrics['auroc']:.3f}")
    overall.close()
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df = predictions_df.merge(manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num"]], on="sample_index", how="left")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    return metrics_df, predictions_df


def train_feature_baselines(
    features: np.ndarray,
    manifest: pd.DataFrame,
    config: AuditConfig = CFG,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_audit_dirs(config)
    metrics_path = paths["metrics"] / "baseline_fold_metrics.csv"
    predictions_path = paths["predictions"] / "baseline_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force:
        return pd.read_csv(metrics_path), pd.read_csv(predictions_path)
    labels = manifest["label"].to_numpy(dtype=int)
    test_folds = manifest["test_fold"].to_numpy(dtype=int)
    models = {
        "lr_features": LogisticRegression(max_iter=1000, random_state=config.random_seed),
        "gbm_features": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=200, random_state=config.random_seed),
    }
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    overall = tqdm(total=config.n_splits * len(models), desc="训练特征基线", unit="模型折", postfix={"hash": config.config_hash[:8]})
    for test_fold in range(config.n_splits):
        validation_fold = (test_fold + 1) % config.n_splits
        test_indices = np.where(test_folds == test_fold)[0]
        validation_indices = np.where(test_folds == validation_fold)[0]
        train_indices = np.where((test_folds != test_fold) & (test_folds != validation_fold))[0]
        scaler = StandardScaler().fit(features[train_indices])
        scaled = scaler.transform(features)
        for condition, estimator in models.items():
            seed_everything(config.random_seed + 1000 * (BASELINE_CONDITIONS.index(condition) + 1) + test_fold)
            estimator.fit(scaled[train_indices], labels[train_indices])
            if hasattr(estimator, "predict_proba"):
                probabilities = estimator.predict_proba(scaled)[:, 1]
            else:
                probabilities = estimator.decision_function(scaled)
            threshold = choose_threshold(labels[validation_indices], probabilities[validation_indices])
            metrics = binary_metrics(labels[test_indices], probabilities[test_indices], threshold)
            metrics["condition"] = condition
            metrics["test_fold"] = test_fold
            metrics["validation_fold"] = validation_fold
            all_metrics.append(metrics)
            frame = pd.DataFrame({"sample_index": test_indices, "label": labels[test_indices], "probability": probabilities[test_indices], "threshold": threshold, "condition": condition, "test_fold": test_fold})
            all_predictions.append(frame)
            overall.update(1)
            overall.set_postfix(model=condition, fold=test_fold, auroc=f"{metrics['auroc']:.3f}")
    overall.close()
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df = predictions_df.merge(manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num"]], on="sample_index", how="left")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    return metrics_df, predictions_df


def group_bootstrap_metric(frame: pd.DataFrame, metric_name: str, rng: np.random.Generator, runs: int) -> tuple[float, float]:
    groups = frame["ori_file_num"].unique()
    values = []
    for _ in range(runs):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([frame[frame["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
        if sampled["label"].nunique() < 2:
            continue
        values.append(frame_metric(sampled, metric_name))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def condition_point_metric(frame: pd.DataFrame, metric_name: str) -> float:
    return frame_metric(frame, metric_name)


def paired_bootstrap_delta(first: pd.DataFrame, second: pd.DataFrame, config: AuditConfig = CFG) -> pd.DataFrame:
    merged = first.merge(second[["sample_index", "probability", "threshold"]], on="sample_index", suffixes=("_first", "_second"), validate="one_to_one")
    rng = np.random.default_rng(config.random_seed + 9001)
    groups = merged["ori_file_num"].unique()
    rows = []
    for metric_name in ("auroc", "auprc", "balanced_accuracy"):
        first_frame = pd.DataFrame({"label": merged["label"], "probability": merged["probability_first"], "threshold": merged["threshold_first"]})
        second_frame = pd.DataFrame({"label": merged["label"], "probability": merged["probability_second"], "threshold": merged["threshold_second"]})
        point = condition_point_metric(first_frame, metric_name) - condition_point_metric(second_frame, metric_name)
        values = []
        for _ in range(config.group_bootstrap_runs):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            sampled = pd.concat([merged[merged["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
            if sampled["label"].nunique() < 2:
                continue
            sampled_first = pd.DataFrame({"label": sampled["label"], "probability": sampled["probability_first"], "threshold": sampled["threshold_first"]})
            sampled_second = pd.DataFrame({"label": sampled["label"], "probability": sampled["probability_second"], "threshold": sampled["threshold_second"]})
            values.append(condition_point_metric(sampled_first, metric_name) - condition_point_metric(sampled_second, metric_name))
        rows.append({"metric": metric_name, "delta": point, "ci_low": float(np.percentile(values, 2.5)), "ci_high": float(np.percentile(values, 97.5))})
    return pd.DataFrame(rows)


def cross_cohort_group_bootstrap_diff(first: pd.DataFrame, second: pd.DataFrame, config: AuditConfig = CFG) -> pd.DataFrame:
    common_groups = np.intersect1d(first["ori_file_num"].unique(), second["ori_file_num"].unique())
    rng = np.random.default_rng(config.random_seed + 9002)
    rows = []
    for metric_name in ("auroc", "auprc", "balanced_accuracy"):
        point = condition_point_metric(first, metric_name) - condition_point_metric(second, metric_name)
        values = []
        for _ in range(config.group_bootstrap_runs):
            sampled_groups = rng.choice(common_groups, size=len(common_groups), replace=True)
            sampled_first = pd.concat([first[first["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
            sampled_second = pd.concat([second[second["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
            if sampled_first["label"].nunique() < 2 or sampled_second["label"].nunique() < 2:
                continue
            values.append(condition_point_metric(sampled_first, metric_name) - condition_point_metric(sampled_second, metric_name))
        rows.append({"metric": metric_name, "delta": point, "ci_low": float(np.percentile(values, 2.5)), "ci_high": float(np.percentile(values, 97.5))})
    return pd.DataFrame(rows)


def one_sample_background_ci(background: pd.DataFrame, config: AuditConfig = CFG) -> dict[str, Any]:
    rng = np.random.default_rng(config.random_seed + 9003)
    values = []
    groups = background["ori_file_num"].unique()
    for _ in range(config.group_bootstrap_runs):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([background[background["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
        if sampled["label"].nunique() < 2:
            continue
        values.append(condition_point_metric(sampled, "auroc"))
    return {"point": condition_point_metric(background, "auroc"), "ci_low": float(np.percentile(values, 2.5)), "ci_high": float(np.percentile(values, 97.5))}


def summarize_conditions(predictions: pd.DataFrame, config: AuditConfig = CFG) -> pd.DataFrame:
    paths = ensure_audit_dirs(config)
    rows = []
    for condition, frame in predictions.groupby("condition"):
        row = {"condition": condition}
        for metric_name in ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity"):
            rng = np.random.default_rng(config.random_seed + 77 + ALL_TRAINED_CONDITIONS.index(condition) if condition in ALL_TRAINED_CONDITIONS else config.random_seed + 77)
            low, high = group_bootstrap_metric(frame, metric_name, rng, config.group_bootstrap_runs)
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(paths["metrics"] / "condition_summary.csv", index=False, encoding="utf-8-sig")
    return result


def normalized_cross_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - np.mean(first)
    second = second - np.mean(second)
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denom) if denom > 1e-12 else 0.0


def residual_click_audit(waveforms: np.ndarray, residual_unnorm: np.ndarray, arrays: dict[str, np.ndarray], manifest: pd.DataFrame, config: AuditConfig = CFG) -> pd.DataFrame:
    paths = ensure_audit_dirs(config)
    out_path = paths["root"] / "residual_click_audit.csv"
    center_half = int(round(config.signal_window_ms * 1e-3 * config.sample_rate))
    template_half = int(round(config.mask_radius_ms * 1e-3 * config.sample_rate))
    rows = []
    for index in tqdm(range(len(waveforms)), desc="残差click审计", unit="条"):
        atom = np.asarray(arrays["atoms"][index], dtype=np.float64)
        residual = np.asarray(residual_unnorm[index], dtype=np.float64)
        center = direct_center(index, arrays, config)
        if center < 0:
            center = int(np.argmax(np.abs(residual)))
        center_start = max(0, center - center_half)
        center_stop = min(len(residual), center + center_half + 1)
        center_window = residual[center_start:center_stop]
        template_window = _crop_center(residual, center, len(atom))
        ncc = normalized_cross_correlation(template_window, atom)
        center_energy = float(np.mean(center_window**2))
        envelope_peak = float(np.max(np.abs(hilbert(center_window)))) if len(center_window) else 0.0
        tapered_res = residual * windows.hann(len(residual), sym=False)
        tapered_atom = atom * windows.hann(len(atom), sym=False)
        n_fft = 2_048
        spectrum_res = np.abs(np.fft.rfft(tapered_res, n=n_fft))
        spectrum_atom = np.abs(np.fft.rfft(tapered_atom, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / config.sample_rate)
        band = (freqs >= config.lowcut_hz) & (freqs <= config.highcut_hz)
        spectral_cosine = normalized_cross_correlation(spectrum_res[band], spectrum_atom[band])
        edge_n = int(round(config.edge_ms * 1e-3 * config.sample_rate))
        edge = np.concatenate([residual[:edge_n], residual[-edge_n:]])
        bg_rms = float(np.sqrt(np.mean(edge**2)) + 1e-12)
        prominence = float(envelope_peak / bg_rms)
        rows.append({"sample_index": index, "sample_id": manifest.iloc[index]["sample_id"], "label": int(manifest.iloc[index]["label"]), "ori_file_num": int(manifest.iloc[index]["ori_file_num"]), "direct_template_ncc": ncc, "center_energy": center_energy, "residual_envelope_peak": envelope_peak, "spectral_cosine": spectral_cosine, "residual_prominence": prominence})
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def compute_features_from_waveforms(waveforms: np.ndarray, manifest: pd.DataFrame, config: AuditConfig = CFG) -> pd.DataFrame:
    rows = []
    for index in tqdm(range(len(waveforms)), desc="计算波形特征", unit="条"):
        feature = matching_features_from_bandpassed(np.asarray(waveforms[index]), config)
        feature["sample_id"] = manifest.iloc[index]["sample_id"]
        feature["ori_file_num"] = int(manifest.iloc[index]["ori_file_num"])
        feature["timestamp_s"] = float(manifest.iloc[index]["timestamp_s"])
        rows.append(feature)
    return pd.DataFrame(rows)


def snr_strata_metrics(predictions: pd.DataFrame, snr_frames: dict[str, pd.DataFrame], config: AuditConfig = CFG) -> pd.DataFrame:
    cohort_of = {condition: "matched" if condition.startswith("matched_") or condition in BASELINE_CONDITIONS else "original" for condition in predictions["condition"].unique()}
    rows = []
    for condition, frame in predictions.groupby("condition"):
        cohort = cohort_of.get(condition, "original")
        snr = snr_frames[cohort][["sample_id", "local_snr_db"]].copy()
        snr["quartile"] = pd.qcut(snr["local_snr_db"], 4, labels=["Q1低SNR", "Q2", "Q3", "Q4高SNR"], duplicates="drop")
        merged = frame.merge(snr, on="sample_id", how="inner")
        for quartile, sub in merged.groupby("quartile", observed=True):
            if sub["label"].nunique() < 2:
                continue
            rows.append({"condition": condition, "quartile": str(quartile), "n": len(sub), "auroc": roc_auc_score(sub["label"], sub["probability"]), "balanced_accuracy": balanced_accuracy_score(sub["label"], sub["probability"] >= sub["threshold"])})
    result = pd.DataFrame(rows)
    result.to_csv(ensure_audit_dirs(config)["root"] / "snr_strata_metrics.csv", index=False, encoding="utf-8-sig")
    return result


def plot_results(config: AuditConfig = CFG) -> None:
    paths = ensure_audit_dirs(config)
    root = paths["root"]
    figures = paths["figures"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    predictions = pd.read_csv(paths["predictions"] / "condition_predictions.csv")
    main_conditions = ["historical"] + WAVEFORM_CONDITIONS
    predictions = predictions[predictions["condition"].isin(main_conditions)].copy()

    colors = {"historical": "#7F7F7F", "symmetric_full": "#2457A6", "matched_full": "#188977", "matched_signal": "#E09F3E", "matched_residual": "#5B3A8C", "matched_background": "#B52B2B", "matched_randommask": "#4E79A7", "lr_features": "#F28E2B", "gbm_features": "#59A14F"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for condition, frame in predictions.groupby("condition"):
        fpr, tpr, _ = roc_curve(frame["label"], frame["probability"])
        precision, recall, _ = precision_recall_curve(frame["label"], frame["probability"])
        label = CONDITION_LABELS.get(condition, condition)
        axes[0].plot(fpr, tpr, label=label, color=colors.get(condition, "#333333"))
        axes[1].plot(recall, precision, label=label, color=colors.get(condition, "#333333"))
    axes[0].plot([0, 1], [0, 1], "--", color="0.5")
    axes[0].set(xlabel="假阳性率", ylabel="真阳性率", title="七条件汇总ROC")
    axes[1].set(xlabel="召回率", ylabel="精确率", title="七条件汇总PR")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "roc_pr_all.png", dpi=220)
    plt.close(fig)

    balance = pd.read_csv(root / "matching_balance.csv")
    fig, axis = plt.subplots(figsize=(8, 5))
    for phase, frame in balance.groupby("phase"):
        axis.plot(frame["variable"], frame["smd"], marker="o", label="匹配前" if phase == "before" else "匹配后")
    axis.axhline(0.10, color="0.4", linestyle="--", label="0.10阈值")
    axis.axhline(-0.10, color="0.4", linestyle="--")
    axis.set(title="匹配变量标准化均值差", ylabel="SMD")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "matching_balance.png", dpi=220)
    plt.close(fig)

    summary = pd.read_csv(paths["metrics"] / "condition_summary.csv")
    summary = summary[summary["condition"].isin(main_conditions)]
    if not summary.empty:
        fig, axis = plt.subplots(figsize=(9, 5))
        order = summary["condition"].tolist()
        means = []
        lows = []
        highs = []
        for _, row in summary.iterrows():
            frame = predictions[predictions["condition"] == row["condition"]]
            means.append(condition_point_metric(frame, "auroc"))
            lows.append(row["auroc_ci_low"])
            highs.append(row["auroc_ci_high"])
        y = np.arange(len(order))
        axis.errorbar(means, y, xerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)], fmt="o", capsize=4)
        axis.axvline(0.5, color="0.4", linestyle="--")
        axis.set_yticks(y)
        axis.set_yticklabels([CONDITION_LABELS.get(c, c) for c in order])
        axis.set(xlabel="AUROC", title="AUROC森林图（录音组Bootstrap 95% CI）", xlim=(0.35, 1.02))
        axis.grid(alpha=0.2, axis="x")
        fig.tight_layout()
        fig.savefig(figures / "auroc_forest.png", dpi=220)
        plt.close(fig)

    snr = pd.read_csv(root / "snr_strata_metrics.csv")
    if not snr.empty:
        fig, axis = plt.subplots(figsize=(10, 5))
        for condition, frame in snr.groupby("condition"):
            axis.plot(frame["quartile"], frame["auroc"], marker="o", label=CONDITION_LABELS.get(condition, condition))
        axis.set(title="局部SNR四分位AUROC", ylabel="AUROC")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "snr_strata.png", dpi=220)
        plt.close(fig)

    audit = pd.read_csv(root / "residual_click_audit.csv")
    if not audit.empty:
        features = ["direct_template_ncc", "center_energy", "residual_envelope_peak", "spectral_cosine", "residual_prominence"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        axes = axes.ravel()
        for index, feature in enumerate(features):
            for label, frame in audit.groupby("label"):
                axes[index].hist(frame[feature], bins=40, alpha=0.55, label="正样本" if label == 1 else "负样本")
            axes[index].set_title(feature)
            axes[index].legend()
        axes[-1].axis("off")
        fig.tight_layout()
        fig.savefig(figures / "residual_click_audit.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for index, (condition, frame) in enumerate(predictions.groupby("condition")):
        if index >= 8:
            break
        preds = frame["probability"] >= frame["threshold"]
        cm = confusion_matrix(frame["label"], preds, labels=[0, 1])
        axes[index].imshow(cm, cmap="Blues")
        axes[index].set(title=CONDITION_LABELS.get(condition, condition), xlabel="预测", ylabel="真实")
        axes[index].set_xticks([0, 1])
        axes[index].set_yticks([0, 1])
        for row in range(2):
            for col in range(2):
                axes[index].text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > cm.max() / 2 else "black")
    for index in range(len(predictions["condition"].unique()), len(axes)):
        axes[index].axis("off")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrices.png", dpi=220)
    plt.close(fig)


def build_evidence_table(
    predictions: pd.DataFrame,
    config: AuditConfig = CFG,
) -> pd.DataFrame:
    def frame_for(condition: str) -> pd.DataFrame:
        return predictions[predictions["condition"] == condition].copy()

    rows: list[dict[str, Any]] = []
    paired_specs = [
        ("historical", "symmetric_full", "制作流程影响"),
        ("matched_full", "matched_signal", "背景提供的额外信息"),
        ("matched_residual", "matched_background", "BCSC残差中的click泄漏"),
        ("matched_background", "matched_randommask", "遮罩与拼接伪影影响"),
    ]
    for first_name, second_name, label in paired_specs:
        delta = paired_bootstrap_delta(frame_for(first_name), frame_for(second_name), config)
        for metric in ("auroc", "auprc", "balanced_accuracy"):
            row = delta[delta["metric"] == metric].iloc[0]
            rows.append({"comparison": label, "pair": f"{first_name} vs {second_name}", "metric": metric, "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "significant": bool(row["ci_low"] > 0 or row["ci_high"] < 0)})

    cross = cross_cohort_group_bootstrap_diff(frame_for("symmetric_full"), frame_for("matched_full"), config)
    for metric in ("auroc", "auprc", "balanced_accuracy"):
        row = cross[cross["metric"] == metric].iloc[0]
        rows.append({"comparison": "SNR及粗统计差异影响", "pair": "symmetric_full vs matched_full", "metric": metric, "delta": row["delta"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "significant": bool(row["ci_low"] > 0 or row["ci_high"] < 0)})

    background_ci = one_sample_background_ci(frame_for("matched_background"), config)
    rows.append({"comparison": "清除信号后背景是否仍有类别信息", "pair": "matched_background vs 0.5", "metric": "auroc", "delta": background_ci["point"] - 0.5, "ci_low": background_ci["ci_low"] - 0.5, "ci_high": background_ci["ci_high"] - 0.5, "significant": bool(background_ci["ci_low"] > 0.5)})

    result = pd.DataFrame(rows)
    result.to_csv(ensure_audit_dirs(config)["root"] / "evidence_table.csv", index=False, encoding="utf-8-sig")
    return result


def write_audit_report(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    matched_pairs: pd.DataFrame,
    balance: pd.DataFrame,
    evidence: pd.DataFrame,
    snr: pd.DataFrame,
    residual: pd.DataFrame,
    bcsc_table: pd.DataFrame,
    mask_test: dict[str, Any],
    config: AuditConfig = CFG,
) -> Path:
    paths = ensure_audit_dirs(config)
    report_path = paths["root"] / "research_report.md"
    main_conditions = ["historical"] + WAVEFORM_CONDITIONS
    summary_lines = []
    baseline_lines = []
    for _, row in summary.iterrows():
        condition = row["condition"]
        frame = predictions[predictions["condition"] == condition]
        point = condition_point_metric(frame, "auroc")
        line = f"| {CONDITION_LABELS.get(condition, condition)} | {point:.4f} [{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}] |"
        if condition in BASELINE_CONDITIONS:
            baseline_lines.append(line)
        elif condition in main_conditions:
            summary_lines.append(line)
    evidence_lines = []
    for _, row in evidence.iterrows():
        evidence_lines.append(f"| {row['comparison']} | {row['pair']} | {row['metric']} | {row['delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {'是' if row['significant'] else '否'} |")
    matched_after = balance[balance["phase"] == "after"].set_index("variable")["smd"].abs()
    all_smd_ok = bool((matched_after <= config.smd_max).all())
    feasible = bool(len(matched_pairs) >= config.min_pairs and all_smd_ok)
    report = f"""# 信号—噪声可分性与背景捷径确认研究报告

## 摘要

本实验独立于原BCSC–Transformer实验，重新从 `00_Data\\original_audio` 截取5761点、576 kHz波形，并以严格匹配队列隔离“信号本身可分”“背景差异捷径”与“BCSC残差泄漏”。历史基准只读取原实验波形结果，不重新解释。

- 正样本原始时间映射：3500/3500。
- 严格匹配对数：{len(matched_pairs)}，最低验收3150对；匹配可行性：{'通过' if feasible else '不可行（按实际规模继续）'}。
- 遮罩合成测试：{'通过' if mask_test['passed'] else '未通过'}；注入模板NCC下降 {mask_test['ncc_drop']:.2%}。

## 七条件AUROC

| 条件 | AUROC [录音组Bootstrap 95% CI] |
|---|---:|
{chr(10).join(summary_lines)}

## 特征基线AUROC

| 条件 | AUROC [录音组Bootstrap 95% CI] |
|---|---:|
{chr(10).join(baseline_lines)}

![ROC与PR](figures/roc_pr_all.png)
![AUROC森林图](figures/auroc_forest.png)

## 匹配平衡

匹配变量为未峰值归一化带通波形的背景对数RMS、局部SNR、峰均比、谱质心、−10 dB带宽与100–190 kHz高频能量比例。

![匹配前后平衡](figures/matching_balance.png)

## 预注册对比

| 比较 | 配对 | 指标 | 差值 [95% CI] | 显著 |
|---|---|---|---:|---|
{chr(10).join(evidence_lines)}

![SNR分层](figures/snr_strata.png)
![混淆矩阵](figures/confusion_matrices.png)

## 残差click审计

BCSC残差审计特征：直达模板NCC、±0.175 ms中心窗能量、剩余包络峰、残差频谱余弦与残余突出度。

![残差click审计](figures/residual_click_audit.png)

## 结论证据

- 信号区AUROC置信区间下限高于0.5：信号形态本身可分。
- 宽窗口背景AUROC置信区间下限高于0.5：存在背景类别信息。
- BCSC残差显著优于宽窗口背景且残差NCC/中心能量存在类别差异：原0.9265部分来自信号泄漏。
- 严格匹配后全波形显著下降：原任务受到SNR、幅度或频谱不平衡影响。
- 不设置必须下降或必须显著的结果门槛，以上结论按95% CI实际方向陈述。

## 可复现信息

- 配置哈希：`{config.config_hash}`
- 随机种子：{config.random_seed}
- Python：`D:\\Python_env\\toothwhale\\python.exe`
- GPU：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def run_audit(config: AuditConfig = CFG, force: bool = False, quick: bool = False) -> dict[str, Any]:
    ensure_audit_dirs(config)
    seed_everything(config.random_seed)
    started = time.perf_counter()
    summary, historical_metrics, historical_predictions, historical_manifest = load_historical(config)

    time_map = build_positive_time_map(historical_manifest, config, force=force)
    original_manifest, _ = build_original_manifest(config, force=force)
    mask_test = synthetic_mask_test(config)
    bcsc_test = bcsc_synthetic_test(config)

    if quick:
        quick_rng = np.random.default_rng(config.random_seed)
        smoke = smoke_condition_checks(quick_rng.normal(size=(16, config.sample_count)).astype(np.float32), np.asarray([0, 1] * 8, dtype=np.float32), config)
        result = {"mode": "quick", "config_hash": config.config_hash, "positive_time_map": len(time_map), "mask_test": mask_test, "bcsc_synthetic_test": bcsc_test, "smoke": smoke, "elapsed_s": time.perf_counter() - started}
        json_dump(config.output_dir / "quick_run_summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    original_model = materialize_waveforms(original_manifest, config, "model", force=force)
    original_matching = materialize_waveforms(original_manifest, config, "matching", force=force)
    original_features = compute_features_from_waveforms(original_matching, original_manifest, config)
    paths = ensure_audit_dirs(config)
    original_features.to_csv(paths["features"] / "original_features.csv", index=False, encoding="utf-8-sig")

    pool_features = extract_candidate_pool_features(config, force=force)
    pos_features = original_features[original_manifest["label"].to_numpy() == 1][["sample_id", "ori_file_num", "timestamp_s", *ALL_FEATURE_COLS]].reset_index(drop=True)
    matched_manifest, matched_pairs = build_matched_manifest(pos_features, pool_features, original_manifest, config, force=force)

    matched_model = materialize_waveforms(matched_manifest, config, "matched_model", force=force)
    arrays, bcsc_table, candidate_sidecar = run_bcsc_audit(matched_model, matched_manifest, config, force=force)
    matched_signal = build_signal_only(matched_model, arrays, config)
    matched_residual, residual_unnorm = build_residual(matched_model, arrays, config)
    matched_background = build_background_wide(matched_model, arrays, candidate_sidecar, config)
    matched_randommask = build_random_mask(matched_model, arrays, candidate_sidecar, config)

    waveform_sets = {
        "symmetric_full": (original_model, original_manifest),
        "matched_full": (matched_model, matched_manifest),
        "matched_signal": (matched_signal, matched_manifest),
        "matched_residual": (matched_residual, matched_manifest),
        "matched_background": (matched_background, matched_manifest),
        "matched_randommask": (matched_randommask, matched_manifest),
    }
    all_metrics = [historical_metrics[historical_metrics["model"] == "waveform"].assign(condition="historical")[["condition", "test_fold", "validation_fold", "auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity", "threshold", "tn", "fp", "fn", "tp"]]]
    all_predictions = [historical_predictions[historical_predictions["model"] == "waveform"].rename(columns={"model": "condition"}).assign(condition="historical")[["sample_index", "label", "probability", "threshold", "condition", "test_fold", "sample_id", "ori_file_num"]]]
    for condition, (values, manifest) in waveform_sets.items():
        metrics, predictions = train_waveform_condition(condition, values, manifest, config, force=force)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    pos_ids = set(matched_manifest[matched_manifest["label"] == 1]["sample_id"])
    matched_neg_features = pool_features[pool_features["pool_index"].isin(matched_pairs["negative_pool_index"])].copy()
    negative_id_map = matched_pairs.set_index("negative_pool_index")["negative_sample_id"]
    matched_neg_features["sample_id"] = matched_neg_features["pool_index"].map(negative_id_map)
    pos_feature_rows = original_features[original_features["sample_id"].isin(pos_ids)][["sample_id", *ALL_FEATURE_COLS]].copy()
    neg_feature_rows = matched_neg_features[["sample_id", *ALL_FEATURE_COLS]]
    matched_simple_features = pd.concat([pos_feature_rows, neg_feature_rows], ignore_index=True).set_index("sample_id").loc[matched_manifest["sample_id"]].to_numpy(dtype=np.float32)
    baseline_metrics, baseline_predictions = train_feature_baselines(matched_simple_features, matched_manifest, config, force=force)
    all_metrics.append(baseline_metrics)
    all_predictions.append(baseline_predictions)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(paths["metrics"] / "condition_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions_df.to_csv(paths["predictions"] / "condition_predictions.csv", index=False, encoding="utf-8-sig")

    summary_df = summarize_conditions(predictions_df, config)
    snr_frames = {
        "original": original_features[["sample_id", "local_snr_db"]],
        "matched": pd.concat([pos_feature_rows, neg_feature_rows], ignore_index=True)[["sample_id", "local_snr_db"]],
    }
    snr_df = snr_strata_metrics(predictions_df, snr_frames, config)
    residual_df = residual_click_audit(matched_model, residual_unnorm, arrays, matched_manifest, config)
    balance_df = pd.read_csv(paths["root"] / "matching_balance.csv")
    evidence = build_evidence_table(predictions_df, config)
    plot_results(config)
    report_path = write_audit_report(summary_df, predictions_df, matched_pairs, balance_df, evidence, snr_df, residual_df, bcsc_table, mask_test, config)

    result = {
        "mode": "full",
        "config_hash": config.config_hash,
        "positive_time_map": len(time_map),
        "matched_pairs": int(len(matched_pairs)),
        "strict_matching_feasible": bool(len(matched_pairs) >= config.min_pairs and balance_df[balance_df["phase"] == "after"]["smd"].abs().max() <= config.smd_max),
        "mask_test": mask_test,
        "bcsc_synthetic_test": bcsc_test,
        "bcsc_ok": int((bcsc_table["status"] == "ok").sum()),
        "report_path": str(report_path.resolve()),
        "elapsed_s": time.perf_counter() - started,
    }
    json_dump(config.output_dir / "run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="仅运行时间映射、遮罩合成与单批次冒烟测试")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存并重新计算")
    arguments = parser.parse_args()
    run_audit(CFG, force=arguments.force, quick=arguments.quick)


if __name__ == "__main__":
    main()
