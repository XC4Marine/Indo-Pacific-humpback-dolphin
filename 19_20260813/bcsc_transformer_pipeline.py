#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BCSC 路径提取、显式特征 Transformer 与捷径学习检验流水线。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.fft import dct
from scipy.signal import (
    butter,
    correlate,
    fftconvolve,
    find_peaks,
    hilbert,
    sosfiltfilt,
    windows,
)
from scipy.stats import kurtosis
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Config:
    """保存实验的固定参数与绝对路径。"""

    project_root: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
    module_dir: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813")
    output_dir: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\19_20260813\outputs")
    positive_root: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\Positive_Clicks_3500")
    clicktrains_csv: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\ClickTrains.csv")
    negative_csv: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\14\negative_samples_log.csv")
    original_audio_root: Path = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin\00_Data\original_audio")
    sample_rate: int = 576_000
    sample_count: int = 5_761
    lowcut_hz: float = 20_000.0
    highcut_hz: float = 190_000.0
    filter_order: int = 5
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
    group_bootstrap_runs: int = 1_000
    bcsc_checkpoint_interval: int = 25
    bcsc_workers: int = 6

    @property
    def atom_length(self) -> int:
        """返回最接近85微秒的奇数原子长度。"""
        length = int(round(self.atom_duration_us * 1e-6 * self.sample_rate))
        return length + (length % 2 == 0)

    @property
    def config_hash(self) -> str:
        """返回影响缓存正确性的参数哈希。"""
        payload = asdict(self).copy()
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


CFG = Config()
DIRECT_FEATURE_NAMES: list[str] = []
RELATION_FEATURE_NAMES = [
    "multipath_present",
    "direct_confidence",
    "multipath_confidence",
    "delay_ms",
    "delay_ci_width_ms",
    "relative_amplitude",
    "attenuation_db",
    "polarity",
    "energy_ratio",
    "ncc",
    "spectral_cosine",
    "coherence",
    "overlap_ratio",
    "explained_variance",
    "residual_energy_change",
]


def seed_everything(seed: int) -> None:
    """固定Python、NumPy和PyTorch随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_output_dirs(config: Config) -> dict[str, Path]:
    """创建并返回流水线输出子目录。"""
    paths = {
        "root": config.output_dir,
        "cache": config.output_dir / "cache",
        "models": config.output_dir / "models",
        "figures": config.output_dir / "figures",
        "logs": config.output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def json_dump(path: Path, value: Any) -> None:
    """以UTF-8原子方式写入JSON。"""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def parse_pulse_train(path: Path) -> int:
    """从正样本路径解析PulseTrain编号。"""
    match = re.search(r"PulseTrain_(\d+)", str(path))
    if not match:
        raise ValueError(f"无法解析PulseTrain编号：{path}")
    return int(match.group(1))


def parse_original_number(audio_file: str) -> int:
    """从Ori_Recording文件名解析原始录音编号。"""
    match = re.search(r"Ori_Recording_(\d+)", audio_file)
    if not match:
        raise ValueError(f"无法解析原始录音编号：{audio_file}")
    return int(match.group(1))


def _select_nonoverlapping_rows(
    source_df: pd.DataFrame,
    target_count: int,
    rng: np.random.Generator,
    min_separation_s: float = 0.010,
) -> pd.DataFrame:
    """按噪声类型近似分层并选择互不重叠的时间戳。"""
    source_df = source_df.copy()
    type_counts = source_df["Noise_Type"].value_counts(normalize=True)
    quotas = {name: int(math.floor(target_count * fraction)) for name, fraction in type_counts.items()}
    for name in type_counts.index[: target_count - sum(quotas.values())]:
        quotas[name] += 1

    selected_indices: list[int] = []
    selected_times: list[float] = []

    def can_add(timestamp: float) -> bool:
        return all(abs(timestamp - previous) >= min_separation_s for previous in selected_times)

    for noise_type, quota in quotas.items():
        if quota <= 0:
            continue
        candidates = source_df[source_df["Noise_Type"] == noise_type]
        order = rng.permutation(candidates.index.to_numpy())
        for index in order:
            timestamp = float(source_df.at[index, "Noise_Timestamp(s)"])
            if can_add(timestamp):
                selected_indices.append(int(index))
                selected_times.append(timestamp)
                if sum(source_df.loc[selected_indices, "Noise_Type"] == noise_type) >= quota:
                    break

    if len(selected_indices) < target_count:
        remaining = source_df.drop(index=selected_indices)
        for index in rng.permutation(remaining.index.to_numpy()):
            timestamp = float(source_df.at[index, "Noise_Timestamp(s)"])
            if can_add(timestamp):
                selected_indices.append(int(index))
                selected_times.append(timestamp)
                if len(selected_indices) == target_count:
                    break
    if len(selected_indices) != target_count:
        raise RuntimeError(
            f"来源{source_df['ori_file_num'].iloc[0]}仅选择到{len(selected_indices)}/{target_count}条非重叠负样本"
        )
    return source_df.loc[selected_indices].sort_values("Noise_Timestamp(s)").copy()


def build_dataset_manifest(config: Config = CFG, force: bool = False) -> pd.DataFrame:
    """构建3500正样本与来源匹配3500负样本的可追踪清单。"""
    paths = ensure_output_dirs(config)
    manifest_path = paths["root"] / "dataset_manifest.csv"
    if manifest_path.exists() and not force:
        manifest = pd.read_csv(manifest_path)
        if len(manifest) == 7_000 and manifest["label"].value_counts().to_dict() == {1: 3500, 0: 3500}:
            return manifest

    clicktrains = pd.read_csv(config.clicktrains_csv)
    train_column = "Train_num(No.)"
    source_column = "Ori_file_num(No.)"
    train_to_source = (
        clicktrains[[train_column, source_column]]
        .dropna()
        .drop_duplicates(train_column)
        .set_index(train_column)[source_column]
        .astype(int)
        .to_dict()
    )
    positive_rows: list[dict[str, Any]] = []
    positive_paths = sorted(config.positive_root.rglob("*.wav"))
    if len(positive_paths) != 3_500:
        raise RuntimeError(f"正样本数量应为3500，实际为{len(positive_paths)}")
    for wav_path in positive_paths:
        pulse_train_num = parse_pulse_train(wav_path)
        if pulse_train_num not in train_to_source:
            raise KeyError(f"ClickTrains.csv缺少PulseTrain_{pulse_train_num:03d}")
        source_num = int(train_to_source[pulse_train_num])
        positive_rows.append(
            {
                "sample_id": f"P_{wav_path.parent.parent.name}_{wav_path.parent.name}",
                "label": 1,
                "ori_file_num": source_num,
                "audio_file": f"Ori_Recording_{source_num:02d}.wav",
                "timestamp_s": np.nan,
                "noise_type": "positive_click",
                "wav_path": str(wav_path.resolve()),
                "source_kind": "positive_wav",
            }
        )
    positive_df = pd.DataFrame(positive_rows)

    negative_df = pd.read_csv(config.negative_csv)
    negative_df["ori_file_num"] = negative_df["Audio_File"].map(parse_original_number)
    rng = np.random.default_rng(config.random_seed)
    selected_negative: list[pd.DataFrame] = []
    positive_source_counts = positive_df["ori_file_num"].value_counts().sort_index()
    half_window_s = (config.sample_count // 2) / config.sample_rate
    for source_num, target_count in positive_source_counts.items():
        audio_path = config.original_audio_root / f"Ori_Recording_{int(source_num):02d}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        info = sf.info(str(audio_path))
        source_candidates = negative_df[negative_df["ori_file_num"] == int(source_num)].copy()
        source_candidates = source_candidates[
            (source_candidates["Noise_Timestamp(s)"] >= half_window_s)
            & (source_candidates["Noise_Timestamp(s)"] <= info.duration - half_window_s)
        ]
        selected_negative.append(
            _select_nonoverlapping_rows(source_candidates, int(target_count), rng)
        )
    selected_negative_df = pd.concat(selected_negative, ignore_index=True)
    selected_negative_df["sample_id"] = selected_negative_df.apply(
        lambda row: (
            f"N_{int(row['ori_file_num']):02d}_"
            f"{float(row['Noise_Timestamp(s)']):.6f}_{row['Noise_Type']}"
        ),
        axis=1,
    )
    selected_negative_df["label"] = 0
    selected_negative_df["audio_file"] = selected_negative_df["Audio_File"]
    selected_negative_df["timestamp_s"] = selected_negative_df["Noise_Timestamp(s)"].astype(float)
    selected_negative_df["noise_type"] = selected_negative_df["Noise_Type"]
    selected_negative_df["wav_path"] = selected_negative_df["audio_file"].map(
        lambda name: str((config.original_audio_root / name).resolve())
    )
    selected_negative_df["source_kind"] = "original_audio_timestamp"
    negative_manifest = selected_negative_df[
        ["sample_id", "label", "ori_file_num", "audio_file", "timestamp_s", "noise_type", "wav_path", "source_kind"]
    ]

    manifest = pd.concat([positive_df, negative_manifest], ignore_index=True)
    splitter = StratifiedGroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_seed)
    manifest["test_fold"] = -1
    for fold, (_, test_indices) in enumerate(
        splitter.split(manifest, manifest["label"], groups=manifest["ori_file_num"])
    ):
        manifest.loc[test_indices, "test_fold"] = fold
    if (manifest["test_fold"] < 0).any():
        raise RuntimeError("存在未分配交叉验证折的样本")
    manifest["config_hash"] = config.config_hash
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest


def read_manifest_waveform(row: pd.Series, config: Config = CFG) -> np.ndarray:
    """按清单读取正样本或从原始录音精确截取负样本。"""
    path = Path(str(row["wav_path"]))
    if row["source_kind"] == "positive_wav":
        signal, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    else:
        sample_rate = sf.info(str(path)).samplerate
        center = int(round(float(row["timestamp_s"]) * sample_rate))
        start = center - config.sample_count // 2
        with sf.SoundFile(str(path), mode="r") as handle:
            handle.seek(start)
            signal = handle.read(config.sample_count, dtype="float64", always_2d=False)
    if sample_rate != config.sample_rate:
        raise ValueError(f"采样率错误：{path} -> {sample_rate}")
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.ndim != 1 or len(signal) != config.sample_count:
        raise ValueError(f"波形尺寸错误：{path} -> {signal.shape}")
    return signal


def preprocess_waveform(signal: np.ndarray, config: Config = CFG) -> np.ndarray:
    """对两类样本执行完全相同的BCSC/模型预处理。"""
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    sos = butter(
        config.filter_order,
        [config.lowcut_hz / (config.sample_rate / 2), config.highcut_hz / (config.sample_rate / 2)],
        btype="bandpass",
        output="sos",
    )
    filtered = sosfiltfilt(sos, signal)
    filtered *= windows.tukey(len(filtered), alpha=0.04)
    scale = np.max(np.abs(filtered))
    if not np.isfinite(scale) or scale < 1e-12:
        raise RuntimeError("预处理后波形幅度过小")
    return (filtered / scale).astype(np.float32)


def materialize_waveforms(
    manifest: pd.DataFrame,
    config: Config = CFG,
    force: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """读取并统一预处理全部7000条波形，保存可恢复缓存。"""
    paths = ensure_output_dirs(config)
    waveforms_path = paths["cache"] / "preprocessed_waveforms.npy"
    audit_path = paths["root"] / "waveform_audit.csv"
    if waveforms_path.exists() and audit_path.exists() and not force:
        waveforms = np.load(waveforms_path, mmap_mode="r")
        audit = pd.read_csv(audit_path)
        if waveforms.shape == (len(manifest), config.sample_count) and len(audit) == len(manifest):
            return waveforms, audit

    waveforms = np.lib.format.open_memmap(
        waveforms_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(manifest), config.sample_count),
    )
    audit_rows: list[dict[str, Any]] = []
    for index, row in tqdm(manifest.iterrows(), total=len(manifest), desc="总体进度 1/4：读取与统一预处理", unit="条"):
        raw = read_manifest_waveform(row, config)
        processed = preprocess_waveform(raw, config)
        waveforms[index] = processed
        audit_rows.append(
            {
                "sample_id": row["sample_id"],
                "sample_rate_hz": config.sample_rate,
                "sample_count": len(raw),
                "raw_rms": float(np.sqrt(np.mean(raw**2))),
                "raw_peak": float(np.max(np.abs(raw))),
                "processed_rms": float(np.sqrt(np.mean(processed**2))),
                "processed_peak": float(np.max(np.abs(processed))),
                "finite": bool(np.all(np.isfinite(processed))),
            }
        )
    waveforms.flush()
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return np.load(waveforms_path, mmap_mode="r"), audit


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """执行带符号软阈值。"""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def reconstruct(atom: np.ndarray, activation: np.ndarray, signal_length: int) -> np.ndarray:
    """由原子与激活重构固定长度波形。"""
    return fftconvolve(activation, atom, mode="full")[:signal_length]


def adjoint_atom(residual: np.ndarray, atom: np.ndarray, activation_length: int) -> np.ndarray:
    """计算关于稀疏激活的伴随卷积。"""
    return correlate(residual, atom, mode="valid", method="fft")[:activation_length]


def adjoint_code(residual: np.ndarray, activation: np.ndarray, atom_length: int) -> np.ndarray:
    """计算关于原子的伴随卷积。"""
    return correlate(residual, activation, mode="valid", method="fft")[:atom_length]


def spectral_lipschitz(kernel: np.ndarray, output_length: int) -> float:
    """估计卷积算子的Lipschitz常数。"""
    n_fft = 1 << int(np.ceil(np.log2(max(2, output_length))))
    return float(np.max(np.abs(np.fft.rfft(kernel, n=n_fft)) ** 2) + 1e-10)


def normalize_atom(atom: np.ndarray) -> np.ndarray:
    """施加零均值、单位范数与主峰正号约束。"""
    atom = np.asarray(atom, dtype=np.float64) - np.mean(atom)
    norm = np.linalg.norm(atom)
    if norm < 1e-12:
        raise RuntimeError("原子范数过小")
    atom /= norm
    if atom[np.argmax(np.abs(atom))] < 0:
        atom = -atom
    return atom


def measure_3db_bandwidth(
    atom: np.ndarray,
    sample_rate: int,
    n_fft: int = 16_384,
) -> tuple[float, float]:
    """测量绝对主谱峰所在连续负3 dB频带宽度。"""
    spectrum = np.abs(np.fft.rfft(atom, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    peak_index = int(np.argmax(spectrum))
    threshold = spectrum[peak_index] / np.sqrt(2.0)
    left = peak_index
    while left > 0 and spectrum[left - 1] >= threshold:
        left -= 1
    right = peak_index
    while right < len(spectrum) - 1 and spectrum[right + 1] >= threshold:
        right += 1

    def crossing(index_below: int, index_above: int) -> float:
        f0, f1 = frequencies[index_below], frequencies[index_above]
        m0, m1 = spectrum[index_below], spectrum[index_above]
        if abs(m1 - m0) < 1e-15:
            return float(f1)
        return float(f0 + (threshold - m0) * (f1 - f0) / (m1 - m0))

    left_frequency = 0.0 if left == 0 else crossing(left - 1, left)
    right_frequency = sample_rate / 2 if right == len(spectrum) - 1 else crossing(right + 1, right)
    return float(max(0.0, right_frequency - left_frequency)), float(frequencies[peak_index])


def hard_project_3db_bandwidth(atom: np.ndarray, config: Config = CFG) -> np.ndarray:
    """用高斯频谱包络将原子硬投影到指定负3 dB带宽。"""
    projected = normalize_atom(atom)
    bandwidth, _ = measure_3db_bandwidth(projected, config.sample_rate, config.bandwidth_nfft)
    if config.min_bandwidth_hz <= bandwidth <= config.max_bandwidth_hz:
        return projected
    target = float(np.clip(bandwidth, config.min_bandwidth_hz, config.max_bandwidth_hz))
    frequencies = np.fft.rfftfreq(config.bandwidth_nfft, d=1.0 / config.sample_rate)
    taper = windows.tukey(len(projected), alpha=0.20)
    for _ in range(12):
        spectrum = np.fft.rfft(projected, n=config.bandwidth_nfft)
        peak_index = int(np.argmax(np.abs(spectrum)))
        center_frequency = float(frequencies[peak_index])
        sigma = max(target / (2.0 * np.sqrt(np.log(2.0))), frequencies[1])
        envelope = np.exp(-0.5 * ((frequencies - center_frequency) / sigma) ** 2)
        candidate = np.fft.irfft(envelope * np.exp(1j * np.angle(spectrum)), n=config.bandwidth_nfft)
        projected = normalize_atom(candidate[: len(projected)] * taper)
        bandwidth, _ = measure_3db_bandwidth(projected, config.sample_rate, config.bandwidth_nfft)
        if config.min_bandwidth_hz <= bandwidth <= config.max_bandwidth_hz:
            return projected
        if bandwidth < config.min_bandwidth_hz:
            target *= 1.05 * config.min_bandwidth_hz / max(bandwidth, 1.0)
        else:
            target *= 0.95 * config.max_bandwidth_hz / bandwidth
        target = float(np.clip(target, 1_000.0, config.sample_rate * 0.80))
    raise RuntimeError(f"原子带宽硬约束失败：{bandwidth / 1e3:.2f} kHz")


def initialize_atom(signal: np.ndarray, config: Config = CFG) -> np.ndarray:
    """从全局绝对峰附近初始化受带宽约束的BCSC原子。"""
    atom_length = config.atom_length
    peak = int(np.argmax(np.abs(signal)))
    half = atom_length // 2
    padded = np.pad(signal, (half, half))
    atom = padded[peak : peak + atom_length].copy()
    atom = normalize_atom(atom * windows.tukey(atom_length, alpha=0.35))
    return hard_project_3db_bandwidth(atom, config)


def fista_sparse_code(
    signal: np.ndarray,
    atom: np.ndarray,
    lam: float,
    initial: np.ndarray | None = None,
    iterations: int = 120,
) -> np.ndarray:
    """固定原子时用FISTA求解L1卷积稀疏编码。"""
    activation_length = len(signal) - len(atom) + 1
    x = np.zeros(activation_length) if initial is None else initial.copy()
    z = x.copy()
    momentum = 1.0
    lipschitz = spectral_lipschitz(atom, len(signal))
    for _ in range(iterations):
        residual = reconstruct(atom, z, len(signal)) - signal
        gradient = adjoint_atom(residual, atom, activation_length)
        x_new = soft_threshold(z - gradient / lipschitz, lam / lipschitz)
        momentum_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        z = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
        x, momentum = x_new, momentum_new
    return x


def update_atom(
    signal: np.ndarray,
    activation: np.ndarray,
    atom: np.ndarray,
    config: Config = CFG,
) -> np.ndarray:
    """固定激活时执行带硬约束的投影梯度原子更新。"""
    if np.linalg.norm(activation) < 1e-12:
        return atom
    step = 0.8 / spectral_lipschitz(activation, len(signal))
    updated = atom.copy()
    taper = windows.tukey(len(atom), alpha=0.20)
    for _ in range(config.dictionary_iterations):
        residual = reconstruct(updated, activation, len(signal)) - signal
        gradient = adjoint_code(residual, activation, len(atom))
        updated = normalize_atom((updated - step * gradient) * taper)
        updated = hard_project_3db_bandwidth(updated, config)
    return updated


def shift_atom_and_code(atom: np.ndarray, activation: np.ndarray, config: Config = CFG) -> tuple[np.ndarray, np.ndarray]:
    """将原子绝对主峰移到中心并反向平移激活。"""
    center = len(atom) // 2
    shift = center - int(np.argmax(np.abs(atom)))
    shifted_atom = np.roll(atom, shift)
    if shift > 0:
        shifted_atom[:shift] = 0.0
    elif shift < 0:
        shifted_atom[shift:] = 0.0
    shifted_atom = hard_project_3db_bandwidth(normalize_atom(shifted_atom), config)
    shifted_code = np.roll(activation, -shift)
    if shift > 0:
        shifted_code[-shift:] = 0.0
    elif shift < 0:
        shifted_code[:-shift] = 0.0
    return shifted_atom, shifted_code


def bcsc(signal: np.ndarray, config: Config = CFG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行单原子盲卷积稀疏编码。"""
    signal = np.asarray(signal, dtype=np.float64)
    atom = initialize_atom(signal, config)
    activation = np.zeros(len(signal) - len(atom) + 1)
    for iteration in range(config.outer_iterations):
        max_correlation = np.max(np.abs(adjoint_atom(signal, atom, len(activation))))
        anneal = 1.8 - 0.8 * iteration / max(config.outer_iterations - 1, 1)
        activation = fista_sparse_code(
            signal,
            atom,
            config.lambda_ratio * anneal * max_correlation,
            activation,
            config.fista_iterations,
        )
        atom = update_atom(signal, activation, atom, config)
        atom, activation = shift_atom_and_code(atom, activation, config)
    max_correlation = np.max(np.abs(adjoint_atom(signal, atom, len(activation))))
    activation = fista_sparse_code(
        signal,
        atom,
        config.lambda_ratio * max_correlation,
        activation,
        2 * config.fista_iterations,
    )
    estimate = reconstruct(atom, activation, len(signal))
    return atom, activation, estimate


def extract_candidate_indices(activation: np.ndarray, config: Config = CFG) -> np.ndarray:
    """从稀疏激活提取满足相对峰值和最小间距的候选。"""
    magnitude = np.abs(activation)
    min_distance = max(1, int(round(config.min_path_delay_ms * 1e-3 * config.sample_rate)))
    peaks, _ = find_peaks(magnitude, height=config.peak_ratio * np.max(magnitude), distance=min_distance)
    strongest = int(np.argmax(magnitude))
    if strongest not in peaks:
        peaks = np.sort(np.append(peaks, strongest))
    return np.asarray(peaks, dtype=int)


def moving_block_bootstrap(residual: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """对残差执行循环移动块Bootstrap。"""
    starts = rng.integers(0, len(residual), size=int(np.ceil(len(residual) / block_length)))
    indices = np.concatenate([(start + np.arange(block_length)) % len(residual) for start in starts])
    return residual[indices[: len(residual)]]


def _path_component(atom: np.ndarray, activation_index: int, coefficient: float, signal_length: int) -> np.ndarray:
    """构建单条BCSC路径在完整波形中的分量。"""
    component = np.zeros(signal_length, dtype=np.float64)
    start = int(activation_index)
    stop = min(signal_length, start + len(atom))
    if start < signal_length and stop > 0:
        atom_start = max(0, -start)
        target_start = max(0, start)
        component[target_start:stop] = coefficient * atom[atom_start : atom_start + stop - target_start]
    return component


def process_bcsc_signal(signal: np.ndarray, seed: int, config: Config = CFG) -> dict[str, Any]:
    """对单条波形拟合BCSC并用8次扰动复核路径稳定性。"""
    started = time.perf_counter()
    try:
        atom, activation, estimate = bcsc(signal, config)
        residual = np.asarray(signal, dtype=np.float64) - estimate
        candidates = extract_candidate_indices(activation, config)
        direct_index = int(np.argmax(np.abs(activation)))
        min_delay = int(round(config.min_path_delay_ms * 1e-3 * config.sample_rate))
        max_delay = int(round(config.max_path_delay_ms * 1e-3 * config.sample_rate))
        reference_multipaths = [
            int(index)
            for index in candidates
            if direct_index + min_delay <= int(index) <= direct_index + max_delay
        ]
        reference_indices = [direct_index] + reference_multipaths
        matched: dict[int, list[tuple[int, float]]] = {index: [] for index in reference_indices}
        rng = np.random.default_rng(seed)
        block_length = max(1, int(round(config.residual_block_ms * 1e-3 * config.sample_rate)))
        match_tolerance = int(round(config.match_tolerance_ms * 1e-3 * config.sample_rate))
        max_correlation = np.max(np.abs(adjoint_atom(signal, atom, len(activation))))
        for _ in range(config.bootstrap_runs):
            synthetic = estimate + moving_block_bootstrap(residual, block_length, rng)
            lambda_scale = rng.uniform(1.0 - config.lambda_perturbation, 1.0 + config.lambda_perturbation)
            code = fista_sparse_code(
                synthetic,
                atom,
                config.lambda_ratio * lambda_scale * max_correlation,
                activation,
                config.fista_iterations,
            )
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
            stable_multipaths.append(
                {
                    "reference_index": float(reference_index),
                    "median_index": float(np.median(indices)),
                    "ci_low_index": float(np.percentile(indices, 2.5)),
                    "ci_high_index": float(np.percentile(indices, 97.5)),
                    "median_coefficient": float(np.median(coefficients)),
                    "confidence": len(records) / config.bootstrap_runs,
                    "relative_amplitude": float(np.median(np.abs(coefficients)) / direct_abs),
                }
            )
        stable_multipaths.sort(key=lambda row: (-row["relative_amplitude"], row["median_index"]))
        selected = stable_multipaths[0] if stable_multipaths else None
        multipath_index = int(round(selected["median_index"])) if selected else -1
        multipath_coefficient = float(selected["median_coefficient"]) if selected else 0.0
        multipath_confidence = float(selected["confidence"]) if selected else 0.0
        delay_ms = (
            (multipath_index - direct_index) / config.sample_rate * 1e3 if selected else 0.0
        )
        delay_ci_low_ms = (
            (selected["ci_low_index"] - direct_index) / config.sample_rate * 1e3 if selected else 0.0
        )
        delay_ci_high_ms = (
            (selected["ci_high_index"] - direct_index) / config.sample_rate * 1e3 if selected else 0.0
        )
        bandwidth_hz, center_frequency_hz = measure_3db_bandwidth(atom, config.sample_rate, config.bandwidth_nfft)
        explained_variance = 1.0 - np.sum(residual**2) / (np.sum(np.asarray(signal) ** 2) + 1e-12)
        status = "ok" if len(direct_records) >= config.bootstrap_required else "unstable_direct"
        return {
            "status": status,
            "error": "",
            "atom": atom.astype(np.float32),
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
    except Exception as error:  # 每条独立记录失败时保留错误并继续全量运行。
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "atom": np.zeros(config.atom_length, dtype=np.float32),
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


def _empty_bcsc_arrays(sample_count: int, config: Config = CFG) -> dict[str, np.ndarray]:
    """初始化可断点保存的BCSC结果数组。"""
    return {
        "completed": np.zeros(sample_count, dtype=bool),
        "status_code": np.full(sample_count, -1, dtype=np.int8),
        "atoms": np.zeros((sample_count, config.atom_length), dtype=np.float32),
        "direct_index": np.full(sample_count, -1, dtype=np.int32),
        "direct_coefficient": np.zeros(sample_count, dtype=np.float32),
        "direct_confidence": np.zeros(sample_count, dtype=np.float32),
        "multipath_index": np.full(sample_count, -1, dtype=np.int32),
        "multipath_coefficient": np.zeros(sample_count, dtype=np.float32),
        "multipath_confidence": np.zeros(sample_count, dtype=np.float32),
        "delay_ms": np.zeros(sample_count, dtype=np.float32),
        "delay_ci_low_ms": np.zeros(sample_count, dtype=np.float32),
        "delay_ci_high_ms": np.zeros(sample_count, dtype=np.float32),
        "explained_variance": np.zeros(sample_count, dtype=np.float32),
        "residual_energy": np.zeros(sample_count, dtype=np.float32),
        "bandwidth_hz": np.zeros(sample_count, dtype=np.float32),
        "center_frequency_hz": np.zeros(sample_count, dtype=np.float32),
        "candidate_count": np.zeros(sample_count, dtype=np.int16),
        "stable_multipath_count": np.zeros(sample_count, dtype=np.int16),
        "elapsed_s": np.zeros(sample_count, dtype=np.float32),
        "errors": np.full(sample_count, "", dtype="U256"),
    }


def _save_bcsc_cache(path: Path, arrays: dict[str, np.ndarray], config: Config = CFG) -> None:
    """原子写入BCSC断点缓存。"""
    temp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_path, config_hash=np.asarray(config.config_hash), **arrays)
    temp_path.replace(path)


def _process_bcsc_payload(payload: tuple[int, np.ndarray, int, Config]) -> tuple[int, dict[str, Any]]:
    """在独立进程中处理一个BCSC样本。"""
    index, signal, seed, config = payload
    return index, process_bcsc_signal(np.asarray(signal, dtype=np.float64), seed, config)


def run_bcsc_batch(
    waveforms: np.ndarray,
    manifest: pd.DataFrame,
    config: Config = CFG,
    force: bool = False,
    limit: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """批量运行BCSC，逐条显示总体进度并定期保存断点。"""
    paths = ensure_output_dirs(config)
    cache_path = paths["cache"] / "bcsc_results.npz"
    table_path = paths["root"] / "bcsc_paths.csv"
    arrays = _empty_bcsc_arrays(len(manifest), config)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_hash = str(cached["config_hash"].item()) if "config_hash" in cached else ""
            if cached_hash == config.config_hash:
                for key in arrays:
                    if key in cached and cached[key].shape == arrays[key].shape:
                        arrays[key] = cached[key].copy()
    indices = np.where(~arrays["completed"])[0]
    if limit is not None:
        indices = indices[:limit]
    progress = tqdm(total=len(indices), desc="总体进度 2/4：BCSC与8次稳定性复核", unit="条")

    payloads = (
        (int(index), np.asarray(waveforms[index], dtype=np.float32), config.random_seed + int(index), config)
        for index in indices
    )
    with ProcessPoolExecutor(max_workers=config.bcsc_workers) as executor:
        result_iterator = executor.map(_process_bcsc_payload, payloads, chunksize=1)
        for completed_in_run, (index, result) in enumerate(result_iterator, start=1):
            arrays["completed"][index] = True
            arrays["status_code"][index] = {"ok": 1, "unstable_direct": 0, "failed": -2}[result["status"]]
            arrays["atoms"][index] = result["atom"]
            for key in (
                "direct_index", "direct_coefficient", "direct_confidence", "multipath_index",
                "multipath_coefficient", "multipath_confidence", "delay_ms", "delay_ci_low_ms",
                "delay_ci_high_ms", "explained_variance", "residual_energy", "bandwidth_hz",
                "center_frequency_hz", "candidate_count", "stable_multipath_count", "elapsed_s",
            ):
                arrays[key][index] = result[key]
            arrays["errors"][index] = result["error"][:255]
            progress.update(1)
            progress.set_postfix(
                ok=int(np.sum(arrays["status_code"] == 1)),
                multipath=int(np.sum(arrays["multipath_index"] >= 0)),
                last_s=f"{result['elapsed_s']:.1f}",
            )
            if completed_in_run % config.bcsc_checkpoint_interval == 0:
                _save_bcsc_cache(cache_path, arrays, config)
    progress.close()
    _save_bcsc_cache(cache_path, arrays, config)

    status_text = np.where(
        arrays["status_code"] == 1,
        "ok",
        np.where(arrays["status_code"] == 0, "unstable_direct", np.where(arrays["completed"], "failed", "pending")),
    )
    table = manifest[["sample_id", "label", "ori_file_num", "test_fold"]].copy()
    table["status"] = status_text
    for key in (
        "direct_index", "direct_coefficient", "direct_confidence", "multipath_index",
        "multipath_coefficient", "multipath_confidence", "delay_ms", "delay_ci_low_ms",
        "delay_ci_high_ms", "explained_variance", "residual_energy", "bandwidth_hz",
        "center_frequency_hz", "candidate_count", "stable_multipath_count", "elapsed_s", "errors",
    ):
        table[key] = arrays[key]
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    return arrays, table


def _crop_center(signal: np.ndarray, center: int, length: int) -> np.ndarray:
    """围绕中心点裁剪定长波形，越界处补零。"""
    half = length // 2
    padded = np.pad(signal, (half, half))
    start = int(center)
    return padded[start : start + length].astype(np.float64, copy=False)


def path_features(signal: np.ndarray, sample_rate: int = CFG.sample_rate) -> tuple[np.ndarray, list[str]]:
    """提取单条路径的时域、频域、分带能量和倒谱特征。"""
    signal = np.asarray(signal, dtype=np.float64)
    envelope = np.abs(hilbert(signal))
    envelope_peak = float(np.max(envelope)) + 1e-12
    normalized_envelope = envelope / envelope_peak
    active10 = np.where(normalized_envelope >= 0.10)[0]
    active50 = np.where(normalized_envelope >= 0.50)[0]
    active90 = np.where(normalized_envelope >= 0.90)[0]

    def duration(indices: np.ndarray) -> float:
        return float((indices[-1] - indices[0] + 1) / sample_rate * 1e6) if len(indices) else 0.0

    peak_index = int(np.argmax(envelope))
    rise_us = peak_index / sample_rate * 1e6
    fall_us = (len(signal) - 1 - peak_index) / sample_rate * 1e6
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / (rms + 1e-12)
    signal_kurtosis = float(kurtosis(signal, fisher=False, bias=False)) if np.std(signal) > 1e-12 else 0.0
    zero_crossing_rate = float(np.mean(np.diff(np.signbit(signal)) != 0))

    tapered = signal * windows.hann(len(signal), sym=False)
    n_fft = 2_048
    spectrum = np.abs(np.fft.rfft(tapered, n=n_fft)) ** 2
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band_mask = (frequencies >= CFG.lowcut_hz) & (frequencies <= CFG.highcut_hz)
    frequencies_band = frequencies[band_mask]
    power = spectrum[band_mask] + 1e-18
    probability = power / np.sum(power)
    peak_frequency = float(frequencies_band[int(np.argmax(power))])
    centroid = float(np.sum(frequencies_band * probability))
    cumulative = np.cumsum(probability)

    def rolloff(quantile: float) -> float:
        return float(frequencies_band[min(np.searchsorted(cumulative, quantile), len(frequencies_band) - 1)])

    peak_power = float(np.max(power))

    def bandwidth_at(db: float) -> float:
        mask = power >= peak_power * 10 ** (-db / 10)
        selected = frequencies_band[mask]
        return float(selected[-1] - selected[0]) if len(selected) else 0.0

    spectral_entropy = float(-np.sum(probability * np.log(probability + 1e-18)) / np.log(len(probability)))
    spectral_flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    subbands = [(20_000, 50_000), (50_000, 90_000), (90_000, 130_000), (130_000, 190_000)]
    subband_energy = []
    for low, high in subbands:
        mask = (frequencies_band >= low) & (frequencies_band < high)
        subband_energy.append(float(np.sum(power[mask]) / np.sum(power)))
    log_spectrum = np.log(power)
    cepstral = dct(log_spectrum, type=2, norm="ortho")[1:7]
    values = np.asarray(
        [
            duration(active10), duration(active50), duration(active90), rise_us, fall_us,
            rms, peak, crest, signal_kurtosis, zero_crossing_rate,
            peak_frequency / 1e3, centroid / 1e3, bandwidth_at(3) / 1e3, bandwidth_at(10) / 1e3,
            rolloff(0.25) / 1e3, rolloff(0.50) / 1e3, rolloff(0.75) / 1e3, rolloff(0.95) / 1e3,
            spectral_entropy, spectral_flatness,
            *subband_energy,
            *cepstral.tolist(),
        ],
        dtype=np.float32,
    )
    names = [
        "duration_10_us", "duration_50_us", "duration_90_us", "rise_us", "fall_us",
        "rms", "peak", "crest_factor", "kurtosis", "zero_crossing_rate",
        "peak_frequency_khz", "spectral_centroid_khz", "bandwidth_3db_khz", "bandwidth_10db_khz",
        "rolloff_25_khz", "rolloff_50_khz", "rolloff_75_khz", "rolloff_95_khz",
        "spectral_entropy", "spectral_flatness",
        "energy_20_50", "energy_50_90", "energy_90_130", "energy_130_190",
        *[f"cepstral_{index}" for index in range(1, 7)],
    ]
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), names


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """返回两个向量的余弦相似度。"""
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-12))


def build_features(
    waveforms: np.ndarray,
    manifest: pd.DataFrame,
    bcsc_arrays: dict[str, np.ndarray],
    config: Config = CFG,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """构建直达径、多径及路径关系显式特征。"""
    paths = ensure_output_dirs(config)
    cache_path = paths["cache"] / "explicit_features.npz"
    table_path = paths["root"] / "explicit_features.csv"
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["config_hash"].item()) == config.config_hash:
            names = [str(value) for value in cached["feature_names"]]
            return cached["direct"], cached["multipath"], cached["relation"], names

    direct_rows: list[np.ndarray] = []
    multipath_rows: list[np.ndarray] = []
    relation_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for index in tqdm(range(len(manifest)), desc="总体进度 3/4：路径特征提取", unit="条"):
        signal = np.asarray(waveforms[index], dtype=np.float64)
        atom = np.asarray(bcsc_arrays["atoms"][index], dtype=np.float64)
        direct_index = int(bcsc_arrays["direct_index"][index])
        multipath_index = int(bcsc_arrays["multipath_index"][index])
        direct_valid = direct_index >= 0 and bcsc_arrays["status_code"][index] == 1
        multipath_present = multipath_index >= 0 and direct_valid
        if not direct_valid:
            if feature_names is None:
                _, feature_names = path_features(np.zeros(config.local_window_samples))
            direct_values = np.zeros(len(feature_names), dtype=np.float32)
            multipath_values = np.zeros(len(feature_names), dtype=np.float32)
            relation_values = np.zeros(len(RELATION_FEATURE_NAMES), dtype=np.float32)
        else:
            direct_component = _path_component(
                atom, direct_index, float(bcsc_arrays["direct_coefficient"][index]), len(signal)
            )
            multipath_component = (
                _path_component(atom, multipath_index, float(bcsc_arrays["multipath_coefficient"][index]), len(signal))
                if multipath_present else np.zeros_like(signal)
            )
            direct_center = direct_index + len(atom) // 2
            multipath_center = multipath_index + len(atom) // 2 if multipath_present else -1
            direct_local = _crop_center(signal - multipath_component, direct_center, config.local_window_samples)
            direct_values, feature_names = path_features(direct_local, config.sample_rate)
            if multipath_present:
                multipath_local = _crop_center(signal - direct_component, multipath_center, config.local_window_samples)
                multipath_values, _ = path_features(multipath_local, config.sample_rate)
                direct_reference = _crop_center(signal, direct_center, config.local_window_samples)
                multipath_reference = _crop_center(signal, multipath_center, config.local_window_samples)
                ncc = _cosine_similarity(direct_reference, multipath_reference)
                spectrum_direct = np.abs(np.fft.rfft(direct_reference * windows.hann(len(direct_reference))))
                spectrum_multipath = np.abs(np.fft.rfft(multipath_reference * windows.hann(len(multipath_reference))))
                spectral_cosine = _cosine_similarity(spectrum_direct, spectrum_multipath)
                coherence = float(np.corrcoef(direct_reference, multipath_reference)[0, 1])
                direct_coefficient = float(bcsc_arrays["direct_coefficient"][index])
                multipath_coefficient = float(bcsc_arrays["multipath_coefficient"][index])
                relative_amplitude = abs(multipath_coefficient) / (abs(direct_coefficient) + 1e-12)
                attenuation_db = 20 * np.log10(relative_amplitude + 1e-12)
                polarity = float(np.sign(direct_coefficient * multipath_coefficient))
                energy_ratio = float(np.sum(multipath_component**2) / (np.sum(direct_component**2) + 1e-12))
                overlap_samples = max(0, len(atom) - abs(multipath_index - direct_index))
                overlap_ratio = overlap_samples / len(atom)
                residual_without_mp = signal - direct_component
                residual_with_paths = signal - direct_component - multipath_component
                residual_energy_change = float(
                    (np.mean(residual_without_mp**2) - np.mean(residual_with_paths**2))
                    / (np.mean(residual_without_mp**2) + 1e-12)
                )
            else:
                multipath_values = np.zeros_like(direct_values)
                ncc = spectral_cosine = coherence = relative_amplitude = attenuation_db = 0.0
                polarity = energy_ratio = overlap_ratio = residual_energy_change = 0.0
            relation_values = np.asarray(
                [
                    float(multipath_present),
                    float(bcsc_arrays["direct_confidence"][index]),
                    float(bcsc_arrays["multipath_confidence"][index]),
                    float(bcsc_arrays["delay_ms"][index]),
                    float(bcsc_arrays["delay_ci_high_ms"][index] - bcsc_arrays["delay_ci_low_ms"][index]),
                    relative_amplitude,
                    attenuation_db,
                    polarity,
                    energy_ratio,
                    ncc,
                    spectral_cosine,
                    coherence,
                    overlap_ratio,
                    float(bcsc_arrays["explained_variance"][index]),
                    residual_energy_change,
                ],
                dtype=np.float32,
            )
        direct_rows.append(np.nan_to_num(direct_values))
        multipath_rows.append(np.nan_to_num(multipath_values))
        relation_rows.append(np.nan_to_num(relation_values))
    direct = np.asarray(direct_rows, dtype=np.float32)
    multipath = np.asarray(multipath_rows, dtype=np.float32)
    relation = np.asarray(relation_rows, dtype=np.float32)
    assert feature_names is not None
    combined_names = (
        [f"direct__{name}" for name in feature_names]
        + [f"multipath__{name}" for name in feature_names]
        + [f"relation__{name}" for name in RELATION_FEATURE_NAMES]
    )
    np.savez_compressed(
        cache_path,
        config_hash=np.asarray(config.config_hash),
        direct=direct,
        multipath=multipath,
        relation=relation,
        feature_names=np.asarray(combined_names),
    )
    table = manifest[["sample_id", "label", "ori_file_num", "test_fold"]].copy()
    feature_matrix = np.concatenate([direct, multipath, relation], axis=1)
    for column_index, name in enumerate(combined_names):
        table[name] = feature_matrix[:, column_index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        table.to_csv(table_path, index=False, encoding="utf-8-sig")
    return direct, multipath, relation, combined_names


class ArrayDataset(Dataset):
    """将NumPy数组包装为PyTorch数据集。"""

    def __init__(self, values: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.values = values
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (
            torch.as_tensor(np.asarray(self.values[index]), dtype=torch.float32),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class TransformerCore(nn.Module):
    """三组模型共享的Transformer编码核心。"""

    def __init__(self, d_model: int = 64, heads: int = 4, layers: int = 2, feedforward: int = 128):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.10), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        return self.head(self.norm(encoded[:, 0])).squeeze(-1)


def sinusoidal_positions(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    """生成确定性的正弦位置编码。"""
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dimension, 2, device=device) * (-math.log(10_000.0) / dimension))
    encoding = torch.zeros(length, dimension, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class WaveformTransformer(nn.Module):
    """以波形补丁作为token的端到端Transformer。"""

    def __init__(self, patch_size: int = 32, stride: int = 16, d_model: int = 64):
        super().__init__()
        self.patch = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)
        self.adapter = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.core = TransformerCore(d_model=d_model)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(waveform.unsqueeze(1)).transpose(1, 2)
        tokens = self.adapter(tokens)
        cls = self.cls.expand(len(waveform), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + sinusoidal_positions(tokens.shape[1], tokens.shape[2], tokens.device).unsqueeze(0)
        return self.core(tokens)


class FeatureTransformer(nn.Module):
    """将每个连续特征编码为语义token的Transformer。"""

    def __init__(self, feature_count: int, direct_count: int, d_model: int = 64):
        super().__init__()
        self.feature_count = feature_count
        self.direct_count = direct_count
        self.scale = nn.Parameter(torch.empty(feature_count, d_model))
        self.bias = nn.Parameter(torch.empty(feature_count, d_model))
        self.feature_embedding = nn.Parameter(torch.empty(feature_count, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.core = TransformerCore(d_model=d_model)
        nn.init.normal_(self.scale, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, features: torch.Tensor, direct_only: bool = False) -> torch.Tensor:
        tokens = features.unsqueeze(-1) * self.scale.unsqueeze(0) + self.bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        cls = self.cls.expand(len(features), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        padding_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=features.device)
        if direct_only:
            padding_mask[:, 1 + self.direct_count :] = True
        return self.core(tokens, padding_mask)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def fit_feature_scaler(features: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """仅使用训练折拟合稳健中位数和标准差。"""
    train = features[train_indices]
    center = np.nanmedian(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def apply_feature_scaler(features: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """应用训练折特征标准化并替换非有限值。"""
    return np.nan_to_num((features - center) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device, direct_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回索引、标签和预测概率。"""
    model.eval()
    all_indices, all_labels, all_probabilities = [], [], []
    for values, labels, indices in loader:
        values = values.to(device, non_blocking=True)
        logits = model(values, direct_only=direct_only) if isinstance(model, FeatureTransformer) else model(values)
        all_indices.append(indices.numpy())
        all_labels.append(labels.numpy())
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_labels), np.concatenate(all_probabilities)


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """在验证集选择平衡准确率最高的分类阈值。"""
    candidates = np.unique(np.concatenate([[0.0, 0.5, 1.0], probabilities]))
    scores = [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """计算预注册的二分类指标。"""
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_one_model(
    model: nn.Module,
    values: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    model_path: Path,
    config: Config = CFG,
    direct_only: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """训练单折模型并返回测试预测。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        "train": DataLoader(ArrayDataset(values, labels, train_indices), batch_size=config.batch_size, shuffle=True, num_workers=0),
        "validation": DataLoader(ArrayDataset(values, labels, validation_indices), batch_size=config.batch_size * 2, shuffle=False, num_workers=0),
        "test": DataLoader(ArrayDataset(values, labels, test_indices), batch_size=config.batch_size * 2, shuffle=False, num_workers=0),
    }
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -np.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.max_epochs):
        model.train()
        losses = []
        for batch_values, batch_labels, _ in loaders["train"]:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch_values, direct_only=direct_only) if isinstance(model, FeatureTransformer) else model(batch_values)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        _, validation_labels, validation_probabilities = predict_model(model, loaders["validation"], device, direct_only)
        validation_score = roc_auc_score(validation_labels, validation_probabilities)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_auroc": float(validation_score)})
        if validation_score > best_score + 1e-4:
            best_score = float(validation_score)
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    _, validation_labels, validation_probabilities = predict_model(model, loaders["validation"], device, direct_only)
    threshold = choose_threshold(validation_labels, validation_probabilities)
    test_order, test_labels, test_probabilities = predict_model(model, loaders["test"], device, direct_only)
    metrics = binary_metrics(test_labels, test_probabilities, threshold)
    metrics.update({"best_epoch": best_epoch, "validation_auroc": best_score, "parameters": count_parameters(model), "history": history})
    predictions = pd.DataFrame(
        {"sample_index": test_order, "label": test_labels.astype(int), "probability": test_probabilities, "threshold": threshold}
    )
    return metrics, predictions


def _build_model(model_name: str, feature_count: int, direct_count: int) -> nn.Module:
    """按实验名称构建对应Transformer。"""
    if model_name == "waveform":
        return WaveformTransformer()
    return FeatureTransformer(feature_count=feature_count, direct_count=direct_count)


def train_cross_validated_models(
    waveforms: np.ndarray,
    manifest: pd.DataFrame,
    direct: np.ndarray,
    multipath: np.ndarray,
    relation: np.ndarray,
    config: Config = CFG,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在来源隔离五折上训练三组Transformer。"""
    paths = ensure_output_dirs(config)
    metrics_path = paths["root"] / "fold_metrics.csv"
    predictions_path = paths["root"] / "test_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force:
        metrics = pd.read_csv(metrics_path)
        predictions = pd.read_csv(predictions_path)
        if set(metrics["model"]) == {"waveform", "direct_multipath", "direct_only"} and len(metrics) == 15:
            return metrics, predictions

    labels = manifest["label"].to_numpy(dtype=np.float32)
    test_folds = manifest["test_fold"].to_numpy(dtype=int)
    raw_features = np.concatenate([direct, multipath, relation], axis=1)
    direct_count = direct.shape[1]
    model_names = ["waveform", "direct_multipath", "direct_only"]
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    overall = tqdm(total=config.n_splits * len(model_names), desc="总体进度 4/4：三模型五折训练", unit="模型折")
    for test_fold in range(config.n_splits):
        validation_fold = (test_fold + 1) % config.n_splits
        test_indices = np.where(test_folds == test_fold)[0]
        validation_indices = np.where(test_folds == validation_fold)[0]
        train_indices = np.where((test_folds != test_fold) & (test_folds != validation_fold))[0]
        center, scale = fit_feature_scaler(raw_features, train_indices)
        scaled_features = apply_feature_scaler(raw_features, center, scale)
        for model_name in model_names:
            seed_everything(config.random_seed + 100 * test_fold + model_names.index(model_name))
            model_path = paths["models"] / f"{model_name}_fold_{test_fold}.pt"
            values = np.asarray(waveforms) if model_name == "waveform" else scaled_features
            model = _build_model(model_name, scaled_features.shape[1], direct_count)
            metrics, predictions = train_one_model(
                model,
                values,
                labels,
                train_indices,
                validation_indices,
                test_indices,
                model_path,
                config,
                direct_only=model_name == "direct_only",
            )
            serializable_metrics = {key: value for key, value in metrics.items() if key != "history"}
            serializable_metrics.update({"model": model_name, "test_fold": test_fold, "validation_fold": validation_fold})
            all_metrics.append(serializable_metrics)
            predictions["model"] = model_name
            predictions["test_fold"] = test_fold
            all_predictions.append(predictions)
            json_dump(paths["logs"] / f"{model_name}_fold_{test_fold}_history.json", metrics["history"])
            overall.update(1)
            overall.set_postfix(model=model_name, fold=test_fold, auroc=f"{metrics['auroc']:.3f}")
    overall.close()
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df = predictions_df.merge(
        manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num"]],
        on="sample_index",
        how="left",
    )
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    return metrics_df, predictions_df


def group_bootstrap_metric(
    frame: pd.DataFrame,
    metric_name: str,
    rng: np.random.Generator,
    runs: int,
) -> tuple[float, float]:
    """按原始录音组重采样计算指标95%区间。"""
    groups = frame["ori_file_num"].unique()
    values = []
    for _ in range(runs):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat([frame[frame["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
        if sampled["label"].nunique() < 2:
            continue
        values.append(frame_metric(sampled, metric_name))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def frame_metric(frame: pd.DataFrame, metric_name: str) -> float:
    """使用每个测试折自己的验证阈值计算合并预测指标。"""
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    if metric_name == "auroc":
        return float(roc_auc_score(labels, probabilities))
    if metric_name == "auprc":
        return float(average_precision_score(labels, probabilities))
    predictions = probabilities >= frame["threshold"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    if metric_name == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, predictions))
    if metric_name == "f1":
        return float(f1_score(labels, predictions, zero_division=0))
    if metric_name == "sensitivity":
        return float(tp / max(tp + fn, 1))
    if metric_name == "specificity":
        return float(tn / max(tn + fp, 1))
    raise KeyError(metric_name)


def paired_model_deltas(
    predictions: pd.DataFrame,
    config: Config = CFG,
) -> pd.DataFrame:
    """按录音组Bootstrap计算直达+多径相对仅直达径的配对指标差。"""
    first = predictions[predictions["model"] == "direct_multipath"].copy()
    second = predictions[predictions["model"] == "direct_only"].copy()
    merged = first.merge(
        second[["sample_index", "probability", "threshold"]],
        on="sample_index",
        suffixes=("_direct_multipath", "_direct_only"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(config.random_seed + 1001)
    groups = merged["ori_file_num"].unique()
    rows = []
    for metric_name in ("auroc", "auprc", "balanced_accuracy"):
        first_frame = pd.DataFrame(
            {
                "label": merged["label"],
                "probability": merged["probability_direct_multipath"],
                "threshold": merged["threshold_direct_multipath"],
            }
        )
        second_frame = pd.DataFrame(
            {
                "label": merged["label"],
                "probability": merged["probability_direct_only"],
                "threshold": merged["threshold_direct_only"],
            }
        )
        point = frame_metric(first_frame, metric_name) - frame_metric(second_frame, metric_name)
        bootstrap_values = []
        for _ in range(config.group_bootstrap_runs):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            sampled = pd.concat([merged[merged["ori_file_num"] == group] for group in sampled_groups], ignore_index=True)
            if sampled["label"].nunique() < 2:
                continue
            sampled_first = pd.DataFrame(
                {
                    "label": sampled["label"],
                    "probability": sampled["probability_direct_multipath"],
                    "threshold": sampled["threshold_direct_multipath"],
                }
            )
            sampled_second = pd.DataFrame(
                {
                    "label": sampled["label"],
                    "probability": sampled["probability_direct_only"],
                    "threshold": sampled["threshold_direct_only"],
                }
            )
            bootstrap_values.append(frame_metric(sampled_first, metric_name) - frame_metric(sampled_second, metric_name))
        rows.append(
            {
                "metric": metric_name,
                "delta": point,
                "ci_low": float(np.percentile(bootstrap_values, 2.5)),
                "ci_high": float(np.percentile(bootstrap_values, 97.5)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(config.output_dir / "paired_model_deltas.csv", index=False, encoding="utf-8-sig")
    return result


def summarize_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Config = CFG,
) -> pd.DataFrame:
    """汇总五折均值和录音组Bootstrap置信区间。"""
    rows = []
    rng = np.random.default_rng(config.random_seed + 99)
    for model_name, model_predictions in predictions.groupby("model"):
        model_metrics = metrics[metrics["model"] == model_name]
        row: dict[str, Any] = {"model": model_name}
        for metric_name in ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity"):
            row[f"{metric_name}_fold_mean"] = float(model_metrics[metric_name].mean())
            low, high = group_bootstrap_metric(model_predictions, metric_name, rng, config.group_bootstrap_runs)
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(config.output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    paired_model_deltas(predictions, config)
    return summary


def synthetic_bcsc_test(config: Config = CFG) -> dict[str, Any]:
    """用已知直达径和延迟副本验证BCSC路径定位误差。"""
    rng = np.random.default_rng(config.random_seed)
    atom = windows.gaussian(config.atom_length, std=config.atom_length / 7)
    atom *= np.cos(2 * np.pi * 90_000 * np.arange(config.atom_length) / config.sample_rate)
    atom = normalize_atom(atom)
    signal = np.zeros(config.sample_count)
    direct_index = 2_700
    delay_samples = int(round(0.35e-3 * config.sample_rate))
    signal[direct_index : direct_index + len(atom)] += atom
    signal[direct_index + delay_samples : direct_index + delay_samples + len(atom)] += 0.45 * atom
    signal += rng.normal(scale=0.004, size=len(signal))
    processed = process_bcsc_signal(signal, config.random_seed, config)
    error_ms = abs(processed["delay_ms"] - delay_samples / config.sample_rate * 1e3)
    result = {
        "status": processed["status"],
        "true_delay_ms": delay_samples / config.sample_rate * 1e3,
        "estimated_delay_ms": processed["delay_ms"],
        "absolute_error_ms": error_ms,
        "passed": bool(processed["multipath_index"] >= 0 and error_ms <= config.match_tolerance_ms),
    }
    json_dump(config.output_dir / "synthetic_bcsc_test.json", result)
    return result


def _counterfactual_waveforms(
    waveforms: np.ndarray,
    bcsc_arrays: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    config: Config = CFG,
) -> dict[str, np.ndarray]:
    """构造波形模型的直达、去多径、残差及背景交换反事实。"""
    full = np.asarray(waveforms, dtype=np.float32)
    direct_only = np.zeros_like(full)
    remove_multipath = full.copy()
    residual_only = np.zeros_like(full)
    structured = np.zeros_like(full)
    residuals = np.zeros_like(full)
    for index in range(len(manifest)):
        atom = np.asarray(bcsc_arrays["atoms"][index], dtype=np.float64)
        direct_index = int(bcsc_arrays["direct_index"][index])
        if direct_index < 0:
            residuals[index] = full[index]
            residual_only[index] = full[index]
            continue
        direct_component = _path_component(atom, direct_index, float(bcsc_arrays["direct_coefficient"][index]), config.sample_count)
        multipath_index = int(bcsc_arrays["multipath_index"][index])
        multipath_component = (
            _path_component(atom, multipath_index, float(bcsc_arrays["multipath_coefficient"][index]), config.sample_count)
            if multipath_index >= 0 else np.zeros(config.sample_count)
        )
        direct_only[index] = direct_component
        remove_multipath[index] = full[index] - multipath_component
        structured[index] = direct_component + multipath_component
        residuals[index] = full[index] - structured[index]
        residual_only[index] = residuals[index]
    swapped = structured.copy()
    rng = np.random.default_rng(config.random_seed + 123)
    labels = manifest["label"].to_numpy()
    groups = manifest["ori_file_num"].to_numpy()
    folds = manifest["test_fold"].to_numpy()
    for index in range(len(manifest)):
        candidates = np.where((labels != labels[index]) & (groups == groups[index]) & (folds == folds[index]))[0]
        if len(candidates) == 0:
            candidates = np.where((labels != labels[index]) & (folds == folds[index]))[0]
        donor = int(rng.choice(candidates))
        donor_residual = residuals[donor]
        target_rms = np.sqrt(np.mean(residuals[index] ** 2))
        donor_rms = np.sqrt(np.mean(donor_residual**2))
        swapped[index] += donor_residual * (target_rms / (donor_rms + 1e-12))
    return {
        "full": full,
        "direct_only": direct_only,
        "remove_multipath": remove_multipath,
        "residual_only": residual_only,
        "background_swapped": swapped,
    }


def run_counterfactual_tests(
    waveforms: np.ndarray,
    manifest: pd.DataFrame,
    bcsc_arrays: dict[str, np.ndarray],
    config: Config = CFG,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用已训练波形Transformer评估反事实输入与梯度归因。"""
    paths = ensure_output_dirs(config)
    metrics_path = paths["root"] / "counterfactual_metrics.csv"
    predictions_path = paths["root"] / "counterfactual_predictions.csv"
    attribution_path = paths["root"] / "waveform_attribution.csv"
    if metrics_path.exists() and predictions_path.exists() and attribution_path.exists() and not force:
        return pd.read_csv(metrics_path), pd.read_csv(predictions_path)
    variants = _counterfactual_waveforms(waveforms, bcsc_arrays, manifest, config)
    labels = manifest["label"].to_numpy(dtype=np.float32)
    folds = manifest["test_fold"].to_numpy(dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    trained_metrics = pd.read_csv(paths["root"] / "fold_metrics.csv")
    for test_fold in tqdm(range(config.n_splits), desc="捷径反事实检验", unit="折"):
        test_indices = np.where(folds == test_fold)[0]
        threshold = float(
            trained_metrics[
                (trained_metrics["model"] == "waveform") & (trained_metrics["test_fold"] == test_fold)
            ]["threshold"].iloc[0]
        )
        model = WaveformTransformer().to(device)
        model.load_state_dict(torch.load(paths["models"] / f"waveform_fold_{test_fold}.pt", map_location=device, weights_only=True))
        baseline_frame: pd.DataFrame | None = None
        for variant_name, variant_values in variants.items():
            loader = DataLoader(ArrayDataset(variant_values, labels, test_indices), batch_size=config.batch_size * 2, shuffle=False)
            order, variant_labels, probabilities = predict_model(model, loader, device)
            frame = pd.DataFrame({"sample_index": order, "label": variant_labels.astype(int), "probability": probabilities})
            frame["variant"] = variant_name
            frame["test_fold"] = test_fold
            prediction_rows.append(frame)
            if variant_name == "full":
                baseline_frame = frame
            metric_rows.append({
                "test_fold": test_fold,
                "variant": variant_name,
                **binary_metrics(variant_labels, probabilities, threshold),
            })
        assert baseline_frame is not None

        # 梯度×输入只作为反事实结果的辅助归因证据。
        model.eval()
        for start in range(0, len(test_indices), config.batch_size):
            batch_indices = test_indices[start : start + config.batch_size]
            batch = torch.as_tensor(np.asarray(waveforms[batch_indices]), dtype=torch.float32, device=device)
            batch.requires_grad_(True)
            logits = model(batch)
            gradients = torch.autograd.grad(logits.sum(), batch)[0]
            importance = torch.abs(gradients * batch).detach().cpu().numpy()
            for local_index, sample_index in enumerate(batch_indices):
                direct_mask = np.zeros(config.sample_count, dtype=bool)
                multipath_mask = np.zeros(config.sample_count, dtype=bool)
                direct_start = int(bcsc_arrays["direct_index"][sample_index])
                multipath_start = int(bcsc_arrays["multipath_index"][sample_index])
                if direct_start >= 0:
                    direct_mask[max(0, direct_start) : min(config.sample_count, direct_start + config.atom_length)] = True
                if multipath_start >= 0:
                    multipath_mask[max(0, multipath_start) : min(config.sample_count, multipath_start + config.atom_length)] = True
                multipath_mask &= ~direct_mask
                background_mask = ~(direct_mask | multipath_mask)
                total = float(np.sum(importance[local_index])) + 1e-12
                attribution_rows.append(
                    {
                        "sample_index": int(sample_index),
                        "test_fold": test_fold,
                        "direct_fraction": float(np.sum(importance[local_index, direct_mask]) / total),
                        "multipath_fraction": float(np.sum(importance[local_index, multipath_mask]) / total),
                        "background_fraction": float(np.sum(importance[local_index, background_mask]) / total),
                    }
                )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions = predictions.merge(
        manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num"]],
        on="sample_index",
        how="left",
    )
    baseline = predictions[predictions["variant"] == "full"][["sample_index", "probability"]].rename(columns={"probability": "baseline_probability"})
    predictions = predictions.merge(baseline, on="sample_index", how="left")
    predictions["probability_change"] = predictions["probability"] - predictions["baseline_probability"]
    predictions["prediction_flip"] = (predictions["probability"] >= 0.5) != (predictions["baseline_probability"] >= 0.5)
    metrics = pd.DataFrame(metric_rows)
    attribution = pd.DataFrame(attribution_rows).merge(
        manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "label", "ori_file_num"]],
        on="sample_index",
        how="left",
    )
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    attribution.to_csv(attribution_path, index=False, encoding="utf-8-sig")
    return metrics, predictions


def feature_ablation_tests(
    direct: np.ndarray,
    multipath: np.ndarray,
    relation: np.ndarray,
    manifest: pd.DataFrame,
    config: Config = CFG,
    force: bool = False,
) -> pd.DataFrame:
    """对多径数值特征清零并在检出子集上评估已训练特征模型。"""
    paths = ensure_output_dirs(config)
    result_path = paths["root"] / "feature_ablation_metrics.csv"
    if result_path.exists() and not force:
        return pd.read_csv(result_path)
    raw_features = np.concatenate([direct, multipath, relation], axis=1)
    labels = manifest["label"].to_numpy(dtype=np.float32)
    folds = manifest["test_fold"].to_numpy(dtype=int)
    direct_count = direct.shape[1]
    multipath_count = multipath.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_metrics = pd.read_csv(paths["root"] / "fold_metrics.csv")
    rows = []
    for test_fold in range(config.n_splits):
        validation_fold = (test_fold + 1) % config.n_splits
        train_indices = np.where((folds != test_fold) & (folds != validation_fold))[0]
        test_indices = np.where(folds == test_fold)[0]
        center, scale = fit_feature_scaler(raw_features, train_indices)
        scaled = apply_feature_scaler(raw_features, center, scale)
        model = FeatureTransformer(scaled.shape[1], direct_count).to(device)
        model.load_state_dict(torch.load(paths["models"] / f"direct_multipath_fold_{test_fold}.pt", map_location=device, weights_only=True))
        threshold = float(
            trained_metrics[
                (trained_metrics["model"] == "direct_multipath")
                & (trained_metrics["test_fold"] == test_fold)
            ]["threshold"].iloc[0]
        )
        variants = {"full_features": scaled.copy(), "multipath_numeric_zero": scaled.copy()}
        variants["multipath_numeric_zero"][:, direct_count : direct_count + multipath_count] = 0.0
        relation_start = direct_count + multipath_count
        keep_relation = {0, 1, 2}
        for offset in range(relation.shape[1]):
            if offset not in keep_relation:
                variants["multipath_numeric_zero"][:, relation_start + offset] = 0.0
        for variant_name, values in variants.items():
            loader = DataLoader(ArrayDataset(values, labels, test_indices), batch_size=config.batch_size * 2, shuffle=False)
            _, variant_labels, probabilities = predict_model(model, loader, device)
            rows.append({"test_fold": test_fold, "variant": variant_name, **binary_metrics(variant_labels, probabilities, threshold)})
            detected = relation[test_indices, 0] > 0.5
            if detected.sum() > 2 and len(np.unique(variant_labels[detected])) == 2:
                rows.append({
                    "test_fold": test_fold,
                    "variant": variant_name + "__detected_subset",
                    **binary_metrics(variant_labels[detected], probabilities[detected], threshold),
                })
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    return result


def plot_results(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    bcsc_table: pd.DataFrame,
    counterfactual_metrics: pd.DataFrame,
    config: Config = CFG,
) -> None:
    """生成报告使用的核心图表。"""
    figures = ensure_output_dirs(config)["figures"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"waveform": "#2457A6", "direct_multipath": "#188977", "direct_only": "#E09F3E"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, frame in predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(frame["label"], frame["probability"])
        precision, recall, _ = precision_recall_curve(frame["label"], frame["probability"])
        axes[0].plot(fpr, tpr, label=model_name, color=colors[model_name])
        axes[1].plot(recall, precision, label=model_name, color=colors[model_name])
    axes[0].plot([0, 1], [0, 1], "--", color="0.5")
    axes[0].set(xlabel="假阳性率", ylabel="真阳性率", title="来源隔离五折汇总ROC")
    axes[1].set(xlabel="召回率", ylabel="精确率", title="来源隔离五折汇总PR")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "model_roc_pr.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, frame in bcsc_table.groupby("label"):
        name = "正样本" if label == 1 else "负样本"
        axes[0].hist(frame["direct_confidence"], bins=np.linspace(0, 1, 10), alpha=0.55, label=name)
        selected = frame[frame["multipath_index"] >= 0]
        axes[1].hist(selected["delay_ms"], bins=30, alpha=0.55, label=name)
    axes[0].set(title="直达候选稳定度", xlabel="8次复现比例", ylabel="数量")
    axes[1].set(title="高置信度次到达时延", xlabel="时延 (ms)", ylabel="数量")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "bcsc_confidence_delay.png", dpi=220)
    plt.close(fig)

    counter_summary = counterfactual_metrics.groupby("variant")["auroc"].agg(["mean", "std"]).reset_index()
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(counter_summary["variant"], counter_summary["mean"], yerr=counter_summary["std"], color="#5B3A8C", alpha=0.85)
    axis.axhline(0.5, color="0.4", linestyle="--")
    axis.set(title="波形Transformer反事实AUROC", ylabel="五折AUROC", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "counterfactual_auroc.png", dpi=220)
    plt.close(fig)


def write_research_report(
    manifest: pd.DataFrame,
    bcsc_table: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    counterfactual_metrics: pd.DataFrame,
    counterfactual_predictions: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    synthetic_test: dict[str, Any],
    config: Config = CFG,
) -> Path:
    """生成包含三组结果和两项研究问题结论的Markdown报告。"""
    report_path = config.output_dir / "research_report.md"
    model_labels = {
        "waveform": "波形端到端 Transformer",
        "direct_multipath": "直达径+多径显式特征 Transformer",
        "direct_only": "仅直达径显式特征 Transformer",
    }
    summary_lines = []
    for _, row in summary.iterrows():
        summary_lines.append(
            f"| {model_labels[row['model']]} | {row['auroc_fold_mean']:.4f} "
            f"[{row['auroc_ci_low']:.4f}, {row['auroc_ci_high']:.4f}] | "
            f"{row['auprc_fold_mean']:.4f} | {row['balanced_accuracy_fold_mean']:.4f} | {row['f1_fold_mean']:.4f} |"
        )
    direct_multi = predictions[predictions["model"] == "direct_multipath"].sort_values("sample_index")
    direct_only = predictions[predictions["model"] == "direct_only"].sort_values("sample_index")
    delta_auroc = roc_auc_score(direct_multi["label"], direct_multi["probability"]) - roc_auc_score(direct_only["label"], direct_only["probability"])
    paired_deltas = pd.read_csv(config.output_dir / "paired_model_deltas.csv").set_index("metric")
    delta_auroc_low = float(paired_deltas.loc["auroc", "ci_low"])
    delta_auroc_high = float(paired_deltas.loc["auroc", "ci_high"])
    counter_summary = counterfactual_metrics.groupby("variant")["auroc"].mean().to_dict()
    swaps = counterfactual_predictions[counterfactual_predictions["variant"] == "background_swapped"]
    flip_rate = float(swaps["prediction_flip"].mean())
    mean_probability_change = float(swaps["probability_change"].abs().mean())
    attribution = pd.read_csv(config.output_dir / "waveform_attribution.csv")
    attribution_means = attribution[["direct_fraction", "multipath_fraction", "background_fraction"]].mean()
    positive_mp_rate = float((bcsc_table.loc[bcsc_table["label"] == 1, "multipath_index"] >= 0).mean())
    negative_mp_rate = float((bcsc_table.loc[bcsc_table["label"] == 0, "multipath_index"] >= 0).mean())
    success_rate = float((bcsc_table["status"] == "ok").mean())
    shortcut_evidence = (
        counter_summary.get("residual_only", 0.5) > 0.55
        or flip_rate > 0.10
        or mean_probability_change > 0.10
    )
    if delta_auroc_low > 0:
        multipath_conclusion = "点估计和95%置信区间均支持显式多径表征提高总体AUROC。"
    elif delta_auroc_high < 0:
        multipath_conclusion = "点估计和95%置信区间均显示显式多径表征降低总体AUROC。"
    elif delta_auroc >= 0:
        multipath_conclusion = (
            "显式多径表征的AUROC点估计更高，但95%置信区间包含0，"
            "因此尚无统计上明确的总体AUROC增益证据。"
        )
    else:
        multipath_conclusion = (
            "显式多径表征的AUROC点估计更低，但95%置信区间包含0，"
            "因此尚无统计上明确的总体AUROC下降证据。"
        )
    report = f"""# BCSC–Transformer 二分类与捷径学习研究报告

## 摘要

本实验使用3500条中华白海豚click正样本和3500条来源匹配的困难负样本。所有样本统一从原始波形获得5761点、576 kHz片段，并执行相同预处理。数据按原始录音编号严格隔离为5折，避免同一录音跨训练、验证和测试集合。

BCSC处理成功率为 **{success_rate:.2%}**；正样本高置信度多径检出率为 **{positive_mp_rate:.2%}**，负样本稳定次到达候选检出率为 **{negative_mp_rate:.2%}**。这里的7/8复现率表示算法稳定性，不是物理路径为真的后验概率。

## 三组主要结果

| 模型 | AUROC均值 [录音组Bootstrap 95% CI] | AUPRC均值 | 平衡准确率均值 | F1均值 |
|---|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

![三模型ROC和PR](figures/model_roc_pr.png)

## 问题一：波形模型是否存在背景或多径捷径

波形模型完整输入五折平均AUROC为 **{counter_summary.get('full', float('nan')):.4f}**；仅背景残差AUROC为 **{counter_summary.get('residual_only', float('nan')):.4f}**；移除高置信度多径后的AUROC为 **{counter_summary.get('remove_multipath', float('nan')):.4f}**；跨类别背景交换造成 **{flip_rate:.2%}** 的预测翻转，平均绝对概率变化为 **{mean_probability_change:.4f}**。

梯度×输入归因中，直达径、多径和背景区域的平均贡献比例分别为 **{attribution_means['direct_fraction']:.2%}**、**{attribution_means['multipath_fraction']:.2%}** 和 **{attribution_means['background_fraction']:.2%}**。该归因仅作为反事实检验的辅助证据。

基于预注册反事实判据，当前结果{'提供了' if shortcut_evidence else '没有提供明确的'}背景/多径捷径证据。该判断依赖反事实扰动而不是仅依赖注意力或显著性图。

![反事实AUROC](figures/counterfactual_auroc.png)

## 问题二：显式多径是否辅助仅直达径识别

“直达径+多径”相对“仅直达径”的汇总AUROC差值为 **{delta_auroc:+.4f}**，录音组Bootstrap 95% CI为 **[{delta_auroc_low:+.4f}, {delta_auroc_high:+.4f}]**。{multipath_conclusion}

附加消融同时报告保留多径存在/置信度掩码但清零数值特征，以及只在两类均检出稳定次到达的子集上的结果，用于区分路径特性、检出存在性和选择效应。详见 `feature_ablation_metrics.csv`。

## BCSC与数据质量控制

- 合成直达+延迟副本测试：真实时延 {synthetic_test['true_delay_ms']:.4f} ms，估计时延 {synthetic_test['estimated_delay_ms']:.4f} ms，绝对误差 {synthetic_test['absolute_error_ms']:.4f} ms，结果：{'通过' if synthetic_test['passed'] else '未通过'}。
- 数据量：正样本 {(manifest['label'] == 1).sum()}，负样本 {(manifest['label'] == 0).sum()}，总计 {len(manifest)}。
- BCSC未静默删除任何记录；失败或直达径不稳定的状态完整保存在 `bcsc_paths.csv`。
- 单个样本最多保存1条满足7/8复现和±0.03 ms延迟匹配的次到达候选。

![BCSC稳定度与时延](figures/bcsc_confidence_delay.png)

## 限制

单通道盲卷积分解存在尺度、符号、整体时移及源—信道分配不唯一性。负样本中的稳定次到达只称为算法候选。结果适用于当前35个原始录音来源，外推到其他水域、设备或季节前需要独立外部验证。

## 可复现信息

- 随机种子：{config.random_seed}
- 配置哈希：`{config.config_hash}`
- Python：`D:\\Python_env\\toothwhale\\python.exe`
- GPU：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def validate_dataset(manifest: pd.DataFrame, audit: pd.DataFrame, config: Config = CFG) -> dict[str, Any]:
    """执行数据数量、格式、重叠和来源泄漏验收。"""
    counts = manifest["label"].value_counts().to_dict()
    fold_group_counts = manifest.groupby("ori_file_num")["test_fold"].nunique()
    negative = manifest[manifest["label"] == 0].sort_values(["ori_file_num", "timestamp_s"])
    minimum_gaps = negative.groupby("ori_file_num")["timestamp_s"].diff().dropna()
    checks = {
        "positive_count": int(counts.get(1, 0)),
        "negative_count": int(counts.get(0, 0)),
        "unique_sample_ids": bool(manifest["sample_id"].is_unique),
        "all_waveforms_finite": bool(audit["finite"].all()),
        "all_sample_rates_correct": bool((audit["sample_rate_hz"] == config.sample_rate).all()),
        "all_sample_counts_correct": bool((audit["sample_count"] == config.sample_count).all()),
        "minimum_negative_gap_s": float(minimum_gaps.min()),
        "source_group_leakage_count": int((fold_group_counts != 1).sum()),
    }
    checks["passed"] = bool(
        checks["positive_count"] == 3_500
        and checks["negative_count"] == 3_500
        and checks["unique_sample_ids"]
        and checks["all_waveforms_finite"]
        and checks["all_sample_rates_correct"]
        and checks["all_sample_counts_correct"]
        and checks["minimum_negative_gap_s"] >= 0.010 - 1e-9
        and checks["source_group_leakage_count"] == 0
    )
    json_dump(config.output_dir / "data_validation.json", checks)
    return checks


def run_pipeline(
    config: Config = CFG,
    force: bool = False,
    quick: bool = False,
) -> dict[str, Any]:
    """运行完整实验；quick模式只完成数据与少量BCSC烟雾测试。"""
    ensure_output_dirs(config)
    seed_everything(config.random_seed)
    started = time.perf_counter()
    manifest = build_dataset_manifest(config, force=force)
    waveforms, audit = materialize_waveforms(manifest, config, force=force)
    data_validation = validate_dataset(manifest, audit, config)
    if not data_validation["passed"]:
        raise RuntimeError(f"数据验收失败：{data_validation}")
    synthetic_test = synthetic_bcsc_test(config)
    if not synthetic_test["passed"]:
        raise RuntimeError(f"BCSC合成测试失败：{synthetic_test}")
    bcsc_arrays, bcsc_table = run_bcsc_batch(
        waveforms,
        manifest,
        config,
        force=force,
        limit=10 if quick else None,
    )
    result: dict[str, Any] = {
        "mode": "quick" if quick else "full",
        "config_hash": config.config_hash,
        "data_validation": data_validation,
        "synthetic_test": synthetic_test,
        "bcsc_completed": int(bcsc_arrays["completed"].sum()),
        "elapsed_s": time.perf_counter() - started,
    }
    if quick:
        json_dump(config.output_dir / "quick_run_summary.json", result)
        return result
    success_rate = float((bcsc_table["status"] == "ok").mean())
    if success_rate < 0.99:
        raise RuntimeError(f"BCSC成功率{success_rate:.2%}低于99%验收门槛")
    direct, multipath, relation, feature_names = build_features(
        waveforms, manifest, bcsc_arrays, config, force=force
    )
    metrics, predictions = train_cross_validated_models(
        waveforms, manifest, direct, multipath, relation, config, force=force
    )
    summary = summarize_results(metrics, predictions, config)
    counterfactual_metrics, counterfactual_predictions = run_counterfactual_tests(
        waveforms, manifest, bcsc_arrays, config, force=force
    )
    feature_ablation = feature_ablation_tests(
        direct, multipath, relation, manifest, config, force=force
    )
    plot_results(summary, predictions, bcsc_table, counterfactual_metrics, config)
    report_path = write_research_report(
        manifest,
        bcsc_table,
        summary,
        metrics,
        predictions,
        counterfactual_metrics,
        counterfactual_predictions,
        feature_ablation,
        synthetic_test,
        config,
    )
    result.update(
        {
            "bcsc_success_rate": success_rate,
            "high_confidence_multipath_count": int((bcsc_table["multipath_index"] >= 0).sum()),
            "feature_count": len(feature_names),
            "report_path": str(report_path.resolve()),
            "elapsed_s": time.perf_counter() - started,
        }
    )
    json_dump(config.output_dir / "run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """解析命令行并运行研究流水线。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="仅运行数据审计、合成测试和10条BCSC烟雾测试")
    parser.add_argument("--force", action="store_true", help="忽略已有缓存并重新计算")
    arguments = parser.parse_args()
    run_pipeline(CFG, force=arguments.force, quick=arguments.quick)


if __name__ == "__main__":
    main()
