#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双分支物理约束海豚 Click 分类实验流水线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from bisect import bisect_left, insort
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt, windows
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


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
EXPERIMENT_DIR = ROOT / "21_PhysicsInformedClickClassification"
BASELINE_OUTPUT = EXPERIMENT_DIR / "outputs_full_5seeds"
LOW_SNR_REFERENCE_OUTPUT = EXPERIMENT_DIR / "outputs_snr_m15_p15_full_5seeds"
DEFAULT_CONDITIONS = ("waveform_transformer", "recon_bce", "recon_physics_bce")
LOCAL_ENERGY_CONDITION = "recon_local_energy_physics_bce"
NO_RESIDUAL_CONDITIONS = ("recon_bce_no_residual", "recon_physics_bce_no_residual")
SOURCE_ONLY_CONDITIONS = ("recon_bce_source_only", "recon_physics_bce_source_only")
ALL_CONDITIONS = (DEFAULT_CONDITIONS + (LOCAL_ENERGY_CONDITION,) + NO_RESIDUAL_CONDITIONS
                  + SOURCE_ONLY_CONDITIONS)
INITIALIZATION_OFFSETS = {
    "waveform_transformer": 0,
    "recon_bce": 1_000,
    "recon_bce_no_residual": 1_000,
    "recon_physics_bce": 2_000,
    "recon_physics_bce_no_residual": 2_000,
    "recon_bce_source_only": 1_000,
    "recon_physics_bce_source_only": 2_000,
    LOCAL_ENERGY_CONDITION: 0,
}
CONDITION_LABELS = {
    "waveform_transformer": "Waveform Transformer",
    "recon_bce": "Recon + BCE",
    "recon_physics_bce": "Recon + Pointwise Physics + BCE",
    "recon_bce_no_residual": "Recon + BCE (no residual feature)",
    "recon_physics_bce_no_residual": "Recon + Pointwise Physics + BCE (no residual feature)",
    "recon_bce_source_only": "Recon + BCE (source feature only)",
    "recon_physics_bce_source_only": "Recon + Pointwise Physics + BCE (source feature only)",
    LOCAL_ENERGY_CONDITION: "Recon + Local-Energy Physics + BCE",
}


def is_no_residual_experiment(conditions: Iterable[str]) -> bool:
    return set(conditions) == set(NO_RESIDUAL_CONDITIONS)


def is_source_only_experiment(conditions: Iterable[str]) -> bool:
    return set(conditions) == set(SOURCE_ONLY_CONDITIONS)


def is_feature_ablation_experiment(conditions: Iterable[str]) -> bool:
    return is_no_residual_experiment(conditions) or is_source_only_experiment(conditions)


@dataclass(frozen=True)
class Config:
    project_root: Path = ROOT
    experiment_dir: Path = EXPERIMENT_DIR
    output_dir: Path = EXPERIMENT_DIR / "outputs"
    positive_root: Path = ROOT / "00_Data" / "Positive_Clicks_3500"
    clicktrains_csv: Path = ROOT / "00_Data" / "ClickTrains.csv"
    negative_csv: Path = ROOT / "14" / "negative_samples_log.csv"
    original_audio_root: Path = ROOT / "00_Data" / "original_audio"
    historical_output: Path = ROOT / "19_20260813" / "outputs"
    sample_rate: int = 576_000
    sample_count: int = 5_761
    response_count: int = 1_153
    lowcut_hz: float = 20_000.0
    highcut_hz: float = 190_000.0
    filter_order: int = 5
    tukey_alpha: float = 0.04
    beta: float = 0.001
    bce_weight: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 25
    patience: int = 5
    n_splits: int = 5
    bootstrap_runs: int = 2_000
    random_seed: int = 20_260_826
    num_workers: int = 0
    epsilon: float = 1e-8
    positive_noise_mix: bool = False
    snr_min_db: float = -15.0
    snr_max_db: float = 15.0
    mix_seed: int = 20_260_826
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
    local_energy_window: int = 57
    local_energy_stride: int = 1
    local_energy_activity_threshold: float = 0.05

    @property
    def gamma(self) -> float:
        return math.log(10.0) / 0.002

    @property
    def config_hash(self) -> str:
        payload = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
        if LOCAL_ENERGY_CONDITION not in self.conditions:
            for key in ("local_energy_window", "local_energy_stride", "local_energy_activity_threshold"):
                payload.pop(key)
            if not is_feature_ablation_experiment(self.conditions):
                payload.pop("conditions")
            else:
                payload["classification_features"] = (
                    "source_only" if is_source_only_experiment(self.conditions)
                    else "source_response")
        else:
            payload["physics_types"] = {
                "recon_physics_bce": "pointwise_exponential",
                LOCAL_ENERGY_CONDITION: "overlapping_local_energy_group_lasso",
            }
        if not self.positive_noise_mix:
            # 新功能未启用时保持历史配置哈希和检查点兼容性。
            for key in ("positive_noise_mix", "snr_min_db", "snr_max_db", "mix_seed"):
                payload.pop(key)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def output_paths(config: Config) -> dict[str, Path]:
    paths = {
        "root": config.output_dir,
        "cache": config.output_dir / "cache",
        "models": config.output_dir / "models",
        "logs": config.output_dir / "logs",
        "metrics": config.output_dir / "metrics",
        "predictions": config.output_dir / "predictions",
        "figures": config.output_dir / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_pulse_train(path: Path) -> int:
    match = re.search(r"PulseTrain_(\d+)", str(path))
    if not match:
        raise ValueError(f"无法解析 PulseTrain 编号：{path}")
    return int(match.group(1))


def parse_original_number(audio_file: str) -> int:
    match = re.search(r"Ori_Recording_(\d+)", audio_file)
    if not match:
        raise ValueError(f"无法解析原始录音编号：{audio_file}")
    return int(match.group(1))


def select_nonoverlapping_rows(source: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """按噪声类型近似分层，选取相隔至少10 ms的负样本。"""
    source = source.copy()
    fractions = source["Noise_Type"].value_counts(normalize=True)
    quotas = {name: int(math.floor(target * value)) for name, value in fractions.items()}
    for name in fractions.index[: target - sum(quotas.values())]:
        quotas[name] += 1
    chosen: list[int] = []
    times: list[float] = []

    def addable(value: float) -> bool:
        return all(abs(value - old) >= 0.010 for old in times)

    for kind, quota in quotas.items():
        if quota <= 0:
            continue
        accepted = 0
        for index in rng.permutation(source.index[source["Noise_Type"] == kind].to_numpy()):
            timestamp = float(source.at[index, "Noise_Timestamp(s)"])
            if addable(timestamp):
                chosen.append(int(index)); times.append(timestamp); accepted += 1
                if accepted == quota:
                    break
    if len(chosen) < target:
        for index in rng.permutation(source.drop(index=chosen).index.to_numpy()):
            timestamp = float(source.at[index, "Noise_Timestamp(s)"])
            if addable(timestamp):
                chosen.append(int(index)); times.append(timestamp)
                if len(chosen) == target:
                    break
    if len(chosen) != target:
        group = int(source["ori_file_num"].iloc[0])
        raise RuntimeError(f"录音 {group} 只选到 {len(chosen)}/{target} 条非重叠负样本")
    return source.loc[chosen].sort_values("Noise_Timestamp(s)").copy()


def is_nonoverlapping(timestamp: float, sorted_times: list[float], spacing_s: float) -> bool:
    """检查以 timestamp 为中心的窗口是否与已排序窗口集合重叠。"""
    position = bisect_left(sorted_times, timestamp)
    if position > 0 and timestamp - sorted_times[position - 1] < spacing_s:
        return False
    if position < len(sorted_times) and sorted_times[position] - timestamp < spacing_s:
        return False
    return True


def select_background_rows(source: pd.DataFrame, target: int, rng: np.random.Generator,
                           blocked_times: Iterable[float], spacing_s: float) -> pd.DataFrame:
    """按剩余池噪声类型比例抽样，并与负样本及已选背景保持窗口不重叠。"""
    source = source.copy()
    occupied = sorted(float(value) for value in blocked_times)
    available_mask = [is_nonoverlapping(float(value), occupied, spacing_s)
                      for value in source["Noise_Timestamp(s)"]]
    source = source.loc[available_mask]
    fractions = source["Noise_Type"].value_counts(normalize=True)
    quotas = {name: int(math.floor(target * value)) for name, value in fractions.items()}
    for name in fractions.index[: target - sum(quotas.values())]:
        quotas[name] += 1
    chosen: list[int] = []

    def try_add(index: int) -> bool:
        timestamp = float(source.at[index, "Noise_Timestamp(s)"])
        if not is_nonoverlapping(timestamp, occupied, spacing_s):
            return False
        chosen.append(int(index))
        insort(occupied, timestamp)
        return True

    for kind, quota in quotas.items():
        if quota <= 0:
            continue
        accepted = 0
        indices = source.index[source["Noise_Type"] == kind].to_numpy()
        for index in rng.permutation(indices):
            if try_add(int(index)):
                accepted += 1
                if accepted == quota:
                    break
    if len(chosen) < target:
        remaining = source.drop(index=chosen).index.to_numpy()
        for index in rng.permutation(remaining):
            if try_add(int(index)) and len(chosen) == target:
                break
    if len(chosen) != target:
        group = int(source["ori_file_num"].iloc[0]) if len(source) else -1
        raise RuntimeError(f"录音 {group} 只选到 {len(chosen)}/{target} 条混合背景")
    return source.loc[chosen].copy()


def build_mixed_manifest(config: Config) -> pd.DataFrame:
    """复用高SNR基线清单，固定负样本、折划分和低SNR背景配对。"""
    baseline_path = BASELINE_OUTPUT / "dataset_manifest.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"缺少高SNR基线清单：{baseline_path}")
    manifest = pd.read_csv(baseline_path)
    if len(manifest) != 7000 or manifest["label"].value_counts().to_dict() != {1: 3500, 0: 3500}:
        raise RuntimeError("高SNR基线清单不是正负各3500条")
    manifest = manifest.copy()
    mix_columns = {
        "mix_noise_audio_file": "", "mix_noise_wav_path": "",
        "mix_noise_timestamp_s": np.nan, "mix_noise_type": "",
        "target_snr_db": np.nan, "realized_snr_db": np.nan, "mix_alpha": np.nan,
    }
    for name, default in mix_columns.items():
        manifest[name] = default

    negative_pool = pd.read_csv(config.negative_csv)
    negative_pool["ori_file_num"] = negative_pool["Audio_File"].map(parse_original_number)
    negatives = manifest[manifest["label"] == 0]
    positives = manifest[manifest["label"] == 1]
    spacing_s = config.sample_count / config.sample_rate
    half_window = (config.sample_count // 2) / config.sample_rate
    rng = np.random.default_rng(config.mix_seed)

    for source, positive_group in positives.groupby("ori_file_num", sort=True):
        source = int(source)
        audio_file = f"Ori_Recording_{source:02d}.wav"
        audio_path = config.original_audio_root / audio_file
        info = sf.info(str(audio_path))
        candidates = negative_pool[negative_pool["ori_file_num"] == source].copy()
        candidates = candidates[(candidates["Noise_Timestamp(s)"] >= half_window) &
                                (candidates["Noise_Timestamp(s)"] <= info.duration - half_window)]
        blocked = negatives.loc[negatives["ori_file_num"] == source, "timestamp_s"].astype(float)
        selected = select_background_rows(candidates, len(positive_group), rng, blocked, spacing_s)
        selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
        positive_indices = positive_group.index.to_numpy()
        manifest.loc[positive_indices, "mix_noise_audio_file"] = selected["Audio_File"].to_numpy()
        manifest.loc[positive_indices, "mix_noise_wav_path"] = selected["Audio_File"].map(
            lambda name: str((config.original_audio_root / name).resolve())).to_numpy()
        manifest.loc[positive_indices, "mix_noise_timestamp_s"] = selected["Noise_Timestamp(s)"].to_numpy(float)
        manifest.loc[positive_indices, "mix_noise_type"] = selected["Noise_Type"].to_numpy()

    positive_indices = positives.index.to_numpy()
    manifest.loc[positive_indices, "target_snr_db"] = rng.uniform(
        config.snr_min_db, config.snr_max_db, len(positive_indices))
    manifest["config_hash"] = config.config_hash
    return manifest


def build_manifest(config: Config, force: bool = False) -> pd.DataFrame:
    paths = output_paths(config)
    destination = paths["root"] / "dataset_manifest.csv"
    if destination.exists() and not force:
        frame = pd.read_csv(destination)
        valid_hash = ("config_hash" in frame and
                      frame["config_hash"].astype(str).eq(config.config_hash).all())
        if (len(frame) == 7000 and frame["label"].value_counts().to_dict() == {1: 3500, 0: 3500}
                and valid_hash):
            return frame

    if config.positive_noise_mix:
        manifest = build_mixed_manifest(config)
        manifest.to_csv(destination, index=False, encoding="utf-8-sig")
        return manifest

    clicktrains = pd.read_csv(config.clicktrains_csv)
    mapping = (clicktrains[["Train_num(No.)", "Ori_file_num(No.)"]].dropna()
               .drop_duplicates("Train_num(No.)").set_index("Train_num(No.)")["Ori_file_num(No.)"]
               .astype(int).to_dict())
    positives = []
    wav_paths = sorted(config.positive_root.rglob("*.wav"))
    if len(wav_paths) != 3500:
        raise RuntimeError(f"正样本应为3500条，实际为 {len(wav_paths)}")
    for wav_path in wav_paths:
        train = parse_pulse_train(wav_path)
        source = int(mapping[train])
        positives.append({
            "sample_id": f"P_{source:02d}_{train:03d}_{wav_path.stem}", "label": 1,
            "ori_file_num": source, "audio_file": f"Ori_Recording_{source:02d}.wav",
            "timestamp_s": np.nan, "noise_type": "positive_click",
            "wav_path": str(wav_path.resolve()), "source_kind": "positive_wav",
        })
    positive = pd.DataFrame(positives)
    negative_pool = pd.read_csv(config.negative_csv)
    negative_pool["ori_file_num"] = negative_pool["Audio_File"].map(parse_original_number)
    rng = np.random.default_rng(config.random_seed)
    selected = []
    half_window = (config.sample_count // 2) / config.sample_rate
    for source, target in positive["ori_file_num"].value_counts().sort_index().items():
        audio = config.original_audio_root / f"Ori_Recording_{int(source):02d}.wav"
        info = sf.info(str(audio))
        candidates = negative_pool[negative_pool["ori_file_num"] == int(source)].copy()
        candidates = candidates[(candidates["Noise_Timestamp(s)"] >= half_window) &
                                (candidates["Noise_Timestamp(s)"] <= info.duration - half_window)]
        selected.append(select_nonoverlapping_rows(candidates, int(target), rng))
    negative = pd.concat(selected, ignore_index=True)
    negative = negative.assign(
        sample_id=negative.apply(lambda r: f"N_{int(r.ori_file_num):02d}_{float(r['Noise_Timestamp(s)']):.6f}_{r.Noise_Type}", axis=1),
        label=0,
        audio_file=negative["Audio_File"],
        timestamp_s=negative["Noise_Timestamp(s)"].astype(float),
        noise_type=negative["Noise_Type"],
        wav_path=negative["Audio_File"].map(lambda n: str((config.original_audio_root / n).resolve())),
        source_kind="original_audio_timestamp",
    )[["sample_id", "label", "ori_file_num", "audio_file", "timestamp_s", "noise_type", "wav_path", "source_kind"]]
    manifest = pd.concat([positive, negative], ignore_index=True)
    splitter = StratifiedGroupKFold(config.n_splits, shuffle=True, random_state=config.random_seed)
    manifest["test_fold"] = -1
    for fold, (_, indices) in enumerate(splitter.split(manifest, manifest["label"], manifest["ori_file_num"])):
        manifest.loc[indices, "test_fold"] = fold
    manifest["config_hash"] = config.config_hash
    manifest.to_csv(destination, index=False, encoding="utf-8-sig")
    return manifest


def read_waveform(row: pd.Series, config: Config) -> np.ndarray:
    path = Path(str(row["wav_path"]))
    if row["source_kind"] == "positive_wav":
        signal, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    else:
        sample_rate = sf.info(str(path)).samplerate
        center = int(round(float(row["timestamp_s"]) * sample_rate))
        start = center - config.sample_count // 2
        with sf.SoundFile(str(path)) as handle:
            handle.seek(start)
            signal = handle.read(config.sample_count, dtype="float64", always_2d=False)
    signal = np.asarray(signal).squeeze()
    if sample_rate != config.sample_rate or signal.shape != (config.sample_count,):
        raise ValueError(f"波形格式错误：{path}, fs={sample_rate}, shape={signal.shape}")
    return signal


def read_timestamp_waveform(path: Path, timestamp_s: float, config: Config) -> np.ndarray:
    sample_rate = sf.info(str(path)).samplerate
    center = int(round(timestamp_s * sample_rate))
    start = center - config.sample_count // 2
    with sf.SoundFile(str(path)) as handle:
        handle.seek(start)
        signal = handle.read(config.sample_count, dtype="float64", always_2d=False)
    signal = np.asarray(signal).squeeze()
    if sample_rate != config.sample_rate or signal.shape != (config.sample_count,):
        raise ValueError(f"背景波形格式错误：{path}, fs={sample_rate}, shape={signal.shape}")
    return signal


def preprocess_component(signal: np.ndarray, config: Config) -> np.ndarray:
    """去直流、零相位带通和Tukey窗，不执行幅度归一化。"""
    signal = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    sos = butter(config.filter_order,
                 [config.lowcut_hz / (config.sample_rate / 2), config.highcut_hz / (config.sample_rate / 2)],
                 btype="bandpass", output="sos")
    signal = sosfiltfilt(sos, signal)
    signal *= windows.tukey(config.sample_count, alpha=config.tukey_alpha)
    if not np.isfinite(signal).all():
        raise RuntimeError("预处理后波形包含非有限值")
    return signal


def peak_normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if not np.isfinite(peak) or peak < 1e-12:
        raise RuntimeError("预处理后波形无有效幅度")
    return (signal / peak).astype(np.float32)


def preprocess_waveform(signal: np.ndarray, config: Config) -> np.ndarray:
    return peak_normalize(preprocess_component(signal, config))


def materialize_waveforms(manifest: pd.DataFrame, config: Config, force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    paths = output_paths(config)
    array_path = paths["cache"] / "preprocessed_waveforms.npy"
    audit_path = paths["root"] / "waveform_audit.csv"
    if array_path.exists() and audit_path.exists() and not force:
        values = np.load(array_path, mmap_mode="r")
        audit = pd.read_csv(audit_path)
        if values.shape == (7000, config.sample_count) and len(audit) == 7000:
            return values, audit
    values = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32,
                                       shape=(len(manifest), config.sample_count))
    rows = []
    for index, (_, row) in enumerate(tqdm(manifest.iterrows(), total=len(manifest),
                                          desc="总体阶段 1/5：预处理7000条波形", unit="条")):
        target_snr = float(row["target_snr_db"]) if config.positive_noise_mix and int(row["label"]) == 1 else np.nan
        realized_snr = np.nan
        alpha = np.nan
        click_rms = np.nan
        noise_rms = np.nan
        if config.positive_noise_mix and int(row["label"]) == 1:
            click = preprocess_component(read_waveform(row, config), config)
            noise_path = Path(str(row["mix_noise_wav_path"]))
            noise = preprocess_component(read_timestamp_waveform(
                noise_path, float(row["mix_noise_timestamp_s"]), config), config)
            click_rms = float(np.sqrt(np.mean(np.square(click))))
            noise_rms = float(np.sqrt(np.mean(np.square(noise))))
            if click_rms < 1e-12 or noise_rms < 1e-12:
                raise RuntimeError(f"混合分量RMS无效：{row['sample_id']}")
            alpha = click_rms / (noise_rms * 10.0 ** (target_snr / 20.0))
            realized_snr = float(20.0 * np.log10(click_rms / (alpha * noise_rms)))
            values[index] = peak_normalize(click + alpha * noise)
            manifest.at[index, "realized_snr_db"] = realized_snr
            manifest.at[index, "mix_alpha"] = alpha
        else:
            values[index] = preprocess_waveform(read_waveform(row, config), config)
        rows.append({"sample_index": index, "sample_id": row["sample_id"],
                     "sample_rate_hz": config.sample_rate, "sample_count": config.sample_count,
                     "finite": bool(np.isfinite(values[index]).all()),
                     "peak": float(np.max(np.abs(values[index]))),
                     "mean": float(np.mean(values[index])),
                     "target_snr_db": target_snr, "realized_snr_db": realized_snr,
                     "snr_error_db": abs(realized_snr - target_snr) if np.isfinite(realized_snr) else np.nan,
                     "mix_alpha": alpha, "click_rms": click_rms, "noise_rms": noise_rms})
    values.flush()
    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(paths["root"] / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    return np.load(array_path, mmap_mode="r"), audit


def compute_data_hash(manifest: pd.DataFrame, waveforms: np.ndarray, config: Config) -> dict[str, Any]:
    digest = hashlib.sha256()
    columns = ["sample_id", "label", "ori_file_num", "test_fold"]
    if config.positive_noise_mix:
        columns.extend(["mix_noise_audio_file", "mix_noise_timestamp_s", "mix_noise_type",
                        "target_snr_db", "realized_snr_db", "mix_alpha"])
        mix_config = {"positive_noise_mix": True, "mix_seed": config.mix_seed,
                      "snr_min_db": config.snr_min_db, "snr_max_db": config.snr_max_db}
        digest.update(json.dumps(mix_config, sort_keys=True).encode())
    digest.update(manifest[columns].to_csv(index=False).encode())
    for block_start in range(0, len(waveforms), 128):
        digest.update(np.asarray(waveforms[block_start:block_start + 128]).tobytes())
    result = {"sha256": digest.hexdigest(), "samples": len(manifest),
              "shape": list(waveforms.shape), "dtype": str(waveforms.dtype)}
    json_dump(output_paths(config)["root"] / "data_hash.json", result)
    return result


def validate_data(manifest: pd.DataFrame, waveforms: np.ndarray, audit: pd.DataFrame, config: Config) -> dict[str, Any]:
    group_folds = manifest.groupby("ori_file_num")["test_fold"].nunique()
    checks = {
        "sample_count": int(len(manifest)),
        "positive_count": int((manifest["label"] == 1).sum()),
        "negative_count": int((manifest["label"] == 0).sum()),
        "shape": list(waveforms.shape),
        "all_finite": bool(audit["finite"].all()),
        "all_peak_normalized": bool(np.allclose(audit["peak"], 1.0, atol=1e-5)),
        "unique_sample_ids": bool(manifest["sample_id"].is_unique),
        "source_group_leakage_count": int((group_folds != 1).sum()),
    }
    mix_passed = True
    if config.positive_noise_mix:
        positives = manifest[manifest["label"] == 1].copy()
        negatives = manifest[manifest["label"] == 0].copy()
        positive_keys = set(zip(positives["mix_noise_audio_file"], positives["mix_noise_timestamp_s"],
                                positives["mix_noise_type"]))
        negative_keys = set(zip(negatives["audio_file"], negatives["timestamp_s"], negatives["noise_type"]))
        spacing_s = config.sample_count / config.sample_rate
        background_overlap_count = 0
        negative_overlap_count = 0
        for source, group in positives.groupby("ori_file_num"):
            background_times = np.sort(group["mix_noise_timestamp_s"].to_numpy(float))
            if len(background_times) > 1:
                background_overlap_count += int(np.sum(np.diff(background_times) < spacing_s))
            negative_times = np.sort(negatives.loc[negatives["ori_file_num"] == source,
                                                    "timestamp_s"].to_numpy(float))
            if len(negative_times):
                positions = np.searchsorted(negative_times, background_times)
                for timestamp, position in zip(background_times, positions):
                    neighbours = []
                    if position > 0:
                        neighbours.append(negative_times[position - 1])
                    if position < len(negative_times):
                        neighbours.append(negative_times[position])
                    negative_overlap_count += int(any(abs(timestamp - old) < spacing_s for old in neighbours))
        snr_errors = audit.loc[manifest["label"].to_numpy() == 1, "snr_error_db"].to_numpy(float)
        checks.update({
            "mixed_positive_count": int(len(positives)),
            "background_same_recording": bool((positives["ori_file_num"].astype(int) ==
                                                positives["mix_noise_audio_file"].map(parse_original_number)).all()),
            "background_unique_count": int(len(positive_keys)),
            "background_selected_negative_exact_match_count": int(len(positive_keys & negative_keys)),
            "background_background_overlap_count": background_overlap_count,
            "background_negative_overlap_count": negative_overlap_count,
            "target_snr_min_db": float(positives["target_snr_db"].min()),
            "target_snr_max_db": float(positives["target_snr_db"].max()),
            "target_snr_in_bounds": bool(positives["target_snr_db"].between(
                config.snr_min_db, config.snr_max_db, inclusive="both").all()),
            "maximum_snr_error_db": float(np.max(snr_errors)),
            "snr_error_within_1e_6_db": bool(np.all(snr_errors <= 1e-6)),
            "negative_samples_match_baseline": bool(negatives["sample_id"].tolist() ==
                pd.read_csv(BASELINE_OUTPUT / "dataset_manifest.csv").query("label == 0")["sample_id"].tolist()),
        })
        mix_passed = bool(checks["mixed_positive_count"] == 3500
                          and checks["background_same_recording"]
                          and checks["background_unique_count"] == 3500
                          and checks["background_selected_negative_exact_match_count"] == 0
                          and checks["background_background_overlap_count"] == 0
                          and checks["background_negative_overlap_count"] == 0
                          and checks["target_snr_in_bounds"] and checks["snr_error_within_1e_6_db"]
                          and checks["negative_samples_match_baseline"])
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.hist(positives["target_snr_db"], bins=np.arange(config.snr_min_db, config.snr_max_db + 5.0, 5.0),
                  edgecolor="black", alpha=0.8)
        axis.set(xlabel="Target SNR (dB)", ylabel="Positive samples", title="Positive-mixture SNR distribution")
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_paths(config)["figures"] / "snr_histogram.png", dpi=180)
        plt.close(figure)
    checks["passed"] = bool(checks["sample_count"] == 7000 and checks["positive_count"] == 3500
                            and checks["negative_count"] == 3500
                            and checks["shape"] == [7000, config.sample_count]
                            and checks["all_finite"] and checks["all_peak_normalized"]
                            and checks["unique_sample_ids"] and checks["source_group_leakage_count"] == 0
                            and mix_passed)
    json_dump(output_paths(config)["root"] / "data_validation.json", checks)
    return checks


class WaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray, indices: Iterable[int]):
        self.waveforms = waveforms
        self.labels = labels
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (torch.from_numpy(np.asarray(self.waveforms[index]).copy()),
                torch.tensor(float(self.labels[index]), dtype=torch.float32),
                torch.tensor(index, dtype=torch.long))


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 9, stride=stride, padding=4, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, 5, padding=2, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class SharedEncoder(nn.Module):
    """只向两个解码器提供共享表示，不直接连接分类头。"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 64),
            ConvBlock(64, 96), ConvBlock(96, 128),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.layers(waveform.unsqueeze(1))


class FullLengthDecoder(nn.Module):
    """从共享表示预测完整5761点信源，不读取或拼接输入波形。"""
    def __init__(self, output_count: int):
        super().__init__()
        self.output_count = output_count
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(128, 96, 8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(96, 64, 8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(32, 16, 8, stride=2, padding=3), nn.GELU(),
            nn.ConvTranspose1d(16, 8, 8, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(8, 1, 9, padding=4),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.layers(latent)
        decoded = nn.functional.interpolate(decoded, size=self.output_count, mode="linear", align_corners=False)
        return torch.tanh(decoded[:, 0])


class CausalResponseDecoder(nn.Module):
    """预测0–2 ms因果响应；首点严格固定为1。"""
    def __init__(self, response_count: int):
        super().__init__()
        self.response_count = response_count
        self.layers = nn.Sequential(
            nn.Conv1d(128, 96, 3, padding=1), nn.GELU(),
            nn.Conv1d(96, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tail = self.layers(latent)
        tail = nn.functional.interpolate(tail, size=self.response_count - 1,
                                         mode="linear", align_corners=False)[:, 0]
        tail = 0.25 * torch.tanh(tail)
        first = torch.ones((len(latent), 1), dtype=tail.dtype, device=tail.device)
        return torch.cat([first, tail], dim=1)


def causal_fft_convolution(source: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """批量可微线性卷积，返回与信源等长的因果前段。"""
    output_count = source.shape[-1]
    linear_count = source.shape[-1] + response.shape[-1] - 1
    fft_count = 1 << (linear_count - 1).bit_length()
    source_fft = torch.fft.rfft(source, n=fft_count)
    response_fft = torch.fft.rfft(response, n=fft_count)
    return torch.fft.irfft(source_fft * response_fft, n=fft_count)[..., :output_count]


class BranchFeatureEncoder(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(16, channels, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(channels, channels, 5, stride=4, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values.unsqueeze(1)).squeeze(-1)


class PhysicalClassifier(nn.Module):
    """分类头从信源、可选信道响应及可选重建残差取得信息。"""
    def __init__(self, sample_count: int = 5761, response_count: int = 1153,
                 include_residual: bool = True, include_response_feature: bool = True):
        super().__init__()
        self.include_residual = include_residual
        self.include_response_feature = include_response_feature
        self.shared_encoder = SharedEncoder()
        self.source_decoder = FullLengthDecoder(sample_count)
        self.response_decoder = CausalResponseDecoder(response_count)
        self.source_classifier = BranchFeatureEncoder()
        if include_response_feature:
            self.response_classifier = BranchFeatureEncoder()
        if include_residual:
            self.residual_classifier = BranchFeatureEncoder()
        self.classification_head = nn.Sequential(
            nn.Linear(32 * (1 + int(include_response_feature) + int(include_residual)), 64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 1),
        )

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.shared_encoder(waveform)
        source = self.source_decoder(latent)
        response = self.response_decoder(latent)
        reconstruction = causal_fft_convolution(source, response)
        residual = waveform - reconstruction
        feature_parts = [self.source_classifier(source)]
        if self.include_response_feature:
            feature_parts.append(self.response_classifier(response))
        if self.include_residual:
            feature_parts.append(self.residual_classifier(residual))
        features = torch.cat(feature_parts, dim=1)
        logit = self.classification_head(features).squeeze(-1)
        return {"logit": logit, "source": source, "response": response,
                "reconstruction": reconstruction, "residual": residual}


def sinusoidal_positions(length: int, dimension: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dimension, 2, device=device) *
                            (-math.log(10_000.0) / dimension))
    result = torch.zeros(length, dimension, device=device)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies)
    return result


class WaveformTransformer(nn.Module):
    """与历史实验结构一致的波形基线。"""
    def __init__(self, patch_size: int = 32, stride: int = 16, d_model: int = 64):
        super().__init__()
        self.patch = nn.Conv1d(1, d_model, patch_size, stride=stride)
        self.adapter = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, 4, 128, 0.1, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, 1))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        tokens = self.adapter(self.patch(waveform.unsqueeze(1)).transpose(1, 2))
        tokens = torch.cat([self.cls.expand(len(waveform), -1, -1), tokens], dim=1)
        tokens = tokens + sinusoidal_positions(tokens.shape[1], tokens.shape[2], tokens.device)[None]
        return self.head(self.norm(self.encoder(tokens)[:, 0])).squeeze(-1)


def reconstruction_loss(target: torch.Tensor, prediction: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    numerator = torch.sum((target - prediction) ** 2, dim=1)
    denominator = torch.sum(target ** 2, dim=1) + epsilon
    return torch.mean(numerator / denominator)


def channel_physics_loss(response: torch.Tensor, sample_rate: int = 576_000) -> torch.Tensor:
    indices = torch.arange(1, response.shape[1], device=response.device, dtype=response.dtype)
    gamma = math.log(10.0) / 0.002
    weights = torch.exp(gamma * indices / sample_rate)
    return torch.mean(torch.abs(response[:, 1:]) * weights[None])


def local_energy_curve(response: torch.Tensor, window: int = 57, stride: int = 1,
                       epsilon: float = 1e-8) -> torch.Tensor:
    """计算不含固定h0的重叠局部RMS能量，零尾部严格映射为零。"""
    tail = response[:, 1:]
    pooled = nn.functional.avg_pool1d(tail.square().unsqueeze(1), kernel_size=window,
                                      stride=stride, padding=0).squeeze(1)
    return torch.sqrt(pooled + epsilon) - math.sqrt(epsilon)


def local_energy_physics_loss(response: torch.Tensor, window: int = 57, stride: int = 1,
                              epsilon: float = 1e-8) -> torch.Tensor:
    """返回批次和重叠窗口上的局部能量Group-Lasso均值。"""
    return local_energy_curve(response, window, stride, epsilon).mean()


def physics_loss_for_condition(response: torch.Tensor, condition: str, config: Config) -> torch.Tensor:
    """按实验条件选择旧逐点或新局部能量Physics损失。"""
    if condition in ("recon_physics_bce", "recon_physics_bce_no_residual",
                     "recon_physics_bce_source_only"):
        return channel_physics_loss(response, config.sample_rate)
    if condition == LOCAL_ENERGY_CONDITION:
        return local_energy_physics_loss(response, config.local_energy_window,
                                         config.local_energy_stride, config.epsilon)
    return response[:, 1:].sum() * 0.0


def model_losses(outputs: dict[str, torch.Tensor], waveform: torch.Tensor, labels: torch.Tensor,
                 condition: str, config: Config) -> dict[str, torch.Tensor]:
    """组合重建、指定Physics和BCE，并显式返回加权Physics贡献。"""
    recon = reconstruction_loss(waveform, outputs["reconstruction"], config.epsilon)
    physics = physics_loss_for_condition(outputs["response"], condition, config)
    beta = config.beta if condition in (
        "recon_physics_bce", "recon_physics_bce_no_residual",
        "recon_physics_bce_source_only", LOCAL_ENERGY_CONDITION) else 0.0
    physics_weighted = beta * physics
    bce = nn.functional.binary_cross_entropy_with_logits(outputs["logit"], labels)
    return {"recon": recon, "physics": physics, "physics_weighted": physics_weighted, "bce": bce,
            "total": recon + physics_weighted + config.bce_weight * bce}


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def gradient_norm(module: nn.Module) -> float:
    squares = [parameter.grad.detach().float().square().sum() for parameter in module.parameters()
               if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(squares).sum()).cpu()) if squares else 0.0


def diagnostics(outputs: dict[str, torch.Tensor], waveform: torch.Tensor,
                config: Config) -> dict[str, np.ndarray]:
    """计算NCC、尾能量、局部能量Hoyer稀疏度和相对活动比例。"""
    source = outputs["source"]
    response = outputs["response"]
    centered_source = source - source.mean(dim=1, keepdim=True)
    centered_waveform = waveform - waveform.mean(dim=1, keepdim=True)
    ncc = torch.sum(centered_source * centered_waveform, dim=1) / (
        torch.linalg.vector_norm(centered_source, dim=1) * torch.linalg.vector_norm(centered_waveform, dim=1) + 1e-8)
    tail_energy = torch.sum(response[:, 1:] ** 2, dim=1)
    energy = local_energy_curve(response, config.local_energy_window,
                                config.local_energy_stride, config.epsilon)
    count = energy.shape[1]
    l1 = torch.linalg.vector_norm(energy, ord=1, dim=1)
    l2 = torch.linalg.vector_norm(energy, ord=2, dim=1)
    hoyer = (math.sqrt(count) - l1 / (l2 + config.epsilon)) / (math.sqrt(count) - 1.0)
    relative_cutoff = config.local_energy_activity_threshold * energy.amax(dim=1, keepdim=True)
    active_ratio = (energy > relative_cutoff).float().mean(dim=1)
    return {"ncc_source_input": ncc.detach().cpu().numpy(),
            "response_tail_energy": tail_energy.detach().cpu().numpy(),
            "local_energy_hoyer": hoyer.clamp(0.0, 1.0).detach().cpu().numpy(),
            "local_energy_active_ratio": active_ratio.detach().cpu().numpy()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            physical: bool, config: Config) -> tuple[pd.DataFrame, np.ndarray | None]:
    model.eval()
    frames = []
    response_tails = []
    for waveforms, labels, indices in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        if physical:
            outputs = model(waveforms)
            probabilities = torch.sigmoid(outputs["logit"]).cpu().numpy()
            diag = diagnostics(outputs, waveforms, config)
            response_tails.append(outputs["response"][:, 1:].cpu().numpy())
        else:
            probabilities = torch.sigmoid(model(waveforms)).cpu().numpy()
            diag = {"ncc_source_input": np.full(len(labels), np.nan),
                    "response_tail_energy": np.full(len(labels), np.nan),
                    "local_energy_hoyer": np.full(len(labels), np.nan),
                    "local_energy_active_ratio": np.full(len(labels), np.nan)}
        frames.append(pd.DataFrame({"sample_index": indices.numpy(), "label": labels.numpy().astype(int),
                                    "probability": probabilities, **diag}))
    tails = np.concatenate(response_tails) if response_tails else None
    return pd.concat(frames, ignore_index=True), tails


def response_diversity(response_tails: np.ndarray | None, seed: int) -> float:
    """以随机样本对的1减余弦相似度衡量响应多样性。"""
    if response_tails is None or len(response_tails) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    pair_count = min(10_000, len(response_tails) * 4)
    left = rng.integers(0, len(response_tails), pair_count)
    right = rng.integers(0, len(response_tails), pair_count)
    valid = left != right
    a, b = response_tails[left[valid]], response_tails[right[valid]]
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return float(np.mean(1.0 - cosine))


def create_loaders(waveforms: np.ndarray, labels: np.ndarray, train_indices: np.ndarray,
                   validation_indices: np.ndarray, test_indices: np.ndarray,
                   config: Config) -> dict[str, DataLoader]:
    kwargs = {"num_workers": config.num_workers, "pin_memory": torch.cuda.is_available()}
    return {
        "train": DataLoader(WaveformDataset(waveforms, labels, train_indices), config.batch_size,
                            shuffle=True, **kwargs),
        "validation": DataLoader(WaveformDataset(waveforms, labels, validation_indices), config.batch_size * 2,
                                 shuffle=False, **kwargs),
        "test": DataLoader(WaveformDataset(waveforms, labels, test_indices), config.batch_size * 2,
                           shuffle=False, **kwargs),
    }


def fit_condition(condition: str, fold: int, experiment_seed: int,
                  waveforms: np.ndarray, labels: np.ndarray,
                  train_indices: np.ndarray, validation_indices: np.ndarray, test_indices: np.ndarray,
                  config: Config, force: bool, overall: tqdm) -> tuple[dict[str, Any], pd.DataFrame]:
    paths = output_paths(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physical = condition != "waveform_transformer"
    if LOCAL_ENERGY_CONDITION in config.conditions:
        initialization_seed = experiment_seed + fold if physical else experiment_seed + 100_000 + fold
    else:
        initialization_seed = experiment_seed + INITIALIZATION_OFFSETS[condition] + fold
    seed_everything(initialization_seed)
    model: nn.Module = (PhysicalClassifier(
                        config.sample_count, config.response_count,
                        include_residual=condition not in NO_RESIDUAL_CONDITIONS and condition not in SOURCE_ONLY_CONDITIONS,
                        include_response_feature=condition not in SOURCE_ONLY_CONDITIONS)
                        if physical else WaveformTransformer())
    model = model.to(device)
    loaders = create_loaders(waveforms, labels, train_indices, validation_indices, test_indices, config)
    checkpoint = paths["models"] / f"{condition}_seed_{experiment_seed}_fold_{fold}.pt"
    history_path = paths["logs"] / f"{condition}_seed_{experiment_seed}_fold_{fold}_history.csv"
    batch_path = paths["logs"] / f"{condition}_seed_{experiment_seed}_fold_{fold}_batch.csv"

    needs_training = force or not checkpoint.exists()
    if not needs_training:
        candidate = torch.load(checkpoint, map_location="cpu", weights_only=True)
        needs_training = bool(candidate.get("config_hash") != config.config_hash
                              or not candidate.get("training_complete", False)
                              or not history_path.exists() or not batch_path.exists())
    if needs_training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        best_score, best_epoch, stale = -np.inf, 0, 0
        history_rows: list[dict[str, Any]] = []
        batch_rows: list[dict[str, Any]] = []
        epoch_bar = tqdm(range(1, config.max_epochs + 1), desc=f"{condition} 折{fold} epoch",
                         unit="epoch", leave=False)
        for epoch in epoch_bar:
            model.train()
            sums = {"total": 0.0, "recon": 0.0, "physics": 0.0,
                    "physics_weighted": 0.0, "bce": 0.0}
            batch_bar = tqdm(loaders["train"], desc=f"折{fold} epoch {epoch} batch",
                             unit="batch", leave=False)
            for batch_number, (batch_waveforms, batch_labels, _) in enumerate(batch_bar, start=1):
                batch_waveforms = batch_waveforms.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if physical:
                    outputs = model(batch_waveforms)
                    losses = model_losses(outputs, batch_waveforms, batch_labels, condition, config)
                else:
                    bce = nn.functional.binary_cross_entropy_with_logits(model(batch_waveforms), batch_labels)
                    zero = torch.zeros((), device=device)
                    losses = {"total": bce, "recon": zero, "physics": zero,
                              "physics_weighted": zero, "bce": bce}
                losses["total"].backward()
                total_grad = gradient_norm(model)
                source_grad = gradient_norm(model.source_decoder) if physical else float("nan")
                response_grad = gradient_norm(model.response_decoder) if physical else float("nan")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                row = {name: float(value.detach().cpu()) for name, value in losses.items()}
                for name in sums:
                    sums[name] += row[name]
                batch_rows.append({"condition": condition, "seed": experiment_seed,
                                   "fold": fold, "epoch": epoch,
                                   "batch": batch_number, **row, "gradient_norm": total_grad,
                                   "source_decoder_gradient_norm": source_grad,
                                   "response_decoder_gradient_norm": response_grad})
                batch_bar.set_postfix(loss=f"{row['total']:.4f}", recon=f"{row['recon']:.3f}",
                                      physics=f"{row['physics']:.3f}",
                                      beta_physics=f"{row['physics_weighted']:.3e}", bce=f"{row['bce']:.3f}",
                                      grad=f"{total_grad:.2f}", refresh=False)
            validation, _ = predict(model, loaders["validation"], device, physical, config)
            score = float(roc_auc_score(validation["label"], validation["probability"]))
            count = len(loaders["train"])
            history_row = {"condition": condition, "seed": experiment_seed,
                           "fold": fold, "epoch": epoch,
                           **{name: value / count for name, value in sums.items()},
                           "validation_auroc": score}
            history_rows.append(history_row)
            epoch_bar.set_postfix(loss=f"{history_row['total']:.4f}", val_auroc=f"{score:.4f}")
            if score > best_score + 1e-4:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({"state_dict": model.state_dict(), "condition": condition,
                            "seed": experiment_seed, "fold": fold,
                            "initialization_seed": initialization_seed,
                            "physics_type": ("local_energy" if condition == LOCAL_ENERGY_CONDITION
                                             else "pointwise" if condition in (
                                                 "recon_physics_bce",
                                                 "recon_physics_bce_no_residual",
                                                 "recon_physics_bce_source_only") else "none"),
                            "include_residual_feature": (condition not in NO_RESIDUAL_CONDITIONS
                                                         and condition not in SOURCE_ONLY_CONDITIONS),
                            "include_response_feature": condition not in SOURCE_ONLY_CONDITIONS,
                            "local_energy_window": config.local_energy_window,
                            "local_energy_stride": config.local_energy_stride,
                            "best_epoch": best_epoch, "validation_auroc": best_score,
                            "config_hash": config.config_hash}, checkpoint)
            else:
                stale += 1
                if stale >= config.patience:
                    break
        pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(batch_rows).to_csv(batch_path, index=False, encoding="utf-8-sig")
        completed = torch.load(checkpoint, map_location="cpu", weights_only=True)
        completed["training_complete"] = True
        torch.save(completed, checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    validation, _ = predict(model, loaders["validation"], device, physical, config)
    threshold = choose_threshold(validation["label"].to_numpy(), validation["probability"].to_numpy())
    test, tails = predict(model, loaders["test"], device, physical, config)
    metrics = binary_metrics(test["label"].to_numpy(), test["probability"].to_numpy(), threshold)
    metrics.update({"condition": condition, "seed": experiment_seed,
                    "test_fold": fold, "validation_fold": (fold + 1) % 5,
                    "best_epoch": int(saved["best_epoch"]),
                    "validation_auroc": float(saved["validation_auroc"]),
                    "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    "mean_ncc_source_input": float(test["ncc_source_input"].mean()) if physical else float("nan"),
                    "mean_response_tail_energy": float(test["response_tail_energy"].mean()) if physical else float("nan"),
                    "mean_local_energy_hoyer": float(test["local_energy_hoyer"].mean()) if physical else float("nan"),
                    "mean_local_energy_active_ratio": float(test["local_energy_active_ratio"].mean()) if physical else float("nan"),
                    "response_diversity": response_diversity(tails, experiment_seed + fold)})
    if tails is not None:
        energy = local_energy_curve(torch.as_tensor(tails), config.local_energy_window,
                                    config.local_energy_stride, config.epsilon).numpy()
        pd.DataFrame({"window_index": np.arange(energy.shape[1]),
                      "time_ms": (np.arange(energy.shape[1]) + config.local_energy_window / 2.0)
                                 / config.sample_rate * 1000.0,
                      "mean_local_energy": energy.mean(axis=0),
                      "median_local_energy": np.median(energy, axis=0)}).to_csv(
            paths["metrics"] / f"{condition}_seed_{experiment_seed}_fold_{fold}_local_energy_curve.csv",
            index=False, encoding="utf-8-sig")
    test["threshold"] = threshold
    test["condition"] = condition
    test["seed"] = experiment_seed
    test["test_fold"] = fold
    overall.update(1)
    overall.set_postfix(condition=condition, seed=experiment_seed, fold=fold,
                        auroc=f"{metrics['auroc']:.4f}")
    return metrics, test


def split_indices(manifest: pd.DataFrame, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = manifest["test_fold"].to_numpy(dtype=int)
    validation_fold = (test_fold + 1) % 5
    test = np.flatnonzero(folds == test_fold)
    validation = np.flatnonzero(folds == validation_fold)
    train = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    train_groups = set(manifest.iloc[train]["ori_file_num"])
    validation_groups = set(manifest.iloc[validation]["ori_file_num"])
    test_groups = set(manifest.iloc[test]["ori_file_num"])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise RuntimeError("训练、验证、测试之间存在录音来源泄漏")
    return train, validation, test


def train_experiments(waveforms: np.ndarray, manifest: pd.DataFrame, config: Config,
                      mode: str, seeds: tuple[int, ...], force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = output_paths(config)
    metrics_path = paths["metrics"] / "fold_metrics.csv"
    predictions_path = paths["predictions"] / "sample_predictions.csv"
    expected_folds = [0] if mode == "quick" else list(range(5))
    if metrics_path.exists() and predictions_path.exists() and not force:
        metrics = pd.read_csv(metrics_path)
        if (set(metrics["condition"]) == set(config.conditions)
                and set(metrics["test_fold"]) == set(expected_folds)
                and "seed" in metrics and set(metrics["seed"]) == set(seeds)
                and len(metrics) == len(config.conditions) * len(expected_folds) * len(seeds)):
            return metrics, pd.read_csv(predictions_path)
    labels = manifest["label"].to_numpy(dtype=np.float32)
    all_metrics, all_predictions = [], []
    overall = tqdm(total=len(seeds) * len(expected_folds) * len(config.conditions),
                   desc="总体阶段 3/5：实验训练", unit="条件折")
    for experiment_seed in seeds:
        for fold in expected_folds:
            train, validation, test = split_indices(manifest, fold)
            for condition in config.conditions:
                metrics, predictions = fit_condition(condition, fold, experiment_seed,
                                                      waveforms, labels, train, validation, test,
                                                      config, force, overall)
                all_metrics.append(metrics)
                all_predictions.append(predictions)
    overall.close()
    metrics = pd.DataFrame(all_metrics)
    manifest_columns = ["sample_index", "sample_id", "ori_file_num", "noise_type"]
    if config.positive_noise_mix:
        manifest_columns.extend(["target_snr_db", "realized_snr_db", "mix_noise_type"])
    predictions = pd.concat(all_predictions, ignore_index=True).merge(
        manifest.reset_index(names="sample_index")[manifest_columns],
        on="sample_index", how="left")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics[metrics["condition"] != "waveform_transformer"].to_csv(
        paths["root"] / "ablation_results.csv", index=False, encoding="utf-8-sig")
    return metrics, predictions


def metric_from_frame(frame: pd.DataFrame, metric: str) -> float:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    if metric == "auroc":
        return float(roc_auc_score(labels, probabilities))
    if metric == "auprc":
        return float(average_precision_score(labels, probabilities))
    predicted = probabilities >= frame["threshold"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, predicted))
    if metric == "f1":
        return float(f1_score(labels, predicted, zero_division=0))
    if metric == "sensitivity":
        return float(tp / max(tp + fn, 1))
    if metric == "specificity":
        return float(tn / max(tn + fp, 1))
    raise KeyError(metric)


def sampled_groups_frame(frame: pd.DataFrame, sampled_groups: np.ndarray) -> pd.DataFrame:
    pieces = [frame[frame["ori_file_num"] == group] for group in sampled_groups]
    return pd.concat(pieces, ignore_index=True)


METRIC_NAMES = ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity")


def all_frame_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {metric: metric_from_frame(frame, metric) for metric in METRIC_NAMES}


def summarize_seed_metrics(predictions: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows = []
    for condition in config.conditions:
        for experiment_seed in sorted(predictions["seed"].unique()):
            frame = predictions[(predictions["condition"] == condition) &
                                (predictions["seed"] == experiment_seed)]
            rows.append({"condition": condition, "seed": int(experiment_seed), **all_frame_metrics(frame)})
    result = pd.DataFrame(rows)
    result.to_csv(output_paths(config)["metrics"] / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    return result


def summarize_physics_contributions(config: Config, seeds: tuple[int, ...], mode: str) -> pd.DataFrame:
    """汇总原始Physics及beta加权贡献，供量级失效诊断和报告使用。"""
    paths = output_paths(config)
    folds = [0] if mode == "quick" else list(range(config.n_splits))
    frames = []
    for condition in config.conditions:
        for seed in seeds:
            for fold in folds:
                path = paths["logs"] / f"{condition}_seed_{seed}_fold_{fold}_batch.csv"
                frame = pd.read_csv(path, usecols=["condition", "physics", "physics_weighted", "total"])
                frames.append(frame)
    batches = pd.concat(frames, ignore_index=True)
    result = batches.groupby("condition", as_index=False).agg(
        mean_physics=("physics", "mean"),
        mean_weighted_physics=("physics_weighted", "mean"),
        mean_total_loss=("total", "mean"),
    )
    result["weighted_physics_fraction_of_total"] = (
        result["mean_weighted_physics"] / result["mean_total_loss"].clip(lower=config.epsilon))
    result.to_csv(paths["metrics"] / "physics_contribution_summary.csv", index=False, encoding="utf-8-sig")
    return result


def snr_binned_metrics(predictions: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每5 dB区间使用该区间正样本与完整负样本计算性能。"""
    if not config.positive_noise_mix:
        return pd.DataFrame(), pd.DataFrame()
    edges = np.arange(config.snr_min_db, config.snr_max_db + 5.0, 5.0)
    rows = []
    for condition in config.conditions:
        for experiment_seed in sorted(predictions["seed"].unique()):
            frame = predictions[(predictions["condition"] == condition) &
                                (predictions["seed"] == experiment_seed)]
            negative_bin = frame[frame["label"] == 0]
            positives = frame[frame["label"] == 1]
            for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                include_upper = bin_index == len(edges) - 2
                mask = ((positives["target_snr_db"] >= lower) &
                        (positives["target_snr_db"] <= upper if include_upper
                         else positives["target_snr_db"] < upper))
                positive_bin = positives[mask]
                evaluation = pd.concat([negative_bin, positive_bin], ignore_index=True)
                rows.append({"condition": condition, "seed": int(experiment_seed),
                             "snr_lower_db": float(lower), "snr_upper_db": float(upper),
                             "positive_count": int(len(positive_bin)),
                             "negative_count": int(len(negative_bin)),
                             "auroc": metric_from_frame(evaluation, "auroc"),
                             "auprc": metric_from_frame(evaluation, "auprc"),
                             "sensitivity": metric_from_frame(evaluation, "sensitivity")})
    detailed = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in detailed.groupby(["condition", "snr_lower_db", "snr_upper_db"], sort=False):
        condition, lower, upper = keys
        row = {"condition": condition, "snr_lower_db": lower, "snr_upper_db": upper,
               "positive_count_mean": float(group["positive_count"].mean())}
        for metric in ("auroc", "auprc", "sensitivity"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_seed_sd"] = (float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    paths = output_paths(config)
    detailed.to_csv(paths["metrics"] / "snr_bin_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths["metrics"] / "snr_bin_summary.csv", index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for condition in config.conditions:
        frame = summary[summary["condition"] == condition]
        centers = (frame["snr_lower_db"] + frame["snr_upper_db"]) / 2.0
        for axis, metric, title in zip(axes, ("auroc", "auprc", "sensitivity"),
                                       ("AUROC", "AUPRC", "Sensitivity")):
            axis.plot(centers, frame[f"{metric}_mean"], marker="o", label=CONDITION_LABELS[condition])
            axis.set(title=title, xlabel="Target SNR bin center (dB)", ylim=(0.0, 1.02))
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Metric")
    axes[-1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(paths["figures"] / "snr_metric_curves.png", dpi=180)
    plt.close(figure)
    return detailed, summary


def load_no_residual_reference_predictions(predictions: pd.DataFrame,
                                           config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取旧低SNR预测，并严格验证新旧配对键、来源、折号和SNR。"""
    reference_path = LOW_SNR_REFERENCE_OUTPUT / "predictions" / "sample_predictions.csv"
    if not reference_path.exists():
        raise FileNotFoundError(f"缺少原低SNR预测：{reference_path}")
    reference = pd.read_csv(reference_path)
    seeds = set(predictions["seed"].astype(int).unique())
    folds = set(predictions["test_fold"].astype(int).unique())
    reference = reference[
        reference["seed"].astype(int).isin(seeds)
        & reference["test_fold"].astype(int).isin(folds)
        & reference["condition"].isin(("recon_bce", "recon_physics_bce"))
    ].copy()
    mapping = {
        "recon_bce_no_residual": "recon_bce",
        "recon_physics_bce_no_residual": "recon_physics_bce",
        "recon_bce_source_only": "recon_bce",
        "recon_physics_bce_source_only": "recon_physics_bce",
    }
    mapping = {new_name: old_name for new_name, old_name in mapping.items()
               if new_name in config.conditions}
    keys = ["sample_id", "seed", "test_fold"]
    exact_columns = ["label", "ori_file_num", "noise_type"]
    float_columns = ["target_snr_db", "realized_snr_db"]
    checks: dict[str, Any] = {
        "reference_path": str(reference_path.resolve()),
        "paired_key_columns": keys,
        "conditions": {},
    }
    all_passed = True
    for new_name, old_name in mapping.items():
        new = predictions[predictions["condition"] == new_name]
        old = reference[reference["condition"] == old_name]
        if new.duplicated(keys).any() or old.duplicated(keys).any():
            raise RuntimeError(f"{new_name} 与 {old_name} 存在重复配对键")
        merged = new[keys + exact_columns + float_columns].merge(
            old[keys + exact_columns + float_columns], on=keys, how="outer",
            suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
        key_match = bool(merged["_merge"].eq("both").all() and len(merged) == len(new) == len(old))
        exact_match = key_match and all(np.array_equal(
            merged[f"{column}_new"].to_numpy(), merged[f"{column}_old"].to_numpy())
            for column in exact_columns)
        float_match = key_match and all(np.allclose(
            merged[f"{column}_new"].to_numpy(float),
            merged[f"{column}_old"].to_numpy(float),
            rtol=0.0, atol=1e-12, equal_nan=True) for column in float_columns)
        condition_passed = bool(key_match and exact_match and float_match)
        checks["conditions"][f"{new_name}_vs_{old_name}"] = {
            "new_rows": int(len(new)), "reference_rows": int(len(old)),
            "paired_rows": int(merged["_merge"].eq("both").sum()),
            "keys_match": key_match, "labels_sources_noise_types_match": exact_match,
            "target_and_realized_snr_match": float_match, "passed": condition_passed,
        }
        all_passed = all_passed and condition_passed
    checks["passed"] = bool(all_passed)
    json_dump(output_paths(config)["root"] / "pairing_validation.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"新旧实验预测无法严格配对：{checks}")
    return reference, checks


def bootstrap_no_residual_results(predictions: pd.DataFrame, config: Config
                                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对无残差模型本身及其相对原模型的差值执行种子×录音来源Bootstrap。"""
    paths = output_paths(config)
    reference, _ = load_no_residual_reference_predictions(predictions, config)
    rng = np.random.default_rng(config.random_seed + 991)
    groups = np.sort(predictions["ori_file_num"].unique())
    seeds = np.sort(predictions["seed"].unique())
    bootstrap_draws = [(rng.choice(seeds, len(seeds), replace=True),
                        rng.choice(groups, len(groups), replace=True))
                       for _ in range(config.bootstrap_runs)]
    summary_rows = []
    for condition in config.conditions:
        frame = predictions[predictions["condition"] == condition]
        seed_frames = {seed: frame[frame["seed"] == seed] for seed in seeds}
        estimates_by_seed = [all_frame_metrics(seed_frames[seed]) for seed in seeds]
        values_by_metric = {metric: [] for metric in METRIC_NAMES}
        for sampled_seeds, sampled_groups in tqdm(
                bootstrap_draws, desc=f"总体阶段 4/5：{condition} 种子×录音组Bootstrap", leave=False):
            replicate = []
            for seed in sampled_seeds:
                sample = sampled_groups_frame(seed_frames[seed], sampled_groups)
                if sample["label"].nunique() == 2:
                    replicate.append(all_frame_metrics(sample))
            if replicate:
                for metric in METRIC_NAMES:
                    values_by_metric[metric].append(float(np.mean([row[metric] for row in replicate])))
        for metric in METRIC_NAMES:
            values = values_by_metric[metric]
            estimates = [row[metric] for row in estimates_by_seed]
            summary_rows.append({
                "condition": condition, "metric": metric, "estimate": float(np.mean(estimates)),
                "seed_sd": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "bootstrap_runs_valid": len(values),
            })

    comparisons = (("recon_bce_source_only", "recon_bce", "source_only_recon_minus_original_recon"),
                   ("recon_physics_bce_source_only", "recon_physics_bce",
                    "source_only_physics_minus_original_physics")) if is_source_only_experiment(config.conditions) else (
                   ("recon_bce_no_residual", "recon_bce", "no_residual_recon_minus_original_recon"),
                   ("recon_physics_bce_no_residual", "recon_physics_bce",
                    "no_residual_physics_minus_original_physics"))
    paired_rows = []
    for left_name, right_name, comparison in comparisons:
        left = predictions[predictions["condition"] == left_name]
        right = reference[reference["condition"] == right_name]
        left_seed_frames = {seed: left[left["seed"] == seed] for seed in seeds}
        right_seed_frames = {seed: right[right["seed"] == seed] for seed in seeds}
        seed_differences = []
        for seed in seeds:
            left_metrics = all_frame_metrics(left_seed_frames[seed])
            right_metrics = all_frame_metrics(right_seed_frames[seed])
            seed_differences.append({metric: left_metrics[metric] - right_metrics[metric]
                                     for metric in METRIC_NAMES})
        values_by_metric = {metric: [] for metric in METRIC_NAMES}
        for sampled_seeds, sampled_groups in tqdm(
                bootstrap_draws, desc=f"总体阶段 4/5：{comparison} 分层配对Bootstrap", leave=False):
            replicate = []
            for seed in sampled_seeds:
                left_sample = sampled_groups_frame(left_seed_frames[seed], sampled_groups)
                right_sample = sampled_groups_frame(right_seed_frames[seed], sampled_groups)
                if left_sample["label"].nunique() == 2 and right_sample["label"].nunique() == 2:
                    left_metrics = all_frame_metrics(left_sample)
                    right_metrics = all_frame_metrics(right_sample)
                    replicate.append({metric: left_metrics[metric] - right_metrics[metric]
                                      for metric in METRIC_NAMES})
            if replicate:
                for metric in METRIC_NAMES:
                    values_by_metric[metric].append(float(np.mean([row[metric] for row in replicate])))
        for metric in METRIC_NAMES:
            values = values_by_metric[metric]
            estimates = [row[metric] for row in seed_differences]
            ci_low = float(np.percentile(values, 2.5))
            paired_rows.append({
                "comparison": comparison, "left": left_name, "right": right_name,
                "metric": metric, "estimate": float(np.mean(estimates)),
                "seed_sd": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                "ci_low": ci_low, "ci_high": float(np.percentile(values, 97.5)),
                "bootstrap_runs_valid": len(values),
                "noninferiority_margin": -0.005 if metric == "auroc" else np.nan,
                "noninferior": bool(ci_low > -0.005) if metric == "auroc" else np.nan,
                "significantly_better": bool(ci_low > 0.0) if metric == "auroc" else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(paths["metrics"] / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(paths["metrics"] / "paired_differences.csv", index=False, encoding="utf-8-sig")
    return summary, paired


def bootstrap_results(predictions: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    if is_feature_ablation_experiment(config.conditions):
        return bootstrap_no_residual_results(predictions, config)
    paths = output_paths(config)
    summary_path = paths["metrics"] / "bootstrap_summary.csv"
    paired_path = paths["metrics"] / "paired_differences.csv"
    rng = np.random.default_rng(config.random_seed + 991)
    groups = predictions["ori_file_num"].unique()
    seeds = predictions["seed"].unique()
    bootstrap_draws = [(rng.choice(seeds, len(seeds), replace=True),
                        rng.choice(groups, len(groups), replace=True))
                       for _ in range(config.bootstrap_runs)]
    summary_rows = []
    for condition in config.conditions:
        frame = predictions[predictions["condition"] == condition]
        seed_frames = {seed: frame[frame["seed"] == seed] for seed in seeds}
        estimates_by_seed = [all_frame_metrics(seed_frames[seed]) for seed in seeds]
        bootstrap_values = {metric: [] for metric in METRIC_NAMES}
        for sampled_seeds, sampled_groups in tqdm(bootstrap_draws,
                      desc=f"总体阶段 4/5：{condition} 种子×录音组Bootstrap", leave=False):
            replicate = []
            for seed in sampled_seeds:
                sample = sampled_groups_frame(seed_frames[seed], sampled_groups)
                if sample["label"].nunique() == 2:
                    replicate.append(all_frame_metrics(sample))
            if replicate:
                for metric in METRIC_NAMES:
                    bootstrap_values[metric].append(float(np.mean([row[metric] for row in replicate])))
        for metric in METRIC_NAMES:
            values = bootstrap_values[metric]
            summary_rows.append({"condition": condition, "metric": metric,
                                 "estimate": float(np.mean([row[metric] for row in estimates_by_seed])),
                                 "seed_sd": float(np.std([row[metric] for row in estimates_by_seed], ddof=1))
                                 if len(estimates_by_seed) > 1 else 0.0,
                                 "ci_low": float(np.percentile(values, 2.5)),
                                 "ci_high": float(np.percentile(values, 97.5)),
                                 "bootstrap_runs_valid": len(values)})
    if LOCAL_ENERGY_CONDITION in config.conditions:
        pairs = ((LOCAL_ENERGY_CONDITION, "recon_physics_bce", "local_energy_minus_pointwise_physics"),
                 (LOCAL_ENERGY_CONDITION, "recon_bce", "local_energy_minus_recon_bce"),
                 (LOCAL_ENERGY_CONDITION, "waveform_transformer", "local_energy_minus_waveform_transformer"))
    else:
        pairs = (("recon_physics_bce", "recon_bce", "physics_minus_recon_bce"),
                 ("recon_physics_bce", "waveform_transformer", "physics_minus_waveform_transformer"))
    paired_rows = []
    for left_name, right_name, comparison in pairs:
        left = predictions[predictions["condition"] == left_name]
        right = predictions[predictions["condition"] == right_name]
        left_seed_frames = {seed: left[left["seed"] == seed] for seed in seeds}
        right_seed_frames = {seed: right[right["seed"] == seed] for seed in seeds}
        seed_differences = []
        for seed in seeds:
            left_metrics = all_frame_metrics(left_seed_frames[seed])
            right_metrics = all_frame_metrics(right_seed_frames[seed])
            seed_differences.append({metric: left_metrics[metric] - right_metrics[metric]
                                     for metric in METRIC_NAMES})
        bootstrap_values = {metric: [] for metric in METRIC_NAMES}
        for sampled_seeds, sampled_groups in tqdm(bootstrap_draws,
                      desc=f"总体阶段 4/5：{comparison} 分层配对Bootstrap", leave=False):
            replicate = []
            for seed in sampled_seeds:
                left_sample = sampled_groups_frame(left_seed_frames[seed], sampled_groups)
                right_sample = sampled_groups_frame(right_seed_frames[seed], sampled_groups)
                if left_sample["label"].nunique() == 2 and right_sample["label"].nunique() == 2:
                    left_metrics = all_frame_metrics(left_sample)
                    right_metrics = all_frame_metrics(right_sample)
                    replicate.append({metric: left_metrics[metric] - right_metrics[metric]
                                      for metric in METRIC_NAMES})
            if replicate:
                for metric in METRIC_NAMES:
                    bootstrap_values[metric].append(float(np.mean([row[metric] for row in replicate])))
        for metric in METRIC_NAMES:
            values = bootstrap_values[metric]
            estimate = float(np.mean([row[metric] for row in seed_differences]))
            paired_rows.append({"comparison": comparison, "left": left_name, "right": right_name,
                                "metric": metric, "estimate": estimate,
                                "seed_sd": float(np.std([row[metric] for row in seed_differences], ddof=1))
                                if len(seed_differences) > 1 else 0.0,
                                "ci_low": float(np.percentile(values, 2.5)),
                                "ci_high": float(np.percentile(values, 97.5)),
                                "bootstrap_runs_valid": len(values)})
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    return summary, paired


def bootstrap_physics_diagnostics(predictions: pd.DataFrame, config: Config) -> pd.DataFrame:
    """按录音来源配对比较Local-Energy与旧逐点Physics的响应结构。"""
    if LOCAL_ENERGY_CONDITION not in config.conditions:
        return pd.DataFrame()
    left = predictions[predictions["condition"] == LOCAL_ENERGY_CONDITION]
    right = predictions[predictions["condition"] == "recon_physics_bce"]
    keys = ["seed", "sample_id", "ori_file_num"]
    paired = left[keys + ["local_energy_hoyer", "local_energy_active_ratio"]].merge(
        right[keys + ["local_energy_hoyer", "local_energy_active_ratio"]],
        on=keys, suffixes=("_local", "_pointwise"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("局部能量与逐点Physics诊断样本无法一一配对")
    metric_columns = {
        "local_energy_hoyer": ("local_energy_hoyer_local", "local_energy_hoyer_pointwise"),
        "local_energy_active_ratio": ("local_energy_active_ratio_local", "local_energy_active_ratio_pointwise"),
    }
    groups = paired["ori_file_num"].unique()
    rng = np.random.default_rng(config.random_seed + 1992)
    group_draws = [rng.choice(groups, len(groups), replace=True) for _ in range(config.bootstrap_runs)]
    rows = []
    for metric, (left_column, right_column) in metric_columns.items():
        paired["difference"] = paired[left_column] - paired[right_column]
        values = []
        for sampled_groups in tqdm(group_draws,
                                   desc=f"总体阶段 4/5：{metric} 录音组配对Bootstrap", leave=False):
            values.append(float(sampled_groups_frame(paired, sampled_groups)["difference"].mean()))
        rows.append({"comparison": "local_energy_minus_pointwise_physics",
                     "metric": metric, "estimate": float(paired["difference"].mean()),
                     "ci_low": float(np.percentile(values, 2.5)),
                     "ci_high": float(np.percentile(values, 97.5)),
                     "bootstrap_runs_valid": len(values)})
    result = pd.DataFrame(rows)
    result.to_csv(output_paths(config)["metrics"] / "physics_diagnostic_paired_differences.csv",
                  index=False, encoding="utf-8-sig")
    return result


def compare_with_high_snr(predictions: pd.DataFrame, config: Config) -> pd.DataFrame:
    """对每种模型执行低SNR减原高SNR的种子×录音组配对AUROC比较。"""
    if not config.positive_noise_mix:
        return pd.DataFrame()
    baseline_path = BASELINE_OUTPUT / "predictions" / "sample_predictions.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"缺少高SNR基线预测：{baseline_path}")
    high = pd.read_csv(baseline_path)
    seeds = np.sort(predictions["seed"].unique())
    high = high[high["seed"].isin(seeds)]
    groups = np.sort(predictions["ori_file_num"].unique())
    rng = np.random.default_rng(config.mix_seed + 1991)
    rows = []
    comparable_conditions = tuple(condition for condition in config.conditions
                                  if condition in set(high["condition"]))
    for condition in comparable_conditions:
        low_condition = predictions[predictions["condition"] == condition]
        high_condition = high[high["condition"] == condition]
        low_seed_frames = {seed: low_condition[low_condition["seed"] == seed] for seed in seeds}
        high_seed_frames = {seed: high_condition[high_condition["seed"] == seed] for seed in seeds}
        seed_differences = []
        for seed in seeds:
            low_ids = set(low_seed_frames[seed]["sample_id"])
            high_seed_frames[seed] = high_seed_frames[seed][high_seed_frames[seed]["sample_id"].isin(low_ids)]
            high_ids = set(high_seed_frames[seed]["sample_id"])
            if low_ids != high_ids:
                raise RuntimeError(f"{condition} 种子{seed}的高低SNR测试样本不一致")
            seed_differences.append(metric_from_frame(low_seed_frames[seed], "auroc") -
                                    metric_from_frame(high_seed_frames[seed], "auroc"))
        values = []
        for _ in tqdm(range(config.bootstrap_runs),
                      desc=f"总体阶段 4/5：{condition} 低SNR−高SNR配对Bootstrap", leave=False):
            sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
            sampled_groups = rng.choice(groups, len(groups), replace=True)
            replicate = []
            for seed in sampled_seeds:
                low_sample = sampled_groups_frame(low_seed_frames[seed], sampled_groups)
                high_sample = sampled_groups_frame(high_seed_frames[seed], sampled_groups)
                if low_sample["label"].nunique() == 2 and high_sample["label"].nunique() == 2:
                    replicate.append(metric_from_frame(low_sample, "auroc") -
                                     metric_from_frame(high_sample, "auroc"))
            if replicate:
                values.append(float(np.mean(replicate)))
        rows.append({"condition": condition, "metric": "auroc",
                     "comparison": "low_snr_minus_high_snr",
                     "estimate": float(np.mean(seed_differences)),
                     "seed_sd": float(np.std(seed_differences, ddof=1)) if len(seed_differences) > 1 else 0.0,
                     "ci_low": float(np.percentile(values, 2.5)),
                     "ci_high": float(np.percentile(values, 97.5)),
                     "bootstrap_runs_valid": len(values)})
    result = pd.DataFrame(rows)
    result.to_csv(output_paths(config)["metrics"] / "low_minus_high_paired_auroc.csv",
                  index=False, encoding="utf-8-sig")
    return result


def plot_results(predictions: pd.DataFrame, config: Config) -> None:
    paths = output_paths(config)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in config.conditions:
        frame = predictions[predictions["condition"] == condition]
        fpr, tpr, _ = roc_curve(frame["label"], frame["probability"])
        precision, recall, _ = precision_recall_curve(frame["label"], frame["probability"])
        axes[0].plot(fpr, tpr, label=f"{CONDITION_LABELS[condition]} ({roc_auc_score(frame['label'], frame['probability']):.3f})")
        axes[1].plot(recall, precision, label=f"{CONDITION_LABELS[condition]} ({average_precision_score(frame['label'], frame['probability']):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    for axis in axes:
        axis.legend(fontsize=8); axis.grid(alpha=0.2)
    figure.tight_layout(); figure.savefig(paths["figures"] / "roc_pr.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, len(config.conditions), figsize=(4 * len(config.conditions), 4))
    axes = np.atleast_1d(axes)
    for axis, condition in zip(axes, config.conditions):
        frame = predictions[predictions["condition"] == condition]
        matrix = confusion_matrix(frame["label"], frame["probability"] >= frame["threshold"], labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
        axis.set(title=CONDITION_LABELS[condition], xlabel="Predicted", ylabel="True", xticks=[0, 1], yticks=[0, 1])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout(); figure.savefig(paths["figures"] / "confusion_matrices.png", dpi=180); plt.close(figure)

    curve_paths = sorted(paths["metrics"].glob("*_local_energy_curve.csv"))
    curves = []
    for path in curve_paths:
        condition = next((name for name in config.conditions if path.name.startswith(name + "_")), None)
        if condition is not None and condition != "waveform_transformer":
            frame = pd.read_csv(path)
            frame["condition"] = condition
            curves.append(frame)
    if curves:
        aggregate = pd.concat(curves, ignore_index=True).groupby(
            ["condition", "window_index", "time_ms"], as_index=False)["mean_local_energy"].mean()
        aggregate.to_csv(paths["metrics"] / "local_energy_curves.csv", index=False, encoding="utf-8-sig")
        figure, axis = plt.subplots(figsize=(10, 5))
        for condition in config.conditions:
            frame = aggregate[aggregate["condition"] == condition]
            if len(frame):
                axis.plot(frame["time_ms"], frame["mean_local_energy"], label=CONDITION_LABELS[condition])
        axis.set(xlabel="Tail time (ms)", ylabel="Mean local energy",
                 title=f"Local-energy response curves (window={config.local_energy_window})")
        axis.grid(alpha=0.2); axis.legend(fontsize=8)
        figure.tight_layout(); figure.savefig(paths["figures"] / "local_energy_response_curves.png", dpi=180)
        plt.close(figure)


def write_no_residual_report(summary: pd.DataFrame, paired: pd.DataFrame,
                             validation: dict[str, Any], data_hash: dict[str, Any],
                             config: Config, mode: str, seeds: tuple[int, ...]) -> Path:
    """生成独立临时中文消融报告，不触碰原实验报告。"""
    paths = output_paths(config)
    table_rows = []
    metric_labels = {
        "auroc": "AUROC", "auprc": "AUPRC", "balanced_accuracy": "Balanced accuracy",
        "f1": "F1", "sensitivity": "灵敏度", "specificity": "特异度",
    }
    for condition in config.conditions:
        values = []
        for metric in metric_labels:
            row = summary[(summary["condition"] == condition) & (summary["metric"] == metric)].iloc[0]
            values.append(f"{row['estimate']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        table_rows.append(f"| {CONDITION_LABELS[condition]} | " + " | ".join(values) + " |")
    recon = paired[(paired["comparison"] == "no_residual_recon_minus_original_recon")
                   & (paired["metric"] == "auroc")].iloc[0]
    physics = paired[(paired["comparison"] == "no_residual_physics_minus_original_physics")
                     & (paired["metric"] == "auroc")].iloc[0]
    report = f"""# 无残差特征拼接消融实验临时报告

## 实验范围

- 运行模式：`{mode}`；样本数：{validation['sample_count']}；独立训练种子：{', '.join(map(str, seeds))}。
- 数据 SHA-256：`{data_hash['sha256']}`；混合种子：`{config.mix_seed}`。
- 两个模型的分类输入均为 `[z_x; z_h]`（64维），分类头为 `64→64→1`；残差仅用于诊断。
- 新旧预测按 `sample_id + seed + test_fold` 一一配对，并核验录音来源和目标/实现SNR。
- 置信区间采用随机种子×录音来源分组 {config.bootstrap_runs} 次 Bootstrap。
- 预设AUROC非劣效界值为 -0.005；95% CI下限大于 -0.005 判为非劣，大于0才判为显著更优。

## 无残差模型结果

| 条件 | AUROC [95% CI] | AUPRC [95% CI] | Balanced accuracy [95% CI] | F1 [95% CI] | 灵敏度 [95% CI] | 特异度 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 预注册配对结论

- 主要终点（Physics）：AUROC差值 {physics['estimate']:+.4f} [{physics['ci_low']:+.4f}, {physics['ci_high']:+.4f}]；非劣：**{'是' if physics['noninferior'] else '否'}**；显著更优：**{'是' if physics['significantly_better'] else '否'}**。
- 一致性验证（Recon）：AUROC差值 {recon['estimate']:+.4f} [{recon['ci_low']:+.4f}, {recon['ci_high']:+.4f}]；非劣：**{'是' if recon['noninferior'] else '否'}**；显著更优：**{'是' if recon['significantly_better'] else '否'}**。

“置信区间包含0”不作为等价证据；非劣结论只依据预设 -0.005 界值。本报告为独立消融产物，未修改原 `research_report_zh.md`。
"""
    destination = paths["root"] / "no_residual_ablation_report_zh.md"
    destination.write_text(report, encoding="utf-8")
    json_dump(paths["root"] / "no_residual_acceptance_decisions.json", {
        "evaluation_scope": mode, "full_five_fold": mode == "full",
        "seeds": list(seeds), "noninferiority_margin_auroc": -0.005,
        "primary_endpoint": "no_residual_physics_minus_original_physics",
        "physics_noninferior": bool(physics["noninferior"]),
        "physics_significantly_better": bool(physics["significantly_better"]),
        "recon_noninferior": bool(recon["noninferior"]),
        "recon_significantly_better": bool(recon["significantly_better"]),
    })
    return destination


def write_report(metrics: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame,
                 snr_summary: pd.DataFrame, high_comparison: pd.DataFrame,
                 diagnostic_paired: pd.DataFrame, physics_contributions: pd.DataFrame,
                 validation: dict[str, Any], data_hash: dict[str, Any], config: Config,
                 mode: str, seeds: tuple[int, ...]) -> Path:
    if is_feature_ablation_experiment(config.conditions):
        return write_no_residual_report(summary, paired, validation, data_hash, config, mode, seeds)
    paths = output_paths(config)
    auroc = summary[summary["metric"] == "auroc"].set_index("condition")
    local_experiment = LOCAL_ENERGY_CONDITION in config.conditions
    main_condition = LOCAL_ENERGY_CONDITION if local_experiment else "recon_physics_bce"
    if local_experiment:
        delta_physics = paired[(paired["comparison"] == "local_energy_minus_pointwise_physics") &
                               (paired["metric"] == "auroc")].iloc[0]
        delta_recon = paired[(paired["comparison"] == "local_energy_minus_recon_bce") &
                             (paired["metric"] == "auroc")].iloc[0]
        delta_baseline = paired[(paired["comparison"] == "local_energy_minus_waveform_transformer") &
                                (paired["metric"] == "auroc")].iloc[0]
        hoyer_delta = diagnostic_paired[diagnostic_paired["metric"] == "local_energy_hoyer"].iloc[0]
        activity_delta = diagnostic_paired[
            diagnostic_paired["metric"] == "local_energy_active_ratio"].iloc[0]
        block_sparse_gain = bool(hoyer_delta["ci_low"] > 0.0)
    else:
        delta_physics = paired[(paired["comparison"] == "physics_minus_recon_bce") &
                               (paired["metric"] == "auroc")].iloc[0]
        delta_recon = delta_physics
        delta_baseline = paired[(paired["comparison"] == "physics_minus_waveform_transformer") &
                                (paired["metric"] == "auroc")].iloc[0]
        hoyer_delta = activity_delta = None
        block_sparse_gain = False
    has_capacity = bool(auroc.loc[main_condition, "ci_low"] > 0.5)
    physics_gain = bool(delta_physics["ci_low"] > 0.0)
    overall_gain = bool(delta_baseline["ci_low"] > 0.0)
    physical_rows = metrics[metrics["condition"] == main_condition]
    mean_ncc = float(physical_rows["mean_ncc_source_input"].mean())
    mean_tail = float(physical_rows["mean_response_tail_energy"].mean())
    diversity = float(physical_rows["response_diversity"].mean())
    degenerate = bool(mean_ncc >= 0.95 and mean_tail <= 1e-4)

    table_rows = []
    metric_labels = {"auroc": "AUROC", "auprc": "AUPRC", "balanced_accuracy": "Balanced accuracy",
                     "f1": "F1", "sensitivity": "灵敏度", "specificity": "特异度"}
    for condition in config.conditions:
        values = []
        for metric_name in metric_labels:
            row = summary[(summary["condition"] == condition) & (summary["metric"] == metric_name)].iloc[0]
            values.append(f"{row['estimate']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        table_rows.append(f"| {CONDITION_LABELS[condition]} | " + " | ".join(values) + " |")
    snr_rows = []
    if len(snr_summary):
        for _, row in snr_summary.iterrows():
            snr_rows.append(f"| {CONDITION_LABELS[row['condition']]} | [{row['snr_lower_db']:.0f}, {row['snr_upper_db']:.0f}] | "
                            f"{row['auroc_mean']:.4f} | {row['auprc_mean']:.4f} | {row['sensitivity_mean']:.4f} |")
    high_rows = []
    if len(high_comparison):
        for _, row in high_comparison.iterrows():
            high_rows.append(f"| {CONDITION_LABELS[row['condition']]} | {row['estimate']:+.4f} "
                             f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {int(row['bootstrap_runs_valid'])} |")
    contribution_rows = []
    for _, row in physics_contributions.iterrows():
        contribution_rows.append(
            f"| {CONDITION_LABELS[row['condition']]} | {row['mean_physics']:.6e} | "
            f"{row['mean_weighted_physics']:.6e} | {row['weighted_physics_fraction_of_total']:.6%} |")
    mixture_scope = ""
    snr_section = ""
    high_section = ""
    if config.positive_noise_mix:
        mixture_scope = f"""
- 固定混合种子：{config.mix_seed}；每条正样本目标SNR独立采自 `Uniform({config.snr_min_db:g}, {config.snr_max_db:g})` dB。
- 3500个负样本及原五折编号与高SNR基线完全一致；3500个混合背景均来自同录音的未选噪声，且与负样本及其他背景的10 ms窗口无重叠。
- Click与背景分别去直流、20–190 kHz零相位带通和Tukey窗，按完整5761点RMS定标；混合后统一peak归一化。
- 目标SNR范围为 [{validation['target_snr_min_db']:.6f}, {validation['target_snr_max_db']:.6f}] dB；最大实现误差为 {validation['maximum_snr_error_db']:.3e} dB。
"""
        snr_section = f"""
## 分SNR退化曲线

每个5 dB区间均使用该区间正样本与完整3500个负样本比较；阈值仍由对应验证折确定。

| 条件 | 目标SNR区间 (dB) | AUROC | AUPRC | 灵敏度 |
|---|---:|---:|---:|---:|
{chr(10).join(snr_rows)}
"""
        high_section = f"""
## 相对原高SNR实验的配对变化

下表为“低SNR实验−原高SNR实验”的配对AUROC差值；两侧复用相同训练种子、测试样本和录音组重采样。

| 条件 | AUROC差值 [95% CI] | 有效Bootstrap次数 |
|---|---:|---:|
{chr(10).join(high_rows)}
"""
    if local_experiment:
        physics_conclusion = f"""
- Local-Energy相对旧逐点Physics的分类改善：**{'是' if physics_gain else '否'}**；AUROC差值为 {delta_physics['estimate']:+.4f} [{delta_physics['ci_low']:+.4f}, {delta_physics['ci_high']:+.4f}]。
- Local-Energy相对 `Recon + BCE` 的AUROC差值为 {delta_recon['estimate']:+.4f} [{delta_recon['ci_low']:+.4f}, {delta_recon['ci_high']:+.4f}]。
- Local-Energy相对 Waveform Transformer 的AUROC差值为 {delta_baseline['estimate']:+.4f} [{delta_baseline['ci_low']:+.4f}, {delta_baseline['ci_high']:+.4f}]。
- 块状稀疏更明显：**{'是' if block_sparse_gain else '否'}**；局部能量Hoyer差值为 {hoyer_delta['estimate']:+.4f} [{hoyer_delta['ci_low']:+.4f}, {hoyer_delta['ci_high']:+.4f}]。
- 相对阈值活动块比例（阈值为各样本峰值的 {config.local_energy_activity_threshold:.0%}）差值为 {activity_delta['estimate']:+.4f} [{activity_delta['ci_low']:+.4f}, {activity_delta['ci_high']:+.4f}]。
"""
    else:
        physics_conclusion = f"""
- 物理损失带来分类增益：**{'是' if physics_gain else '否'}**；相对 `Recon + BCE` 的AUROC差值为 {delta_physics['estimate']:+.4f} [{delta_physics['ci_low']:+.4f}, {delta_physics['ci_high']:+.4f}]。
- 总体性能优于 Waveform Transformer：**{'是' if overall_gain else '否'}**；AUROC差值为 {delta_baseline['estimate']:+.4f} [{delta_baseline['ci_low']:+.4f}, {delta_baseline['ci_high']:+.4f}]。
"""
    report = f"""# {'低SNR噪声混合：' if config.positive_noise_mix else ''}双分支物理约束海豚 Click 分类实验报告

## 实验范围

- 运行模式：`{mode}`；样本数：{validation['sample_count']}（正/负各3500）。
{'- **本报告是单一预注册划分的初步结果，不能替代完整五折结论。**' if mode == 'quick' else '- 本报告汇总完整五折测试预测。'}
- 独立训练随机种子：{', '.join(str(seed) for seed in seeds)}（共{len(seeds)}个）。
- 数据 SHA-256：`{data_hash['sha256']}`。
- 所有输入均为576 kHz、5761点。
- 按原始录音编号严格隔离训练、验证和测试；来源泄漏计数为 {validation['source_group_leakage_count']}。
- 置信区间使用种子×录音组分层 {config.bootstrap_runs} 次 Bootstrap；配对比较复用相同的种子和录音组重采样。
- 单种子结果不能证明跨初始化稳定性；全部置信区间仅描述录音来源不确定性。
- Local-Energy使用窗口 {config.local_energy_window}、stride {config.local_energy_stride}、无padding；固定 `h0=1` 不参与约束。
{mixture_scope}

## 主要结果

| 条件 | AUROC [95% CI] | AUPRC [95% CI] | Balanced accuracy [95% CI] | F1 [95% CI] | 灵敏度 [95% CI] | 特异度 [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

- 完整模型具有当前数据分类能力：**{'是' if has_capacity else '否'}**（AUROC置信区间下限 {'>' if has_capacity else '≤'} 0.5）。
{physics_conclusion}

{snr_section}

{high_section}

## 分解退化审计

- Physics损失量级及加权贡献：

| 条件 | 原始Physics均值 | beta×Physics均值 | 占总损失比例 |
|---|---:|---:|---:|
{chr(10).join(contribution_rows)}

- `NCC(x_hat, y)` 均值：{mean_ncc:.6f}。
- `h_hat` 尾部能量均值：{mean_tail:.6e}。
- 传播响应多样性（随机样本对的1减余弦相似度）：{diversity:.6e}。
- 启发式退化标记（NCC≥0.95且尾部能量≤1e-4）：**{'触发' if degenerate else '未触发'}**。

即使 AUROC 较高，也不能由此自动声称完成信源—传播响应物理解耦。若上述诊断显示
`x_hat≈y, h_hat≈delta` 或响应多样性很低，结果只能证明监督分类有效。

## 方法边界

低SNR混合定标将原Click波形（包括其中少量既有背景）整体视为信号分量；负样本不重新抽样，
训练、验证和测试中的正样本均使用固定混合版本，不另保留干净正样本副本。

输入的一维性来自既有数据格式，本实验不称为“用户指定的单通道盲反卷积”。加入 BCE 后，
模型属于物理约束监督分类，而非纯自监督方法。统计结论只适用于当前录音来源及预注册判据。
"""
    destination = paths["root"] / "research_report_zh.md"
    destination.write_text(report, encoding="utf-8")
    json_dump(paths["root"] / "acceptance_decisions.json",
              {"evaluation_scope": mode, "full_five_fold": mode == "full",
               "seeds": list(seeds), "seed_count": len(seeds),
               "classification_capacity": has_capacity, "physics_loss_gain": physics_gain,
               "local_energy_classification_gain_over_pointwise": physics_gain if local_experiment else None,
               "local_energy_block_sparsity_gain": block_sparse_gain if local_experiment else None,
               "better_than_waveform_transformer": overall_gain, "degenerate_solution_flag": degenerate,
               "confidence_interval_scope": "recording_source_uncertainty_only" if len(seeds) == 1 else "seed_and_recording_source"})
    return destination


def smoke_tests(config: Config, manifest: pd.DataFrame, waveforms: np.ndarray,
                validation: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {"data_validation": validation}
    # 用双精度直接求和作为FFT因果卷积的独立参照。
    source = torch.zeros(2, 5761, dtype=torch.float64, device=device)
    response = torch.zeros(2, 1153, dtype=torch.float64, device=device)
    source[0, 17] = 1.0; source[0, 101] = -0.25
    source[1, 3] = 0.7; source[1, 301] = 0.2
    response[:, 0] = 1.0; response[0, 9] = 0.3; response[1, 1152] = -0.1
    fft_value = causal_fft_convolution(source, response)
    direct = torch.zeros_like(source)
    for k in (0, 9, 1152):
        if k < response.shape[1]:
            direct[:, k:] += response[:, k, None] * source[:, :source.shape[1] - k]
    convolution_error = float(torch.max(torch.abs(fft_value - direct)).cpu())
    results["causal_convolution_max_error"] = convolution_error
    results["causal_convolution_passed"] = convolution_error < 1e-6

    no_residual = is_no_residual_experiment(config.conditions)
    source_only = is_source_only_experiment(config.conditions)
    model = PhysicalClassifier(config.sample_count, config.response_count,
                               include_residual=not no_residual and not source_only,
                               include_response_feature=not source_only).to(device)
    batch = torch.as_tensor(np.asarray(waveforms[:2]).copy(), dtype=torch.float32, device=device)
    labels = torch.as_tensor(manifest["label"].to_numpy(dtype=np.float32)[:2], device=device)
    outputs = model(batch)
    if no_residual or source_only:
        outputs["residual"].retain_grad()
    results["source_shape"] = list(outputs["source"].shape)
    results["response_shape"] = list(outputs["response"].shape)
    results["classification_head_input_features"] = int(model.classification_head[0].in_features)
    results["residual_classifier_absent"] = not hasattr(model, "residual_classifier")
    results["response_classifier_absent"] = not hasattr(model, "response_classifier")
    results["response_h0_exact"] = bool(torch.equal(outputs["response"][:, 0], torch.ones(2, device=device)))
    results["source_is_decoder_tensor"] = bool(outputs["source"].grad_fn is not None and
                                                outputs["source"].data_ptr() != batch.data_ptr())
    loss_condition = (LOCAL_ENERGY_CONDITION if LOCAL_ENERGY_CONDITION in config.conditions
                      else "recon_physics_bce_source_only" if source_only
                      else "recon_physics_bce_no_residual" if no_residual else "recon_physics_bce")
    losses = model_losses(outputs, batch, labels, loss_condition, config)
    results["losses_finite"] = {name: bool(torch.isfinite(value)) for name, value in losses.items()}
    model.zero_grad(set_to_none=True)
    nn.functional.binary_cross_entropy_with_logits(outputs["logit"], labels).backward()
    results["residual_outside_classification_gradient_path"] = bool(
        not (no_residual or source_only) or outputs["residual"].grad is None)
    modules = {"classification_head": model.classification_head, "source_decoder": model.source_decoder,
               "response_decoder": model.response_decoder, "shared_encoder": model.shared_encoder}
    results["bce_only_gradient_norms"] = {name: gradient_norm(module) for name, module in modules.items()}
    required_bce_modules = ("classification_head", "source_decoder", "shared_encoder")
    results["bce_gradient_passed"] = bool(all(
        np.isfinite(results["bce_only_gradient_norms"][name])
        and results["bce_only_gradient_norms"][name] > 0 for name in required_bce_modules)
        and (not source_only or results["bce_only_gradient_norms"]["response_decoder"] == 0.0))
    manual_response = torch.zeros(4, config.response_count, dtype=torch.float64, device=device,
                                  requires_grad=True)
    with torch.no_grad():
        manual_response[:, 0] = 1.0
        manual_response[1, 10:80] = 0.5
        manual_response[2, 100] = 1.0
        manual_response[3, 1:] = 0.1
    curve = local_energy_curve(manual_response, config.local_energy_window,
                               config.local_energy_stride, config.epsilon)
    hand_first = torch.sqrt(torch.mean(manual_response[1, 1:1 + config.local_energy_window].square())
                            + config.epsilon) - math.sqrt(config.epsilon)
    curve.mean().backward()
    h0_probe = manual_response.detach().clone(); h0_probe[:, 0] = 123.0
    results["local_energy_curve_shape"] = list(curve.shape)
    results["local_energy_manual_pool_error"] = float(torch.abs(curve[1, 0] - hand_first).cpu())
    results["local_energy_zero_tail_exact_zero"] = bool(curve[0].eq(0).all())
    results["local_energy_patterns_finite"] = bool(torch.isfinite(curve).all())
    results["local_energy_gradients_finite"] = bool(torch.isfinite(manual_response.grad[:, 1:]).all())
    results["local_energy_h0_gradient_zero"] = bool(manual_response.grad[:, 0].eq(0).all())
    results["local_energy_h0_invariant"] = bool(torch.equal(
        local_energy_curve(h0_probe, config.local_energy_window, config.local_energy_stride, config.epsilon),
        curve.detach()))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.response_decoder.layers[-1].weight.normal_(0.0, 1e-3)
    local_outputs = model(batch)
    physics_loss_for_condition(local_outputs["response"], loss_condition, config).backward()
    results["local_energy_only_gradient_norms"] = {
        "response_decoder": gradient_norm(model.response_decoder),
        "shared_encoder": gradient_norm(model.shared_encoder),
    }
    results["local_energy_only_gradient_passed"] = bool(all(
        np.isfinite(value) and value > 0 for value in results["local_energy_only_gradient_norms"].values()))
    results["passed"] = bool(validation["passed"] and results["causal_convolution_passed"]
                             and results["source_shape"] == [2, 5761]
                             and results["response_shape"] == [2, 1153]
                             and results["classification_head_input_features"] == (32 if source_only else 64 if no_residual else 96)
                             and results["residual_classifier_absent"] == (no_residual or source_only)
                             and results["response_classifier_absent"] == source_only
                             and results["residual_outside_classification_gradient_path"]
                             and results["response_h0_exact"] and results["source_is_decoder_tensor"]
                             and all(results["losses_finite"].values()) and results["bce_gradient_passed"]
                             and results["local_energy_curve_shape"] == [4, 1096]
                             and results["local_energy_manual_pool_error"] < 1e-12
                             and results["local_energy_zero_tail_exact_zero"]
                             and results["local_energy_patterns_finite"]
                             and results["local_energy_gradients_finite"]
                             and results["local_energy_h0_gradient_zero"]
                             and results["local_energy_h0_invariant"]
                             and results["local_energy_only_gradient_passed"])
    json_dump(output_paths(config)["root"] / "smoke_test_results.json", results)
    return results


def serializable_config(config: Config, mode: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    result = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    result.update({"mode": mode, "seeds": list(seeds), "gamma": config.gamma,
                   "conditions": list(config.conditions),
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                   "torch_version": torch.__version__, "config_hash": config.config_hash})
    return result


def verify_artifacts(config: Config, mode: str, seeds: tuple[int, ...], manifest: pd.DataFrame,
                     metrics: pd.DataFrame, predictions: pd.DataFrame,
                     summary: pd.DataFrame, paired: pd.DataFrame,
                     snr_metrics: pd.DataFrame, high_comparison: pd.DataFrame,
                     diagnostic_paired: pd.DataFrame, physics_contributions: pd.DataFrame) -> dict[str, Any]:
    paths = output_paths(config)
    folds = [0] if mode == "quick" else list(range(config.n_splits))
    expected_models = len(config.conditions) * len(seeds) * len(folds)
    expected_predictions = len(config.conditions) * len(seeds) * int(manifest["test_fold"].isin(folds).sum())
    expected_names = {f"{condition}_seed_{seed}_fold_{fold}.pt"
                      for condition in config.conditions for seed in seeds for fold in folds}
    model_paths = [paths["models"] / name for name in sorted(expected_names)]
    completed_models = 0
    for path in model_paths:
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=True)
            completed_models += int(saved.get("training_complete", False)
                                    and saved.get("config_hash") == config.config_hash)
    history_names = {f"{condition}_seed_{seed}_fold_{fold}_history.csv"
                     for condition in config.conditions for seed in seeds for fold in folds}
    batch_names = {f"{condition}_seed_{seed}_fold_{fold}_batch.csv"
                   for condition in config.conditions for seed in seeds for fold in folds}
    seed_metrics_path = paths["metrics"] / "seed_metrics.csv"
    seed_metric_rows = len(pd.read_csv(seed_metrics_path)) if seed_metrics_path.exists() else 0
    bootstrap_frames = [summary, paired]
    if config.positive_noise_mix and not is_feature_ablation_experiment(config.conditions):
        bootstrap_frames.append(high_comparison)
    bootstrap_valid = bool(all(len(frame) and
                               frame["bootstrap_runs_valid"].eq(config.bootstrap_runs).all()
                               for frame in bootstrap_frames))
    checks = {
        "expected_model_count": expected_models,
        "weight_file_count": int(sum(path.exists() for path in model_paths)),
        "training_complete_weight_count": completed_models,
        "history_log_count": int(sum((paths["logs"] / name).exists() for name in history_names)),
        "batch_log_count": int(sum((paths["logs"] / name).exists() for name in batch_names)),
        "fold_metric_rows": int(len(metrics)),
        "expected_seed_metric_rows": len(config.conditions) * len(seeds),
        "seed_metric_rows": int(seed_metric_rows),
        "expected_prediction_rows": expected_predictions,
        "prediction_rows": int(len(predictions)),
        "prediction_key_unique": bool(not predictions.duplicated(
            ["condition", "seed", "sample_id"]).any()),
        "bootstrap_all_valid": bootstrap_valid,
        "bootstrap_summary_rows": int(len(summary)),
        "expected_bootstrap_summary_rows": len(config.conditions) * len(METRIC_NAMES),
        "paired_difference_rows": int(len(paired)),
        "expected_paired_difference_rows": (2 * len(METRIC_NAMES)
                                            if is_feature_ablation_experiment(config.conditions)
                                            else len(paired)),
        "snr_metric_rows": int(len(snr_metrics)),
        "expected_snr_metric_rows": len(config.conditions) * len(seeds) * 6 if config.positive_noise_mix else 0,
        "physics_diagnostic_rows": int(len(diagnostic_paired)),
        "expected_physics_diagnostic_rows": 2 if LOCAL_ENERGY_CONDITION in config.conditions else 0,
        "physics_contribution_rows": int(len(physics_contributions)),
        "expected_physics_contribution_rows": len(config.conditions),
        "local_energy_curve_fold_files": len(list(paths["metrics"].glob("*_local_energy_curve.csv"))),
        "expected_local_energy_curve_fold_files": int(sum(
            condition != "waveform_transformer" for condition in config.conditions) * len(seeds) * len(folds)),
        "local_energy_curve_summary_exists": (paths["metrics"] / "local_energy_curves.csv").exists(),
    }
    pairing_path = paths["root"] / "pairing_validation.json"
    checks["pairing_validation_passed"] = (bool(json.loads(pairing_path.read_text(
        encoding="utf-8"))["passed"]) if is_feature_ablation_experiment(config.conditions)
        and pairing_path.exists() else not is_feature_ablation_experiment(config.conditions))
    checks["passed"] = bool(checks["weight_file_count"] == expected_models
                            and checks["training_complete_weight_count"] == expected_models
                            and checks["history_log_count"] == expected_models
                            and checks["batch_log_count"] == expected_models
                            and checks["fold_metric_rows"] == expected_models
                            and checks["seed_metric_rows"] == checks["expected_seed_metric_rows"]
                            and checks["prediction_rows"] == expected_predictions
                            and checks["prediction_key_unique"] and checks["bootstrap_all_valid"]
                            and checks["bootstrap_summary_rows"] == checks["expected_bootstrap_summary_rows"]
                            and checks["paired_difference_rows"] == checks["expected_paired_difference_rows"]
                            and checks["pairing_validation_passed"]
                            and checks["snr_metric_rows"] == checks["expected_snr_metric_rows"]
                            and checks["physics_diagnostic_rows"] == checks["expected_physics_diagnostic_rows"]
                            and checks["physics_contribution_rows"] == checks["expected_physics_contribution_rows"]
                            and checks["local_energy_curve_fold_files"] == checks["expected_local_energy_curve_fold_files"]
                            and checks["local_energy_curve_summary_exists"])
    json_dump(paths["root"] / "artifact_validation.json", checks)
    return checks


def run(config: Config, mode: str, seeds: tuple[int, ...], force: bool, skip_report: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    output_paths(config)
    seed_everything(config.random_seed)
    json_dump(config.output_dir / "run_config.json", serializable_config(config, mode, seeds))
    manifest = build_manifest(config, force)
    waveforms, audit = materialize_waveforms(manifest, config, force)
    # 统一以落盘清单为哈希和后续分析输入，避免首次内存浮点与CSV回读格式造成哈希漂移。
    manifest = pd.read_csv(config.output_dir / "dataset_manifest.csv")
    validation = validate_data(manifest, waveforms, audit, config)
    if not validation["passed"]:
        raise RuntimeError(f"数据验收失败：{validation}")
    data_hash = compute_data_hash(manifest, waveforms, config)
    if config.positive_noise_mix and (LOCAL_ENERGY_CONDITION in config.conditions
                                      or is_feature_ablation_experiment(config.conditions)):
        reference_hash_path = LOW_SNR_REFERENCE_OUTPUT / "data_hash.json"
        if not reference_hash_path.exists():
            raise FileNotFoundError(f"缺少既有低SNR数据哈希：{reference_hash_path}")
        reference_hash = json.loads(reference_hash_path.read_text(encoding="utf-8"))["sha256"]
        if data_hash["sha256"] != reference_hash:
            raise RuntimeError("新Manifest/波形与既有低SNR实验数据哈希不一致，停止训练："
                               f"new={data_hash['sha256']}, reference={reference_hash}")
    print("总体阶段 2/5：尺寸、卷积、损失和梯度验收")
    smoke = smoke_tests(config, manifest, waveforms, validation)
    if not smoke["passed"]:
        raise RuntimeError(f"Smoke测试失败：{smoke}")
    result: dict[str, Any] = {"mode": mode, "seeds": list(seeds), "config_hash": config.config_hash,
                              "data_hash": data_hash["sha256"], "smoke": smoke}
    if mode == "smoke":
        result["elapsed_s"] = time.perf_counter() - started
        json_dump(config.output_dir / "run_summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    metrics, predictions = train_experiments(waveforms, manifest, config, mode, seeds, force)
    summarize_seed_metrics(predictions, config)
    physics_contributions = summarize_physics_contributions(config, seeds, mode)
    snr_metrics, snr_summary = snr_binned_metrics(predictions, config)
    summary, paired = bootstrap_results(predictions, config)
    diagnostic_paired = bootstrap_physics_diagnostics(predictions, config)
    high_comparison = (pd.DataFrame() if is_feature_ablation_experiment(config.conditions)
                       else compare_with_high_snr(predictions, config))
    print("总体阶段 5/5：图表与产物验收")
    plot_results(predictions, config)
    report = None if skip_report else write_report(
        metrics, summary, paired, snr_summary, high_comparison, diagnostic_paired,
        physics_contributions, validation, data_hash, config, mode, seeds)
    artifacts = verify_artifacts(config, mode, seeds, manifest, metrics, predictions,
                                 summary, paired, snr_metrics, high_comparison,
                                 diagnostic_paired, physics_contributions)
    result.update({"fold_metrics": str((config.output_dir / "metrics" / "fold_metrics.csv").resolve()),
                   "predictions": str((config.output_dir / "predictions" / "sample_predictions.csv").resolve()),
                   "report": None if report is None else str(report.resolve()),
                   "skip_report": skip_report, "artifact_validation": artifacts,
                   "elapsed_s": time.perf_counter() - started})
    json_dump(config.output_dir / "run_summary.json", result)
    if not artifacts["passed"]:
        raise RuntimeError(f"实验产物验收失败：{artifacts}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "quick", "full"))
    parser.add_argument("--force", action="store_true", help="显式重建检查点和结果")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20_260_826],
                        help="一个或多个独立训练随机种子")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs",
                        help="输出目录（建议使用绝对路径）")
    parser.add_argument("--positive-noise-mix", action="store_true",
                        help="为正样本固定配对同录音背景并按目标SNR混合")
    parser.add_argument("--snr-min-db", type=float, default=-15.0, help="混合目标SNR下限(dB)")
    parser.add_argument("--snr-max-db", type=float, default=15.0, help="混合目标SNR上限(dB)")
    parser.add_argument("--mix-seed", type=int, default=20_260_826, help="背景配对与SNR固定种子")
    parser.add_argument("--conditions", nargs="+", choices=ALL_CONDITIONS,
                        default=list(DEFAULT_CONDITIONS), help="选择实验条件；默认保持历史三条件")
    parser.add_argument("--skip-report", action="store_true",
                        help="保留全部数值、图表和验收产物，但不生成或修改Markdown报告")
    parser.add_argument("--local-energy-window", type=int, default=57,
                        help="Local-Energy AvgPool1d窗口点数")
    parser.add_argument("--local-energy-stride", type=int, default=1,
                        help="Local-Energy AvgPool1d步长")
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if arguments.positive_noise_mix and not arguments.snr_min_db < arguments.snr_max_db:
        raise ValueError("--snr-min-db 必须小于 --snr-max-db")
    if not 1 <= arguments.local_energy_window <= 1152:
        raise ValueError("--local-energy-window 必须介于1和1152之间")
    if arguments.local_energy_stride < 1:
        raise ValueError("--local-energy-stride 必须为正整数")
    conditions = tuple(dict.fromkeys(arguments.conditions))
    config = Config(output_dir=output_dir, positive_noise_mix=arguments.positive_noise_mix,
                    snr_min_db=arguments.snr_min_db, snr_max_db=arguments.snr_max_db,
                    mix_seed=arguments.mix_seed, conditions=conditions,
                    local_energy_window=arguments.local_energy_window,
                    local_energy_stride=arguments.local_energy_stride)
    seeds = tuple(dict.fromkeys(arguments.seeds))
    if not seeds:
        raise ValueError("至少需要一个随机种子")
    run(config, arguments.mode, seeds, arguments.force, arguments.skip_report)


if __name__ == "__main__":
    main()
