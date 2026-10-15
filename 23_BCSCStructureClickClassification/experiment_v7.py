#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v7：OMP 多径全局软标签辅助分类。

命令：verify、extract-omp、train、report、export、all。
只读取 v4 的清单和加噪分类波形；OMP 使用 manifest 的正样本 wav_path 与
ClickTrains 直达模板，绝不读取 BCSC 缓存。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE = ROOT / "23_BCSCStructureClickClassification" / "v4_rebalanced_snr_m5_p15_step2_3seeds"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v7_omp_global_softlabel_proxy"
CLICK_TRAINS = ROOT / "00_Data" / "ClickTrains"
N_SAMPLES, WAVE_SAMPLES, SAMPLE_RATE, CENTER = 7000, 5761, 576_000, 2880
MIN_DELAY, MAX_DELAY, ATOM_COUNT, AUX_DIM = 20, 2880, 15, 6
SEED, TEST_FOLDS = 20260826, (0, 2)
CONDITIONS = ("C0_waveform_control", "C1_aux_only", "C2_proxy_no_aux",
              "C3_full_proxy", "C4_shuffled_proxy")
CONDITION_NAMES = {
    "C0_waveform_control": "C0 Waveform-control", "C1_aux_only": "C1 Aux-only",
    "C2_proxy_no_aux": "C2 Proxy-no-aux", "C3_full_proxy": "C3 Full-proxy",
    "C4_shuffled_proxy": "C4 Shuffled-proxy",
}
TARGET_NAMES = ("mean_extra_delay_us", "rms_delay_spread_us", "max_extra_delay_us",
                "k_db", "total_gain", "coherence_bandwidth_hz")
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SOURCE / "dataset_manifest.csv"
    waveform_path: Path = SOURCE / "cache" / "preprocessed_waveforms.npy"
    batch_size: int = 64
    max_epochs: int = 12
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    auxiliary_weight: float = .1
    dropout: float = .1
    device: str = "auto"


def output_paths(config: Config) -> dict[str, Path]:
    result = {name: config.output_dir / name for name in
              ("labels", "models", "metrics", "predictions", "deployment", "tests")}
    result["root"] = config.output_dir
    for item in result.values(): item.mkdir(parents=True, exist_ok=True)
    return result


def dump_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path): return str(item)
        if isinstance(item, (np.floating, float)): return float(item) if math.isfinite(float(item)) else None
        if isinstance(item, (np.integer, int)): return int(item)
        if isinstance(item, (str, bool)) or item is None: return item
        if isinstance(item, dict): return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [convert(v) for v in item]
        return str(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def device_for(value: str) -> torch.device:
    if value == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("指定 CUDA，但当前环境没有 CUDA")
    return device


def append_record(config: Config, command: str, started: datetime, ended: datetime, status: str, detail: str) -> None:
    record = config.output_dir / "实验记录.md"
    if not record.exists(): return
    row = f"| {started.isoformat(timespec='seconds')} | {ended.isoformat(timespec='seconds')} | {command} | {status} | `{config.output_dir}` | {detail} |\n"
    with record.open("a", encoding="utf-8") as handle:
        handle.write("\n## 实际调用\n\n| 开始时间 | 结束时间 | 命令 | 状态 | 实际输出路径 | 事实记录 |\n| --- | --- | --- | --- | --- | --- |\n" + row)


def parse_sample_id(sample_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"P_\d+_(\d+)_Pulse_(\d+)", str(sample_id))
    if match is None: raise RuntimeError(f"无法解析正样本 sample_id：{sample_id}")
    return int(match.group(1)), int(match.group(2))


def template_path(sample_id: str) -> Path:
    train, pulse = parse_sample_id(sample_id)
    return CLICK_TRAINS / f"PulseTrain_{train:03d}" / f"Pulse_{pulse}.wav"


def split(manifest: pd.DataFrame, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = manifest.test_fold.to_numpy(dtype=int)
    validation_fold = (test_fold + 1) % 5
    return (np.flatnonzero((folds != test_fold) & (folds != validation_fold)),
            np.flatnonzero(folds == validation_fold), np.flatnonzero(folds == test_fold))


def load_data(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    if not config.manifest_path.exists() or not config.waveform_path.exists():
        raise FileNotFoundError("缺少 v4 清单或预处理波形")
    manifest = pd.read_csv(config.manifest_path).reset_index(drop=True)
    waves = np.load(config.waveform_path, mmap_mode="r")
    positives = manifest.label.eq(1)
    snr = set(manifest.loc[positives, "target_snr_db"].astype(float).unique())
    expected_snr = {-5., -3., -1., 1., 3., 5., 7., 9., 11., 13., 15.}
    checks = {
        "shape": len(manifest) == N_SAMPLES and tuple(waves.shape) == (N_SAMPLES, WAVE_SAMPLES),
        "classes": manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
        "groups": manifest.ori_file_num.nunique() == 34,
        "folds": set(manifest.test_fold.astype(int).unique()) == set(range(5)),
        "group_isolation": manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
        "snr": snr == expected_snr,
        "finite_waves": bool(np.isfinite(waves).all()),
    }
    if not all(checks.values()): raise RuntimeError(f"v4 数据合同失败：{checks}")
    for fold in range(5):
        levels = set(manifest.loc[positives & manifest.test_fold.eq(fold), "target_snr_db"].astype(float))
        if levels != expected_snr: raise RuntimeError(f"test_fold={fold} 未覆盖全部 SNR 档")
    return manifest, waves


def read_wave(path: Path) -> np.ndarray:
    rate, wave = wavfile.read(path, mmap=True)
    if rate != SAMPLE_RATE: raise RuntimeError(f"采样率不是 576 kHz：{path}")
    value = np.asarray(wave, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all(): raise RuntimeError(f"非有限或非单通道波形：{path}")
    return value


def positive_source_contract(manifest: pd.DataFrame) -> dict[str, int]:
    positive = manifest.loc[manifest.label.eq(1)]
    wavs = [Path(value) for value in positive.wav_path]
    templates = [template_path(value) for value in positive.sample_id]
    if len(set(wavs)) != 3500 or len(set(templates)) != 3500: raise RuntimeError("正样本接收波形或直达模板不是一一对应")
    for path in tqdm(wavs, desc="验证干净接收波形", unit="样本"):
        value = read_wave(path)
        if len(value) != WAVE_SAMPLES: raise RuntimeError(f"干净接收波形长度不是 5761：{path}")
    for path in tqdm(templates, desc="验证直达模板", unit="样本"):
        read_wave(path)
    return {"positive_wav_count": len(wavs), "template_count": len(templates)}


def shifted_atom(template: np.ndarray, peak: int, delay: int) -> np.ndarray:
    atom = np.zeros(WAVE_SAMPLES, dtype=np.float64)
    start = CENTER - peak + delay
    source_start = max(0, -start)
    target_start = max(0, start)
    count = min(len(template) - source_start, WAVE_SAMPLES - target_start)
    if count > 0: atom[target_start:target_start + count] = template[source_start:source_start + count]
    return atom


def omp_atoms(signal: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(signal) != WAVE_SAMPLES or not np.isfinite(signal).all() or not np.isfinite(template).all():
        raise RuntimeError("OMP 输入必须是有限的 5761 点接收波形与有限模板")
    peak = int(np.argmax(np.abs(template)))
    if not np.any(template): raise RuntimeError("OMP 模板能量为零")
    candidates = np.arange(MIN_DELAY, MAX_DELAY + 1, dtype=int)
    atoms = [shifted_atom(template, peak, 0)]
    selected = [0]
    direct_matrix = np.column_stack(atoms)
    direct_coefficient, _, direct_rank, _ = np.linalg.lstsq(direct_matrix, signal, rcond=None)
    if direct_rank != 1: raise RuntimeError("OMP 直达原子最小二乘出现秩异常")
    residual = signal - direct_matrix @ direct_coefficient
    for _ in range(ATOM_COUNT - 1):
        correlations = np.empty(len(candidates), dtype=np.float64)
        for position, delay in enumerate(candidates):
            if int(delay) in selected:
                correlations[position] = -np.inf
                continue
            atom = shifted_atom(template, peak, int(delay))
            norm = np.linalg.norm(atom)
            correlations[position] = abs(np.dot(atom, residual)) / (norm + EPS)
        delay = int(candidates[int(np.argmax(correlations))])
        selected.append(delay); atoms.append(shifted_atom(template, peak, delay))
        matrix = np.column_stack(atoms)
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, signal, rcond=None)
        if rank != len(atoms): raise RuntimeError("OMP 联合最小二乘出现秩异常")
        residual = signal - matrix @ coefficients
    matrix = np.column_stack(atoms)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, signal, rcond=None)
    if rank != ATOM_COUNT or len(selected) != ATOM_COUNT or len(set(selected)) != ATOM_COUNT:
        raise RuntimeError("OMP 未得到 15 个不重复原子")
    return np.asarray(selected, dtype=np.int16), coefficients.astype(np.float64)


def physical_label(delays: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    powers = coefficients ** 2
    delay_us = delays.astype(np.float64) / SAMPLE_RATE * 1e6
    power_total = powers.sum()
    mean = (powers * delay_us).sum() / (power_total + EPS)
    spread = math.sqrt((powers * (delay_us - mean) ** 2).sum() / (power_total + EPS))
    extra_power = powers[1:].sum()
    k_db = 10. * math.log10(powers[0] / (extra_power + EPS))
    bandwidth = 1. / (5. * (spread * 1e-6) + EPS)
    label = np.asarray((mean, spread, delay_us[1:].max(), k_db, power_total, bandwidth), dtype=np.float64)
    if not np.isfinite(label).all(): raise RuntimeError("OMP 物理标签出现非有限数")
    return label


def omp_paths(config: Config) -> dict[str, Path]:
    labels = output_paths(config)["labels"]
    return {"archive": labels / "omp_labels.npz", "metadata": labels / "omp_metadata.json"}


def validate_omp_archive(config: Config, manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = omp_paths(config)["archive"]
    if not archive.exists(): raise FileNotFoundError("请先执行 extract-omp")
    with np.load(archive, allow_pickle=True) as item:
        labels = item["physical_labels"].astype(np.float64)
        coefficients = item["coefficients"].astype(np.float64)
        delays = item["delays"].astype(np.int16)
        valid = item["aux_valid"].astype(bool)
        sample_id = item["sample_id"].astype(str)
    positive = manifest.label.to_numpy(dtype=int) == 1
    checks = {
        "sample_order": np.array_equal(sample_id, manifest.sample_id.astype(str).to_numpy()),
        "labels": labels.shape == (N_SAMPLES, AUX_DIM) and np.isfinite(labels).all(),
        "coefficients": coefficients.shape == (N_SAMPLES, ATOM_COUNT) and np.isfinite(coefficients).all(),
        "delays": delays.shape == (N_SAMPLES, ATOM_COUNT),
        "positive_valid": bool(valid[positive].all()),
        "negative_masked": bool((~valid[~positive]).all() and np.all(labels[~positive] == 0.) and np.all(coefficients[~positive] == 0.) and np.all(delays[~positive] == 0)),
        "atoms": bool(np.all(delays[positive, 0] == 0) and np.all((delays[positive, 1:] >= MIN_DELAY) & (delays[positive, 1:] <= MAX_DELAY)) and all(len(set(row)) == ATOM_COUNT for row in delays[positive])),
    }
    if not all(checks.values()): raise RuntimeError(f"OMP 标签合同失败：{checks}")
    return labels, coefficients, delays, valid


def extract_omp(config: Config) -> None:
    manifest, _ = load_data(config); archive_paths = omp_paths(config)
    if archive_paths["archive"].exists():
        validate_omp_archive(config, manifest)
        print("OMP 标签已通过完整合同检查，安全复用。", flush=True)
        return
    labels = np.zeros((N_SAMPLES, AUX_DIM), dtype=np.float64)
    coefficients = np.zeros((N_SAMPLES, ATOM_COUNT), dtype=np.float64)
    delays = np.zeros((N_SAMPLES, ATOM_COUNT), dtype=np.int16)
    valid = np.zeros(N_SAMPLES, dtype=bool)
    positive_indices = np.flatnonzero(manifest.label.to_numpy(dtype=int) == 1)
    bar = tqdm(positive_indices, total=len(positive_indices), desc="总体进度：OMP 标签", unit="样本")
    for index in bar:
        row = manifest.iloc[int(index)]
        signal, template = read_wave(Path(row.wav_path)), read_wave(template_path(row.sample_id))
        selected, fitted = omp_atoms(signal, template)
        delays[index], coefficients[index], labels[index], valid[index] = selected, fitted, physical_label(selected, fitted), True
        bar.set_postfix(sample=int(index), sample_id=row.sample_id)
    np.savez_compressed(archive_paths["archive"], physical_labels=labels, coefficients=coefficients,
                        delays=delays, aux_valid=valid, sample_id=np.asarray(manifest.sample_id.astype(str).to_numpy(), dtype="U"))
    validate_omp_archive(config, manifest)
    dump_json(archive_paths["metadata"], {"sample_rate_hz": SAMPLE_RATE, "center_index": CENTER,
              "delay_samples": [MIN_DELAY, MAX_DELAY], "atom_count": ATOM_COUNT,
              "selection": "absolute residual correlation normalized by atom norm", "least_squares": "joint float64 signed coefficients",
              "labels": list(TARGET_NAMES), "positive_count": int(valid.sum()), "negative_count": int((~valid).sum())})


class ClickDataset(Dataset):
    def __init__(self, waves: np.ndarray, labels: np.ndarray, targets: np.ndarray, valid: np.ndarray, indices: np.ndarray) -> None:
        self.waves, self.labels, self.targets, self.valid, self.indices = waves, labels, targets, valid, indices
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[position])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32),
                torch.as_tensor(self.targets[index], dtype=torch.float32), torch.tensor(self.valid[index], dtype=torch.bool), index)


def make_loader(waves: np.ndarray, labels: np.ndarray, targets: np.ndarray, valid: np.ndarray, indices: np.ndarray,
                batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(ClickDataset(waves, labels, targets, valid, indices), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0, pin_memory=torch.cuda.is_available())


class WaveformEncoder(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv1d(1, 64, kernel_size=32, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64)); nn.init.normal_(self.cls_token, std=.02)
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(wave.unsqueeze(1)).transpose(1, 2)
        position = torch.arange(tokens.shape[1], device=wave.device, dtype=tokens.dtype).unsqueeze(-1)
        dimension = torch.arange(64, device=wave.device, dtype=tokens.dtype).unsqueeze(0)
        scale = torch.pow(torch.tensor(10000., device=wave.device, dtype=tokens.dtype), 2 * torch.floor(dimension / 2) / 64)
        angle = position / scale
        positional = torch.where((dimension.long() % 2) == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        tokens = torch.cat((self.cls_token.expand(wave.shape[0], -1, -1), tokens + positional), dim=1)
        return self.encoder(tokens)[:, 0]


class OMPAuxClassifier(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.waveform = WaveformEncoder(config.dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.auxiliary = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, AUX_DIM))
    def forward(self, wave: torch.Tensor, detach_auxiliary: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.waveform(wave)
        return self.classifier(z).squeeze(1), self.auxiliary(z.detach() if detach_auxiliary else z)


class WaveformDeploymentModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__(); self.waveform = WaveformEncoder(config.dropout); self.classifier = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, wave: torch.Tensor) -> torch.Tensor: return self.classifier(self.waveform(wave)).squeeze(1)


def standardize_targets(labels: np.ndarray, valid: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use = train_indices[valid[train_indices]]
    median = np.median(labels[use], axis=0)
    iqr = np.percentile(labels[use], 75, axis=0) - np.percentile(labels[use], 25, axis=0)
    scale = np.where(iqr == 0., 1., iqr)
    target = np.zeros_like(labels, dtype=np.float32)
    target[valid] = np.clip((labels[valid] - median) / scale, -5., 5.).astype(np.float32)
    return target, median, scale


def shuffled_train_targets(target: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, fold: int) -> np.ndarray:
    result = target.copy(); positive = train_indices[labels[train_indices] == 1]
    rng = np.random.default_rng(SEED + 10000 + fold)
    result[positive] = target[rng.permutation(positive)]
    return result


def sigmoid(value: np.ndarray) -> np.ndarray: return 1. / (1. + np.exp(-np.clip(value, -80., 80.)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., logits / temperature) - labels * logits / temperature))
    result = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.


def threshold_for(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0., .5, 1.], probabilities)))
    return float(candidates[int(np.argmax([balanced_accuracy_score(labels, probabilities >= value) for value in candidates]))])


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)), "auprc": float(average_precision_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "recall": float(recall_score(labels, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)), "f1": float(f1_score(labels, predicted, zero_division=0)),
            "brier": float(brier_score_loss(labels, probabilities)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@torch.no_grad()
def predict(model: OMPAuxClassifier, data: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval(); collected: dict[str, list[np.ndarray]] = {name: [] for name in ("index", "label", "logit", "auxiliary")}
    for wave, label, _, _, index in data:
        logit, auxiliary = model(wave.to(device, non_blocking=True))
        collected["index"].append(np.asarray(index)); collected["label"].append(label.numpy())
        collected["logit"].append(logit.cpu().numpy()); collected["auxiliary"].append(auxiliary.cpu().numpy())
    return {name: np.concatenate(value) for name, value in collected.items()}


def model_dir(config: Config, condition: str, fold: int) -> Path:
    result = output_paths(config)["models"] / condition / f"fold_{fold}"
    result.mkdir(parents=True, exist_ok=True); return result


def train_one(config: Config, condition: str, fold: int, manifest: pd.DataFrame, waves: np.ndarray,
              labels: np.ndarray, valid: np.ndarray, progress: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    directory = model_dir(config, condition, fold)
    complete, checkpoint = directory / "COMPLETE.json", directory / "best_checkpoint.pt"
    prediction_path, auxiliary_path = directory / "test_predictions.csv", directory / "auxiliary_mae.csv"
    if all(path.exists() for path in (complete, checkpoint, prediction_path, auxiliary_path)):
        result = json.loads(complete.read_text(encoding="utf-8"))
        if result.get("condition") == condition and result.get("test_fold") == fold:
            return result, pd.read_csv(prediction_path), pd.read_csv(auxiliary_path)
    hard_labels = manifest.label.to_numpy(dtype=np.int64)
    train_idx, valid_idx, test_idx = split(manifest, fold)
    standardized, median, scale = standardize_targets(labels, valid, train_idx)
    train_targets = shuffled_train_targets(standardized, hard_labels, train_idx, fold) if condition == "C4_shuffled_proxy" else standardized
    uses_auxiliary = condition in ("C1_aux_only", "C3_full_proxy", "C4_shuffled_proxy")
    detach_auxiliary = condition == "C1_aux_only"
    seed_all(SEED + fold * 100); device = device_for(config.device); model = OMPAuxClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    validation_loader = make_loader(waves, hard_labels, standardized, valid, valid_idx, config.batch_size * 2, False)
    test_loader = make_loader(waves, hard_labels, standardized, valid, test_idx, config.batch_size * 2, False)
    best_score, stale, best_epoch, history = -np.inf, 0, 0, []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train(); loss_rows = []
        training_loader = make_loader(waves, hard_labels, train_targets, valid, train_idx, config.batch_size, True, SEED + fold * 1000 + epoch)
        for wave, label, target, valid_batch, _ in training_loader:
            optimizer.zero_grad(set_to_none=True)
            logit, estimate = model(wave.to(device, non_blocking=True), detach_auxiliary)
            bce = nn.functional.binary_cross_entropy_with_logits(logit, label.to(device, non_blocking=True))
            if uses_auxiliary:
                mask = (label.to(device) == 1) & valid_batch.to(device)
                auxiliary = nn.functional.smooth_l1_loss(estimate[mask], target.to(device)[mask]) if bool(mask.any()) else logit.sum() * 0.
            else:
                auxiliary = logit.sum() * 0.
            loss = bce + config.auxiliary_weight * auxiliary
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"{condition}/fold={fold}/epoch={epoch} 出现非有限损失")
            loss.backward()
            if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters() if parameter.grad is not None):
                raise FloatingPointError(f"{condition}/fold={fold}/epoch={epoch} 出现非有限梯度")
            optimizer.step(); loss_rows.append((float(loss.detach().cpu()), float(bce.detach().cpu()), float(auxiliary.detach().cpu())))
        validation = predict(model, validation_loader, device)
        score = float(roc_auc_score(validation["label"], validation["logit"]))
        history.append({"epoch": epoch, "loss": float(np.mean([x[0] for x in loss_rows])), "bce": float(np.mean([x[1] for x in loss_rows])),
                        "auxiliary": float(np.mean([x[2] for x in loss_rows])), "validation_auroc": score})
        progress(epoch, score)
        if score > best_score:
            best_score, stale, best_epoch = score, 0, epoch
            torch.save({"state_dict": model.state_dict(), "condition": condition, "fold": fold, "epoch": epoch,
                        "validation_auroc": score, "target_median": median, "target_scale": scale}, checkpoint)
        else: stale += 1
        if stale >= config.patience: break
    payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["state_dict"])
    deployment = WaveformDeploymentModel(config).to(device).eval()
    deployment.waveform.load_state_dict(model.waveform.state_dict()); deployment.classifier.load_state_dict(model.classifier.state_dict())
    with torch.no_grad():
        probe = torch.as_tensor(np.asarray(waves[test_idx[:4]]), dtype=torch.float32, device=device)
        trained_logit, _ = model.eval()(probe)
        if not torch.allclose(trained_logit, deployment(probe), atol=1e-6, rtol=0.): raise RuntimeError("移除 OMP 头后分类 logit 改变")
    torch.save({"state_dict": deployment.state_dict(), "input": "waveform [batch,5761]", "removed_module": "auxiliary"}, directory / "deployment_waveform_only.pt")
    validation = predict(model, validation_loader, device); temperature = fit_temperature(validation["logit"], validation["label"])
    threshold = threshold_for(validation["label"].astype(int), sigmoid(validation["logit"] / temperature))
    test = predict(model, test_loader, device); probability = sigmoid(test["logit"] / temperature)
    result = classification_metrics(test["label"].astype(int), probability, threshold)
    result.update({"condition": condition, "test_fold": fold, "validation_fold": (fold + 1) % 5,
                   "best_epoch": best_epoch, "best_validation_auroc": best_score, "temperature": temperature,
                   "threshold": threshold, "parameters": int(sum(p.numel() for p in model.parameters())),
                   "training_seconds": time.perf_counter() - started})
    test_raw = labels[test["index"]]; test_pred_raw = np.clip(test["auxiliary"], -5., 5.) * scale + median
    baseline = np.median(labels[train_idx[valid[train_idx]]], axis=0)
    auxiliary_rows = pd.DataFrame({"condition": condition, "test_fold": fold, "target": TARGET_NAMES,
                                   "mae": np.mean(np.abs(test_pred_raw - test_raw), axis=0),
                                   "train_median_baseline_mae": np.mean(np.abs(test_raw - baseline), axis=0)})
    test_frame = manifest.iloc[test["index"]].reset_index(drop=True)
    prediction_frame = test_frame[["sample_id", "ori_file_num", "test_fold", "target_snr_db", "noise_type"]].copy()
    prediction_frame.insert(0, "sample_index", test["index"]); prediction_frame["label"] = test["label"].astype(int)
    prediction_frame["logit"] = test["logit"]; prediction_frame["p_click"] = probability
    prediction_frame["predicted_label"] = (probability >= threshold).astype(int); prediction_frame["condition"] = condition
    for dimension, name in enumerate(TARGET_NAMES): prediction_frame[f"aux_{name}"] = test_pred_raw[:, dimension]
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig"); auxiliary_rows.to_csv(auxiliary_path, index=False, encoding="utf-8-sig")
    dump_json(complete, result)
    return result, prediction_frame, auxiliary_rows


def train(config: Config) -> None:
    manifest, waves = load_data(config); labels, _, _, valid = validate_omp_archive(config, manifest)
    jobs = [(condition, fold) for condition in CONDITIONS for fold in TEST_FOLDS]
    metric_rows, prediction_rows, auxiliary_rows = [], [], []
    bar = tqdm(jobs, total=len(jobs), desc="总体进度：v7 训练", unit="模型折")
    for condition, fold in bar:
        def progress(epoch: int, score: float) -> None:
            print(f"[总体进度 {len(metric_rows)}/{len(jobs)} | {condition} | test_fold={fold} | epoch={epoch}/{config.max_epochs}] validation_AUROC={score:.6f}", flush=True)
        bar.set_postfix(condition=condition, test_fold=fold)
        metric, prediction, auxiliary = train_one(config, condition, fold, manifest, waves, labels, valid, progress)
        metric_rows.append(metric); prediction_rows.append(prediction); auxiliary_rows.append(auxiliary)
    paths = output_paths(config)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_rows, ignore_index=True).to_csv(paths["predictions"] / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(auxiliary_rows, ignore_index=True).to_csv(paths["metrics"] / "auxiliary_mae.csv", index=False, encoding="utf-8-sig")


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = [f"{float(row[column]):.{digits}f}" if isinstance(row[column], (float, np.floating)) else str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report(config: Config) -> Path:
    paths = output_paths(config); metric_path, auxiliary_path = paths["metrics"] / "fold_metrics.csv", paths["metrics"] / "auxiliary_mae.csv"
    if not metric_path.exists() or not auxiliary_path.exists(): raise FileNotFoundError("请先完成全部十个模型折训练")
    metrics, auxiliary = pd.read_csv(metric_path), pd.read_csv(auxiliary_path)
    if len(metrics) != 10 or set(metrics.condition) != set(CONDITIONS) or set(metrics.test_fold) != set(TEST_FOLDS):
        raise RuntimeError("报告要求完整的五条件×两测试折结果")
    pivot = metrics.pivot(index="test_fold", columns="condition", values="auroc")
    criterion = all(pivot.loc[fold, "C3_full_proxy"] > pivot.loc[fold, opponent]
                    for fold in TEST_FOLDS for opponent in ("C0_waveform_control", "C2_proxy_no_aux", "C4_shuffled_proxy"))
    differences = pd.DataFrame([{"test_fold": fold, "C3_minus_C0_auroc": pivot.loc[fold, "C3_full_proxy"] - pivot.loc[fold, "C0_waveform_control"],
                                 "C3_minus_C2_auroc": pivot.loc[fold, "C3_full_proxy"] - pivot.loc[fold, "C2_proxy_no_aux"],
                                 "C3_minus_C4_auroc": pivot.loc[fold, "C3_full_proxy"] - pivot.loc[fold, "C4_shuffled_proxy"]} for fold in TEST_FOLDS])
    report_path = paths["root"] / "实验记录.md"
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## 实际报告结论\n\n")
        handle.write("结论：达到目标，可考虑扩展五折。\n" if criterion else "结论：未达到双 fold AUROC 判定，不扩展五折。\n")
        handle.write("\n### 完整测试折分类指标\n\n" + markdown_table(metrics[["condition", "test_fold", "auroc", "auprc", "balanced_accuracy", "recall", "specificity", "f1", "brier"]]) + "\n")
        handle.write("\n### C3 AUROC 差值\n\n" + markdown_table(differences) + "\n")
        handle.write("\n### 六维 OMP 任务质量\n\n" + markdown_table(auxiliary) + "\n")
    differences.to_csv(paths["metrics"] / "condition_differences.csv", index=False, encoding="utf-8-sig")
    return report_path


def export_c3(config: Config) -> Path:
    destination = output_paths(config)["deployment"]
    for fold in TEST_FOLDS:
        source = model_dir(config, "C3_full_proxy", fold) / "deployment_waveform_only.pt"
        if not source.exists(): raise FileNotFoundError(f"请先完成 C3 fold {fold} 训练")
        (destination / f"C3_waveform_encoder_detector_fold{fold}.pt").write_bytes(source.read_bytes())
    return destination


def verify(config: Config) -> dict[str, Any]:
    manifest, waves = load_data(config); source = positive_source_contract(manifest)
    sample_rows = manifest.loc[manifest.label.eq(1)].iloc[:3]
    for _, row in sample_rows.iterrows():
        selected, coefficients = omp_atoms(read_wave(Path(row.wav_path)), read_wave(template_path(row.sample_id)))
        physical_label(selected, coefficients)
    device = device_for(config.device); batch = torch.as_tensor(np.asarray(waves[:4]), dtype=torch.float32, device=device)
    target = torch.zeros((4, AUX_DIM), dtype=torch.float32, device=device); hard = torch.as_tensor(manifest.label.iloc[:4].to_numpy(np.float32), device=device)
    models = {}; initial_states = {}
    for condition in CONDITIONS:
        seed_all(SEED); model = OMPAuxClassifier(config).to(device).train(); initial_states[condition] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        logit, auxiliary = model(batch, condition == "C1_aux_only")
        loss = nn.functional.binary_cross_entropy_with_logits(logit, hard)
        if condition in ("C1_aux_only", "C3_full_proxy", "C4_shuffled_proxy"): loss = loss + config.auxiliary_weight * nn.functional.smooth_l1_loss(auxiliary, target)
        loss.backward()
        models[condition] = {"parameters": sum(p.numel() for p in model.parameters()), "finite": bool(torch.isfinite(loss)),
                             "auxiliary_grad": bool(model.auxiliary[1].weight.grad is not None and torch.isfinite(model.auxiliary[1].weight.grad).all()),
                             "encoder_grad": float(model.waveform.patch_embedding.weight.grad.abs().sum().cpu())}
    same_init = all(all(torch.equal(initial_states["C0_waveform_control"][key], value) for key, value in initial_states[condition].items()) for condition in CONDITIONS[1:])
    detector_states = {}
    for condition in ("C0_waveform_control", "C1_aux_only", "C2_proxy_no_aux"):
        seed_all(SEED); model = OMPAuxClassifier(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizer.zero_grad(set_to_none=True); logit, estimate = model(batch, condition == "C1_aux_only")
        loss = nn.functional.binary_cross_entropy_with_logits(logit, hard)
        if condition == "C1_aux_only": loss = loss + config.auxiliary_weight * nn.functional.smooth_l1_loss(estimate, target)
        loss.backward(); optimizer.step()
        detector_states[condition] = {name: value.detach().clone() for name, value in model.state_dict().items()
                                      if name.startswith("waveform.") or name.startswith("classifier.")}
    control_state = detector_states["C0_waveform_control"]
    detector_updates_match = all(all(torch.equal(control_state[name], value) for name, value in detector_states[condition].items())
                                 for condition in ("C1_aux_only", "C2_proxy_no_aux"))
    seed_all(SEED); c1_aux = OMPAuxClassifier(config).to(device); _, estimate = c1_aux(batch, True)
    nn.functional.smooth_l1_loss(estimate, target).backward()
    c1_aux_only_isolated = all(parameter.grad is None for parameter in c1_aux.waveform.parameters()) and c1_aux.auxiliary[1].weight.grad is not None
    seed_all(SEED); c3_aux = OMPAuxClassifier(config).to(device); _, estimate = c3_aux(batch, False)
    nn.functional.smooth_l1_loss(estimate, target).backward()
    c3_aux_gradient = float(c3_aux.waveform.patch_embedding.weight.grad.abs().sum().cpu())
    seed_all(SEED); full = OMPAuxClassifier(config).to(device).eval(); deploy = WaveformDeploymentModel(config).to(device).eval()
    deploy.waveform.load_state_dict(full.waveform.state_dict()); deploy.classifier.load_state_dict(full.classifier.state_dict())
    with torch.no_grad():
        before, _ = full(batch); auxiliary_before = full.auxiliary[-1].weight.detach().clone(); full.auxiliary[-1].weight.add_(1.); after, _ = full(batch)
        labels_changed_logits = bool(torch.allclose(before, after, atol=1e-6, rtol=0.)); full.auxiliary[-1].weight.copy_(auxiliary_before)
        deployment_match = bool(torch.allclose(before, deploy(batch), atol=1e-6, rtol=0.))
    train_idx, valid_idx, test_idx = split(manifest, 0)
    fake = np.arange(N_SAMPLES * AUX_DIM, dtype=np.float32).reshape(N_SAMPLES, AUX_DIM)
    shuffled = shuffled_train_targets(fake, manifest.label.to_numpy(int), train_idx, 0)
    changed = np.flatnonzero(np.any(shuffled != fake, axis=1))
    c4_only_train_positive = bool(np.all(np.isin(changed, train_idx)) and np.all(manifest.label.iloc[changed].to_numpy(int) == 1) and np.array_equal(shuffled[valid_idx], fake[valid_idx]) and np.array_equal(shuffled[test_idx], fake[test_idx]))
    result = {"data": source, "small_omp_samples": len(sample_rows), "same_initialization": same_init,
              "same_parameter_count": len({item["parameters"] for item in models.values()}) == 1,
              "finite_networks": all(item["finite"] for item in models.values()), "c1_auxiliary_gradient": models["C1_aux_only"]["auxiliary_grad"],
              "c0_c1_c2_detector_updates_match": detector_updates_match, "c1_auxiliary_isolated": c1_aux_only_isolated,
              "c3_encoder_auxiliary_gradient": math.isfinite(c3_aux_gradient) and c3_aux_gradient > 0., "c4_training_positive_only": c4_only_train_positive,
              "auxiliary_parameter_does_not_change_logit": labels_changed_logits, "deployment_matches": deployment_match, "networks": models}
    result["passed"] = bool(result["same_initialization"] and result["same_parameter_count"] and result["finite_networks"] and result["c1_auxiliary_gradient"] and result["c0_c1_c2_detector_updates_match"] and result["c1_auxiliary_isolated"] and result["c3_encoder_auxiliary_gradient"] and result["c4_training_positive_only"] and result["auxiliary_parameter_does_not_change_logit"] and result["deployment_matches"])
    dump_json(output_paths(config)["tests"] / "verification.json", result)
    if not result["passed"]: raise RuntimeError(f"v7 验证失败：{result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "extract-omp", "train", "report", "export", "all"))
    parser.add_argument("--device", default="auto"); parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def run_step(config: Config, name: str, function: Any) -> Any:
    started = datetime.now()
    try:
        result = function(); append_record(config, name, started, datetime.now(), "完成", "命令实际完成。"); return result
    except Exception as error:
        append_record(config, name, started, datetime.now(), "失败", str(error).replace("|", "/")); raise


def main() -> None:
    arguments = parse_args(); config = Config(output_dir=arguments.output_dir or OUT, device=arguments.device); output_paths(config)
    dump_json(config.output_dir / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir), "seed": SEED,
              "test_folds": TEST_FOLDS, "validation_folds": (1, 3), "classification_input": "v4 preprocessed_waveforms only",
              "forbidden_input": "BCSC cache and structure_maps", "primary_endpoint": "full test fold AUROC"})
    steps = {"verify": lambda: verify(config), "extract-omp": lambda: extract_omp(config), "train": lambda: train(config),
             "report": lambda: report(config), "export": lambda: export_c3(config)}
    selected = ("verify", "extract-omp", "train", "report", "export") if arguments.command == "all" else (arguments.command,)
    for name in selected:
        print(f"[总体进度] 开始 {name}", flush=True); run_step(config, name, steps[name])


if __name__ == "__main__": main()
