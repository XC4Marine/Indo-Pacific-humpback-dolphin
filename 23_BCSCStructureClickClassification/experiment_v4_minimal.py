#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v4：BCSC可靠性门控最小消融实验。

仅复用实验21的波形缓存和实验23-v2的完整BCSC结构缓存，不会提取BCSC。
命令：verify、screen、train-core、report、all。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Windows工作站上的中文图表字体；不存在时保留Matplotlib默认回退。
CN_FONT: Any | None = None
if Path(r"C:\Windows\Fonts\simhei.ttf").exists():
    from matplotlib import font_manager
    CN_FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    plt.rcParams["font.family"] = CN_FONT.get_name()
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE = ROOT / "21_PhysicsInformedClickClassification" / "outputs_snr_m15_p15_full_5seeds"
V2_CACHE = ROOT / "23_BCSCStructureClickClassification" / "v2_low_snr_gated" / "cache"
OUT = Path(os.environ.get("BCSC_V4_OUTPUT", str(ROOT / "23_BCSCStructureClickClassification" / "v4_minimal_reliability_ablation")))
WAVE_SAMPLES, PATH_BINS, SEED = 5761, 1133, int(os.environ.get("BCSC_V4_SEED", "20260826"))
CONDITIONS = ("C0_waveform", "C1_full_zero", "C2_v2_scalar", "C3_reliability", "C4_local_gate", "C5_structure_dropout")
CORE_CONDITIONS = ("C0_waveform", "C1_full_zero", "C5_structure_dropout")
SCREEN_FOLDS = (0, 2)
LABELS = {
    "C0_waveform": "C0 Waveform",
    "C1_full_zero": "C1 Full Zero",
    "C2_v2_scalar": "C2 v2 Scalar",
    "C3_reliability": "C3 +Reliability",
    "C4_local_gate": "C4 +Local Gate",
    "C5_structure_dropout": "C5 +Structure Dropout",
}
STEP_PAIRS = (("reliability", "C2_v2_scalar", "C3_reliability"),
              ("local_gate", "C3_reliability", "C4_local_gate"),
              ("structure_dropout", "C4_local_gate", "C5_structure_dropout"))
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SOURCE / "dataset_manifest.csv"
    waveform_path: Path = SOURCE / "cache" / "preprocessed_waveforms.npy"
    structure_path: Path = V2_CACHE / "structure_maps.npy"
    structure_done_path: Path = V2_CACHE / "structure_completed.npy"
    structure_status_path: Path = V2_CACHE / "structure_status.csv"
    batch_size: int = 64
    max_epochs: int = 12
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    structure_dropout: float = 0.20
    bootstrap_runs: int = 2000
    device: str = "auto"


def output_paths(config: Config) -> dict[str, Path]:
    result = {name: config.output_dir / name for name in ("models", "logs", "tests")}
    result["root"] = config.output_dir
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def dump_json(path: Path, value: Any) -> None:
    def json_safe(item: Any) -> Any:
        if isinstance(item, (float, np.floating)):
            return float(item) if math.isfinite(float(item)) else None
        if isinstance(item, (int, np.integer, str, bool)) or item is None:
            return item
        if isinstance(item, dict):
            return {str(key): json_safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(child) for child in item]
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, default=str,
                               allow_nan=False), encoding="utf-8")
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定CUDA设备，但CUDA不可用")
    return device


def load_data(config: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    required = (config.manifest_path, config.waveform_path, config.structure_path,
                config.structure_done_path, config.structure_status_path)
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError("缺少固定实验21数据或v2 BCSC缓存：" + "; ".join(absent))
    manifest = pd.read_csv(config.manifest_path)
    waves = np.load(config.waveform_path, mmap_mode="r")
    structures = np.load(config.structure_path, mmap_mode="r")
    completed = np.load(config.structure_done_path)
    status = pd.read_csv(config.structure_status_path)
    checks = {
        "manifest_rows": len(manifest) == 7000,
        "wave_shape": tuple(waves.shape) == (7000, WAVE_SAMPLES),
        "structure_shape": tuple(structures.shape) == (7000, 3, PATH_BINS),
        "structure_completed": tuple(completed.shape) == (7000,) and bool(completed.all()),
        "class_balance": manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
        "recording_groups": manifest.ori_file_num.nunique() == 34,
        "five_folds": set(manifest.test_fold.unique()) == set(range(5)),
        "no_group_leakage": manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
        "finite_wave": bool(np.isfinite(waves).all()),
        "finite_structure": bool(np.isfinite(structures).all()),
        "status_rows": len(status) == 7000 and status.sample_index.nunique() == 7000,
    }
    if not all(checks.values()):
        raise RuntimeError(f"输入数据合同失败：{checks}")
    return manifest, waves, structures, status


class ClickDataset(Dataset):
    def __init__(self, waves: np.ndarray, structures: np.ndarray, labels: np.ndarray,
                 indices: np.ndarray) -> None:
        self.waves, self.structures, self.labels, self.indices = waves, structures, labels, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32),
                torch.as_tensor(np.asarray(self.structures[index]), dtype=torch.float32),
                torch.tensor(self.labels[index], dtype=torch.float32), index)


def make_loader(waves: np.ndarray, structures: np.ndarray, labels: np.ndarray,
                indices: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(ClickDataset(waves, structures, labels, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


class WaveformTransformer(nn.Module):
    """单层、d_model=64、4 heads、FFN=128 的波形编码器。"""
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.patch = nn.Conv1d(1, 64, kernel_size=32, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(wave.unsqueeze(1)).transpose(1, 2)
        positions = torch.arange(tokens.shape[1], device=wave.device, dtype=tokens.dtype).unsqueeze(-1)
        dims = torch.arange(64, device=wave.device, dtype=tokens.dtype).unsqueeze(0)
        scale = torch.pow(torch.tensor(10000., device=wave.device, dtype=tokens.dtype),
                          2 * torch.floor(dims / 2) / 64)
        angle = positions / scale
        positional = torch.where((dims.long() % 2) == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        return self.encoder(torch.cat([self.cls.expand(wave.shape[0], -1, -1),
                                       tokens + positional], dim=1))[:, 0]


class PathCNN(nn.Module):
    """实验23原有三层Path-CNN。"""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3), nn.GroupNorm(4, 16), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.project = nn.Linear(128, 32)

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        value = self.net(structure)
        return self.project(torch.cat([value.mean(dim=-1), value.max(dim=-1).values], dim=1))


class FusionModel(nn.Module):
    """C0--C5。C1和C5同构；C1在模型入口逐样本置零结构输入。"""
    def __init__(self, condition: str, dropout: float, structure_dropout: float) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition, self.structure_dropout = condition, structure_dropout
        self.waveform = WaveformTransformer(dropout)
        self.structure = None if condition == "C0_waveform" else PathCNN()
        self.global_gate = None if condition == "C0_waveform" else nn.Sequential(
            nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
        # C1必须与C5完全同构；区别仅是其结构输入在入口处置零。
        self.local_conv = None if condition not in ("C1_full_zero", "C4_local_gate", "C5_structure_dropout") else nn.Conv1d(3, 1, 1)
        self.local_context = None if self.local_conv is None else nn.Linear(64, 1)
        self.head = nn.Sequential(nn.Linear(64 if condition == "C0_waveform" else 96, 64),
                                  nn.GELU(), nn.Linear(64, 1))
        self.last_gates: dict[str, torch.Tensor] = {}

    def _structure_input(self, structure: torch.Tensor, wave_features: torch.Tensor) -> torch.Tensor:
        raw = torch.zeros_like(structure) if self.condition == "C1_full_zero" else structure
        p, uncertainty = raw[:, 0:1], raw[:, 2:3]
        r = torch.clamp(p * (1. - uncertainty), 0., 1.)
        a = torch.ones_like(r)
        if self.local_conv is not None:
            a = torch.sigmoid(self.local_conv(raw) + self.local_context(wave_features).unsqueeze(-1))
        g = r * a
        value = raw if self.condition == "C2_v2_scalar" else raw * g
        if self.condition in ("C1_full_zero", "C5_structure_dropout") and self.training:
            # 同一时延的P/A/U共同随机失活，缩放保持训练期望幅度。
            keep = (torch.rand_like(g) >= self.structure_dropout).to(value.dtype)
            value = value * keep / (1. - self.structure_dropout)
        self.last_gates = {"r": r, "a": a, "g": g}
        return value

    def forward(self, wave: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        wave_features = self.waveform(wave)
        if self.condition == "C0_waveform":
            self.last_gates = {}
            return self.head(wave_features).squeeze(1)
        path_features = self.structure(self._structure_input(structure, wave_features))
        global_gate = torch.sigmoid(self.global_gate(torch.cat([wave_features, path_features], dim=1)))
        self.last_gates["global"] = global_gate
        return self.head(torch.cat([wave_features, global_gate * path_features], dim=1)).squeeze(1)


def split(manifest: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = manifest.test_fold.to_numpy(dtype=int)
    validation_fold = (fold + 1) % 5
    return (np.flatnonzero((values != fold) & (values != validation_fold)),
            np.flatnonzero(values == validation_fold), np.flatnonzero(values == fold))


def low_snr_mask(frame: pd.DataFrame) -> np.ndarray:
    labels = frame.label.to_numpy(dtype=int)
    snr = frame.target_snr_db.to_numpy(dtype=float)
    return (labels == 0) | ((labels == 1) & (snr >= -15.) & (snr < 0.))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-np.clip(value, -80., 80.)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., logits / temperature) - labels * logits / temperature))
    result = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0., .5, 1.], probabilities)))
    values = [balanced_accuracy_score(labels, probabilities >= candidate) for candidate in candidates]
    return float(candidates[int(np.argmax(values))])


def score_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    result = {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= threshold)),
    }
    return result


class RunningStats:
    def __init__(self) -> None:
        self.count, self.sum, self.sum_sq, self.zeros, self.finite = 0, 0., 0., 0, True

    def add(self, value: torch.Tensor) -> None:
        flat = value.detach().float().reshape(-1)
        self.count += int(flat.numel())
        self.finite = self.finite and bool(torch.isfinite(flat).all())
        if flat.numel():
            self.sum += float(flat.sum().cpu())
            self.sum_sq += float((flat * flat).sum().cpu())
            self.zeros += int((flat == 0).sum().cpu())

    def result(self, prefix: str) -> dict[str, float | bool]:
        if not self.count:
            return {f"{prefix}_mean": math.nan, f"{prefix}_std": math.nan,
                    f"{prefix}_zero_ratio": math.nan, f"{prefix}_finite": True}
        mean = self.sum / self.count
        variance = max(self.sum_sq / self.count - mean * mean, 0.)
        return {f"{prefix}_mean": mean, f"{prefix}_std": math.sqrt(variance),
                f"{prefix}_zero_ratio": self.zeros / self.count, f"{prefix}_finite": self.finite}


def collect_gates(model: FusionModel, targets: dict[str, RunningStats]) -> None:
    for name, value in model.last_gates.items():
        targets[name].add(value)


def gradient_rms(model: FusionModel) -> dict[str, float | bool]:
    buckets: dict[str, list[torch.Tensor]] = {name: [] for name in
                                               ("waveform", "path_cnn", "local_gate", "global_gate", "head")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("waveform."):
            bucket = "waveform"
        elif name.startswith("structure."):
            bucket = "path_cnn"
        elif name.startswith("local_"):
            bucket = "local_gate"
        elif name.startswith("global_gate."):
            bucket = "global_gate"
        elif name.startswith("head."):
            bucket = "head"
        else:
            raise RuntimeError(f"未分类参数：{name}")
        buckets[bucket].append(parameter.grad.detach().reshape(-1))
    result: dict[str, float | bool] = {}
    for name, values in buckets.items():
        if not values:
            result[f"grad_{name}_rms"] = math.nan
            result[f"grad_{name}_finite"] = True
            continue
        joined = torch.cat(values)
        result[f"grad_{name}_rms"] = float(torch.sqrt(torch.mean(joined.float() ** 2)).cpu())
        result[f"grad_{name}_finite"] = bool(torch.isfinite(joined).all())
    return result


@torch.no_grad()
def predict(model: FusionModel, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float | bool]], float]:
    model.eval()
    identifiers, labels, logits = [], [], []
    gates = {name: RunningStats() for name in ("r", "a", "g", "global")}
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for wave, structure, label, identifier in data:
        output = model(wave.to(device, non_blocking=True), structure.to(device, non_blocking=True))
        collect_gates(model, gates)
        identifiers.append(np.asarray(identifier)); labels.append(label.numpy()); logits.append(output.cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize()
    return (np.concatenate(identifiers), np.concatenate(labels), np.concatenate(logits),
            {name: stats.result(name) for name, stats in gates.items()}, time.perf_counter() - started)


def run_directory(config: Config, condition: str, fold: int) -> Path:
    return output_paths(config)["models"] / condition / f"seed_{SEED}" / f"fold_{fold}"


def train_one(condition: str, fold: int, waves: np.ndarray, structures: np.ndarray,
              manifest: pd.DataFrame, config: Config, resume: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    directory = run_directory(config, condition, fold)
    directory.mkdir(parents=True, exist_ok=True)
    complete, checkpoint, prediction_path, diagnostic_path = (directory / "COMPLETE.json",
        directory / "best_checkpoint.pt", directory / "test_predictions.csv", directory / "epoch_diagnostics.csv")
    if resume and all(path.exists() for path in (complete, checkpoint, prediction_path, diagnostic_path)):
        return json.loads(complete.read_text(encoding="utf-8")), pd.read_csv(prediction_path), pd.read_csv(diagnostic_path)

    seed_all(SEED + fold * 100)
    device = device_for(config.device)
    labels = manifest.label.to_numpy(dtype=np.float32)
    train_ids, validation_ids, test_ids = split(manifest, fold)
    loaders = {
        "train": make_loader(waves, structures, labels, train_ids, config.batch_size, shuffle=True),
        "validation": make_loader(waves, structures, labels, validation_ids, config.batch_size * 2),
        "test": make_loader(waves, structures, labels, test_ids, config.batch_size * 2),
    }
    model = FusionModel(condition, config.dropout, config.structure_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    # 原始结构图在CUDA float16缩放下曾产生非有限梯度；为保持所有条件的
    # 数值协议一致，本实验固定采用float32训练，不使用混合精度或GradScaler。
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best_score, stale, history, started = -np.inf, 0, [], time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses: list[float] = []
        gate_stats = {name: RunningStats() for name in ("r", "a", "g", "global")}
        gradient_values: list[dict[str, float | bool]] = []
        for wave, structure, label, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                output = model(wave.to(device, non_blocking=True), structure.to(device, non_blocking=True))
                loss = criterion(output, label.to(device, non_blocking=True))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{condition} fold {fold} epoch {epoch}: 非有限loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients = gradient_rms(model)
            if not all(bool(value) for name, value in gradients.items() if name.endswith("_finite")):
                raise FloatingPointError(f"{condition} fold {fold} epoch {epoch}: 非有限梯度")
            gradient_values.append(gradients)
            collect_gates(model, gate_stats)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_ids, val_labels, val_logits, _, _ = predict(model, loaders["validation"], device)
        val_frame = manifest.iloc[val_ids].reset_index(drop=True).assign(label=val_labels.astype(int))
        mask = low_snr_mask(val_frame)
        value = float(roc_auc_score(val_labels[mask], val_logits[mask]))
        row: dict[str, Any] = {"condition": condition, "test_fold": fold, "epoch": epoch,
                               "training_loss": float(np.mean(losses)), "validation_low_snr_auroc": value,
                               "nan_or_inf": False}
        for name, stats in gate_stats.items(): row.update(stats.result(name))
        for key in gradient_values[0]:
            vals = [item[key] for item in gradient_values]
            row[key] = bool(all(vals)) if key.endswith("_finite") else (
                math.nan if all(not math.isfinite(float(value)) for value in vals) else float(np.nanmean(vals)))
        row["nan_or_inf"] = not all(bool(row[f"{name}_finite"]) for name in ("r", "a", "g", "global")) or not all(
            bool(value) for name, value in row.items() if name.endswith("_finite"))
        saturation = ((row.get("a_mean", .5) <= .02 or row.get("a_mean", .5) >= .98) or
                      (row.get("global_mean", .5) <= .02 or row.get("global_mean", .5) >= .98))
        row["gate_saturated"] = bool(saturation)
        history.append(row)
        if value > best_score + 1e-7:
            best_score, stale = value, 0
            torch.save({"state_dict": model.state_dict(), "condition": condition, "fold": fold,
                        "seed": SEED, "best_validation_low_snr_auroc": value}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    val_ids, val_labels, val_logits, _, _ = predict(model, loaders["validation"], device)
    temperature = fit_temperature(val_logits, val_labels)
    threshold = select_threshold(val_labels.astype(int), sigmoid(val_logits / temperature))
    test_ids, test_labels, test_logits, test_gates, inference_seconds = predict(model, loaders["test"], device)
    probabilities = sigmoid(test_logits / temperature)
    test_frame = manifest.iloc[test_ids].reset_index(drop=True).assign(label=test_labels.astype(int))
    low_mask = low_snr_mask(test_frame)
    details: dict[str, Any] = score_metrics(test_labels.astype(int), probabilities, threshold)
    details.update({f"low_snr_{key}": value for key, value in score_metrics(test_labels[low_mask].astype(int), probabilities[low_mask], threshold).items()})
    details.update({"condition": condition, "seed": SEED, "test_fold": fold,
                    "validation_fold": (fold + 1) % 5, "threshold": threshold,
                    "temperature": temperature, "best_validation_low_snr_auroc": best_score,
                    "best_epoch": int(max(item["epoch"] for item in history if item["validation_low_snr_auroc"] == best_score)),
                    "parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
                    "inference_ms_per_sample": inference_seconds / len(test_ids) * 1000.,
                    "training_seconds": time.perf_counter() - started})
    prediction = pd.DataFrame({"sample_index": test_ids, "label": test_labels.astype(int),
                               "p_click": probabilities, "logit": test_logits,
                               "predicted_label": (probabilities >= threshold).astype(int),
                               "threshold": threshold, "temperature": temperature,
                               "condition": condition, "seed": SEED, "test_fold": fold})
    diagnostic = pd.DataFrame(history)
    for name, values in test_gates.items():
        for key, value in values.items(): details[f"test_{key}"] = value
    diagnostic.to_csv(diagnostic_path, index=False, encoding="utf-8-sig")
    prediction.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    dump_json(complete, details)
    return details, prediction, diagnostic


def merge_metadata(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    meta = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    return predictions.merge(meta, on=["sample_index", "test_fold"], how="left", validate="many_to_one")


def train_jobs(jobs: list[tuple[str, int]], waves: np.ndarray, structures: np.ndarray,
               manifest: pd.DataFrame, config: Config, resume: bool, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, predictions, diagnostics = [], [], []
    bar = tqdm(total=len(jobs), desc=f"总体进度：{label}", unit="模型折")
    for position, (condition, fold) in enumerate(jobs, 1):
        row, prediction, diagnostic = train_one(condition, fold, waves, structures, manifest, config, resume)
        rows.append(row); predictions.append(prediction); diagnostics.append(diagnostic)
        bar.update(1); bar.set_postfix_str(f"{position}/{len(jobs)} {condition} fold={fold} lowAUROC={row['low_snr_auroc']:.3f}")
    bar.close()
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True), pd.concat(diagnostics, ignore_index=True)


def screen(config: Config, resume: bool = True) -> pd.DataFrame:
    manifest, waves, structures, _ = load_data(config)
    jobs = [(condition, fold) for fold in SCREEN_FOLDS for condition in CONDITIONS]
    metrics, predictions, diagnostics = train_jobs(jobs, waves, structures, manifest, config, resume, "阶段A筛选：12/21")
    del predictions
    metric_path = config.output_dir / "screen_fold_metrics.csv"
    metrics.to_csv(metric_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(config.output_dir / "screen_epoch_diagnostics.csv", index=False, encoding="utf-8-sig")
    records: list[dict[str, Any]] = []
    for step, left, right in STEP_PAIRS:
        for fold in SCREEN_FOLDS:
            before = metrics.query("condition == @left and test_fold == @fold").iloc[0]
            after = metrics.query("condition == @right and test_fold == @fold").iloc[0]
            records.append({"step": step, "fold": fold, "from_condition": left, "to_condition": right,
                            "from_low_snr_auroc": before.low_snr_auroc, "to_low_snr_auroc": after.low_snr_auroc,
                            "delta_low_snr_auroc": after.low_snr_auroc - before.low_snr_auroc,
                            "from_auroc": before.auroc, "to_auroc": after.auroc,
                            "delta_auroc": after.auroc - before.auroc})
    result = pd.DataFrame(records)
    directions = []
    for step, item in result.groupby("step", sort=False):
        deltas = item.delta_low_snr_auroc.to_numpy()
        decision = "一致的改善方向" if np.all(deltas > 0) else "效果不稳定" if np.any(deltas > 0) else "当前配置下可能无效"
        directions.append({"step": step, "fold": "summary", "direction": decision,
                           "delta_low_snr_auroc": float(np.mean(deltas))})
    result = pd.concat([result, pd.DataFrame(directions)], ignore_index=True, sort=False)
    result.to_csv(config.output_dir / "screening_ablation.csv", index=False, encoding="utf-8-sig")
    plot_ablation_steps(result[result.fold != "summary"], config.output_dir / "ablation_steps.png")
    return result


def train_core(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, waves, structures, _ = load_data(config)
    jobs = [(condition, fold) for condition in CORE_CONDITIONS for fold in range(5)]
    metrics, predictions, diagnostics = train_jobs(jobs, waves, structures, manifest, config, resume, "阶段B核心五折：15/21")
    predictions = merge_metadata(predictions, manifest)
    if len(predictions) != 21000 or predictions.duplicated(["condition", "sample_index"]).any():
        raise RuntimeError(f"核心OOF合同失败：{len(predictions)}行")
    metrics.to_csv(config.output_dir / "core_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.output_dir / "core_oof_predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(config.output_dir / "core_epoch_diagnostics.csv", index=False, encoding="utf-8-sig")
    screen_diagnostic_path = config.output_dir / "screen_epoch_diagnostics.csv"
    if screen_diagnostic_path.exists():
        diagnostics = pd.concat([diagnostics, pd.read_csv(screen_diagnostic_path)], ignore_index=True)
        diagnostics = diagnostics.drop_duplicates(["condition", "test_fold", "epoch"], keep="first")
    gate_diagnostics = diagnostics[[column for column in diagnostics.columns if column.startswith(("r_", "a_", "g_", "global_", "condition", "test_fold", "epoch", "gate_", "nan_"))]].copy()
    gate_diagnostics.to_csv(config.output_dir / "gate_diagnostics.csv", index=False, encoding="utf-8-sig")
    return metrics, predictions


def train_full(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一个随机种子下C0--C5全部五折，用于三种子完整确认。"""
    manifest, waves, structures, _ = load_data(config)
    jobs = [(condition, fold) for condition in CONDITIONS for fold in range(5)]
    metrics, predictions, diagnostics = train_jobs(jobs, waves, structures, manifest, config, resume,
                                                    "完整五折：30个模型折")
    predictions = merge_metadata(predictions, manifest)
    if len(metrics) != 30 or len(predictions) != 42_000 or predictions.duplicated(["condition", "sample_index"]).any():
        raise RuntimeError(f"完整五折OOF合同失败：metrics={len(metrics)}, predictions={len(predictions)}")
    metrics.to_csv(config.output_dir / "full_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.output_dir / "full_oof_predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(config.output_dir / "full_epoch_diagnostics.csv", index=False, encoding="utf-8-sig")
    return metrics, predictions


def plot_ablation_steps(frame: pd.DataFrame, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    steps = ["reliability", "local_gate", "structure_dropout"]
    for fold, item in frame.groupby("fold"):
        ordered = item.set_index("step").reindex(steps)
        axis.plot(range(len(steps)), ordered.delta_low_snr_auroc, marker="o", label=f"fold {fold}")
    axis.axhline(0, color="black", lw=.8)
    axis.set(xticks=range(len(steps)), xticklabels=["+Reliability", "+Local Gate", "+Structure Dropout"])
    axis.set_ylabel("低SNR AUROC增量", fontproperties=CN_FONT)
    axis.set_title("阶段A：两个fold的逐步消融方向", fontproperties=CN_FONT)
    axis.legend()
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def paired_bootstrap(predictions: pd.DataFrame, left: str, right: str, endpoint: str,
                     runs: int, seed: int) -> dict[str, Any]:
    first = predictions[predictions.condition == left].sort_values("sample_index").reset_index(drop=True)
    second = predictions[predictions.condition == right].sort_values("sample_index").reset_index(drop=True)
    if not np.array_equal(first.sample_index.to_numpy(), second.sample_index.to_numpy()):
        raise RuntimeError("配对OOF样本不一致")
    if endpoint == "low_snr_auroc":
        mask = low_snr_mask(first)
        first, second = first[mask].copy(), second[mask].copy()
    groups = first.ori_file_num.unique()
    blocks = {group: (first[first.ori_file_num == group], second[second.ori_file_num == group]) for group in groups}
    def calculate(a: pd.DataFrame, b: pd.DataFrame) -> float:
        return float(roc_auc_score(a.label, a.p_click) - roc_auc_score(b.label, b.p_click))
    point = calculate(first, second)
    rng, values = np.random.default_rng(seed), []
    for _ in range(runs):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        a = pd.concat([blocks[group][0] for group in chosen], ignore_index=True)
        b = pd.concat([blocks[group][1] for group in chosen], ignore_index=True)
        if a.label.nunique() == 2:
            values.append(calculate(a, b))
    if len(values) < runs * .95:
        raise RuntimeError("有效Bootstrap重复数不足")
    values_np = np.asarray(values)
    return {"left": left, "right": right, "endpoint": endpoint, "point_difference": point,
            "ci_low": float(np.quantile(values_np, .025)), "ci_high": float(np.quantile(values_np, .975)),
            "bootstrap_runs_requested": runs, "bootstrap_runs_valid": len(values),
            "supports_positive_gain": bool(np.quantile(values_np, .025) > 0)}


def plot_core_roc_pr(predictions: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition in CORE_CONDITIONS:
        item = predictions[predictions.condition == condition]
        fpr, tpr, _ = roc_curve(item.label, item.p_click)
        precision, recall, _ = precision_recall_curve(item.label, item.p_click)
        axes[0].plot(fpr, tpr, label=f"{LABELS[condition]} (AUC={roc_auc_score(item.label, item.p_click):.3f})")
        axes[1].plot(recall, precision, label=f"{LABELS[condition]} (AP={average_precision_score(item.label, item.p_click):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set(xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].set_title("核心OOF ROC", fontproperties=CN_FONT)
    axes[1].set_title("核心OOF PR", fontproperties=CN_FONT)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns = list(frame.columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) and math.isfinite(value) else str(value) for value in row])
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def report(config: Config) -> Path:
    core_path, metrics_path = config.output_dir / "core_oof_predictions.csv", config.output_dir / "core_fold_metrics.csv"
    if not core_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("请先完成train-core")
    predictions, metrics = pd.read_csv(core_path), pd.read_csv(metrics_path)
    if len(predictions) != 21000 or len(metrics) != 15:
        raise RuntimeError("报告拒绝使用不完整核心结果")
    comparisons = []
    for number, right in enumerate(("C0_waveform", "C1_full_zero")):
        for endpoint in ("low_snr_auroc", "auroc"):
            comparisons.append(paired_bootstrap(predictions, "C5_structure_dropout", right, endpoint,
                                                config.bootstrap_runs, SEED + number * 1000 + len(endpoint)))
    paired = pd.DataFrame(comparisons)
    paired.to_csv(config.output_dir / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    plot_core_roc_pr(predictions, config.output_dir / "core_roc_pr.png")
    screening_path = config.output_dir / "screening_ablation.csv"
    screening = pd.read_csv(screening_path) if screening_path.exists() else pd.DataFrame()
    primary = paired[paired.endpoint == "low_snr_auroc"].copy()
    conclusion = bool(primary.supports_positive_gain.all())
    fold_summary = metrics.groupby("condition", as_index=False).agg(
        mean_low_snr_auroc=("low_snr_auroc", "mean"), mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"), mean_parameters=("parameters", "mean"))
    text = ["# 实验23-v4：BCSC可靠性门控最小消融实验报告", "",
            "## 实际运行证据", "", "核心三条件均完成五折，核心OOF共21,000行。主终点是低SNR AUROC；Bootstrap以34个录音组为配对重采样单元。", "",
            markdown_table(fold_summary), "", "## 核心配对Bootstrap", "", markdown_table(paired), ""]
    if conclusion:
        text.extend(["## 核心结论", "", "C5相对C0和C1的低SNR AUROC配对Bootstrap 95%置信区间下限均大于0。因此，在本固定五折与本训练配置下，BCSC结构增益成立。", ""])
    else:
        text.extend(["## 核心结论", "", "至少一个主要比较的95%置信区间下限不大于0。因此只能报告提升趋势，不能声称BCSC结构增益成立。", ""])
    text.extend(["## 阶段A最小消融筛选", "", "中间组件仅在fold 0和fold 2运行，用于方向诊断；不报告统计显著性。"])
    if not screening.empty:
        text.extend(["", markdown_table(screening)])
    text.extend(["", "## 证据等级", "", "C0/C1/C5：完整五折核心证据。C2/C3/C4：两个fold的最小消融证据，仅可表述为可能有效、结果不稳定或当前配置下可能无效。"])
    destination = config.output_dir / "research_report_zh.md"
    destination.write_text("\n".join(text) + "\n", encoding="utf-8")
    return destination


def verify(config: Config) -> dict[str, Any]:
    output_paths(config)
    manifest, waves, structures, status = load_data(config)
    device = device_for(config.device)
    ids = np.arange(4)
    wave = torch.as_tensor(np.asarray(waves[ids]), dtype=torch.float32, device=device)
    structure = torch.as_tensor(np.asarray(structures[ids]), dtype=torch.float32, device=device)
    labels = torch.as_tensor(manifest.label.iloc[ids].to_numpy(copy=True), dtype=torch.float32, device=device)
    networks: dict[str, Any] = {}
    c1_parameters = c5_parameters = None
    for condition in CONDITIONS:
        seed_all(SEED)
        model = FusionModel(condition, config.dropout, config.structure_dropout).to(device)
        model.train()
        logits = model(wave, structure)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        gradients = gradient_rms(model)
        gates = {name: value.detach() for name, value in model.last_gates.items()}
        finite_gradient = all(bool(value) for name, value in gradients.items() if name.endswith("_finite"))
        bounded = all(bool(((value >= 0) & (value <= 1)).all()) for name, value in gates.items() if name in ("r", "a", "g", "global"))
        networks[condition] = {"logit_shape": list(logits.shape), "finite_loss": bool(torch.isfinite(loss)),
                               "finite_gradients": finite_gradient, "gates_bounded": bounded,
                               "parameter_count": int(sum(parameter.numel() for parameter in model.parameters()))}
        if condition == "C1_full_zero": c1_parameters = networks[condition]["parameter_count"]
        if condition == "C5_structure_dropout": c5_parameters = networks[condition]["parameter_count"]
    c1_model = FusionModel("C1_full_zero", config.dropout, config.structure_dropout).to(device).eval()
    captured: dict[str, torch.Tensor] = {}
    def capture_c1_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["value"] = inputs[0].detach().cpu()
    handle = c1_model.structure.net[0].register_forward_pre_hook(capture_c1_input)
    with torch.no_grad(): c1_model(wave, structure)
    handle.remove()
    c1_zero = bool(torch.count_nonzero(captured["value"]) == 0)
    c5 = FusionModel("C5_structure_dropout", config.dropout, config.structure_dropout).to(device)
    # 关闭Transformer内常规dropout，只检验结构随机失活是否严格受train/eval控制。
    captured_dropout: list[torch.Tensor] = []
    dropout_hook = c5.structure.register_forward_pre_hook(
        lambda _module, inputs: captured_dropout.append(inputs[0].detach().cpu()))
    seed_all(SEED + 7); c5.train(); c5.waveform.eval(); c5(wave, structure)
    c5.eval(); c5(wave, structure); c5(wave, structure)
    dropout_hook.remove()
    dropout_only_train = bool(len(captured_dropout) == 3 and not torch.equal(captured_dropout[0], captured_dropout[1])
                             and torch.equal(captured_dropout[1], captured_dropout[2]))
    result = {"data": {"wave_shape": list(waves.shape), "structure_shape": list(structures.shape),
                       "recording_groups": int(manifest.ori_file_num.nunique()), "status_rows": len(status)},
              "networks": networks, "c1_zero_structure": c1_zero,
              "c1_c5_same_parameter_count": c1_parameters == c5_parameters,
              "dropout_only_in_training": dropout_only_train,
              "transformer_layers": len(c5.waveform.encoder.layers)}
    result["passed"] = bool(result["data"]["wave_shape"] == [7000, 5761] and result["data"]["structure_shape"] == [7000, 3, 1133]
                            and result["data"]["recording_groups"] == 34 and c1_zero
                            and result["c1_c5_same_parameter_count"] and dropout_only_train
                            and result["transformer_layers"] == 1
                            and all(item["logit_shape"] == [4] and item["finite_loss"] and item["finite_gradients"] and item["gates_bounded"] for item in networks.values()))
    dump_json(config.output_dir / "tests" / "verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"v4验证失败：{result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "screen", "train-core", "train-full", "report", "all"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(device=args.device, bootstrap_runs=args.bootstrap_runs)
    output_paths(config)
    dump_json(config.output_dir / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir),
              "manifest_path": str(config.manifest_path), "waveform_path": str(config.waveform_path),
              "structure_path": str(config.structure_path), "seed": SEED,
              "unique_model_folds": 21, "screen_model_folds": 12, "core_model_folds": 15,
              "primary_endpoint": "low_snr_auroc"})
    resume = not args.no_resume
    if args.command in ("verify", "all"):
        print("[总体进度 0%] 验证固定缓存、六条件和随机失活边界", flush=True)
        verify(config)
    if args.command in ("screen", "all"):
        print("[总体进度 5%] 阶段A：运行12个筛选模型折", flush=True)
        screen(config, resume)
    if args.command in ("train-core", "all"):
        print("[总体进度 45%] 阶段B：运行15个核心模型折（复用已完成折）", flush=True)
        train_core(config, resume)
    if args.command == "train-full":
        print("[总体进度 0%] 完整五折：运行30个模型折", flush=True)
        train_full(config, resume)
        print("[总体进度 100%] 完整五折训练完成", flush=True)
    if args.command in ("report", "all"):
        print("[总体进度 90%] 生成核心OOF统计、图表与研究报告", flush=True)
        destination = report(config)
        print(f"[总体进度 100%] 完成：{destination}", flush=True)


if __name__ == "__main__":
    main()
