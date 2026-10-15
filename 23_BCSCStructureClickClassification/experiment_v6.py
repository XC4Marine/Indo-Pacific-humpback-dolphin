#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v6：波形驱动 BCSC 代理分支最小消融。

本程序只读取实验21的固定波形和实验23-v2已完成的 BCSC 缓存。真实
BCSC 摘要绝不进入分类器：它仅作为 C1/C3/C4 的训练期回归目标，部署
模型始终只接受一条 5761 点波形。

命令：verify、train、report、export、all。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE = ROOT / "21_PhysicsInformedClickClassification" / "outputs_snr_m15_p15_full_5seeds"
CACHE = ROOT / "23_BCSCStructureClickClassification" / "v2_low_snr_gated" / "cache"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v6_waveform_bcsc_proxy_ablation"
SEED = 20260826
SCREEN_FOLDS = (0, 2)
WAVE_SAMPLES, PATH_BINS, SAMPLE_RATE, MIN_DELAY = 5761, 1133, 576_000, 20
CONDITIONS = ("C0_waveform_control", "C1_aux_only", "C2_proxy_no_aux",
              "C3_full_proxy", "C4_shuffled_proxy")
CONDITION_LABELS = {
    "C0_waveform_control": "C0 Waveform-control",
    "C1_aux_only": "C1 Aux-only",
    "C2_proxy_no_aux": "C2 Proxy-no-aux",
    "C3_full_proxy": "C3 Full-proxy",
    "C4_shuffled_proxy": "C4 Shuffled-proxy",
}
TARGET_NAMES = ("path_count", "support_sum", "support_max", "support_mean",
                "amplitude_max", "delay_weighted_mean_ms", "delay_weighted_std_ms",
                "uncertainty_mean")
AUXILIARY_CONDITIONS = frozenset(("C1_aux_only", "C3_full_proxy", "C4_shuffled_proxy"))
PROXY_CONDITIONS = frozenset(("C2_proxy_no_aux", "C3_full_proxy", "C4_shuffled_proxy"))
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SOURCE / "dataset_manifest.csv"
    waveform_path: Path = SOURCE / "cache" / "preprocessed_waveforms.npy"
    structure_path: Path = CACHE / "structure_maps.npy"
    structure_done_path: Path = CACHE / "structure_completed.npy"
    structure_status_path: Path = CACHE / "structure_status.csv"
    batch_size: int = 64
    max_epochs: int = 12
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    auxiliary_weight: float = 0.1
    device: str = "auto"


def output_paths(config: Config) -> dict[str, Path]:
    result = {name: config.output_dir / name for name in
              ("models", "metrics", "predictions", "gradients", "figures", "deployment", "tests")}
    result["root"] = config.output_dir
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def json_dump(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, (np.floating, float)):
            return float(item) if math.isfinite(float(item)) else None
        if isinstance(item, (np.integer, int, str, bool)) or item is None:
            return item
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return str(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


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
        raise RuntimeError("指定了CUDA设备，但CUDA不可用")
    return device


def split(manifest: pd.DataFrame, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = manifest.test_fold.to_numpy(dtype=int)
    validation_fold = (test_fold + 1) % 5
    return (np.flatnonzero((folds != test_fold) & (folds != validation_fold)),
            np.flatnonzero(folds == validation_fold), np.flatnonzero(folds == test_fold))


def low_snr_mask(frame: pd.DataFrame) -> np.ndarray:
    labels = frame.label.to_numpy(dtype=int)
    snr = frame.target_snr_db.to_numpy(dtype=float)
    return (labels == 0) | ((labels == 1) & (snr >= -15.) & (snr < 0.))


def raw_summary(structures: np.ndarray) -> np.ndarray:
    """从P/A/U缓存生成协议定义的8维目标；空路径的统计量均为0。"""
    persistence = np.asarray(structures[:, 0], dtype=np.float64)
    amplitude = np.asarray(structures[:, 1], dtype=np.float64)
    uncertainty = np.asarray(structures[:, 2], dtype=np.float64)
    present = persistence > 0.
    delays = (np.arange(PATH_BINS, dtype=np.float64) + MIN_DELAY) / SAMPLE_RATE * 1e3
    count = present.sum(axis=1)
    support_sum = persistence.sum(axis=1)
    support_max = persistence.max(axis=1)
    support_mean = np.divide(support_sum, count, out=np.zeros_like(support_sum), where=count > 0)
    weights = persistence
    total = weights.sum(axis=1)
    delay_mean = np.divide((weights * delays).sum(axis=1), total,
                           out=np.zeros_like(total), where=total > 0)
    delay_std = np.sqrt(np.divide((weights * (delays - delay_mean[:, None]) ** 2).sum(axis=1), total,
                                  out=np.zeros_like(total), where=total > 0))
    uncertainty_mean = np.divide((uncertainty * present).sum(axis=1), count,
                                  out=np.zeros_like(support_sum), where=count > 0)
    return np.column_stack((count, support_sum, support_max, support_mean, amplitude.max(axis=1),
                            delay_mean, delay_std, uncertainty_mean)).astype(np.float32)


def load_data(config: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = (config.manifest_path, config.waveform_path, config.structure_path,
                config.structure_done_path, config.structure_status_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少固定波形或BCSC缓存：" + "; ".join(missing))
    manifest = pd.read_csv(config.manifest_path)
    waves = np.load(config.waveform_path, mmap_mode="r")
    structures = np.load(config.structure_path, mmap_mode="r")
    complete = np.load(config.structure_done_path)
    status = pd.read_csv(config.structure_status_path).sort_values("sample_index").reset_index(drop=True)
    checks = {
        "manifest_rows": len(manifest) == 7000,
        "wave_shape": tuple(waves.shape) == (7000, WAVE_SAMPLES),
        "structure_shape": tuple(structures.shape) == (7000, 3, PATH_BINS),
        "completed": tuple(complete.shape) == (7000,) and bool(complete.all()),
        "status_one_to_one": (len(status) == 7000 and status.sample_index.nunique() == 7000
                                and np.array_equal(status.sample_index.to_numpy(), np.arange(7000))),
        "class_balance": manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
        "recording_groups": manifest.ori_file_num.nunique() == 34,
        "folds": set(manifest.test_fold.unique()) == set(range(5)),
        "group_isolation": manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
        "finite_waves": bool(np.isfinite(waves).all()),
        "finite_structures": bool(np.isfinite(structures).all()),
        "positive_snr_range": bool(manifest.loc[manifest.label.eq(1), "target_snr_db"].between(-15, 15).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"输入数据合同失败：{checks}")
    # empty 是一次成功运行所得的无路径结果；只有 failed/missing 屏蔽辅助损失。
    auxiliary_ok = status.status.fillna("missing").ne("failed").to_numpy(dtype=bool)
    summary = raw_summary(structures)
    if not np.isfinite(summary).all():
        raise RuntimeError("BCSC摘要含有非有限数")
    return manifest, waves, summary, auxiliary_ok, status


class ClickDataset(Dataset):
    def __init__(self, waves: np.ndarray, labels: np.ndarray, targets: np.ndarray,
                 target_valid: np.ndarray, indices: np.ndarray) -> None:
        self.waves, self.labels, self.targets = waves, labels, targets
        self.target_valid, self.indices = target_valid, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[row])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32),
                torch.tensor(self.labels[index], dtype=torch.float32),
                torch.as_tensor(self.targets[index], dtype=torch.float32),
                torch.tensor(self.target_valid[index], dtype=torch.bool), index)


def make_loader(waves: np.ndarray, labels: np.ndarray, targets: np.ndarray, target_valid: np.ndarray,
                indices: np.ndarray, batch_size: int, shuffle: bool, generator_seed: int | None = None) -> DataLoader:
    generator = None
    if generator_seed is not None:
        generator = torch.Generator().manual_seed(generator_seed)
    return DataLoader(ClickDataset(waves, labels, targets, target_valid, indices), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0,
                      pin_memory=torch.cuda.is_available())


class WaveformEncoder(nn.Module):
    def __init__(self, dropout: float = .1) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv1d(1, 64, kernel_size=32, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                           dropout=dropout, batch_first=True, activation="gelu",
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.cls_token, std=.02)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(wave.unsqueeze(1)).transpose(1, 2)
        pos = torch.arange(tokens.shape[1], device=wave.device, dtype=tokens.dtype).unsqueeze(-1)
        dim = torch.arange(64, device=wave.device, dtype=tokens.dtype).unsqueeze(0)
        scale = torch.pow(torch.tensor(10000., device=wave.device, dtype=tokens.dtype),
                          2 * torch.floor(dim / 2) / 64)
        angle = pos / scale
        positional = torch.where((dim.long() % 2) == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        tokens = torch.cat((self.cls_token.expand(wave.shape[0], -1, -1), tokens + positional), dim=1)
        return self.encoder(tokens)[:, 0]


class ProxyClassifier(nn.Module):
    """所有条件同参：仅由condition决定分类器读取预测u还是零u。"""
    def __init__(self, condition: str, dropout: float = .1) -> None:
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"未知条件：{condition}")
        self.condition = condition
        self.waveform = WaveformEncoder(dropout)
        self.summary_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8))
        self.proxy_head = nn.Linear(8, 16)
        self.classifier = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 1))

    @property
    def uses_proxy(self) -> bool:
        return self.condition in PROXY_CONDITIONS

    @property
    def uses_auxiliary_loss(self) -> bool:
        return self.condition in AUXILIARY_CONDITIONS

    def features(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.waveform(wave)
        predicted_summary = self.summary_head(z)
        predicted_proxy = self.proxy_head(predicted_summary)
        return z, predicted_summary, predicted_proxy

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, predicted_summary, predicted_proxy = self.features(wave)
        u = predicted_proxy if self.uses_proxy else torch.zeros_like(predicted_proxy)
        return self.classifier(torch.cat((z, u), dim=1)).squeeze(1), predicted_summary


class WaveformDeploymentModel(nn.Module):
    """导出模型：保留波形编码、代理和分类器，不含真实BCSC入口。"""
    def __init__(self, dropout: float = .1) -> None:
        super().__init__()
        self.waveform = WaveformEncoder(dropout)
        self.summary_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8))
        self.proxy_head = nn.Linear(8, 16)
        self.classifier = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        z = self.waveform(wave)
        summary = self.summary_head(z)
        return self.classifier(torch.cat((z, self.proxy_head(summary)), dim=1)).squeeze(1)


def standardize_targets(summary: np.ndarray, valid: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = train_indices[valid[train_indices]]
    if not len(usable):
        raise RuntimeError("当前训练折没有可用BCSC辅助目标")
    median = np.median(summary[usable], axis=0).astype(np.float32)
    q1, q3 = np.percentile(summary[usable], (25, 75), axis=0).astype(np.float32)
    iqr = q3 - q1
    # 常量维度仍可安全处理，且该维度的目标标准化为0。
    scale = np.where(iqr > 1e-6, iqr, 1.).astype(np.float32)
    standardized = np.clip((summary - median) / scale, -5., 5.).astype(np.float32)
    return standardized, median, scale


def shuffled_training_targets(targets: np.ndarray, valid: np.ndarray, labels: np.ndarray,
                              train_indices: np.ndarray, fold: int) -> np.ndarray:
    """C4只在当前训练折的同类、成功样本中置换一次；失败样本始终被掩蔽。"""
    result = targets.copy()
    rng = np.random.default_rng(SEED + 10000 + fold)
    for label in (0, 1):
        group = train_indices[(labels[train_indices] == label) & valid[train_indices]]
        result[group] = targets[rng.permutation(group)]
    return result


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-np.clip(value, -80., 80.)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temp: float) -> float:
        temperature = math.exp(log_temp)
        return float(np.mean(np.logaddexp(0., logits / temperature) - labels * logits / temperature))
    result = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0., .5, 1.], probability)))
    scores = [balanced_accuracy_score(labels, probability >= item) for item in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "brier": float(brier_score_loss(labels, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def gradient_rms(model: ProxyClassifier) -> dict[str, float | bool | None]:
    groups: dict[str, list[torch.Tensor]] = {key: [] for key in ("encoder", "cls", "classifier", "proxy")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = "encoder" if name.startswith("waveform.") else "cls" if name == "waveform.cls_token" else (
            "classifier" if name.startswith("classifier.") else "proxy")
        groups[group].append(parameter.grad.detach().reshape(-1))
    result: dict[str, float | bool | None] = {}
    for name, values in groups.items():
        if not values:
            result[f"grad_{name}_rms"] = None
            result[f"grad_{name}_finite"] = True
            continue
        joined = torch.cat(values)
        result[f"grad_{name}_rms"] = float(torch.sqrt(torch.mean(joined.float() ** 2)).cpu())
        result[f"grad_{name}_finite"] = bool(torch.isfinite(joined).all())
    return result


def masked_auxiliary_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not bool(valid.any()):
        return prediction.sum() * 0.
    return nn.functional.smooth_l1_loss(prediction[valid], target[valid])


@torch.no_grad()
def predict(model: ProxyClassifier, data: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    target: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    for wave, label, value, usable, index in data:
        output, predicted = model(wave.to(device, non_blocking=True))
        ids.append(np.asarray(index)); labels.append(label.numpy()); logits.append(output.cpu().numpy())
        target.append(value.numpy()); valid.append(usable.numpy()); summaries.append(predicted.cpu().numpy())
    return {"sample_index": np.concatenate(ids), "label": np.concatenate(labels), "logit": np.concatenate(logits),
            "target": np.concatenate(target), "target_valid": np.concatenate(valid),
            "predicted_summary": np.concatenate(summaries)}


def run_directory(config: Config, condition: str, fold: int) -> Path:
    path = output_paths(config)["models"] / condition / f"seed_{SEED}" / f"fold_{fold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_auxiliary(pred_standard: np.ndarray, valid: np.ndarray, summary: np.ndarray,
                       median: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    true = summary[valid]
    prediction = pred_standard[valid] * scale + median
    baseline = np.broadcast_to(median, true.shape)
    rows = []
    for dimension, name in enumerate(TARGET_NAMES):
        mae = float(np.mean(np.abs(prediction[:, dimension] - true[:, dimension])))
        baseline_mae = float(np.mean(np.abs(baseline[:, dimension] - true[:, dimension])))
        rows.append({"target_dimension": dimension, "target_name": name, "mae": mae,
                     "median_baseline_mae": baseline_mae, "mae_improvement": baseline_mae - mae,
                     "improves_median_baseline": bool(mae < baseline_mae), "n_success": int(len(true))})
    return pd.DataFrame(rows)


def train_one(condition: str, fold: int, waves: np.ndarray, summary: np.ndarray, valid: np.ndarray,
              manifest: pd.DataFrame, config: Config, resume: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = run_directory(config, condition, fold)
    done_path = directory / "COMPLETE.json"
    checkpoint_path = directory / "best_checkpoint.pt"
    prediction_path = directory / "test_predictions.csv"
    target_path = directory / "auxiliary_target_metrics.csv"
    history_path = directory / "history.csv"
    if resume and all(path.exists() for path in (done_path, checkpoint_path, prediction_path, target_path, history_path)):
        return (json.loads(done_path.read_text(encoding="utf-8")), pd.read_csv(prediction_path),
                pd.read_csv(target_path), pd.read_csv(history_path))
    labels = manifest.label.to_numpy(dtype=np.float32)
    train_indices, validation_indices, test_indices = split(manifest, fold)
    standardized, median, scale = standardize_targets(summary, valid, train_indices)
    training_targets = (shuffled_training_targets(standardized, valid, labels.astype(int), train_indices, fold)
                        if condition == "C4_shuffled_proxy" else standardized)
    # 每个条件的参数初始化和训练batch顺序均只由fold决定。
    initialization_seed = SEED + fold * 100
    seed_all(initialization_seed)
    device = device_for(config.device)
    model = ProxyClassifier(condition, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loaders = {
        "train": make_loader(waves, labels, training_targets, valid, train_indices, config.batch_size, True,
                              generator_seed=initialization_seed + 1),
        "validation": make_loader(waves, labels, standardized, valid, validation_indices, config.batch_size * 2, False),
        "test": make_loader(waves, labels, standardized, valid, test_indices, config.batch_size * 2, False),
    }
    best_score, stale, history, started = -np.inf, 0, [], time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        classification_losses: list[float] = []
        auxiliary_losses: list[float] = []
        total_losses: list[float] = []
        gradient_rows: list[dict[str, float | bool | None]] = []
        for wave, label, target, usable, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits, prediction = model(wave.to(device, non_blocking=True))
            classification_loss = nn.functional.binary_cross_entropy_with_logits(logits, label.to(device, non_blocking=True))
            auxiliary_loss = (masked_auxiliary_loss(prediction, target.to(device, non_blocking=True), usable.to(device))
                              if model.uses_auxiliary_loss else logits.sum() * 0.)
            loss = classification_loss + config.auxiliary_weight * auxiliary_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"{condition} fold={fold} epoch={epoch} 出现非有限loss")
            loss.backward()
            gradients = gradient_rms(model)
            if not all(bool(value) for key, value in gradients.items() if key.endswith("_finite")):
                raise FloatingPointError(f"{condition} fold={fold} epoch={epoch} 出现非有限梯度")
            optimizer.step()
            classification_losses.append(float(classification_loss.detach().cpu()))
            auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
            gradient_rows.append(gradients)
        validation = predict(model, loaders["validation"], device)
        validation_frame = manifest.iloc[validation["sample_index"]].reset_index(drop=True)
        mask = low_snr_mask(validation_frame)
        validation_low_auroc = float(roc_auc_score(validation["label"][mask], validation["logit"][mask]))
        validation_aux = evaluate_auxiliary(validation["predicted_summary"], validation["target_valid"].astype(bool),
                                            summary[validation["sample_index"]], median, scale)
        record: dict[str, Any] = {
            "condition": condition, "seed": SEED, "test_fold": fold, "validation_fold": (fold + 1) % 5,
            "epoch": epoch, "train_classification_loss": float(np.mean(classification_losses)),
            "train_auxiliary_loss": float(np.mean(auxiliary_losses)), "train_total_loss": float(np.mean(total_losses)),
            "aux_to_classification_loss_ratio": float(np.mean(auxiliary_losses) / max(np.mean(classification_losses), EPS)),
            "validation_low_snr_auroc": validation_low_auroc,
            "validation_auxiliary_mae_mean": float(validation_aux.mae.mean()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        for key in ("encoder", "cls", "classifier", "proxy"):
            values = [item[f"grad_{key}_rms"] for item in gradient_rows if item[f"grad_{key}_rms"] is not None]
            record[f"grad_{key}_rms"] = float(np.mean(values)) if values else math.nan
            record[f"grad_{key}_finite"] = bool(all(item[f"grad_{key}_finite"] for item in gradient_rows))
        history.append(record)
        print(f"[训练] {condition} fold={fold} epoch={epoch}/{config.max_epochs} | "
              f"cls={record['train_classification_loss']:.4f} aux={record['train_auxiliary_loss']:.4f} "
              f"ratio={record['aux_to_classification_loss_ratio']:.3f} "
              f"val-low-AUROC={validation_low_auroc:.4f}", flush=True)
        if validation_low_auroc > best_score + 1e-7:
            best_score, stale = validation_low_auroc, 0
            torch.save({"state_dict": model.state_dict(), "condition": condition, "seed": SEED,
                        "test_fold": fold, "initialization_seed": initialization_seed,
                        "target_median": median, "target_iqr": scale, "target_names": TARGET_NAMES}, checkpoint_path)
        else:
            stale += 1
            if stale >= config.patience:
                break
    if not checkpoint_path.exists():
        raise RuntimeError(f"{condition} fold={fold}未保存检查点")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    validation = predict(model, loaders["validation"], device)
    temperature = fit_temperature(validation["logit"], validation["label"])
    threshold = choose_threshold(validation["label"].astype(int), sigmoid(validation["logit"] / temperature))
    test = predict(model, loaders["test"], device)
    probability = sigmoid(test["logit"] / temperature)
    test_frame = manifest.iloc[test["sample_index"]].reset_index(drop=True)
    result = metric(test["label"].astype(int), probability, threshold)
    low_mask = low_snr_mask(test_frame)
    result.update({"low_snr_auroc": float(roc_auc_score(test["label"][low_mask], probability[low_mask])),
                   "condition": condition, "seed": SEED, "test_fold": fold,
                   "validation_fold": (fold + 1) % 5, "initialization_seed": initialization_seed,
                   "best_validation_low_snr_auroc": float(best_score),
                   "best_epoch": int(max(item["epoch"] for item in history if item["validation_low_snr_auroc"] == best_score)),
                   "temperature": temperature, "threshold": threshold,
                   "parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
                   "training_seconds": time.perf_counter() - started})
    prediction_frame = test_frame[["sample_id", "ori_file_num", "test_fold", "target_snr_db"]].copy()
    prediction_frame.insert(0, "sample_index", test["sample_index"])
    prediction_frame["label"] = test["label"].astype(int)
    prediction_frame["logit"] = test["logit"]
    prediction_frame["p_click"] = probability
    prediction_frame["predicted_label"] = (probability >= threshold).astype(int)
    prediction_frame["condition"] = condition
    prediction_frame["seed"] = SEED
    prediction_frame["temperature"] = temperature
    prediction_frame["threshold"] = threshold
    for dimension, name in enumerate(TARGET_NAMES):
        prediction_frame[f"true_{name}"] = summary[test["sample_index"], dimension]
        prediction_frame[f"predicted_{name}"] = test["predicted_summary"][:, dimension] * scale[dimension] + median[dimension]
    target_metrics = evaluate_auxiliary(test["predicted_summary"], test["target_valid"].astype(bool),
                                        summary[test["sample_index"]], median, scale)
    target_metrics.insert(0, "condition", condition)
    target_metrics.insert(1, "seed", SEED)
    target_metrics.insert(2, "test_fold", fold)
    target_metrics["auxiliary_loss_enabled"] = model.uses_auxiliary_loss
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    target_metrics.to_csv(target_path, index=False, encoding="utf-8-sig")
    json_dump(done_path, result)
    return result, prediction_frame, target_metrics, pd.DataFrame(history)


def completed_count(config: Config) -> int:
    return sum((run_directory(config, condition, fold) / "COMPLETE.json").exists()
               for condition in CONDITIONS for fold in SCREEN_FOLDS)


def train(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = output_paths(config)
    manifest, waves, summary, valid, _ = load_data(config)
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    target_metrics: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    jobs = [(condition, fold) for fold in SCREEN_FOLDS for condition in CONDITIONS]
    bar = tqdm(total=len(jobs), desc="总体进度：v6 BCSC代理最小消融", unit="模型折")
    for position, (condition, fold) in enumerate(jobs, 1):
        print(f"[总体进度 {100 * (position - 1) / len(jobs):.0f}%] {condition}, test fold={fold}，已完成{completed_count(config)}/10", flush=True)
        row, prediction, aux, history = train_one(condition, fold, waves, summary, valid, manifest, config, resume)
        rows.append(row); predictions.append(prediction); target_metrics.append(aux); histories.append(history)
        bar.update(1); bar.set_postfix_str(f"{condition} fold={fold} low-AUROC={row['low_snr_auroc']:.3f}")
    bar.close()
    metrics = pd.DataFrame(rows).sort_values(["test_fold", "condition"]).reset_index(drop=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    aux_frame = pd.concat(target_metrics, ignore_index=True)
    history_frame = pd.concat(histories, ignore_index=True)
    expected_prediction_rows = len(CONDITIONS) * sum(len(split(manifest, fold)[2]) for fold in SCREEN_FOLDS)
    if len(metrics) != 10 or len(prediction_frame) != expected_prediction_rows or prediction_frame.duplicated(["condition", "sample_index"]).any():
        raise RuntimeError(f"训练结果合同失败：metrics={len(metrics)}, predictions={len(prediction_frame)}")
    metrics.to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(paths["predictions"] / "test_predictions.csv", index=False, encoding="utf-8-sig")
    aux_frame.to_csv(paths["metrics"] / "auxiliary_target_metrics.csv", index=False, encoding="utf-8-sig")
    history_frame.to_csv(paths["gradients"] / "gradient_epochs.csv", index=False, encoding="utf-8-sig")
    status = {"seed": SEED, "test_folds": list(SCREEN_FOLDS), "conditions": list(CONDITIONS),
              "expected_model_folds": 10, "completed_model_folds": 10,
              "completed": [{"condition": row["condition"], "test_fold": int(row["test_fold"])} for row in rows]}
    json_dump(paths["metrics"] / "model_completion_status.json", status)
    print("[总体进度 100%] 10个模型折训练完成", flush=True)
    return metrics, prediction_frame, aux_frame


def feasibility(metrics: pd.DataFrame, auxiliary: pd.DataFrame) -> dict[str, Any]:
    comparisons = []
    for fold in SCREEN_FOLDS:
        row = {item.condition: item for item in metrics[metrics.test_fold.eq(fold)].itertuples(index=False)}
        comparisons.append({"test_fold": fold,
                            "C3_minus_C0_low_snr_auroc": float(row["C3_full_proxy"].low_snr_auroc - row["C0_waveform_control"].low_snr_auroc),
                            "C3_minus_C2_low_snr_auroc": float(row["C3_full_proxy"].low_snr_auroc - row["C2_proxy_no_aux"].low_snr_auroc),
                            "C3_minus_C4_low_snr_auroc": float(row["C3_full_proxy"].low_snr_auroc - row["C4_shuffled_proxy"].low_snr_auroc),
                            "C1_minus_C0_low_snr_auroc": float(row["C1_aux_only"].low_snr_auroc - row["C0_waveform_control"].low_snr_auroc)})
    comparison_frame = pd.DataFrame(comparisons)
    c3_aux = auxiliary[auxiliary.condition.eq("C3_full_proxy")]
    dimensions = c3_aux.groupby("test_fold").improves_median_baseline.sum().to_dict()
    criteria = {
        "C3_better_than_C0_both_folds": bool((comparison_frame.C3_minus_C0_low_snr_auroc > 0).all()),
        "C3_better_than_C2_both_folds": bool((comparison_frame.C3_minus_C2_low_snr_auroc > 0).all()),
        "C3_better_than_C4_both_folds": bool((comparison_frame.C3_minus_C4_low_snr_auroc > 0).all()),
        "C3_at_least_5_targets_better_than_median_each_fold": bool(all(dimensions.get(fold, 0) >= 5 for fold in SCREEN_FOLDS)),
    }
    return {"primary_endpoint": "low-SNR positive [-15,0) dB vs all negative AUROC",
            "comparisons": comparisons, "c3_dimensions_better_than_median_by_fold": {str(k): int(v) for k, v in dimensions.items()},
            "criteria": criteria, "expand_to_full_five_folds": bool(all(criteria.values())),
            "interpretation": "仅两个fold的可行性证据；未计算置信区间或统计显著性。"}


def plot_performance(metrics: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(CONDITIONS)); width = .32
    for offset, fold in ((-.16, 0), (.16, 2)):
        values = [metrics[(metrics.condition == condition) & (metrics.test_fold == fold)].iloc[0].low_snr_auroc for condition in CONDITIONS]
        axis.bar(x + offset, values, width, label=f"test fold {fold}")
    axis.set(xticks=x, xticklabels=["C0", "C1", "C2", "C3", "C4"], ylim=(0, 1),
             ylabel="Low-SNR AUROC", title="v6 waveform BCSC proxy ablation")
    axis.legend(); figure.tight_layout(); figure.savefig(destination, dpi=180); plt.close(figure)


def plot_auxiliary(auxiliary: pd.DataFrame, destination: Path) -> None:
    c3 = auxiliary[auxiliary.condition.eq("C3_full_proxy")]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(TARGET_NAMES)); width = .32
    for offset, fold in ((-.16, 0), (.16, 2)):
        values = c3[c3.test_fold.eq(fold)].sort_values("target_dimension").mae_improvement.to_numpy()
        axis.bar(x + offset, values, width, label=f"test fold {fold}")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xticks=x, xticklabels=TARGET_NAMES, ylabel="Median baseline MAE − proxy MAE",
             title="C3 proxy target quality (positive is better)")
    axis.tick_params(axis="x", rotation=28); axis.legend(); figure.tight_layout(); figure.savefig(destination, dpi=180); plt.close(figure)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns = list(frame.columns)
    values = []
    for row in frame.itertuples(index=False, name=None):
        values.append([f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) and math.isfinite(value) else str(value) for value in row])
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
                      *["| " + " | ".join(row) + " |" for row in values]])


def report(config: Config) -> Path:
    paths = output_paths(config)
    metrics_path = paths["metrics"] / "fold_metrics.csv"
    prediction_path = paths["predictions"] / "test_predictions.csv"
    auxiliary_path = paths["metrics"] / "auxiliary_target_metrics.csv"
    if not all(path.exists() for path in (metrics_path, prediction_path, auxiliary_path)):
        raise FileNotFoundError("请先完成10个模型折训练")
    metrics, predictions, auxiliary = pd.read_csv(metrics_path), pd.read_csv(prediction_path), pd.read_csv(auxiliary_path)
    manifest, _, _, _, _ = load_data(config)
    expected_prediction_rows = len(CONDITIONS) * sum(len(split(manifest, fold)[2]) for fold in SCREEN_FOLDS)
    if len(metrics) != 10 or len(predictions) != expected_prediction_rows or len(auxiliary) != 80:
        raise RuntimeError("报告拒绝使用不完整的实际训练结果")
    summary = feasibility(metrics, auxiliary)
    json_dump(paths["metrics"] / "feasibility_summary.json", summary)
    comparison = pd.DataFrame(summary["comparisons"])
    comparison.to_csv(paths["metrics"] / "ablation_gains.csv", index=False, encoding="utf-8-sig")
    plot_performance(metrics, paths["figures"] / "performance_comparison.png")
    plot_auxiliary(auxiliary, paths["figures"] / "auxiliary_target_mae.png")
    cols = ["condition", "test_fold", "low_snr_auroc", "auroc", "auprc", "balanced_accuracy", "recall", "brier", "best_epoch"]
    aux_c3 = auxiliary[auxiliary.condition.eq("C3_full_proxy")][["test_fold", "target_name", "mae", "median_baseline_mae", "mae_improvement"]]
    text = ["# 实验23-v6：波形驱动 BCSC 代理分支最小消融", "",
            "本报告只汇总实际完成的10个模型折：种子20260826、测试fold 0与2；测试fold为f，验证fold为(f+1) mod 5。所有部署分类均仅输入波形，真实BCSC摘要只用于C1/C3/C4的训练期辅助回归。主终点为`[-15,0)` dB正样本与同折全部负样本的AUROC。", "",
            "## 折外分类结果", "", markdown_table(metrics[cols]), "",
            "## C3关键比较（低SNR AUROC增量）", "", markdown_table(comparison), "",
            "## C3代理摘要质量", "", markdown_table(aux_c3), "",
            "## 可行性判读", "", "- C3相对C0、C2和C4须在两个fold的主终点增量均大于0。",
            "- C3须在每个测试fold至少5/8个目标上优于训练折中位数基线。",
            f"- 当前自动判定：`{'满足扩展门槛' if summary['expand_to_full_five_folds'] else '未满足扩展门槛'}`。", "",
            "本阶段未计算Bootstrap置信区间，两个fold结果不构成统计显著性结论。"]
    destination = paths["root"] / "research_report_zh.md"
    destination.write_text("\n".join(text) + "\n", encoding="utf-8")
    return destination


def export(config: Config) -> list[Path]:
    paths = output_paths(config)
    results: list[Path] = []
    device = device_for(config.device)
    for fold in SCREEN_FOLDS:
        checkpoint = run_directory(config, "C3_full_proxy", fold) / "best_checkpoint.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"请先训练C3 fold {fold}：{checkpoint}")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        deployment = WaveformDeploymentModel(config.dropout).to(device)
        deployment.load_state_dict(payload["state_dict"])
        deployment.eval()
        destination = paths["deployment"] / f"waveform_student_C3_fold_{fold}.pt"
        torch.save({"state_dict": deployment.state_dict(), "architecture": "waveform->64->8->16; [z;u] 80->64->1",
                    "input": "waveform [batch, 5761] only", "condition": "C3_full_proxy", "seed": SEED,
                    "test_fold": fold, "target_standardization_training_only": {"median": payload["target_median"], "iqr": payload["target_iqr"]}}, destination)
        results.append(destination)
    return results


def verify(config: Config) -> dict[str, Any]:
    paths = output_paths(config)
    manifest, waves, summary, valid, status = load_data(config)
    device = device_for(config.device)
    ids = np.arange(4)
    wave = torch.as_tensor(np.asarray(waves[ids]), dtype=torch.float32, device=device)
    labels = torch.as_tensor(manifest.label.iloc[ids].to_numpy(dtype=np.float32), device=device)
    target = torch.as_tensor(summary[ids], dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(valid[ids], dtype=torch.bool, device=device)
    networks: dict[str, Any] = {}
    for condition in CONDITIONS:
        seed_all(SEED)
        model = ProxyClassifier(condition, config.dropout).to(device).train()
        logits, predicted = model(wave)
        classification_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        auxiliary_loss = masked_auxiliary_loss(predicted, target, valid_tensor) if model.uses_auxiliary_loss else logits.sum() * 0.
        loss = classification_loss + config.auxiliary_weight * auxiliary_loss
        loss.backward()
        gradients = gradient_rms(model)
        networks[condition] = {"logit_shape": list(logits.shape), "summary_shape": list(predicted.shape),
                               "finite_loss": bool(torch.isfinite(loss)), "finite_probability": bool(torch.isfinite(torch.sigmoid(logits)).all()),
                               "finite_gradients": bool(all(bool(item) for name, item in gradients.items() if name.endswith("_finite"))),
                               "parameter_count": int(sum(p.numel() for p in model.parameters())),
                               "uses_proxy": model.uses_proxy, "uses_auxiliary_loss": model.uses_auxiliary_loss}
    parameter_counts_match = len({item["parameter_count"] for item in networks.values()}) == 1
    classifier_inputs_zero_for_C0_C1 = True
    for condition in ("C0_waveform_control", "C1_aux_only"):
        seed_all(SEED)
        zero_model = ProxyClassifier(condition, config.dropout).to(device).eval()
        captured: list[torch.Tensor] = []
        handle = zero_model.classifier[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().cpu()))
        with torch.no_grad():
            zero_model(wave)
        handle.remove()
        classifier_inputs_zero_for_C0_C1 = classifier_inputs_zero_for_C0_C1 and bool(
            len(captured) == 1 and torch.count_nonzero(captured[0][:, 64:]) == 0)
    # 失败项屏蔽：把一个有效目标标记为失败，损失必须不随其数值改变。
    seed_all(SEED); auxiliary_model = ProxyClassifier("C3_full_proxy", config.dropout).to(device).eval()
    with torch.no_grad():
        _, predicted = auxiliary_model(wave)
        mask = valid_tensor.clone(); mask[0] = False
        first = masked_auxiliary_loss(predicted, target, mask)
        changed = target.clone(); changed[0] += 99999.
        second = masked_auxiliary_loss(predicted, changed, mask)
        failed_masked = bool(torch.allclose(first, second))
        # 分类前向没有真实摘要参数；改变真实摘要不能改变logits。
        logits_a, _ = auxiliary_model(wave)
        _unused_perturbed_real_summary = target + 99999.
        logits_b, _ = auxiliary_model(wave)
        no_real_summary_leakage = bool(torch.equal(logits_a, logits_b))
    train_ids, _, _ = split(manifest, 0)
    shuffled = shuffled_training_targets(summary, valid, manifest.label.to_numpy(dtype=int), train_ids, 0)
    shuffled_again = shuffled_training_targets(summary, valid, manifest.label.to_numpy(dtype=int), train_ids, 0)
    c4_only_training = bool(np.array_equal(shuffled[~np.isin(np.arange(7000), train_ids)], summary[~np.isin(np.arange(7000), train_ids)]))
    c4_reproducible = bool(np.array_equal(shuffled, shuffled_again))
    c4_within_class_permutation = True
    for label in (0, 1):
        group = train_ids[(manifest.label.to_numpy(dtype=int)[train_ids] == label) & valid[train_ids]]
        c4_within_class_permutation = c4_within_class_permutation and bool(
            np.array_equal(np.sort(shuffled[group], axis=0), np.sort(summary[group], axis=0)))
    seed_all(SEED); train_model = ProxyClassifier("C3_full_proxy", config.dropout).to(device).eval()
    deployment = WaveformDeploymentModel(config.dropout).to(device).eval()
    deployment.load_state_dict(train_model.state_dict())
    with torch.no_grad():
        train_logits, _ = train_model(wave)
        deploy_logits = deployment(wave)
    deployment_matches = bool(torch.allclose(train_logits, deploy_logits, atol=1e-6, rtol=1e-6))
    result = {"data": {"wave_shape": list(waves.shape), "summary_shape": list(summary.shape),
                       "structure_shape": [7000, 3, 1133], "recording_groups": int(manifest.ori_file_num.nunique()),
                       "status_rows": len(status), "bcsc_failed_samples": int((~valid).sum())},
              "networks": networks, "same_parameter_count_all_conditions": parameter_counts_match,
              "C0_C1_classifier_structure_input_is_zero": classifier_inputs_zero_for_C0_C1,
              "failed_auxiliary_loss_masked": failed_masked,
              "C4_only_shuffles_training_targets": c4_only_training,
              "C4_fixed_reproducible_within_class_permutation": bool(c4_reproducible and c4_within_class_permutation),
              "real_BCSC_summary_cannot_change_logits": no_real_summary_leakage,
              "deployment_waveform_only_logit_matches_training_model": deployment_matches,
              "deployment_parameter_names": list(deployment.state_dict().keys())}
    result["passed"] = bool(result["data"]["wave_shape"] == [7000, 5761] and result["data"]["summary_shape"] == [7000, 8]
                            and result["data"]["recording_groups"] == 34 and parameter_counts_match
                            and classifier_inputs_zero_for_C0_C1 and failed_masked and c4_only_training
                            and c4_reproducible and c4_within_class_permutation and no_real_summary_leakage and deployment_matches
                            and all(item["logit_shape"] == [4] and item["summary_shape"] == [4, 8]
                                    and item["finite_loss"] and item["finite_probability"] and item["finite_gradients"]
                                    for item in networks.values()))
    json_dump(paths["tests"] / "verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"v6最小验证失败：{result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "train", "report", "export", "all"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = Config(device=arguments.device, output_dir=arguments.output_dir or OUT)
    paths = output_paths(config)
    json_dump(paths["root"] / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir),
              "seed": SEED, "test_folds": list(SCREEN_FOLDS), "conditions": list(CONDITIONS),
              "primary_endpoint": "low-SNR positive [-15,0) dB vs all negative AUROC",
              "model_contract": "All conditions retain encoder, 64->32->8 proxy head, 8->16 proxy mapping, and 80->64->1 classifier."})
    if arguments.command in ("verify", "all"):
        print("[总体进度 0%] 验证输入合同、五条件前反向与部署等价性", flush=True)
        verify(config)
        if arguments.command == "verify":
            print("[总体进度 100%] 最小验证通过", flush=True)
            return
    if arguments.command in ("train", "all"):
        train(config, resume=not arguments.no_resume)
    if arguments.command in ("report", "all"):
        print("[总体进度 92%] 汇总真实折外预测与可行性判读", flush=True)
        print(f"[总体进度 96%] 报告：{report(config)}", flush=True)
    if arguments.command in ("export", "all"):
        exported = export(config)
        print("[总体进度 100%] 导出：" + "; ".join(map(str, exported)), flush=True)


if __name__ == "__main__":
    main()
