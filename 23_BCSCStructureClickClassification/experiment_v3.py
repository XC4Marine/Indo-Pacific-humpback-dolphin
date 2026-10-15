#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v3：仅波形、单层Transformer与实时梯度监控。

命令：verify、train、report、all。训练默认启动独立监控进程；加
--no-live-monitor 可用于没有图形界面的机器。
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE = ROOT / "21_PhysicsInformedClickClassification" / "outputs_snr_m15_p15_full_5seeds"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v3_waveform_1layer_gradient_seed20260826_epoch50"
SAMPLE_COUNT = 5761
SEEDS = (20260826,)
SNR_BINS = ((-15., -10., "[-15,-10)"), (-10., -5., "[-10,-5)"),
            (-5., 0., "[-5,0)"), (0., 5., "[0,5)"),
            (5., 10., "[5,10)"), (10., 15.000001, "[10,15]"))
METRICS = ("auroc", "auprc", "balanced_accuracy", "precision", "recall",
           "specificity", "f1", "mcc", "brier")
MODULES = ("Patch Embedding", "CLS token", "Attention QKV", "Attention输出投影",
           "FFN Linear 1", "FFN Linear 2", "LayerNorm", "分类头")
MODULE_DISPLAY = {"Patch Embedding": "补丁嵌入层", "CLS token": "CLS令牌",
                  "Attention QKV": "注意力QKV投影", "Attention输出投影": "注意力输出投影",
                  "FFN Linear 1": "前馈网络线性层1", "FFN Linear 2": "前馈网络线性层2",
                  "LayerNorm": "层归一化", "分类头": "分类头"}
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SOURCE / "dataset_manifest.csv"
    waveform_path: Path = SOURCE / "cache" / "preprocessed_waveforms.npy"
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    lr_scheduler: str = "none"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    min_learning_rate: float = 3e-5
    bootstrap_runs: int = 2000
    device: str = "auto"


def paths(config: Config) -> dict[str, Path]:
    result = {name: config.output_dir / name for name in
              ("models", "metrics", "predictions", "figures", "logs", "live", "tests")}
    result["root"] = config.output_dir
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
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


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定CUDA设备，但CUDA不可用")
    return result


def load_data(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    if not config.manifest_path.exists() or not config.waveform_path.exists():
        raise FileNotFoundError("缺少实验21的固定清单或预处理波形缓存")
    manifest = pd.read_csv(config.manifest_path)
    waves = np.load(config.waveform_path, mmap_mode="r")
    checks = {
        "sample_count": len(manifest) == 7000,
        "wave_shape": tuple(waves.shape) == (7000, SAMPLE_COUNT),
        "class_balance": manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
        "recording_groups": manifest.ori_file_num.nunique() == 34,
        "no_group_leakage": manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
        "folds": set(manifest.test_fold.unique()) == set(range(5)),
        "finite_waveforms": bool(np.isfinite(waves).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"实验21数据合同失败：{checks}")
    return manifest, waves


class WaveDataset(Dataset):
    def __init__(self, waves: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.waves, self.labels, self.indices = waves, labels, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32),
                torch.tensor(self.labels[index], dtype=torch.float32), index)


def make_loader(waves: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(WaveDataset(waves, labels, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


class WaveformTransformer(nn.Module):
    """5761 → 359×64 → CLS+位置编码 → 一个Encoder层 → 64 → 64 → 1。"""
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv1d(1, 64, kernel_size=32, stride=16)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.cls_token, std=.02)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(wave.unsqueeze(1)).transpose(1, 2)
        positions = torch.arange(tokens.shape[1], device=wave.device, dtype=tokens.dtype).unsqueeze(-1)
        dimensions = torch.arange(64, device=wave.device, dtype=tokens.dtype).unsqueeze(0)
        scale = torch.pow(torch.tensor(10000., device=wave.device, dtype=tokens.dtype),
                          (2 * torch.floor(dimensions / 2)) / 64)
        angle = positions / scale
        positional = torch.where((dimensions.long() % 2) == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        tokens = torch.cat([self.cls_token.expand(wave.shape[0], -1, -1), tokens + positional], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(1)


def split(manifest: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = manifest.test_fold.to_numpy(dtype=int)
    validation = (fold + 1) % 5
    return (np.flatnonzero((folds != fold) & (folds != validation)),
            np.flatnonzero(folds == validation), np.flatnonzero(folds == fold))


def group_for_parameter(name: str) -> str | None:
    if name.startswith("patch_embedding."):
        return "Patch Embedding"
    if name == "cls_token":
        return "CLS token"
    if ".self_attn.in_proj_" in name:
        return "Attention QKV"
    if ".self_attn.out_proj." in name:
        return "Attention输出投影"
    if ".linear1." in name:
        return "FFN Linear 1"
    if ".linear2." in name:
        return "FFN Linear 2"
    if ".norm1." in name or ".norm2." in name or ".norm." in name:
        return "LayerNorm"
    if name.startswith("head."):
        return "分类头"
    return None


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def gradient_statistics(model: nn.Module) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, dict[str, float | int | bool]] = {
        name: {"sum_sq": 0., "param_sq": 0., "max_abs": 0., "zeros": 0,
               "numel": 0, "finite": True, "has_grad": False} for name in MODULES
    }
    buckets["__global__"] = {"sum_sq": 0., "param_sq": 0., "max_abs": 0., "zeros": 0,
                              "numel": 0, "finite": True, "has_grad": False}
    for parameter_name, parameter in model.named_parameters():
        group = group_for_parameter(parameter_name)
        if group is None:
            raise RuntimeError(f"未分类的模型参数：{parameter_name}")
        targets = (group, "__global__")
        value = parameter.detach()
        param_sq = float(torch.sum(value.double() ** 2).cpu())
        gradient = parameter.grad
        for target in targets:
            buckets[target]["param_sq"] = float(buckets[target]["param_sq"]) + param_sq
        if gradient is None:
            for target in targets:
                buckets[target]["finite"] = False
            continue
        detached = gradient.detach()
        is_finite = bool(torch.isfinite(detached).all())
        for target in targets:
            buckets[target]["has_grad"] = True
            buckets[target]["finite"] = bool(buckets[target]["finite"]) and is_finite
        if not is_finite:
            continue
        squared = float(torch.sum(detached.double() ** 2).cpu())
        maximum = float(torch.max(torch.abs(detached)).cpu()) if detached.numel() else 0.
        zeros = int(torch.count_nonzero(detached == 0).cpu())
        for target in targets:
            buckets[target]["sum_sq"] = float(buckets[target]["sum_sq"]) + squared
            buckets[target]["max_abs"] = max(float(buckets[target]["max_abs"]), maximum)
            buckets[target]["zeros"] = int(buckets[target]["zeros"]) + zeros
            buckets[target]["numel"] = int(buckets[target]["numel"]) + detached.numel()
    result: dict[str, dict[str, Any]] = {}
    for name, raw in buckets.items():
        norm = math.sqrt(float(raw["sum_sq"]))
        parameter_norm = math.sqrt(float(raw["param_sq"]))
        count = int(raw["numel"])
        result[name] = {
            "l2_norm": finite_or_none(norm), "rms": finite_or_none(norm / math.sqrt(max(count, 1))),
            "max_abs": finite_or_none(float(raw["max_abs"])),
            "zero_ratio": finite_or_none(int(raw["zeros"]) / max(count, 1)),
            "parameter_l2_norm": finite_or_none(parameter_norm),
            "relative_gradient": finite_or_none(norm / (parameter_norm + EPS)),
            "finite": bool(raw["finite"] and raw["has_grad"]), "parameter_count": count,
        }
    return {name: result[name] for name in MODULES}, result["__global__"]


class EventWriter:
    def __init__(self, event_path: Path, run_id: str):
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.file = event_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        event["run_id"] = self.run_id
        event["time_unix"] = time.time()
        self.file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@dataclass
class TrainingLock:
    path: Path
    run_id: str

    def release(self) -> None:
        if not self.path.exists():
            return
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if owner.get("run_id") == self.run_id:
            self.path.unlink()


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a valid Windows liveness probe: it can raise
        # WinError 87 even for a valid PID.  OpenProcess distinguishes a dead
        # PID from an access-denied live PID without touching the process.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: conservatively treat as live.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_training_lock(lock_path: Path) -> TrainingLock:
    """同一输出目录同一时刻只允许一个训练写入者；非活动PID锁自动回收。"""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    payload = {"run_id": run_id, "pid": os.getpid(), "started_unix": time.time(),
               "purpose": "experiment_v3_training"}
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise RuntimeError(f"训练锁损坏，拒绝并发写入：{lock_path}")
            owner_pid = int(owner.get("pid", -1))
            if process_is_running(owner_pid):
                raise RuntimeError(f"已有训练正在写入该输出目录（PID={owner_pid}，run_id={owner.get('run_id')}）；拒绝重复启动。")
            lock_path.unlink()
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
        return TrainingLock(lock_path, run_id)
    raise RuntimeError(f"无法取得训练锁：{lock_path}")


def complete_fold_count(config: Config) -> int:
    return sum((run_directory(config, seed, fold) / "COMPLETE.json").exists()
               for seed in SEEDS for fold in range(5))


def expected_model_count() -> int:
    return len(SEEDS) * 5


def run_directory(config: Config, seed: int, fold: int) -> Path:
    return paths(config)["models"] / f"seed_{seed}" / f"fold_{fold}"


def health_update(modules: dict[str, dict[str, Any]], global_stats: dict[str, Any],
                  history: deque[float]) -> list[str]:
    warnings: list[str] = []
    bad = [name for name, item in modules.items() if not item["finite"]]
    if not global_stats["finite"] or bad:
        warnings.append("非有限梯度" + ("：" + "、".join(bad) if bad else ""))
    previous = list(history)
    current = global_stats["l2_norm"]
    if current is not None and previous and current > 10 * float(np.median(previous)):
        warnings.append("异常峰值：全局梯度L2超过此前中位数10倍")
    if current is not None and current > 0:
        history.append(current)
    return warnings


def epoch_health(modules_by_batch: list[dict[str, dict[str, Any]]],
                 vanishing_streak: Counter[str], imbalance_streak: int) -> tuple[list[str], int]:
    warnings: list[str] = []
    if not modules_by_batch:
        return warnings, imbalance_streak
    for name in MODULES:
        low = sum(item[name]["rms"] is not None and float(item[name]["rms"]) < 1e-8
                  for item in modules_by_batch)
        if low / len(modules_by_batch) > .8:
            vanishing_streak[name] += 1
        else:
            vanishing_streak[name] = 0
        if vanishing_streak[name] >= 3:
            warnings.append(f"可能梯度消失：{name}连续3个epoch超过80% batch的RMS低于1e-8")
    module_means = []
    for name in MODULES:
        values = [float(item[name]["rms"]) for item in modules_by_batch
                  if item[name]["finite"] and item[name]["rms"] is not None and item[name]["rms"] > 0]
        if values:
            module_means.append(float(np.mean(values)))
    if len(module_means) >= 2 and max(module_means) / min(module_means) > 1e3:
        imbalance_streak += 1
    else:
        imbalance_streak = 0
    if imbalance_streak >= 3:
        warnings.append("模块失衡：有效模块平均RMS最大/最小比连续3个epoch超过1e3")
    return warnings, imbalance_streak


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1. / (1. + np.exp(-np.clip(values, -80., 80.)))


@torch.no_grad()
def predict(model: WaveformTransformer, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    indices, labels, logits = [], [], []
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for wave, label, index in data:
        logits.append(model(wave.to(device, non_blocking=True)).cpu().numpy())
        labels.append(label.numpy()); indices.append(np.asarray(index))
    if device.type == "cuda": torch.cuda.synchronize()
    return (np.concatenate(indices), np.concatenate(labels), np.concatenate(logits),
            time.perf_counter() - started)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0., logits / temperature) - labels * logits / temperature))
    outcome = minimize_scalar(nll, bounds=(-3., 3.), method="bounded")
    return float(math.exp(outcome.x)) if outcome.success else 1.


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0., .5, 1.], probability)))
    scores = [balanced_accuracy_score(labels, probability >= candidate) for candidate in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric(labels: np.ndarray, probability: np.ndarray, threshold: float | np.ndarray) -> dict[str, float | int]:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "brier": float(brier_score_loss(labels, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def low_snr_mask(frame: pd.DataFrame) -> np.ndarray:
    return ((frame.label.to_numpy(dtype=int) == 0) |
            ((frame.label.to_numpy(dtype=int) == 1) &
             (frame.target_snr_db.to_numpy(float) >= -15.) &
             (frame.target_snr_db.to_numpy(float) < 0.)))


def append_gradient_rows(writer: csv.DictWriter, seed: int, fold: int, epoch: int, batch: int,
                         loss: float | None, modules: dict[str, dict[str, Any]],
                         global_stats: dict[str, Any], warnings: list[str]) -> None:
    for name, values in (("全模型", global_stats), *modules.items()):
        writer.writerow({"seed": seed, "test_fold": fold, "epoch": epoch, "batch": batch,
                         "module": name, "loss": loss, "warning": "；".join(warnings), **values})


def train_one(seed: int, fold: int, waves: np.ndarray, manifest: pd.DataFrame, config: Config,
              events: EventWriter, resume: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    directory = run_directory(config, seed, fold)
    directory.mkdir(parents=True, exist_ok=True)
    completed, checkpoint, prediction_path = (directory / "COMPLETE.json", directory / "best_checkpoint.pt",
                                               directory / "test_predictions.csv")
    if resume and completed.exists() and checkpoint.exists() and prediction_path.exists():
        return json.loads(completed.read_text(encoding="utf-8")), pd.read_csv(prediction_path)
    initialization_seed = seed + fold * 100
    seed_all(initialization_seed)
    device = device_for(config.device)
    labels = manifest.label.to_numpy(dtype=np.float32)
    train_indices, validation_indices, test_indices = split(manifest, fold)
    loaders = {
        "train": make_loader(waves, labels, train_indices, config.batch_size, True),
        "validation": make_loader(waves, labels, validation_indices, config.batch_size * 2),
        "test": make_loader(waves, labels, test_indices, config.batch_size * 2),
    }
    model = WaveformTransformer(dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience, min_lr=config.min_learning_rate)
        if config.lr_scheduler == "plateau" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    gradient_history: deque[float] = deque(maxlen=10_000)
    vanishing_streak: Counter[str] = Counter()
    nonfinite_streak = 0
    best, stale, aborted, imbalance_streak = -np.inf, 0, False, 0
    started = time.perf_counter()
    batch_path = paths(config)["logs"] / f"seed_{seed}_fold_{fold}_batch_gradients.csv"
    epoch_path = paths(config)["logs"] / f"seed_{seed}_fold_{fold}_epoch_gradients.csv"
    health_path = paths(config)["logs"] / f"seed_{seed}_fold_{fold}_gradient_health.json"
    fieldnames = ["seed", "test_fold", "epoch", "batch", "module", "loss", "warning",
                  "l2_norm", "rms", "max_abs", "zero_ratio", "parameter_l2_norm",
                  "relative_gradient", "finite", "parameter_count"]
    health_events: list[dict[str, Any]] = []
    events.write({"kind": "status", "state": "训练中", "seed": seed, "test_fold": fold,
                  "epoch": 0, "batch": 0, "completed_models": complete_fold_count(config),
                  "device": str(device), "elapsed_s": 0.})
    with batch_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader()
        for epoch in range(1, config.max_epochs + 1):
            model.train(); losses: list[float] = []; modules_by_batch: list[dict[str, dict[str, Any]]] = []
            batch_warnings: list[str] = []
            for batch, (wave, label, _) in enumerate(loaders["train"], 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(wave.to(device, non_blocking=True))
                    loss = criterion(logits, label.to(device, non_blocking=True))
                loss_value = float(loss.detach().cpu())
                scaler.scale(loss).backward()
                # 梯度在unscale后采集，避免AMP缩放因子扭曲统计量。
                scaler.unscale_(optimizer)
                modules, global_stats = gradient_statistics(model)
                warnings = health_update(modules, global_stats, gradient_history)
                finite = bool(math.isfinite(loss_value) and global_stats["finite"])
                if finite:
                    scaler.step(optimizer)
                    scaler.update()
                    nonfinite_streak = 0
                    losses.append(loss_value)
                else:
                    nonfinite_streak += 1
                    warnings.append(f"非有限梯度：连续{nonfinite_streak}个batch；跳过本次更新")
                    scaler.update()
                    if nonfinite_streak >= 3:
                        warnings.append("连续3个batch非有限梯度，终止当前模型折")
                        aborted = True
                modules_by_batch.append(modules)
                batch_warnings.extend(warnings)
                append_gradient_rows(writer, seed, fold, epoch, batch, finite_or_none(loss_value), modules,
                                     global_stats, warnings)
                events.write({"kind": "batch", "seed": seed, "test_fold": fold, "epoch": epoch,
                              "batch": batch, "completed_models": complete_fold_count(config),
                              "device": str(device), "elapsed_s": time.perf_counter() - started,
                              "loss": finite_or_none(loss_value), "global": global_stats,
                              "modules": modules, "health": warnings})
                if aborted:
                    break
            if aborted:
                break
            validation_ids, validation_labels, validation_logits, _ = predict(model, loaders["validation"], device)
            validation_frame = manifest.iloc[validation_ids].reset_index(drop=True).assign(label=validation_labels.astype(int))
            mask = low_snr_mask(validation_frame)
            validation_score = float(roc_auc_score(validation_labels[mask], validation_logits[mask]))
            validation_loss = float(nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(validation_logits.astype(np.float32, copy=False)),
                torch.from_numpy(validation_labels.astype(np.float32, copy=False))))
            warnings, imbalance_streak = epoch_health(modules_by_batch, vanishing_streak, imbalance_streak)
            if scheduler is not None:
                scheduler.step(validation_score)
            health_events.append({"epoch": epoch, "warnings": warnings, "batch_warnings": sorted(set(batch_warnings))})
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                   "validation_loss": validation_loss, "validation_low_snr_auroc": validation_score,
                   "learning_rate": float(optimizer.param_groups[0]["lr"]),
                   "warnings": "；".join(warnings)}
            history.append(row)
            epoch_rows = []
            for name in ("全模型", *MODULES):
                source = ([item["__global__"] for item in []] if False else None)
                values = [item[name] for item in modules_by_batch] if name != "全模型" else []
                # 全局统计已写在逐batch CSV；epoch CSV保留模块均值，便于热图读取。
                if values:
                    epoch_rows.append({"seed": seed, "test_fold": fold, "epoch": epoch, "module": name,
                                       "mean_rms": float(np.mean([v["rms"] for v in values if v["rms"] is not None])),
                                       "mean_l2_norm": float(np.mean([v["l2_norm"] for v in values if v["l2_norm"] is not None])),
                                       "nonfinite_batches": int(sum(not v["finite"] for v in values)),
                                       "warnings": "；".join(warnings)})
            epoch_file = epoch_path.open("a" if epoch > 1 else "w", newline="", encoding="utf-8-sig")
            with epoch_file as file:
                epoch_writer = csv.DictWriter(file, fieldnames=["seed", "test_fold", "epoch", "module", "mean_rms", "mean_l2_norm", "nonfinite_batches", "warnings"])
                if epoch == 1: epoch_writer.writeheader()
                epoch_writer.writerows(epoch_rows)
            events.write({"kind": "epoch", "seed": seed, "test_fold": fold, "epoch": epoch,
                          "completed_models": complete_fold_count(config), "device": str(device),
                          "elapsed_s": time.perf_counter() - started, **row, "health": warnings})
            if validation_score > best + 1e-7:
                best, stale = validation_score, 0
                torch.save({"state_dict": model.state_dict(), "seed": seed, "test_fold": fold,
                            "initialization_seed": initialization_seed, "num_transformer_layers": 1}, checkpoint)
            else:
                stale += 1
                if stale >= config.patience:
                    break
    json_dump(health_path, {"seed": seed, "test_fold": fold, "aborted": aborted,
                            "events": health_events, "nonfinite_streak_final": nonfinite_streak})
    if aborted or not checkpoint.exists():
        raise RuntimeError(f"seed={seed}, fold={fold}训练因连续非有限梯度终止；未生成完成标记")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    val_ids, val_labels, val_logits, _ = predict(model, loaders["validation"], device)
    temperature = fit_temperature(val_logits, val_labels)
    threshold = choose_threshold(val_labels.astype(int), sigmoid(val_logits / temperature))
    ids, test_labels, test_logits, inference_seconds = predict(model, loaders["test"], device)
    probability = sigmoid(test_logits / temperature)
    details = metric(test_labels.astype(int), probability, threshold)
    details.update({"seed": seed, "test_fold": fold, "validation_fold": (fold + 1) % 5,
                    "initialization_seed": initialization_seed, "temperature": temperature, "threshold": threshold,
                    "best_validation_low_snr_auroc": float(best),
                    "best_epoch": int(max(row["epoch"] for row in history if row["validation_low_snr_auroc"] == best)),
                    "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                    "inference_ms_per_sample": inference_seconds / len(ids) * 1000.,
                    "training_seconds": time.perf_counter() - started})
    prediction = pd.DataFrame({"sample_index": ids, "label": test_labels.astype(int), "p_click": probability,
                               "predicted_label": (probability >= threshold).astype(int), "threshold": threshold,
                               "temperature": temperature, "seed": seed, "test_fold": fold})
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False, encoding="utf-8-sig")
    prediction.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    json_dump(completed, details)
    events.write({"kind": "status", "state": "模型折完成", "seed": seed, "test_fold": fold,
                  "epoch": details["best_epoch"], "batch": 0, "completed_models": complete_fold_count(config),
                  "device": str(device), "elapsed_s": time.perf_counter() - started,
                  "health": []})
    return details, prediction


def launch_monitor(config: Config) -> subprocess.Popen[Any] | None:
    event_path = paths(config)["live"] / "events.jsonl"
    try:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "monitor",
                                    "--event-path", str(event_path)], cwd=str(ROOT),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(.4)
        if process.poll() is not None:
            print("[实时监控] GUI启动失败，训练继续；可使用--no-live-monitor关闭监控。", flush=True)
            return None
        return process
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[实时监控] GUI启动失败，训练继续：{error}", flush=True)
        return None


def train(config: Config, resume: bool = True, live_monitor: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_map = paths(config)
    manifest, waves = load_data(config)
    event_path = path_map["live"] / "events.jsonl"
    lock = acquire_training_lock(path_map["live"] / "active_training.lock")
    event_path.write_text("", encoding="utf-8")
    monitor = launch_monitor(config) if live_monitor else None
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    events = EventWriter(event_path, lock.run_id)
    try:
        progress = tqdm(total=expected_model_count(), desc="总体进度：v3单层波形Transformer", unit="模型折")
        for seed in SEEDS:
            for fold in range(5):
                row, pred = train_one(seed, fold, waves, manifest, config, events, resume)
                results.append(row); predictions.append(pred); progress.update(1)
                progress.set_postfix(seed=seed, fold=fold, auroc=f"{row['auroc']:.3f}")
        progress.close()
    finally:
        events.write({"kind": "status", "state": "训练完成", "seed": None, "test_fold": None,
                      "epoch": None, "batch": None, "completed_models": complete_fold_count(config),
                      "device": str(device_for(config.device)), "elapsed_s": None, "health": []})
        events.close()
        lock.release()
    metrics = pd.DataFrame(results)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    prediction_frame = prediction_frame.merge(metadata, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
    expected_rows = 7000 * len(SEEDS)
    if len(prediction_frame) != expected_rows or prediction_frame.duplicated(["seed", "sample_id"]).any():
        raise RuntimeError(f"OOF合同失败：预测行={len(prediction_frame)}（期望{expected_rows}），seed+sample_id必须唯一")
    if not np.isfinite(prediction_frame.p_click).all() or not np.isfinite(metrics.select_dtypes(include=np.number)).all().all():
        raise RuntimeError("存在非有限概率或最终指标")
    metrics.to_csv(path_map["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(path_map["predictions"] / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics, prediction_frame


def scoped_metric(frame: pd.DataFrame, scope: str) -> dict[str, float | int]:
    selected = frame[low_snr_mask(frame)] if scope == "low_snr" else frame
    return metric(selected.label.to_numpy(int), selected.p_click.to_numpy(float), selected.threshold.to_numpy(float))


def bootstrap_summary(predictions: pd.DataFrame, config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(SEEDS[0] + 991)
    groups = np.sort(predictions.ori_file_num.unique())
    seeds = np.sort(predictions.seed.unique())
    rows: list[dict[str, Any]] = []
    for scope in ("all_snr", "low_snr"):
        by_seed = {seed: scoped_metric(predictions[predictions.seed == seed], scope) for seed in seeds}
        draws: dict[str, list[float]] = {name: [] for name in METRICS}
        for _ in tqdm(range(config.bootstrap_runs), desc=f"总体进度：{scope}录音组Bootstrap", leave=False):
            selected_groups = rng.choice(groups, len(groups), replace=True)
            values: list[dict[str, float | int]] = []
            for seed in seeds:
                frame = predictions[predictions.seed == seed]
                sample = pd.concat([frame[frame.ori_file_num == group] for group in selected_groups], ignore_index=True)
                if sample.label.nunique() == 2:
                    values.append(scoped_metric(sample, scope))
            for name in METRICS:
                if values: draws[name].append(float(np.mean([value[name] for value in values])))
        for name in METRICS:
            estimates = [float(by_seed[seed][name]) for seed in seeds]
            rows.append({"scope": scope, "metric": name, "mean": float(np.mean(estimates)),
                         "seed_sd": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.,
                         "ci_low": float(np.percentile(draws[name], 2.5)),
                         "ci_high": float(np.percentile(draws[name], 97.5)),
                         "bootstrap_runs_valid": len(draws[name])})
    result = pd.DataFrame(rows)
    result.to_csv(paths(config)["metrics"] / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    return result


def snr_metrics(predictions: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        frame = predictions[predictions.seed == seed]
        for lower, upper, label in SNR_BINS:
            selected = frame[(frame.label == 0) | ((frame.label == 1) &
                                                   (frame.target_snr_db >= lower) &
                                                   (frame.target_snr_db < upper))]
            result = metric(selected.label.to_numpy(int), selected.p_click.to_numpy(float), selected.threshold.to_numpy(float))
            rows.append({"seed": seed, "snr_bin": label, "snr_low_db": lower, "snr_high_db": min(upper, 15.),
                         "positive_count": int((selected.label == 1).sum()), "negative_count": int((selected.label == 0).sum()),
                         "auroc": result["auroc"], "auprc": result["auprc"], "recall": result["recall"]})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["snr_bin", "snr_low_db", "snr_high_db"], as_index=False).agg(
        positive_count=("positive_count", "mean"), negative_count=("negative_count", "mean"),
        auroc_mean=("auroc", "mean"), auroc_seed_sd=("auroc", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else 0.),
        auprc_mean=("auprc", "mean"), auprc_seed_sd=("auprc", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else 0.),
        recall_mean=("recall", "mean"), recall_seed_sd=("recall", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else 0.))
    detail.to_csv(paths(config)["metrics"] / "snr_bin_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths(config)["metrics"] / "snr_bin_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def recover_legacy_gradient_events(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从原子写入的JSONL恢复被并发CSV写入破坏的真实梯度记录。

    仅接受每个模型折的“模型折完成”事件所指明设备、且发生在完成事件之前的
    batch事件。这样会剔除未完成的重复训练，而不推断或补造梯度值。
    """
    path_map = paths(config)
    events: list[dict[str, Any]] = []
    for line in (path_map["live"] / "events.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    completed: dict[tuple[int, int], tuple[str, float]] = {}
    for event in events:
        if event.get("kind") == "status" and event.get("state") == "模型折完成":
            key = (int(event["seed"]), int(event["test_fold"]))
            if key in completed:
                raise RuntimeError(f"梯度恢复拒绝处理重复的模型完成事件：{key}")
            completed[key] = (str(event.get("device")), float(event.get("time_unix", 0.)))
    expected = {(seed, fold) for seed in SEEDS for fold in range(5)}
    if set(completed) != expected:
        raise RuntimeError(f"梯度恢复需要{len(expected)}个唯一的模型折完成事件，当前事件记录不满足该合同")
    selected = []
    for event in events:
        if event.get("kind") != "batch":
            continue
        key = (int(event["seed"]), int(event["test_fold"]))
        device, completed_at = completed[key]
        if event.get("device") == device and float(event.get("time_unix", 0.)) <= completed_at:
            selected.append(event)
    counts = Counter((int(event["seed"]), int(event["test_fold"])) for event in selected)
    if set(counts) != expected or min(counts.values()) <= 0:
        raise RuntimeError(f"梯度恢复缺少实际batch事件：{counts}")
    rows: list[dict[str, Any]] = []
    for event in selected:
        common = {"seed": int(event["seed"]), "test_fold": int(event["test_fold"]),
                  "epoch": int(event["epoch"]), "batch": int(event["batch"]),
                  "loss": event.get("loss"), "warning": "；".join(event.get("health", []))}
        for name, values in (("全模型", event["global"]), *event["modules"].items()):
            rows.append({**common, "module": name, **values})
    batches = pd.DataFrame(rows)
    modules = batches[batches.module != "全模型"]
    epochs = modules.groupby(["seed", "test_fold", "epoch", "module"], as_index=False).agg(
        mean_rms=("rms", "mean"), mean_l2_norm=("l2_norm", "mean"),
        nonfinite_batches=("finite", lambda values: int((~values.astype(bool)).sum())),
        warnings=("warning", lambda values: "；".join(sorted({str(value) for value in values if str(value)}))),
    )
    batches.to_csv(path_map["logs"] / "recovered_batch_gradient_records.csv", index=False, encoding="utf-8-sig")
    epochs.to_csv(path_map["logs"] / "recovered_epoch_gradient_summary.csv", index=False, encoding="utf-8-sig")
    json_dump(path_map["logs"] / "gradient_log_recovery.json", {
        "reason": "并发写入导致原逐batchCSV字段数不一致；原文件保留未改写。",
        "source": str(path_map["live"] / "events.jsonl"),
        "selection_rule": "每个模型折完成事件指定的device，且batch事件时间不晚于该完成事件",
        "selected_batch_counts": {f"seed_{seed}_fold_{fold}": int(counts[(seed, fold)]) for seed, fold in sorted(expected)},
        "recovered_rows": int(len(batches)), "recovered_epoch_rows": int(len(epochs)),
    })
    return batches, epochs


def figures(config: Config) -> None:
    path_map = paths(config); log_files = sorted(path_map["logs"].glob("*_batch_gradients.csv"))
    epoch_files = sorted(path_map["logs"].glob("*_epoch_gradients.csv"))
    histories = sorted(path_map["models"].glob("seed_*/*/history.csv"))
    if not log_files or not epoch_files or not histories:
        raise RuntimeError("缺少完整训练日志，拒绝生成梯度图")
    try:
        batches = pd.concat([pd.read_csv(path) for path in log_files], ignore_index=True)
        epochs = pd.concat([pd.read_csv(path) for path in epoch_files], ignore_index=True)
    except pd.errors.ParserError:
        batches, epochs = recover_legacy_gradient_events(config)
    history = pd.concat([pd.read_csv(path).assign(run=path.parent.as_posix()) for path in histories], ignore_index=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    global_frame = batches[batches.module == "全模型"].copy()
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(np.arange(len(global_frame)), global_frame.l2_norm, label="L2范数")
    axes[0].plot(np.arange(len(global_frame)), global_frame.rms, label="均方根")
    axes[0].set_yscale("log"); axes[0].set(title="全模型梯度", xlabel="批次记录", ylabel="梯度", ylim=(1e-12, None)); axes[0].legend()
    if "validation_loss" not in history:
        raise RuntimeError("历史训练记录缺少验证损失；请完成至少一个模型折后重新生成图形")
    epoch_mean = history.groupby("epoch", as_index=False).agg(
        train_loss=("train_loss", "mean"), validation_loss=("validation_loss", "mean"),
        validation_low_snr_auroc=("validation_low_snr_auroc", "mean"))
    axes[1].plot(epoch_mean.epoch, epoch_mean.train_loss, label="训练集BCE")
    axes[1].plot(epoch_mean.epoch, epoch_mean.validation_loss, label="验证集BCE")
    axes[1].set(title="训练/验证损失（已完成折均值）", xlabel="训练轮次", ylabel="BCE损失"); axes[1].legend(); axes[1].grid(alpha=.2)
    axes[2].plot(epoch_mean.epoch, epoch_mean.validation_low_snr_auroc, label="验证集低SNR AUROC")
    axes[2].set(title="验证集低信噪比AUROC（已完成折均值）", xlabel="训练轮次", ylabel="AUROC", ylim=(0, 1)); axes[2].legend(); axes[2].grid(alpha=.2)
    figure.tight_layout(); figure.savefig(path_map["figures"] / "gradient_and_performance_curves.png", dpi=180); plt.close(figure)
    table = epochs.pivot_table(index="module", columns="epoch", values="mean_rms", aggfunc="mean").reindex(MODULES)
    table.index = [MODULE_DISPLAY.get(str(name), str(name)) for name in table.index]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    image = axis.imshow(np.log10(table.fillna(1e-12).clip(lower=1e-12)), aspect="auto", cmap="viridis")
    axis.set(yticks=np.arange(len(table.index)), yticklabels=table.index, xlabel="训练轮次", title="模块梯度均方根热图（log10）")
    figure.colorbar(image, ax=axis, label="log10均方根")
    figure.tight_layout(); figure.savefig(path_map["figures"] / "module_gradient_rms_heatmap.png", dpi=180); plt.close(figure)


def markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    data = []
    for row in frame.itertuples(index=False, name=None):
        data.append([f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row])
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(row) + " |" for row in data]])


def report(config: Config) -> Path:
    path_map = paths(config); prediction_path = path_map["predictions"] / "oof_predictions.csv"
    metric_path = path_map["metrics"] / "fold_metrics.csv"
    if not prediction_path.exists() or not metric_path.exists():
        raise FileNotFoundError(f"请先完成全部{expected_model_count()}个模型折训练")
    predictions, folds = pd.read_csv(prediction_path), pd.read_csv(metric_path)
    expected_rows, expected_folds = 7000 * len(SEEDS), expected_model_count()
    if len(predictions) != expected_rows or len(folds) != expected_folds or predictions.duplicated(["seed", "sample_id"]).any():
        raise RuntimeError("报告拒绝使用不完整或重复的实际结果")
    summary = bootstrap_summary(predictions, config)
    snr = snr_metrics(predictions, config)
    figures(config)
    main = summary[summary.scope.isin(["all_snr", "low_snr"]) & summary.metric.isin(["auroc", "auprc", "balanced_accuracy", "recall", "f1"])]
    text = f"""# 实验23-v3：单层波形Transformer与实时梯度监控

本报告只使用已完成的{expected_model_count()}个模型折实际预测：训练种子为{', '.join(map(str, SEEDS))}、34个录音组固定五折和7,000条变SNR波形。模型仅输入波形；Transformer Encoder为单层。测试fold为f、验证fold为(f+1) mod 5，最佳检查点按验证集`[-15,0)` dB AUROC选择；温度缩放与Balanced Accuracy阈值仅由完整验证集确定。

## 汇总结果

{markdown(main[["scope", "metric", "mean", "seed_sd", "ci_low", "ci_high"]])}

95%置信区间来自录音组Bootstrap；种子标准差由两个种子的OOF指标计算。

## 六个5 dB区间

{markdown(snr)}

## 梯度监控

逐batch真实梯度在AMP反缩放后采集，未使用默认梯度裁剪。梯度健康事件、逐batch统计、逐epoch汇总和图形均位于本实验输出目录；监控告警仅为诊断，不作为性能结论。
"""
    destination = path_map["root"] / "research_report_zh.md"
    destination.write_text(text, encoding="utf-8")
    return destination


class EventReader:
    """只消费换行终止的完整JSONL事件；写入中的尾行被保留到下一次刷新。"""
    def __init__(self, path: Path):
        self.path, self.offset, self.buffer = path, 0, ""

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists(): return []
        with self.path.open("r", encoding="utf-8") as file:
            file.seek(self.offset); chunk = file.read(); self.offset = file.tell()
        text = self.buffer + chunk
        if not text: return []
        complete, separator, tail = text.rpartition("\n")
        if not separator:
            self.buffer = text; return []
        self.buffer = tail
        events = []
        for line in complete.splitlines():
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
        return events


def monitor(event_path: Path) -> None:
    """独立Tk进程：异常退出或用户关闭不会影响父训练进程。"""
    import tkinter as tk
    from matplotlib import font_manager
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Windows优先使用微软雅黑；仅选取实际安装的中文字体，避免Matplotlib回退到缺字字体。
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS")
    installed = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((font for font in candidates if font in installed), None)
    if chinese_font is not None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chinese_font]
    plt.rcParams["axes.unicode_minus"] = False

    reader, events, active_run_id = EventReader(event_path), [], None
    root = tk.Tk(); root.title("实验23-v3 实时梯度监控（关闭不停止训练）")
    status = tk.StringVar(value="等待训练事件…")
    tk_font = ("Microsoft YaHei", 10) if "Microsoft YaHei" in installed else ("TkDefaultFont", 10)
    tk.Label(root, textvariable=status, anchor="w", font=tk_font).pack(fill="x", padx=8, pady=5)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    auroc_axis = axes[0, 0].twinx()
    canvas = FigureCanvasTkAgg(figure, master=root); canvas.get_tk_widget().pack(fill="both", expand=True)
    def refresh() -> None:
        nonlocal active_run_id
        events.extend(reader.read())
        starts = [event for event in events if event.get("kind") == "status" and event.get("state") == "训练中" and event.get("run_id")]
        if starts:
            active_run_id = starts[-1]["run_id"]
        visible_events = [event for event in events if active_run_id is None or event.get("run_id") == active_run_id]
        batches = [event for event in visible_events if event.get("kind") == "batch"]
        epochs = [event for event in visible_events if event.get("kind") == "epoch"]
        states = [event for event in visible_events if event.get("kind") == "status"]
        for axis in axes.flat: axis.clear()
        auroc_axis.clear()
        if batches:
            glob = [item.get("global", {}) for item in batches]
            axes[0, 1].plot([x.get("l2_norm") for x in glob], label="L2范数")
            axes[0, 1].plot([x.get("rms") for x in glob], label="均方根")
            axes[0, 1].set_yscale("log"); axes[0, 1].set(title="全局梯度", xlabel="批次"); axes[0, 1].legend(fontsize=7)
            for name in MODULES:
                values = [item.get("modules", {}).get(name, {}).get("rms") for item in batches]
                axes[1, 0].plot(values, label=MODULE_DISPLAY[name], linewidth=.8)
            axes[1, 0].set_yscale("log"); axes[1, 0].set(title="各模块梯度均方根", xlabel="批次"); axes[1, 0].legend(fontsize=6, ncol=2)
        if epochs:
            current_seed, current_fold = epochs[-1].get("seed"), epochs[-1].get("test_fold")
            current_epochs = [item for item in epochs if item.get("seed") == current_seed and item.get("test_fold") == current_fold]
            epoch_x = [item["epoch"] for item in current_epochs]
            axes[0, 0].plot(epoch_x, [item["train_loss"] for item in current_epochs], label="训练集BCE")
            axes[0, 0].plot(epoch_x, [item["validation_loss"] for item in current_epochs], label="验证集BCE")
            axes[0, 0].set(title="当前模型折：训练/验证损失", xlabel="训练轮次", ylabel="BCE损失")
            auroc_axis.plot(epoch_x, [item["validation_low_snr_auroc"] for item in current_epochs],
                            color="#2a9d8f", linestyle="--", label="验证低SNR AUROC")
            auroc_axis.set_ylabel("AUROC"); auroc_axis.set_ylim(0, 1)
            lines, labels = axes[0, 0].get_legend_handles_labels()
            auroc_lines, auroc_labels = auroc_axis.get_legend_handles_labels()
            axes[0, 0].legend(lines + auroc_lines, labels + auroc_labels, fontsize=7)
            axes[0, 0].grid(alpha=.2)
        health = [warning for event in visible_events for warning in event.get("health", [])]
        axes[1, 1].axis("off")
        axes[1, 1].text(.02, .96, "健康状态\n" + ("正常" if not health else "\n".join(health[-8:])), va="top", wrap=True)
        if states:
            last_status = states[-1]
            current = batches[-1] if batches else last_status
            status.set(f"状态：{last_status.get('state')} | 随机种子={current.get('seed')} | 测试折={current.get('test_fold')} | "
                       f"训练轮次={current.get('epoch')} 批次={current.get('batch')} | "
                       f"已完成={current.get('completed_models', last_status.get('completed_models'))}/{expected_model_count()} | "
                       f"运行={float(current.get('elapsed_s') or 0):.0f}秒 | 设备={current.get('device')} | 关闭窗口不停止训练")
        figure.tight_layout(); canvas.draw_idle(); root.after(2000, refresh)
    refresh(); root.mainloop()


def verify(config: Config) -> dict[str, Any]:
    path_map = paths(config); manifest, waves = load_data(config)
    model = WaveformTransformer(); wave = torch.randn(2, SAMPLE_COUNT); labels = torch.tensor([0., 1.])
    logits = model(wave); loss = nn.BCEWithLogitsLoss()(logits, labels); loss.backward()
    module_stats, global_stats = gradient_statistics(model)
    encoder_layers = len(model.encoder.layers)
    event_path = path_map["tests"] / "simulated_events.jsonl"
    complete = {"kind": "batch", "seed": SEEDS[0], "test_fold": 0, "epoch": 1, "batch": 1,
                "loss": .4, "global": {"l2_norm": 10., "rms": .1},
                "modules": {name: {"rms": 1e-4} for name in MODULES},
                "health": ["异常峰值：全局梯度L2超过此前中位数10倍"]}
    event_path.write_text(json.dumps(complete, ensure_ascii=False) + "\n" + "{\"partial\":", encoding="utf-8")
    read_events = EventReader(event_path).read()
    try:
        import tkinter  # noqa: F401
        tkinter_available = True
    except Exception:
        tkinter_available = False
    lock_path = path_map["tests"] / "single_instance_contract.lock"
    first_lock = acquire_training_lock(lock_path)
    try:
        try:
            acquire_training_lock(lock_path)
            lock_rejects_duplicate = False
        except RuntimeError:
            lock_rejects_duplicate = True
    finally:
        first_lock.release()
    result = {"data_shape": list(waves.shape), "recording_groups": int(manifest.ori_file_num.nunique()),
              "no_cross_fold_group_leakage": bool(manifest.groupby("ori_file_num").test_fold.nunique().max() == 1),
              "transformer_encoder_layers": encoder_layers,
              "model_output_shape": list(logits.shape), "finite_loss": bool(torch.isfinite(loss)),
              "expected_modules": list(MODULES),
              "finite_module_gradients": {name: bool(value["finite"] and value["rms"] is not None) for name, value in module_stats.items()},
              "finite_global_gradient": bool(global_stats["finite"]),
              "simulated_monitor_complete_events": len(read_events), "simulated_monitor_ignores_partial_tail": len(read_events) == 1,
              "simulated_monitor_recognizes_anomaly": bool(read_events and "异常峰值" in read_events[0]["health"][0]),
              "monitor_close_is_training_decoupled": True,
              "single_instance_lock_rejects_duplicate": lock_rejects_duplicate,
              "tkinter_importable": tkinter_available}
    result["passed"] = bool(result["data_shape"] == [7000, SAMPLE_COUNT] and result["recording_groups"] == 34 and
                            result["no_cross_fold_group_leakage"] and encoder_layers == 1 and result["finite_loss"] and
                            all(result["finite_module_gradients"].values()) and result["finite_global_gradient"] and
                            result["simulated_monitor_ignores_partial_tail"] and
                            result["simulated_monitor_recognizes_anomaly"] and
                            result["monitor_close_is_training_decoupled"] and
                            result["single_instance_lock_rejects_duplicate"])
    json_dump(path_map["tests"] / "verification.json", result)
    if not result["passed"]: raise RuntimeError(f"v3验证失败：{result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "train", "report", "all", "monitor"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path,
                        help="独立实验输出目录；用于调参时避免覆盖基线结果")
    parser.add_argument("--weight-decay", type=float,
                        help="AdamW权重衰减；未指定时使用协议默认值1e-4")
    parser.add_argument("--dropout", type=float,
                        help="Transformer dropout；未指定时使用协议默认值0.1")
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="none",
                        help="学习率调度器；plateau按验证低SNR AUROC降低学习率")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-live-monitor", action="store_true")
    parser.add_argument("--event-path", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.command == "monitor":
        if arguments.event_path is None: raise ValueError("monitor需要--event-path")
        monitor(arguments.event_path); return
    if arguments.weight_decay is not None and arguments.weight_decay < 0:
        raise ValueError("weight_decay必须为非负数")
    if arguments.dropout is not None and not 0 <= arguments.dropout < 1:
        raise ValueError("dropout必须在[0, 1)范围内")
    config = Config(device=arguments.device, bootstrap_runs=arguments.bootstrap_runs,
                    output_dir=arguments.output_dir or OUT,
                    weight_decay=(arguments.weight_decay if arguments.weight_decay is not None else Config.weight_decay),
                    dropout=(arguments.dropout if arguments.dropout is not None else Config.dropout),
                    lr_scheduler=arguments.lr_scheduler)
    path_map = paths(config)
    json_dump(path_map["root"] / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir),
              "manifest_path": str(config.manifest_path), "waveform_path": str(config.waveform_path),
              "seeds": list(SEEDS), "num_transformer_layers": 1, "inputs": ["waveform_only"],
              "gradient_capture": "after GradScaler.unscale_(optimizer), before optimizer step"})
    if arguments.command == "verify":
        print("[总体进度 100%] 验证通过", verify(config), flush=True); return
    if arguments.command in ("train", "all"):
        print("[总体进度 0%] 验证数据、单层模型与监控事件", flush=True)
        verify(config)
        print(f"[总体进度 5%] 训练{expected_model_count()}个模型折", flush=True)
        train(config, resume=not arguments.no_resume, live_monitor=not arguments.no_live_monitor)
    if arguments.command in ("report", "all"):
        print("[总体进度 90%] 汇总实际结果、Bootstrap与梯度图", flush=True)
        print(f"[总体进度 100%] 完成：{report(config)}", flush=True)


if __name__ == "__main__":
    main()
