#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23：BCSC多径结构分支辅助海豚click识别。

本脚本只读取实验19已锁定的7000条清单及预处理波形；所有新产物均写入本目录。
命令：all / extract-structure / train / report / predict --wav <绝对路径>。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
EXPERIMENT_DIR = PROJECT_ROOT / "23_BCSCStructureClickClassification"
REFERENCE_DIR = PROJECT_ROOT / "19_20260813" / "outputs"
REFERENCE_PIPELINE = PROJECT_ROOT / "19_20260813" / "bcsc_transformer_pipeline.py"
SAMPLE_RATE, SAMPLE_COUNT = 576_000, 5_761
MIN_DELAY, MAX_DELAY = 20, 1_152
PATH_BINS = MAX_DELAY - MIN_DELAY + 1
SEEDS = (20260912, 20260913, 20260914)
NEURAL_CONDITIONS = ("waveform", "structure_only", "waveform_structure", "waveform_zero_structure")
CLASSIC_CONDITIONS = ("teager", "bcsc_summary")
CONDITIONS = CLASSIC_CONDITIONS + NEURAL_CONDITIONS
CONDITION_LABELS = {
    "teager": "Teager基线", "bcsc_summary": "BCSC-Summary",
    "waveform": "Waveform Transformer", "structure_only": "Structure-only Path-CNN",
    "waveform_structure": "Waveform + Structure", "waveform_zero_structure": "Waveform + Zero-Structure",
}
METRICS = ("auroc", "auprc", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "brier")


@dataclass(frozen=True)
class Config:
    output_dir: Path = EXPERIMENT_DIR / "outputs"
    reference_dir: Path = REFERENCE_DIR
    workers: int = max(1, min(6, os.cpu_count() or 1))
    max_epochs: int = 25
    patience: int = 5
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bootstrap_runs: int = 2_000
    device: str = "auto"


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def paths(config: Config) -> dict[str, Path]:
    output = config.output_dir
    result = {"root": output, "cache": output / "cache", "models": output / "models",
              "metrics": output / "metrics", "predictions": output / "predictions",
              "figures": output / "figures", "logs": output / "logs", "tests": output / "tests"}
    for item in result.values():
        item.mkdir(parents=True, exist_ok=True)
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA但当前PyTorch无法使用CUDA")
    return device


def load_legacy() -> Any:
    """显式加载实验19的原始BCSC实现，避免复制后发生配置漂移。"""
    if not REFERENCE_PIPELINE.exists():
        raise FileNotFoundError(f"缺少实验19 BCSC实现：{REFERENCE_PIPELINE}")
    module_name = "experiment19_bcsc_for_experiment23"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{REFERENCE_PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def bcsc_config() -> Any:
    legacy = load_legacy()
    # 85微秒在576 kHz下被实验19转换为49点；其余BCSC超参数保持原值。
    config = legacy.Config()
    if config.atom_length != 49 or config.outer_iterations != 35 or not math.isclose(config.lambda_ratio, 0.075):
        raise RuntimeError("实验19 BCSC配置不符合实验23锁定合同")
    return config


def load_reference_data(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    manifest_path = config.reference_dir / "dataset_manifest.csv"
    waveform_path = config.reference_dir / "cache" / "preprocessed_waveforms.npy"
    if not manifest_path.exists() or not waveform_path.exists():
        raise FileNotFoundError("实验19的固定清单或预处理波形缓存缺失")
    manifest = pd.read_csv(manifest_path)
    waveforms = np.load(waveform_path, mmap_mode="r")
    if len(manifest) != 7_000 or tuple(waveforms.shape) != (7_000, SAMPLE_COUNT):
        raise RuntimeError(f"实验19输入合同错误：manifest={len(manifest)}，waveforms={waveforms.shape}")
    if manifest["label"].value_counts().to_dict() != {1: 3500, 0: 3500}:
        raise RuntimeError("实验19清单类别计数错误")
    if manifest["ori_file_num"].nunique() != 34 or manifest.groupby("ori_file_num")["test_fold"].nunique().max() != 1:
        raise RuntimeError("录音组数量或折分配不符合固定五折合同")
    return manifest, waveforms


def _match_bootstrap_paths(reference: list[int], boot: list[int], code: np.ndarray, tolerance: int) -> dict[int, list[tuple[int, float]]]:
    """每个Bootstrap重复内，对所有参考候选作一对一最近邻匹配。"""
    matches: dict[int, list[tuple[int, float]]] = {item: [] for item in reference}
    unused = set(boot)
    for item in reference:
        available = [candidate for candidate in unused if abs(candidate - item) <= tolerance]
        if available:
            chosen = min(available, key=lambda candidate: (abs(candidate - item), -abs(float(code[candidate]))))
            unused.remove(chosen)
            matches[item].append((chosen, float(code[chosen])))
    return matches


def extract_structure_one(signal: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """将BCSC全候选延迟路径编码为(P,A,U)三通道结构图。"""
    legacy, cfg = load_legacy(), bcsc_config()
    started = time.perf_counter()
    structure = np.zeros((3, PATH_BINS), dtype=np.float32)
    try:
        atom, activation, estimate = legacy.bcsc(np.asarray(signal, dtype=np.float64), cfg)
        direct = int(np.argmax(np.abs(activation)))
        candidates = legacy.extract_candidate_indices(activation, cfg)
        references = [int(index) for index in candidates if MIN_DELAY <= int(index) - direct <= MAX_DELAY]
        residual = np.asarray(signal, dtype=np.float64) - estimate
        rng = np.random.default_rng(seed)
        block_length = int(round(0.10e-3 * SAMPLE_RATE))
        tolerance = int(round(0.03e-3 * SAMPLE_RATE))
        scale = cfg.lambda_ratio * np.max(np.abs(legacy.adjoint_atom(signal, atom, len(activation))))
        records: dict[int, list[tuple[int, float]]] = {item: [] for item in references}
        for _ in range(8):
            synthetic = estimate + legacy.moving_block_bootstrap(residual, block_length, rng)
            lam = scale * rng.uniform(0.85, 1.15)
            code = legacy.fista_sparse_code(synthetic, atom, lam, activation, cfg.fista_iterations)
            match = _match_bootstrap_paths(references, list(map(int, legacy.extract_candidate_indices(code, cfg))), code, tolerance)
            for item in references:
                records[item].extend(match[item])
        direct_abs = abs(float(activation[direct])) + 1e-12
        nonzero_paths = 0
        for item in references:
            found = records[item]
            # 用户确认：若8次均未检出，该格点三个通道均为零。
            if not found:
                continue
            bin_index = item - direct - MIN_DELAY
            indices = np.asarray([pair[0] for pair in found], dtype=float)
            coefficients = np.asarray([pair[1] for pair in found], dtype=float)
            persistence = len(found) / 8.0
            amplitude = math.log1p(float(np.median(np.abs(coefficients))) / direct_abs)
            uncertainty = 1.0 if len(found) == 1 else float(np.clip((np.percentile(indices, 97.5) - np.percentile(indices, 2.5)) / SAMPLE_RATE * 1e3 / 0.06, 0.0, 1.0))
            # 一个延迟格仅可能来自一个初始候选；仍使用max保证数值健壮性。
            structure[0, bin_index] = max(structure[0, bin_index], persistence)
            structure[1, bin_index] = max(structure[1, bin_index], amplitude)
            structure[2, bin_index] = max(structure[2, bin_index], uncertainty)
            nonzero_paths += 1
        return structure, {"status": "ok", "error": "", "direct_index": direct,
                           "candidate_path_count": len(references), "detected_path_count": nonzero_paths,
                           "elapsed_s": time.perf_counter() - started}
    except Exception as error:
        return structure, {"status": "failed", "error": f"{type(error).__name__}: {error}", "direct_index": -1,
                           "candidate_path_count": 0, "detected_path_count": 0, "elapsed_s": time.perf_counter() - started}


def _extract_worker(payload: tuple[int, np.ndarray, int]) -> tuple[int, np.ndarray, dict[str, Any]]:
    index, signal, seed = payload
    structure, metadata = extract_structure_one(signal, seed)
    return index, structure, metadata


def extract_structure(config: Config, force: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    output = paths(config)
    manifest, waveforms = load_reference_data(config)
    map_path, done_path, status_path = output["cache"] / "structure_maps.npy", output["cache"] / "structure_completed.npy", output["cache"] / "structure_status.csv"
    if force:
        for path in (map_path, done_path, status_path):
            if path.exists(): path.unlink()
    if map_path.exists() and done_path.exists():
        maps = np.lib.format.open_memmap(map_path, mode="r+", dtype=np.float32, shape=(7000, 3, PATH_BINS))
        completed = np.load(done_path)
        if tuple(maps.shape) != (7000, 3, PATH_BINS) or completed.shape != (7000,):
            raise RuntimeError("既有结构缓存尺寸不符合7000×3×1133合同")
    else:
        maps = np.lib.format.open_memmap(map_path, mode="w+", dtype=np.float32, shape=(7000, 3, PATH_BINS))
        maps[:] = 0
        maps.flush()
        completed = np.zeros(7000, dtype=bool)
        np.save(done_path, completed)
    old_status = pd.read_csv(status_path).set_index("sample_index") if status_path.exists() else pd.DataFrame()
    rows: dict[int, dict[str, Any]] = old_status.to_dict("index") if not old_status.empty else {}
    pending = np.flatnonzero(~completed)
    if len(pending):
        print(f"[总体进度 10%] BCSC结构图：缓存已完成 {completed.sum()}/7000，待处理 {len(pending)} 条", flush=True)
        progress = tqdm(total=len(pending), desc="总体进度：BCSC样本提取", unit="样本")
        payloads = ((int(index), np.asarray(waveforms[index], dtype=np.float32), 20260912 + int(index)) for index in pending)
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for count, (index, structure, metadata) in enumerate(pool.map(_extract_worker, payloads, chunksize=1), start=1):
                maps[index] = structure
                completed[index] = True
                metadata["sample_index"] = index
                rows[index] = metadata
                progress.update(1)
                if count % 25 == 0 or count == len(pending):
                    maps.flush(); np.save(done_path, completed)
                    status = pd.DataFrame.from_dict(rows, orient="index").sort_values("sample_index")
                    status.to_csv(status_path, index=False, encoding="utf-8-sig")
        progress.close()
    maps.flush(); np.save(done_path, completed)
    status = pd.DataFrame.from_dict(rows, orient="index").sort_values("sample_index")
    status = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "label", "ori_file_num", "noise_type", "test_fold"]].merge(status, on="sample_index", how="left")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    if not completed.all() or tuple(maps.shape) != (7000, 3, PATH_BINS):
        raise RuntimeError("BCSC结构缓存未完整写入")
    json_dump(output["cache"] / "structure_cache_contract.json", {"shape": [7000, 3, PATH_BINS], "dtype": "float32", "completed": int(completed.sum()),
        "bcsc": {"atom_points": 49, "outer_iterations": 35, "lambda_ratio": 0.075, "bootstrap_runs": 8, "all_initial_candidates": True}})
    return status, np.load(map_path, mmap_mode="r")


def load_structure(config: Config) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    output = paths(config)
    manifest, _ = load_reference_data(config)
    map_path, done_path, status_path = output["cache"] / "structure_maps.npy", output["cache"] / "structure_completed.npy", output["cache"] / "structure_status.csv"
    if not map_path.exists() or not done_path.exists() or not status_path.exists():
        raise FileNotFoundError("请先运行 extract-structure")
    maps, done = np.load(map_path, mmap_mode="r"), np.load(done_path)
    if maps.shape != (7000, 3, PATH_BINS) or not done.all():
        raise RuntimeError("结构缓存不完整或尺寸错误")
    return manifest, maps, pd.read_csv(status_path)


class ExperimentDataset(Dataset):
    def __init__(self, waves: np.ndarray, structures: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.waves, self.structures, self.labels, self.indices = waves, structures, labels, indices
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32),
                torch.as_tensor(np.asarray(self.structures[index]), dtype=torch.float32),
                torch.tensor(self.labels[index], dtype=torch.float32), index)


class WaveformEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv1d(1, 64, kernel_size=32, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.cls, std=0.02)
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(wave.unsqueeze(1)).transpose(1, 2)
        positions = torch.arange(tokens.shape[1], device=wave.device, dtype=tokens.dtype).unsqueeze(-1)
        dimensions = torch.arange(64, device=wave.device, dtype=tokens.dtype).unsqueeze(0)
        angle = positions / torch.pow(torch.tensor(10000.0, device=wave.device, dtype=tokens.dtype), (2 * torch.floor(dimensions / 2)) / 64)
        positional = torch.where((dimensions.long() % 2) == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        cls = self.cls.expand(wave.shape[0], -1, -1)
        return self.encoder(torch.cat([cls, tokens + positional], dim=1))[:, 0]


class StructureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(3, 16, 7, padding=3), nn.GroupNorm(4, 16), nn.GELU(), nn.MaxPool1d(2),
                                 nn.Conv1d(16, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(), nn.MaxPool1d(2),
                                 nn.Conv1d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.project = nn.Linear(128, 32)
    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        value = self.net(structure)
        return self.project(torch.cat([value.mean(dim=-1), value.max(dim=-1).values], dim=1))


class ClickModel(nn.Module):
    def __init__(self, condition: str) -> None:
        super().__init__()
        self.condition = condition
        self.waveform = WaveformEncoder() if condition != "structure_only" else None
        self.structure = StructureEncoder() if condition != "waveform" else None
        input_size = 32 if condition == "structure_only" else 64 if condition == "waveform" else 96
        self.head = nn.Sequential(nn.Linear(input_size, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, wave: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        if self.condition == "waveform": features = self.waveform(wave)
        elif self.condition == "structure_only": features = self.structure(structure)
        else:
            if self.condition == "waveform_zero_structure": structure = torch.zeros_like(structure)
            features = torch.cat([self.waveform(wave), self.structure(structure)], dim=1)
        return self.head(features).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_loader(waves: np.ndarray, structures: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(ExperimentDataset(waves, structures, labels, indices), batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict_neural(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval(); indices: list[np.ndarray] = []; labels: list[np.ndarray] = []; logits: list[np.ndarray] = []
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for wave, structure, label, index in loader:
        output = model(wave.to(device, non_blocking=True), structure.to(device, non_blocking=True))
        indices.append(np.asarray(index)); labels.append(label.numpy()); logits.append(output.cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return np.concatenate(indices), np.concatenate(labels), np.concatenate(logits), elapsed


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        values = np.logaddexp(0, logits / temperature) - labels * logits / temperature
        return float(np.mean(values))
    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.0


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    values = [balanced_accuracy_score(labels, probability >= candidate) for candidate in candidates]
    return float(candidates[int(np.argmax(values))])


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probability)), "auprc": float(average_precision_score(labels, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)), "specificity": float(tn / max(tn + fp, 1)),
            "f1": float(f1_score(labels, predicted, zero_division=0)), "mcc": float(matthews_corrcoef(labels, predicted)),
            "brier": float(brier_score_loss(labels, probability)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def split_indices(manifest: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = manifest.test_fold.to_numpy()
    validation = (fold + 1) % 5
    return np.flatnonzero((values != fold) & (values != validation)), np.flatnonzero(values == validation), np.flatnonzero(values == fold)


def neural_run_dir(config: Config, condition: str, seed: int, fold: int) -> Path:
    return paths(config)["models"] / condition / f"seed_{seed}" / f"fold_{fold}"


def train_neural_fold(condition: str, seed: int, fold: int, waves: np.ndarray, structures: np.ndarray, manifest: pd.DataFrame, config: Config, resume: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = neural_run_dir(config, condition, seed, fold); run_dir.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint_path, complete_path = run_dir / "test_predictions.csv", run_dir / "best_checkpoint.pt", run_dir / "COMPLETE.json"
    if resume and result_path.exists() and checkpoint_path.exists() and complete_path.exists():
        return json.loads(complete_path.read_text(encoding="utf-8")), pd.read_csv(result_path)
    seed_everything(seed + fold * 100)
    device, labels = resolve_device(config.device), manifest.label.to_numpy(dtype=np.float32)
    train_index, validation_index, test_index = split_indices(manifest, fold)
    loaders = {"train": make_loader(waves, structures, labels, train_index, config.batch_size, True),
               "validation": make_loader(waves, structures, labels, validation_index, config.batch_size * 2),
               "test": make_loader(waves, structures, labels, test_index, config.batch_size * 2)}
    model = ClickModel(condition).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = nn.BCEWithLogitsLoss(); scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, stale, history = -np.inf, 0, []
    began = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train(); losses = []
        for wave, structure, label, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(wave.to(device, non_blocking=True), structure.to(device, non_blocking=True))
                loss = loss_function(logits, label.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        _, val_labels, val_logits, _ = predict_neural(model, loaders["validation"], device)
        score = float(roc_auc_score(val_labels, val_logits)); history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_auroc": score})
        if score > best + 1e-7:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "condition": condition, "seed": seed, "fold": fold}, checkpoint_path)
        else:
            stale += 1
            if stale >= config.patience: break
    package = torch.load(checkpoint_path, map_location=device, weights_only=False); model.load_state_dict(package["state_dict"])
    _, val_labels, val_logits, _ = predict_neural(model, loaders["validation"], device)
    temperature = fit_temperature(val_logits, val_labels); threshold = choose_threshold(val_labels, 1.0 / (1.0 + np.exp(-val_logits / temperature)))
    test_order, test_labels, test_logits, elapsed = predict_neural(model, loaders["test"], device)
    probability = 1.0 / (1.0 + np.exp(-test_logits / temperature))
    fold_metrics = metrics(test_labels.astype(int), probability, threshold)
    fold_metrics.update({"condition": condition, "seed": seed, "test_fold": fold, "validation_fold": (fold + 1) % 5, "temperature": temperature,
                         "threshold": threshold, "best_validation_auroc": best, "best_epoch": max(item["epoch"] for item in history if item["validation_auroc"] == best),
                         "parameters": count_parameters(model), "inference_ms_per_sample": elapsed / len(test_order) * 1e3, "training_seconds": time.perf_counter() - began})
    prediction = pd.DataFrame({"sample_index": test_order, "label": test_labels.astype(int), "logit": test_logits, "p_click": probability,
                               "predicted_label": (probability >= threshold).astype(int), "threshold": threshold, "temperature": temperature,
                               "condition": condition, "seed": seed, "test_fold": fold})
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    prediction.to_csv(result_path, index=False, encoding="utf-8-sig")
    json_dump(complete_path, fold_metrics)
    return fold_metrics, prediction


def teager_scores(waves: np.ndarray) -> np.ndarray:
    values = np.asarray(waves, dtype=np.float64)
    energy = values[:, 1:-1] ** 2 - values[:, :-2] * values[:, 2:]
    peak = np.argmax(energy, axis=1)
    result = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        keep = np.ones(energy.shape[1], dtype=bool); keep[max(0, peak[index] - 32): min(len(keep), peak[index] + 33)] = False
        result[index] = max(float(energy[index, peak[index]]), 0.0) / (float(np.median(np.abs(energy[index, keep]))) + 1e-9)
    return result


def summary_features(structures: np.ndarray) -> np.ndarray:
    persistence, amplitude, uncertainty = np.asarray(structures[:, 0]), np.asarray(structures[:, 1]), np.asarray(structures[:, 2])
    detected = persistence > 0; delays = (np.arange(PATH_BINS) + MIN_DELAY) / SAMPLE_RATE * 1e3
    weights = persistence + 1e-12; weighted_mean = (weights * delays).sum(1) / weights.sum(1)
    weighted_std = np.sqrt((weights * (delays - weighted_mean[:, None]) ** 2).sum(1) / weights.sum(1))
    count = detected.sum(1); support_sum = persistence.sum(1); support_max = persistence.max(1)
    support_mean = np.divide(support_sum, count, out=np.zeros_like(support_sum), where=count > 0)
    uncertainty_mean = np.divide((uncertainty * detected).sum(1), count, out=np.zeros_like(support_sum), where=count > 0)
    return np.column_stack([count, support_sum, support_max, support_mean, amplitude.max(1), weighted_mean, weighted_std, uncertainty_mean]).astype(np.float32)


def train_classic(config: Config, waves: np.ndarray, structures: np.ndarray, manifest: pd.DataFrame, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = paths(config); metrics_path, prediction_path = output["metrics"] / "classic_fold_metrics.csv", output["predictions"] / "classic_oof_predictions.csv"
    if resume and metrics_path.exists() and prediction_path.exists(): return pd.read_csv(metrics_path), pd.read_csv(prediction_path)
    labels = manifest.label.to_numpy(dtype=int); teager = np.log1p(teager_scores(waves)).reshape(-1, 1); summary = summary_features(structures)
    all_metrics, all_predictions = [], []
    progress = tqdm(total=10, desc="总体进度：确定性基线五折训练", unit="模型折")
    for fold in range(5):
        train_index, validation_index, test_index = split_indices(manifest, fold)
        for condition, values in (("teager", teager), ("bcsc_summary", summary)):
            started = time.perf_counter()
            if condition == "teager": model = LogisticRegression(random_state=fold, max_iter=200)
            else: model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=fold)
            model.fit(values[train_index], labels[train_index]); val_prob = model.predict_proba(values[validation_index])[:, 1]
            threshold = choose_threshold(labels[validation_index], val_prob); before = time.perf_counter(); probability = model.predict_proba(values[test_index])[:, 1]; infer = time.perf_counter() - before
            row = metrics(labels[test_index], probability, threshold)
            parameter_count = 0 if condition == "teager" else int(sum(len(predictor.nodes) for stage in model._predictors for predictor in stage))
            row.update({"condition": condition, "seed": "deterministic", "test_fold": fold, "validation_fold": (fold + 1) % 5, "threshold": threshold, "temperature": np.nan,
                        "parameters": parameter_count, "inference_ms_per_sample": infer / len(test_index) * 1e3, "training_seconds": time.perf_counter() - started})
            all_metrics.append(row); all_predictions.append(pd.DataFrame({"sample_index": test_index, "label": labels[test_index], "logit": np.nan, "p_click": probability,
                "predicted_label": (probability >= threshold).astype(int), "threshold": threshold, "temperature": np.nan, "condition": condition, "seed": "deterministic", "test_fold": fold}))
            progress.update(1)
    progress.close(); frame_metrics, frame_predictions = pd.DataFrame(all_metrics), pd.concat(all_predictions, ignore_index=True)
    frame_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig"); frame_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    return frame_metrics, frame_predictions


def train_all(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = paths(config); manifest, waves = load_reference_data(config); _, structures, _ = load_structure(config)
    classic_metrics, classic_predictions = train_classic(config, waves, structures, manifest, resume)
    metrics_path, prediction_path = output["metrics"] / "neural_fold_metrics.csv", output["predictions"] / "neural_oof_predictions.csv"
    if resume and metrics_path.exists() and prediction_path.exists(): neural_metrics, neural_predictions = pd.read_csv(metrics_path), pd.read_csv(prediction_path)
    else:
        records, prediction_records = [], []
        progress = tqdm(total=60, desc="总体进度：60个神经网络模型折", unit="模型折")
        for seed in SEEDS:
            for fold in range(5):
                for condition in NEURAL_CONDITIONS:
                    row, prediction = train_neural_fold(condition, seed, fold, waves, structures, manifest, config, resume)
                    records.append(row); prediction_records.append(prediction); progress.update(1); progress.set_postfix(condition=condition, seed=seed, fold=fold, auroc=f"{row['auroc']:.3f}")
        progress.close(); neural_metrics, neural_predictions = pd.DataFrame(records), pd.concat(prediction_records, ignore_index=True)
        neural_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig"); neural_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    all_metrics = pd.concat([classic_metrics, neural_metrics], ignore_index=True); all_predictions = pd.concat([classic_predictions, neural_predictions], ignore_index=True)
    # CSV round-trips mix deterministic and integer seeds; use one canonical key type.
    all_metrics["seed"] = all_metrics["seed"].astype(str)
    all_predictions["seed"] = all_predictions["seed"].astype(str)
    metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "test_fold"]]
    all_predictions = all_predictions.merge(metadata, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
    if len(all_predictions) != 98_000 or all_predictions.duplicated(["condition", "seed", "sample_id"]).any():
        raise RuntimeError(f"折外预测合同失败：期望98000行，得到{len(all_predictions)}行")
    all_metrics.to_csv(output["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig"); all_predictions.to_csv(output["predictions"] / "oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    return all_metrics, all_predictions


def aggregate_probability(predictions: pd.DataFrame, condition: str) -> pd.DataFrame:
    frame = predictions[predictions.condition == condition]
    return frame.groupby(["sample_id", "label", "ori_file_num", "noise_type"], as_index=False).agg(p_click=("p_click", "mean"), threshold=("threshold", "median"), predicted_label=("predicted_label", "mean"))


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    """无额外tabulate依赖地生成报告所需的紧凑Markdown表。"""
    columns = list(frame.columns)
    def render(value: Any) -> str:
        return f"{value:.{decimals}f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def bootstrap_ci(frame: pd.DataFrame, metric_name: str, rng: np.random.Generator, runs: int) -> tuple[float, float]:
    groups = frame.ori_file_num.unique(); values = []
    for _ in range(runs):
        selected = rng.choice(groups, len(groups), replace=True)
        sample = pd.concat([frame[frame.ori_file_num == group] for group in selected], ignore_index=True)
        if sample.label.nunique() < 2: continue
        values.append(metrics(sample.label.to_numpy(), sample.p_click.to_numpy(), float(sample.threshold.median()))[metric_name])
    return tuple(map(float, np.percentile(values, [2.5, 97.5])))


def paired_bootstrap(predictions: pd.DataFrame, left: str, right: str, rng: np.random.Generator, runs: int) -> tuple[float, float, list[float]]:
    left_frame, right_frame = predictions[predictions.condition == left], predictions[predictions.condition == right]
    seeds = [seed for seed in left_frame.seed.unique() if seed != "deterministic"]
    values = []; groups = left_frame.ori_file_num.unique()
    for _ in tqdm(range(runs), desc=f"录音组配对Bootstrap：{left}-{right}", leave=False):
        selected = rng.choice(groups, len(groups), replace=True); differences = []
        for seed in seeds:
            a, b = left_frame[left_frame.seed == seed], right_frame[right_frame.seed == seed]
            sa = pd.concat([a[a.ori_file_num == group] for group in selected], ignore_index=True); sb = pd.concat([b[b.ori_file_num == group] for group in selected], ignore_index=True)
            differences.append(roc_auc_score(sa.label, sa.p_click) - roc_auc_score(sb.label, sb.p_click))
        values.append(float(np.mean(differences)))
    point = float(np.mean([roc_auc_score(left_frame[left_frame.seed == seed].label, left_frame[left_frame.seed == seed].p_click) - roc_auc_score(right_frame[right_frame.seed == seed].label, right_frame[right_frame.seed == seed].p_click) for seed in seeds]))
    low, high = np.percentile(values, [2.5, 97.5]); return point, float(low), float(high), values


def summarize(config: Config, fold_metrics: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = paths(config); rng = np.random.default_rng(20260913); rows = []
    for condition in CONDITIONS:
        seed_frames = [aggregate_probability(predictions[predictions.seed == str(seed)], condition) for seed in SEEDS] if condition in NEURAL_CONDITIONS else [aggregate_probability(predictions, condition)]
        row: dict[str, Any] = {"condition": condition, "condition_label": CONDITION_LABELS[condition], "parameters": float(fold_metrics[fold_metrics.condition == condition].parameters.max()),
                                "inference_ms_per_sample": float(fold_metrics[fold_metrics.condition == condition].inference_ms_per_sample.mean())}
        for metric_name in METRICS:
            values = [metrics(frame.label, frame.p_click, float(frame.threshold.median()))[metric_name] for frame in seed_frames]
            row[metric_name] = float(np.mean(values)); row[f"{metric_name}_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ensemble = aggregate_probability(predictions, condition); low, high = bootstrap_ci(ensemble, metric_name, rng, config.bootstrap_runs)
            row[f"{metric_name}_ci_low"], row[f"{metric_name}_ci_high"] = low, high
        rows.append(row)
    table = pd.DataFrame(rows); table.to_csv(output["metrics"] / "main_results.csv", index=False, encoding="utf-8-sig")
    paired_rows = []
    for right in ("waveform", "waveform_zero_structure"):
        point, low, high, _ = paired_bootstrap(predictions, "waveform_structure", right, rng, config.bootstrap_runs)
        paired_rows.append({"comparison": f"Waveform + Structure − {CONDITION_LABELS[right]}", "left": "waveform_structure", "right": right, "metric": "AUROC", "difference": point, "ci_low": low, "ci_high": high,
                            "evidence": "支持结构分支增益" if low > 0 else "提升趋势，证据不足" if point > 0 else "当前结构设计未产生有效增益"})
    paired = pd.DataFrame(paired_rows); paired.to_csv(output["metrics"] / "paired_auroc_bootstrap.csv", index=False, encoding="utf-8-sig")
    noise_rows = []
    for condition in CONDITIONS:
        frames = [aggregate_probability(predictions[predictions.seed == str(seed)], condition) for seed in SEEDS] if condition in NEURAL_CONDITIONS else [aggregate_probability(predictions, condition)]
        for noise_type in ("Type_3_Multipath_Noise", "Type_4_Isolated_Noise"):
            result = []
            for frame in frames:
                negative = frame[frame.noise_type == noise_type]; positive = frame[frame.label == 1]; combined = pd.concat([positive, negative])
                threshold = float(frame.threshold.median()); result.append({"specificity": float((negative.p_click < threshold).mean()), "false_positives": int((negative.p_click >= threshold).sum()), "auroc": float(roc_auc_score(combined.label, combined.p_click)), "auprc": float(average_precision_score(combined.label, combined.p_click))})
            noise_rows.append({"condition": condition, "noise_type": noise_type, **{key: float(np.mean([item[key] for item in result])) for key in result[0]}, **{f"{key}_seed_sd": float(np.std([item[key] for item in result], ddof=1)) if len(result)>1 else 0.0 for key in result[0]}})
    noise = pd.DataFrame(noise_rows); noise.to_csv(output["metrics"] / "noise_type_results.csv", index=False, encoding="utf-8-sig")
    return table, paired, noise


def plot_figures(config: Config, manifest: pd.DataFrame, waves: np.ndarray, structures: np.ndarray, predictions: pd.DataFrame, paired: pd.DataFrame, noise: pd.DataFrame) -> None:
    output = paths(config); plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    # 图1：每类一个可复现样本的原始波形与P/A/U结构图。
    chosen = [int(manifest[manifest.label == 1].index[0]), int(manifest[manifest.noise_type == "Type_3_Multipath_Noise"].index[0]), int(manifest[manifest.noise_type == "Type_4_Isolated_Noise"].index[0])]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), gridspec_kw={"width_ratios": [1.3, 1]})
    for row, index in enumerate(chosen):
        axes[row, 0].plot(np.arange(SAMPLE_COUNT) / SAMPLE_RATE * 1e3, waves[index], lw=.65); axes[row, 0].set(title=f"{manifest.iloc[index].noise_type} 波形", xlabel="时间 (ms)", ylabel="归一化幅度")
        image = axes[row, 1].imshow(structures[index], aspect="auto", origin="lower", extent=[MIN_DELAY / SAMPLE_RATE * 1e3, MAX_DELAY / SAMPLE_RATE * 1e3, 0, 3], vmin=0, vmax=1, cmap="viridis")
        axes[row, 1].set(title="BCSC结构图", xlabel="相对时延 (ms)", yticks=[.5, 1.5, 2.5], yticklabels=["P", "A", "U"])
    fig.colorbar(image, ax=axes[:, 1], label="归一化值"); fig.tight_layout(); fig.savefig(output["figures"] / "waveform_structure_examples.png", dpi=180); plt.close(fig)
    # 图2：所有条件的OOF ROC/PR。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in CONDITIONS:
        frame = aggregate_probability(predictions, condition); fpr, tpr, _ = roc_curve(frame.label, frame.p_click); precision, recall, _ = precision_recall_curve(frame.label, frame.p_click)
        axes[0].plot(fpr, tpr, label=CONDITION_LABELS[condition]); axes[1].plot(recall, precision, label=CONDITION_LABELS[condition])
    axes[0].plot([0, 1], [0, 1], "k--", lw=.7); axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC"); axes[1].set(xlabel="Recall", ylabel="Precision", title="PR"); axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(output["figures"] / "roc_pr_curves.png", dpi=180); plt.close(fig)
    # 图3：配对增益与两类负样本FP。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); y = np.arange(len(paired)); axes[0].errorbar(paired.difference, y, xerr=[paired.difference-paired.ci_low, paired.ci_high-paired.difference], fmt="o", capsize=4); axes[0].axvline(0, color="k", lw=.8); axes[0].set(yticks=y, yticklabels=paired.comparison, xlabel="配对 AUROC 差值", title="结构分支配对增益")
    subset = noise[noise.condition.isin(["waveform", "waveform_structure", "waveform_zero_structure"])]; pivot = subset.pivot(index="condition", columns="noise_type", values="false_positives"); pivot.plot.bar(ax=axes[1]); axes[1].set(title="两类负样本误报数", xlabel="条件", ylabel="误报数"); axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(output["figures"] / "paired_gain_noise_false_positives.png", dpi=180); plt.close(fig)
    # 图4：主模型与波形基线校准及置信度分布。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in ("waveform", "waveform_structure"):
        frame = aggregate_probability(predictions, condition); observed, predicted = calibration_curve(frame.label, frame.p_click, n_bins=10, strategy="quantile"); axes[0].plot(predicted, observed, marker="o", label=CONDITION_LABELS[condition]); axes[1].hist(frame.p_click[frame.label == 1], bins=20, alpha=.45, density=True, label=f"{CONDITION_LABELS[condition]} click")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.7); axes[0].set(xlabel="预测概率", ylabel="实际click比例", title="校准曲线"); axes[1].set(xlabel="p_click", ylabel="密度", title="正类置信度分布"); axes[0].legend(fontsize=8); axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(output["figures"] / "calibration_confidence.png", dpi=180); plt.close(fig)


def write_report(config: Config, table: pd.DataFrame, paired: pd.DataFrame, noise: pd.DataFrame, status: pd.DataFrame) -> Path:
    output = paths(config); main = table.set_index("condition").loc["waveform_structure"]; base = table.set_index("condition").loc["waveform"]; comparison = paired[paired.right == "waveform"].iloc[0]
    conclusion = str(comparison.evidence)
    report = f"""# 实验23：BCSC多径结构分支辅助海豚Click识别

## 问题与数据

在实验19固定的7,000个10 ms候选片段（3,500 click、3,500非click；576 kHz、5,761点）上，检验BCSC全候选多径置信结构图能否提高click/no-click识别。清单覆盖34段原始录音，采用固定五折录音组隔离：每个测试折的下一折作验证，其余三折训练。波形沿用去均值、20–190 kHz五阶零相位带通、Tukey(0.04)及峰值归一化的实验19缓存。

## 方法

BCSC使用49点原子、35次外循环和0.075稀疏比例。对每个初始激活候选路径执行8次0.10 ms残差块Bootstrap，正则系数扰动±15%，在±0.03 ms内作一对一最近邻匹配。结构缓存尺寸为`7000×3×1133`；Bootstrap从未检出的候选格点(P/A/U)均置零。成功BCSC样本为{int((status.status == 'ok').sum())}/7,000，失败样本保留全零图并记录状态。

双分支主模型由64维Waveform Transformer和32维Path-CNN结构编码器融合而成。四个神经网络条件运行3种子×5折，即60个模型折；Teager与BCSC-Summary为确定性五折基线。因此长格式折外预测严格为98,000行。

## 主结果

Waveform + Structure 的AUROC为{main.auroc:.4f}（种子SD {main.auroc_seed_sd:.4f}，录音组Bootstrap 95% CI [{main.auroc_ci_low:.4f}, {main.auroc_ci_high:.4f}]），Waveform Transformer为{base.auroc:.4f}。主比较的配对AUROC差为{comparison.difference:.4f}，95% CI [{comparison.ci_low:.4f}, {comparison.ci_high:.4f}]；按预注册规则，结论为：**{conclusion}**。

{markdown_table(table)}

## 噪声类型分析

{markdown_table(noise)}

由于非click样本为抽样候选而非连续录音的全部触发，本实验不报告“每小时误报数”。

## 图表与可复核产物

- 原始波形与(P/A/U)结构图：`figures/waveform_structure_examples.png`
- ROC/PR：`figures/roc_pr_curves.png`
- 配对增益与两类噪声误报：`figures/paired_gain_noise_false_positives.png`
- 校准曲线与置信度：`figures/calibration_confidence.png`
- 长格式折外预测：`predictions/oof_predictions_long.csv`
"""
    report_path = output["root"] / "research_report_zh.md"; report_path.write_text(report, encoding="utf-8"); return report_path


def verify(config: Config) -> dict[str, Any]:
    output = paths(config); legacy = load_legacy(); cfg = bcsc_config(); rng = np.random.default_rng(20260912)
    atom = legacy.normalize_atom(legacy.windows.gaussian(cfg.atom_length, std=cfg.atom_length / 7) * np.cos(2 * np.pi * 90_000 * np.arange(cfg.atom_length) / SAMPLE_RATE))
    signal = rng.normal(0, .002, SAMPLE_COUNT); direct, delay = 2700, int(round(.35e-3 * SAMPLE_RATE)); signal[direct:direct+len(atom)] += atom; signal[direct+delay:direct+delay+len(atom)] += .45 * atom; signal[direct+2*delay:direct+2*delay+len(atom)] += .25 * atom
    structure, metadata = extract_structure_one(signal, 20260912); expected_bin = delay - MIN_DELAY
    synthetic = {"expected_delay_samples": delay, "expected_bin": expected_bin, "p": float(structure[0, expected_bin]), "a": float(structure[1, expected_bin]), "u": float(structure[2, expected_bin]), "status": metadata["status"], "passed": bool(metadata["status"] == "ok" and structure[0, expected_bin] > 0 and structure[1, expected_bin] > 0 and 0 <= structure[2, expected_bin] <= 1)}
    neural = {}; wave, map_value, label = torch.randn(4, SAMPLE_COUNT), torch.rand(4, 3, PATH_BINS), torch.tensor([0., 1., 0., 1.])
    for condition in NEURAL_CONDITIONS:
        model = ClickModel(condition); logits = model(wave, map_value); loss = nn.BCEWithLogitsLoss()(logits, label); loss.backward(); probability = torch.sigmoid(logits)
        neural[condition] = {"shape": list(logits.shape), "loss": float(loss.detach()), "finite_probability": bool(torch.isfinite(probability).all()), "passed": bool(logits.shape == (4,) and torch.isfinite(loss) and torch.isfinite(probability).all())}
    result = {"synthetic_bcsc": synthetic, "neural_forward_backward": neural, "passed": bool(synthetic["passed"] and all(item["passed"] for item in neural.values()))}
    json_dump(output["tests"] / "verification.json", result)
    if not result["passed"]: raise RuntimeError(f"验证失败：{result}")
    return result


def report(config: Config) -> Path:
    manifest, waves = load_reference_data(config); _, structures, status = load_structure(config); output = paths(config)
    metric_path, prediction_path = output["metrics"] / "fold_metrics.csv", output["predictions"] / "oof_predictions_long.csv"
    if not metric_path.exists() or not prediction_path.exists(): raise FileNotFoundError("请先完成 train")
    fold_metrics, predictions = pd.read_csv(metric_path), pd.read_csv(prediction_path)
    fold_metrics["seed"] = fold_metrics["seed"].astype(str)
    predictions["seed"] = predictions["seed"].astype(str)
    if len(predictions) != 98_000: raise RuntimeError("报告拒绝使用不完整折外预测")
    table, paired, noise = summarize(config, fold_metrics, predictions); plot_figures(config, manifest, waves, structures, predictions, paired, noise); path = write_report(config, table, paired, noise, status)
    json_dump(output["root"] / "artifact_validation.json", {"structure_shape": list(structures.shape), "neural_model_folds": int(len(fold_metrics[fold_metrics.condition.isin(NEURAL_CONDITIONS)])), "oof_prediction_rows": int(len(predictions)), "report": str(path), "passed": bool(structures.shape == (7000, 3, PATH_BINS) and len(predictions) == 98000)})
    return path


def predict_single(config: Config, wav_path: Path) -> dict[str, Any]:
    if not wav_path.is_absolute(): raise ValueError("--wav 必须为10 ms WAV的绝对路径")
    if not wav_path.exists(): raise FileNotFoundError(wav_path)
    signal, sample_rate = sf.read(wav_path, dtype="float64", always_2d=False)
    signal = np.asarray(signal).squeeze()
    if sample_rate != SAMPLE_RATE or signal.ndim != 1 or len(signal) != SAMPLE_COUNT: raise ValueError(f"输入必须为576 kHz、5761点单声道WAV；当前为{sample_rate} Hz、{signal.shape}")
    wave = load_legacy().preprocess_waveform(signal, bcsc_config()); structure, extraction = extract_structure_one(wave, 20260912)
    device = resolve_device(config.device); probabilities, thresholds = [], []
    for seed in SEEDS:
        for fold in range(5):
            path = neural_run_dir(config, "waveform_structure", seed, fold) / "best_checkpoint.pt"
            completion = neural_run_dir(config, "waveform_structure", seed, fold) / "COMPLETE.json"
            if not path.exists() or not completion.exists(): raise FileNotFoundError("主模型15个检查点不完整，无法预测")
            model = ClickModel("waveform_structure").to(device); model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"]); model.eval()
            with torch.no_grad(): logit = float(model(torch.as_tensor(wave[None], device=device), torch.as_tensor(structure[None], device=device)).cpu())
            saved = json.loads(completion.read_text(encoding="utf-8")); probabilities.append(1 / (1 + math.exp(-logit / float(saved["temperature"])))); thresholds.append(float(saved["threshold"]))
    threshold = float(np.median(thresholds)); probability = float(np.mean(probabilities))
    result = {"p_click": probability, "predicted_label": int(probability >= threshold), "threshold": threshold, "ensemble_members": 15, "bcsc_status": extraction["status"]}
    print(json.dumps(result, ensure_ascii=False, indent=2)); return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("all", "extract-structure", "train", "report", "predict", "verify")); parser.add_argument("--wav", type=Path); parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1))); parser.add_argument("--device", default="auto"); parser.add_argument("--no-resume", action="store_true"); parser.add_argument("--bootstrap-runs", type=int, default=2000); return parser.parse_args()


def main() -> None:
    arguments = parse_args(); config = Config(workers=arguments.workers, device=arguments.device, bootstrap_runs=arguments.bootstrap_runs); output = paths(config); json_dump(output["root"] / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir), "reference_dir": str(config.reference_dir), "seeds": list(SEEDS), "path_shape": [3, PATH_BINS]})
    if arguments.command == "predict":
        if arguments.wav is None: raise ValueError("predict需要--wav <10ms绝对路径>")
        predict_single(config, arguments.wav.resolve()); return
    if arguments.command == "verify": verify(config); return
    if arguments.command in ("all", "extract-structure"):
        print("[总体进度  0%] 验证合成路径与四个神经网络条件", flush=True); verify(config); extract_structure(config, force=arguments.no_resume)
    if arguments.command in ("all", "train"):
        print("[总体进度 45%] 训练六个对比条件", flush=True); train_all(config, resume=not arguments.no_resume)
    if arguments.command in ("all", "report"):
        print("[总体进度 90%] 统计、绘图并写入真实结果报告", flush=True); result = report(config); print(f"[总体进度100%] 完成：{result}", flush=True)


if __name__ == "__main__": main()
