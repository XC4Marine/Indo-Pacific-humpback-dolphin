#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v5：噪声感知置信 BCSC 的最小低SNR分类验证。

本脚本只读实验21的固定波形，以及实验23-v2已完成的原BCSC结构缓存。
命令：all / extract-structure / train / report / verify。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
V2_PATH = ROOT / "23_BCSCStructureClickClassification" / "experiment_v2.py"
SNR_ROOT = ROOT / "21_PhysicsInformedClickClassification" / "outputs_snr_m15_p15_full_5seeds"
V2_ROOT = ROOT / "23_BCSCStructureClickClassification" / "v2_low_snr_gated"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v5_noise_adaptive_bcsc_pilot"
SEED = 20260826
SAMPLE_RATE, SAMPLE_COUNT, PATH_BINS = 576_000, 5_761, 1_133
MIN_DELAY, MAX_DELAY = 20, 1_152
BOOTSTRAP_REPLICATES = 8
MIN_PERSISTENCE = 2 / BOOTSTRAP_REPLICATES
SNR_BINS = ((-15.0, -10.0, "[-15,-10)"), (-10.0, -5.0, "[-10,-5)"),
            (-5.0, 0.0, "[-5,0)"), (0.0, 5.0, "[0,5)"),
            (5.0, 10.0, "[5,10)"), (10.0, 15.000001, "[10,15]"))
CONDITIONS = ("waveform", "gated_original_bcsc", "gated_na_bcsc", "gated_na_zero_structure")
LABELS = {
    "waveform": "Waveform Transformer",
    "gated_original_bcsc": "原BCSC门控融合",
    "gated_na_bcsc": "v5噪声感知BCSC门控融合",
    "gated_na_zero_structure": "v5门控+Zero-Structure",
}
METRICS = ("auroc", "auprc", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "brier")


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SNR_ROOT / "dataset_manifest.csv"
    waveform_path: Path = SNR_ROOT / "cache" / "preprocessed_waveforms.npy"
    original_structure_path: Path = V2_ROOT / "cache" / "structure_maps.npy"
    original_completed_path: Path = V2_ROOT / "cache" / "structure_completed.npy"
    original_contract_path: Path = V2_ROOT / "cache" / "structure_contract.json"
    workers: int = max(1, min(6, os.cpu_count() or 1))
    batch_size: int = 64
    max_epochs: int = 25
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bootstrap_runs: int = 2_000
    device: str = "auto"


def v2() -> Any:
    """加载已锁定的v2实现，确保模型、指标与折分合同不漂移。"""
    name = "experiment23_v2_shared_for_v5"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载实验23-v2：{V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paths(config: Config) -> dict[str, Path]:
    result = {name: config.output_dir / name for name in ("cache", "models", "metrics", "predictions", "figures", "logs", "tests")}
    result["root"] = config.output_dir
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
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
        raise RuntimeError("指定CUDA但当前PyTorch不可用")
    return device


def load_data(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    if not config.manifest_path.exists() or not config.waveform_path.exists():
        raise FileNotFoundError("缺少实验21固定低SNR清单或预处理波形")
    manifest = pd.read_csv(config.manifest_path)
    waves = np.load(config.waveform_path, mmap_mode="r")
    positive = manifest.label.eq(1)
    checks = [
        len(manifest) == 7_000,
        tuple(waves.shape) == (7_000, SAMPLE_COUNT),
        manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
        manifest.ori_file_num.nunique() == 34,
        manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
        manifest.loc[positive, "target_snr_db"].between(-15, 15).all(),
        bool(np.isfinite(waves).all()),
    ]
    if not all(checks):
        raise RuntimeError("实验21低SNR数据合同验证失败")
    return manifest, waves


def load_original_structures(config: Config) -> np.ndarray:
    """只读v2缓存；拒绝用不匹配的原BCSC对照替代。"""
    required = (config.original_structure_path, config.original_completed_path, config.original_contract_path)
    if not all(path.exists() for path in required):
        raise FileNotFoundError("缺少v2原BCSC结构缓存、完成标记或合同")
    contract = json.loads(config.original_contract_path.read_text(encoding="utf-8"))
    expected_source = str(config.waveform_path)
    if contract.get("shape") != [7000, 3, PATH_BINS] or contract.get("source") != expected_source:
        raise RuntimeError("v2原BCSC缓存与v5波形来源或尺寸不一致")
    completed = np.load(config.original_completed_path)
    maps = np.load(config.original_structure_path, mmap_mode="r")
    if maps.shape != (7000, 3, PATH_BINS) or completed.shape != (7000,) or not completed.all():
        raise RuntimeError("v2原BCSC结构缓存不完整")
    if not np.isfinite(maps).all():
        raise RuntimeError("v2原BCSC结构缓存包含非有限值")
    return maps


def apply_frequency_weight(signal: np.ndarray, frequency_weight: np.ndarray) -> np.ndarray:
    """对实信号施加零相位循环频率权重。"""
    value = np.asarray(signal, dtype=np.float64)
    if len(frequency_weight) != len(np.fft.rfft(value)):
        raise ValueError("频率权重长度与信号不一致")
    return np.fft.irfft(np.fft.rfft(value) * frequency_weight, n=len(value)).real


def residual_whitener(residual: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """由初始BCSC残差的平滑周期图构造有限的有色噪声白化权重。"""
    value = np.asarray(residual, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(value)) ** 2 / max(len(value), 1)
    smooth = np.convolve(spectrum, np.ones(33, dtype=np.float64) / 33.0, mode="same")
    positive = smooth[np.isfinite(smooth) & (smooth > 0)]
    if len(positive) == 0:
        raise RuntimeError("残差功率谱无有效正值")
    floor = max(float(np.percentile(positive, 5)) * 0.10, np.finfo(np.float64).tiny)
    raw = 1.0 / np.sqrt(np.maximum(smooth, floor))
    low, high = np.percentile(raw, [5, 95])
    weight = np.clip(raw, max(float(low), np.finfo(np.float64).tiny), max(float(high), float(low)))
    if not np.isfinite(weight).all() or np.max(weight) <= 0:
        raise RuntimeError("白化权重无效")
    return weight, {"psd_floor": floor, "weight_min": float(np.min(weight)), "weight_max": float(np.max(weight))}


def mad_sigma(value: np.ndarray) -> float:
    centered = np.asarray(value, dtype=np.float64) - float(np.median(value))
    return max(float(np.median(np.abs(centered)) / 0.6744897501960817), 1e-10)


def weighted_residual(signal: np.ndarray, atom: np.ndarray, activation: np.ndarray, weight: np.ndarray, legacy: Any) -> np.ndarray:
    return apply_frequency_weight(legacy.reconstruct(atom, activation, len(signal)) - signal, weight)


def weighted_code_gradient(residual_white: np.ndarray, atom: np.ndarray, activation_length: int, weight: np.ndarray, legacy: Any) -> np.ndarray:
    return legacy.adjoint_atom(apply_frequency_weight(residual_white, weight), atom, activation_length)


def weighted_fista(signal: np.ndarray, atom: np.ndarray, lam: float, weight: np.ndarray, legacy: Any,
                   initial: np.ndarray | None, iterations: int) -> np.ndarray:
    activation_length = len(signal) - len(atom) + 1
    x = np.zeros(activation_length, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    z, momentum = x.copy(), 1.0
    lipschitz = legacy.spectral_lipschitz(atom, len(signal)) * float(np.max(weight ** 2))
    for _ in range(iterations):
        residual_white = weighted_residual(signal, atom, z, weight, legacy)
        gradient = weighted_code_gradient(residual_white, atom, activation_length, weight, legacy)
        x_new = legacy.soft_threshold(z - gradient / lipschitz, lam / lipschitz)
        momentum_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        z = x_new + ((momentum - 1.0) / momentum_new) * (x_new - x)
        x, momentum = x_new, momentum_new
    return x


def weighted_update_atom(signal: np.ndarray, activation: np.ndarray, atom: np.ndarray, weight: np.ndarray,
                         legacy: Any, legacy_config: Any) -> np.ndarray:
    if np.linalg.norm(activation) < 1e-12:
        return atom
    step = 0.8 / (legacy.spectral_lipschitz(activation, len(signal)) * float(np.max(weight ** 2)))
    updated = np.asarray(atom, dtype=np.float64).copy()
    taper = legacy.windows.tukey(len(atom), alpha=0.20)
    for _ in range(legacy_config.dictionary_iterations):
        residual_white = weighted_residual(signal, updated, activation, weight, legacy)
        gradient = legacy.adjoint_code(apply_frequency_weight(residual_white, weight), activation, len(updated))
        updated = legacy.hard_project_3db_bandwidth(legacy.normalize_atom((updated - step * gradient) * taper), legacy_config)
    return updated


def shift_atom_and_code(atom: np.ndarray, activation: np.ndarray, legacy: Any, legacy_config: Any) -> tuple[np.ndarray, np.ndarray]:
    return legacy.shift_atom_and_code(atom, activation, legacy_config)


def noise_adaptive_bcsc(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    """初始BCSC残差估计噪声，再以白化目标与MAD通用阈值重求解。"""
    shared, legacy_config = v2(), v2().base().bcsc_config()
    legacy = shared.base().load_legacy()
    y = np.asarray(signal, dtype=np.float64)
    initial_atom, _, initial_estimate = legacy.bcsc(y, legacy_config)
    weight, noise_info = residual_whitener(y - initial_estimate)
    initial = np.zeros(len(y) - len(initial_atom) + 1, dtype=np.float64)
    atom = initial_atom.copy()
    for _ in range(legacy_config.outer_iterations):
        white_signal = apply_frequency_weight(y, weight)
        sigma = mad_sigma(white_signal - apply_frequency_weight(legacy.reconstruct(atom, initial, len(y)), weight))
        lam = sigma * math.sqrt(2.0 * math.log(max(len(initial), 2)))
        initial = weighted_fista(y, atom, lam, weight, legacy, initial, legacy_config.fista_iterations)
        atom = weighted_update_atom(y, initial, atom, weight, legacy, legacy_config)
        atom, initial = shift_atom_and_code(atom, initial, legacy, legacy_config)
    white_signal = apply_frequency_weight(y, weight)
    sigma = mad_sigma(white_signal - apply_frequency_weight(legacy.reconstruct(atom, initial, len(y)), weight))
    lam = sigma * math.sqrt(2.0 * math.log(max(len(initial), 2)))
    activation = weighted_fista(y, atom, lam, weight, legacy, initial, 2 * legacy_config.fista_iterations)
    estimate = legacy.reconstruct(atom, activation, len(y))
    noise_info.update({"mad_sigma": sigma, "lambda": lam, "atom_norm": float(np.linalg.norm(atom))})
    return atom, activation, estimate, weight, lam, noise_info


def match_paths(reference: list[int], candidates: list[int], code: np.ndarray, tolerance: int) -> dict[int, list[tuple[int, float]]]:
    records = {item: [] for item in reference}
    unused = set(candidates)
    for item in reference:
        available = [candidate for candidate in unused if abs(candidate - item) <= tolerance]
        if available:
            chosen = min(available, key=lambda candidate: (abs(candidate - item), -abs(float(code[candidate]))))
            unused.remove(chosen)
            records[item].append((chosen, float(code[chosen])))
    return records


def extract_na_structure_one(signal: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """只依赖波形；不读取标签、真实SNR或任何测试真值。"""
    shared, legacy_config = v2(), v2().base().bcsc_config()
    legacy = shared.base().load_legacy()
    started = time.perf_counter()
    structure = np.zeros((3, PATH_BINS), dtype=np.float32)
    try:
        atom, activation, estimate, weight, lam, noise_info = noise_adaptive_bcsc(signal)
        if not np.isfinite(atom).all() or not np.isfinite(activation).all() or not np.isfinite(weight).all():
            raise RuntimeError("NA-BCSC产物包含非有限值")
        if not np.isclose(np.linalg.norm(atom), 1.0, atol=1e-5):
            raise RuntimeError("NA-BCSC原子未满足单位范数")
        if np.max(np.abs(activation)) < 1e-12:
            return structure, {"status": "empty", "error": "", "direct_index": -1, "candidate_path_count": 0,
                               "detected_path_count": 0, "elapsed_s": time.perf_counter() - started, **noise_info}
        direct = int(np.argmax(np.abs(activation)))
        candidates = legacy.extract_candidate_indices(activation, legacy_config)
        references = [int(index) for index in candidates if MIN_DELAY <= int(index) - direct <= MAX_DELAY]
        residual = np.asarray(signal, dtype=np.float64) - estimate
        rng = np.random.default_rng(seed)
        block_length = int(round(0.10e-3 * SAMPLE_RATE))
        tolerance = int(round(0.03e-3 * SAMPLE_RATE))
        records: dict[int, list[tuple[int, float]]] = {item: [] for item in references}
        for _ in range(BOOTSTRAP_REPLICATES):
            synthetic = estimate + legacy.moving_block_bootstrap(residual, block_length, rng)
            code = weighted_fista(synthetic, atom, lam, weight, legacy, activation, legacy_config.fista_iterations)
            matched = match_paths(references, list(map(int, legacy.extract_candidate_indices(code, legacy_config))), code, tolerance)
            for item in references:
                records[item].extend(matched[item])
        direct_abs = abs(float(activation[direct])) + 1e-12
        stable = 0
        for item, found in records.items():
            persistence = len(found) / BOOTSTRAP_REPLICATES
            if persistence < MIN_PERSISTENCE:
                continue
            indices = np.asarray([pair[0] for pair in found], dtype=np.float64)
            coefficients = np.asarray([pair[1] for pair in found], dtype=np.float64)
            bin_index = item - direct - MIN_DELAY
            if not 0 <= bin_index < PATH_BINS:
                continue
            amplitude = persistence * math.log1p(float(np.median(np.abs(coefficients))) / direct_abs)
            spread = 1.0 if len(found) == 1 else float(np.clip((np.percentile(indices, 97.5) - np.percentile(indices, 2.5)) / SAMPLE_RATE * 1e3 / 0.06, 0.0, 1.0))
            uncertainty = float(np.clip(0.5 * spread + 0.5 * (1.0 - persistence), 0.0, 1.0))
            structure[0, bin_index] = max(structure[0, bin_index], persistence)
            structure[1, bin_index] = max(structure[1, bin_index], amplitude)
            structure[2, bin_index] = max(structure[2, bin_index], uncertainty)
            stable += 1
        status = "ok" if stable else "empty"
        return structure, {"status": status, "error": "", "direct_index": direct, "candidate_path_count": len(references),
                           "detected_path_count": stable, "elapsed_s": time.perf_counter() - started, **noise_info}
    except Exception as error:
        return structure, {"status": "failed", "error": f"{type(error).__name__}: {error}", "direct_index": -1,
                           "candidate_path_count": 0, "detected_path_count": 0, "elapsed_s": time.perf_counter() - started,
                           "psd_floor": float("nan"), "weight_min": float("nan"), "weight_max": float("nan"),
                           "mad_sigma": float("nan"), "lambda": float("nan"), "atom_norm": float("nan")}


def extract_worker(payload: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, dict[str, Any]]:
    index, signal = payload
    structure, status = extract_na_structure_one(signal, SEED + index)
    status["sample_index"] = index
    return index, structure, status


def extract_structure(config: Config, force: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    output = paths(config)
    manifest, waves = load_data(config)
    map_path, done_path, status_path = output["cache"] / "na_structure_maps.npy", output["cache"] / "na_structure_completed.npy", output["cache"] / "na_structure_status.csv"
    if force:
        for path in (map_path, done_path, status_path):
            if path.exists():
                path.unlink()
    if map_path.exists() and done_path.exists():
        maps, completed = np.lib.format.open_memmap(map_path, mode="r+"), np.load(done_path)
        if maps.shape != (7000, 3, PATH_BINS) or completed.shape != (7000,):
            raise RuntimeError("v5结构缓存尺寸错误")
    else:
        maps = np.lib.format.open_memmap(map_path, mode="w+", dtype=np.float32, shape=(7000, 3, PATH_BINS))
        maps[:] = 0.0
        maps.flush()
        completed = np.zeros(7000, dtype=bool)
        np.save(done_path, completed)
    records: dict[int, dict[str, Any]] = {}
    if status_path.exists():
        prior = pd.read_csv(status_path)
        if "sample_index" in prior:
            records = {int(row["sample_index"]): row for row in prior.dropna(subset=["sample_index"]).to_dict("records")}
    pending = np.flatnonzero(~completed)
    if len(pending):
        print(f"[总体进度 10%] v5 NA-BCSC：已完成{int(completed.sum())}/7000，待处理{len(pending)}", flush=True)
        progress = tqdm(total=len(pending), desc="总体进度：v5 NA-BCSC结构提取", unit="样本")
        payloads = ((int(index), np.asarray(waves[index], dtype=np.float32)) for index in pending)
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for count, (index, structure, status) in enumerate(pool.map(extract_worker, payloads, chunksize=1), 1):
                maps[index], completed[index], records[index] = structure, True, status
                progress.update(1)
                if count % 25 == 0 or count == len(pending):
                    maps.flush()
                    np.save(done_path, completed)
                    pd.DataFrame.from_dict(records, orient="index").sort_values("sample_index").to_csv(status_path, index=False, encoding="utf-8-sig")
        progress.close()
    maps.flush()
    np.save(done_path, completed)
    status = pd.DataFrame.from_dict(records, orient="index").sort_values("sample_index")
    metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "label", "ori_file_num", "test_fold", "noise_type", "target_snr_db"]]
    status = metadata.merge(status, on="sample_index", how="left", validate="one_to_one")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    if not completed.all() or maps.shape != (7000, 3, PATH_BINS) or not np.isfinite(maps).all():
        raise RuntimeError("v5 NA-BCSC结构缓存不完整或包含非有限值")
    dump(output["cache"] / "na_structure_contract.json", {
        "shape": [7000, 3, PATH_BINS], "source": str(config.waveform_path), "seed": SEED,
        "method": {"initial_bcsc": "experiment19 fixed configuration", "whitened_objective": True,
                   "threshold": "MAD universal threshold", "bootstrap_runs": BOOTSTRAP_REPLICATES,
                   "minimum_persistence": MIN_PERSISTENCE, "target_snr_read": False, "label_read": False},
    })
    return status, np.load(map_path, mmap_mode="r")


def load_na_structures(config: Config) -> np.ndarray:
    output = paths(config)
    map_path, done_path = output["cache"] / "na_structure_maps.npy", output["cache"] / "na_structure_completed.npy"
    if not map_path.exists() or not done_path.exists():
        raise FileNotFoundError("请先运行extract-structure")
    maps, done = np.load(map_path, mmap_mode="r"), np.load(done_path)
    if maps.shape != (7000, 3, PATH_BINS) or not done.all() or not np.isfinite(maps).all():
        raise RuntimeError("v5 NA-BCSC结构缓存不完整")
    return maps


class DualStructureDataset(Dataset):
    def __init__(self, waves: np.ndarray, original: np.ndarray, na: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.waves, self.original, self.na, self.labels, self.indices = waves, original, na, labels, indices
    def __len__(self) -> int:
        return len(self.indices)
    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32),
                torch.as_tensor(np.asarray(self.original[index]), dtype=torch.float32),
                torch.as_tensor(np.asarray(self.na[index]), dtype=torch.float32),
                torch.tensor(self.labels[index], dtype=torch.float32), index)


def loader(waves: np.ndarray, original: np.ndarray, na: np.ndarray, labels: np.ndarray, indices: np.ndarray,
           batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(DualStructureDataset(waves, original, na, labels, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def structure_for(condition: str, original: torch.Tensor, na: torch.Tensor) -> torch.Tensor:
    if condition == "gated_original_bcsc":
        return original
    if condition == "gated_na_bcsc":
        return na
    if condition == "gated_na_zero_structure":
        return torch.zeros_like(na)
    raise ValueError(f"不支持的结构条件：{condition}")


def model_for(condition: str) -> nn.Module:
    shared = v2()
    if condition == "waveform":
        return shared.Model("waveform")
    return shared.Model("gated_fusion")


@torch.no_grad()
def predict(model: nn.Module, condition: str, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    ids, labels, logits = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for wave, original, na, label, index in data:
        wave = wave.to(device, non_blocking=True)
        if condition == "waveform":
            output = model(wave, torch.zeros_like(na, device=device))
        else:
            structure = structure_for(condition, original, na).to(device, non_blocking=True)
            output = model(wave, structure)
        logits.append(output.cpu().numpy())
        ids.append(np.asarray(index))
        labels.append(label.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(ids), np.concatenate(labels), np.concatenate(logits), time.perf_counter() - started


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.0, logits / temperature) - labels * logits / temperature))
    result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.0


def train_condition(condition: str, fold: int, waves: np.ndarray, original: np.ndarray, na: np.ndarray,
                    manifest: pd.DataFrame, config: Config, resume: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    output = paths(config)
    directory = output["models"] / condition / f"seed_{SEED}" / f"fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    complete, checkpoint, prediction_file = directory / "COMPLETE.json", directory / "best_checkpoint.pt", directory / "test_predictions.csv"
    if resume and complete.exists() and checkpoint.exists() and prediction_file.exists():
        return json.loads(complete.read_text(encoding="utf-8")), pd.read_csv(prediction_file)
    shared = v2()
    seed_all(SEED + fold * 100)
    device = device_for(config.device)
    labels = manifest.label.to_numpy(dtype=np.float32)
    train_index, validation_index, test_index = shared.split(manifest, fold)
    loads = {"train": loader(waves, original, na, labels, train_index, config.batch_size, True),
             "validation": loader(waves, original, na, labels, validation_index, config.batch_size * 2),
             "test": loader(waves, original, na, labels, test_index, config.batch_size * 2)}
    model = model_for(condition).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion, history, best, stale = nn.BCEWithLogitsLoss(), [], -np.inf, 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for wave, original_map, na_map, label, _ in loads["train"]:
            optimizer.zero_grad(set_to_none=True)
            wave = wave.to(device, non_blocking=True)
            if condition == "waveform":
                structure = torch.zeros_like(na_map, device=device)
            else:
                structure = structure_for(condition, original_map, na_map).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(wave, structure), label.to(device, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val_ids, val_labels, val_logits, _ = predict(model, condition, loads["validation"], device)
        frame = manifest.iloc[val_ids].reset_index(drop=True).assign(label=val_labels.astype(int), logit=val_logits)
        score = float(roc_auc_score(val_labels[v2().low_snr_mask(frame)], val_logits[v2().low_snr_mask(frame)]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_low_snr_auroc": score})
        if score > best + 1e-7:
            best, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "condition": condition, "fold": fold, "seed": SEED}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    val_ids, val_labels, val_logits, _ = predict(model, condition, loads["validation"], device)
    temperature = fit_temperature(val_logits, val_labels)
    probability = 1.0 / (1.0 + np.exp(-val_logits / temperature))
    cut = v2().threshold(val_labels, probability)
    test_ids, test_labels, test_logits, inference_seconds = predict(model, condition, loads["test"], device)
    test_probability = 1.0 / (1.0 + np.exp(-test_logits / temperature))
    details = v2().metric(test_labels.astype(int), test_probability, cut)
    details.update({"condition": condition, "seed": SEED, "test_fold": fold, "validation_fold": (fold + 1) % 5,
                    "threshold": cut, "temperature": temperature, "best_validation_low_snr_auroc": best,
                    "best_epoch": int(max(row["epoch"] for row in history if row["validation_low_snr_auroc"] == best)),
                    "parameters": int(sum(item.numel() for item in model.parameters() if item.requires_grad)),
                    "inference_ms_per_sample": inference_seconds / len(test_ids) * 1e3,
                    "training_seconds": time.perf_counter() - started})
    prediction = pd.DataFrame({"sample_index": test_ids, "label": test_labels.astype(int), "p_click": test_probability,
                               "predicted_label": (test_probability >= cut).astype(int), "threshold": cut,
                               "temperature": temperature, "condition": condition, "seed": SEED, "test_fold": fold})
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False, encoding="utf-8-sig")
    prediction.to_csv(prediction_file, index=False, encoding="utf-8-sig")
    dump(complete, details)
    return details, prediction


def train(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = paths(config)
    manifest, waves = load_data(config)
    original, na = load_original_structures(config), load_na_structures(config)
    metrics_path, prediction_path = output["metrics"] / "fold_metrics.csv", output["predictions"] / "oof_predictions_long.csv"
    if resume and metrics_path.exists() and prediction_path.exists():
        metrics, predictions = pd.read_csv(metrics_path), pd.read_csv(prediction_path)
    else:
        results, values = [], []
        progress = tqdm(total=20, desc="总体进度：v5四条件五折", unit="模型折")
        for fold in range(5):
            for condition in CONDITIONS:
                row, prediction = train_condition(condition, fold, waves, original, na, manifest, config, resume)
                results.append(row)
                values.append(prediction)
                progress.update(1)
                progress.set_postfix(fold=fold, condition=condition, auroc=f"{row['auroc']:.3f}")
        progress.close()
        metrics, predictions = pd.DataFrame(results), pd.concat(values, ignore_index=True)
        metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
        predictions = predictions.merge(metadata, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
        if len(predictions) != 28_000 or predictions.duplicated(["condition", "seed", "sample_id"]).any():
            raise RuntimeError(f"v5 OOF合同失败：{len(predictions)}行")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    return metrics, predictions


def bootstrap_ci(frame: pd.DataFrame, metric_name: str, rng: np.random.Generator, runs: int) -> tuple[float, float]:
    groups, values = frame.ori_file_num.unique(), []
    for _ in range(runs):
        selected = rng.choice(groups, len(groups), replace=True)
        sample = pd.concat([frame[frame.ori_file_num == group] for group in selected], ignore_index=True)
        if sample.label.nunique() == 2:
            values.append(v2().metric(sample.label.to_numpy(), sample.p_click.to_numpy(), sample.threshold.to_numpy())[metric_name])
    return tuple(map(float, np.percentile(values, [2.5, 97.5])))


def paired_low_snr(left: pd.DataFrame, right: pd.DataFrame, rng: np.random.Generator, runs: int) -> tuple[float, float, float]:
    joined = left[["sample_id", "label", "ori_file_num", "target_snr_db", "p_click"]].merge(
        right[["sample_id", "p_click"]], on="sample_id", suffixes=("_left", "_right"), validate="one_to_one")
    joined = joined[(joined.label == 0) | ((joined.label == 1) & joined.target_snr_db.between(-15, 0, inclusive="left"))]
    point = float(roc_auc_score(joined.label, joined.p_click_left) - roc_auc_score(joined.label, joined.p_click_right))
    groups, values = joined.ori_file_num.unique(), []
    for _ in range(runs):
        selected = rng.choice(groups, len(groups), replace=True)
        sample = pd.concat([joined[joined.ori_file_num == group] for group in selected], ignore_index=True)
        if sample.label.nunique() == 2:
            values.append(float(roc_auc_score(sample.label, sample.p_click_left) - roc_auc_score(sample.label, sample.p_click_right)))
    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def summarize(config: Config, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output, rng, rows = paths(config), np.random.default_rng(SEED), []
    for condition in CONDITIONS:
        frame = predictions[predictions.condition == condition]
        row: dict[str, Any] = {"condition": condition, "condition_label": LABELS[condition]}
        for name in METRICS:
            row[name] = v2().metric(frame.label.to_numpy(), frame.p_click.to_numpy(), frame.threshold.to_numpy())[name]
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = bootstrap_ci(frame, name, rng, config.bootstrap_runs)
        rows.append(row)
    main = pd.DataFrame(rows)
    snr_rows, noise_rows = [], []
    for condition in CONDITIONS:
        frame = predictions[predictions.condition == condition]
        for low, high, label in SNR_BINS:
            selected = frame[(frame.label == 0) | ((frame.label == 1) & (frame.target_snr_db >= low) & (frame.target_snr_db < high))]
            values = v2().metric(selected.label.to_numpy(), selected.p_click.to_numpy(), selected.threshold.to_numpy())
            snr_rows.append({"condition": condition, "snr_bin": label, "snr_low_db": low, "snr_high_db": min(high, 15.0),
                             "positive_count": int((selected.label == 1).sum()), "negative_count": int((selected.label == 0).sum()),
                             **{name: values[name] for name in ("auroc", "auprc", "recall")}})
        for noise_type in ("Type_3_Multipath_Noise", "Type_4_Isolated_Noise"):
            selected = pd.concat([frame[frame.label == 1], frame[(frame.label == 0) & (frame.noise_type == noise_type)]], ignore_index=True)
            values = v2().metric(selected.label.to_numpy(), selected.p_click.to_numpy(), selected.threshold.to_numpy())
            noise_rows.append({"condition": condition, "noise_type": noise_type, "false_positives": values["fp"],
                               "auroc": values["auroc"], "auprc": values["auprc"], "recall": values["recall"]})
    comparisons = []
    left = predictions[predictions.condition == "gated_na_bcsc"]
    for right_name in ("gated_original_bcsc", "gated_na_zero_structure"):
        point, low, high = paired_low_snr(left, predictions[predictions.condition == right_name], rng, config.bootstrap_runs)
        comparisons.append({"left": "gated_na_bcsc", "right": right_name, "scope": "低SNR[-15,0)",
                            "auroc_difference": point, "ci_low": low, "ci_high": high, "supports": bool(low > 0)})
    comparison_frame = pd.DataFrame(comparisons)
    main.to_csv(output["metrics"] / "main_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(snr_rows).to_csv(output["metrics"] / "snr_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(noise_rows).to_csv(output["metrics"] / "noise_type_results.csv", index=False, encoding="utf-8-sig")
    comparison_frame.to_csv(output["metrics"] / "paired_low_snr_auroc_bootstrap.csv", index=False, encoding="utf-8-sig")
    return main, pd.DataFrame(snr_rows), pd.DataFrame(noise_rows), comparison_frame


def figures(config: Config, predictions: pd.DataFrame, snr: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    directory = paths(config)["figures"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in CONDITIONS:
        frame = predictions[predictions.condition == condition]
        fpr, tpr, _ = roc_curve(frame.label, frame.p_click)
        precision, recall, _ = precision_recall_curve(frame.label, frame.p_click)
        axes[0].plot(fpr, tpr, label=LABELS[condition])
        axes[1].plot(recall, precision, label=LABELS[condition])
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set(title="全SNR ROC", xlabel="FPR", ylabel="TPR")
    axes[1].set(title="全SNR PR", xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "roc_pr_all_snr.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in CONDITIONS:
        item = snr[snr.condition == condition]
        center = (item.snr_low_db + item.snr_high_db) / 2
        axes[0].plot(center, item.auroc, marker="o", label=LABELS[condition])
        axes[1].plot(center, item.auprc, marker="o", label=LABELS[condition])
    axes[0].set(title="分SNR AUROC", xlabel="正样本目标SNR (dB)", ylabel="AUROC", ylim=(0, 1.02))
    axes[1].set(title="分SNR AUPRC", xlabel="正样本目标SNR (dB)", ylabel="AUPRC", ylim=(0, 1.02))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "snr_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y = np.arange(len(comparisons))
    axes[0].errorbar(comparisons.auroc_difference, y,
                     xerr=[comparisons.auroc_difference - comparisons.ci_low, comparisons.ci_high - comparisons.auroc_difference], fmt="o", capsize=3)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set(yticks=y, yticklabels=comparisons.right, xlabel="v5 NA-BCSC AUROC差", title="低SNR配对Bootstrap")
    for condition in ("waveform", "gated_na_bcsc"):
        frame = predictions[predictions.condition == condition]
        observed, estimated = calibration_curve(frame.label, frame.p_click, n_bins=10, strategy="quantile")
        axes[1].plot(estimated, observed, marker="o", label=LABELS[condition])
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[1].set(title="校准曲线", xlabel="预测概率", ylabel="click比例")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "paired_gain_calibration.png", dpi=180)
    plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [[f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
                      *["| " + " | ".join(row) + " |" for row in rows]])


def report(config: Config) -> Path:
    output = paths(config)
    prediction_path = output["predictions"] / "oof_predictions_long.csv"
    if not prediction_path.exists():
        raise FileNotFoundError("请先完成train")
    predictions = pd.read_csv(prediction_path)
    if len(predictions) != 28_000:
        raise RuntimeError("报告拒绝使用不完整的v5 OOF预测")
    main, snr, noise, comparisons = summarize(config, predictions)
    figures(config, predictions, snr, comparisons)
    passed = len(comparisons) == 2 and comparisons.ci_low.gt(0).all()
    conclusion = "初步支持低SNR结构增益" if passed else "当前最小验证未同时支持低SNR结构增益"
    text = f"""# 实验23-v5：噪声感知置信 BCSC 最小分类验证

本实验只使用实验21固定低SNR混合数据：7,000条样本、34段录音、正样本目标SNR在[-15,15] dB。采用单训练种子（{SEED}）与五折录音组隔离。真实`target_snr_db`只用于验证集低SNR模型选择与结果分层，不输入BCSC特征提取或分类模型。

v5先以原BCSC残差估计有色噪声白化权重，再以MAD通用阈值执行白化残差度量下的BCSC，并使用8次残差块bootstrap生成P/A/U置信结构图。该最小验证不含路径真值恢复评估，也不量化随机初始化不确定性。

主终点是`[-15,0)` dB正样本与同测试fold全部负样本的AUROC。只有v5相对原BCSC门控融合和v5 Zero-Structure门控融合的两项配对95%置信区间下界均大于0，才给出“初步支持”结论。

## 全范围结果

{markdown(main[["condition_label", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "balanced_accuracy", "f1", "mcc", "brier"]])}

## 主比较

{markdown(comparisons)}

**结论：{conclusion}。** 单种子录音组Bootstrap只反映录音组抽样不确定性，不能作为随机初始化鲁棒性或完整方法创新成立的证据。

## 分SNR结果

{markdown(snr)}

## 噪声类型结果

{markdown(noise)}
"""
    destination = output["root"] / "research_report_zh.md"
    destination.write_text(text, encoding="utf-8")
    dump(output["root"] / "artifact_validation.json", {"na_structure_shape": [7000, 3, PATH_BINS], "neural_model_folds": 20,
                                                          "oof_rows": len(predictions), "seed": SEED,
                                                          "primary_screen_passed": bool(passed), "passed": True})
    return destination


def verify(config: Config) -> dict[str, Any]:
    output = paths(config)
    shared, legacy_config = v2(), v2().base().bcsc_config()
    legacy = shared.base().load_legacy()
    rng = np.random.default_rng(SEED)
    atom = legacy.normalize_atom(legacy.windows.gaussian(legacy_config.atom_length, std=legacy_config.atom_length / 7) *
                                 np.cos(2 * np.pi * 90_000 * np.arange(legacy_config.atom_length) / SAMPLE_RATE))
    signal, direct, delay = rng.normal(0, 0.002, SAMPLE_COUNT), 2700, int(round(0.35e-3 * SAMPLE_RATE))
    signal[direct:direct + len(atom)] += atom
    signal[direct + delay:direct + delay + len(atom)] += 0.45 * atom
    structure, status = extract_na_structure_one(signal, SEED)
    bin_index = delay - MIN_DELAY
    tolerance = int(round(0.03e-3 * SAMPLE_RATE))
    candidate_slice = structure[0, max(0, bin_index - tolerance): min(PATH_BINS, bin_index + tolerance + 1)]
    detected_bin = max(0, bin_index - tolerance) + int(np.argmax(candidate_slice))
    synthetic = {"status": status["status"], "expected_bin": bin_index, "detected_bin": detected_bin,
                 "p": float(structure[0, detected_bin]), "a": float(structure[1, detected_bin]), "u": float(structure[2, detected_bin]),
                 "finite": bool(np.isfinite(structure).all()), "atom_norm": status.get("atom_norm"),
                 "passed": bool(status["status"] == "ok" and structure[0, detected_bin] >= MIN_PERSISTENCE and
                                np.isfinite(structure).all() and np.isfinite(status.get("weight_min", np.nan)) and
                                np.isfinite(status.get("weight_max", np.nan)))}
    network = {}
    wave, original, na, labels = torch.randn(4, SAMPLE_COUNT), torch.rand(4, 3, PATH_BINS), torch.rand(4, 3, PATH_BINS), torch.tensor([0., 1., 0., 1.])
    for condition in CONDITIONS:
        model = model_for(condition)
        structure_input = torch.zeros_like(na) if condition == "waveform" else structure_for(condition, original, na)
        logits = model(wave, structure_input)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        network[condition] = {"shape": list(logits.shape), "loss": float(loss.detach()), "finite": bool(torch.isfinite(logits).all()),
                              "passed": bool(logits.shape == (4,) and torch.isfinite(loss))}
    result = {"synthetic_na_bcsc": synthetic, "network": network,
              "passed": bool(synthetic["passed"] and all(item["passed"] for item in network.values()))}
    dump(output["tests"] / "verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"v5验证失败：{result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "extract-structure", "train", "report", "verify"))
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-runs", type=int, default=2_000)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = Config(workers=arguments.workers, device=arguments.device, bootstrap_runs=arguments.bootstrap_runs)
    output = paths(config)
    dump(output["root"] / "run_config.json", {**asdict(config), "seed": SEED, "conditions": list(CONDITIONS),
                                                "primary_endpoint": "low_snr_[-15,0)_auroc", "pilot": True})
    if arguments.command == "verify":
        verify(config)
        return
    if arguments.command in ("all", "extract-structure"):
        print("[总体进度 0%] 验证v5 NA-BCSC与输入合同", flush=True)
        verify(config)
        load_original_structures(config)
        extract_structure(config, force=arguments.no_resume)
    if arguments.command in ("all", "train"):
        print("[总体进度 45%] 训练20个v5神经网络折", flush=True)
        train(config, resume=not arguments.no_resume)
    if arguments.command in ("all", "report"):
        print("[总体进度 90%] 汇总v5真实结果", flush=True)
        print(f"[总体进度100%] 完成：{report(config)}", flush=True)


if __name__ == "__main__":
    main()
