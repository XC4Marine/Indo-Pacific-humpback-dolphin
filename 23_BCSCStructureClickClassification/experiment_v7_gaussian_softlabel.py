#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v7：BCSC 高斯软标签辅助分类。

真实 BCSC P/A/U 结构仅构造训练期的 64 维时延—能量目标；分类器和导出
模型始终只接收一条 5761 点波形。命令：verify、prepare-labels、train、
report、template、all。``report`` 会拒绝不完整的 75 模型折结果。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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
BCSC_CACHE = ROOT / "23_BCSCStructureClickClassification" / "v2_low_snr_gated" / "cache"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v7_gaussian_softlabel_aux"
N_SAMPLES, WAVE_SAMPLES, PATH_BINS = 7000, 5761, 1133
SAMPLE_RATE, MIN_DELAY_SAMPLES, MAX_DELAY_SAMPLES = 576_000, 20, 1152
TARGET_BINS, SIGMA_BINS, TRUNCATION_SIGMA = 64, 1.0, 3.0
SEEDS = (20260826, 20260827, 20260828)
CONDITIONS = ("C0_waveform", "C1_hard64_gated", "C2_gaussian64_ungated",
              "C3_gaussian64_gated", "C4_shuffled_gaussian")
EPS = 1e-12
LOW_SNR_LABEL = "[-15,0) dB"
SNR_BANDS = ((-15., -10., "[-15,-10) dB"), (-10., -5., "[-10,-5) dB"),
             (-5., 0., "[-5,0) dB"), (0., 5., "[0,5) dB"),
             (5., 10., "[5,10) dB"), (10., 15.000001, "[10,15] dB"))


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SOURCE / "dataset_manifest.csv"
    waveform_path: Path = SOURCE / "cache" / "preprocessed_waveforms.npy"
    structure_path: Path = BCSC_CACHE / "structure_maps.npy"
    structure_done_path: Path = BCSC_CACHE / "structure_completed.npy"
    structure_status_path: Path = BCSC_CACHE / "structure_status.csv"
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    encoder_dropout: float = 0.1
    auxiliary_dropout: float = 0.2
    auxiliary_weight: float = 0.1
    bootstrap_replicates: int = 2000
    device: str = "auto"


def paths(config: Config) -> dict[str, Path]:
    names = ("models", "metrics", "predictions", "labels", "reports", "tests")
    result = {name: config.output_dir / name for name in names}
    result["root"] = config.output_dir
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def dump_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path): return str(item)
        if isinstance(item, (np.floating, float)): return float(item) if math.isfinite(float(item)) else None
        if isinstance(item, (np.integer, int, str, bool)) or item is None: return item
        if isinstance(item, dict): return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [convert(v) for v in item]
        return str(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state: torch.cuda.set_rng_state_all(state["cuda"])


def device_for(name: str) -> torch.device:
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA，但当前环境没有 CUDA")
    return result


def split(manifest: pd.DataFrame, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = manifest.test_fold.to_numpy(dtype=int)
    valid_fold = (test_fold + 1) % 5
    return (np.flatnonzero((folds != test_fold) & (folds != valid_fold)),
            np.flatnonzero(folds == valid_fold), np.flatnonzero(folds == test_fold))


def low_snr_mask(frame: pd.DataFrame) -> np.ndarray:
    labels, snr = frame.label.to_numpy(dtype=int), frame.target_snr_db.to_numpy(dtype=float)
    return (labels == 0) | ((labels == 1) & (snr >= -15.) & (snr < 0.))


def load_inputs(config: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    required = (config.manifest_path, config.waveform_path, config.structure_path,
                config.structure_done_path, config.structure_status_path)
    missing = [str(item) for item in required if not item.exists()]
    if missing: raise FileNotFoundError("缺少输入或 BCSC 缓存：" + "; ".join(missing))
    manifest = pd.read_csv(config.manifest_path).reset_index(drop=True)
    waves = np.load(config.waveform_path, mmap_mode="r")
    structure = np.load(config.structure_path, mmap_mode="r")
    done = np.load(config.structure_done_path)
    status = pd.read_csv(config.structure_status_path).sort_values("sample_index").reset_index(drop=True)
    checks = {
        "manifest_rows": len(manifest) == N_SAMPLES,
        "waves": tuple(waves.shape) == (N_SAMPLES, WAVE_SAMPLES),
        "structure": tuple(structure.shape) == (N_SAMPLES, 3, PATH_BINS),
        "completed": tuple(done.shape) == (N_SAMPLES,) and bool(done.all()),
        "sample_order": len(status) == N_SAMPLES and np.array_equal(status.sample_index.to_numpy(), np.arange(N_SAMPLES))
                        and np.array_equal(status.sample_id.astype(str).to_numpy(), manifest.sample_id.astype(str).to_numpy()),
        "groups": manifest.ori_file_num.nunique() == 34 and manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
        "folds": set(manifest.test_fold.unique()) == set(range(5)),
        "classes": manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
        "finite": bool(np.isfinite(waves).all() and np.isfinite(structure).all()),
    }
    if not all(checks.values()): raise RuntimeError(f"输入/分组合同失败：{checks}")
    return manifest, waves, structure, status


def delay_ms_from_source_bins() -> np.ndarray:
    return (np.arange(PATH_BINS, dtype=np.float64) + MIN_DELAY_SAMPLES) / SAMPLE_RATE * 1000.


def soft_label_cache(structure: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按连续时延做截断高斯沉积；hard 用同一连续位置的最近中心 bin。"""
    p = np.asarray(structure[:, 0], dtype=np.float64)
    a = np.asarray(structure[:, 1], dtype=np.float64)
    u = np.asarray(structure[:, 2], dtype=np.float64)
    weight = np.maximum(p * a * (1. - u), 0.)
    denominator = np.maximum((p * a).sum(axis=1), 0.)
    total = weight.sum(axis=1)
    q = np.clip(total / (denominator + EPS), 0., 1.).astype(np.float32)
    valid = (total > 1e-8)
    gaussian = np.zeros((N_SAMPLES, TARGET_BINS), dtype=np.float32)
    hard = np.zeros_like(gaussian)
    source_delay = delay_ms_from_source_bins()
    target_delay = np.linspace(source_delay[0], source_delay[-1], TARGET_BINS)
    coordinate = (source_delay - target_delay[0]) / (target_delay[-1] - target_delay[0]) * (TARGET_BINS - 1)
    radius = int(math.ceil(TRUNCATION_SIGMA * SIGMA_BINS))
    for sample in np.flatnonzero(valid):
        support = np.flatnonzero(weight[sample] > 0.)
        for source_bin in support:
            center = float(coordinate[source_bin]); w = float(weight[sample, source_bin])
            nearest = int(np.clip(np.rint(center), 0, TARGET_BINS - 1)); hard[sample, nearest] += w
            left, right = max(0, int(math.ceil(center - radius))), min(TARGET_BINS - 1, int(math.floor(center + radius)))
            bins = np.arange(left, right + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * ((bins - center) / SIGMA_BINS) ** 2)
            gaussian[sample, left:right + 1] += (w * kernel).astype(np.float32)
        gaussian[sample] /= gaussian[sample].sum()
        hard[sample] /= hard[sample].sum()
    return gaussian, hard, valid.astype(bool), q


def prepare_labels(config: Config) -> dict[str, Any]:
    result_paths = paths(config)
    manifest, _, structure, _ = load_inputs(config)
    gaussian, hard, valid, q = soft_label_cache(structure)
    label_path = result_paths["labels"] / "softlabel_cache.npz"
    np.savez_compressed(label_path, gaussian=gaussian, hard=hard, valid=valid, q=q,
                        sample_id=manifest.sample_id.to_numpy(dtype=str),
                        target_delay_ms=np.linspace(delay_ms_from_source_bins()[0], delay_ms_from_source_bins()[-1], TARGET_BINS))
    metadata = {"source_structure": str(config.structure_path), "shape": [N_SAMPLES, TARGET_BINS],
                "source_delay_ms": [float(delay_ms_from_source_bins()[0]), float(delay_ms_from_source_bins()[-1])],
                "includes_direct_path_tau_0": False, "weight": "P*A*(1-U)", "sigma_bins": SIGMA_BINS,
                "truncation_sigma": TRUNCATION_SIGMA, "valid_rule": "sum_j(P*A*(1-U)) > 1e-8",
                "valid_count": int(valid.sum()), "invalid_count": int((~valid).sum()),
                "q_range": [float(q.min()), float(q.max())], "normalization": "each valid H sums to 1; invalid H is all zero"}
    dump_json(result_paths["labels"] / "softlabel_metadata.json", metadata)
    return metadata


def load_labels(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_path = paths(config)["labels"] / "softlabel_cache.npz"
    if not label_path.exists(): prepare_labels(config)
    with np.load(label_path) as value:
        gaussian, hard = value["gaussian"], value["hard"]
        valid, q = value["valid"].astype(bool), value["q"].astype(np.float32)
    checks = (gaussian.shape == (N_SAMPLES, TARGET_BINS) and hard.shape == gaussian.shape and valid.shape == (N_SAMPLES,)
              and q.shape == (N_SAMPLES,) and np.isfinite(gaussian).all() and np.isfinite(hard).all()
              and np.allclose(gaussian[valid].sum(axis=1), 1., atol=1e-6) and np.allclose(hard[valid].sum(axis=1), 1., atol=1e-6)
              and np.all(gaussian[~valid] == 0.) and np.all(hard[~valid] == 0.) and np.all((q >= 0.) & (q <= 1.)))
    if not checks: raise RuntimeError("64维软标签缓存合同失败")
    return gaussian, hard, valid, q


class ClickDataset(Dataset):
    def __init__(self, waves: np.ndarray, labels: np.ndarray, target: np.ndarray, valid: np.ndarray,
                 q: np.ndarray, indices: np.ndarray) -> None:
        self.waves, self.labels, self.target, self.valid, self.q, self.indices = waves, labels, target, valid, q, indices
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[row])
        return (torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32),
                torch.as_tensor(self.target[index], dtype=torch.float32), torch.tensor(self.valid[index], dtype=torch.bool),
                torch.tensor(self.q[index], dtype=torch.float32), index)


def loader(waves: np.ndarray, labels: np.ndarray, target: np.ndarray, valid: np.ndarray, q: np.ndarray,
           indices: np.ndarray, batch_size: int, shuffle: bool, seed: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(ClickDataset(waves, labels, target, valid, q, indices), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0, pin_memory=torch.cuda.is_available())


class WaveformEncoder(nn.Module):
    """固定 v6 的 64维单层 Waveform Transformer 编码器。"""
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv1d(1, 64, kernel_size=32, stride=16)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 64)); nn.init.normal_(self.cls_token, std=.02)
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(wave.unsqueeze(1)).transpose(1, 2)
        pos = torch.arange(tokens.shape[1], device=wave.device, dtype=tokens.dtype).unsqueeze(-1)
        dim = torch.arange(64, device=wave.device, dtype=tokens.dtype).unsqueeze(0)
        scale = torch.pow(torch.tensor(10000., device=wave.device, dtype=tokens.dtype), 2 * torch.floor(dim / 2) / 64)
        angle = pos / scale; sinusoid = torch.where((dim.long() % 2) == 0, torch.sin(angle), torch.cos(angle)).unsqueeze(0)
        tokens = torch.cat((self.cls_token.expand(wave.shape[0], -1, -1), tokens + sinusoid), dim=1)
        return self.encoder(tokens)[:, 0]


class GaussianAuxClassifier(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.waveform = WaveformEncoder(config.encoder_dropout)
        self.auxiliary = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128), nn.GELU(), nn.Dropout(config.auxiliary_dropout), nn.Linear(128, 64))
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.waveform(wave)
        # classifier strictly consumes z; the BCSC-derived distribution is never concatenated or read here.
        return self.classifier(z).squeeze(1), torch.softmax(self.auxiliary(z), dim=1)


class WaveformDeploymentModel(nn.Module):
    """部署模型物理删除辅助头，只保留 waveform -> classifier。"""
    def __init__(self, config: Config) -> None:
        super().__init__(); self.waveform = WaveformEncoder(config.encoder_dropout); self.classifier = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, wave: torch.Tensor) -> torch.Tensor: return self.classifier(self.waveform(wave)).squeeze(1)


def path_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kl = (target * (torch.log(target.clamp_min(EPS)) - torch.log(prediction.clamp_min(EPS)))).sum(dim=1) / math.log(TARGET_BINS)
    cdf = torch.abs(torch.cumsum(target, dim=1) - torch.cumsum(prediction, dim=1)).mean(dim=1)
    return .5 * kl + .5 * cdf


def auxiliary_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, q: torch.Tensor, gated: bool) -> torch.Tensor:
    weights = valid.float() * (q if gated else 1.)
    return (weights * path_loss(prediction, target)).sum() / (weights.sum() + EPS)


def condition_target(condition: str, gaussian: np.ndarray, hard: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    if condition == "C0_waveform": return gaussian, False, False
    if condition == "C1_hard64_gated": return hard, True, True
    if condition == "C2_gaussian64_ungated": return gaussian, True, False
    if condition in ("C3_gaussian64_gated", "C4_shuffled_gaussian"): return gaussian, True, True
    raise ValueError(condition)


def shuffled_targets(target: np.ndarray, valid: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, seed: int, fold: int) -> np.ndarray:
    result = target.copy(); rng = np.random.default_rng(seed + 10000 + fold)
    for label_value in (0, 1):
        group = train_indices[(labels[train_indices] == label_value) & valid[train_indices]]
        result[group] = target[rng.permutation(group)]
    return result


def sigmoid(value: np.ndarray) -> np.ndarray: return 1. / (1. + np.exp(-np.clip(value, -80., 80.)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temp: float) -> float:
        temperature = math.exp(log_temp); return float(np.mean(np.logaddexp(0., logits / temperature) - labels * logits / temperature))
    fit = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
    return float(math.exp(fit.x)) if fit.success else 1.


def threshold_for(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0., .5, 1.], probabilities)))
    scores = [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=(0, 1)).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)), "auprc": float(average_precision_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "recall": float(recall_score(labels, predicted, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)), "brier": float(brier_score_loss(labels, probabilities)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


@torch.no_grad()
def predict(model: GaussianAuxClassifier, data: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval(); result: dict[str, list[np.ndarray]] = {key: [] for key in ("sample_index", "label", "logit", "target", "valid", "q", "predicted_h")}
    for wave, label, target, valid, q, index in data:
        logits, predicted = model(wave.to(device, non_blocking=True))
        result["sample_index"].append(np.asarray(index)); result["label"].append(label.numpy()); result["logit"].append(logits.cpu().numpy())
        result["target"].append(target.numpy()); result["valid"].append(valid.numpy()); result["q"].append(q.numpy()); result["predicted_h"].append(predicted.cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def weighted_aux_scores(truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray, q: np.ndarray,
                        constant: np.ndarray) -> dict[str, float | int]:
    usable = valid.astype(bool); weights = q[usable].astype(np.float64)
    if not len(weights) or weights.sum() <= EPS: return {"n_valid": int(usable.sum()), "weight_sum": 0., "weighted_emd": math.nan, "weighted_js": math.nan, "weighted_cosine": math.nan}
    true, pred = truth[usable].astype(np.float64), prediction[usable].astype(np.float64)
    def calculate(value: np.ndarray) -> tuple[float, float, float]:
        emd = np.abs(np.cumsum(true, axis=1) - np.cumsum(value, axis=1)).mean(axis=1)
        midpoint = .5 * (true + value)
        js = .5 * ((true * (np.log(true + EPS) - np.log(midpoint + EPS))).sum(axis=1) + (value * (np.log(value + EPS) - np.log(midpoint + EPS))).sum(axis=1))
        cosine = (true * value).sum(axis=1) / (np.linalg.norm(true, axis=1) * np.linalg.norm(value, axis=1) + EPS)
        return tuple(float(np.average(item, weights=weights)) for item in (emd, js, cosine))
    emd, js, cosine = calculate(pred)
    base_emd, base_js, base_cosine = calculate(np.broadcast_to(constant, true.shape))
    return {"n_valid": int(usable.sum()), "weight_sum": float(weights.sum()), "weighted_emd": emd, "weighted_js": js, "weighted_cosine": cosine,
            "constant_weighted_emd": base_emd, "constant_weighted_js": base_js, "constant_weighted_cosine": base_cosine,
            "emd_improvement": base_emd - emd, "js_improvement": base_js - js, "cosine_improvement": cosine - base_cosine}


def q_stratified_scores(truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray, q: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for low, high, name in ((0., 1/3, "[0,1/3)"), (1/3, 2/3, "[1/3,2/3)"), (2/3, 1.000001, "[2/3,1]")):
        use = valid & (q >= low) & (q < high)
        if not use.any(): rows.append({"q_band": name, "n": 0, "emd": math.nan, "js": math.nan, "cosine": math.nan}); continue
        rows.append({"q_band": name, "n": int(use.sum()),
                     "emd": float(np.abs(np.cumsum(truth[use], 1) - np.cumsum(prediction[use], 1)).mean()),
                     "js": float(.5 * ((truth[use] * (np.log(truth[use] + EPS) - np.log(.5 * (truth[use] + prediction[use]) + EPS))).sum(1) + (prediction[use] * (np.log(prediction[use] + EPS) - np.log(.5 * (truth[use] + prediction[use]) + EPS))).sum(1)).mean()),
                     "cosine": float(np.mean((truth[use] * prediction[use]).sum(1) / (np.linalg.norm(truth[use], axis=1) * np.linalg.norm(prediction[use], axis=1) + EPS)))})
    return rows


def model_directory(config: Config, condition: str, seed: int, fold: int) -> Path:
    destination = paths(config)["models"] / condition / f"seed_{seed}" / f"fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True); return destination


def train_one(config: Config, condition: str, seed: int, fold: int, waves: np.ndarray, manifest: pd.DataFrame,
              gaussian: np.ndarray, hard: np.ndarray, valid: np.ndarray, q: np.ndarray, resume: bool,
              progress: callable) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    destination = model_directory(config, condition, seed, fold)
    complete, best_path, latest_path = destination / "COMPLETE.json", destination / "best_checkpoint.pt", destination / "latest_checkpoint.pt"
    prediction_path, aux_path, history_path = destination / "test_predictions.csv", destination / "auxiliary_metrics.csv", destination / "history.csv"
    if resume and all(item.exists() for item in (complete, best_path, prediction_path, aux_path, history_path)):
        return json.loads(complete.read_text(encoding="utf-8")), pd.read_csv(prediction_path), pd.read_csv(aux_path), pd.read_csv(history_path)
    labels = manifest.label.to_numpy(dtype=np.float32); train_idx, validation_idx, test_idx = split(manifest, fold)
    target, uses_aux, gated = condition_target(condition, gaussian, hard)
    train_target = shuffled_targets(target, valid, labels.astype(int), train_idx, seed, fold) if condition == "C4_shuffled_gaussian" else target
    initialization_seed = seed + fold * 100
    seed_all(initialization_seed); device = device_for(config.device); model = GaussianAuxClassifier(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_score, stale, start_epoch, history, started = -np.inf, 0, 1, [], time.perf_counter()
    if resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False); model.load_state_dict(payload["state_dict"]); optimizer.load_state_dict(payload["optimizer"])
        best_score, stale, start_epoch, history, started = payload["best_score"], payload["stale"], payload["epoch"] + 1, payload["history"], time.perf_counter() - payload["elapsed_seconds"]
        restore_rng_state(payload["rng_state"])
    validation_loader = loader(waves, labels, target, valid, q, validation_idx, config.batch_size * 2, False)
    test_loader = loader(waves, labels, target, valid, q, test_idx, config.batch_size * 2, False)
    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train(); losses: list[float] = []; cls_losses: list[float] = []; aux_losses: list[float] = []
        training_loader = loader(waves, labels, train_target, valid, q, train_idx, config.batch_size, True, initialization_seed + 1000 + epoch)
        for wave, label, target_batch, valid_batch, q_batch, _ in training_loader:
            optimizer.zero_grad(set_to_none=True); logits, predicted = model(wave.to(device, non_blocking=True))
            cls = nn.functional.binary_cross_entropy_with_logits(logits, label.to(device, non_blocking=True))
            aux = auxiliary_loss(predicted, target_batch.to(device), valid_batch.to(device), q_batch.to(device), gated) if uses_aux else logits.sum() * 0.
            loss = cls + config.auxiliary_weight * aux
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"{condition}/seed={seed}/fold={fold}/epoch={epoch}: 非有限损失")
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None): raise FloatingPointError("出现非有限梯度")
            optimizer.step(); losses.append(float(loss.detach().cpu())); cls_losses.append(float(cls.detach().cpu())); aux_losses.append(float(aux.detach().cpu()))
        validation = predict(model, validation_loader, device); validation_frame = manifest.iloc[validation["sample_index"]].reset_index(drop=True)
        score = float(roc_auc_score(validation["label"][low_snr_mask(validation_frame)], validation["logit"][low_snr_mask(validation_frame)]))
        record = {"condition": condition, "seed": seed, "test_fold": fold, "epoch": epoch, "train_loss": float(np.mean(losses)),
                  "train_bce": float(np.mean(cls_losses)), "train_auxiliary": float(np.mean(aux_losses)), "validation_low_snr_auroc": score,
                  "elapsed_seconds": time.perf_counter() - started}
        history.append(record); progress(epoch, score, record["elapsed_seconds"])
        if score > best_score + 1e-7:
            best_score, stale = score, 0
            torch.save({"state_dict": model.state_dict(), "condition": condition, "seed": seed, "fold": fold, "best_score": best_score,
                        "epoch": epoch,
                        "architecture": "WaveformTransformer64; classifier(z); LayerNorm64->Linear128->GELU->Dropout0.2->Linear64->Softmax"}, best_path)
        else: stale += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_score": best_score, "stale": stale,
                    "epoch": epoch, "history": history, "elapsed_seconds": time.perf_counter() - started, "rng_state": rng_state()}, latest_path)
        if stale >= config.patience: break
    if not best_path.exists(): raise RuntimeError("没有生成最佳检查点")
    payload = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(payload["state_dict"])
    deployment = WaveformDeploymentModel(config).to(device).eval()
    deployment.waveform.load_state_dict(model.waveform.state_dict()); deployment.classifier.load_state_dict(model.classifier.state_dict())
    with torch.no_grad():
        check_wave = torch.as_tensor(np.asarray(waves[test_idx[:min(4, len(test_idx))]]), dtype=torch.float32, device=device)
        if not torch.allclose(model.eval()(check_wave)[0], deployment(check_wave), atol=1e-6, rtol=0.): raise RuntimeError("部署模型移除辅助头后分类logit发生变化")
    torch.save({"state_dict": deployment.state_dict(), "input": "waveform [batch, 5761] only", "removed_module": "auxiliary",
                "condition": condition, "seed": seed, "test_fold": fold}, destination / "deployment_waveform_only.pt")
    validation = predict(model, validation_loader, device); temperature = fit_temperature(validation["logit"], validation["label"])
    threshold = threshold_for(validation["label"].astype(int), sigmoid(validation["logit"] / temperature))
    test = predict(model, test_loader, device); probability = sigmoid(test["logit"] / temperature); test_frame = manifest.iloc[test["sample_index"]].reset_index(drop=True)
    result = metrics(test["label"].astype(int), probability, threshold); low = low_snr_mask(test_frame)
    result.update({"low_snr_auroc": float(roc_auc_score(test["label"][low], probability[low])), "condition": condition, "seed": seed, "test_fold": fold,
                   "validation_fold": (fold + 1) % 5, "best_validation_low_snr_auroc": best_score, "best_epoch": int(payload.get("epoch", 0)),
                   "temperature": temperature, "threshold": threshold, "parameters": int(sum(p.numel() for p in model.parameters())), "training_seconds": time.perf_counter() - started})
    for low_bound, high_bound, name in SNR_BANDS:
        band = (test_frame.label.to_numpy(dtype=int) == 0) | ((test_frame.label.to_numpy(dtype=int) == 1) & (test_frame.target_snr_db.to_numpy(float) >= low_bound) & (test_frame.target_snr_db.to_numpy(float) < high_bound))
        result[f"snr_{name}_auroc"] = float(roc_auc_score(test["label"][band], probability[band]))
    for noise_type in ("Type_3_Multipath_Noise", "Type_4_Isolated_Noise"):
        result[f"false_positive_{noise_type}"] = int(((test_frame.noise_type.to_numpy() == noise_type) & (probability >= threshold)).sum())
    train_weights = valid[train_idx].astype(np.float64) * q[train_idx]; constant = np.average(target[train_idx], axis=0, weights=train_weights)
    aux_result = weighted_aux_scores(test["target"], test["predicted_h"], test["valid"].astype(bool), test["q"], constant)
    aux_result.update({"condition": condition, "seed": seed, "test_fold": fold, "target_type": "hard" if condition == "C1_hard64_gated" else "gaussian", "auxiliary_loss_enabled": uses_aux})
    prediction_frame = test_frame[["sample_id", "ori_file_num", "test_fold", "target_snr_db", "noise_type"]].copy(); prediction_frame.insert(0, "sample_index", test["sample_index"])
    prediction_frame["label"] = test["label"].astype(int); prediction_frame["logit"] = test["logit"]; prediction_frame["p_click"] = probability
    prediction_frame["predicted_label"] = (probability >= threshold).astype(int); prediction_frame["condition"] = condition; prediction_frame["seed"] = seed; prediction_frame["temperature"] = temperature; prediction_frame["threshold"] = threshold
    prediction_frame["m"] = test["valid"].astype(int); prediction_frame["q"] = test["q"]
    for dimension in range(TARGET_BINS): prediction_frame[f"pred_h_{dimension:02d}"] = test["predicted_h"][:, dimension]
    q_rows = pd.DataFrame(q_stratified_scores(test["target"], test["predicted_h"], test["valid"].astype(bool), test["q"])); q_rows["condition"] = condition; q_rows["seed"] = seed; q_rows["test_fold"] = fold
    aux_frame = pd.concat((pd.DataFrame([aux_result]), q_rows), ignore_index=True, sort=False)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig"); prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig"); aux_frame.to_csv(aux_path, index=False, encoding="utf-8-sig")
    dump_json(complete, result); return result, prediction_frame, aux_frame, pd.DataFrame(history)


def train(config: Config, resume: bool, active_conditions: tuple[str, ...] = CONDITIONS,
          active_seeds: tuple[int, ...] = SEEDS, active_folds: tuple[int, ...] = tuple(range(5))) -> None:
    result_paths = paths(config); manifest, waves, _, _ = load_inputs(config); gaussian, hard, valid, q = load_labels(config)
    jobs = [(condition, seed, fold) for condition in active_conditions for seed in active_seeds for fold in active_folds]
    total_jobs = len(jobs)
    if not total_jobs: raise ValueError("至少需要一个条件、随机种子和测试折")
    started, durations = time.perf_counter(), []
    rows: list[dict[str, Any]] = []; predictions: list[pd.DataFrame] = []; auxiliary: list[pd.DataFrame] = []; histories: list[pd.DataFrame] = []
    bar = tqdm(total=len(jobs), initial=0, desc="总体进度：v7 Gaussian soft-label", unit="模型折")
    for job_number, (condition, seed, fold) in enumerate(jobs, 1):
        def progress(epoch: int, score: float, elapsed: float, c=condition, s=seed, f=fold, n=job_number) -> None:
            overall_completed = sum((model_directory(config, jc, js, jf) / "COMPLETE.json").exists() for jc, js, jf in jobs)
            mean_seconds = np.mean(durations) if durations else elapsed / max(epoch, 1) * config.max_epochs
            eta = max(0., (len(jobs) - overall_completed - 1) * mean_seconds)
            print(f"[总体进度 {100*overall_completed/total_jobs:.1f}% | {overall_completed}/{total_jobs}] {c} seed={s} fold={f} epoch={epoch}/{config.max_epochs} val-low-AUROC={score:.4f} ETA≈{eta/60:.1f}min", flush=True)
        begin = time.perf_counter(); row, prediction, aux, history = train_one(config, condition, seed, fold, waves, manifest, gaussian, hard, valid, q, resume, progress)
        durations.append(time.perf_counter() - begin); rows.append(row); predictions.append(prediction); auxiliary.append(aux); histories.append(history); bar.update(1)
    bar.close(); metric_frame = pd.DataFrame(rows).sort_values(["condition", "seed", "test_fold"]); prediction_frame = pd.concat(predictions, ignore_index=True); auxiliary_frame = pd.concat(auxiliary, ignore_index=True); history_frame = pd.concat(histories, ignore_index=True)
    expected = len(active_conditions) * len(active_seeds) * sum(len(split(manifest, fold)[2]) for fold in active_folds)
    if len(metric_frame) != total_jobs or len(prediction_frame) != expected or prediction_frame.duplicated(["condition", "seed", "sample_index"]).any(): raise RuntimeError("训练汇总合同失败")
    metric_frame.to_csv(result_paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig"); prediction_frame.to_csv(result_paths["predictions"] / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    auxiliary_frame.to_csv(result_paths["metrics"] / "auxiliary_metrics.csv", index=False, encoding="utf-8-sig"); history_frame.to_csv(result_paths["metrics"] / "epoch_history.csv", index=False, encoding="utf-8-sig")
    dump_json(result_paths["metrics"] / "model_completion_status.json", {"expected_model_folds": total_jobs, "completed_model_folds": total_jobs,
              "conditions": active_conditions, "seeds": active_seeds, "test_folds": active_folds, "elapsed_seconds": time.perf_counter()-started})
    print(f"[总体进度 100.0% | {total_jobs}/{total_jobs}] 所有请求模型折训练完成", flush=True)


def require_complete(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_path, prediction_path, auxiliary_path = paths(config)["metrics"] / "fold_metrics.csv", paths(config)["predictions"] / "oof_predictions.csv", paths(config)["metrics"] / "auxiliary_metrics.csv"
    if not all(item.exists() for item in (metric_path, prediction_path, auxiliary_path)): raise RuntimeError("正式汇总拒绝：缺少训练产物")
    metric_frame, prediction_frame, auxiliary_frame = pd.read_csv(metric_path), pd.read_csv(prediction_path), pd.read_csv(auxiliary_path)
    expected = len(CONDITIONS) * len(SEEDS) * N_SAMPLES
    if len(metric_frame) != 75 or len(prediction_frame) != expected or prediction_frame.duplicated(["condition", "seed", "sample_index"]).any() or set(prediction_frame.seed.unique()) != set(SEEDS):
        raise RuntimeError("正式汇总拒绝：模型折、折外预测或随机种子不完整/重复")
    for condition in CONDITIONS:
        for seed in SEEDS:
            part = prediction_frame[(prediction_frame.condition == condition) & (prediction_frame.seed == seed)]
            if len(part) != N_SAMPLES or part.sample_index.nunique() != N_SAMPLES: raise RuntimeError("正式汇总拒绝：每个条件/种子必须恰有7000条折外预测")
    return metric_frame, prediction_frame, auxiliary_frame


def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "sample_index"]
    fixed = ["sample_id", "ori_file_num", "test_fold", "target_snr_db", "noise_type", "label"]
    aggregate = predictions.groupby(keys, as_index=False).agg({**{key: "first" for key in fixed}, "p_click": "mean"})
    if len(aggregate) != len(CONDITIONS) * N_SAMPLES: raise RuntimeError("三种子聚合行数异常")
    return aggregate


def primary_auroc(frame: pd.DataFrame) -> float:
    use = (frame.label.to_numpy(int) == 0) | ((frame.label.to_numpy(int) == 1) & (frame.target_snr_db.to_numpy(float) >= -15.) & (frame.target_snr_db.to_numpy(float) < 0.))
    return float(roc_auc_score(frame.label.to_numpy(int)[use], frame.p_click.to_numpy(float)[use]))


def paired_bootstrap(aggregate: pd.DataFrame, comparisons: Iterable[tuple[str, str]], runs: int) -> pd.DataFrame:
    groups = np.sort(aggregate.ori_file_num.unique()); rng = np.random.default_rng(20260915); rows = []
    by_condition = {condition: aggregate[aggregate.condition == condition].set_index("sample_index") for condition in CONDITIONS}
    for replicate in range(runs):
        chosen = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_condition["C0_waveform"].index[by_condition["C0_waveform"].ori_file_num.eq(group)].to_numpy() for group in chosen])
        scores = {condition: primary_auroc(part.loc[indices].reset_index()) for condition, part in by_condition.items()}
        for left, right in comparisons: rows.append({"replicate": replicate, "comparison": f"{left}_minus_{right}", "auroc_difference": scores[left] - scores[right]})
    return pd.DataFrame(rows)


def snr_metrics(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        part = aggregate[aggregate.condition == condition]
        for low, high, name in SNR_BANDS:
            use = (part.label.to_numpy(int) == 0) | ((part.label.to_numpy(int) == 1) & (part.target_snr_db.to_numpy(float) >= low) & (part.target_snr_db.to_numpy(float) < high))
            rows.append({"condition": condition, "snr_band": name, "auroc": float(roc_auc_score(part.label.to_numpy(int)[use], part.p_click.to_numpy(float)[use])), "n": int(use.sum())})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    columns = [str(item) for item in frame.columns]
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)): return f"{value:.{digits}f}" if math.isfinite(float(value)) else "NA"
        return str(value)
    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def report(config: Config) -> Path:
    metrics_frame, predictions, auxiliary = require_complete(config); aggregate = aggregate_predictions(predictions)
    comparisons = (("C3_gaussian64_gated", "C0_waveform"), ("C3_gaussian64_gated", "C4_shuffled_gaussian"),
                   ("C3_gaussian64_gated", "C1_hard64_gated"), ("C3_gaussian64_gated", "C2_gaussian64_ungated"))
    bootstrap = paired_bootstrap(aggregate, comparisons, config.bootstrap_replicates); summary_rows = []
    for name, part in bootstrap.groupby("comparison"):
        summary_rows.append({"comparison": name, "mean_difference": float(part.auroc_difference.mean()), "ci95_low": float(part.auroc_difference.quantile(.025)), "ci95_high": float(part.auroc_difference.quantile(.975)), "significant_positive": bool(part.auroc_difference.quantile(.025) > 0.)})
    bootstrap_summary = pd.DataFrame(summary_rows); result_paths = paths(config); bootstrap.to_csv(result_paths["metrics"] / "paired_bootstrap_2000.csv", index=False, encoding="utf-8-sig"); bootstrap_summary.to_csv(result_paths["metrics"] / "paired_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(result_paths["predictions"] / "oof_seed_mean_predictions.csv", index=False, encoding="utf-8-sig")
    bands = snr_metrics(aggregate); bands.to_csv(result_paths["metrics"] / "snr_stratified_auroc.csv", index=False, encoding="utf-8-sig")
    efficiency = metrics_frame.groupby("condition", as_index=False).agg(parameters=("parameters", "mean"), training_seconds_per_fold=("training_seconds", "mean"), best_epoch=("best_epoch", "mean")); efficiency["deployment_parameters"] = int(sum(p.numel() for p in WaveformDeploymentModel(config).parameters())); efficiency.to_csv(result_paths["metrics"] / "efficiency.csv", index=False, encoding="utf-8-sig")
    c3_aux = auxiliary[(auxiliary.condition == "C3_gaussian64_gated") & auxiliary.q_band.isna()].copy(); learned = bool((c3_aux.emd_improvement > 0).all() and (c3_aux.js_improvement > 0).all() and (c3_aux.cosine_improvement > 0).all())
    success = bool(bootstrap_summary.loc[bootstrap_summary.comparison.eq("C3_gaussian64_gated_minus_C0_waveform"), "significant_positive"].iloc[0]); mechanism = bool(success and bootstrap_summary.loc[bootstrap_summary.comparison.eq("C3_gaussian64_gated_minus_C4_shuffled_gaussian"), "significant_positive"].iloc[0])
    text = ["# 实验23-v7：BCSC 高斯软标签辅助分类", "", "本报告仅在75个模型折、每条样本三个种子折外预测均完整时生成。分类推理仅输入波形；BCSC标签只作为训练期辅助目标。", "", "## 主终点与配对 Bootstrap", "", markdown_table(bootstrap_summary), "", f"- 主成功条件（C3−C0 CI下限>0）：`{success}`。", f"- 强物理机制条件（另有C3−C4 CI下限>0）：`{mechanism}`。", f"- 辅助分布是否三项均优于训练折常数预测器：`{learned}`。若为否，不得主张学到样本级BCSC结构。", "", "## 逐折分类指标", "", markdown_table(metrics_frame), "", "## SNR分层 AUROC", "", markdown_table(bands), "", "## 辅助预测指标与q分层", "", markdown_table(auxiliary), "", "## 训练开销与部署参数量", "", markdown_table(efficiency), "", "两类既有负样本的误报数已作为逐折指标的 `false_positive_Type_3_Multipath_Noise` 与 `false_positive_Type_4_Isolated_Noise` 字段保存。"]
    destination = result_paths["reports"] / "research_report_zh.md"; destination.write_text("\n".join(text) + "\n", encoding="utf-8"); return destination


def template(config: Config) -> Path:
    destination = paths(config)["reports"] / "research_report_zh_TBD.md"
    text = """# 实验23-v7：BCSC 高斯软标签辅助分类（待实际训练）

本文件不包含任何推测性结果。完成75个模型折训练、三种子折外预测平均及34个录音组的2,000次配对Bootstrap后，使用 `report` 生成正式报告。

| 项目 | 真实值 |
| --- | --- |
| C3−C0 低SNR AUROC差值及95% CI | TBD |
| C3−C4 低SNR AUROC差值及95% CI | TBD |
| C3−C1 平滑贡献 | TBD |
| C3−C2 门控贡献 | TBD |
| 加权 EMD/JS/余弦相似度与常数基线比较 | TBD |
| 六个SNR区间 AUROC | TBD |
| 两类负样本误报数 | TBD |
| 训练开销与部署参数量 | TBD |
"""
    destination.write_text(text, encoding="utf-8"); return destination


def verify(config: Config) -> dict[str, Any]:
    result_paths = paths(config); manifest, waves, structure, _ = load_inputs(config); gaussian, hard, valid, q = load_labels(config); device = device_for(config.device)
    target_delay = np.linspace(delay_ms_from_source_bins()[0], delay_ms_from_source_bins()[-1], TARGET_BINS); center = 31
    synthetic = np.zeros((N_SAMPLES, 3, PATH_BINS), dtype=np.float32); source_coordinate = int(round((target_delay[center] * 1e-3 * SAMPLE_RATE) - MIN_DELAY_SAMPLES)); synthetic[0, 0, source_coordinate] = 1.; synthetic[0, 1, source_coordinate] = 1.; synthetic[0, 2, source_coordinate] = 0.
    test_gaussian, test_hard, test_valid, _ = soft_label_cache(synthetic); centroid = float((np.arange(TARGET_BINS) * test_gaussian[0]).sum())
    expected_coordinate = float((delay_ms_from_source_bins()[source_coordinate] - target_delay[0]) / (target_delay[-1] - target_delay[0]) * (TARGET_BINS - 1))
    network = {}; wave = torch.as_tensor(np.asarray(waves[:4]), dtype=torch.float32, device=device); label = torch.as_tensor(manifest.label.iloc[:4].to_numpy(np.float32), device=device)
    for condition in CONDITIONS:
        seed_all(20260826); model = GaussianAuxClassifier(config).to(device).train(); logits, predicted = model(wave); target, uses_aux, gated = condition_target(condition, gaussian, hard)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, label) + (config.auxiliary_weight * auxiliary_loss(predicted, torch.as_tensor(target[:4], device=device), torch.as_tensor(valid[:4], device=device), torch.as_tensor(q[:4], device=device), gated) if uses_aux else 0.)
        loss.backward(); network[condition] = {"logits": list(logits.shape), "h": list(predicted.shape), "parameters": int(sum(p.numel() for p in model.parameters())), "finite": bool(torch.isfinite(loss))}
    seed_all(1); full = GaussianAuxClassifier(config).to(device).eval(); deploy = WaveformDeploymentModel(config).to(device).eval(); deploy.waveform.load_state_dict(full.waveform.state_dict()); deploy.classifier.load_state_dict(full.classifier.state_dict())
    with torch.no_grad(): full_logits, _ = full(wave); deployment_matches = bool(torch.allclose(full_logits, deploy(wave), atol=1e-6, rtol=0.))
    train_idx, _, _ = split(manifest, 0); shuffled = shuffled_targets(gaussian, valid, manifest.label.to_numpy(int), train_idx, SEEDS[0], 0); shuffled_again = shuffled_targets(gaussian, valid, manifest.label.to_numpy(int), train_idx, SEEDS[0], 0)
    c4_classes = all(np.array_equal(np.sort(shuffled[group], axis=0), np.sort(gaussian[group], axis=0))
                     for label_value in (0, 1) for group in [train_idx[(manifest.label.to_numpy(int)[train_idx] == label_value) & valid[train_idx]]])
    c4_ok = bool(np.array_equal(shuffled, shuffled_again) and np.array_equal(shuffled[~np.isin(np.arange(N_SAMPLES), train_idx)], gaussian[~np.isin(np.arange(N_SAMPLES), train_idx)]) and c4_classes)
    result = {"input_shapes": {"waves": list(waves.shape), "bcsc": list(structure.shape), "labels": list(gaussian.shape)}, "finite_and_group_contract": True,
              "label_normalization": bool(np.allclose(gaussian[valid].sum(1), 1., atol=1e-6) and np.all(gaussian[~valid] == 0.) and np.all((q >= 0.) & (q <= 1.))),
              "single_path_expected_hard_bin": int(np.rint(expected_coordinate)), "single_path_gaussian_centroid": centroid, "single_path_expected_continuous_coordinate": expected_coordinate,
              "single_path_mapping_passed": bool(test_valid[0] and np.argmax(test_hard[0]) == int(np.rint(expected_coordinate)) and abs(centroid-expected_coordinate) < .02),
              "c4_training_only_reproducible": c4_ok, "same_parameter_count": len({x["parameters"] for x in network.values()}) == 1,
              "classification_logits_ignore_real_bcsc": True, "deployment_without_auxiliary_matches": deployment_matches, "networks": network}
    result["passed"] = bool(result["label_normalization"] and result["single_path_mapping_passed"] and result["c4_training_only_reproducible"] and result["same_parameter_count"] and result["deployment_without_auxiliary_matches"] and all(x["finite"] and x["logits"] == [4] and x["h"] == [4,64] for x in network.values()))
    dump_json(result_paths["tests"] / "verification.json", result)
    if not result["passed"]: raise RuntimeError(f"v7验证失败：{result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("verify", "prepare-labels", "train", "report", "template", "all")); parser.add_argument("--device", default="auto"); parser.add_argument("--output-dir", type=Path); parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, help="受限训练的条件子集；省略时使用全部五个条件")
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, help="受限训练的随机种子子集；省略时使用全部三个种子")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), help="受限训练的测试折子集；省略时使用全部五折")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args(); config = Config(output_dir=arguments.output_dir or OUT, device=arguments.device); result_paths = paths(config)
    active_conditions, active_seeds, active_folds = tuple(arguments.conditions or CONDITIONS), tuple(arguments.seeds or SEEDS), tuple(arguments.folds or range(5))
    dump_json(result_paths["root"] / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir), "conditions": active_conditions, "seeds": active_seeds, "test_folds": active_folds, "fold_protocol": "test=f; validation=(f+1) mod 5; train=other 3 folds", "primary_endpoint": "[-15,0) dB positives vs all same-fold negatives AUROC", "classification_contract": "classifier reads waveform feature z only; auxiliary head is training-only and removed at deployment"})
    if arguments.command in ("verify", "all"): print("[总体进度 0% | 0/75] 验证输入、64维标签和前后向合同", flush=True); verify(config)
    if arguments.command in ("prepare-labels", "all"): prepare_labels(config)
    if arguments.command in ("template", "all"): template(config)
    if arguments.command in ("train", "all"): train(config, resume=not arguments.no_resume, active_conditions=active_conditions, active_seeds=active_seeds, active_folds=active_folds)
    if arguments.command in ("report", "all"): print(f"正式报告：{report(config)}", flush=True)


if __name__ == "__main__": main()
