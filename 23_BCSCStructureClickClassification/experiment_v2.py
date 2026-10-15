#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v2：低SNR BCSC可靠性门控融合。

仅读取实验21固定低SNR混合缓存；命令：all、extract-structure、train、report、verify。
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


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
V1_PATH = ROOT / "23_BCSCStructureClickClassification" / "experiment.py"
SNR_ROOT = ROOT / "21_PhysicsInformedClickClassification" / "outputs_snr_m15_p15_full_5seeds"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v2_low_snr_gated"
SAMPLE_COUNT, SAMPLE_RATE, PATH_BINS = 5761, 576000, 1133
SEED = 20260826
NEURAL = ("waveform", "structure_only", "direct_fusion", "gated_fusion", "gated_zero_structure")
CLASSIC = ("teager", "bcsc_summary")
CONDITIONS = CLASSIC + NEURAL
LABELS = {"teager": "Teager", "bcsc_summary": "BCSC-Summary", "waveform": "Waveform Transformer",
          "structure_only": "Structure-only Path-CNN", "direct_fusion": "v1直接融合",
          "gated_fusion": "v2可靠性门控融合", "gated_zero_structure": "v2门控+Zero-Structure"}
METRICS = ("auroc", "auprc", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "brier")
SNR_BINS = ((-15., -10., "[-15,-10)"), (-10., -5., "[-10,-5)"), (-5., 0., "[-5,0)"),
            (0., 5., "[0,5)"), (5., 10., "[5,10)"), (10., 15.000001, "[10,15]"))


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    manifest_path: Path = SNR_ROOT / "dataset_manifest.csv"
    waveform_path: Path = SNR_ROOT / "cache" / "preprocessed_waveforms.npy"
    workers: int = max(1, min(6, os.cpu_count() or 1))
    batch_size: int = 64
    max_epochs: int = 25
    patience: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bootstrap_runs: int = 2000
    device: str = "auto"


def base() -> Any:
    name = "experiment23_v1_shared"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载v1：{V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def output_paths(config: Config) -> dict[str, Path]:
    result = {key: config.output_dir / key for key in ("cache", "models", "metrics", "predictions", "figures", "logs", "tests")}
    result["root"] = config.output_dir
    for path in result.values(): path.mkdir(parents=True, exist_ok=True)
    return result


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def device_for(value: str) -> torch.device:
    if value == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA不可用")
    return device


def load_data(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    if not config.manifest_path.exists() or not config.waveform_path.exists():
        raise FileNotFoundError("缺少实验21的固定低SNR清单或波形缓存")
    manifest, waves = pd.read_csv(config.manifest_path), np.load(config.waveform_path, mmap_mode="r")
    positives = manifest.label.eq(1)
    checks = [len(manifest) == 7000, tuple(waves.shape) == (7000, SAMPLE_COUNT), manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500},
              manifest.ori_file_num.nunique() == 34, manifest.groupby("ori_file_num").test_fold.nunique().max() == 1,
              manifest.loc[positives, "target_snr_db"].between(-15, 15).all(), np.isfinite(waves).all()]
    if not all(checks): raise RuntimeError("实验21低SNR数据合同验证失败")
    return manifest, waves


def _extract_one(payload: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, dict[str, Any]]:
    index, wave = payload
    structure, status = base().extract_structure_one(wave, SEED + index)
    status["sample_index"] = index
    return index, structure, status


def extract_structure(config: Config, force: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    paths = output_paths(config); manifest, waves = load_data(config)
    map_path, completed_path, status_path = paths["cache"] / "structure_maps.npy", paths["cache"] / "structure_completed.npy", paths["cache"] / "structure_status.csv"
    if force:
        for path in (map_path, completed_path, status_path):
            if path.exists(): path.unlink()
    if map_path.exists() and completed_path.exists():
        maps, completed = np.lib.format.open_memmap(map_path, mode="r+"), np.load(completed_path)
        if maps.shape != (7000, 3, PATH_BINS) or completed.shape != (7000,): raise RuntimeError("v2结构缓存尺寸错误")
    else:
        maps = np.lib.format.open_memmap(map_path, mode="w+", dtype=np.float32, shape=(7000, 3, PATH_BINS)); maps[:] = 0.; maps.flush()
        completed = np.zeros(7000, dtype=bool); np.save(completed_path, completed)
    records: dict[int, dict[str, Any]] = {}
    if status_path.exists():
        prior = pd.read_csv(status_path)
        if "sample_index" in prior:
            extraction_columns = ["sample_index", "status", "error", "direct_index", "candidate_path_count", "detected_path_count", "elapsed_s"]
            available = [column for column in extraction_columns if column in prior.columns]
            records = {int(row["sample_index"]): row for row in prior[available].dropna(subset=["sample_index"]).to_dict("records")}
    pending = np.flatnonzero(~completed)
    if len(pending):
        print(f"[总体进度 10%] 低SNR BCSC：已完成{int(completed.sum())}/7000，待处理{len(pending)}", flush=True)
        bar = tqdm(total=len(pending), desc="总体进度：v2 BCSC结构提取", unit="样本")
        payloads = ((int(index), np.asarray(waves[index], dtype=np.float32)) for index in pending)
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for done, (index, structure, status) in enumerate(pool.map(_extract_one, payloads, chunksize=1), 1):
                maps[index], completed[index], records[index] = structure, True, status; bar.update(1)
                if done % 25 == 0 or done == len(pending):
                    maps.flush(); np.save(completed_path, completed)
                    pd.DataFrame.from_dict(records, orient="index").sort_values("sample_index").to_csv(status_path, index=False, encoding="utf-8-sig")
        bar.close()
    maps.flush(); np.save(completed_path, completed)
    status = pd.DataFrame.from_dict(records, orient="index").sort_values("sample_index")
    status = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "label", "ori_file_num", "test_fold", "noise_type", "target_snr_db"]].merge(status, on="sample_index", how="left", validate="one_to_one")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")
    if not completed.all() or maps.shape != (7000, 3, PATH_BINS): raise RuntimeError("v2结构缓存未完整写入")
    dump(paths["cache"] / "structure_contract.json", {"shape": [7000, 3, PATH_BINS], "source": str(config.waveform_path), "seed": SEED,
        "bcsc": {"atom_points": 49, "outer_iterations": 35, "lambda_ratio": .075, "bootstrap_runs": 8, "all_initial_candidates": True}})
    dump(paths["cache"] / "structure_status_summary.json", {"samples": 7000, "completed": int(completed.sum()),
        "status_counts": {str(key): int(value) for key, value in status["status"].fillna("missing").value_counts().items()},
        "failed_samples": int(status["status"].eq("failed").sum())})
    return status, np.load(map_path, mmap_mode="r")


def load_structure(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    paths = output_paths(config); manifest, _ = load_data(config); map_path, done_path = paths["cache"] / "structure_maps.npy", paths["cache"] / "structure_completed.npy"
    if not map_path.exists() or not done_path.exists(): raise FileNotFoundError("请先运行extract-structure")
    maps, done = np.load(map_path, mmap_mode="r"), np.load(done_path)
    if maps.shape != (7000, 3, PATH_BINS) or not done.all(): raise RuntimeError("v2结构缓存不完整")
    return manifest, maps


class Dataset(Dataset):
    def __init__(self, waves: np.ndarray, structures: np.ndarray, labels: np.ndarray, indices: np.ndarray): self.waves, self.structures, self.labels, self.indices = waves, structures, labels, indices
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        return torch.as_tensor(np.asarray(self.waves[index]), dtype=torch.float32), torch.as_tensor(np.asarray(self.structures[index]), dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32), index


def loader(waves: np.ndarray, structures: np.ndarray, labels: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(Dataset(waves, structures, labels, indices), batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


class Model(nn.Module):
    """v1直接融合与v2可靠性门控共享的编码器；门控不读取真实SNR。"""
    def __init__(self, condition: str):
        super().__init__(); self.condition = condition; shared = base()
        self.waveform = shared.WaveformEncoder() if condition != "structure_only" else None
        self.structure = shared.StructureEncoder() if condition != "waveform" else None
        self.gate = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1)) if condition.startswith("gated_") else None
        input_size = 64 if condition == "waveform" else 32 if condition == "structure_only" else 96
        self.head = nn.Sequential(nn.Linear(input_size, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, wave: torch.Tensor, structure: torch.Tensor) -> torch.Tensor:
        if self.condition == "waveform": value = self.waveform(wave)
        elif self.condition == "structure_only": value = self.structure(structure)
        else:
            if self.condition == "gated_zero_structure": structure = torch.zeros_like(structure)
            x, m = self.waveform(wave), self.structure(structure)
            if self.gate is not None: m = torch.sigmoid(self.gate(torch.cat([x, m], dim=1))) * m
            value = torch.cat([x, m], dim=1)
        return self.head(value).squeeze(1)


def split(manifest: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, validation = manifest.test_fold.to_numpy(), (fold + 1) % 5
    return np.flatnonzero((values != fold) & (values != validation)), np.flatnonzero(values == validation), np.flatnonzero(values == fold)


@torch.no_grad()
def predict(model: Model, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval(); ids, labels, logits = [], [], []
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for wave, structure, label, index in data:
        logits.append(model(wave.to(device, non_blocking=True), structure.to(device, non_blocking=True)).cpu().numpy()); ids.append(np.asarray(index)); labels.append(label.numpy())
    if device.type == "cuda": torch.cuda.synchronize()
    return np.concatenate(ids), np.concatenate(labels), np.concatenate(logits), time.perf_counter() - started


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    def nll(log_temp: float) -> float:
        temp = math.exp(log_temp); return float(np.mean(np.logaddexp(0., logits / temp) - labels * logits / temp))
    result = minimize_scalar(nll, bounds=(-3., 3.), method="bounded")
    return float(math.exp(result.x)) if result.success else 1.


def threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0., .5, 1.], probability)))
    scores = [balanced_accuracy_score(labels, probability >= candidate) for candidate in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric(labels: np.ndarray, probability: np.ndarray, thresholds: float | np.ndarray) -> dict[str, float]:
    predicted = probability >= thresholds; tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probability)), "auprc": float(average_precision_score(labels, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)), "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)), "specificity": float(tn / max(tn + fp, 1)),
            "f1": float(f1_score(labels, predicted, zero_division=0)), "mcc": float(matthews_corrcoef(labels, predicted)), "brier": float(brier_score_loss(labels, probability)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def low_snr_mask(frame: pd.DataFrame) -> np.ndarray:
    return (frame.label.to_numpy() == 0) | ((frame.label.to_numpy() == 1) & (frame.target_snr_db.to_numpy(dtype=float) >= -15) & (frame.target_snr_db.to_numpy(dtype=float) < 0))


def run_dir(config: Config, condition: str, fold: int) -> Path: return output_paths(config)["models"] / condition / f"seed_{SEED}" / f"fold_{fold}"


def train_neural(condition: str, fold: int, waves: np.ndarray, structures: np.ndarray, manifest: pd.DataFrame, config: Config, resume: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    directory = run_dir(config, condition, fold); directory.mkdir(parents=True, exist_ok=True)
    complete, checkpoint, pred_file = directory / "COMPLETE.json", directory / "best_checkpoint.pt", directory / "test_predictions.csv"
    if resume and complete.exists() and checkpoint.exists() and pred_file.exists(): return json.loads(complete.read_text(encoding="utf-8")), pd.read_csv(pred_file)
    seed_all(SEED + fold * 100); device, labels = device_for(config.device), manifest.label.to_numpy(dtype=np.float32); train, validation, test = split(manifest, fold)
    loads = {"train": loader(waves, structures, labels, train, config.batch_size, True), "validation": loader(waves, structures, labels, validation, config.batch_size * 2), "test": loader(waves, structures, labels, test, config.batch_size * 2)}
    model, optimizer, criterion = Model(condition).to(device), None, nn.BCEWithLogitsLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler, history, best, stale = torch.amp.GradScaler("cuda", enabled=device.type == "cuda"), [], -np.inf, 0; started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train(); losses = []
        for wave, structure, label, _ in loads["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(wave.to(device, non_blocking=True), structure.to(device, non_blocking=True)), label.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        val_ids, val_labels, val_logits, _ = predict(model, loads["validation"], device)
        val_frame = manifest.iloc[val_ids].reset_index(drop=True).assign(label=val_labels.astype(int), logit=val_logits)
        mask = low_snr_mask(val_frame); score = float(roc_auc_score(val_labels[mask], val_logits[mask]))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_low_snr_auroc": score})
        if score > best + 1e-7:
            best, stale = score, 0; torch.save({"state_dict": model.state_dict(), "condition": condition, "fold": fold, "seed": SEED}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience: break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    val_ids, val_labels, val_logits, _ = predict(model, loads["validation"], device); temperature = fit_temperature(val_logits, val_labels); cut = threshold(val_labels, 1 / (1 + np.exp(-val_logits / temperature)))
    test_ids, test_labels, test_logits, inference_seconds = predict(model, loads["test"], device); probability = 1 / (1 + np.exp(-test_logits / temperature)); details = metric(test_labels.astype(int), probability, cut)
    details.update({"condition": condition, "seed": SEED, "test_fold": fold, "validation_fold": (fold + 1) % 5, "threshold": cut, "temperature": temperature,
                    "best_validation_low_snr_auroc": best, "best_epoch": int(max(row["epoch"] for row in history if row["validation_low_snr_auroc"] == best)),
                    "parameters": int(sum(item.numel() for item in model.parameters() if item.requires_grad)), "inference_ms_per_sample": inference_seconds / len(test_ids) * 1e3, "training_seconds": time.perf_counter() - started})
    prediction = pd.DataFrame({"sample_index": test_ids, "label": test_labels.astype(int), "p_click": probability, "predicted_label": (probability >= cut).astype(int), "threshold": cut, "temperature": temperature, "condition": condition, "seed": SEED, "test_fold": fold})
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False, encoding="utf-8-sig"); prediction.to_csv(pred_file, index=False, encoding="utf-8-sig"); dump(complete, details)
    return details, prediction


def classic_train(config: Config, waves: np.ndarray, structures: np.ndarray, manifest: pd.DataFrame, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = output_paths(config); met_path, pred_path = paths["metrics"] / "classic_fold_metrics.csv", paths["predictions"] / "classic_oof_predictions.csv"
    if resume and met_path.exists() and pred_path.exists(): return pd.read_csv(met_path), pd.read_csv(pred_path)
    labels = manifest.label.to_numpy(dtype=int); teager = np.log1p(base().teager_scores(waves)).reshape(-1, 1); summary = base().summary_features(structures); metrics_rows, predictions = [], []
    bar = tqdm(total=10, desc="总体进度：v2确定性基线", unit="模型折")
    for fold in range(5):
        train, validation, test = split(manifest, fold)
        for condition, values in (("teager", teager), ("bcsc_summary", summary)):
            started = time.perf_counter(); model = LogisticRegression(random_state=SEED, max_iter=200) if condition == "teager" else HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=SEED)
            model.fit(values[train], labels[train]); val_prob = model.predict_proba(values[validation])[:, 1]; cut = threshold(labels[validation], val_prob); before = time.perf_counter(); prob = model.predict_proba(values[test])[:, 1]; elapsed = time.perf_counter() - before
            row = metric(labels[test], prob, cut); row.update({"condition": condition, "seed": SEED, "test_fold": fold, "validation_fold": (fold + 1) % 5, "threshold": cut, "temperature": np.nan,
                "parameters": 0 if condition == "teager" else int(sum(len(predictor.nodes) for stage in model._predictors for predictor in stage)), "inference_ms_per_sample": elapsed / len(test) * 1e3, "training_seconds": time.perf_counter() - started})
            metrics_rows.append(row); predictions.append(pd.DataFrame({"sample_index": test, "label": labels[test], "p_click": prob, "predicted_label": (prob >= cut).astype(int), "threshold": cut, "temperature": np.nan, "condition": condition, "seed": SEED, "test_fold": fold})); bar.update(1)
    bar.close(); metric_frame, pred_frame = pd.DataFrame(metrics_rows), pd.concat(predictions, ignore_index=True); metric_frame.to_csv(met_path, index=False, encoding="utf-8-sig"); pred_frame.to_csv(pred_path, index=False, encoding="utf-8-sig"); return metric_frame, pred_frame


def train(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = output_paths(config); manifest, waves = load_data(config); _, structures = load_structure(config); classic_metrics, classic_predictions = classic_train(config, waves, structures, manifest, resume)
    neural_met, neural_pred = paths["metrics"] / "neural_fold_metrics.csv", paths["predictions"] / "neural_oof_predictions.csv"
    if resume and neural_met.exists() and neural_pred.exists(): nmetrics, npredictions = pd.read_csv(neural_met), pd.read_csv(neural_pred)
    else:
        results, predictions = [], []; bar = tqdm(total=25, desc="总体进度：v2五条件五折", unit="模型折")
        for fold in range(5):
            for condition in NEURAL:
                row, prediction = train_neural(condition, fold, waves, structures, manifest, config, resume); results.append(row); predictions.append(prediction); bar.update(1); bar.set_postfix(fold=fold, condition=condition, auroc=f"{row['auroc']:.3f}")
        bar.close(); nmetrics, npredictions = pd.DataFrame(results), pd.concat(predictions, ignore_index=True); nmetrics.to_csv(neural_met, index=False, encoding="utf-8-sig"); npredictions.to_csv(neural_pred, index=False, encoding="utf-8-sig")
    all_metrics, all_predictions = pd.concat([classic_metrics, nmetrics], ignore_index=True), pd.concat([classic_predictions, npredictions], ignore_index=True)
    metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    all_predictions = all_predictions.merge(metadata, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
    if len(all_predictions) != 49_000 or all_predictions.duplicated(["condition", "seed", "sample_id"]).any(): raise RuntimeError(f"OOF合同失败：{len(all_predictions)}行")
    all_metrics.to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig"); all_predictions.to_csv(paths["predictions"] / "oof_predictions_long.csv", index=False, encoding="utf-8-sig")
    return all_metrics, all_predictions


def bootstrap(frame: pd.DataFrame, metric_name: str, rng: np.random.Generator, runs: int) -> tuple[float, float]:
    groups, values = frame.ori_file_num.unique(), []
    for _ in range(runs):
        selected = rng.choice(groups, len(groups), replace=True); sample = pd.concat([frame[frame.ori_file_num == group] for group in selected], ignore_index=True)
        if sample.label.nunique() == 2: values.append(metric(sample.label.to_numpy(), sample.p_click.to_numpy(), sample.threshold.to_numpy())[metric_name])
    return tuple(map(float, np.percentile(values, [2.5, 97.5])))


def paired(left: pd.DataFrame, right: pd.DataFrame, low_snr: bool, rng: np.random.Generator, runs: int) -> tuple[float, float, float]:
    joined = left[["sample_id", "label", "ori_file_num", "target_snr_db", "p_click"]].merge(right[["sample_id", "p_click"]], on="sample_id", suffixes=("_left", "_right"), validate="one_to_one")
    if low_snr: joined = joined[(joined.label == 0) | ((joined.label == 1) & joined.target_snr_db.between(-15, 0, inclusive="left"))]
    groups, values = joined.ori_file_num.unique(), []
    point = float(roc_auc_score(joined.label, joined.p_click_left) - roc_auc_score(joined.label, joined.p_click_right))
    for _ in range(runs):
        selected = rng.choice(groups, len(groups), replace=True); sample = pd.concat([joined[joined.ori_file_num == group] for group in selected], ignore_index=True)
        if sample.label.nunique() == 2: values.append(float(roc_auc_score(sample.label, sample.p_click_left) - roc_auc_score(sample.label, sample.p_click_right)))
    low, high = np.percentile(values, [2.5, 97.5]); return point, float(low), float(high)


def summarize(config: Config, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths, rng, rows = output_paths(config), np.random.default_rng(SEED), []
    for condition in CONDITIONS:
        frame = predictions[predictions.condition == condition].copy(); row: dict[str, Any] = {"condition": condition, "condition_label": LABELS[condition]}
        for key in METRICS:
            row[key] = metric(frame.label.to_numpy(), frame.p_click.to_numpy(), frame.threshold.to_numpy())[key]; row[f"{key}_ci_low"], row[f"{key}_ci_high"] = bootstrap(frame, key, rng, config.bootstrap_runs)
        row["parameters"] = float("nan") if condition in CLASSIC else float(0)  # 用fold表在报告时补全。
        rows.append(row)
    main = pd.DataFrame(rows)
    snr_rows = []
    for condition in CONDITIONS:
        frame = predictions[predictions.condition == condition]
        for low, high, label in SNR_BINS:
            selected = frame[(frame.label == 0) | ((frame.label == 1) & (frame.target_snr_db >= low) & (frame.target_snr_db < high))]
            value = metric(selected.label.to_numpy(), selected.p_click.to_numpy(), selected.threshold.to_numpy())
            snr_rows.append({"condition": condition, "snr_bin": label, "snr_low_db": low, "snr_high_db": min(high, 15.), "positive_count": int(((selected.label == 1)).sum()), "negative_count": int((selected.label == 0).sum()), **{key: value[key] for key in ("auroc", "auprc", "recall")}})
    snr = pd.DataFrame(snr_rows); comparison_rows = []
    left = predictions[predictions.condition == "gated_fusion"]
    for right_name in ("waveform", "gated_zero_structure", "direct_fusion"):
        right = predictions[predictions.condition == right_name]
        for low_snr, scope in ((True, "低SNR[-15,0)"), (False, "全SNR")):
            point, low, high = paired(left, right, low_snr, rng, config.bootstrap_runs)
            comparison_rows.append({"left": "gated_fusion", "right": right_name, "scope": scope, "auroc_difference": point, "ci_low": low, "ci_high": high,
                "supports": bool(low > 0)})
    comparisons = pd.DataFrame(comparison_rows); main.to_csv(paths["metrics"] / "main_results.csv", index=False, encoding="utf-8-sig"); snr.to_csv(paths["metrics"] / "snr_results.csv", index=False, encoding="utf-8-sig"); comparisons.to_csv(paths["metrics"] / "paired_auroc_bootstrap.csv", index=False, encoding="utf-8-sig")
    return main, snr, comparisons


def figures(config: Config, predictions: pd.DataFrame, snr: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    directory = output_paths(config)["figures"]; plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in CONDITIONS:
        frame = predictions[predictions.condition == condition]; fpr, tpr, _ = roc_curve(frame.label, frame.p_click); precision, recall, _ = precision_recall_curve(frame.label, frame.p_click); axes[0].plot(fpr, tpr, label=LABELS[condition]); axes[1].plot(recall, precision, label=LABELS[condition])
    axes[0].plot([0, 1], [0, 1], "k--", lw=.7); axes[0].set(title="全SNR ROC", xlabel="FPR", ylabel="TPR"); axes[1].set(title="全SNR PR", xlabel="Recall", ylabel="Precision"); axes[0].legend(fontsize=7); axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(directory / "roc_pr_all_snr.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition in CONDITIONS:
        item = snr[snr.condition == condition]; axes[0].plot((item.snr_low_db + item.snr_high_db) / 2, item.auroc, marker="o", label=LABELS[condition]); axes[1].plot((item.snr_low_db + item.snr_high_db) / 2, item.auprc, marker="o", label=LABELS[condition])
    axes[0].set(title="分SNR AUROC", xlabel="正样本目标SNR (dB)", ylabel="AUROC", ylim=(0, 1.02)); axes[1].set(title="分SNR AUPRC", xlabel="正样本目标SNR (dB)", ylabel="AUPRC", ylim=(0, 1.02)); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(directory / "snr_curves.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); y = np.arange(len(comparisons)); axes[0].errorbar(comparisons.auroc_difference, y, xerr=[comparisons.auroc_difference-comparisons.ci_low, comparisons.ci_high-comparisons.auroc_difference], fmt="o", capsize=3); axes[0].axvline(0, color="k", lw=.8); axes[0].set(yticks=y, yticklabels=(comparisons.scope + ": " + comparisons.right), xlabel="v2门控融合 AUROC差", title="配对Bootstrap")
    for condition in ("waveform", "gated_fusion"):
        frame = predictions[predictions.condition == condition]; observed, estimated = calibration_curve(frame.label, frame.p_click, n_bins=10, strategy="quantile"); axes[1].plot(estimated, observed, marker="o", label=LABELS[condition])
    axes[1].plot([0, 1], [0, 1], "k--", lw=.7); axes[1].set(title="校准曲线", xlabel="预测概率", ylabel="click比例"); axes[1].legend(); fig.tight_layout(); fig.savefig(directory / "paired_gain_calibration.png", dpi=180); plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns); rows = []
    for row in frame.itertuples(index=False, name=None): rows.append([f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row])
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def report(config: Config) -> Path:
    paths = output_paths(config); prediction_path = paths["predictions"] / "oof_predictions_long.csv"
    if not prediction_path.exists(): raise FileNotFoundError("请先完成train")
    predictions = pd.read_csv(prediction_path)
    if len(predictions) != 49_000: raise RuntimeError("报告拒绝使用不完整的OOF预测")
    main, snr, comparisons = summarize(config, predictions); figures(config, predictions, snr, comparisons)
    low = comparisons[(comparisons.scope == "低SNR[-15,0)") & comparisons.right.isin(["waveform", "gated_zero_structure"])]
    conclusion = "支持低SNR结构增益" if len(low) == 2 and low.ci_low.gt(0).all() else "当前低SNR结构设计未获支持"
    text = f"""# 实验23-v2：低SNR BCSC可靠性门控融合

本报告仅使用实验21固定低SNR混合数据：7,000条样本、34段录音、正样本目标SNR在[-15,15] dB；单训练种子为{SEED}，五折录音组隔离。真实`target_snr_db`只用于验证选择与结果分层，不输入任何模型。

主要终点是`[-15,0)` dB正样本与同测试fold全部负样本的AUROC。v2门控融合相对Waveform和门控Zero-Structure的低SNR配对95%置信区间均须大于0才支持结构信息增益。

## 全范围结果

{markdown(main[["condition_label", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "balanced_accuracy", "f1", "mcc", "brier"]])}

## 主要配对比较

{markdown(comparisons)}

**结论：{conclusion}。** 单种子Bootstrap仅反映录音组抽样不确定性，不代表随机初始化稳健性。

## 分SNR结果

{markdown(snr)}
"""
    destination = paths["root"] / "research_report_zh.md"; destination.write_text(text, encoding="utf-8")
    dump(paths["root"] / "artifact_validation.json", {"structure_shape": [7000, 3, PATH_BINS], "neural_model_folds": 25, "oof_rows": len(predictions), "seed": SEED, "passed": True})
    return destination


def verify(config: Config) -> dict[str, Any]:
    paths = output_paths(config); shared = base(); cfg = shared.bcsc_config(); rng = np.random.default_rng(SEED); atom = shared.load_legacy().normalize_atom(shared.load_legacy().windows.gaussian(cfg.atom_length, std=cfg.atom_length / 7) * np.cos(2 * np.pi * 90_000 * np.arange(cfg.atom_length) / SAMPLE_RATE))
    signal, direct, delay = rng.normal(0, .002, SAMPLE_COUNT), 2700, int(round(.35e-3 * SAMPLE_RATE)); signal[direct:direct+len(atom)] += atom; signal[direct+delay:direct+delay+len(atom)] += .45 * atom; signal[direct+2*delay:direct+2*delay+len(atom)] += .25 * atom
    structure, status = shared.extract_structure_one(signal, SEED); bin_index = delay - 20; synthetic = {"status": status["status"], "expected_bin": bin_index, "p": float(structure[0, bin_index]), "a": float(structure[1, bin_index]), "u": float(structure[2, bin_index]), "passed": bool(status["status"] == "ok" and structure[0, bin_index] > 0 and structure[1, bin_index] > 0 and 0 <= structure[2, bin_index] <= 1)}
    network = {}; wave, map_value, labels = torch.randn(4, SAMPLE_COUNT), torch.rand(4, 3, PATH_BINS), torch.tensor([0., 1., 0., 1.])
    for condition in NEURAL:
        model = Model(condition); logits = model(wave, map_value); loss = nn.BCEWithLogitsLoss()(logits, labels); loss.backward(); network[condition] = {"shape": list(logits.shape), "loss": float(loss.detach()), "finite": bool(torch.isfinite(torch.sigmoid(logits)).all()), "passed": bool(logits.shape == (4,) and torch.isfinite(loss))}
    result = {"synthetic_bcsc": synthetic, "network": network, "passed": bool(synthetic["passed"] and all(item["passed"] for item in network.values()))}; dump(paths["tests"] / "verification.json", result)
    if not result["passed"]: raise RuntimeError(f"v2验证失败：{result}")
    return result


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("all", "extract-structure", "train", "report", "verify")); parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1))); parser.add_argument("--device", default="auto"); parser.add_argument("--bootstrap-runs", type=int, default=2000); parser.add_argument("--no-resume", action="store_true"); return parser.parse_args()


def main() -> None:
    value = args(); config = Config(workers=value.workers, device=value.device, bootstrap_runs=value.bootstrap_runs); paths = output_paths(config); dump(paths["root"] / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir), "manifest_path": str(config.manifest_path), "waveform_path": str(config.waveform_path), "seed": SEED, "neural_conditions": list(NEURAL), "primary_endpoint": "low_snr_[-15,0)_auroc"})
    if value.command == "verify": verify(config); return
    if value.command in ("all", "extract-structure"):
        print("[总体进度 0%] 验证低SNR数据合同与模型", flush=True); verify(config); extract_structure(config, force=value.no_resume)
    if value.command in ("all", "train"):
        print("[总体进度 45%] 训练25个v2神经网络折", flush=True); train(config, resume=not value.no_resume)
    if value.command in ("all", "report"):
        print("[总体进度 90%] 统计与生成真实结果报告", flush=True); print(f"[总体进度100%] 完成：{report(config)}", flush=True)


if __name__ == "__main__": main()
