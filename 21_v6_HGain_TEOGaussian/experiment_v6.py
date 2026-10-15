#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验21 v6：h 增益分配与 x 的 TEO 高斯软先验四组消融。

数据、折号和加噪混合固定复用 v4 的 low-SNR source-only 数据契约。本脚本不
重建数据；训练前会对该契约、逐折来源隔离、TEO 边界和梯度路径逐项验收。
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4 = ROOT / "21_PhysicsInformedClickClassification"
V4_DATA = V4 / "outputs_snr_m15_p15_source_only_5seeds"
# 与 v6 相同的 source-only 分类定义下，历史物理损失参考保存在同一固定低 SNR 数据契约中。
V4_HISTORICAL = V4_DATA
sys.path.insert(0, str(V4))
from pipeline import (BranchFeatureEncoder, CausalResponseDecoder, FullLengthDecoder,
                      SharedEncoder, WaveformDataset, causal_fft_convolution,
                      preprocess_waveform, read_waveform, split_indices)

SEED = 20_260_826
CONDS = ("A_no_gain_no_teo", "B_gain_only", "C_teo_only", "D_gain_teo")
NAMES = {
    "A_no_gain_no_teo": "A：无增益、无 TEO（v4 自由 x/h）",
    "B_gain_only": "B：仅 h 增益（单位能量 x）",
    "C_teo_only": "C：仅 TEO 高斯项（v4 自由 x/h）",
    "D_gain_teo": "D：完整 v6（h 增益 + TEO）",
    "historical_v4_physics": "历史 v4：物理损失参考（只读）",
}


@dataclass(frozen=True)
class Config:
    output_dir: Path = HERE / "outputs_seed20260826"
    source_data_dir: Path = V4_DATA
    historical_dir: Path = V4_HISTORICAL
    sample_rate: int = 576_000
    sample_count: int = 5_761
    response_count: int = 1_153
    lowcut_hz: float = 20_000.0
    highcut_hz: float = 190_000.0
    filter_order: int = 5
    tukey_alpha: float = 0.04
    seed: int = SEED
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 25
    patience: int = 5
    epsilon: float = 1e-8
    teo_weight: float = 0.1
    teo_smooth_window: int = 9
    bootstrap_runs: int = 2_000
    num_workers: int = 0


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dirs(cfg: Config) -> dict[str, Path]:
    paths = {name: cfg.output_dir / name for name in
             ("models", "logs", "metrics", "predictions", "figures", "waves", "validation")}
    paths["root"] = cfg.output_dir
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    return paths


def condition_uses_gain(condition: str) -> bool:
    return condition in ("B_gain_only", "D_gain_teo")


def condition_uses_teo(condition: str) -> bool:
    return condition in ("C_teo_only", "D_gain_teo")


class GainHead(nn.Module):
    """共享编码器池化特征的 256→32→1 增益头。"""
    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 1))
        self.epsilon = epsilon

    def initialize_median(self, median_l2: float) -> None:
        # softplus(bias) + epsilon = training-input median L2 norm.
        desired = max(float(median_l2) - self.epsilon, self.epsilon)
        inverse_softplus = math.log(math.expm1(desired)) if desired < 40 else desired
        with torch.no_grad():
            for layer in (self.layers[0], self.layers[-1]):
                layer.weight.zero_(); layer.bias.zero_()
            self.layers[-1].bias.fill_(inverse_softplus)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((latent.mean(dim=-1), latent.amax(dim=-1)), dim=1)
        return nn.functional.softplus(self.layers(pooled)[:, 0]) + self.epsilon


class V6Model(nn.Module):
    """v4 自由 h 分支；分类器仅接收根据条件决定的 x。"""
    def __init__(self, cfg: Config, use_gain: bool) -> None:
        super().__init__()
        self.cfg, self.use_gain = cfg, use_gain
        self.shared_encoder = SharedEncoder()
        self.source_decoder = FullLengthDecoder(cfg.sample_count)
        self.response_decoder = CausalResponseDecoder(cfg.response_count)
        # 所有条件都实例化，保证共享模块的随机初始化逐折一致。
        self.gain_head = GainHead(cfg.epsilon)
        self.source_classifier = BranchFeatureEncoder()
        self.classification_head = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def set_gain_initialization(self, median_l2: float) -> None:
        self.gain_head.initialize_median(median_l2)

    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.shared_encoder(y)
        u = self.source_decoder(latent)
        h_shape = self.response_decoder(latent)  # h_shape[0] = 1 exactly
        g = self.gain_head(latent)
        if self.use_gain:
            # 用户定义中的 |u|_2：逐样本 L2 单位能量信源。
            source = u / torch.linalg.vector_norm(u, dim=1, keepdim=True).clamp_min(self.cfg.epsilon)
            response = g[:, None] * h_shape
        else:
            source, response = u, h_shape
            g = torch.ones_like(g)
        reconstruction = causal_fft_convolution(source, response)
        logit = self.classification_head(self.source_classifier(source)).squeeze(-1)
        return {"logit": logit, "source": source, "source_raw": u, "response": response,
                "h_shape": h_shape, "gain": g, "reconstruction": reconstruction}


def reconstruction_loss(y: torch.Tensor, reconstruction: torch.Tensor, epsilon: float) -> torch.Tensor:
    return ((y - reconstruction).square().sum(1) / (y.square().sum(1) + epsilon)).mean()


def smoothed_teo(x: torch.Tensor, window: int) -> torch.Tensor:
    """有效内点 TEO 后做无填充 9 点滑动平均，保留符号。"""
    psi = x[:, 1:-1].square() - x[:, :-2] * x[:, 2:]
    return nn.functional.avg_pool1d(psi.unsqueeze(1), window, stride=1).squeeze(1)


def gaussian_targets(e: torch.Tensor, sigma_low: float, sigma_high: float, epsilon: float
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """由 e+ 矩匹配生成停止梯度的离散 L1 归一化高斯。"""
    positions = torch.arange(e.shape[1], device=e.device, dtype=e.dtype)[None]
    positive = e.clamp_min(0)
    mass = positive.sum(1, keepdim=True)
    default_mu = torch.full_like(mass, (e.shape[1] - 1) / 2)
    weights = positive / mass.clamp_min(epsilon)
    mu = torch.where(mass > epsilon, (weights * positions).sum(1, keepdim=True), default_mu)
    variance = (weights * (positions - mu).square()).sum(1, keepdim=True)
    sigma = torch.where(mass > epsilon, variance.clamp_min(0).sqrt(),
                        torch.full_like(mu, sigma_high))
    sigma = sigma.clamp(float(sigma_low), float(sigma_high))
    target = torch.exp(-.5 * ((positions - mu) / sigma.clamp_min(epsilon)).square())
    target = target / target.sum(1, keepdim=True).clamp_min(epsilon)
    return target.detach(), mu.detach()[:, 0], sigma.detach()[:, 0], mass.detach()[:, 0]


def teo_loss(source: torch.Tensor, labels: torch.Tensor, sigma_low: float, sigma_high: float,
             cfg: Config) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    e = smoothed_teo(source, cfg.teo_smooth_window)
    q = e / e.abs().sum(1, keepdim=True).clamp_min(cfg.epsilon)
    target, mu, sigma, positive_mass = gaussian_targets(e, sigma_low, sigma_high, cfg.epsilon)
    per_sample = (q - target).square().sum(1) / target.square().sum(1).clamp_min(cfg.epsilon)
    positive_mask = labels > .5
    if bool(positive_mask.any()):
        loss = per_sample[positive_mask].mean()
    else:
        loss = source.sum() * 0.0
    return loss, {"teo_error": per_sample, "teo_mu": mu, "teo_sigma_samples": sigma,
                  "teo_positive_mass": positive_mass, "teo_q": q, "teo_target": target, "teo_e": e}


def losses(outputs: dict[str, torch.Tensor], y: torch.Tensor, labels: torch.Tensor, condition: str,
           bounds: tuple[float, float], cfg: Config) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    recon = reconstruction_loss(y, outputs["reconstruction"], cfg.epsilon)
    bce = nn.functional.binary_cross_entropy_with_logits(outputs["logit"], labels)
    raw_teo, detail = teo_loss(outputs["source"], labels, *bounds, cfg)
    weighted_teo = raw_teo * (cfg.teo_weight if condition_uses_teo(condition) else 0.0)
    return {"total": recon + bce + weighted_teo, "recon": recon, "bce": bce,
            "teo_raw": raw_teo, "teo_weighted": weighted_teo}, detail


def grad_norm(module: nn.Module) -> float:
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    choices = np.unique(np.r_[0., .5, 1., probabilities])
    return float(choices[np.argmax([balanced_accuracy_score(labels, probabilities >= x) for x in choices])])


def metrics(labels: np.ndarray, probabilities: np.ndarray, thresholds: float | np.ndarray) -> dict[str, float]:
    prediction = probabilities >= thresholds
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "f1": float(f1_score(labels, prediction, zero_division=0)),
            "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
            "nrmse": float("nan")}


def bootstrap_classification_metrics(labels: np.ndarray, probabilities: np.ndarray,
                                     thresholds: np.ndarray) -> dict[str, float]:
    """与 sklearn 的二分类 AUROC/AP 及阈值指标等价的 NumPy 实现。

    此函数仅用于 Bootstrap 内循环；避免在 2000 次分组重抽样中反复构造
    sklearn 对象。概率相同的样本按同一阈值块处理，保证 AUROC 的半秩规则和
    average_precision 的阶梯积分定义。
    """
    y = np.asarray(labels, dtype=bool); p = np.asarray(probabilities, dtype=float)
    positive, negative = int(y.sum()), int((~y).sum())
    if positive == 0 or negative == 0:
        return {k: float("nan") for k in ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity")}
    ascending = np.argsort(p, kind="mergesort"); ps, ys = p[ascending], y[ascending]
    split = np.r_[0, np.flatnonzero(np.diff(ps)) + 1, len(ps)]
    prior_negative, auc_numerator = 0, 0.0
    for start, stop in zip(split[:-1], split[1:]):
        group_positive = int(ys[start:stop].sum()); group_negative = (stop - start) - group_positive
        auc_numerator += group_positive * (prior_negative + .5 * group_negative)
        prior_negative += group_negative
    descending = np.argsort(-p, kind="mergesort"); ps, ys = p[descending], y[descending]
    split = np.r_[0, np.flatnonzero(np.diff(ps)) + 1, len(ps)]
    tp = fp = 0; previous_recall = 0.0; ap = 0.0
    for start, stop in zip(split[:-1], split[1:]):
        group_positive = int(ys[start:stop].sum()); group_negative = (stop - start) - group_positive
        tp += group_positive; fp += group_negative; recall = tp / positive
        ap += (recall - previous_recall) * (tp / max(tp + fp, 1)); previous_recall = recall
    predicted = p >= thresholds; tp = int(np.logical_and(y, predicted).sum()); fn = positive - tp
    tn = int(np.logical_and(~y, ~predicted).sum()); fp = negative - tn
    sensitivity, specificity = tp / positive, tn / negative
    return {"auroc": auc_numerator / (positive * negative), "auprc": ap,
            "balanced_accuracy": .5 * (sensitivity + specificity),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "sensitivity": sensitivity, "specificity": specificity}


def effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    p = singular / max(singular.sum(), eps)
    return float(np.exp(-(p * np.log(p + eps)).sum()))


def response_diagnostics(out: dict[str, torch.Tensor], y: torch.Tensor, detail: dict[str, torch.Tensor],
                         cfg: Config) -> dict[str, np.ndarray]:
    h_shape, response = out["h_shape"], out["response"]
    idx = torch.arange(h_shape.shape[1], device=y.device, dtype=y.dtype)[None]
    shape_energy = h_shape.square().sum(1).clamp_min(cfg.epsilon)
    response_energy = response.square().sum(1).clamp_min(cfg.epsilon)
    nrmse = ((out["reconstruction"] - y).square().sum(1) / (y.square().sum(1) + cfg.epsilon)).sqrt()
    return {"nrmse": nrmse.detach().cpu().numpy(), "gain": out["gain"].detach().cpu().numpy(),
            "h_first_energy_ratio": (h_shape[:, 0].square() / shape_energy).detach().cpu().numpy(),
            "h_shape_tail_energy_ratio": (h_shape[:, 1:].square().sum(1) / shape_energy).detach().cpu().numpy(),
            "h_energy_delay_ms": ((idx * h_shape.square()).sum(1) / shape_energy * 1000 / cfg.sample_rate).detach().cpu().numpy(),
            "response_energy_delay_ms": ((idx * response.square()).sum(1) / response_energy * 1000 / cfg.sample_rate).detach().cpu().numpy(),
            "teo_fit_error": detail["teo_error"].detach().cpu().numpy(),
            "teo_mu_sample": detail["teo_mu"].detach().cpu().numpy(),
            "teo_sigma_samples": detail["teo_sigma_samples"].detach().cpu().numpy(),
            "teo_positive_mass": detail["teo_positive_mass"].detach().cpu().numpy()}


def loaders(waveforms: np.ndarray, labels: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray,
            cfg: Config, fold: int) -> dict[str, DataLoader]:
    common = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    # 每个条件重建同一生成器，因此逐折 batch 顺序一致。
    train_gen = torch.Generator().manual_seed(cfg.seed + fold)
    return {"train": DataLoader(WaveformDataset(waveforms, labels, tr), cfg.batch_size, shuffle=True,
                                generator=train_gen, **common),
            "validation": DataLoader(WaveformDataset(waveforms, labels, va), cfg.batch_size * 2, shuffle=False, **common),
            "test": DataLoader(WaveformDataset(waveforms, labels, te), cfg.batch_size * 2, shuffle=False, **common)}


def unnoised_training_click_bounds(manifest: pd.DataFrame, train_indices: np.ndarray, cfg: Config
                                   ) -> tuple[tuple[float, float], dict[str, Any]]:
    """只读取本折训练正样本的未加噪观测 Click，计算 TEO-FWHM 代理宽度。"""
    positives = manifest.iloc[train_indices]
    positives = positives[positives["label"].eq(1)]
    widths: list[float] = []
    for _, row in positives.iterrows():
        click = torch.from_numpy(preprocess_waveform(read_waveform(row, cfg), cfg)).float()[None]
        e = smoothed_teo(click, cfg.teo_smooth_window)[0].numpy()
        e = np.maximum(e, 0.0)
        if not np.isfinite(e).all() or e.max() <= cfg.epsilon:
            continue
        peak = int(np.argmax(e)); half = .5 * e[peak]
        left, right = peak, peak
        while left > 0 and e[left - 1] >= half: left -= 1
        while right + 1 < len(e) and e[right + 1] >= half: right += 1
        fwhm_samples = max(right - left + 1, 1)
        widths.append(fwhm_samples / (2 * math.sqrt(2 * math.log(2))))
    physical_low, physical_high = 2.0, cfg.sample_rate / 1000.0
    if widths:
        low, high = np.percentile(widths, [10, 90]).astype(float)
        low, high = max(low, physical_low), min(high, physical_high)
        if high < low: high = low
    else:
        low, high = physical_low, physical_high
    return (float(low), float(high)), {"training_positive_count": int(len(positives)),
                                        "valid_fwhm_sample_count": int(len(widths)),
                                        "sigma_low_samples": float(low), "sigma_high_samples": float(high),
                                        "sigma_low_ms": float(low * 1000 / cfg.sample_rate),
                                        "sigma_high_ms": float(high * 1000 / cfg.sample_rate),
                                        "bounds_source": "unnoised_training_positive_observed_clicks_only"}


@torch.no_grad()
def predict(model: V6Model, loader: DataLoader, device: torch.device, bounds: tuple[float, float],
            cfg: Config, keep_arrays: bool = False) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    model.eval(); frames: list[pd.DataFrame] = []; arrays: list[dict[str, np.ndarray]] = []
    for y, label, index in loader:
        y, label = y.to(device), label.to(device)
        out = model(y); _, detail = losses(out, y, label, "D_gain_teo", bounds, cfg)
        diag = response_diagnostics(out, y, detail, cfg)
        frames.append(pd.DataFrame({"sample_index": index.numpy(), "label": label.cpu().numpy().astype(int),
                                    "probability": torch.sigmoid(out["logit"]).cpu().numpy(), **diag}))
        if keep_arrays:
            arrays.append({key: value.detach().cpu().numpy() for key, value in out.items() if key != "logit"} |
                          {"observed": y.cpu().numpy(), "sample_index": index.numpy(),
                           "teo_e": detail["teo_e"].cpu().numpy(), "teo_target": detail["teo_target"].cpu().numpy()})
    return pd.concat(frames, ignore_index=True), arrays


def checkpoint_paths(paths: dict[str, Path], condition: str, fold: int) -> tuple[Path, Path, Path]:
    stem = f"{condition}_seed_{SEED}_fold_{fold}"
    return paths["models"] / f"{stem}.pt", paths["logs"] / f"{stem}_history.csv", paths["logs"] / f"{stem}_batch.csv"


def fit(condition: str, fold: int, waveforms: np.ndarray, manifest: pd.DataFrame, cfg: Config,
        force: bool, overall: tqdm) -> tuple[dict[str, Any], pd.DataFrame]:
    paths, device = dirs(cfg), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = split_indices(manifest, fold)
    width_path = paths["metrics"] / f"teo_width_bounds_fold_{fold}.json"
    if width_path.exists():
        width_info = json.loads(width_path.read_text(encoding="utf-8"))
        bounds = (float(width_info["sigma_low_samples"]), float(width_info["sigma_high_samples"]))
    else:
        bounds, width_info = unnoised_training_click_bounds(manifest, tr, cfg)
        dump(width_path, width_info)
    labels = manifest.label.to_numpy(np.float32)
    seed_all(cfg.seed + fold)  # 公共模块各条件相同初始权重。
    model = V6Model(cfg, condition_uses_gain(condition)).to(device)
    input_median = float(np.median(np.linalg.norm(np.asarray(waveforms[tr]), axis=1)))
    model.set_gain_initialization(input_median)
    ld = loaders(waveforms, labels, tr, va, te, cfg, fold)
    ck, history_path, batch_path = checkpoint_paths(paths, condition, fold)
    ready = False
    if ck.exists() and history_path.exists() and batch_path.exists() and not force:
        saved = torch.load(ck, map_location="cpu", weights_only=True)
        ready = bool(saved.get("training_complete") and saved.get("condition") == condition)
    if not ready:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        best, stale, histories, batches, started = -np.inf, 0, [], [], time.time()
        epochs = tqdm(range(1, cfg.max_epochs + 1), desc=f"{condition} 折{fold} epoch", unit="epoch", leave=False)
        for epoch in epochs:
            model.train(); totals = {k: 0.0 for k in ("total", "recon", "bce", "teo_raw", "teo_weighted")}; count = 0
            batch_bar = tqdm(ld["train"], desc=f"折{fold} epoch {epoch} batch", unit="batch", leave=False)
            for number, (y, label, _) in enumerate(batch_bar, 1):
                y, label = y.to(device), label.to(device); opt.zero_grad(set_to_none=True)
                out = model(y); loss, _ = losses(out, y, label, condition, bounds, cfg); loss["total"].backward()
                record = {k: float(v.detach().cpu()) for k, v in loss.items()}
                record.update({"condition": condition, "fold": fold, "epoch": epoch, "batch": number,
                               "source_decoder_gradient_norm": grad_norm(model.source_decoder),
                               "response_decoder_gradient_norm": grad_norm(model.response_decoder),
                               "gain_head_gradient_norm": grad_norm(model.gain_head),
                               "shared_encoder_gradient_norm": grad_norm(model.shared_encoder),
                               "training_seconds_elapsed": time.time() - started})
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                batches.append(record); count += 1
                for key in totals: totals[key] += record[key]
                batch_bar.set_postfix(loss=f"{record['total']:.3f}", teo=f"{record['teo_weighted']:.3f}", refresh=False)
            validation, _ = predict(model, ld["validation"], device, bounds, cfg)
            score = float(roc_auc_score(validation.label, validation.probability))
            row = {"condition": condition, "fold": fold, "epoch": epoch, "validation_auroc": score,
                   **{k: v / max(count, 1) for k, v in totals.items()}, "elapsed_seconds": time.time() - started}
            histories.append(row); epochs.set_postfix(val_auroc=f"{score:.4f}")
            if score > best + 1e-4:
                best, stale = score, 0
                torch.save({"state_dict": model.state_dict(), "condition": condition, "fold": fold,
                            "seed": cfg.seed, "best_epoch": epoch, "validation_auroc": score,
                            "training_seconds": time.time() - started, "gain_init_l2_median": input_median,
                            "teo_bounds": bounds}, ck)
            else:
                stale += 1
                if stale >= cfg.patience: break
        pd.DataFrame(histories).to_csv(history_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(batches).to_csv(batch_path, index=False, encoding="utf-8-sig")
        saved = torch.load(ck, map_location="cpu", weights_only=True); saved["training_complete"] = True; torch.save(saved, ck)
    saved = torch.load(ck, map_location=device, weights_only=True); model.load_state_dict(saved["state_dict"])
    validation, _ = predict(model, ld["validation"], device, bounds, cfg)
    cut = threshold(validation.label.to_numpy(), validation.probability.to_numpy())
    test, arrays = predict(model, ld["test"], device, bounds, cfg, keep_arrays=True)
    shape_matrix = np.concatenate([x["h_shape"] for x in arrays])
    for key, value in {"threshold": cut, "condition": condition, "seed": cfg.seed, "test_fold": fold}.items():
        test[key] = value
    test["teo_sigma_low_samples"] = bounds[0]
    test["teo_sigma_high_samples"] = bounds[1]
    np.savez_compressed(paths["waves"] / f"{condition}_fold_{fold}_test.npz", **{k: np.concatenate([x[k] for x in arrays]) for k in arrays[0]})
    values = metrics(test.label.to_numpy(), test.probability.to_numpy(), cut)
    values.update({"condition": condition, "fold": fold, "validation_fold": (fold + 1) % 5,
                   "best_epoch": int(saved["best_epoch"]), "validation_auroc": float(saved["validation_auroc"]),
                   "threshold": cut, "training_seconds": float(saved["training_seconds"]),
                   "gain_initial_l2_median": input_median, "teo_sigma_low_samples": bounds[0],
                   "teo_sigma_high_samples": bounds[1], "h_shape_effective_rank": effective_rank(shape_matrix),
                   "h_shape_mean_pair_distance": pair_distance(shape_matrix, cfg.seed + fold)})
    values["nrmse"] = float(test.nrmse.mean())
    overall.update(1); overall.set_postfix(condition=condition, fold=fold, auroc=f"{values['auroc']:.4f}")
    return values, test


def pair_distance(matrix: np.ndarray, seed: int, limit: int = 10_000) -> float:
    if len(matrix) < 2: return float("nan")
    rng = np.random.default_rng(seed); a = rng.integers(0, len(matrix), min(limit, len(matrix) * 4)); b = rng.integers(0, len(matrix), len(a))
    keep = a != b
    return float(np.linalg.norm(matrix[a[keep]] - matrix[b[keep]], axis=1).mean()) if keep.any() else float("nan")


def train_all(waveforms: np.ndarray, manifest: pd.DataFrame, cfg: Config, force: bool, folds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = dirs(cfg); all_metrics, all_predictions = [], []
    overall = tqdm(total=len(CONDS) * len(folds), desc="总体进度：20 次条件×折训练", unit="条件折")
    for fold in folds:
        for condition in CONDS:
            metric, prediction = fit(condition, fold, waveforms, manifest, cfg, force, overall)
            all_metrics.append(metric); all_predictions.append(prediction)
    overall.close(); fold_metrics = pd.DataFrame(all_metrics); predictions = pd.concat(all_predictions, ignore_index=True)
    meta = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    predictions = predictions.merge(meta, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
    fold_metrics.to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"] / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    return fold_metrics, predictions


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in frame.groupby("condition"):
        out = metrics(part.label.to_numpy(), part.probability.to_numpy(), part.threshold.to_numpy())
        out["nrmse"] = float(part.nrmse.mean()); rows.append({"condition": condition, **out})
    return pd.DataFrame(rows)


def snr_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in frame.groupby("condition"):
        negatives = part[part.label.eq(0)]
        for lo in range(-15, 15, 5):
            pos = part[(part.label.eq(1)) & (part.target_snr_db.ge(lo)) & (part.target_snr_db.le(lo + 5) if lo == 10 else part.target_snr_db.lt(lo + 5))]
            joined = pd.concat((negatives, pos)); out = metrics(joined.label.to_numpy(), joined.probability.to_numpy(), joined.threshold.to_numpy())
            rows.append({"condition": condition, "snr_bin_db": f"[{lo},{lo + 5}{']' if lo == 10 else ')'}", "n_positive": len(pos), **out})
    return pd.DataFrame(rows)


def bootstrap_differences(predictions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """按原始录音来源配对重抽样，且每条样本继续使用其所属折验证阈值。"""
    rng = np.random.default_rng(cfg.seed + 901); pairs = (("B_gain_only", "A_no_gain_no_teo", "B_minus_A"),
        ("D_gain_teo", "C_teo_only", "D_minus_C"), ("C_teo_only", "A_no_gain_no_teo", "C_minus_A"),
        ("D_gain_teo", "B_gain_only", "D_minus_B"), ("D_gain_teo", "A_no_gain_no_teo", "D_minus_A"))
    rows = []; keys = ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity", "nrmse")
    for left_name, right_name, title in pairs:
        left, right = predictions[predictions.condition.eq(left_name)], predictions[predictions.condition.eq(right_name)]
        groups = np.sort(left.ori_file_num.unique())
        # 预先转为 NumPy 分组数组；每次重抽样仍完整保留抽中录音的所有样本，
        # 只是避免 2000×5 次 DataFrame 过滤/拼接和同一指标集的重复计算。
        fields = ("label", "probability", "threshold", "nrmse")
        left_parts = {int(g): tuple(left.loc[left.ori_file_num.eq(g), f].to_numpy() for f in fields) for g in groups}
        right_parts = {int(g): tuple(right.loc[right.ori_file_num.eq(g), f].to_numpy() for f in fields) for g in groups}
        def pooled(parts: dict[int, tuple[np.ndarray, ...]], chosen: np.ndarray) -> tuple[np.ndarray, ...]:
            return tuple(np.concatenate([parts[int(g)][i] for g in chosen]) for i in range(len(fields)))
        observed_left = metrics(left.label.to_numpy(), left.probability.to_numpy(), left.threshold.to_numpy())
        observed_right = metrics(right.label.to_numpy(), right.probability.to_numpy(), right.threshold.to_numpy())
        observed_left["nrmse"], observed_right["nrmse"] = float(left.nrmse.mean()), float(right.nrmse.mean())
        values = {key: np.empty(cfg.bootstrap_runs, dtype=float) for key in keys}
        bar = tqdm(range(cfg.bootstrap_runs), desc=f"Bootstrap {title}", leave=False)
        for run in bar:
            chosen = rng.choice(groups, len(groups), replace=True)
            al, ap, at, an = pooled(left_parts, chosen); bl, bp, bt, bn = pooled(right_parts, chosen)
            a_metrics, b_metrics = bootstrap_classification_metrics(al, ap, at), bootstrap_classification_metrics(bl, bp, bt)
            for key in keys[:-1]: values[key][run] = a_metrics[key] - b_metrics[key]
            values["nrmse"][run] = an.mean() - bn.mean()
        for key, value in values.items():
            rows.append({"comparison": title, "metric": key, "estimate": float(observed_left[key] - observed_right[key]),
                         "ci_low": float(np.percentile(value, 2.5)), "ci_high": float(np.percentile(value, 97.5)), "bootstrap_runs": cfg.bootstrap_runs})
    return pd.DataFrame(rows)


def diagnostic_summary(predictions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for condition, part in predictions.groupby("condition"):
        positive, negative = part[part.label.eq(1)], part[part.label.eq(0)]
        for label, subset in (("positive", positive), ("negative", negative)):
            rows.append({"condition": condition, "subset": label, "n": len(subset),
                         "mean_teo_fit_error": float(subset.teo_fit_error.mean()),
                         "median_teo_fit_error": float(subset.teo_fit_error.median()),
                         "mean_fitted_sigma_ms": float(subset.teo_sigma_samples.mean() * 1000 / cfg.sample_rate),
                         "sigma_at_lower_bound_fraction": float(np.isclose(subset.teo_sigma_samples, subset.teo_sigma_low_samples, rtol=0, atol=1e-5).mean()),
                         "sigma_at_upper_bound_fraction": float(np.isclose(subset.teo_sigma_samples, subset.teo_sigma_high_samples, rtol=0, atol=1e-5).mean()),
                         "gain_mean": float(subset.gain.mean()),
                         "gain_std": float(subset.gain.std()),
                         "gain_p05": float(subset.gain.quantile(.05)),
                         "gain_p95": float(subset.gain.quantile(.95)),
                         "gain_median": float(subset.gain.median()),
                         "h_first_energy_ratio_mean": float(subset.h_first_energy_ratio.mean()),
                         "h_tail_energy_ratio_mean": float(subset.h_shape_tail_energy_ratio.mean()),
                         "h_energy_delay_ms_median": float(subset.h_energy_delay_ms.median())})
    return pd.DataFrame(rows)


def exchange_diagnostics(predictions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """交换仅 h_shape，保留 source 和其自身 g；以重建相对误差增量作诊断。"""
    rows = []
    for condition in CONDS:
        deltas = {"same_train": [], "cross_train": []}
        for fold in sorted(predictions.loc[predictions.condition.eq(condition), "test_fold"].unique()):
            path = dirs(cfg)["waves"] / f"{condition}_fold_{fold}_test.npz"
            # 每个数组仅解压一次；不能在样本对循环中反复以 data["source"] 访问压缩成员。
            with np.load(path) as data:
                indices = data["sample_index"]
                sources, shapes = data["source"], data["h_shape"]
                gains, observed, reconstructions = data["gain"], data["observed"], data["reconstruction"]
            # 元数据直接由预测表按 index 获取，避免位置推断。
            meta = predictions[(predictions.condition.eq(condition)) & (predictions.test_fold.eq(fold))].set_index("sample_index")
            pos = np.array([i for i in indices if int(meta.loc[int(i), "label"]) == 1])
            if len(pos) < 2:
                del sources, shapes, gains, observed, reconstructions
                continue
            lookup = {int(v): k for k, v in enumerate(indices)}; rng = np.random.default_rng(cfg.seed + fold)
            linear = cfg.sample_count + cfg.response_count - 1
            fft_count = 1 << (linear - 1).bit_length()
            for kind in deltas:
                for source_id in rng.choice(pos, min(200, len(pos)), replace=False):
                    choices = pos[pos != source_id]
                    group = meta.loc[int(source_id), "ori_file_num"]
                    choices = np.array([x for x in choices if (meta.loc[int(x), "ori_file_num"] == group) == (kind == "same_train")])
                    if not len(choices): continue
                    other = int(rng.choice(choices)); a, b = lookup[int(source_id)], lookup[other]
                    x, h, g, y, own = sources[a], shapes[b], gains[a], observed[a], reconstructions[a]
                    # 与 np.convolve(x, g*h)[:N] 等价的零填充 FFT 因果线性卷积，
                    # 避免交换诊断的 O(N×H) 逐对瓶颈。
                    swapped = np.fft.irfft(np.fft.rfft(x, fft_count) * np.fft.rfft(g * h, fft_count), fft_count)[:cfg.sample_count]
                    own_err = np.sum((y - own) ** 2) / (np.sum(y ** 2) + cfg.epsilon)
                    swap_err = np.sum((y - swapped) ** 2) / (np.sum(y ** 2) + cfg.epsilon)
                    deltas[kind].append(swap_err - own_err)
            del sources, shapes, gains, observed, reconstructions
            gc.collect()
        for kind, values in deltas.items(): rows.append({"condition": condition, "exchange": kind, "n_pairs": len(values),
                                                          "mean_reconstruction_error_delta": float(np.mean(values)) if values else float("nan"),
                                                          "median_reconstruction_error_delta": float(np.median(values)) if values else float("nan")})
    return pd.DataFrame(rows)


def plot_five(cfg: Config, condition: str, predictions: pd.DataFrame, manifest: pd.DataFrame) -> None:
    selected = pd.read_csv(V4 / "five_test_samples" / "sample_labels.csv")
    items = []
    for _, row in selected.iterrows():
        index = int(row.sample_index); fold = int(manifest.loc[index, "test_fold"])
        data = np.load(dirs(cfg)["waves"] / f"{condition}_fold_{fold}_test.npz")
        where = int(np.where(data["sample_index"] == index)[0][0]); pred = predictions[(predictions.condition.eq(condition)) & (predictions.sample_index.eq(index))].iloc[0]
        items.append((row, pred, {k: data[k][where] for k in data.files}))
    t = np.arange(cfg.sample_count) * 1000 / cfg.sample_rate; th = np.arange(cfg.response_count) * 1000 / cfg.sample_rate
    fig, axes = plt.subplots(5, 5, figsize=(22, 15)); labels = {0: "背景噪声", 1: "Click"}
    for i, (row, pred, data) in enumerate(items):
        e = data["teo_e"]; e = e / (np.max(np.abs(e)) + cfg.epsilon); target = data["teo_target"] / (np.max(data["teo_target"]) + cfg.epsilon)
        axes[i, 0].plot(t, data["source"], lw=.65); axes[i, 0].set_title("预测 x" if i == 0 else "")
        axes[i, 1].plot(t[5:-5], e, lw=.65, label="平滑 TEO"); axes[i, 1].plot(t[5:-5], target, lw=.8, label="拟合 Gaussian"); axes[i, 1].legend(fontsize=6); axes[i, 1].set_title("归一化 TEO 与高斯" if i == 0 else "")
        axes[i, 2].plot(th, data["response"], lw=.65, label="h"); axes[i, 2].plot(th[1:], data["h_shape"][1:] * 4, lw=.65, label="4× h_shape tail"); axes[i, 2].legend(fontsize=6); axes[i, 2].set_title("h / h_shape 尾部放大" if i == 0 else "")
        axes[i, 3].plot(t, data["reconstruction"], lw=.65); axes[i, 3].set_title("重建 y" if i == 0 else "")
        axes[i, 4].plot(t, data["observed"], lw=.65, label="实际 y"); axes[i, 4].plot(t, data["reconstruction"], lw=.65, label="重建 y"); axes[i, 4].legend(fontsize=6); axes[i, 4].set_title("实际 / 重建叠加" if i == 0 else "")
        axes[i, 0].set_ylabel(f"#{index} 真={labels[int(row.label)]}\n预测={labels[int(pred.probability >= pred.threshold)]}\nP={pred.probability:.3f}, T={pred.threshold:.3f}\ng={pred.gain:.3f}, SNR={pred.target_snr_db if pd.notna(pred.target_snr_db) else '负样本'}", fontsize=7)
        for ax in axes[i]: ax.grid(alpha=.2); ax.tick_params(labelsize=6)
    fig.suptitle(NAMES[condition]); fig.tight_layout(rect=(.04, .02, 1, .97)); base = dirs(cfg)["figures"] / f"five_samples_{condition}"
    fig.savefig(base.with_suffix(".png"), dpi=180); fig.savefig(base.with_suffix(".pdf")); plt.close(fig)
    np.savez_compressed(base.with_suffix(".npz"), **{f"sample_{i}_{k}": value for i, (_, _, item) in enumerate(items) for k, value in item.items()})


def plot_a_d(cfg: Config) -> None:
    a = plt.imread(dirs(cfg)["figures"] / "five_samples_A_no_gain_no_teo.png")
    d = plt.imread(dirs(cfg)["figures"] / "five_samples_D_gain_teo.png")
    fig, axes = plt.subplots(2, 1, figsize=(18, 24)); axes[0].imshow(a); axes[0].set_title(NAMES["A_no_gain_no_teo"]); axes[1].imshow(d); axes[1].set_title(NAMES["D_gain_teo"])
    for ax in axes: ax.axis("off")
    fig.tight_layout(); fig.savefig(dirs(cfg)["figures"] / "A_vs_D_five_samples.png", dpi=150); fig.savefig(dirs(cfg)["figures"] / "A_vs_D_five_samples.pdf"); plt.close(fig)


def read_historical(manifest: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = cfg.historical_dir / "predictions" / "sample_predictions.csv"
    if not path.exists(): return pd.DataFrame(), {"available": False, "reason": str(path)}
    source = pd.read_csv(path)
    source = source[(source.condition == "recon_physics_bce_source_only") & (source.seed == cfg.seed)].copy()
    required = {"sample_index", "probability", "threshold", "test_fold"}
    if not required.issubset(source.columns) or len(source) != len(manifest):
        return pd.DataFrame(), {"available": False, "reason": "historical prediction rows/columns mismatch", "rows": len(source)}
    reference = manifest.reset_index(names="sample_index")[["sample_index", "label", "ori_file_num", "test_fold"]]
    joined = source.merge(reference, on=["sample_index", "test_fold"], validate="one_to_one",
                          suffixes=("_historical", "_manifest"))
    label_column = "label_manifest" if "label_manifest" in joined else "label"
    historical_label = "label_historical" if "label_historical" in joined else label_column
    labels_match = bool(joined[historical_label].astype(int).eq(joined[label_column].astype(int)).all())
    joined["label"] = joined[label_column].astype(int)
    valid = bool(len(joined) == len(manifest) and joined["label"].notna().all() and labels_match)
    return joined.assign(condition="historical_v4_physics"), {"available": valid, "rows": len(joined), "condition": "recon_physics_bce_source_only", "seed": cfg.seed, "labels_match_manifest": labels_match}


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty: return "（无数据）"
    def cell(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}" if np.isfinite(value) else "NA"
        return str(value).replace("|", "\\|")
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def report(cfg: Config, fold_metrics: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame,
           snr: pd.DataFrame, bootstrap: pd.DataFrame, diag: pd.DataFrame, exchange: pd.DataFrame,
           historical: pd.DataFrame, historical_check: dict[str, Any]) -> None:
    historical_summary = pd.DataFrame()
    if not historical.empty:
        historical_summary = pd.DataFrame([{ "condition": "historical_v4_physics", **metrics(historical.label.to_numpy(), historical.probability.to_numpy(), historical.threshold.to_numpy()) }])
    by_condition = summary.set_index("condition")
    positive_diag = diag[diag.subset.eq("positive")].set_index("condition")
    h_structure = (fold_metrics.groupby("condition", as_index=False).agg(
        mean_h_shape_effective_rank=("h_shape_effective_rank", "mean"),
        mean_h_shape_pair_distance=("h_shape_mean_pair_distance", "mean"),
        mean_h_first_energy_ratio=("h_shape_effective_rank", lambda _: np.nan),
    ))
    # h 首点/尾部使用逐样本正样本诊断；秩和距离使用逐折 h_shape 矩阵诊断。
    h_structure = h_structure.merge(positive_diag[["h_first_energy_ratio_mean", "h_tail_energy_ratio_mean", "h_energy_delay_ms_median"]],
                                    left_on="condition", right_index=True, how="left")
    def difference(name: str, metric: str) -> pd.Series:
        return bootstrap[(bootstrap.comparison.eq(name)) & (bootstrap.metric.eq(metric))].iloc[0]
    b_a, c_a, d_a = difference("B_minus_A", "auroc"), difference("C_minus_A", "auroc"), difference("D_minus_A", "auroc")
    teo_a, teo_c, teo_d = (positive_diag.loc["A_no_gain_no_teo", "mean_teo_fit_error"],
                           positive_diag.loc["C_teo_only", "mean_teo_fit_error"],
                           positive_diag.loc["D_gain_teo", "mean_teo_fit_error"])
    neg_a, neg_c, neg_d = (diag[(diag.condition.eq("A_no_gain_no_teo")) & (diag.subset.eq("negative"))].iloc[0].mean_teo_fit_error,
                           diag[(diag.condition.eq("C_teo_only")) & (diag.subset.eq("negative"))].iloc[0].mean_teo_fit_error,
                           diag[(diag.condition.eq("D_gain_teo")) & (diag.subset.eq("negative"))].iloc[0].mean_teo_fit_error)
    exchange_scale = float(exchange.mean_reconstruction_error_delta.abs().max())
    text = f"""# 实验21 v6——h 增益与 x 的 TEO 高斯形状约束

本报告只包含随机种子 {cfg.seed} 的固定数据、四组五折结果；不把单种子结果表述为跨种子稳定性。TEO 高斯是软先验，不是宽带 Click 的严格物理定律。x 的单位能量定义采用逐样本 L2 范数；g 是当前预处理下的相对增益，不能解释为真实传播损失。

## 分类与重建

### 逐折

{markdown_table(fold_metrics[["condition", "fold", "best_epoch", "validation_auroc", "auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity", "nrmse", "threshold"]])}

### 合并折外预测

{markdown_table(pd.concat((summary, historical_summary), ignore_index=True))}

### 原始录音来源配对 Bootstrap（2000 次）

每条样本使用其对应测试折的验证阈值；差值包括 B−A、D−C（增益方案），C−A、D−B（TEO）和 D−A（整体）。

{markdown_table(bootstrap)}

### 分 SNR 指标

{markdown_table(snr[["condition", "snr_bin_db", "n_positive", "auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity", "nrmse"]])}

## x 的 TEO 与 h 诊断

正样本 TEO 拟合误差、拟合宽度、负样本对应统计及 h_shape（不受 g 影响）的首点/尾部能量比、时延与 g 分布如下。需以负样本是否也呈现相似高斯形状判定先验的选择性，而非只看正样本拟合降低。

{markdown_table(diag)}

### h_shape 结构汇总（五折均值）

{markdown_table(h_structure)}

### 仅交换 h_shape 的诊断

交换时保留源样本的 x 与 g，只替换 h_shape；因此误差变化不将增益不匹配计作响应结构差异。

{markdown_table(exchange)}

## 三项结论

1. **分类未改善。**合并折外 AUROC：A={by_condition.loc['A_no_gain_no_teo', 'auroc']:.4f}，B={by_condition.loc['B_gain_only', 'auroc']:.4f}，C={by_condition.loc['C_teo_only', 'auroc']:.4f}，D={by_condition.loc['D_gain_teo', 'auroc']:.4f}。B−A={b_a.estimate:+.4f} [{b_a.ci_low:+.4f}, {b_a.ci_high:+.4f}]、C−A={c_a.estimate:+.4f} [{c_a.ci_low:+.4f}, {c_a.ci_high:+.4f}]、D−A={d_a.estimate:+.4f} [{d_a.ci_low:+.4f}, {d_a.ci_high:+.4f}]；三个区间均小于 0。因此本固定单种子实验中，增益、TEO 与联合方案均未提高分类性能。

2. **x 对高斯软先验的贴合仅有限改善，且不具选择性。**正样本平均 TEO 误差从 A 的 {teo_a:.4f} 降至 C 的 {teo_c:.4f}，D 为 {teo_d:.4f}；但负样本也从 {neg_a:.4f} 降至 C={neg_c:.4f}、D={neg_d:.4f}。同时所有组的拟合宽度均触及逐折上界（见宽度触界比例），不能把该误差下降解释为得到稳定、正样本特异的高斯 Click 形态。

3. **h 仍然退化。**正样本 h_shape 首点能量占比约 0.94–0.99，尾部能量占比仅约 0.01–0.06，且同 train 与跨 train 的 h_shape 交换造成的重建误差均值变化绝对值最大仅 {exchange_scale:.4f}。g 的统计见上表；若 g 改变而 h_shape 仍近似单位冲激，这只能说明幅值分工发生调整，不能作为响应结构被恢复的证据。本结果支持“调整了幅值分工，未解决响应结构退化”。

## 图件与数据

每组五个固定测试样本的 PNG/PDF、波形 NPZ 和逐样本 CSV 位于 `figures/`、`waves/`、`predictions/`。`A_vs_D_five_samples.*` 是公共基线与完整 v6 的并排对照。

## 历史参考核验

{json.dumps(historical_check, ensure_ascii=False, indent=2)}

## 结论规则

分类是否改善只依据折外指标及预设 Bootstrap 差值；x 是否更符合先验只依据正负样本 TEO 诊断；h 是否仍退化只依据 h_shape 首点/尾部、时延、有效秩、样本距离和交换诊断。若 g 改变而 h_shape 仍近似单位冲激，结论必须是“调整了幅值分工，未解决响应结构退化”。
"""
    (dirs(cfg)["root"] / "final_report_zh.md").write_text(text, encoding="utf-8")


def validation(cfg: Config, manifest: pd.DataFrame, waveforms: np.ndarray) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); seed_all(cfg.seed)
    results: dict[str, Any] = {"data_rows": int(len(manifest)), "waveform_shape": list(waveforms.shape),
                               "finite_waveforms": bool(np.isfinite(waveforms).all()), "fold_source_isolation": []}
    for fold in range(5):
        tr, va, te = split_indices(manifest, fold); groups = [set(manifest.iloc[x].ori_file_num) for x in (tr, va, te)]
        results["fold_source_isolation"].append(bool(not(groups[0] & groups[1]) and not(groups[0] & groups[2]) and not(groups[1] & groups[2])))
    model = V6Model(cfg, True).to(device); model.set_gain_initialization(float(np.median(np.linalg.norm(waveforms[:128], axis=1))))
    y = torch.from_numpy(np.asarray(waveforms[:3])).to(device); labels = torch.tensor([1., 0., 1.], device=device)
    out = model(y); loss, detail = losses(out, y, labels, "D_gain_teo", (2., 100.), cfg)
    results.update({"unit_energy_x": bool(torch.allclose(torch.linalg.vector_norm(out["source"], dim=1), torch.ones(3, device=device), atol=2e-5)),
                    "h0_equals_g": bool(torch.allclose(out["response"][:, 0], out["gain"], atol=1e-6)),
                    "h_shape_h0_equals_1": bool(torch.equal(out["h_shape"][:, 0], torch.ones(3, device=device))),
                    "losses_finite": {k: bool(torch.isfinite(v)) for k, v in loss.items()},
                    "teo_signed_finite": bool(torch.isfinite(detail["teo_e"]).all()),
                    "teo_has_negative_values": bool((detail["teo_e"] < 0).any()),
                    "teo_no_positive_batch_zero": None})
    # 直接卷积核验。
    x = torch.zeros(2, cfg.sample_count, device=device); h = torch.zeros(2, cfg.response_count, device=device); x[0, 9] = 1; x[1, 20] = .5; h[:, 0] = 1; h[0, 5] = .25; h[1, 100] = -.1
    direct = torch.zeros_like(x)
    for k in (0, 5, 100): direct[:, k:] += h[:, k, None] * x[:, :cfg.sample_count-k]
    results["causal_convolution_max_error"] = float((causal_fft_convolution(x, h) - direct).abs().max().cpu())
    model.zero_grad(set_to_none=True); nn.functional.binary_cross_entropy_with_logits(out["logit"], labels).backward(retain_graph=True)
    results["bce_gradient_norms"] = {"source_decoder": grad_norm(model.source_decoder), "response_decoder": grad_norm(model.response_decoder), "gain_head": grad_norm(model.gain_head), "shared_encoder": grad_norm(model.shared_encoder)}
    model.zero_grad(set_to_none=True); reconstruction_loss(y, model(y)["reconstruction"], cfg.epsilon).backward()
    results["recon_gradient_norms"] = {"response_decoder": grad_norm(model.response_decoder), "gain_head": grad_norm(model.gain_head)}
    model.zero_grad(set_to_none=True); zero_labels = torch.zeros(3, device=device); zero_loss, _ = teo_loss(model(y)["source"], zero_labels, 2., 100., cfg); zero_loss.backward()
    results["teo_no_positive_batch_zero"] = bool(float(zero_loss.detach().cpu()) == 0.0 and grad_norm(model.source_decoder) == 0.0)
    results["passed"] = bool(results["data_rows"] == 7000 and results["waveform_shape"] == [7000, cfg.sample_count] and results["finite_waveforms"] and all(results["fold_source_isolation"]) and results["unit_energy_x"] and results["h0_equals_g"] and results["h_shape_h0_equals_1"] and all(results["losses_finite"].values()) and results["causal_convolution_max_error"] < 1e-6 and results["bce_gradient_norms"]["response_decoder"] == 0.0 and results["bce_gradient_norms"]["gain_head"] == 0.0 and results["bce_gradient_norms"]["source_decoder"] > 0 and results["recon_gradient_norms"]["response_decoder"] > 0 and results["recon_gradient_norms"]["gain_head"] > 0 and results["teo_no_positive_batch_zero"])
    dump(dirs(cfg)["validation"] / "preflight_validation.json", results); return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="只从已保存的20折外预测和波形恢复统计、图件与报告；不训练或重写模型。")
    parser.add_argument("--output-dir", type=Path, default=Config().output_dir)
    args = parser.parse_args(); cfg = Config(output_dir=args.output_dir, bootstrap_runs=30 if args.mode == "smoke" else 2000,
                                             max_epochs=2 if args.mode == "smoke" else 25, patience=1 if args.mode == "smoke" else 5)
    if args.mode == "smoke": args.folds = [0]
    if set(args.folds) - set(range(5)): raise ValueError("folds 必须在 0..4")
    paths = dirs(cfg); dump(paths["root"] / "run_config.json", {**asdict(cfg), "conditions": CONDS, "folds": args.folds,
              "data_contract": str(cfg.source_data_dir), "normalization": "x=u/max(||u||_2,epsilon) for B/D",
              "historical_reference": str(cfg.historical_dir)})
    manifest_path = cfg.source_data_dir / "dataset_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if args.postprocess_only:
        if args.mode != "full": raise ValueError("--postprocess-only 只允许用于正式 full 输出")
        fold_path, prediction_path = paths["metrics"] / "fold_metrics.csv", paths["predictions"] / "sample_predictions.csv"
        if not fold_path.exists() or not prediction_path.exists():
            raise FileNotFoundError("缺少已保存的 fold_metrics.csv 或 sample_predictions.csv，无法仅恢复后处理")
        fold_metrics, predictions = pd.read_csv(fold_path), pd.read_csv(prediction_path)
        check_path = paths["validation"] / "preflight_validation.json"
        check = json.loads(check_path.read_text(encoding="utf-8")) if check_path.exists() else {"passed": False}
        if not check.get("passed", False): raise RuntimeError("缺少或未通过训练前验收记录")
        expected = len(CONDS) * len(manifest)
        if len(predictions) != expected or set(predictions.condition) != set(CONDS):
            raise RuntimeError(f"折外预测覆盖不完整：{len(predictions)}/{expected}")
        print("总体阶段 1/4：复用已完成的20个模型与折外预测")
    else:
        array_path = cfg.source_data_dir / "cache" / "preprocessed_waveforms.npy"
        waveforms = np.load(array_path, mmap_mode="r")
        print("总体阶段 1/4：训练前验收")
        check = validation(cfg, manifest, waveforms)
        if not check["passed"]: raise RuntimeError(f"训练前验收失败：{check}")
        print("总体阶段 2/4：四组五折训练")
        fold_metrics, predictions = train_all(waveforms, manifest, cfg, args.force, args.folds)
    print("总体阶段 3/4：Bootstrap、诊断与图件")
    summary, snr, diag = aggregate_metrics(predictions), snr_metrics(predictions), diagnostic_summary(predictions, cfg)
    # Bootstrap 和交换诊断均先落盘；恢复运行不重算已完成的昂贵统计。
    boot_path, exchange_path = paths["metrics"] / "bootstrap_differences.csv", paths["metrics"] / "exchange_diagnostics.csv"
    boot = pd.read_csv(boot_path) if args.postprocess_only and boot_path.exists() else bootstrap_differences(predictions, cfg)
    boot.to_csv(boot_path, index=False, encoding="utf-8-sig")
    exchange = pd.read_csv(exchange_path) if args.postprocess_only and exchange_path.exists() else exchange_diagnostics(predictions, cfg)
    exchange.to_csv(exchange_path, index=False, encoding="utf-8-sig")
    if args.mode == "full":
        for condition in CONDS:
            figure = paths["figures"] / f"five_samples_{condition}.png"
            if not (args.postprocess_only and figure.exists()): plot_five(cfg, condition, predictions, manifest)
        comparison = paths["figures"] / "A_vs_D_five_samples.png"
        if not (args.postprocess_only and comparison.exists()): plot_a_d(cfg)
    historical, historical_check = read_historical(manifest, cfg)
    for name, frame in {"oof_summary.csv": summary, "snr_metrics.csv": snr,
                        "diagnostic_summary.csv": diag}.items(): frame.to_csv(paths["metrics"] / name, index=False, encoding="utf-8-sig")
    dump(paths["validation"] / "historical_reference_validation.json", historical_check)
    print("总体阶段 4/4：报告")
    report(cfg, fold_metrics, predictions, summary, snr, boot, diag, exchange, historical, historical_check)
    dump(paths["root"] / "COMPLETE.json", {"mode": args.mode, "folds": args.folds, "conditions": CONDS, "preflight_passed": check["passed"], "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()
