#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 21 v7: two-component exponentially damped sinusoid (EDS) source branch.

The script deliberately reuses only the v4 fixed-data contract and free causal h decoder.
It contains no TEO, Gaussian-shape, historical h-physics, or tail-energy loss.
"""
from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(V4))
from pipeline import (BranchFeatureEncoder, CausalResponseDecoder, FullLengthDecoder,
                      SharedEncoder, WaveformDataset, causal_fft_convolution,
                      preprocess_waveform, read_waveform, split_indices)

SEED = 20260826
CONDS = ("A_free", "B_free_gain", "C_eds", "D_eds_gain")
NAMES = {"A_free": "A：自由波形 x，无增益", "B_free_gain": "B：自由波形 x，有增益",
         "C_eds": "C：双分量 EDS x，无增益", "D_eds_gain": "D：双分量 EDS x，有增益"}


@dataclass(frozen=True)
class Config:
    output_dir: Path = HERE / "outputs_seed20260826"
    source_data_dir: Path = V4_DATA
    sample_rate: int = 576000
    sample_count: int = 5761
    response_count: int = 1153
    lowcut_hz: float = 20000.
    highcut_hz: float = 190000.
    filter_order: int = 5
    tukey_alpha: float = .04
    epsilon: float = 1e-8
    seed: int = SEED
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 25
    patience: int = 5
    bootstrap_runs: int = 2000
    num_workers: int = 0


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def dirs(cfg: Config) -> dict[str, Path]:
    paths = {name: cfg.output_dir / name for name in
             ("models", "logs", "metrics", "predictions", "figures", "waves", "fold_configs", "validation")}
    paths["root"] = cfg.output_dir
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    return paths


def use_gain(condition: str) -> bool: return condition in ("B_free_gain", "D_eds_gain")
def use_eds(condition: str) -> bool: return condition in ("C_eds", "D_eds_gain")


class GainHead(nn.Module):
    def __init__(self, epsilon: float) -> None:
        super().__init__(); self.layers = nn.Sequential(nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 1)); self.epsilon = epsilon

    def initialize(self, l2_median: float) -> None:
        value = max(float(l2_median) - self.epsilon, self.epsilon)
        inv_softplus = math.log(math.expm1(value)) if value < 40 else value
        with torch.no_grad():
            for layer in (self.layers[0], self.layers[-1]): layer.weight.zero_(); layer.bias.zero_()
            self.layers[-1].bias.fill_(inv_softplus)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((latent.mean(-1), latent.amax(-1)), 1)
        return nn.functional.softplus(self.layers(pooled)[:, 0]) + self.epsilon


class EDSHead(nn.Module):
    """256 -> 64 -> 10; parameters are [A, delay, rho, frequency, phase] x 2."""
    def __init__(self, cfg: Config, bounds: dict[str, float]) -> None:
        super().__init__(); self.cfg = cfg; self.bounds = bounds
        self.layers = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 10))
        self.initialize()

    @staticmethod
    def inv_sigmoid(value: float) -> float:
        value = min(max(value, 1e-5), 1 - 1e-5)
        return math.log(value / (1 - value))

    def initialize(self) -> None:
        b = self.bounds; delta_lo, delta_hi = b["delay_low_samples"], b["delay_high_samples"]
        rho_lo, rho_hi = b["rho_low_samples"], b["rho_high_samples"]
        init_delays = (b["delay_q01_median_samples"], b["delay_q99_median_samples"])
        # Clip makes init robust when the q01/q99 medians touch the fold bounds.
        init_delays = tuple(min(max(x, delta_lo), delta_hi) for x in init_delays)
        with torch.no_grad():
            self.layers[0].weight.zero_(); self.layers[0].bias.zero_(); self.layers[-1].weight.zero_()
            bias = torch.zeros(2, 5)
            bias[:, 0] = torch.tensor([math.atanh(.75 / 2), math.atanh(.25 / 2)])
            bias[:, 1] = torch.tensor([self.inv_sigmoid((x - delta_lo) / max(delta_hi - delta_lo, 1e-6)) for x in init_delays])
            rho_pos = (b["rho_median_samples"] - rho_lo) / max(rho_hi - rho_lo, 1e-6)
            bias[:, 2] = self.inv_sigmoid(rho_pos)
            for k, name in enumerate(("frequency_init_low_hz", "frequency_init_high_hz")):
                bias[k, 3] = self.inv_sigmoid((b[name] - 20000.) / 170000.)
            bias[:, 4] = torch.tensor([0., math.atanh(.5)])  # raw tanh values -> [0, pi/2]
            self.layers[-1].bias.copy_(bias.flatten())

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.cat((latent.mean(-1), latent.amax(-1)), 1)
        raw = self.layers(pooled).reshape(-1, 2, 5); b = self.bounds
        params = torch.empty_like(raw)
        params[..., 0] = 2 * torch.tanh(raw[..., 0])
        params[..., 1] = b["delay_low_samples"] + (b["delay_high_samples"] - b["delay_low_samples"]) * torch.sigmoid(raw[..., 1])
        params[..., 2] = b["rho_low_samples"] + (b["rho_high_samples"] - b["rho_low_samples"]) * torch.sigmoid(raw[..., 2])
        params[..., 3] = 20000. + 170000. * torch.sigmoid(raw[..., 3])
        params[..., 4] = math.pi * torch.tanh(raw[..., 4])
        # Ordering moves each component's entire parameter tuple, not delay values alone.
        order = params[..., 1].argsort(dim=1)
        params = torch.gather(params, 1, order[..., None].expand_as(params))
        t = torch.arange(self.cfg.sample_count, device=latent.device, dtype=latent.dtype)[None, None, :]
        amp, delay, rho, freq, phase = (params[..., i, None] for i in range(5))
        age = t - delay
        component = amp * (age >= 0).to(latent.dtype) * torch.exp(-age.clamp_min(0) / rho.clamp_min(self.cfg.epsilon)) * torch.cos(2 * math.pi * freq * age.clamp_min(0) / self.cfg.sample_rate + phase)
        return component.sum(1), params, component


class V7Model(nn.Module):
    def __init__(self, cfg: Config, condition: str, bounds: dict[str, float]) -> None:
        super().__init__(); self.cfg, self.condition = cfg, condition
        self.shared_encoder = SharedEncoder(); self.source_decoder = FullLengthDecoder(cfg.sample_count)
        self.eds_head = EDSHead(cfg, bounds); self.response_decoder = CausalResponseDecoder(cfg.response_count)
        self.gain_head = GainHead(cfg.epsilon); self.source_classifier = BranchFeatureEncoder()
        self.classification_head = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))

    def forward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.shared_encoder(y); free = self.source_decoder(latent)
        eds_source, eds_params, eds_components = self.eds_head(latent)
        source_raw = eds_source if use_eds(self.condition) else free
        h_shape = self.response_decoder(latent); gain = self.gain_head(latent)
        if use_gain(self.condition):
            source = source_raw / torch.linalg.vector_norm(source_raw, dim=1, keepdim=True).clamp_min(self.cfg.epsilon)
            response = gain[:, None] * h_shape
        else:
            source, response, gain = source_raw, h_shape, torch.ones_like(gain)
        reconstruction = causal_fft_convolution(source, response)
        logit = self.classification_head(self.source_classifier(source)).squeeze(-1)
        return {"logit": logit, "source": source, "source_raw": source_raw, "free_source": free,
                "eds_params": eds_params, "eds_components": eds_components, "h_shape": h_shape,
                "gain": gain, "response": response, "reconstruction": reconstruction}


def recon_loss(y: torch.Tensor, reconstruction: torch.Tensor, eps: float) -> torch.Tensor:
    return ((y - reconstruction).square().sum(1) / (y.square().sum(1) + eps)).mean()


def losses(out: dict[str, torch.Tensor], y: torch.Tensor, label: torch.Tensor, cfg: Config) -> dict[str, torch.Tensor]:
    rec = recon_loss(y, out["reconstruction"], cfg.epsilon); bce = nn.functional.binary_cross_entropy_with_logits(out["logit"], label)
    return {"total": rec + bce, "reconstruction": rec, "bce": bce}


def grad_norm(module: nn.Module) -> float:
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    choices = np.unique(np.r_[0., .5, 1., probability])
    return float(choices[np.argmax([balanced_accuracy_score(labels, probability >= value) for value in choices])])


def metrics(labels: np.ndarray, probability: np.ndarray, cut: float | np.ndarray) -> dict[str, float]:
    pred = probability >= cut; tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probability)), "auprc": float(average_precision_score(labels, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)), "f1": float(f1_score(labels, pred, zero_division=0)),
            "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1))}


def fold_eds_bounds(manifest: pd.DataFrame, train_index: np.ndarray, cfg: Config) -> dict[str, float]:
    """Estimate all learned EDS time limits from unnoised positive training clicks only."""
    positive = manifest.iloc[train_index]; positive = positive[positive.label.eq(1)]
    starts: list[float] = []; ends: list[float] = []; durations: list[float] = []; rhos: list[float] = []
    spectra: list[np.ndarray] = []
    for _, row in positive.iterrows():
        x = preprocess_waveform(read_waveform(row, cfg), cfg).astype(np.float64)
        energy = x ** 2; total = energy.sum()
        if not np.isfinite(total) or total <= cfg.epsilon: continue
        cdf = np.cumsum(energy) / total; start = float(np.searchsorted(cdf, .01)); end = float(np.searchsorted(cdf, .99))
        duration = max(end - start, 1.); starts.append(start); ends.append(end); durations.append(duration); rhos.append(duration / math.log(20))
        spectra.append(np.abs(np.fft.rfft(x)) ** 2)
    if not durations: raise RuntimeError("训练折没有可用的未加噪正样本 Click，无法估计 EDS 边界")
    # q05(start), q95(end) is the conservative interpretation of the requested 1%-99% pulse window.
    delay_low, delay_high = np.percentile(starts, 5), np.percentile(ends, 95)
    rho_low, rho_high = np.percentile(rhos, [10, 90]); rho_low = max(rho_low, 2.); rho_high = min(max(rho_high, rho_low), cfg.sample_rate / 1000.)
    mean_spec = np.mean(np.stack(spectra), axis=0); fcdf = np.cumsum(mean_spec) / max(mean_spec.sum(), cfg.epsilon)
    freq = np.fft.rfftfreq(cfg.sample_count, 1 / cfg.sample_rate)
    f1, f2 = (float(freq[min(np.searchsorted(fcdf, q), len(freq) - 1)]) for q in (1 / 3, 2 / 3))
    f1, f2 = np.clip([f1, f2], 20000., 190000.)
    return {"estimated_click_count": int(len(durations)), "delay_low_samples": float(delay_low), "delay_high_samples": float(max(delay_high, delay_low + 1)),
            "rho_low_samples": float(rho_low), "rho_high_samples": float(rho_high), "rho_median_samples": float(np.median(rhos)),
            "delay_q01_median_samples": float(np.median(starts)), "delay_q99_median_samples": float(np.median(ends)),
            "duration_median_samples": float(np.median(durations)), "frequency_init_low_hz": float(f1), "frequency_init_high_hz": float(f2),
            "bounds_source": "unnoised_training_positive_clicks_only", "delay_rule": "q05(1pct_energy_time), q95(99pct_energy_time)",
            "rho_rule": "q10/q90((t99-t01)/ln(20)), clipped to [2 samples, 1 ms]"}


def loaders(waveforms: np.ndarray, labels: np.ndarray, tr: np.ndarray, va: np.ndarray, te: np.ndarray, cfg: Config, fold: int) -> dict[str, DataLoader]:
    common = dict(num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    generator = torch.Generator().manual_seed(cfg.seed + fold)
    return {"train": DataLoader(WaveformDataset(waveforms, labels, tr), cfg.batch_size, shuffle=True, generator=generator, **common),
            "validation": DataLoader(WaveformDataset(waveforms, labels, va), cfg.batch_size * 2, shuffle=False, **common),
            "test": DataLoader(WaveformDataset(waveforms, labels, te), cfg.batch_size * 2, shuffle=False, **common)}


def diagnostics(out: dict[str, torch.Tensor], y: torch.Tensor, cfg: Config) -> dict[str, np.ndarray]:
    shape = out["h_shape"]; energy = shape.square().sum(1).clamp_min(cfg.epsilon); idx = torch.arange(shape.shape[1], device=y.device)[None]
    p = out["eds_params"]; components = out["eds_components"]
    comp_energy = components.square().sum(-1); total_comp_energy = comp_energy.sum(1).clamp_min(cfg.epsilon)
    nrmse = ((out["reconstruction"] - y).square().sum(1) / (y.square().sum(1) + cfg.epsilon)).sqrt()
    return {"nrmse": nrmse.detach().cpu().numpy(), "gain": out["gain"].detach().cpu().numpy(),
            "r0": (shape[:, 0].square() / energy).detach().cpu().numpy(), "h_shape_tail_energy_ratio": (shape[:, 1:].square().sum(1) / energy).detach().cpu().numpy(),
            "h_energy_delay_ms": ((idx * shape.square()).sum(1) / energy * 1000 / cfg.sample_rate).detach().cpu().numpy(),
            "eds_a1": p[:, 0, 0].detach().cpu().numpy(), "eds_delta1": p[:, 0, 1].detach().cpu().numpy(), "eds_rho1": p[:, 0, 2].detach().cpu().numpy(), "eds_f1": p[:, 0, 3].detach().cpu().numpy(), "eds_phi1": p[:, 0, 4].detach().cpu().numpy(),
            "eds_a2": p[:, 1, 0].detach().cpu().numpy(), "eds_delta2": p[:, 1, 1].detach().cpu().numpy(), "eds_rho2": p[:, 1, 2].detach().cpu().numpy(), "eds_f2": p[:, 1, 3].detach().cpu().numpy(), "eds_phi2": p[:, 1, 4].detach().cpu().numpy(),
            "eds_component1_energy_fraction": (comp_energy[:, 0] / total_comp_energy).detach().cpu().numpy(), "eds_component2_energy_fraction": (comp_energy[:, 1] / total_comp_energy).detach().cpu().numpy()}


@torch.no_grad()
def predict(model: V7Model, loader: DataLoader, device: torch.device, cfg: Config, arrays: bool = False) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    model.eval(); rows: list[pd.DataFrame] = []; saved: list[dict[str, np.ndarray]] = []
    for y, label, index in loader:
        y, label = y.to(device), label.to(device); out = model(y); diag = diagnostics(out, y, cfg)
        rows.append(pd.DataFrame({"sample_index": index.numpy(), "label": label.cpu().numpy().astype(int), "probability": torch.sigmoid(out["logit"]).cpu().numpy(), **diag}))
        if arrays: saved.append({key: value.detach().cpu().numpy() for key, value in out.items() if key != "logit"} | {"observed": y.cpu().numpy(), "sample_index": index.numpy()})
    return pd.concat(rows, ignore_index=True), saved


def pair_distance(shape: np.ndarray, seed: int) -> float:
    if len(shape) < 2: return float("nan")
    normed = shape / np.linalg.norm(shape, axis=1, keepdims=True).clip(1e-12)
    rng = np.random.default_rng(seed); a = rng.integers(0, len(shape), min(10000, 4 * len(shape))); b = rng.integers(0, len(shape), len(a)); keep = a != b
    return float(np.linalg.norm(normed[a[keep]] - normed[b[keep]], axis=1).mean())


def fit(condition: str, fold: int, waveforms: np.ndarray, manifest: pd.DataFrame, cfg: Config, force: bool, bar: tqdm) -> tuple[dict[str, Any], pd.DataFrame]:
    paths, device = dirs(cfg), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = split_indices(manifest, fold); bounds_path = paths["fold_configs"] / f"eds_bounds_fold_{fold}.json"
    bounds = json.loads(bounds_path.read_text(encoding="utf-8")) if bounds_path.exists() else fold_eds_bounds(manifest, tr, cfg)
    dump(bounds_path, bounds); labels = manifest.label.to_numpy(np.float32)
    # Same condition-independent seed and re-created loader give all four groups the same shared-module start and batches.
    seed_all(cfg.seed + fold); model = V7Model(cfg, condition, bounds).to(device)
    median_l2 = float(np.median(np.linalg.norm(np.asarray(waveforms[tr]), axis=1))); model.gain_head.initialize(median_l2)
    ld = loaders(waveforms, labels, tr, va, te, cfg, fold); stem = f"{condition}_seed_{cfg.seed}_fold_{fold}"
    ck, hp, bp = paths["models"] / f"{stem}.pt", paths["logs"] / f"{stem}_history.csv", paths["logs"] / f"{stem}_batch.csv"
    ready = ck.exists() and hp.exists() and bp.exists() and not force
    if ready:
        previous = torch.load(ck, map_location="cpu", weights_only=True)
        ready = int(previous.get("training_max_epochs", -1)) == cfg.max_epochs
    if not ready:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay); best, stale, history, batches, started = -np.inf, 0, [], [], time.time()
        for epoch in tqdm(range(1, cfg.max_epochs + 1), desc=f"{condition} 折{fold}", leave=False):
            model.train(); total = {k: 0. for k in ("total", "reconstruction", "bce")}; n = 0
            for batch, (y, label, _) in enumerate(ld["train"], 1):
                y, label = y.to(device), label.to(device); opt.zero_grad(set_to_none=True); out = model(y); loss = losses(out, y, label, cfg); loss["total"].backward()
                record = {k: float(v.detach().cpu()) for k, v in loss.items()} | {"condition": condition, "fold": fold, "epoch": epoch, "batch": batch,
                    "free_source_gradient_norm": grad_norm(model.source_decoder), "eds_head_gradient_norm": grad_norm(model.eds_head), "response_gradient_norm": grad_norm(model.response_decoder), "gain_gradient_norm": grad_norm(model.gain_head)}
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); batches.append(record); n += 1
                for k in total: total[k] += record[k]
            valid, _ = predict(model, ld["validation"], device, cfg); score = float(roc_auc_score(valid.label, valid.probability))
            history.append({"condition": condition, "fold": fold, "epoch": epoch, "validation_auroc": score, **{k: v / max(n, 1) for k, v in total.items()}, "elapsed_seconds": time.time()-started})
            if score > best + 1e-4:
                best, stale = score, 0; torch.save({"state_dict": model.state_dict(), "best_epoch": epoch, "validation_auroc": score, "training_seconds": time.time()-started, "training_max_epochs": cfg.max_epochs, "condition": condition, "bounds": bounds}, ck)
            else:
                stale += 1
                if stale >= cfg.patience: break
        pd.DataFrame(history).to_csv(hp, index=False, encoding="utf-8-sig"); pd.DataFrame(batches).to_csv(bp, index=False, encoding="utf-8-sig")
    saved = torch.load(ck, map_location=device, weights_only=True); model.load_state_dict(saved["state_dict"])
    valid, _ = predict(model, ld["validation"], device, cfg); cut = threshold(valid.label.to_numpy(), valid.probability.to_numpy())
    test, arrays = predict(model, ld["test"], device, cfg, arrays=True); test["threshold"], test["condition"], test["seed"], test["test_fold"] = cut, condition, cfg.seed, fold
    for key in ("delay_low_samples", "delay_high_samples", "rho_low_samples", "rho_high_samples"): test[key] = bounds[key]
    np.savez_compressed(paths["waves"] / f"{condition}_fold_{fold}_test.npz", **{key: np.concatenate([x[key] for x in arrays]) for key in arrays[0]})
    shape = np.concatenate([x["h_shape"] for x in arrays]); value = metrics(test.label.to_numpy(), test.probability.to_numpy(), cut)
    value |= {"condition": condition, "fold": fold, "best_epoch": int(saved["best_epoch"]), "validation_auroc": float(saved["validation_auroc"]), "threshold": cut, "training_seconds": float(saved["training_seconds"]), "nrmse": float(test.nrmse.mean()), "h_shape_effective_rank": effective_rank(shape), "h_shape_normalized_pair_distance": pair_distance(shape, cfg.seed + fold), **bounds}
    bar.update(1); bar.set_postfix(condition=condition, fold=fold, auroc=f"{value['auroc']:.4f}")
    return value, test


def effective_rank(x: np.ndarray) -> float:
    s = np.linalg.svd(x, compute_uv=False); p = s / max(s.sum(), 1e-12); return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def train_all(waveforms: np.ndarray, manifest: pd.DataFrame, cfg: Config, force: bool, folds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, tests = [], []; bar = tqdm(total=len(CONDS) * len(folds), desc="总体进度：20 次条件×折训练", unit="条件折")
    for fold in folds:
        for condition in CONDS:
            value, test = fit(condition, fold, waveforms, manifest, cfg, force, bar); values.append(value); tests.append(test)
    bar.close(); fold_metrics, predictions = pd.DataFrame(values), pd.concat(tests, ignore_index=True)
    meta = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    predictions = predictions.merge(meta, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
    paths = dirs(cfg); fold_metrics.to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig"); predictions.to_csv(paths["predictions"] / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    return fold_metrics, predictions


def aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, p in predictions.groupby("condition"):
        rows.append({"condition": condition, **metrics(p.label.to_numpy(), p.probability.to_numpy(), p.threshold.to_numpy()), "nrmse": float(p.nrmse.mean())})
    return pd.DataFrame(rows)


def snr_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, p in predictions.groupby("condition"):
        neg = p[p.label.eq(0)]
        for lo in range(-15, 15, 5):
            pos = p[(p.label.eq(1)) & p.target_snr_db.ge(lo) & (p.target_snr_db.le(lo+5) if lo == 10 else p.target_snr_db.lt(lo+5))]; joined = pd.concat((neg, pos))
            rows.append({"condition": condition, "snr_bin_db": f"[{lo},{lo+5}{']' if lo == 10 else ')'}", "n_positive": len(pos), **metrics(joined.label.to_numpy(), joined.probability.to_numpy(), joined.threshold.to_numpy()), "nrmse": float(joined.nrmse.mean())})
    return pd.DataFrame(rows)


def structure_values(p: pd.DataFrame, fold_metrics: pd.DataFrame) -> dict[str, float]:
    pos = p[p.label.eq(1)]; f = fold_metrics[fold_metrics.condition.eq(p.condition.iloc[0])]
    return {"r0": float(pos.r0.mean()), "tail": float(pos.h_shape_tail_energy_ratio.mean()), "distance": float(f.h_shape_normalized_pair_distance.mean()), "rank": float(f.h_shape_effective_rank.mean())}


def group_structure_metrics(condition: str, predictions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One independent structural statistic per original recording.

    This makes the structure criteria compatible with the stipulated paired,
    original-recording Bootstrap.  Distance is the mean distance between
    normalized h_shape pairs within a recording; exchange keeps source and its
    gain fixed and rotates only h_shape within that same recording.
    """
    subset = predictions[predictions.condition.eq(condition)]
    records: list[dict[str, float]] = []
    for fold in sorted(subset.test_fold.unique()):
        data = np.load(dirs(cfg)["waves"] / f"{condition}_fold_{int(fold)}_test.npz")
        order = {int(index): i for i, index in enumerate(data["sample_index"])}
        fold_rows = subset[subset.test_fold.eq(fold)]
        for origin, rows in fold_rows.groupby("ori_file_num"):
            pos = np.array([order[int(index)] for index in rows.sample_index], dtype=int)
            h, x, g, y = data["h_shape"][pos], data["source"][pos], data["gain"][pos], data["observed"][pos]
            norm = h / np.linalg.norm(h, axis=1, keepdims=True).clip(1e-12)
            if len(pos) > 1:
                rng = np.random.default_rng(cfg.seed + int(fold) + int(origin))
                a = rng.integers(0, len(pos), min(1000, len(pos) * 4)); b = rng.integers(0, len(pos), len(a)); keep = a != b
                distance = float(np.linalg.norm(norm[a[keep]] - norm[b[keep]], axis=1).mean()) if keep.any() else 0.0
                perm = np.roll(np.arange(len(pos)), 1)
            else:
                distance, perm = 0.0, np.arange(1)
            own = causal_fft_convolution(torch.from_numpy(x), torch.from_numpy(g[:, None] * h)).numpy()
            swapped = causal_fft_convolution(torch.from_numpy(x), torch.from_numpy(g[:, None] * h[perm])).numpy()
            own_mse = np.mean((y - own) ** 2, axis=1); swapped_mse = np.mean((y - swapped) ** 2, axis=1)
            records.append({"ori_file_num": int(origin), "r0": float(rows.r0.mean()), "tail": float(rows.h_shape_tail_energy_ratio.mean()),
                            "distance": distance, "exchange": float(np.abs(swapped_mse - own_mse).mean())})
    return pd.DataFrame(records).set_index("ori_file_num").sort_index()


def bootstrap(predictions: pd.DataFrame, fold_metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    comparisons = (("C_eds", "A_free", "C_minus_A"), ("D_eds_gain", "B_free_gain", "D_minus_B"), ("B_free_gain", "A_free", "B_minus_A"), ("D_eds_gain", "C_eds", "D_minus_C"), ("D_eds_gain", "A_free", "D_minus_A"))
    rng = np.random.default_rng(cfg.seed + 901); rows = []
    for left_name, right_name, title in comparisons:
        left, right = predictions.query("condition == @left_name"), predictions.query("condition == @right_name"); groups = np.array(sorted(left.ori_file_num.unique()))
        samples: dict[str, list[dict[str, float]]] = {key: [] for key in ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity", "nrmse", "r0", "tail", "distance", "exchange")}
        observed_l, observed_r = metrics(left.label.to_numpy(), left.probability.to_numpy(), left.threshold.to_numpy()), metrics(right.label.to_numpy(), right.probability.to_numpy(), right.threshold.to_numpy())
        observed = {key: observed_l[key] - observed_r[key] for key in observed_l} | {"nrmse": float(left.nrmse.mean()-right.nrmse.mean())}
        left_struct, right_struct = group_structure_metrics(left_name, predictions, cfg), group_structure_metrics(right_name, predictions, cfg)
        if not left_struct.index.equals(right_struct.index) or not left_struct.index.equals(pd.Index(groups)):
            raise RuntimeError(f"结构诊断的原始录音配对不完整：{title}")
        for key in ("r0", "tail", "distance", "exchange"): observed[key] = float(left_struct[key].mean() - right_struct[key].mean())
        for _ in tqdm(range(cfg.bootstrap_runs), desc=f"Bootstrap {title}", leave=False):
            picked = rng.choice(groups, len(groups), replace=True); l = pd.concat([left[left.ori_file_num.eq(g)] for g in picked]); r = pd.concat([right[right.ori_file_num.eq(g)] for g in picked])
            lm, rm = metrics(l.label.to_numpy(), l.probability.to_numpy(), l.threshold.to_numpy()), metrics(r.label.to_numpy(), r.probability.to_numpy(), r.threshold.to_numpy())
            for key in lm: samples[key].append(lm[key]-rm[key])
            samples["nrmse"].append(float(l.nrmse.mean()-r.nrmse.mean()))
            for key in ("r0", "tail", "distance", "exchange"):
                samples[key].append(float(left_struct.loc[picked, key].mean() - right_struct.loc[picked, key].mean()))
        for key, vals in samples.items(): rows.append({"comparison": title, "metric": key, "estimate": observed[key], "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5)), "bootstrap_runs": cfg.bootstrap_runs})
    return pd.DataFrame(rows)


def eds_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, p in predictions.groupby("condition"):
        low_d, high_d, low_r, high_r = (p.delay_low_samples, p.delay_high_samples, p.rho_low_samples, p.rho_high_samples)
        rows.append({"condition": condition, "n": len(p), "second_component_active_fraction": float((p.eds_component2_energy_fraction > .05).mean()), "delay_sorted_fraction": float((p.eds_delta1 <= p.eds_delta2).mean()),
                     "delay_boundary_fraction": float((np.isclose(p.eds_delta1, low_d, atol=1) | np.isclose(p.eds_delta2, high_d, atol=1)).mean()), "rho_boundary_fraction": float((np.isclose(p.eds_rho1, low_r, atol=1) | np.isclose(p.eds_rho2, high_r, atol=1)).mean()),
                     "frequency_boundary_fraction": float(((p.eds_f1 < 20100) | (p.eds_f1 > 189900) | (p.eds_f2 < 20100) | (p.eds_f2 > 189900)).mean()), "a1_median": float(p.eds_a1.median()), "a2_median": float(p.eds_a2.median()), "component2_energy_median": float(p.eds_component2_energy_fraction.median())})
    return pd.DataFrame(rows)


def plot_five(cfg: Config, condition: str, predictions: pd.DataFrame, manifest: pd.DataFrame) -> None:
    p = predictions[predictions.condition.eq(condition)].copy(); ids = sorted(p.sample_index)[:5]; data = np.load(dirs(cfg)["waves"] / f"{condition}_fold_0_test.npz")
    # Fixed samples are defined globally; retrieve each sample from its corresponding fold NPZ.
    selected = []
    for idx in ids:
        row = p[p.sample_index.eq(idx)].iloc[0]; d = np.load(dirs(cfg)["waves"] / f"{condition}_fold_{int(row.test_fold)}_test.npz"); pos = int(np.where(d["sample_index"] == idx)[0][0]); selected.append((row, {k: d[k][pos] for k in d.files}))
    t = np.arange(cfg.sample_count) * 1000 / cfg.sample_rate; th = np.arange(cfg.response_count) * 1000 / cfg.sample_rate; fig, axes = plt.subplots(5, 6, figsize=(24, 15))
    for i, (r, d) in enumerate(selected):
        axes[i,0].plot(t,d["source"],lw=.55); axes[i,0].set_title("x" if i==0 else "")
        axes[i,1].plot(t,d["eds_components"][0],lw=.55,label="EDS1"); axes[i,1].plot(t,d["eds_components"][1],lw=.55,label="EDS2"); axes[i,1].legend(fontsize=6); axes[i,1].set_title("EDS components" if i==0 else "")
        axes[i,2].plot(th,d["h_shape"],lw=.55); axes[i,2].set_title("h_shape" if i==0 else "")
        axes[i,3].plot(t,d["reconstruction"],lw=.55); axes[i,3].set_title("reconstruction" if i==0 else "")
        axes[i,4].plot(t,d["observed"],lw=.55); axes[i,4].plot(t,d["reconstruction"],lw=.55); axes[i,4].set_title("observed / reconstruction" if i==0 else "")
        axes[i,5].axis("off"); axes[i,5].text(.02,.95,f"idx={r.sample_index}\nP={r.probability:.3f}, T={r.threshold:.3f}\ng={r.gain:.3f}\n1: A={r.eds_a1:.2f}, d={r.eds_delta1:.1f}, rho={r.eds_rho1:.1f}\nf={r.eds_f1/1000:.1f}k, phi={r.eds_phi1:.2f}\n2: A={r.eds_a2:.2f}, d={r.eds_delta2:.1f}, rho={r.eds_rho2:.1f}\nf={r.eds_f2/1000:.1f}k, phi={r.eds_phi2:.2f}",va="top",fontsize=7)
        for a in axes[i,:5]: a.grid(alpha=.2); a.tick_params(labelsize=6)
    fig.suptitle(NAMES[condition]); fig.tight_layout(rect=(0,0,1,.97)); base=dirs(cfg)["figures"] / f"five_samples_{condition}"; fig.savefig(base.with_suffix(".png"),dpi=180); fig.savefig(base.with_suffix(".pdf")); plt.close(fig)


def plot_a_d(cfg: Config) -> None:
    a, d = plt.imread(dirs(cfg)["figures"] / "five_samples_A_free.png"), plt.imread(dirs(cfg)["figures"] / "five_samples_D_eds_gain.png")
    fig, axes = plt.subplots(2,1,figsize=(24,30)); axes[0].imshow(a); axes[1].imshow(d)
    for ax, title in zip(axes,(NAMES["A_free"],NAMES["D_eds_gain"])): ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(dirs(cfg)["figures"] / "A_vs_D_five_samples.png",dpi=150); fig.savefig(dirs(cfg)["figures"] / "A_vs_D_five_samples.pdf"); plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty: return "（无数据）"
    def cell(x: Any) -> str: return f"{x:.4f}" if isinstance(x,(float,np.floating)) and np.isfinite(x) else ("NA" if isinstance(x,(float,np.floating)) else str(x))
    return "\n".join(["| "+" | ".join(frame.columns)+" |", "| "+" | ".join(["---"]*len(frame.columns))+" |"]+["| "+" | ".join(cell(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None)])


def report(cfg: Config, folds: pd.DataFrame, pred: pd.DataFrame, summary: pd.DataFrame, snr: pd.DataFrame, boot: pd.DataFrame, eds: pd.DataFrame) -> None:
    def b(comp: str, metric: str) -> pd.Series: return boot[(boot.comparison.eq(comp)) & (boot.metric.eq(metric))].iloc[0]
    ca, db = b("C_minus_A","auroc"), b("D_minus_B","auroc")
    criteria = []
    for comp in ("C_minus_A","D_minus_B"):
        z = {k:b(comp,k) for k in ("r0","tail","distance","exchange")}; passed=[z["r0"].ci_high < 0, z["tail"].ci_low > 0, z["distance"].ci_low > 0, z["exchange"].ci_low > 0]; criteria.append((comp,sum(passed),passed))
    text=f"""# 实验 21 v7——双分量 EDS 参数化 x 源分支

本报告是种子 {cfg.seed} 的固定数据、原始录音隔离五折结果；单种子结果不表示跨种子稳定性。EDS 仅是受限源先验，并不被表述为真实 Click 声源模型。

## 分类与重建

{markdown_table(folds[["condition","fold","best_epoch","validation_auroc","auroc","auprc","balanced_accuracy","f1","sensitivity","specificity","nrmse","threshold"]])}

### 合并折外

{markdown_table(summary)}

### 原始录音配对 Bootstrap（{cfg.bootstrap_runs} 次）

{markdown_table(boot)}

### 分 SNR

{markdown_table(snr)}

## EDS 与 h 诊断

{markdown_table(eds)}

分类非劣要求 C−A 与 D−B 的 AUROC 95% 区间下界均不低于 −0.005。C−A={ca.estimate:+.4f} [{ca.ci_low:+.4f},{ca.ci_high:+.4f}]；D−B={db.estimate:+.4f} [{db.ci_low:+.4f},{db.ci_high:+.4f}]。h 结构需在同一比较至少两项方向一致（R0 降低、尾能量/距离/仅交换 h_shape 的绝对重建变化提高）：{criteria}。

EDS 的参数分布、两分量能量占比、第二分量活跃率、延迟排序及边界触及率均见诊断表；正负样本请分别在逐样本 CSV 中比较，不以汇总均值掩盖参数塌缩。

若分类未非劣或结构改善少于两项，本数据与本单种子下仅应报告该受限源先验未获支持，不扩展搜索或更改边界。
"""
    (dirs(cfg)["root"] / "final_report_zh.md").write_text(text,encoding="utf-8")


def validation(cfg: Config, manifest: pd.DataFrame, waveforms: np.ndarray) -> dict[str, Any]:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); seed_all(cfg.seed); tr,_,_=split_indices(manifest,0); bounds=fold_eds_bounds(manifest,tr,cfg); model=V7Model(cfg,"D_eds_gain",bounds).to(device); model.gain_head.initialize(float(np.median(np.linalg.norm(np.asarray(waveforms[:128]),axis=1))))
    y=torch.from_numpy(np.asarray(waveforms[:3])).to(device); label=torch.tensor([1.,0.,1.],device=device); out=model(y); loss=losses(out,y,label,cfg); p=out["eds_params"]
    model.zero_grad(set_to_none=True); nn.functional.binary_cross_entropy_with_logits(out["logit"],label).backward(retain_graph=True); bce={"eds_head":grad_norm(model.eds_head),"response":grad_norm(model.response_decoder),"gain":grad_norm(model.gain_head)}
    model.zero_grad(set_to_none=True); recon_loss(y,model(y)["reconstruction"],cfg.epsilon).backward(); rec={"eds_head":grad_norm(model.eds_head),"response":grad_norm(model.response_decoder),"gain":grad_norm(model.gain_head)}
    x=torch.zeros(1,cfg.sample_count,device=device); h=torch.zeros(1,cfg.response_count,device=device); x[0,4]=1; h[0,0]=1; h[0,5]=.25; direct=torch.zeros_like(x)
    for k in (0, 5): direct[:,k:] += h[:,k,None] * x[:,:cfg.sample_count-k]
    groups=[]
    for f in range(5):
        a,b,c=split_indices(manifest,f); gs=[set(manifest.iloc[z].ori_file_num) for z in (a,b,c)]; groups.append(not(gs[0]&gs[1]) and not(gs[0]&gs[2]) and not(gs[1]&gs[2]))
    result={"data_rows":len(manifest),"waveform_shape":list(waveforms.shape),"finite_waveforms":bool(np.isfinite(waveforms).all()),"source_isolation":groups,"eds_finite":bool(torch.isfinite(p).all()),"delay_sorted":bool((p[:,0,1]<=p[:,1,1]).all()),"frequency_under_nyquist":bool((p[...,3] < cfg.sample_rate/2).all()),"unit_energy_x":bool(torch.allclose(torch.linalg.vector_norm(out["source"],dim=1),torch.ones(3,device=device),atol=2e-5)),"h0_equals_g":bool(torch.allclose(out["response"][:,0],out["gain"],atol=1e-6)),"hshape0":bool(torch.equal(out["h_shape"][:,0],torch.ones(3,device=device))),"causal_convolution_error":float((causal_fft_convolution(x,h)-direct).abs().max().cpu()),"bce_gradients":bce,"reconstruction_gradients":rec,"losses_finite":{k:bool(torch.isfinite(v)) for k,v in loss.items()},"bounds":bounds}
    result["passed"]=bool(result["data_rows"]==7000 and result["waveform_shape"]==[7000,cfg.sample_count] and result["finite_waveforms"] and all(groups) and result["eds_finite"] and result["delay_sorted"] and result["frequency_under_nyquist"] and result["unit_energy_x"] and result["h0_equals_g"] and result["hshape0"] and result["causal_convolution_error"]<1e-6 and bce["eds_head"]>0 and bce["response"]==0 and bce["gain"]==0 and rec["eds_head"]>0 and rec["response"]>0 and rec["gain"]>0 and all(result["losses_finite"].values()))
    dump(dirs(cfg)["validation"] / "preflight_validation.json",result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("smoke","full"),required=True); parser.add_argument("--folds",nargs="+",type=int,default=list(range(5))); parser.add_argument("--force",action="store_true"); parser.add_argument("--postprocess-only",action="store_true"); parser.add_argument("--output-dir",type=Path,default=None); args=parser.parse_args()
    output_dir = args.output_dir if args.output_dir is not None else (HERE / "smoke_check" if args.mode == "smoke" else Config().output_dir)
    cfg=Config(output_dir=output_dir,bootstrap_runs=30 if args.mode=="smoke" else 2000,max_epochs=2 if args.mode=="smoke" else 25,patience=1 if args.mode=="smoke" else 5)
    if args.mode=="smoke": args.folds=[0]
    if set(args.folds)-set(range(5)): raise ValueError("folds must be in 0..4")
    paths=dirs(cfg); dump(paths["root"] / "run_config.json",{**asdict(cfg),"conditions":CONDS,"folds":args.folds,"data_contract":str(cfg.source_data_dir),"source_parameterization":"two-component EDS only for C/D; free source only for A/B","loss":"relative reconstruction MSE + BCE only"})
    manifest=pd.read_csv(cfg.source_data_dir / "dataset_manifest.csv"); waveforms=np.load(cfg.source_data_dir / "cache" / "preprocessed_waveforms.npy",mmap_mode="r")
    print("总体阶段 1/4：训练前验收"); check=validation(cfg,manifest,waveforms)
    if not check["passed"]: raise RuntimeError(f"训练前验收失败: {check}")
    print("总体阶段 2/4：四组五折训练"); folds,pred=train_all(waveforms,manifest,cfg,args.force,args.folds)
    print("总体阶段 3/4：Bootstrap、诊断与图件"); summary,snr,eds=aggregate(pred),snr_metrics(pred),eds_summary(pred); boot=bootstrap(pred,folds,cfg)
    for name,frame in {"oof_summary.csv":summary,"snr_metrics.csv":snr,"eds_diagnostics.csv":eds,"bootstrap_differences.csv":boot}.items(): frame.to_csv(paths["metrics"] / name,index=False,encoding="utf-8-sig")
    if args.mode=="full":
        for condition in CONDS: plot_five(cfg,condition,pred,manifest)
        plot_a_d(cfg)
    print("总体阶段 4/4：报告"); report(cfg,folds,pred,summary,snr,boot,eds); dump(paths["root"] / "COMPLETE.json",{"mode":args.mode,"folds":args.folds,"conditions":CONDS,"preflight_passed":check["passed"],"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__": main()
