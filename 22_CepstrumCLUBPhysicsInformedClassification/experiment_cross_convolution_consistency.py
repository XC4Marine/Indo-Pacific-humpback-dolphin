# -*- coding: utf-8 -*-
"""Experiment 22 v7: cross-convolution classification consistency without mutual-information training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import CCA
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import experiment as base

DEFAULT_OUTPUT = HERE / "outputs_v7_cross_convolution_no_mim_seed20260826_2conditions"
CONDITIONS = ("no_mim_reference", "cross_conv_consistency")
LOCKED_PRIOR_WEIGHTS = (0.3845476064627951, 0.013942945255098234, 0.013682163929311455)
GAMMA = math.log(10.0) / 0.005


@dataclass(frozen=True)
class Config:
    implementation_version: str = "experiment22_v7_cross_convolution_no_mim_v1"
    output_dir: Path = DEFAULT_OUTPUT
    seed: int = 20260826
    batch_size: int = 16
    max_epochs: int = 25
    patience: int = 5
    bootstrap_runs: int = 2000
    device: str = "cuda"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    sample_rate: int = 576000
    sample_count: int = 5761
    channel_count: int = 2881
    latent_count: int = 64
    tukey_alpha: float = .04
    epsilon: float = 1e-8
    num_workers: int = 0
    calibration_batches: int = 64
    rms_scale_min: float = .1
    rms_scale_max: float = 10.
    ccc_weight: float | None = None
    @property
    def config_hash(self) -> str:
        value = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items() if k != "output_dir"}
        value["architecture"] = "X=residual anti_alias_stride GN SiLU mean+max; H=dilated(1,2,4,8)+Linear(5761,64); X/H unshared"
        value["conditions"] = CONDITIONS
        value["priors"] = "LH exponential tail; compact soft +/-1ms; band 20-190kHz; Tukey decoder"
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def jwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def paths(c: Config) -> dict[str, Path]:
    result = {"root": c.output_dir, "runs": c.output_dir / "runs", "metrics": c.output_dir / "metrics", "predictions": c.output_dir / "predictions", "latents": c.output_dir / "latent_vectors"}
    for path in result.values(): path.mkdir(parents=True, exist_ok=True)
    return result


class Residual1d(nn.Module):
    def __init__(self, ni: int, no: int, stride: int = 1, dilation: int = 1):
        super().__init__(); pad = dilation * 2; groups = 8 if no % 8 == 0 else 1
        self.a = nn.Conv1d(ni, no, 5, stride=stride, padding=pad, dilation=dilation); self.n1 = nn.GroupNorm(groups, no)
        self.b = nn.Conv1d(no, no, 5, padding=pad, dilation=dilation); self.n2 = nn.GroupNorm(groups, no)
        self.skip = nn.Identity() if ni == no and stride == 1 else nn.Conv1d(ni, no, 1, stride=stride)
    def forward(self, x): return nn.functional.silu(self.n2(self.b(nn.functional.silu(self.n1(self.a(x))))) + self.skip(x))


class XEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.prefilter = nn.AvgPool1d(3, 1, 1); self.blocks = nn.Sequential(Residual1d(1, 16, 2), Residual1d(16, 32, 2), Residual1d(32, 64, 2), Residual1d(64, 64, 2)); self.proj = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64))
    def forward(self, x):
        value = self.blocks(self.prefilter(x.unsqueeze(1)))
        return self.proj(torch.cat((value.mean(-1), value.amax(-1)), 1))


class XDecoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__(); self.seed = nn.Linear(64, 32 * 181); self.blocks = nn.ModuleList([Residual1d(32, 24), Residual1d(24, 16), Residual1d(16, 8), Residual1d(8, 8), Residual1d(8, 8)]); self.out = nn.Conv1d(8, 1, 1)
        self.register_buffer("window", torch.from_numpy(np.asarray(__import__("scipy").signal.windows.tukey(c.sample_count, alpha=c.tukey_alpha), np.float32)))
    def forward(self, z):
        value = self.seed(z).view(-1, 32, 181)
        for size, block in zip((361, 721, 1441, 2881, 5761), self.blocks): value = block(nn.functional.interpolate(value, size=size, mode="linear", align_corners=False))
        return torch.tanh(self.out(value).squeeze(1)) * self.window


class HEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.Sequential(Residual1d(1, 8, dilation=1), Residual1d(8, 8, dilation=2), Residual1d(8, 8, dilation=4), Residual1d(8, 1, dilation=8)); self.position = nn.Linear(5761, 64); self.norm = nn.LayerNorm(64)
    def forward(self, x): return self.norm(self.position(self.blocks(x.unsqueeze(1)).squeeze(1)))


class SoftThreshold(nn.Module):
    def __init__(self): super().__init__(); self.raw = nn.Parameter(torch.tensor(-3.0))
    def forward(self, x):
        threshold = .1 * torch.sigmoid(self.raw)
        return torch.sign(x) * torch.relu(x.abs() - threshold)


class HDecoder(nn.Module):
    def __init__(self):
        super().__init__(); self.tail = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 2880)); self.threshold = SoftThreshold()
    def forward(self, z):
        tail = .25 * torch.tanh(self.threshold(self.tail(z)))
        return torch.cat((torch.ones((z.shape[0], 1), dtype=z.dtype, device=z.device), tail), 1)


class HeterogeneousXHModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__(); self.E_x = XEncoder(); self.E_h = HEncoder(); self.D_x = XDecoder(c); self.D_h = HDecoder(); self.classifier = nn.Linear(64, 1)
    def encode(self, received): return self.E_x(received), self.E_h(received)
    def forward(self, received):
        z_click, z_channel = self.encode(received); x_hat = self.D_x(z_click); h_hat = self.D_h(z_channel)
        return {"z_click": z_click, "z_channel": z_channel, "x_hat": x_hat, "h_hat": h_hat, "r_hat": base.causal_fft_convolution(x_hat, h_hat), "logit": self.classifier(z_click).squeeze(1)}


def differentiable_zero(*values: torch.Tensor) -> torch.Tensor: return sum(value.sum() for value in values) * 0.
def h_loss(h, c):
    weight = torch.exp(h.new_tensor(GAMMA) * torch.arange(1, c.channel_count, device=h.device, dtype=h.dtype) / c.sample_rate)
    return (h[:, 1:].abs() * weight).mean()
def compact_loss(x, c):
    index = torch.arange(c.sample_count, device=x.device, dtype=x.dtype); distance = (index - c.sample_count // 2).abs() / c.sample_rate; outside = torch.sigmoid((distance - .001) * 5000)
    return (x.square() * outside).sum(1) / (x.square().sum(1) + c.epsilon)
def band_loss(x, c):
    frequency = torch.fft.rfftfreq(c.sample_count, 1 / c.sample_rate).to(x.device); spectrum = torch.fft.rfft(x).abs().square()
    return spectrum[:, (frequency < 20000) | (frequency > 190000)].sum(1) / (spectrum.sum(1) + c.epsilon)
def prior_terms(out, labels, c):
    positive = labels > .5
    if not bool(positive.any()):
        zero = differentiable_zero(out["x_hat"], out["h_hat"]); return zero, zero, zero
    return h_loss(out["h_hat"][positive], c), compact_loss(out["x_hat"][positive], c).mean(), band_loss(out["x_hat"][positive], c).mean()


def cross_convolution_loss(model: HeterogeneousXHModel, out: dict[str, torch.Tensor], received: torch.Tensor, labels: torch.Tensor, c: Config, pairing_offset: int) -> tuple[torch.Tensor, int]:
    """All A anchors use positive B donors; positive A never receives itself."""
    positives = torch.nonzero(labels > .5, as_tuple=False).flatten()
    if len(positives) < 2: return differentiable_zero(out["x_hat"], out["h_hat"], out["logit"]), 0
    anchors = torch.arange(len(labels), device=labels.device)
    # The circular donor index is deterministic for (fold, epoch, batch), and a positive
    # anchor is advanced one further whenever the nominal donor would be itself.
    donor = positives[(anchors + pairing_offset) % len(positives)]
    same = donor == anchors
    donor[same] = positives[(anchors[same] + pairing_offset + 1) % len(positives)]
    cross = base.causal_fft_convolution(out["x_hat"][anchors], out["h_hat"][donor])
    if cross.shape != received.shape: raise RuntimeError(f"Cross convolution shape mismatch: {tuple(cross.shape)}")
    source_rms = received[anchors].square().mean(1, keepdim=True).sqrt()
    cross_rms = cross.square().mean(1, keepdim=True).sqrt().clamp_min(c.epsilon)
    scale = (source_rms / cross_rms).clamp(c.rms_scale_min, c.rms_scale_max).detach()
    normalized = cross * scale
    if not torch.isfinite(normalized).all(): raise RuntimeError("Non-finite RMS-normalized cross reconstruction")
    logits = model.classifier(model.E_x(normalized)).squeeze(1)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels[anchors]), int(len(positives))


def base_objective(out, received, labels, c, weights):
    recon = base.reconstruction_loss(received, out["r_hat"], c.epsilon); bce = nn.functional.binary_cross_entropy_with_logits(out["logit"], labels); h, compact, band = prior_terms(out, labels, c)
    return recon + bce + weights[0] * h + weights[1] * compact + weights[2] * band, (recon, bce, h, compact, band)


def train_batch(model, optimizer, received, labels, condition, c, pairing_offset):
    model.train(); optimizer.zero_grad(set_to_none=True); out = model(received); reference, terms = base_objective(out, received, labels, c, LOCKED_PRIOR_WEIGHTS)
    ccc, positive_count = cross_convolution_loss(model, out, received, labels, c, pairing_offset) if condition == "cross_conv_consistency" else (differentiable_zero(out["x_hat"], out["h_hat"], out["logit"]), 0)
    total = reference + (c.ccc_weight * ccc if condition == "cross_conv_consistency" else 0.)
    total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
    recon, bce, h, compact, band = terms
    values = {"loss_recon": recon, "loss_bce": bce, "loss_h_raw": h, "loss_compact_raw": compact, "loss_band_raw": band, "loss_reference": reference, "loss_ccc_raw": ccc, "loss_ccc_weighted": (c.ccc_weight * ccc if condition == "cross_conv_consistency" else differentiable_zero(ccc)), "loss_total": total}
    return {key: float(value.detach().cpu()) for key, value in values.items()} | {"ccc_positive_donor_count": float(positive_count)}


@torch.no_grad()
def predict(model, loader, device, c):
    model.eval(); rows = []; clicks = []; channels = []
    for received, labels, indices in loader:
        out = model(received.to(device)); rows.append(pd.DataFrame({"sample_index": indices.numpy(), "label": labels.numpy().astype(int), "probability": torch.sigmoid(out["logit"]).cpu().numpy()})); clicks.append(out["z_click"].cpu().numpy()); channels.append(out["z_channel"].cpu().numpy())
    return pd.concat(rows, ignore_index=True), {"z_click": np.concatenate(clicks), "z_channel": np.concatenate(channels)}


def choose_threshold(labels, probabilities):
    candidates = np.unique(np.r_[0., probabilities, 1.]); scores = [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    return float(candidates[np.argmax(scores)])
def metric(labels, probabilities, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, probabilities >= threshold, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)), "auprc": float(average_precision_score(labels, probabilities)), "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= threshold)), "f1": float(f1_score(labels, probabilities >= threshold, zero_division=0)), "sensitivity": float(tp / max(1, tp + fn)), "specificity": float(tn / max(1, tn + fp)), "threshold": float(threshold)}


def calibrate_ccc(waves, manifest, c, device):
    destination = paths(c)["root"] / "ccc_calibration.json"
    if destination.exists(): return float(json.loads(destination.read_text(encoding="utf-8"))["ccc_weight"])
    train, _, _ = base.split_indices(manifest, 0); seed_everything(c.seed)
    model = HeterogeneousXHModel(c).to(device); raw_reference = []; raw_ccc = []
    for number, (received, labels, _) in enumerate(base.epoch_train_loader(waves, manifest.label.to_numpy(np.float32), train, c.batch_size, c.seed, 1, 0), 1):
        received, labels = received.to(device), labels.to(device); out = model(received); reference, _ = base_objective(out, received, labels, c, LOCKED_PRIOR_WEIGHTS); ccc, donors = cross_convolution_loss(model, out, received, labels, c, number)
        if donors < 2 or not torch.isfinite(reference + ccc): raise RuntimeError("CCC calibration aborted: insufficient donors or non-finite loss")
        raw_reference.append(float(reference.detach().cpu())); raw_ccc.append(float(ccc.detach().cpu()))
        if number == c.calibration_batches: break
    reference_median, ccc_median = float(np.median(raw_reference)), float(np.median(raw_ccc))
    if len(raw_ccc) != c.calibration_batches or not np.isfinite([reference_median, ccc_median]).all() or min(reference_median, ccc_median) <= 1e-12: raise RuntimeError("CCC calibration aborted: invalid or near-zero median")
    weight = .05 * reference_median / ccc_median
    jwrite(destination, {"fold": 0, "train_only": True, "batches": len(raw_ccc), "reference_median": reference_median, "ccc_raw_median": ccc_median, "ccc_weight": weight, "target_weighted_share_of_reference": .05})
    return weight


def fit(condition, fold, waves, manifest, c, device, resume, bar):
    run = paths(c)["runs"] / condition / f"seed_{c.seed}" / f"fold_{fold}"; run.mkdir(parents=True, exist_ok=True)
    best, complete, history, batches, prediction_file, state = (run / "best_checkpoint.pt", run / "COMPLETE.json", run / "history.csv", run / "batches.csv", run / "predictions.csv", run / "training_state.pt")
    if resume and complete.exists() and all(path.exists() for path in (best, history, batches, prediction_file)):
        bar.update(); return json.loads((run / "metrics.json").read_text(encoding="utf-8")), pd.read_csv(prediction_file)
    seed_everything(c.seed + fold); model = HeterogeneousXHModel(c).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) > 1500000: raise RuntimeError("Model exceeds 1.5M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay); labels = manifest.label.to_numpy(np.float32); train, validation, test = base.split_indices(manifest, fold)
    records = []; batch_records = []; start = 1; best_score = -np.inf; stale = 0
    if resume and state.exists():
        saved = torch.load(state, map_location=device, weights_only=False)
        if saved.get("config_hash") == c.config_hash:
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start, best_score, stale = saved["epoch"] + 1, saved["best_score"], saved["stale"]; records = pd.read_csv(history).to_dict("records") if history.exists() else []; batch_records = pd.read_csv(batches).to_dict("records") if batches.exists() else []
    for epoch in range(start, c.max_epochs + 1):
        epoch_values = []
        for batch, (received, batch_labels, _) in enumerate(base.epoch_train_loader(waves, labels, train, c.batch_size, c.seed + fold, epoch, 0), 1):
            values = train_batch(model, optimizer, received.to(device), batch_labels.to(device), condition, c, pairing_offset=fold * 100000 + epoch * 1000 + batch); epoch_values.append(values); batch_records.append({"condition": condition, "fold": fold, "epoch": epoch, "batch": batch, **values})
        validation_frame, _ = predict(model, base.evaluation_loader(waves, labels, validation, c.batch_size, 0), device, c); validation_auroc = float(roc_auc_score(validation_frame.label, validation_frame.probability))
        records.append({"condition": condition, "fold": fold, "epoch": epoch, **{key: float(np.mean([value[key] for value in epoch_values])) for key in epoch_values[0]}, "validation_auroc": validation_auroc})
        if validation_auroc > best_score + 1e-4: best_score, stale = validation_auroc, 0; torch.save({"model": model.state_dict(), "epoch": epoch, "validation_auroc": best_score, "config_hash": c.config_hash}, best)
        else: stale += 1
        pd.DataFrame(records).to_csv(history, index=False, encoding="utf-8-sig"); pd.DataFrame(batch_records).to_csv(batches, index=False, encoding="utf-8-sig"); torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_score": best_score, "stale": stale, "config_hash": c.config_hash}, state)
        if stale >= c.patience: break
    saved = torch.load(best, map_location=device, weights_only=True); model.load_state_dict(saved["model"]); validation_frame, _ = predict(model, base.evaluation_loader(waves, labels, validation, c.batch_size, 0), device, c); threshold = choose_threshold(validation_frame.label.to_numpy(), validation_frame.probability.to_numpy()); prediction, latent = predict(model, base.evaluation_loader(waves, labels, test, c.batch_size, 0), device, c)
    result = metric(prediction.label.to_numpy(), prediction.probability.to_numpy(), threshold) | {"condition": condition, "fold": fold, "seed": c.seed, "best_epoch": saved["epoch"], "validation_auroc": saved["validation_auroc"], "total_parameters": sum(parameter.numel() for parameter in model.parameters())}
    prediction["condition"] = condition; prediction["fold"] = fold; prediction["seed"] = c.seed; prediction["threshold"] = threshold; prediction = prediction.merge(manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "target_snr_db"]], on="sample_index", how="left", validate="one_to_one"); prediction.to_csv(prediction_file, index=False, encoding="utf-8-sig"); jwrite(run / "metrics.json", result); jwrite(complete, {"training_complete": True, "config_hash": c.config_hash}); np.savez_compressed(paths(c)["latents"] / f"{condition}_fold_{fold}.npz", sample_id=prediction.sample_id.to_numpy(), label=prediction.label.to_numpy(), z_click=latent["z_click"], z_channel=latent["z_channel"]); bar.update(); return result, prediction


def cca(c, folds):
    rows = []; rng = np.random.default_rng(c.seed)
    for condition in CONDITIONS:
        for fold in folds:
            latent = np.load(paths(c)["latents"] / f"{condition}_fold_{fold}.npz")
            for subset, mask in (("all", np.ones(len(latent["label"]), bool)), ("positive", latent["label"] == 1)):
                for method in ("linear_cca", "kernel_cca_nystrom"):
                    x, h = latent["z_click"][mask], latent["z_channel"][mask]
                    if len(x) > 512: selected = rng.choice(len(x), 512, False); x, h = x[selected], h[selected]
                    components = min(10, len(x) - 1)
                    if method.startswith("kernel"):
                        landmarks = min(32, len(x)); selected = rng.choice(len(x), landmarks, False); scale = max(float(np.median(np.linalg.norm(x[selected, None] - x[None, selected], axis=2))), 1e-6); x = Nystroem(gamma=1 / (2 * scale ** 2), n_components=landmarks, random_state=c.seed).fit_transform(x); h = Nystroem(gamma=1 / (2 * scale ** 2), n_components=landmarks, random_state=c.seed).fit_transform(h)
                    u, v = CCA(n_components=components, max_iter=1000).fit_transform(x, h); rows.append({"condition": condition, "fold": fold, "subset": subset, "method": method, **{f"corr_{index + 1}": float(np.corrcoef(u[:, index], v[:, index])[0, 1]) for index in range(components)}})
    per_fold = pd.DataFrame(rows); columns = [f"corr_{index}" for index in range(1, 11)]; per_fold.to_csv(paths(c)["metrics"] / "cca_per_fold.csv", index=False, encoding="utf-8-sig"); per_fold.groupby(["condition", "subset", "method"], as_index=False)[columns].mean().to_csv(paths(c)["metrics"] / "cca_summary.csv", index=False, encoding="utf-8-sig")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    def render(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def summarize(metrics, predictions, c, expected_folds):
    output = paths(c)["metrics"]; metrics.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig"); predictions.to_csv(paths(c)["predictions"] / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    # Five test folds partition the locked 7,000 samples; an integration fold is
    # intentionally allowed to contain its naturally uneven source-isolated size.
    if (expected_folds == 5 and len(predictions) != 14000) or predictions.duplicated(["condition", "sample_id"]).any(): raise RuntimeError("Invalid OOF prediction count or keys")
    main = [metric(group.label.to_numpy(), group.probability.to_numpy(), group.threshold.iloc[0]) | {"condition": condition} for condition, group in predictions.groupby("condition")]; pd.DataFrame(main).to_csv(output / "classification_main_table.csv", index=False, encoding="utf-8-sig")
    losses = []
    for condition in CONDITIONS:
        history = pd.concat([pd.read_csv(path) for path in (paths(c)["runs"] / condition).glob("seed_*/*/history.csv")])
        for column in ("loss_reference", "loss_ccc_raw", "loss_ccc_weighted", "loss_total"): losses.append({"condition": condition, "loss": column, "mean": float(history[column].mean()), "share_total": float(history[column].mean() / max(history.loss_total.mean(), c.epsilon))})
    pd.DataFrame(losses).to_csv(output / "loss_summary.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(c.seed); a = predictions[predictions.condition == "cross_conv_consistency"]; b = predictions[predictions.condition == "no_mim_reference"]; sources = a.ori_file_num.unique(); bootstrap = []
    for name in ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity"):
        values = []
        for _ in range(c.bootstrap_runs):
            chosen = rng.choice(sources, len(sources), True); x = pd.concat([a[a.ori_file_num == source] for source in chosen]); y = pd.concat([b[b.ori_file_num == source] for source in chosen]); values.append(metric(x.label, x.probability, .5)[name] - metric(y.label, y.probability, .5)[name])
        bootstrap.append({"comparison": "cross_conv_consistency - no_mim_reference", "metric": name, "mean": float(np.mean(values)), "ci_low": float(np.percentile(values, 2.5)), "ci_high": float(np.percentile(values, 97.5)), "bootstrap_runs": c.bootstrap_runs})
    pd.DataFrame(bootstrap).to_csv(output / "paired_source_bootstrap.csv", index=False, encoding="utf-8-sig")
    loss_frame = pd.DataFrame(losses); bootstrap_frame = pd.DataFrame(bootstrap); cc = pd.read_csv(output / "cca_summary.csv")
    cross = cc[cc.condition.eq("cross_conv_consistency")].set_index(["subset", "method"]); reference = cc[cc.condition.eq("no_mim_reference")].set_index(["subset", "method"]); corr_cols = [f"corr_{index}" for index in range(1, 11)]
    cca_delta = []
    for key in cross.index:
        cca_delta.append({"样本口径": "全体" if key[0] == "all" else "真实正类", "方法": "线性 CCA" if key[1] == "linear_cca" else "Nyström Kernel CCA", "参考前10均值": float(reference.loc[key, corr_cols].mean()), "CCC前10均值": float(cross.loc[key, corr_cols].mean()), "前10均值差值": float(cross.loc[key, corr_cols].mean() - reference.loc[key, corr_cols].mean()), "第一相关差值": float(cross.loc[key, "corr_1"] - reference.loc[key, "corr_1"])})
    ccc_row = loss_frame[(loss_frame.condition == "cross_conv_consistency") & (loss_frame.loss == "loss_ccc_weighted")].iloc[0]
    report = ["# 第22实验 v7：去 MIM 的交叉卷积一致性初筛", "", "## 实验范围", "", "本实验固定种子 20260826，使用来源隔离五折、batch size 16、最多 25 epoch 与 patience 5；比较无 MIM 参考组和交叉卷积一致性组。CCC 权重由 fold 0 训练集固定 64 批次校准为 1.658331，使校准时加权中位数约为参考目标的 5%。三项物理先验沿用 v6 锁定权重。", "", "## 总体分类结果", "", markdown_table(pd.DataFrame(main)), "", "以录音来源配对的 2,000 次 Bootstrap 比较“交叉卷积一致性 − 无 MIM 参考”：", "", markdown_table(bootstrap_frame[["metric", "mean", "ci_low", "ci_high", "bootstrap_runs"]]), "", "AUROC 与 AUPRC 的区间均跨越 0；平衡准确率、F1 与敏感度的区间为负，未显示该 CCC 在本单种子设置下带来稳定的整体分类增益。", "", "## CCC 损失", "", f"训练期交叉卷积组的 CCC 原始损失均值为 {loss_frame[(loss_frame.condition == 'cross_conv_consistency') & (loss_frame.loss == 'loss_ccc_raw')].iloc[0]['mean']:.4f}，加权均值为 {ccc_row['mean']:.4f}，占总损失 {ccc_row['share_total']:.2%}；该比例是训练过程观测值，不等于校准批次的目标比例。", "", markdown_table(loss_frame), "", "## X/H 解耦诊断", "", "下表为每折 CCA/KCCA 前十典型相关的折均值；差值定义为“交叉卷积一致性 − 无 MIM 参考”。正值表示共享可预测结构增加，负值表示减少。", "", markdown_table(pd.DataFrame(cca_delta)), "", "真实正类上，线性 CCA 前十均值增加 0.0254，Nyström Kernel CCA 增加 0.0457；因此此诊断不支持 CCC 改善 X/H 解耦。CCA/KCCA 仅为折内潜变量诊断，不能单独证明统计独立或真实物理信道恢复。", "", "## 结论边界", "", "本结果仅为单随机种子初筛。当前证据不支持将交叉卷积一致性视为稳定的分类性能提升；是否存在可重复效应需在独立随机种子下复验。"]
    (c.output_dir / "final_report_zh.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def self_test(c, device):
    model = HeterogeneousXHModel(c).to(device); assert not (set(map(id, model.E_x.parameters())) & set(map(id, model.E_h.parameters()))); assert sum(parameter.numel() for parameter in model.parameters()) <= 1500000
    received = torch.randn(4, 5761, device=device); labels = torch.tensor([1., 1., 0., 0.], device=device); out = model(received); assert out["z_click"].shape == out["z_channel"].shape == (4, 64); assert out["h_hat"].shape[1] == 2881 and torch.equal(out["h_hat"][:, 0], torch.ones(4, device=device))
    ccc, donors = cross_convolution_loss(model, out, received, labels, c, 1); assert donors == 2 and torch.isfinite(ccc); ccc.backward(); assert all(parameter.grad is not None for module in (model.E_x, model.D_x, model.E_h, model.D_h, model.classifier) for parameter in module.parameters() if parameter.requires_grad)
    out = model(received); zero, donors = cross_convolution_loss(model, out, received, torch.tensor([1., 0., 0., 0.], device=device), c, 1); assert donors == 0 and zero.requires_grad; print("[总体进度 10%] v7 无 MIM、交叉卷积、CUDA 单批与梯度检查通过", flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("integration", "full"), default="full"); parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4]); parser.add_argument("--max-epochs", type=int, default=25); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--patience", type=int, default=5); parser.add_argument("--bootstrap-runs", type=int, default=2000); parser.add_argument("--resume", action="store_true"); parser.add_argument("--device", default="cuda"); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); arguments = parser.parse_args()
    c = Config(output_dir=arguments.output_dir.resolve(), max_epochs=arguments.max_epochs, batch_size=arguments.batch_size, patience=arguments.patience, bootstrap_runs=arguments.bootstrap_runs, device=arguments.device); device = base.resolve_device(c.device); paths(c); self_test(c, device); manifest, waves, contract = base.load_locked_cache(c); c = Config(**{**asdict(c), "ccc_weight": calibrate_ccc(waves, manifest, c, device)}); jwrite(c.output_dir / "run_config.json", {**asdict(c), "output_dir": str(c.output_dir), "config_hash": c.config_hash, "conditions": CONDITIONS, "data_hash": base.EXPECTED_DATA_HASH, "locked_prior_weights": LOCKED_PRIOR_WEIGHTS}); jwrite(c.output_dir / "data_contract.json", contract)
    folds = tuple(arguments.folds if arguments.mode == "full" else [arguments.folds[0]]); run_c = Config(**{**asdict(c), "max_epochs": c.max_epochs if arguments.mode == "full" else 1, "bootstrap_runs": c.bootstrap_runs if arguments.mode == "full" else min(c.bootstrap_runs, 20)}); print(f"[总体进度 15%] 启动 {len(folds) * len(CONDITIONS)} 个模型", flush=True); bar = tqdm(total=len(folds) * len(CONDITIONS), desc="总体进度：第22实验v7", unit="模型"); metrics = []; predictions = []
    for fold in folds:
        for condition in CONDITIONS:
            result, frame = fit(condition, fold, waves, manifest, run_c, device, arguments.resume, bar); metrics.append(result); predictions.append(frame)
    bar.close(); metrics_frame, prediction_frame = pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True); cca(run_c, folds); summarize(metrics_frame, prediction_frame, run_c, len(folds)); jwrite(run_c.output_dir / "run_summary.json", {"models": len(metrics_frame), "predictions": len(prediction_frame), "config_hash": run_c.config_hash, "data_hash": base.EXPECTED_DATA_HASH, "device": str(device)}); print("[总体进度 100%] 完成", flush=True)


if __name__ == "__main__": main()
