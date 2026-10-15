#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-seed, positive-only cosine-orthogonality ablation for experiment 22."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

import experiment as base


EXPERIMENT_DIR = base.EXPERIMENT_DIR
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs_orthogonal_cosine_positive_seed20260826_6weights"
DEFAULT_WEIGHTS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
METRIC_NAMES = base.METRIC_NAMES


@dataclass(frozen=True)
class OrthConfig:
    implementation_version: str = "experiment22_positive_cosine_orthogonal_v1"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    recon_weight: float = 1.0
    bce_weight: float = 1.0
    orth_weights: tuple[float, ...] = DEFAULT_WEIGHTS
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    random_seed: int = 20_260_826
    num_workers: int = 0
    epsilon: float = 1e-8
    device: str = "auto"
    max_epochs: int = 25
    patience: int = 5
    min_delta: float = 1e-4
    bootstrap_runs: int = 2_000
    orth_definition: str = "positive_only_mean_squared_cosine"

    @property
    def config_hash(self) -> str:
        payload = {key: str(value) if isinstance(value, Path) else value
                   for key, value in asdict(self).items() if key != "output_dir"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def serializable_config(config: OrthConfig) -> dict[str, Any]:
    result = {key: str(value) if isinstance(value, Path) else value
              for key, value in asdict(config).items()}
    result.update({
        "config_hash": config.config_hash,
        "label_definition": {"1": "Indo-Pacific humpback dolphin click", "0": "hard noise"},
        "method_boundary": (
            "Binary click-versus-noise classification; cosine orthogonality is an angular "
            "separation diagnostic, not statistical independence or physical channel recovery."
        ),
    })
    return result


def output_paths(config: OrthConfig) -> dict[str, Path]:
    paths = {"root": config.output_dir, "runs": config.output_dir / "runs",
             "metrics": config.output_dir / "metrics", "predictions": config.output_dir / "predictions"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def weight_key(weight: float) -> str:
    return f"orth_{weight:g}".replace(".", "p")


def weight_label(weight: float) -> str:
    return f"lambda_orth={weight:g}"


def model_run_dir(config: OrthConfig, weight: float, fold: int) -> Path:
    path = output_paths(config)["runs"] / weight_key(weight) / f"seed_{config.random_seed}" / f"fold_{fold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def positive_cosine_orthogonality_loss(
    z_click: torch.Tensor, z_channel: torch.Tensor, labels: torch.Tensor, epsilon: float = 1e-8
) -> tuple[torch.Tensor, int, bool]:
    positive = labels > 0.5
    count = int(positive.sum().item())
    if count == 0:
        return (z_click.sum() + z_channel.sum()) * 0.0, count, False
    numerator = torch.sum(z_click[positive] * z_channel[positive], dim=1)
    denominator = (torch.linalg.vector_norm(z_click[positive], dim=1) *
                   torch.linalg.vector_norm(z_channel[positive], dim=1)).clamp_min(epsilon)
    return (numerator / denominator).square().mean(), count, True


def train_orthogonal_batch(
    model: base.CepstrumCLUBModel,
    optimizer: torch.optim.Optimizer,
    received: torch.Tensor,
    labels: torch.Tensor,
    orth_weight: float,
    config: OrthConfig,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(received)
    loss_recon = base.reconstruction_loss(received, outputs["r_hat"], config.epsilon)
    loss_bce = nn.functional.binary_cross_entropy_with_logits(outputs["logit"], labels)
    loss_orth_raw, positive_count, applied = positive_cosine_orthogonality_loss(
        outputs["z_click"], outputs["z_channel"], labels, config.epsilon
    )
    loss_orth_weighted = orth_weight * loss_orth_raw
    total = config.recon_weight * loss_recon + config.bce_weight * loss_bce + loss_orth_weighted
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return {
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_bce": float(loss_bce.detach().cpu()),
        "loss_orth_raw": float(loss_orth_raw.detach().cpu()),
        "loss_orth_weighted": float(loss_orth_weighted.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "orth_positive_count": float(positive_count),
        "orth_applied": float(applied),
    }


@torch.no_grad()
def predict_orthogonal(
    model: base.CepstrumCLUBModel, loader: torch.utils.data.DataLoader, device: torch.device, epsilon: float
) -> pd.DataFrame:
    model.eval()
    frames: list[pd.DataFrame] = []
    for received, labels, indices in loader:
        received_device = received.to(device, non_blocking=True)
        outputs = model(received_device)
        count = len(labels)
        positive = labels.to(device=device) > 0.5
        cosine_squared = np.full(count, np.nan)
        if bool(positive.any()):
            numerator = torch.sum(outputs["z_click"][positive] * outputs["z_channel"][positive], dim=1)
            denominator = (torch.linalg.vector_norm(outputs["z_click"][positive], dim=1) *
                           torch.linalg.vector_norm(outputs["z_channel"][positive], dim=1)).clamp_min(epsilon)
            cosine_squared[positive.cpu().numpy()] = (numerator / denominator).square().cpu().numpy()
        centered_left = outputs["x_hat"] - outputs["x_hat"].mean(1, keepdim=True)
        centered_right = received_device - received_device.mean(1, keepdim=True)
        ncc = (torch.sum(centered_left * centered_right, 1) /
               (torch.linalg.vector_norm(centered_left, dim=1) *
                torch.linalg.vector_norm(centered_right, dim=1) + epsilon)).cpu().numpy()
        tails = outputs["h_hat"][:, 1:]
        frames.append(pd.DataFrame({
            "sample_index": indices.numpy(), "label": labels.numpy().astype(int),
            "probability": torch.sigmoid(outputs["logit"]).cpu().numpy(),
            "cosine_squared": cosine_squared, "ncc": ncc,
            "channel_tail_energy": tails.square().sum(1).cpu().numpy(),
            "response_diversity": (tails - tails.mean(0, keepdim=True)).square().mean(1).sqrt().cpu().numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def parameter_audit(model: nn.Module) -> dict[str, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    return {"main_total_parameters": total, "q_parameters": 0,
            "total_parameters": total, "trainable_parameters": total}


def save_training_state(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_score: float,
    best_epoch: int, stale: int, weight: float, fold: int, config: OrthConfig
) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
        "orth_weight": weight, "fold": fold, "config_hash": config.config_hash,
        "python_rng": base.random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def restore_training_state(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, config: OrthConfig, device: torch.device
) -> tuple[int, float, int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("config_hash") != config.config_hash:
        raise RuntimeError(f"Resume config mismatch: {path}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    base.random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["epoch"]) + 1, float(state["best_score"]), int(state["best_epoch"]), int(state["stale"])


def restore_best_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return state


def fit_weight_fold(
    weight: float, fold: int, waveforms: np.ndarray, manifest: pd.DataFrame, config: OrthConfig,
    device: torch.device, resume: bool, overall: tqdm | None
) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = model_run_dir(config, weight, fold)
    complete_path, best_path = run_dir / "COMPLETE.json", run_dir / "best_checkpoint.pt"
    state_path, history_path = run_dir / "training_state.pt", run_dir / "history.csv"
    batch_path, prediction_path, metrics_path = run_dir / "batches.csv", run_dir / "predictions.csv", run_dir / "metrics.json"
    required = (best_path, history_path, batch_path, prediction_path, metrics_path)
    if resume and complete_path.exists() and all(path.exists() for path in required):
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if marker.get("config_hash") == config.config_hash:
            if overall is not None:
                overall.update(1)
            return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(prediction_path)

    base.seed_everything(config.random_seed + fold)
    model = base.CepstrumCLUBModel().to(device)
    audit = parameter_audit(model)
    if audit["total_parameters"] > 900_000:
        raise RuntimeError(f"Parameter limit exceeded: {audit}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch, best_score, best_epoch, stale = 1, -np.inf, 0, 0
    resume_incomplete = False
    if resume and state_path.exists():
        header = torch.load(state_path, map_location="cpu", weights_only=False)
        resume_incomplete = header.get("config_hash") == config.config_hash
        del header
    history_rows = pd.read_csv(history_path).to_dict("records") if resume_incomplete and history_path.exists() else []
    batch_rows = pd.read_csv(batch_path).to_dict("records") if resume_incomplete and batch_path.exists() else []
    if resume_incomplete:
        start_epoch, best_score, best_epoch, stale = restore_training_state(
            state_path, model, optimizer, config, device
        )

    labels = manifest["label"].to_numpy(np.float32)
    train_indices, validation_indices, test_indices = base.split_indices(manifest, fold)
    validation_loader = base.evaluation_loader(waveforms, labels, validation_indices, config.batch_size, config.num_workers)
    test_loader = base.evaluation_loader(waveforms, labels, test_indices, config.batch_size, config.num_workers)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, batch_seconds = time.perf_counter(), []
    epochs = tqdm(range(start_epoch, config.max_epochs + 1), desc=f"orth={weight:g} fold {fold}", leave=False, unit="epoch")
    for epoch in epochs:
        loader = base.epoch_train_loader(waveforms, labels, train_indices, config.batch_size,
                                         config.random_seed + fold, epoch, config.num_workers)
        epoch_losses: list[dict[str, float]] = []
        for batch_number, (received, batch_labels, _) in enumerate(loader, start=1):
            received, batch_labels = received.to(device, non_blocking=True), batch_labels.to(device, non_blocking=True)
            batch_started = time.perf_counter()
            losses = train_orthogonal_batch(model, optimizer, received, batch_labels, weight, config)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds.append(time.perf_counter() - batch_started)
            epoch_losses.append(losses)
            batch_rows.append({"orth_weight": weight, "seed": config.random_seed, "fold": fold,
                               "epoch": epoch, "batch": batch_number, **losses,
                               "batch_seconds": batch_seconds[-1]})
        validation = predict_orthogonal(model, validation_loader, device, config.epsilon)
        score = float(base.roc_auc_score(validation["label"], validation["probability"]))
        means = {name: float(np.mean([row[name] for row in epoch_losses])) for name in epoch_losses[0]}
        history_rows.append({"orth_weight": weight, "seed": config.random_seed, "fold": fold,
                             "epoch": epoch, **means, "validation_auroc": score})
        if score > best_score + config.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "orth_weight": weight, "fold": fold,
                        "best_epoch": best_epoch, "validation_auroc": best_score,
                        "config_hash": config.config_hash, "parameter_audit": audit}, best_path)
        else:
            stale += 1
        pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(batch_rows).to_csv(batch_path, index=False, encoding="utf-8-sig")
        save_training_state(state_path, model, optimizer, epoch, best_score, best_epoch, stale, weight, fold, config)
        epochs.set_postfix(val_auroc=f"{score:.4f}", best=f"{best_score:.4f}", stale=stale)
        if stale >= config.patience:
            break

    saved = restore_best_checkpoint(best_path, model, device)
    validation = predict_orthogonal(model, validation_loader, device, config.epsilon)
    threshold = base.choose_validation_threshold(validation["label"].to_numpy(), validation["probability"].to_numpy())
    test = predict_orthogonal(model, test_loader, device, config.epsilon)
    metrics: dict[str, Any] = base.binary_metrics(test["label"].to_numpy(), test["probability"].to_numpy(), threshold)
    metrics.update({"orth_weight": weight, "orth_label": weight_label(weight), "seed": config.random_seed,
                    "test_fold": fold, "validation_fold": (fold + 1) % 5,
                    "best_epoch": int(saved["best_epoch"]), "validation_auroc": float(saved["validation_auroc"]),
                    **audit, "training_seconds": time.perf_counter() - started,
                    "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
                    "mean_batch_seconds": float(np.mean(batch_seconds)),
                    "mean_cosine_squared": float(test.loc[test["label"] == 1, "cosine_squared"].mean()),
                    "mean_ncc": float(test["ncc"].mean()),
                    "mean_channel_tail_energy": float(test["channel_tail_energy"].mean()),
                    "mean_response_diversity": float(test["response_diversity"].mean())})
    test["orth_weight"], test["seed"], test["test_fold"], test["threshold"] = weight, config.random_seed, fold, threshold
    metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num", "target_snr_db"]]
    test = test.merge(metadata, on="sample_index", how="left", validate="one_to_one")
    test.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    base.json_dump(metrics_path, metrics)
    base.json_dump(complete_path, {"training_complete": True, "config_hash": config.config_hash,
                                   "best_checkpoint": str(best_path.resolve())})
    if overall is not None:
        overall.update(1)
        overall.set_postfix(weight=weight, fold=fold, auroc=f"{metrics['auroc']:.4f}")
    return metrics, test


def summarize_results(metrics: pd.DataFrame, predictions: pd.DataFrame, config: OrthConfig) -> dict[str, pd.DataFrame]:
    paths = output_paths(config)
    rows = []
    for weight, group in predictions.groupby("orth_weight", sort=True):
        pooled = base.frame_metrics(group)
        folds = metrics[metrics["orth_weight"] == weight]
        row: dict[str, Any] = {"orth_weight": weight, "orth_label": weight_label(weight), **pooled}
        for metric in METRIC_NAMES:
            row[f"{metric}_fold_sd"] = float(folds[metric].std(ddof=1))
        row.update({"training_seconds": float(folds["training_seconds"].sum()),
                    "peak_memory_mb": float(folds["peak_memory_mb"].max()),
                    "mean_batch_seconds": float(folds["mean_batch_seconds"].mean())})
        rows.append(row)
    main = pd.DataFrame(rows)
    main.to_csv(paths["metrics"] / "weight_sweep_table.csv", index=False, encoding="utf-8-sig")

    diagnostic_rows = []
    for weight, group in metrics.groupby("orth_weight", sort=True):
        for name in ("mean_cosine_squared", "mean_ncc", "mean_channel_tail_energy", "mean_response_diversity"):
            diagnostic_rows.append({"orth_weight": weight, "diagnostic": name,
                                    "mean": float(group[name].mean()), "fold_sd": float(group[name].std(ddof=1)),
                                    "finite_count": int(np.isfinite(group[name]).sum())})
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(paths["metrics"] / "orthogonality_diagnostics.csv", index=False, encoding="utf-8-sig")

    snr_rows = []
    bins = np.arange(-15.0, 15.0 + 5.0, 5.0)
    for weight, frame in predictions.groupby("orth_weight", sort=True):
        negatives = frame[frame["label"] == 0]
        for lower, upper in zip(bins[:-1], bins[1:]):
            positives = frame[(frame["label"] == 1) & (frame["target_snr_db"] >= lower) &
                              (frame["target_snr_db"] < upper if upper < 15 else frame["target_snr_db"].le(upper))]
            combined = pd.concat([positives, negatives], ignore_index=True)
            snr_rows.append({"orth_weight": weight, "snr_lower_db": lower, "snr_upper_db": upper,
                             "positive_count": len(positives), "auroc": float(base.roc_auc_score(combined["label"], combined["probability"])),
                             "sensitivity": float((positives["probability"] >= positives["threshold"]).mean())})
    snr = pd.DataFrame(snr_rows)
    snr.to_csv(paths["metrics"] / "snr_table.csv", index=False, encoding="utf-8-sig")
    paired = paired_source_bootstrap(predictions, config)
    paired.to_csv(paths["metrics"] / "paired_source_bootstrap.csv", index=False, encoding="utf-8-sig")
    return {"main": main, "diagnostics": diagnostics, "snr": snr, "paired": paired}


def paired_source_bootstrap(predictions: pd.DataFrame, config: OrthConfig) -> pd.DataFrame:
    baseline = predictions[predictions["orth_weight"] == 0.0]
    sources = baseline["ori_file_num"].unique()
    rng = np.random.default_rng(config.random_seed)
    rows = []
    for weight in [value for value in config.orth_weights if value != 0.0]:
        candidate = predictions[predictions["orth_weight"] == weight]
        point_left, point_right = base.frame_metrics(candidate), base.frame_metrics(baseline)
        values = {metric: [] for metric in METRIC_NAMES}
        for _ in tqdm(range(config.bootstrap_runs), desc=f"orth={weight:g} source Bootstrap", leave=False):
            sampled_sources = rng.choice(sources, len(sources), replace=True)
            left = base.source_resample(candidate, sampled_sources)
            right = base.source_resample(baseline, sampled_sources)
            if left["label"].nunique() < 2 or right["label"].nunique() < 2:
                continue
            left_metrics, right_metrics = base.frame_metrics(left), base.frame_metrics(right)
            for metric in METRIC_NAMES:
                values[metric].append(left_metrics[metric] - right_metrics[metric])
        for metric in METRIC_NAMES:
            interval = np.percentile(values[metric], [2.5, 97.5])
            rows.append({"orth_weight": weight, "metric": metric,
                         "weight_minus_zero": float(point_left[metric] - point_right[metric]),
                         "ci_low": float(interval[0]), "ci_high": float(interval[1]),
                         "bootstrap_runs_valid": len(values[metric])})
    paired = pd.DataFrame(rows)
    recommendations = []
    for weight in [value for value in config.orth_weights if value != 0.0]:
        current = paired[paired["orth_weight"] == weight].set_index("metric")
        recommended = bool(current.loc["auprc", "ci_low"] > 0.0 and current.loc["auroc", "ci_low"] >= 0.0)
        recommendations.append({"orth_weight": weight, "recommended_over_zero": recommended})
    return paired.merge(pd.DataFrame(recommendations), on="orth_weight", how="left")


def write_preliminary_report(tables: dict[str, pd.DataFrame], config: OrthConfig) -> Path:
    def markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        def render(value: Any) -> str:
            if pd.isna(value):
                return "NA"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6f}"
            return str(value).replace("|", "\\|")
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        rows = ["| " + " | ".join(render(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None)]
        return "\n".join([header, separator, *rows])

    main = tables["main"].copy()
    paired = tables["paired"]
    candidates = paired[paired["recommended_over_zero"]] if not paired.empty else paired
    selected = "lambda_orth=0"
    if not candidates.empty:
        selected = weight_label(float(candidates["orth_weight"].iloc[0]))
    lines = [
        "# 第22实验 v4：正类余弦正交正则初步结果",
        "",
        "本实验使用随机种子 20260826、来源隔离五折和 7,000 条固定样本，比较六个正类余弦正交权重。训练不包含 MIM 或 Q 网络；负类参与 BCE 与接收信号重建，但不参与正交损失。",
        "",
        "## 权重扫描主表",
        "",
        markdown(main),
        "",
        "## 与零权重的来源配对 Bootstrap",
        "",
        markdown(paired),
        "",
        "## 初步选择",
        "",
        f"按预设规则，当前选择为 `{selected}`。该结论仅用于单种子权重初筛。余弦正交反映两个潜变量的方向分离，不等同于统计独立或真实物理信道恢复。",
    ]
    path = config.output_dir / "preliminary_report_zh.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_artifacts(
    config: OrthConfig, folds: tuple[int, ...], metrics: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, Any]:
    expected_models = len(config.orth_weights) * len(folds)
    expected_predictions = len(config.orth_weights) * 7_000 if set(folds) == set(range(5)) else None
    run_dirs = [model_run_dir(config, weight, fold) for weight in config.orth_weights for fold in folds]
    complete_headers = [json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
                        for path in run_dirs if (path / "COMPLETE.json").exists()]
    checks = {
        "expected_models": expected_models,
        "best_checkpoint_count": sum((path / "best_checkpoint.pt").exists() for path in run_dirs),
        "complete_marker_count": sum((path / "COMPLETE.json").exists() for path in run_dirs),
        "history_count": sum((path / "history.csv").exists() for path in run_dirs),
        "batch_log_count": sum((path / "batches.csv").exists() for path in run_dirs),
        "fold_metric_rows": len(metrics), "prediction_rows": len(predictions),
        "expected_prediction_rows": expected_predictions,
        "prediction_keys_unique": not predictions.duplicated(["orth_weight", "sample_id"]).any(),
        "fold_metrics_finite": bool(np.isfinite(metrics[list(METRIC_NAMES)].to_numpy()).all()),
        "required_prediction_finite": bool(np.isfinite(predictions[["label", "probability", "threshold"]].to_numpy()).all()),
        "negative_cosine_missing": bool(predictions.loc[predictions["label"] == 0, "cosine_squared"].isna().all()),
        "positive_cosine_finite": bool(np.isfinite(predictions.loc[predictions["label"] == 1, "cosine_squared"]).all()),
        "complete_config_hashes_match": bool(len(complete_headers) == expected_models and
                                              all(item.get("config_hash") == config.config_hash for item in complete_headers)),
    }
    checks["passed"] = bool(
        checks["best_checkpoint_count"] == expected_models and checks["complete_marker_count"] == expected_models and
        checks["history_count"] == expected_models and checks["batch_log_count"] == expected_models and
        checks["fold_metric_rows"] == expected_models and
        (expected_predictions is None or checks["prediction_rows"] == expected_predictions) and
        checks["prediction_keys_unique"] and checks["fold_metrics_finite"] and checks["required_prediction_finite"] and
        checks["negative_cosine_missing"] and checks["positive_cosine_finite"] and checks["complete_config_hashes_match"]
    )
    base.json_dump(config.output_dir / "artifact_validation.json", checks)
    return checks


def run_full(config: OrthConfig, folds: tuple[int, ...], resume: bool) -> dict[str, Any]:
    started = time.perf_counter()
    paths = output_paths(config)
    manifest, waveforms, contract = base.load_locked_cache(config)
    manifest.to_csv(paths["root"] / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    base.json_dump(paths["root"] / "data_contract.json", contract)
    base.json_dump(paths["root"] / "run_config.json", {**serializable_config(config), "folds": list(folds), "resume": resume})
    device = base.resolve_device(config.device)
    overall = tqdm(total=len(config.orth_weights) * len(folds), desc="orthogonal sweep", unit="model")
    metrics_rows, prediction_rows = [], []
    for weight in config.orth_weights:
        for fold in folds:
            metric, prediction = fit_weight_fold(weight, fold, waveforms, manifest, config, device, resume, overall)
            metrics_rows.append(metric)
            prediction_rows.append(prediction)
    overall.close()
    metrics, predictions = pd.DataFrame(metrics_rows), pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"] / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    tables = summarize_results(metrics, predictions, config)
    report = write_preliminary_report(tables, config)
    artifacts = verify_artifacts(config, folds, metrics, predictions)
    result = {"mode": "full", "data_hash": base.EXPECTED_DATA_HASH, "config_hash": config.config_hash,
              "models": len(metrics), "predictions": len(predictions), "device": str(device),
              "report": str(report.resolve()), "artifact_validation": artifacts,
              "elapsed_seconds": time.perf_counter() - started}
    base.json_dump(paths["root"] / "run_summary.json", result)
    if not artifacts["passed"]:
        raise RuntimeError(f"Orthogonal artifact acceptance failed: {artifacts}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--orth-weights", nargs="+", type=float, default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--seed", type=int, default=20_260_826)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bootstrap-runs", type=int, default=2_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.batch_size < 1 or arguments.max_epochs < 1 or arguments.patience < 1 or arguments.bootstrap_runs < 1:
        raise ValueError("batch-size, max-epochs, patience, and bootstrap-runs must be positive")
    if any(not math.isfinite(value) or value < 0 for value in arguments.orth_weights):
        raise ValueError("orth-weights must be finite and nonnegative")
    if len(set(arguments.orth_weights)) != len(arguments.orth_weights) or 0.0 not in arguments.orth_weights:
        raise ValueError("orth-weights must be unique and include 0")
    if any(fold not in range(5) for fold in arguments.folds) or len(set(arguments.folds)) != len(arguments.folds):
        raise ValueError("folds must be unique values from 0 through 4")
    config = OrthConfig(output_dir=arguments.output_dir.resolve(), recon_weight=arguments.recon_weight,
                        bce_weight=arguments.bce_weight, orth_weights=tuple(arguments.orth_weights),
                        learning_rate=arguments.learning_rate, batch_size=arguments.batch_size,
                        random_seed=arguments.seed, num_workers=arguments.num_workers, device=arguments.device,
                        max_epochs=arguments.max_epochs, patience=arguments.patience,
                        bootstrap_runs=arguments.bootstrap_runs)
    result = run_full(config, tuple(arguments.folds), arguments.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
