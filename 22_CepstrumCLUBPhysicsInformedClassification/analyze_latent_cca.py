#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fold-wise CCA and Nyström RBF Kernel CCA for experiment-22 OOF latents."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh
from tqdm.auto import tqdm

import experiment as base


OUTPUT_DIR = base.EXPERIMENT_DIR / "outputs_latent_cca_three_methods_seed20260826"
V3_OUTPUT = base.FULL_OUTPUT_DIR
V4_OUTPUT = base.EXPERIMENT_DIR / "outputs_orthogonal_cosine_positive_seed20260826_6weights"
SEED = 20_260_826
FOLDS = (0, 1, 2, 3, 4)
COMPONENTS = 10
RCCA_RIDGE = 1e-4
KCCA_RIDGE = 1e-3
LANDMARKS = 256


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    root: Path
    run_relative: str
    expected_hash: str

    def checkpoint(self, fold: int) -> Path:
        return self.root / "runs" / self.run_relative.format(fold=fold) / "best_checkpoint.pt"


METHODS = (
    MethodSpec("recon_bce", "Recon + BCE (without MIM)", V3_OUTPUT,
               "without_mim/seed_20260826/fold_{fold}", "3ec745dac7713453"),
    MethodSpec("positive_mim", "Recon + BCE + positive-only MIM (weight 1.0)", V3_OUTPUT,
               "full/seed_20260826/fold_{fold}", "3ec745dac7713453"),
    MethodSpec("orthogonal_0p3", "Recon + BCE + cosine orthogonality (lambda 0.3)", V4_OUTPUT,
               "orth_0p3/seed_20260826/fold_{fold}", "5c9dadf53bd2ed97"),
)


def _standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    centered = value - value.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=1, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def regularized_cca(x: np.ndarray, y: np.ndarray, components: int = COMPONENTS,
                    ridge: float = RCCA_RIDGE) -> np.ndarray:
    """Return regularized canonical correlations for paired rows of two feature matrices."""
    x, y = _standardize(x), _standardize(y)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("CCA requires at least three paired samples")
    divisor = len(x) - 1
    cxx = x.T @ x / divisor + ridge * np.eye(x.shape[1])
    cyy = y.T @ y / divisor + ridge * np.eye(y.shape[1])
    cxy = x.T @ y / divisor
    dx, ux = eigh(cxx)
    dy, uy = eigh(cyy)
    inv_x = (ux * (1.0 / np.sqrt(np.maximum(dx, 1e-12)))) @ ux.T
    inv_y = (uy * (1.0 / np.sqrt(np.maximum(dy, 1e-12)))) @ uy.T
    values = np.linalg.svd(inv_x @ cxy @ inv_y, compute_uv=False)
    return np.clip(values[:components], 0.0, 1.0)


def median_rbf_gamma(value: np.ndarray, seed: int) -> float:
    """RBF gamma=1/median(squared distance), using a bounded deterministic sample."""
    rng = np.random.default_rng(seed)
    count = min(len(value), 1_024)
    chosen = rng.choice(len(value), size=count, replace=False)
    sample = np.asarray(value[chosen], dtype=np.float64)
    squared = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=-1)
    distances = squared[np.triu_indices(count, k=1)]
    median = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0
    return 1.0 / max(median, 1e-12)


def nystrom_rbf_features(value: np.ndarray, landmarks: int, gamma: float, seed: int) -> np.ndarray:
    """Deterministic Nyström map for the RBF kernel, centered after expansion."""
    rng = np.random.default_rng(seed)
    count = min(landmarks, len(value))
    index = rng.choice(len(value), size=count, replace=False)
    anchors = np.asarray(value[index], dtype=np.float64)
    distances_nm = np.sum((value[:, None, :] - anchors[None, :, :]) ** 2, axis=-1)
    distances_mm = np.sum((anchors[:, None, :] - anchors[None, :, :]) ** 2, axis=-1)
    kernel_nm = np.exp(-gamma * distances_nm)
    kernel_mm = np.exp(-gamma * distances_mm)
    eigenvalues, eigenvectors = eigh(kernel_mm)
    transform = (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, 1e-10)))) @ eigenvectors.T
    return kernel_nm @ transform


def kernel_cca(x: np.ndarray, y: np.ndarray, components: int = COMPONENTS, landmarks: int = LANDMARKS,
               ridge: float = KCCA_RIDGE, seed: int = SEED) -> tuple[np.ndarray, float, float]:
    gamma_x, gamma_y = median_rbf_gamma(x, seed), median_rbf_gamma(y, seed + 1)
    phi_x = nystrom_rbf_features(x, landmarks, gamma_x, seed + 2)
    phi_y = nystrom_rbf_features(y, landmarks, gamma_y, seed + 3)
    return regularized_cca(phi_x, phi_y, components, ridge), gamma_x, gamma_y


def verify_run_config(spec: MethodSpec) -> None:
    payload = json.loads((spec.root / "run_config.json").read_text(encoding="utf-8"))
    if payload.get("config_hash") != spec.expected_hash:
        raise RuntimeError(f"Run config hash mismatch for {spec.key}")


def extract_fold_latents(spec: MethodSpec, fold: int, waveforms: np.ndarray, manifest: pd.DataFrame,
                         device: torch.device, batch_size: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    checkpoint = spec.checkpoint(fold)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("config_hash") != spec.expected_hash or int(saved.get("fold", -1)) != fold:
        raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
    model = base.CepstrumCLUBModel().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    labels = manifest["label"].to_numpy(np.float32)
    _, _, indices = base.split_indices(manifest, fold)
    loader = base.evaluation_loader(waveforms, labels, indices, batch_size, 0)
    rows, z_click_parts, z_channel_parts = [], [], []
    with torch.no_grad():
        for received, batch_labels, batch_indices in loader:
            outputs = model(received.to(device, non_blocking=True))
            z_click = outputs["z_click"].cpu().numpy().astype(np.float32)
            z_channel = outputs["z_channel"].cpu().numpy().astype(np.float32)
            if not np.isfinite(z_click).all() or not np.isfinite(z_channel).all():
                raise RuntimeError(f"Non-finite latent values: {spec.key}, fold {fold}")
            batch_indices_np = batch_indices.numpy().astype(int)
            metadata = manifest.iloc[batch_indices_np][["sample_id", "ori_file_num", "label", "test_fold"]].copy()
            metadata.insert(0, "sample_index", batch_indices_np)
            rows.append(metadata)
            z_click_parts.append(z_click)
            z_channel_parts.append(z_channel)
    metadata = pd.concat(rows, ignore_index=True)
    z_click, z_channel = np.vstack(z_click_parts), np.vstack(z_channel_parts)
    expected = manifest.index[manifest["test_fold"] == fold].to_numpy(int)
    if (len(metadata) != len(expected) or not metadata["test_fold"].eq(fold).all() or
            not metadata["sample_index"].is_unique or set(metadata["sample_index"]) != set(expected)):
        raise RuntimeError(f"Invalid test-fold latent extraction: {spec.key}, fold {fold}")
    if int(metadata["label"].sum()) != int(manifest.iloc[expected]["label"].sum()):
        raise RuntimeError(f"Unexpected positive count: {spec.key}, fold {fold}")
    return metadata, z_click, z_channel


def save_latents(root: Path, spec: MethodSpec, fold: int, metadata: pd.DataFrame,
                 z_click: np.ndarray, z_channel: np.ndarray) -> Path:
    path = root / "latent_vectors" / spec.key / f"fold_{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sample_index=metadata["sample_index"].to_numpy(),
                        sample_id=metadata["sample_id"].to_numpy(str),
                        ori_file_num=metadata["ori_file_num"].to_numpy(), label=metadata["label"].to_numpy(int),
                        test_fold=metadata["test_fold"].to_numpy(int), z_click=z_click, z_channel=z_channel)
    return path


def analyze_population(spec: MethodSpec, fold: int, population: str, metadata: pd.DataFrame,
                       z_click: np.ndarray, z_channel: np.ndarray) -> dict[str, Any]:
    mask = np.ones(len(metadata), dtype=bool) if population == "all_test" else metadata["label"].to_numpy() == 1
    x, y = z_click[mask], z_channel[mask]
    linear = regularized_cca(x, y)
    kernel, gamma_x, gamma_y = kernel_cca(x, y, seed=SEED + fold * 10 + (0 if population == "all_test" else 1))
    result: dict[str, Any] = {"method": spec.key, "method_label": spec.label, "fold": fold,
                              "population": population, "sample_count": int(mask.sum()),
                              "positive_count": int(metadata.loc[mask, "label"].sum())}
    for analysis, values in (("linear_cca", linear), ("nystrom_rbf_kernel_cca", kernel)):
        row = {**result, "analysis": analysis, "mean_first_10": float(values.mean())}
        row.update({f"corr_{index + 1}": float(values[index]) for index in range(len(values))})
        if analysis != "linear_cca":
            row.update({"rbf_gamma_z_click": gamma_x, "rbf_gamma_z_channel": gamma_y,
                        "landmarks": min(LANDMARKS, int(mask.sum())), "ridge": KCCA_RIDGE})
        else:
            row.update({"rbf_gamma_z_click": np.nan, "rbf_gamma_z_channel": np.nan,
                        "landmarks": np.nan, "ridge": RCCA_RIDGE})
        yield row


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    correlations = [f"corr_{number}" for number in range(1, COMPONENTS + 1)] + ["mean_first_10"]
    rows = []
    for keys, group in frame.groupby(["method", "method_label", "population", "analysis"], sort=False):
        row = dict(zip(["method", "method_label", "population", "analysis"], keys, strict=True))
        row["fold_count"] = len(group)
        for column in correlations:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_fold_sd"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, summary: pd.DataFrame) -> None:
    lines = ["# 第22实验：测试 OOF 潜变量 CCA / Kernel CCA 诊断", "",
             "每个折由独立训练的最佳检查点提取潜变量，并在折内完成分析；报告为五折均值 ± 标准差。",
             "不同折的原始潜变量没有拼接，因为其坐标系由不同模型学习。", "",
             "| 方法 | 样本口径 | 分析 | 第一典型相关 | 前10个均值 |", "|---|---|---|---:|---:|"]
    for _, row in summary.iterrows():
        name = row["method_label"]
        population = "全体测试" if row["population"] == "all_test" else "真实正类 click"
        analysis = "线性 CCA" if row["analysis"] == "linear_cca" else "Nyström RBF Kernel CCA"
        lines.append(f"| {name} | {population} | {analysis} | "
                     f"{row['corr_1_mean']:.4f} ± {row['corr_1_fold_sd']:.4f} | "
                     f"{row['mean_first_10_mean']:.4f} ± {row['mean_first_10_fold_sd']:.4f} |")
    positive_linear = summary[(summary["population"] == "positive_click") &
                              (summary["analysis"] == "linear_cca")].set_index("method")
    positive_kernel = summary[(summary["population"] == "positive_click") &
                              (summary["analysis"] == "nystrom_rbf_kernel_cca")].set_index("method")
    lines += ["", "## 本次诊断的直接观察", "",
              "在真实正类 click 上，三种方法的线性典型相关前10个均值依次为："
              f"无 MIM {positive_linear.loc['recon_bce', 'mean_first_10_mean']:.4f}、"
              f"MIM=1.0 {positive_linear.loc['positive_mim', 'mean_first_10_mean']:.4f}、"
              f"正交 λ=0.3 {positive_linear.loc['orthogonal_0p3', 'mean_first_10_mean']:.4f}。",
              "对应的 Nyström RBF Kernel CCA 前10个均值依次为："
              f"{positive_kernel.loc['recon_bce', 'mean_first_10_mean']:.4f}、"
              f"{positive_kernel.loc['positive_mim', 'mean_first_10_mean']:.4f}、"
              f"{positive_kernel.loc['orthogonal_0p3', 'mean_first_10_mean']:.4f}。",
              "因此，本次单种子五折机制诊断没有显示余弦正交将两组潜变量的可预测共享结构降至低于无 MIM 重建基线；"
              "MIM=1.0 的共享结构反而更高。", "", "## 解释边界", "",
              "典型相关高表示一个潜变量群中仍存在可由另一潜变量群线性或非线性预测的共享变化。",
              "余弦正交只约束同一样本潜变量的夹角，因此不保证 CCA 或 Kernel CCA 必然较低。",
              "该诊断不等同于统计独立性，也不证明真实物理信道已经恢复；Kernel CCA 使用 Nyström RBF 近似。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path, device_text: str = "auto", batch_size: int = 16) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in METHODS:
        verify_run_config(spec)
    config = base.Config(output_dir=base.BASE_CACHE_DIR)
    manifest, waveforms, contract = base.load_locked_cache(config)
    if contract["sha256"] != base.EXPECTED_DATA_HASH:
        raise RuntimeError("Locked data hash mismatch")
    device = base.resolve_device(device_text)
    rows = []
    progress = tqdm(total=len(METHODS) * len(FOLDS), desc="latent CCA extraction", unit="model")
    for spec in METHODS:
        for fold in FOLDS:
            metadata, z_click, z_channel = extract_fold_latents(spec, fold, waveforms, manifest, device, batch_size)
            save_latents(output_dir, spec, fold, metadata, z_click, z_channel)
            for population in ("all_test", "positive_click"):
                rows.extend(analyze_population(spec, fold, population, metadata, z_click, z_channel))
            progress.update(1)
            progress.set_postfix(method=spec.key, fold=fold)
    progress.close()
    per_fold, summary = pd.DataFrame(rows), summarize(pd.DataFrame(rows))
    if len(per_fold) != len(METHODS) * len(FOLDS) * 2 * 2 or not np.isfinite(
        per_fold[[f"corr_{number}" for number in range(1, COMPONENTS + 1)]].to_numpy()).all():
        raise RuntimeError("CCA artifact validation failed")
    if not ((per_fold[[f"corr_{number}" for number in range(1, COMPONENTS + 1)]] >= 0).all().all() and
            (per_fold[[f"corr_{number}" for number in range(1, COMPONENTS + 1)]] <= 1).all().all()):
        raise RuntimeError("Canonical correlations outside [0, 1]")
    per_fold.to_csv(output_dir / "cca_per_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "cca_summary.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir / "latent_cca_report_zh.md", summary)
    manifest_rows = []
    for spec in METHODS:
        for fold in FOLDS:
            manifest_rows.append({"method": spec.key, "fold": fold, "checkpoint": str(spec.checkpoint(fold).resolve()),
                                  "config_hash": spec.expected_hash, "latent_file": str((output_dir / "latent_vectors" / spec.key / f"fold_{fold}.npz").resolve())})
    pd.DataFrame(manifest_rows).to_csv(output_dir / "latent_manifest.csv", index=False, encoding="utf-8-sig")
    validation = {"passed": True, "data_hash": contract["sha256"], "method_count": len(METHODS),
                  "fold_count": len(FOLDS), "per_fold_rows": len(per_fold), "summary_rows": len(summary),
                  "latent_file_count": len(manifest_rows), "components": COMPONENTS,
                  "kernel": "Nystrom RBF Kernel CCA", "landmarks": LANDMARKS}
    (output_dir / "artifact_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir, arguments.device, arguments.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
