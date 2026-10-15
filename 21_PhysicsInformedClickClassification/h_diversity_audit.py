#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结既有低 SNR 检查点，审计物理分支响应 h 的结构化多样性。

本入口只读取既有实验的检查点、清单和混合波形缓存。所有响应、统计量、
图件、验收决定和报告均写入独立审计目录；不会训练模型或修改检查点。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 脚本和 pipeline.py 位于同一目录；显式加入路径也支持测试中的文件式导入。
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from pipeline import (  # noqa: E402
    Config,
    PhysicalClassifier,
    causal_fft_convolution,
    peak_normalize,
    preprocess_component,
    read_waveform,
)


AUDIT_CONDITIONS = ("recon_bce", "recon_physics_bce")
CONDITION_NAMES = {
    "recon_bce": "Recon + BCE",
    "recon_physics_bce": "Recon + Physics + BCE",
}
MAIN_METRICS = ("m_train", "m_noise", "m_swap")
METRIC_NAMES = {
    "m_train": "M_train：不同 train 距离 − 同 train 距离",
    "m_noise": "M_noise：不同 train 距离 − 混合/干净距离",
    "m_swap": "M_swap：不同 train 交换误差 − 同 train 交换误差",
}
EXPECTED_SEEDS = (20_260_826, 20_260_827, 20_260_828, 20_260_829, 20_260_830)
EXPECTED_RESPONSE_COUNT = 1_153
EXPECTED_POSITIVES = 3_500


@dataclass(frozen=True)
class AuditConfig:
    mode: str
    checkpoint_output: Path
    output_dir: Path
    seeds: tuple[int, ...]
    bootstrap_runs: int
    permutation_runs: int
    device: str
    force: bool
    batch_size: int = 64
    random_seed: int = 20_260_826

    @property
    def folds(self) -> tuple[int, ...]:
        return (0,) if self.mode == "smoke" else tuple(range(5))

    @property
    def conditions(self) -> tuple[str, ...]:
        return ("recon_physics_bce",) if self.mode == "smoke" else AUDIT_CONDITIONS

    @property
    def active_seeds(self) -> tuple[int, ...]:
        return (self.seeds[0],) if self.mode == "smoke" else self.seeds


class IndexedWaveforms(Dataset):
    def __init__(self, waveforms: np.ndarray, indices: Sequence[int]):
        self.waveforms = waveforms
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        waveform = torch.from_numpy(np.asarray(self.waveforms[index], dtype=np.float32).copy())
        return waveform, torch.tensor(index, dtype=torch.long)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_output(config: AuditConfig) -> None:
    checkpoint = config.checkpoint_output.resolve()
    output = config.output_dir.resolve()
    if output == checkpoint:
        raise ValueError("--output-dir 不得等于既有低 SNR 输出目录")
    try:
        checkpoint.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("--output-dir 不得是既有低 SNR 输出目录的上级目录")
    if output.exists() and any(output.iterdir()):
        if not config.force:
            raise FileExistsError(f"审计输出目录非空；如需重建审计产物请使用 --force：{output}")
        # 只清理明确指定的审计目录，永不触碰 checkpoint_output。
        for child in list(output.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for name in ("responses", "metrics", "figures", "cache"):
        (output / name).mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 环境不可用")
    return torch.device(requested)


def parse_sample_ids(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    parsed = result["sample_id"].astype(str).str.extract(r"^P_(\d+)_(\d+)_Pulse_(\d+)$")
    positive = result["label"].astype(int).eq(1)
    if parsed.loc[positive].isna().any().any():
        bad = result.loc[positive & parsed.isna().any(axis=1), "sample_id"].head().tolist()
        raise RuntimeError(f"正样本 sample_id 无法解析：{bad}")
    result["recording_id"] = pd.to_numeric(parsed[0], errors="coerce").astype("Int64")
    result["train_id"] = pd.to_numeric(parsed[1], errors="coerce").astype("Int64")
    result["pulse_index"] = pd.to_numeric(parsed[2], errors="coerce").astype("Int64")
    result["snr_bin_low_db"] = np.floor(result["target_snr_db"].astype(float) / 5.0) * 5.0
    return result


def build_pair_manifest(positives: pd.DataFrame) -> pd.DataFrame:
    """按预注册规则确定每个目标的同 train 与不同 train 对照。"""
    positives = positives.copy().reset_index(drop=False).rename(columns={"index": "sample_index"})
    group_sizes = positives.groupby(["recording_id", "train_id"])["sample_id"].transform("size")
    candidates = positives[group_sizes >= 2].copy()
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        same = positives[
            (positives["recording_id"] == row.recording_id)
            & (positives["train_id"] == row.train_id)
            & (positives["sample_id"] != row.sample_id)
        ].copy()
        same["pair_delta"] = (same["pulse_index"].astype(int) - int(row.pulse_index)).abs()
        same = same.sort_values(["pair_delta", "sample_id"], kind="stable")
        different = positives[
            (positives["recording_id"] == row.recording_id)
            & (positives["train_id"] != row.train_id)
            & (positives["mix_noise_type"] == row.mix_noise_type)
            & (positives["snr_bin_low_db"] == row.snr_bin_low_db)
        ].copy()
        different["pair_delta"] = (different["target_snr_db"].astype(float) - float(row.target_snr_db)).abs()
        different = different.sort_values(["pair_delta", "sample_id"], kind="stable")
        if same.empty or different.empty:
            continue
        same_row = same.iloc[0]
        different_row = different.iloc[0]
        rows.append(
            {
                "sample_index": int(row.sample_index),
                "sample_id": row.sample_id,
                "recording_id": int(row.recording_id),
                "train_id": int(row.train_id),
                "pulse_index": int(row.pulse_index),
                "test_fold": int(row.test_fold),
                "mix_noise_type": row.mix_noise_type,
                "target_snr_db": float(row.target_snr_db),
                "snr_bin_low_db": float(row.snr_bin_low_db),
                "same_sample_index": int(same_row["sample_index"]),
                "same_sample_id": same_row["sample_id"],
                "different_sample_index": int(different_row["sample_index"]),
                "different_sample_id": different_row["sample_id"],
            }
        )
    return pd.DataFrame(rows)


def validate_dataset_contract(manifest: pd.DataFrame, pair_manifest: pd.DataFrame) -> dict[str, Any]:
    positives = manifest[manifest["label"].astype(int) == 1]
    train_counts = positives.groupby(["recording_id", "train_id"]).size()
    multi = train_counts[train_counts >= 2]
    checks = {
        "positive_count": int(len(positives)),
        "train_count": int(len(train_counts)),
        "multi_click_train_count": int(len(multi)),
        "comparable_click_count": int(multi.sum()),
        "matched_click_count": int(len(pair_manifest)),
        "matched_coverage": float(len(pair_manifest) / max(int(multi.sum()), 1)),
        "positive_fold_counts": {
            str(int(fold)): int(count)
            for fold, count in positives["test_fold"].astype(int).value_counts().sort_index().items()
        },
        "pair_key_unique": bool(pair_manifest["sample_id"].is_unique),
        "same_pair_fold_mismatch_count": 0,
        "different_pair_fold_mismatch_count": 0,
    }
    fold_by_index = manifest["test_fold"].astype(int).to_dict()
    checks["same_pair_fold_mismatch_count"] = int(
        sum(fold_by_index[int(r.sample_index)] != fold_by_index[int(r.same_sample_index)] for r in pair_manifest.itertuples())
    )
    checks["different_pair_fold_mismatch_count"] = int(
        sum(
            fold_by_index[int(r.sample_index)] != fold_by_index[int(r.different_sample_index)]
            for r in pair_manifest.itertuples()
        )
    )
    checks["passed"] = bool(
        checks["positive_count"] == EXPECTED_POSITIVES
        and checks["train_count"] == 640
        and checks["multi_click_train_count"] == 475
        and checks["comparable_click_count"] == 3_335
        and checks["matched_click_count"] >= math.ceil(3_335 * 0.95)
        and checks["pair_key_unique"]
        and checks["same_pair_fold_mismatch_count"] == 0
        and checks["different_pair_fold_mismatch_count"] == 0
    )
    return checks


def checkpoint_path(root: Path, condition: str, seed: int, fold: int) -> Path:
    return root / "models" / f"{condition}_seed_{seed}_fold_{fold}.pt"


def compute_data_hash_readonly(manifest: pd.DataFrame, waveforms: np.ndarray, config: Config) -> dict[str, Any]:
    digest = hashlib.sha256()
    columns = ["sample_id", "label", "ori_file_num", "test_fold"]
    if config.positive_noise_mix:
        columns.extend(
            [
                "mix_noise_audio_file",
                "mix_noise_timestamp_s",
                "mix_noise_type",
                "target_snr_db",
                "realized_snr_db",
                "mix_alpha",
            ]
        )
        digest.update(
            json.dumps(
                {
                    "positive_noise_mix": True,
                    "mix_seed": config.mix_seed,
                    "snr_min_db": config.snr_min_db,
                    "snr_max_db": config.snr_max_db,
                },
                sort_keys=True,
            ).encode()
        )
    digest.update(manifest[columns].to_csv(index=False).encode())
    for start in tqdm(range(0, len(waveforms), 128), desc="总体阶段 1/6：核验数据哈希", unit="块", leave=False):
        digest.update(np.asarray(waveforms[start : start + 128]).tobytes())
    return {"sha256": digest.hexdigest(), "samples": len(manifest), "shape": list(waveforms.shape), "dtype": str(waveforms.dtype)}


def validate_inputs_readonly(
    config: AuditConfig,
) -> tuple[pd.DataFrame, np.ndarray, Config, dict[str, Any], pd.DataFrame]:
    root = config.checkpoint_output
    required = [
        root / "run_config.json",
        root / "dataset_manifest.csv",
        root / "data_hash.json",
        root / "cache" / "preprocessed_waveforms.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"低 SNR 实验产物不完整：{missing}")
    run_config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    stored_hash = json.loads((root / "data_hash.json").read_text(encoding="utf-8"))
    manifest = parse_sample_ids(pd.read_csv(root / "dataset_manifest.csv"))
    waveforms = np.load(root / "cache" / "preprocessed_waveforms.npy", mmap_mode="r")
    pipeline_config = Config(
        output_dir=root,
        positive_noise_mix=bool(run_config["positive_noise_mix"]),
        snr_min_db=float(run_config["snr_min_db"]),
        snr_max_db=float(run_config["snr_max_db"]),
        mix_seed=int(run_config["mix_seed"]),
    )
    if not pipeline_config.positive_noise_mix:
        raise RuntimeError("审计输入不是固定低 SNR 混合实验")
    if tuple(waveforms.shape) != (7_000, pipeline_config.sample_count):
        raise RuntimeError(f"混合波形缓存尺寸错误：{waveforms.shape}")
    computed_hash = compute_data_hash_readonly(manifest, waveforms, pipeline_config)
    if computed_hash["sha256"] != stored_hash.get("sha256"):
        raise RuntimeError("数据哈希不一致；清单或混合波形缓存已改变")

    pair_manifest = build_pair_manifest(manifest[manifest["label"].astype(int) == 1])
    dataset_checks = validate_dataset_contract(manifest, pair_manifest)
    if not dataset_checks["passed"]:
        raise RuntimeError(f"数据契约不满足审计要求：{dataset_checks}")

    checkpoint_rows: list[dict[str, Any]] = []
    # full 模式启动时必须核验 50 个目标检查点；smoke 只核验实际读取的 1 个。
    for condition in config.conditions:
        for seed in config.active_seeds:
            for fold in config.folds:
                path = checkpoint_path(root, condition, seed, fold)
                if not path.exists():
                    raise FileNotFoundError(f"缺少检查点：{path}")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                checks = {
                    "condition": saved.get("condition") == condition,
                    "seed": int(saved.get("seed", -1)) == seed,
                    "fold": int(saved.get("fold", -1)) == fold,
                    "training_complete": bool(saved.get("training_complete", False)),
                    "config_hash": saved.get("config_hash") == run_config.get("config_hash"),
                    "data_hash": computed_hash["sha256"] == stored_hash.get("sha256"),
                }
                checkpoint_rows.append(
                    {
                        "path": str(path.resolve()),
                        "condition": condition,
                        "seed": seed,
                        "fold": fold,
                        **checks,
                        "passed": bool(all(checks.values())),
                    }
                )
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    if not checkpoint_frame["passed"].all():
        raise RuntimeError("一个或多个检查点元数据不一致")
    validation = {
        "run_config_hash": run_config.get("config_hash"),
        "stored_data_hash": stored_hash.get("sha256"),
        "computed_data_hash": computed_hash["sha256"],
        "checkpoint_count": int(len(checkpoint_frame)),
        "dataset": dataset_checks,
        "passed": True,
    }
    json_dump(config.output_dir / "input_validation.json", validation)
    checkpoint_frame.to_csv(config.output_dir / "metrics" / "checkpoint_validation.csv", index=False, encoding="utf-8-sig")
    return manifest, waveforms, pipeline_config, validation, pair_manifest


def materialize_clean_waveforms(
    manifest: pd.DataFrame, pipeline_config: Config, output_dir: Path
) -> np.ndarray:
    positives = manifest[manifest["label"].astype(int) == 1]
    path = output_dir / "cache" / "positive_clean_waveforms.npy"
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(manifest), pipeline_config.sample_count)
    )
    values[:] = 0.0
    for index, row in tqdm(
        positives.iterrows(), total=len(positives), desc="总体阶段 2/6：生成原 Click 干净视图", unit="条"
    ):
        clean = preprocess_component(read_waveform(row, pipeline_config), pipeline_config)
        values[int(index)] = peak_normalize(clean)
    values.flush()
    return np.load(path, mmap_mode="r")


def energy_delay_descriptor(responses: np.ndarray) -> np.ndarray:
    values = np.asarray(responses, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != EXPECTED_RESPONSE_COUNT:
        raise ValueError(f"响应长度必须为 {EXPECTED_RESPONSE_COUNT}，实际为 {values.shape[1]}")
    head = values[:, :1] ** 2
    tail = (values[:, 1:] ** 2).reshape(len(values), 36, 32).sum(axis=2)
    energy = np.concatenate([head, tail], axis=1)
    return energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-15)


def frequency_descriptor(
    responses: np.ndarray, sample_rate: int = 576_000, low_hz: float = 20_000, high_hz: float = 190_000
) -> np.ndarray:
    values = np.asarray(responses, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    spectrum = np.abs(np.fft.rfft(values, n=4096, axis=1))
    frequencies = np.fft.rfftfreq(4096, 1.0 / sample_rate)
    edges = np.linspace(low_hz, high_hz, 33)
    bands = np.empty((len(values), 32), dtype=np.float64)
    for band in range(32):
        mask = (frequencies >= edges[band]) & (frequencies < edges[band + 1])
        bands[:, band] = np.log(np.maximum(spectrum[:, mask].mean(axis=1), 1e-12))
    padded = np.pad(bands, ((0, 0), (1, 1)), mode="edge")
    smooth = 0.25 * padded[:, :-2] + 0.5 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
    smooth -= smooth.mean(axis=1, keepdims=True)
    smooth /= np.maximum(smooth.std(axis=1, keepdims=True), 1e-12)
    return smooth


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return np.linalg.norm(np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0)), axis=-1) / math.sqrt(2.0)


def frequency_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2, axis=-1))


def collapse_diagnostics(responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(responses, dtype=np.float64)
    tail_energy = np.sum(values[:, 1:] ** 2, axis=1)
    total_energy = np.sum(values**2, axis=1)
    ratio = tail_energy / np.maximum(total_energy, 1e-15)
    return ratio, ratio <= 1e-4


def effective_rank(descriptors: np.ndarray) -> float:
    values = np.asarray(descriptors, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) ** 2
    probabilities = singular / np.maximum(singular.sum(), 1e-15)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def centroid_distance(descriptors: np.ndarray) -> np.ndarray:
    center = np.asarray(descriptors, dtype=np.float64).mean(axis=0)
    center /= max(center.sum(), 1e-15)
    return hellinger(descriptors, np.broadcast_to(center, np.asarray(descriptors).shape))


@torch.no_grad()
def infer_model(
    model: PhysicalClassifier,
    waveforms: np.ndarray,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    return_source: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    loader = DataLoader(
        IndexedWaveforms(waveforms, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    responses: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    model.eval()
    for batch, _ in loader:
        outputs = model(batch.to(device, non_blocking=True))
        responses.append(outputs["response"].detach().cpu().numpy())
        if return_source:
            sources.append(outputs["source"].detach().cpu().numpy())
    response = np.concatenate(responses).astype(np.float32, copy=False)
    source = np.concatenate(sources).astype(np.float32, copy=False) if return_source else None
    return response, source


def validate_responses(responses: np.ndarray) -> None:
    if responses.ndim != 2 or responses.shape[1] != EXPECTED_RESPONSE_COUNT:
        raise RuntimeError(f"响应尺寸错误：{responses.shape}")
    if not np.isfinite(responses).all():
        raise RuntimeError("响应包含非有限值")
    if not np.array_equal(responses[:, 0], np.ones(len(responses), dtype=responses.dtype)):
        raise RuntimeError("响应不满足 h_0 = 1")


def response_path(output: Path, condition: str, seed: int, fold: int, view: str) -> Path:
    return output / "responses" / condition / f"seed_{seed}" / f"fold_{fold}_{view}.npz"


def save_responses(
    path: Path, responses: np.ndarray, sample_indices: np.ndarray, sample_ids: np.ndarray, view: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        response=responses.astype(np.float32, copy=False),
        sample_index=sample_indices.astype(np.int64),
        sample_id=sample_ids.astype(str),
        view=np.asarray(view),
    )


def relative_nrmse_batch(inputs: np.ndarray, sources: np.ndarray, responses: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    source_tensor = torch.as_tensor(np.asarray(sources), dtype=torch.float32, device=device)
    response_tensor = torch.as_tensor(np.asarray(responses), dtype=torch.float32, device=device)
    reconstruction = causal_fft_convolution(source_tensor, response_tensor).numpy()
    numerator = np.sum((np.asarray(inputs) - reconstruction) ** 2, axis=1)
    denominator = np.sum(np.asarray(inputs) ** 2, axis=1)
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def fit_confound_probes(
    features: np.ndarray,
    positives: pd.DataFrame,
    fold: int,
    condition: str,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train_mask = ~positives["test_fold"].astype(int).isin([fold, (fold + 1) % 5]).to_numpy()
    validation_mask = positives["test_fold"].astype(int).eq((fold + 1) % 5).to_numpy()
    test_mask = positives["test_fold"].astype(int).eq(fold).to_numpy()
    scaler = StandardScaler().fit(features[train_mask])
    train_x = scaler.transform(features[train_mask])
    validation_x = scaler.transform(features[validation_mask])
    test_x = scaler.transform(features[test_mask])
    snr = positives["target_snr_db"].to_numpy(float)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    ridge_scores = []
    for alpha in alphas:
        candidate = Ridge(alpha=alpha).fit(train_x, snr[train_mask])
        ridge_scores.append(r2_score(snr[validation_mask], candidate.predict(validation_x)))
    ridge_alpha = alphas[int(np.argmax(ridge_scores))]
    ridge = Ridge(alpha=ridge_alpha).fit(train_x, snr[train_mask])
    snr_prediction = ridge.predict(test_x)

    labels = positives["mix_noise_type"].astype(str).to_numpy()
    classes = np.sort(np.unique(labels))
    c_values = (0.001, 0.01, 0.1, 1.0, 10.0)
    logistic_scores = []
    for c_value in c_values:
        candidate = LogisticRegression(C=c_value, max_iter=2_000, solver="lbfgs")
        candidate.fit(train_x, labels[train_mask])
        logistic_scores.append(balanced_accuracy_score(labels[validation_mask], candidate.predict(validation_x)))
    logistic_c = c_values[int(np.argmax(logistic_scores))]
    logistic = LogisticRegression(C=logistic_c, max_iter=2_000, solver="lbfgs")
    logistic.fit(train_x, labels[train_mask])
    background_prediction = logistic.predict(test_x)
    test_rows = positives.loc[test_mask, ["sample_id", "recording_id", "test_fold"]].copy()
    test_rows["condition"] = condition
    test_rows["seed"] = seed
    test_rows["snr_true"] = snr[test_mask]
    test_rows["snr_prediction"] = snr_prediction
    test_rows["background_true"] = labels[test_mask]
    test_rows["background_prediction"] = background_prediction
    rows = [
        {
            "scope": "checkpoint",
            "condition": condition,
            "seed": seed,
            "fold": fold,
            "probe": "snr_r2",
            "selected_regularization": ridge_alpha,
            "estimate": float(r2_score(snr[test_mask], snr_prediction)),
            "class_count": np.nan,
            "random_level": np.nan,
            "sample_count": int(test_mask.sum()),
        },
        {
            "scope": "checkpoint",
            "condition": condition,
            "seed": seed,
            "fold": fold,
            "probe": "background_balanced_accuracy",
            "selected_regularization": logistic_c,
            "estimate": float(balanced_accuracy_score(labels[test_mask], background_prediction)),
            "class_count": int(len(classes)),
            "random_level": float(1.0 / len(classes)),
            "sample_count": int(test_mask.sum()),
        },
    ]
    return rows, test_rows


def checkpoint_metrics(
    condition: str,
    seed: int,
    fold: int,
    saved: dict[str, Any],
    model: PhysicalClassifier,
    manifest: pd.DataFrame,
    mixed: np.ndarray,
    clean: np.ndarray,
    pair_manifest: pd.DataFrame,
    device: torch.device,
    config: AuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    positives = manifest[manifest["label"].astype(int) == 1].copy()
    positive_indices = positives.index.to_numpy(dtype=np.int64)
    test_indices = positives.index[positives["test_fold"].astype(int) == fold].to_numpy(dtype=np.int64)
    all_response, all_source = infer_model(model, mixed, positive_indices, device, config.batch_size, True)
    test_positions = np.flatnonzero(positives["test_fold"].astype(int).to_numpy() == fold)
    mixed_response = all_response[test_positions]
    mixed_source = all_source[test_positions] if all_source is not None else None
    clean_response, _ = infer_model(model, clean, test_indices, device, config.batch_size, False)
    validate_responses(mixed_response)
    validate_responses(clean_response)

    sample_ids = manifest.loc[test_indices, "sample_id"].astype(str).to_numpy()
    save_responses(response_path(config.output_dir, condition, seed, fold, "mixed"), mixed_response, test_indices, sample_ids, "mixed")
    save_responses(response_path(config.output_dir, condition, seed, fold, "clean"), clean_response, test_indices, sample_ids, "clean")

    edp_all = energy_delay_descriptor(all_response)
    freq_all = frequency_descriptor(all_response)
    probe_features = np.concatenate([edp_all, freq_all], axis=1)
    probe_rows, probe_predictions = fit_confound_probes(probe_features, positives, fold, condition, seed)

    edp_mixed = edp_all[test_positions]
    freq_mixed = freq_all[test_positions]
    edp_clean = energy_delay_descriptor(clean_response)
    freq_clean = frequency_descriptor(clean_response)
    test_lookup = {int(index): position for position, index in enumerate(test_indices)}
    diagnostics: list[pd.DataFrame] = []
    fold_stats: dict[str, Any] = {"condition": condition, "seed": seed, "fold": fold}
    for view, response, descriptor in (
        ("mixed", mixed_response, edp_mixed),
        ("clean", clean_response, edp_clean),
    ):
        tail_ratio, collapsed = collapse_diagnostics(response)
        diag = manifest.loc[test_indices, ["sample_id", "test_fold", "mix_noise_type", "target_snr_db"]].copy()
        diag.insert(0, "sample_index", test_indices)
        diag.insert(0, "view", view)
        diag.insert(0, "fold", fold)
        diag.insert(0, "seed", seed)
        diag.insert(0, "condition", condition)
        diag["tail_energy_ratio"] = tail_ratio
        diag["collapsed"] = collapsed
        diag["h_delta"] = np.max(np.abs(response[:, 1:]), axis=1) <= 1e-6
        diag["centroid_hellinger"] = centroid_distance(descriptor)
        diagnostics.append(diag)
        fold_stats[f"{view}_effective_rank"] = effective_rank(descriptor)
        fold_stats[f"{view}_collapse_rate"] = float(np.mean(collapsed))
        fold_stats[f"{view}_h_delta_rate"] = float(diag["h_delta"].mean())

    selected_pairs = pair_manifest[pair_manifest["test_fold"].astype(int) == fold]
    pair_rows: list[dict[str, Any]] = []
    chunk_size = 128
    for start in range(0, len(selected_pairs), chunk_size):
        chunk = selected_pairs.iloc[start : start + chunk_size]
        target_position = np.array([test_lookup[int(v)] for v in chunk["sample_index"]], dtype=int)
        same_position = np.array([test_lookup[int(v)] for v in chunk["same_sample_index"]], dtype=int)
        different_position = np.array([test_lookup[int(v)] for v in chunk["different_sample_index"]], dtype=int)
        target_waveform = np.asarray(mixed[chunk["sample_index"].to_numpy(dtype=int)])
        source = mixed_source[target_position]
        same_swap = relative_nrmse_batch(target_waveform, source, mixed_response[same_position])
        different_swap = relative_nrmse_batch(target_waveform, source, mixed_response[different_position])
        own_reconstruction = relative_nrmse_batch(target_waveform, source, mixed_response[target_position])
        same_distance = hellinger(edp_mixed[target_position], edp_mixed[same_position])
        different_distance = hellinger(edp_mixed[target_position], edp_mixed[different_position])
        noise_distance = hellinger(edp_mixed[target_position], edp_clean[target_position])
        same_frequency = frequency_distance(freq_mixed[target_position], freq_mixed[same_position])
        different_frequency = frequency_distance(freq_mixed[target_position], freq_mixed[different_position])
        noise_frequency = frequency_distance(freq_mixed[target_position], freq_clean[target_position])
        for offset, row in enumerate(chunk.itertuples(index=False)):
            pair_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "fold": fold,
                    **row._asdict(),
                    "d_same_train": float(same_distance[offset]),
                    "d_different_train": float(different_distance[offset]),
                    "d_mix_clean": float(noise_distance[offset]),
                    "m_train": float(different_distance[offset] - same_distance[offset]),
                    "m_noise": float(different_distance[offset] - noise_distance[offset]),
                    "freq_d_same_train": float(same_frequency[offset]),
                    "freq_d_different_train": float(different_frequency[offset]),
                    "freq_d_mix_clean": float(noise_frequency[offset]),
                    "same_swap_nrmse": float(same_swap[offset]),
                    "different_swap_nrmse": float(different_swap[offset]),
                    "own_reconstruction_nrmse": float(own_reconstruction[offset]),
                    "m_swap": float(different_swap[offset] - same_swap[offset]),
                }
            )
    return pd.concat(diagnostics, ignore_index=True), pd.DataFrame(pair_rows), probe_rows, probe_predictions, fold_stats


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    runs: int,
    rng: np.random.Generator,
    progress_description: str | None = None,
) -> np.ndarray:
    seeds = np.sort(frame["seed"].unique())
    # 先聚合每个“种子×录音”簇；Bootstrap 循环不再重复执行 DataFrame 过滤。
    cluster_means = {
        seed: (
            frame.loc[frame["seed"] == seed]
            .groupby("recording_id", sort=True)[value_column]
            .mean()
            .to_numpy(float)
        )
        for seed in seeds
    }
    values = np.empty(runs, dtype=np.float64)
    iterator = tqdm(
        range(runs),
        desc=progress_description,
        unit="次",
        leave=False,
        disable=progress_description is None,
    )
    for run in iterator:
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        seed_means = []
        for seed in sampled_seeds:
            means = cluster_means[seed]
            seed_means.append(float(rng.choice(means, len(means), replace=True).mean()))
        values[run] = float(np.mean(seed_means))
    return values


def stratified_sign_permutation(
    frame: pd.DataFrame,
    value_column: str,
    runs: int,
    rng: np.random.Generator,
    progress_description: str | None = None,
) -> np.ndarray:
    """在录音×背景×5 dB SNR层内破坏差值方向，同时保留层结构。"""
    # 每个层内对每条配对差值独立交换两种对应关系；把各层连续拼接后生成独立
    # Rademacher 符号与逐层执行完全等价，同时避免在 2000 次循环中遍历数千层。
    ordered = frame.sort_values(
        ["seed", "recording_id", "mix_noise_type", "snr_bin_low_db"], kind="stable"
    )
    values = ordered[value_column].to_numpy(float)
    null = np.empty(runs, dtype=np.float64)
    count = len(values)
    iterator = tqdm(
        range(runs),
        desc=progress_description,
        unit="次",
        leave=False,
        disable=progress_description is None,
    )
    for run in iterator:
        signs = rng.choice(np.array([-1.0, 1.0]), count, replace=True)
        null[run] = float(np.sum(values * signs) / max(count, 1))
    return null


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def summarize_main_metrics(
    pair_metrics: pd.DataFrame, config: AuditConfig, show_progress: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.random_seed + 101)
    for condition, condition_frame in pair_metrics.groupby("condition", sort=False):
        temporary: list[dict[str, Any]] = []
        p_values: dict[str, float] = {}
        for metric in MAIN_METRICS:
            prefix = f"{CONDITION_NAMES[condition]} {metric}"
            distribution = hierarchical_bootstrap(
                condition_frame,
                metric,
                config.bootstrap_runs,
                rng,
                f"Bootstrap：{prefix}" if show_progress else None,
            )
            null = stratified_sign_permutation(
                condition_frame,
                metric,
                config.permutation_runs,
                rng,
                f"置换：{prefix}" if show_progress else None,
            )
            estimate = float(condition_frame[metric].mean())
            p_value = float((1 + np.sum(null >= estimate)) / (config.permutation_runs + 1))
            p_values[metric] = p_value
            temporary.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(np.quantile(distribution, 0.025)),
                    "ci_high": float(np.quantile(distribution, 0.975)),
                    "p_one_sided_permutation": p_value,
                    "bootstrap_runs_valid": int(len(distribution)),
                    "permutation_runs_valid": int(len(null)),
                    "sample_count": int(len(condition_frame)),
                }
            )
        adjusted = holm_adjust(p_values)
        for row in temporary:
            row["p_holm"] = adjusted[row["metric"]]
            row["passed"] = bool(row["ci_low"] > 0 and row["p_holm"] < 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired_conditions(
    pair_metrics: pd.DataFrame, config: AuditConfig, show_progress: bool = False
) -> pd.DataFrame:
    if set(pair_metrics["condition"].unique()) != set(AUDIT_CONDITIONS):
        return pd.DataFrame()
    keys = ["seed", "fold", "sample_id", "recording_id", "mix_noise_type", "snr_bin_low_db"]
    left = pair_metrics[pair_metrics["condition"] == "recon_physics_bce"]
    right = pair_metrics[pair_metrics["condition"] == "recon_bce"]
    merged = left.merge(right, on=keys, suffixes=("_physics", "_recon"), validate="one_to_one")
    rng = np.random.default_rng(config.random_seed + 202)
    rows = []
    for metric in MAIN_METRICS:
        delta = f"{metric}_delta"
        merged[delta] = merged[f"{metric}_physics"] - merged[f"{metric}_recon"]
        distribution = hierarchical_bootstrap(
            merged,
            delta,
            config.bootstrap_runs,
            rng,
            f"Bootstrap：Physics−Recon {metric}" if show_progress else None,
        )
        rows.append(
            {
                "comparison": "recon_physics_bce_minus_recon_bce",
                "metric": metric,
                "estimate": float(merged[delta].mean()),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "bootstrap_runs_valid": int(len(distribution)),
                "paired_sample_count": int(len(merged)),
                "physics_increases_structure": bool(np.quantile(distribution, 0.025) > 0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_collapse(
    diagnostics: pd.DataFrame, config: AuditConfig, show_progress: bool = False
) -> pd.DataFrame:
    frame = diagnostics[diagnostics["view"] == "mixed"].copy()
    frame["recording_id"] = frame["sample_id"].str.extract(r"^P_(\d+)_")[0].astype(int)
    rng = np.random.default_rng(config.random_seed + 303)
    rows = []
    for condition, group in frame.groupby("condition"):
        distribution = hierarchical_bootstrap(
            group,
            "collapsed",
            config.bootstrap_runs,
            rng,
            f"Bootstrap：{CONDITION_NAMES[condition]} 坍缩率" if show_progress else None,
        )
        rows.append(
            {
                "condition": condition,
                "metric": "collapse_rate",
                "estimate": float(group["collapsed"].mean()),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "bootstrap_runs_valid": int(len(distribution)),
                "sample_count": int(len(group)),
                "passed": bool(np.quantile(distribution, 0.975) < 0.05),
            }
        )
    return pd.DataFrame(rows)


def summarize_probes(
    checkpoint_rows: list[dict[str, Any]],
    predictions: pd.DataFrame,
    config: AuditConfig,
    show_progress: bool = False,
) -> pd.DataFrame:
    rows = list(checkpoint_rows)
    rng = np.random.default_rng(config.random_seed + 404)
    for condition, frame in predictions.groupby("condition"):
        for probe, true_column, prediction_column in (
            ("snr_r2", "snr_true", "snr_prediction"),
            ("background_balanced_accuracy", "background_true", "background_prediction"),
        ):
            if probe == "snr_r2":
                metric = lambda part: r2_score(part[true_column], part[prediction_column])
            else:
                metric = lambda part: balanced_accuracy_score(part[true_column], part[prediction_column])
            estimate = float(metric(frame))
            seeds = np.sort(frame["seed"].unique())
            clusters: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
            for seed in seeds:
                seed_frame = frame[frame["seed"] == seed]
                clusters[int(seed)] = [
                    (
                        group[true_column].to_numpy(),
                        group[prediction_column].to_numpy(),
                    )
                    for _, group in seed_frame.groupby("recording_id", sort=True)
                ]
            distribution = np.empty(config.bootstrap_runs)
            iterator = tqdm(
                range(config.bootstrap_runs),
                desc=f"Bootstrap：{CONDITION_NAMES[condition]} {probe}" if show_progress else None,
                unit="次",
                leave=False,
                disable=not show_progress,
            )
            for run in iterator:
                sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
                true_parts: list[np.ndarray] = []
                prediction_parts: list[np.ndarray] = []
                for seed in sampled_seeds:
                    seed_clusters = clusters[int(seed)]
                    sampled_indices = rng.integers(0, len(seed_clusters), len(seed_clusters))
                    for cluster_index in sampled_indices:
                        true_values, predicted_values = seed_clusters[int(cluster_index)]
                        true_parts.append(true_values)
                        prediction_parts.append(predicted_values)
                bootstrap_true = np.concatenate(true_parts)
                bootstrap_prediction = np.concatenate(prediction_parts)
                if probe == "snr_r2":
                    distribution[run] = float(r2_score(bootstrap_true, bootstrap_prediction))
                else:
                    distribution[run] = float(
                        balanced_accuracy_score(bootstrap_true, bootstrap_prediction)
                    )
            class_count = int(frame["background_true"].nunique()) if probe.startswith("background") else np.nan
            random_level = 1.0 / class_count if probe.startswith("background") else np.nan
            ci_high = float(np.quantile(distribution, 0.975))
            passed = ci_high < 0.10 if probe == "snr_r2" else ci_high <= random_level + 0.05
            rows.append(
                {
                    "scope": "condition_summary",
                    "condition": condition,
                    "seed": np.nan,
                    "fold": np.nan,
                    "probe": probe,
                    "selected_regularization": np.nan,
                    "estimate": estimate,
                    "ci_low": float(np.quantile(distribution, 0.025)),
                    "ci_high": ci_high,
                    "class_count": class_count,
                    "random_level": random_level,
                    "sample_count": int(len(frame)),
                    "bootstrap_runs_valid": int(len(distribution)),
                    "passed": bool(passed),
                }
            )
    return pd.DataFrame(rows)


def add_structure_diagnostics(summary: pd.DataFrame, fold_stats: list[dict[str, Any]]) -> pd.DataFrame:
    rows = summary.to_dict("records")
    frame = pd.DataFrame(fold_stats)
    for condition, group in frame.groupby("condition"):
        for metric in ("mixed_effective_rank", "mixed_h_delta_rate", "clean_effective_rank"):
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "estimate": float(group[metric].mean()),
                    "ci_low": float(group[metric].quantile(0.025)),
                    "ci_high": float(group[metric].quantile(0.975)),
                    "sample_count": int(len(group)),
                    "passed": np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_outputs(
    pair_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    fold_stats: list[dict[str, Any]],
    representative: dict[str, np.ndarray],
    output: Path,
) -> None:
    figures = output / "figures"
    colors = {"recon_bce": "#4472C4", "recon_physics_bce": "#ED7D31"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    columns = ["d_same_train", "d_mix_clean", "d_different_train"]
    titles = ["Same train", "Mixed vs clean", "Different train"]
    for axis, column, title in zip(axes, columns, titles):
        data = [pair_metrics.loc[pair_metrics.condition == condition, column] for condition in pair_metrics.condition.unique()]
        axis.boxplot(
            data,
            tick_labels=[CONDITION_NAMES[c] for c in pair_metrics.condition.unique()],
            showfliers=False,
        )
        axis.set_title(title)
        axis.set_ylabel("Hellinger distance")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "h_distance_hierarchy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(pair_metrics.condition.unique()))
    width = 0.36
    for offset, column, label in ((-width / 2, "same_swap_nrmse", "Same train"), (width / 2, "different_swap_nrmse", "Different train")):
        values = [pair_metrics.loc[pair_metrics.condition == c, column].mean() for c in pair_metrics.condition.unique()]
        axis.bar(positions + offset, values, width, label=label)
    axis.set_xticks(positions, [CONDITION_NAMES[c] for c in pair_metrics.condition.unique()])
    axis.set_ylabel("Relative-input-energy NRMSE")
    axis.set_title("Channel-swap reconstruction error")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "h_swap_reconstruction.png", dpi=180)
    plt.close(figure)

    stats = pd.DataFrame(fold_stats)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    diagnostics.query("view == 'mixed'").boxplot(column="tail_energy_ratio", by="condition", ax=axes[0], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title("Tail-energy ratio")
    axes[0].set_xlabel("")
    stats.boxplot(column="mixed_effective_rank", by="condition", ax=axes[1])
    axes[1].set_title("EDP effective rank")
    axes[1].set_xlabel("")
    figure.suptitle("")
    figure.tight_layout()
    figure.savefig(figures / "h_collapse_effective_rank.png", dpi=180)
    plt.close(figure)

    if representative:
        figure, axis = plt.subplots(figsize=(9, 5))
        delay_ms = np.concatenate(([0.0], (np.arange(36) * 32 + 16) / 576_000 * 1_000))
        for condition, descriptor in representative.items():
            axis.plot(delay_ms, descriptor, marker="o", markersize=3, label=CONDITION_NAMES[condition], color=colors[condition])
        axis.set(xlabel="Delay (ms)", ylabel="Normalized energy", title="Representative energy-delay profile")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "h_representative_edp.png", dpi=180)
        plt.close(figure)


def build_acceptance(
    config: AuditConfig,
    validation: dict[str, Any],
    diagnostics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    probes: pd.DataFrame,
) -> dict[str, Any]:
    expected_checkpoints = 1 if config.mode == "smoke" else 50
    positives_in_active_folds = sum(
        int(validation["dataset"]["positive_fold_counts"][str(fold)]) for fold in config.folds
    )
    expected_per_view = int(len(config.conditions) * len(config.active_seeds) * positives_in_active_folds)
    actual_mixed = int((diagnostics["view"] == "mixed").sum())
    actual_clean = int((diagnostics["view"] == "clean").sum())
    unique_keys = not diagnostics.duplicated(["condition", "seed", "fold", "view", "sample_id"]).any()
    main = summary[(summary["condition"] == "recon_physics_bce") & summary["metric"].isin(MAIN_METRICS)]
    collapse = summary[(summary["condition"] == "recon_physics_bce") & (summary["metric"] == "collapse_rate")]
    probe = probes[(probes["scope"] == "condition_summary") & (probes["condition"] == "recon_physics_bce")]
    rules: dict[str, Any] = {
        "checkpoint_count": {"actual": validation["checkpoint_count"], "expected": expected_checkpoints, "passed": validation["checkpoint_count"] == expected_checkpoints},
        "response_shape_finite_h0": {"passed": True},
        "response_key_unique": {"passed": bool(unique_keys)},
        "mixed_response_count": {"actual": actual_mixed, "expected": expected_per_view, "passed": actual_mixed == expected_per_view},
        "clean_response_count": {"actual": actual_clean, "expected": expected_per_view, "passed": actual_clean == expected_per_view},
        "matched_coverage": {"actual": validation["dataset"]["matched_coverage"], "minimum": 0.95, "passed": validation["dataset"]["matched_coverage"] >= 0.95},
    }
    if config.mode == "full":
        for metric in MAIN_METRICS:
            row = main[main["metric"] == metric].iloc[0]
            rules[metric] = {"ci_low": safe_float(row.ci_low), "p_holm": safe_float(row.p_holm), "passed": bool(row.passed)}
        collapse_row = collapse.iloc[0]
        rules["collapse_rate"] = {"ci_high": safe_float(collapse_row.ci_high), "passed": bool(collapse_row.passed)}
        for probe_name in ("snr_r2", "background_balanced_accuracy"):
            row = probe[probe["probe"] == probe_name].iloc[0]
            rules[probe_name] = {
                "estimate": safe_float(row.estimate),
                "ci_high": safe_float(row.ci_high),
                "random_level": safe_float(row.random_level) if pd.notna(row.random_level) else None,
                "passed": bool(row.passed),
            }
    core_passed = all(item["passed"] for item in rules.values())
    conclusion = (
        "Physics响应具有结构化多样性"
        if config.mode == "full" and core_passed
        else "未判定：smoke模式只验证流程" if config.mode == "smoke" else "Physics响应未通过全部结构化多样性验收规则"
    )
    attribution = {
        row.metric: {
            "estimate": safe_float(row.estimate),
            "ci_low": safe_float(row.ci_low),
            "ci_high": safe_float(row.ci_high),
            "physics_increases_structure": bool(row.physics_increases_structure),
        }
        for row in paired.itertuples()
    } if len(paired) else {}
    return {
        "mode": config.mode,
        "conclusion": conclusion,
        "structured_diversity_accepted": bool(config.mode == "full" and core_passed),
        "interpretation_limit": "响应仅按结构化多样性解释，不等同于真实水声信道。",
        "rules": rules,
        "physics_attribution": attribution,
    }


def write_report(
    config: AuditConfig,
    validation: dict[str, Any],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    probes: pd.DataFrame,
    acceptance: dict[str, Any],
) -> Path:
    def format_main(condition: str) -> str:
        lines = []
        for row in summary[(summary.condition == condition) & summary.metric.isin(MAIN_METRICS)].itertuples():
            lines.append(
                f"| {METRIC_NAMES[row.metric]} | {row.estimate:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | "
                f"{row.p_holm:.6g} | {'通过' if row.passed else '未通过'} |"
            )
        return "\n".join(lines)

    main_sections = []
    for condition in config.conditions:
        main_sections.append(
            f"### {CONDITION_NAMES[condition]}\n\n"
            "| 指标 | 估计值 | 95% CI | Holm校正p | 结果 |\n|---|---:|---:|---:|---|\n"
            + format_main(condition)
        )
    probe_summary = probes[probes.scope == "condition_summary"]
    probe_lines = [
        f"| {CONDITION_NAMES[row.condition]} | {row.probe} | {row.estimate:.6f} | "
        f"[{row.ci_low:.6f}, {row.ci_high:.6f}] | {'通过' if row.passed else '未通过'} |"
        for row in probe_summary.itertuples()
    ]
    paired_lines = [
        f"| {METRIC_NAMES[row.metric]} | {row.estimate:+.6f} | [{row.ci_low:+.6f}, {row.ci_high:+.6f}] | "
        f"{'是' if row.physics_increases_structure else '否'} |"
        for row in paired.itertuples()
    ]
    report = f"""# 既有模型信道响应 h 多样性审计报告

## 审计范围

- 模式：`{config.mode}`。
- 只读检查点目录：`{config.checkpoint_output.resolve()}`。
- 审计输出目录：`{config.output_dir.resolve()}`。
- 已核验检查点：{validation['checkpoint_count']} 个；数据 SHA-256：`{validation['computed_data_hash']}`。
- 数据固定清单：640 个 Click train、475 个多 Click train、3335 个可比较 Click、{validation['dataset']['matched_click_count']} 个完整匹配（{validation['dataset']['matched_coverage']:.1%}）。
- 响应不跨随机种子直接比较；距离和交换误差均先在单个检查点内计算。

## 主检验

{chr(10).join(main_sections)}

置信区间按“随机种子×录音来源”分层 Bootstrap；单侧置换检验在录音、背景类型和 5 dB SNR 层内破坏差值方向，并对每个条件的三个主检验执行 Holm 校正。

## 混杂探针

| 条件 | 探针 | 测试估计 | 95% CI | 结果 |
|---|---|---:|---:|---|
{chr(10).join(probe_lines)}

Ridge 和多项 Logistic 探针只读取冻结响应；训练折拟合、验证折选择正则、测试折报告。

## Physics 项归因

| 指标 | Physics − Recon+BCE | 95%配对区间 | 区间下限>0 |
|---|---:|---:|---|
{chr(10).join(paired_lines)}

## 验收结论

**{acceptance['conclusion']}**

该结论严格限于“响应具有结构化多样性”。本审计不宣称估计响应 h 等于真实水声传播信道，也不使用测试结果修改阈值、配对规则或模型。

## 图件与明细

- `figures/h_distance_hierarchy.png`：同 train、混合/干净、不同 train 距离层级。
- `figures/h_swap_reconstruction.png`：信道交换重建误差。
- `figures/h_collapse_effective_rank.png`：尾部能量与有效秩。
- `figures/h_representative_edp.png`：代表性归一化能量时延分布。
- `metrics/`：样本诊断、配对指标、Bootstrap/置换结果和混杂探针。
"""
    path = config.output_dir / "h_diversity_report_zh.md"
    path.write_text(report, encoding="utf-8")
    return path


def verify_artifacts(
    config: AuditConfig,
    diagnostics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    probes: pd.DataFrame,
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    response_files = list((config.output_dir / "responses").rglob("*.npz"))
    expected_response_files = len(config.conditions) * len(config.active_seeds) * len(config.folds) * 2
    figure_names = {
        "h_distance_hierarchy.png",
        "h_swap_reconstruction.png",
        "h_collapse_effective_rank.png",
        "h_representative_edp.png",
    }
    checks = {
        "response_file_count": len(response_files),
        "expected_response_file_count": expected_response_files,
        "diagnostic_key_unique": bool(not diagnostics.duplicated(["condition", "seed", "fold", "view", "sample_id"]).any()),
        "pair_key_unique": bool(not pair_metrics.duplicated(["condition", "seed", "fold", "sample_id"]).any()),
        "bootstrap_runs_all_valid": bool(
            summary.loc[summary.metric.isin(list(MAIN_METRICS) + ["collapse_rate"]), "bootstrap_runs_valid"].dropna().eq(config.bootstrap_runs).all()
        ),
        "permutation_runs_all_valid": bool(
            summary.loc[summary.metric.isin(MAIN_METRICS), "permutation_runs_valid"].dropna().eq(config.permutation_runs).all()
        ),
        "paired_bootstrap_all_valid": bool(paired.empty or paired["bootstrap_runs_valid"].eq(config.bootstrap_runs).all()),
        "probe_bootstrap_all_valid": bool(
            probes.loc[probes.scope == "condition_summary", "bootstrap_runs_valid"].dropna().eq(config.bootstrap_runs).all()
        ),
        "figures_complete": figure_names.issubset({path.name for path in (config.output_dir / "figures").glob("*.png")}),
        "acceptance_json_exists": (config.output_dir / "h_diversity_acceptance.json").exists(),
        "report_exists": (config.output_dir / "h_diversity_report_zh.md").exists(),
    }
    checks["passed"] = bool(
        checks["response_file_count"] == checks["expected_response_file_count"]
        and all(value for key, value in checks.items() if key not in {"response_file_count", "expected_response_file_count", "passed"})
    )
    json_dump(config.output_dir / "artifact_validation.json", checks)
    return checks


def run(config: AuditConfig) -> dict[str, Any]:
    started = time.perf_counter()
    prepare_output(config)
    seed_everything(config.random_seed)
    device = resolve_device(config.device)
    manifest, mixed, pipeline_config, validation, pair_manifest = validate_inputs_readonly(config)
    clean = materialize_clean_waveforms(manifest, pipeline_config, config.output_dir)
    positives = manifest[manifest["label"].astype(int) == 1].copy()

    diagnostics_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    probe_rows: list[dict[str, Any]] = []
    probe_prediction_frames: list[pd.DataFrame] = []
    fold_stats: list[dict[str, Any]] = []
    representative: dict[str, list[np.ndarray]] = {condition: [] for condition in config.conditions}
    total = len(config.conditions) * len(config.active_seeds) * len(config.folds)
    overall = tqdm(total=total, desc="总体阶段 3/6：冻结检查点推理与配对", unit="检查点")
    for condition in config.conditions:
        for seed in config.active_seeds:
            for fold in config.folds:
                path = checkpoint_path(config.checkpoint_output, condition, seed, fold)
                saved = torch.load(path, map_location=device, weights_only=True)
                model = PhysicalClassifier(pipeline_config.sample_count, pipeline_config.response_count).to(device)
                model.load_state_dict(saved["state_dict"], strict=True)
                diagnostics, pairs, probes, probe_predictions, stats = checkpoint_metrics(
                    condition,
                    seed,
                    fold,
                    saved,
                    model,
                    manifest,
                    mixed,
                    clean,
                    pair_manifest,
                    device,
                    config,
                )
                diagnostics_frames.append(diagnostics)
                pair_frames.append(pairs)
                probe_rows.extend(probes)
                probe_prediction_frames.append(probe_predictions)
                fold_stats.append(stats)
                mixed_file = np.load(response_path(config.output_dir, condition, seed, fold, "mixed"))
                representative[condition].append(energy_delay_descriptor(mixed_file["response"]).mean(axis=0))
                del mixed_file, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                overall.update(1)
                overall.set_postfix(condition=condition, seed=seed, fold=fold, pairs=len(pairs))
    overall.close()

    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    pair_metrics = pd.concat(pair_frames, ignore_index=True)
    probe_predictions = pd.concat(probe_prediction_frames, ignore_index=True)
    diagnostics.to_csv(config.output_dir / "metrics" / "h_sample_diagnostics.csv", index=False, encoding="utf-8-sig")
    pair_metrics.to_csv(config.output_dir / "metrics" / "h_pair_metrics.csv", index=False, encoding="utf-8-sig")

    print("总体阶段 4/6：分层 Bootstrap 与置换检验")
    summary = summarize_main_metrics(pair_metrics, config, show_progress=True)
    paired = summarize_paired_conditions(pair_metrics, config, show_progress=True)
    collapse = bootstrap_collapse(diagnostics, config, show_progress=True)
    summary = pd.concat([summary, collapse], ignore_index=True, sort=False)
    summary = add_structure_diagnostics(summary, fold_stats)
    probes = summarize_probes(probe_rows, probe_predictions, config, show_progress=True)
    summary.to_csv(config.output_dir / "metrics" / "h_diversity_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(config.output_dir / "metrics" / "h_diversity_paired_differences.csv", index=False, encoding="utf-8-sig")
    probes.to_csv(config.output_dir / "metrics" / "h_confound_probes.csv", index=False, encoding="utf-8-sig")

    print("总体阶段 5/6：图件、验收 JSON 与中文报告")
    representative_mean = {condition: np.mean(values, axis=0) for condition, values in representative.items()}
    plot_outputs(pair_metrics, diagnostics, fold_stats, representative_mean, config.output_dir)
    acceptance = build_acceptance(config, validation, diagnostics, pair_metrics, summary, paired, probes)
    json_dump(config.output_dir / "h_diversity_acceptance.json", acceptance)
    report = write_report(config, validation, summary, paired, probes, acceptance)

    print("总体阶段 6/6：产物完整性核验")
    artifacts = verify_artifacts(config, diagnostics, pair_metrics, summary, paired, probes, acceptance)
    result = {
        "mode": config.mode,
        "device": str(device),
        "checkpoint_count": validation["checkpoint_count"],
        "mixed_response_count": int((diagnostics.view == "mixed").sum()),
        "clean_response_count": int((diagnostics.view == "clean").sum()),
        "pair_metric_rows": int(len(pair_metrics)),
        "acceptance": acceptance,
        "artifacts": artifacts,
        "report": str(report.resolve()),
        "elapsed_s": time.perf_counter() - started,
    }
    json_dump(config.output_dir / "run_summary.json", result)
    if not artifacts["passed"]:
        raise RuntimeError(f"审计产物核验失败：{artifacts}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_arguments() -> AuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--checkpoint-output", type=Path, required=True, help="现有低 SNR 输出目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="独立审计输出目录")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(EXPECTED_SEEDS))
    parser.add_argument("--bootstrap-runs", type=int, default=2_000)
    parser.add_argument("--permutation-runs", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force", action="store_true", help="只重建审计产物；永不修改检查点")
    arguments = parser.parse_args()
    seeds = tuple(dict.fromkeys(arguments.seeds))
    if not seeds:
        parser.error("--seeds 至少需要一个种子")
    if arguments.bootstrap_runs <= 0 or arguments.permutation_runs <= 0:
        parser.error("Bootstrap 与置换次数必须为正整数")
    if arguments.mode == "full" and (len(seeds) != 5 or set(seeds) != set(EXPECTED_SEEDS)):
        parser.error("full 模式必须使用固定五种子 20260826 ... 20260830")
    return AuditConfig(
        mode=arguments.mode,
        checkpoint_output=arguments.checkpoint_output.resolve(),
        output_dir=arguments.output_dir.resolve(),
        seeds=seeds,
        bootstrap_runs=arguments.bootstrap_runs,
        permutation_runs=arguments.permutation_runs,
        device=arguments.device,
        force=arguments.force,
    )


def main() -> None:
    run(parse_arguments())


if __name__ == "__main__":
    main()
