#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验23-v4：重划分五折与离散SNR检测曲线。

独立构造[-5,-3,...,15] dB的7,000条数据，重新提取改变正样本的BCSC，
并在组级平衡的新五折上训练C0/C1/C2各三个随机种子。

命令：prepare-data、extract-structure、make-folds、verify、train、report、all。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# BCSC在多进程下会调用BLAS；每个子进程固定为单线程，避免线程池乘法导致内存耗尽。
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
SOURCE = ROOT / "21_PhysicsInformedClickClassification" / "outputs_snr_m15_p15_full_5seeds"
OLD_CACHE = ROOT / "23_BCSCStructureClickClassification" / "v2_low_snr_gated" / "cache"
BASE_V4 = ROOT / "23_BCSCStructureClickClassification" / "experiment_v4_minimal.py"
BCSC_EXTRACTOR = ROOT / "23_BCSCStructureClickClassification" / "experiment.py"
PIPELINE = ROOT / "21_PhysicsInformedClickClassification" / "pipeline.py"
OUT = ROOT / "23_BCSCStructureClickClassification" / "v4_rebalanced_snr_m5_p15_step2_3seeds"

SNR_LEVELS = np.arange(-5, 16, 2, dtype=np.float64)
SNR_ASSIGNMENT_SEED = 20260915
FOLD_SEARCH_SEED = 20260915
BCSC_SEED_BASE = 20260912  # 与原缓存提取协议一致；仅用于稳定的样本级bootstrap。
SEEDS = (20260826, 20260827, 20260828)
CONDITIONS = ("C0_waveform", "C1_full_zero", "C2_v2_scalar")
LABELS = {"C0_waveform": "C0 Waveform", "C1_full_zero": "C1 Full Zero", "C2_v2_scalar": "C2 v2 Scalar"}
N_SAMPLES, WAVE_SAMPLES, PATH_BINS, N_FOLDS = 7000, 5761, 1133, 5
FOLD_GROUP_CAPACITIES = (7, 7, 7, 7, 6)
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    output_dir: Path = OUT
    batch_size: int = 64
    max_epochs: int = 12
    patience: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    structure_dropout: float = 0.20
    bootstrap_runs: int = 2000
    workers: int = max(1, min(6, os.cpu_count() or 1))
    device: str = "auto"


def load_module(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def base_v4() -> Any:
    return load_module(BASE_V4, "experiment23_v4_rebalanced_base")


def pipeline() -> Any:
    return load_module(PIPELINE, "experiment21_snr_rebalanced_pipeline")


def bcsc_extractor() -> Any:
    return load_module(BCSC_EXTRACTOR, "experiment23_snr_rebalanced_extractor")


def output_paths(config: Config) -> dict[str, Path]:
    paths = {"root": config.output_dir, "cache": config.output_dir / "cache", "models": config.output_dir / "models",
             "logs": config.output_dir / "logs", "tests": config.output_dir / "tests", "figures": config.output_dir / "figures"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dump_json(path: Path, value: Any) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, Path): return str(item)
        if isinstance(item, (np.integer,)): return int(item)
        if isinstance(item, (np.floating,)): return float(item) if math.isfinite(float(item)) else None
        if isinstance(item, dict): return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [safe(v) for v in item]
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_rows(values: np.ndarray, indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(indices), 128):
        digest.update(np.ascontiguousarray(values[indices[start:start + 128]]).tobytes())
    return digest.hexdigest()


def source_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    manifest_path, wave_path = SOURCE / "dataset_manifest.csv", SOURCE / "cache" / "preprocessed_waveforms.npy"
    map_path, done_path, status_path = OLD_CACHE / "structure_maps.npy", OLD_CACHE / "structure_completed.npy", OLD_CACHE / "structure_status.csv"
    required = (manifest_path, wave_path, map_path, done_path, status_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少实验21或旧BCSC缓存：" + "; ".join(missing))
    manifest = pd.read_csv(manifest_path)
    waves, maps, completed, status = np.load(wave_path, mmap_mode="r"), np.load(map_path, mmap_mode="r"), np.load(done_path), pd.read_csv(status_path)
    checks = {"manifest": len(manifest) == N_SAMPLES, "wave": tuple(waves.shape) == (N_SAMPLES, WAVE_SAMPLES),
              "maps": tuple(maps.shape) == (N_SAMPLES, 3, PATH_BINS), "completed": tuple(completed.shape) == (N_SAMPLES,) and bool(completed.all()),
              "classes": manifest.label.value_counts().to_dict() == {1: 3500, 0: 3500}, "groups": manifest.ori_file_num.nunique() == 34,
              "status": len(status) == N_SAMPLES and status.sample_index.nunique() == N_SAMPLES}
    if not all(checks.values()):
        raise RuntimeError(f"源数据合同失败：{checks}")
    return manifest, waves, maps, completed, status


def new_data_paths(config: Config) -> dict[str, Path]:
    cache = output_paths(config)["cache"]
    return {"cache": cache, "manifest": config.output_dir / "dataset_manifest.csv", "waves": cache / "preprocessed_waveforms.npy",
            "completed": cache / "waveform_completed.npy", "audit": config.output_dir / "waveform_audit.csv",
            "construction": config.output_dir / "data_construction.json", "maps": cache / "structure_maps.npy",
            "maps_done": cache / "structure_completed.npy", "raw_status": cache / "structure_status_raw.csv",
            "status": cache / "structure_status.csv", "reuse": cache / "negative_cache_reuse.json"}


def assign_discrete_snr(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    result = result.rename(columns={"test_fold": "original_test_fold"})
    result["target_snr_db"] = np.nan
    result["realized_snr_db"] = np.nan
    result["mix_alpha"] = np.nan
    result["snr_assignment_seed"] = np.nan
    for group, item in result[result.label.eq(1)].groupby("ori_file_num", sort=True):
        positions = item.index.to_numpy(dtype=int)
        rng = np.random.default_rng(SNR_ASSIGNMENT_SEED + int(group))
        targets = np.resize(SNR_LEVELS, len(positions)).copy()
        rng.shuffle(targets)
        result.loc[positions, "target_snr_db"] = targets
        result.loc[positions, "snr_assignment_seed"] = SNR_ASSIGNMENT_SEED + int(group)
    return result


def prepare_data(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    output_paths(config)
    paths = new_data_paths(config)
    source_manifest, source_waves, _, _, _ = source_inputs()
    if paths["manifest"].exists() and paths["waves"].exists() and paths["completed"].exists() and paths["audit"].exists():
        manifest, waves, completed = pd.read_csv(paths["manifest"]), np.load(paths["waves"], mmap_mode="r"), np.load(paths["completed"])
        if len(manifest) == N_SAMPLES and tuple(waves.shape) == (N_SAMPLES, WAVE_SAMPLES) and bool(completed.all()):
            return manifest, waves
    # 若上一次在波形构造阶段中断，保留已写入的目标SNR、实现SNR和混合系数；
    # 仅首次构造时才按固定种子分配离散SNR。
    manifest = pd.read_csv(paths["manifest"]) if paths["manifest"].exists() else assign_discrete_snr(source_manifest)
    if paths["waves"].exists() and paths["completed"].exists():
        waves = np.lib.format.open_memmap(paths["waves"], mode="r+", dtype=np.float32, shape=(N_SAMPLES, WAVE_SAMPLES))
        completed = np.load(paths["completed"])
    else:
        waves = np.lib.format.open_memmap(paths["waves"], mode="w+", dtype=np.float32, shape=(N_SAMPLES, WAVE_SAMPLES))
        waves[:] = 0.; waves.flush(); completed = np.zeros(N_SAMPLES, dtype=bool); np.save(paths["completed"], completed)
    audit_rows: dict[int, dict[str, Any]] = {}
    if paths["audit"].exists():
        audit_rows = pd.read_csv(paths["audit"]).set_index("sample_index").to_dict("index")
    pipe, positive = pipeline(), manifest.label.to_numpy(dtype=int) == 1
    compatibility = pipe.Config(output_dir=config.output_dir, positive_noise_mix=True, snr_min_db=-5., snr_max_db=15., mix_seed=SNR_ASSIGNMENT_SEED)
    negative_ids = np.flatnonzero(~positive)
    if not completed[negative_ids].all():
        waves[negative_ids] = source_waves[negative_ids]
        completed[negative_ids] = True
        for index in negative_ids:
            audit_rows[int(index)] = {"sample_index": int(index), "sample_id": manifest.at[index, "sample_id"], "label": 0,
                                      "target_snr_db": np.nan, "realized_snr_db": np.nan, "mix_alpha": np.nan,
                                      "reused_original_negative": True, "finite": bool(np.isfinite(waves[index]).all())}
        waves.flush(); np.save(paths["completed"], completed)
    pending = np.flatnonzero(positive & ~completed)
    if len(pending):
        print(f"[总体进度 5%] 离散SNR混合：已完成 {int(completed.sum())}/{N_SAMPLES}，待处理正样本 {len(pending)}", flush=True)
        bar = tqdm(total=len(pending), desc="总体进度：重混合正样本", unit="样本")
        for count, index in enumerate(pending, 1):
            row = manifest.iloc[int(index)]
            click = pipe.preprocess_component(pipe.read_waveform(row, compatibility), compatibility)
            noise = pipe.preprocess_component(pipe.read_timestamp_waveform(Path(str(row.mix_noise_wav_path)), float(row.mix_noise_timestamp_s), compatibility), compatibility)
            click_rms, noise_rms = float(np.sqrt(np.mean(click ** 2))), float(np.sqrt(np.mean(noise ** 2)))
            if click_rms < EPS or noise_rms < EPS:
                raise RuntimeError(f"混合分量RMS无效：{row.sample_id}")
            target = float(row.target_snr_db)
            alpha = click_rms / (noise_rms * 10. ** (target / 20.))
            realized = float(20. * np.log10(click_rms / (alpha * noise_rms)))
            waves[index] = pipe.peak_normalize(click + alpha * noise)
            completed[index] = True
            manifest.at[index, "realized_snr_db"], manifest.at[index, "mix_alpha"] = realized, alpha
            audit_rows[int(index)] = {"sample_index": int(index), "sample_id": row.sample_id, "label": 1, "target_snr_db": target,
                                      "realized_snr_db": realized, "snr_error_db": abs(realized - target), "mix_alpha": alpha,
                                      "click_rms": click_rms, "noise_rms": noise_rms, "reused_original_negative": False,
                                      "finite": bool(np.isfinite(waves[index]).all())}
            bar.update(1)
            if count % 25 == 0 or count == len(pending):
                waves.flush(); np.save(paths["completed"], completed)
                pd.DataFrame.from_dict(audit_rows, orient="index").sort_values("sample_index").to_csv(paths["audit"], index=False, encoding="utf-8-sig")
                manifest.to_csv(paths["manifest"], index=False, encoding="utf-8-sig")
        bar.close()
    waves.flush(); np.save(paths["completed"], completed)
    manifest.to_csv(paths["manifest"], index=False, encoding="utf-8-sig")
    audit = pd.DataFrame.from_dict(audit_rows, orient="index").sort_values("sample_index")
    audit.to_csv(paths["audit"], index=False, encoding="utf-8-sig")
    source_hash, current_hash = sha256_rows(source_waves, negative_ids), sha256_rows(waves, negative_ids)
    reuse = {"negative_sample_count": int(len(negative_ids)), "source_negative_sha256": source_hash, "new_negative_sha256": current_hash,
             "byte_identical": source_hash == current_hash, "source_wave_path": str(SOURCE / "cache" / "preprocessed_waveforms.npy")}
    dump_json(paths["reuse"], reuse)
    if not completed.all() or not reuse["byte_identical"]:
        raise RuntimeError("新数据缓存未完成或负样本未保持字节一致")
    dump_json(paths["construction"], {"snr_levels_db": SNR_LEVELS.tolist(), "snr_assignment_seed": SNR_ASSIGNMENT_SEED,
                                        "positive_assignment": "within_recording_balanced_cycle", "negative_reused": True,
                                        "source_manifest": str(SOURCE / "dataset_manifest.csv"), "source_waveforms": str(SOURCE / "cache" / "preprocessed_waveforms.npy")})
    return manifest, np.load(paths["waves"], mmap_mode="r")


def extract_worker(payload: tuple[int, np.ndarray, int]) -> tuple[int, np.ndarray, dict[str, Any]]:
    index, signal, seed = payload
    structure, metadata = bcsc_extractor().extract_structure_one(signal, seed)
    return index, structure, metadata


def status_frame(rows: dict[int, dict[str, Any]]) -> pd.DataFrame:
    """将可续跑的样本字典规范成唯一的sample_index列。"""
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame = frame.drop(columns=["sample_index"], errors="ignore")
    frame.index.name = "sample_index"
    return frame.reset_index().sort_values("sample_index")


def extract_structure(config: Config) -> tuple[pd.DataFrame, np.ndarray]:
    manifest, waves = prepare_data(config)
    paths = new_data_paths(config)
    _, _, old_maps, old_done, old_status = source_inputs()
    if paths["maps"].exists() and paths["maps_done"].exists():
        maps = np.lib.format.open_memmap(paths["maps"], mode="r+", dtype=np.float32, shape=(N_SAMPLES, 3, PATH_BINS))
        completed = np.load(paths["maps_done"])
    else:
        maps = np.lib.format.open_memmap(paths["maps"], mode="w+", dtype=np.float32, shape=(N_SAMPLES, 3, PATH_BINS))
        maps[:] = 0.; maps.flush(); completed = np.zeros(N_SAMPLES, dtype=bool); np.save(paths["maps_done"], completed)
    raw_rows: dict[int, dict[str, Any]] = {}
    if paths["raw_status"].exists():
        # 结构图完成标记才是恢复时的权威来源；状态CSV在进程中断的写入窗口内
        # 可能含重复行，因此保留同一sample_index最后一次完整诊断。
        saved_status = pd.read_csv(paths["raw_status"]).drop_duplicates("sample_index", keep="last")
        raw_rows = saved_status.set_index("sample_index").to_dict("index")
    negative_ids = np.flatnonzero(manifest.label.to_numpy(dtype=int) == 0)
    reuse = json.loads(paths["reuse"].read_text(encoding="utf-8"))
    if not bool(reuse["byte_identical"]) or not old_done.all(): raise RuntimeError("负样本波形或旧BCSC缓存不符合复用条件")
    if not completed[negative_ids].all():
        maps[negative_ids] = old_maps[negative_ids]; completed[negative_ids] = True
        maps.flush(); np.save(paths["maps_done"], completed)
    # 即使结构图已经在先前运行中写入，也恢复每条负样本的状态行，防止一次中断
    # 只留下正样本诊断而导致后续折平衡阶段缺少负样本协变量。
    old_indexed = old_status.drop_duplicates("sample_index", keep="last").set_index("sample_index")
    for index in negative_ids:
        if int(index) not in raw_rows:
            row = old_indexed.loc[int(index)].to_dict(); row.update({"sample_index": int(index), "reused_original_negative_cache": True})
            raw_rows[int(index)] = row
    pending = np.flatnonzero((manifest.label.to_numpy(dtype=int) == 1) & ~completed)
    if len(pending):
        print(f"[总体进度 15%] BCSC结构图：复用负样本 {len(negative_ids)} 条，待提取正样本 {len(pending)} 条", flush=True)
        bar = tqdm(total=len(pending), desc="总体进度：BCSC正样本提取", unit="样本")
        payloads = ((int(index), np.asarray(waves[index], dtype=np.float32), BCSC_SEED_BASE + int(index)) for index in pending)
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            for count, (index, structure, metadata) in enumerate(pool.map(extract_worker, payloads, chunksize=1), 1):
                maps[index] = structure; completed[index] = True; metadata.update({"sample_index": index, "reused_original_negative_cache": False})
                raw_rows[index] = metadata; bar.update(1)
                if count % 25 == 0 or count == len(pending):
                    maps.flush(); np.save(paths["maps_done"], completed)
                    status_frame(raw_rows).to_csv(paths["raw_status"], index=False, encoding="utf-8-sig")
        bar.close()
    maps.flush(); np.save(paths["maps_done"], completed)
    # 若中断恰好发生在结构图落盘和状态CSV落盘之间，结构图仍是有效的完成事实。
    # 对缺失诊断以图本身恢复可用于平衡的路径数量，避免无意义地重算已完成BCSC。
    for index in np.flatnonzero(completed):
        if int(index) not in raw_rows:
            raw_rows[int(index)] = {"status": "ok_recovered_from_map", "error": "", "direct_index": np.nan,
                                    "candidate_path_count": np.nan, "detected_path_count": int(np.count_nonzero(maps[index, 0])),
                                    "elapsed_s": np.nan, "reused_original_negative_cache": bool(manifest.at[index, "label"] == 0)}
    raw = status_frame(raw_rows)
    raw.to_csv(paths["raw_status"], index=False, encoding="utf-8-sig")
    if not completed.all() or tuple(maps.shape) != (N_SAMPLES, 3, PATH_BINS): raise RuntimeError("BCSC结构图缓存未完整写入")
    dump_json(paths["cache"] / "structure_cache_contract.json", {"shape": [N_SAMPLES, 3, PATH_BINS], "completed": int(completed.sum()),
                "positive_reextracted": int((manifest.label == 1).sum()), "negative_reused_after_byte_validation": int((manifest.label == 0).sum()),
                "bcsc": {"atom_points": 49, "outer_iterations": 35, "lambda_ratio": .075, "bootstrap_runs": 8}})
    return raw, np.load(paths["maps"], mmap_mode="r")


def feature_table(manifest: pd.DataFrame, maps: np.ndarray, status: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = np.sort(manifest.ori_file_num.unique()).astype(int)
    sample = manifest[["ori_file_num", "label", "noise_type", "mix_noise_type"]].copy()
    # 每个延迟格最多对应一条初始候选，P非零格数与可检测路径数等价；从结构图直接
    # 计算也使平衡统计独立于可续跑诊断CSV的落盘时机。
    sample["detected_paths"] = np.asarray(np.count_nonzero(maps[:, 0, :], axis=1), dtype=float)
    sample["p_mass"] = np.asarray(maps[:, 0, :].sum(axis=1), dtype=float)
    sample["u_mass"] = np.asarray(maps[:, 2, :].sum(axis=1), dtype=float)
    names, columns = ["samples", "positive", "negative"], [np.ones(N_SAMPLES), sample.label.eq(1).to_numpy(float), sample.label.eq(0).to_numpy(float)]
    for kind in sorted(sample.loc[sample.label.eq(0), "noise_type"].dropna().unique()):
        names.append(f"negative_noise::{kind}"); columns.append((sample.label.eq(0) & sample.noise_type.eq(kind)).to_numpy(float))
    for kind in sorted(sample.loc[sample.label.eq(1), "mix_noise_type"].dropna().unique()):
        names.append(f"positive_background::{kind}"); columns.append((sample.label.eq(1) & sample.mix_noise_type.eq(kind)).to_numpy(float))
    names.extend(["bcsc_detected_paths", "bcsc_p_mass", "bcsc_u_mass"]); columns.extend([sample.detected_paths.to_numpy(float), sample.p_mass.to_numpy(float), sample.u_mass.to_numpy(float)])
    matrix = np.empty((len(groups), len(names)), dtype=float)
    for position, group in enumerate(groups):
        mask = sample.ori_file_num.to_numpy(dtype=int) == group
        matrix[position] = [float(np.sum(column[mask])) for column in columns]
    return groups, matrix, names


def balance_score(assignment: np.ndarray, values: np.ndarray) -> float:
    totals, target = values.sum(axis=0), values.sum(axis=0) / N_FOLDS
    fold_values = np.vstack([values[assignment == fold].sum(axis=0) for fold in range(N_FOLDS)])
    relative = (fold_values - target) / np.maximum(totals, 1.)
    return float(np.sum(relative ** 2))


def optimize_assignment(groups: np.ndarray, values: np.ndarray, original: np.ndarray) -> tuple[np.ndarray, float]:
    rng, best, best_score = np.random.default_rng(FOLD_SEARCH_SEED), None, math.inf
    # 34个录音组下，32次固定种子重启已足以覆盖不同初始组组合；更高重启数只会
    # 重复相同的局部交换搜索而延迟训练阶段。
    for _ in range(32):
        order = rng.permutation(len(groups)); assignment = np.empty(len(groups), dtype=int); start = 0
        for fold, capacity in enumerate(FOLD_GROUP_CAPACITIES):
            assignment[order[start:start + capacity]] = fold; start += capacity
        current = balance_score(assignment, values)
        changed = True
        while changed:
            changed, candidate, candidate_score = False, None, current
            for left in range(len(groups)):
                for right in range(left + 1, len(groups)):
                    if assignment[left] == assignment[right]: continue
                    trial = assignment.copy(); trial[left], trial[right] = trial[right], trial[left]
                    score = balance_score(trial, values)
                    if score < candidate_score - 1e-14:
                        candidate, candidate_score = trial, score
            if candidate is not None: assignment, current, changed = candidate, candidate_score, True
        # 防止仅以fold标签置换的旧划分被误报为新的独立划分。
        same_partition = np.array_equal(assignment[:, None] == assignment[None, :], original[:, None] == original[None, :])
        if not same_partition and current < best_score:
            best, best_score = assignment.copy(), current
    if best is None: raise RuntimeError("未找到不同于原划分的可行组级平衡划分")
    return best, best_score


def make_folds(config: Config) -> pd.DataFrame:
    manifest, _ = prepare_data(config); raw_status, maps = extract_structure(config); paths = new_data_paths(config)
    groups, values, names = feature_table(manifest, maps, raw_status)
    original_by_group = manifest.groupby("ori_file_num").original_test_fold.first().reindex(groups).to_numpy(dtype=int)
    assignment, new_score = optimize_assignment(groups, values, original_by_group)
    mapping = dict(zip(groups, assignment, strict=True)); manifest["test_fold"] = manifest.ori_file_num.map(mapping).astype(int)
    assignment_frame = pd.DataFrame({"ori_file_num": groups, "original_test_fold": original_by_group, "test_fold": assignment})
    assignment_frame["changed_group"] = assignment_frame.original_test_fold.ne(assignment_frame.test_fold)
    assignment_frame.to_csv(config.output_dir / "new_fold_assignment.csv", index=False, encoding="utf-8-sig")
    summaries = []
    for scheme, labels in (("original", original_by_group), ("rebalanced", assignment)):
        score = balance_score(labels, values)
        for fold in range(N_FOLDS):
            group_mask = labels == fold; row = {"scheme": scheme, "fold": fold, "balance_score": score, "group_count": int(group_mask.sum())}
            row.update({name: float(values[group_mask, position].sum()) for position, name in enumerate(names)}); summaries.append(row)
    details = pd.DataFrame(summaries)
    details.to_csv(config.output_dir / "fold_balance_details.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"scheme": "original", "balance_score": balance_score(original_by_group, values), "groups_changed": 0},
                  {"scheme": "rebalanced", "balance_score": new_score, "groups_changed": int(assignment_frame.changed_group.sum())}]).to_csv(config.output_dir / "fold_balance_comparison.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(paths["manifest"], index=False, encoding="utf-8-sig")
    meta = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "label", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    meta.merge(raw_status.drop(columns=[name for name in ("sample_id", "label", "ori_file_num", "noise_type", "test_fold") if name in raw_status], errors="ignore"), on="sample_index", how="left", validate="one_to_one").to_csv(paths["status"], index=False, encoding="utf-8-sig")
    return manifest


def split(manifest: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, validation = manifest.test_fold.to_numpy(dtype=int), (fold + 1) % N_FOLDS
    return np.flatnonzero((values != fold) & (values != validation)), np.flatnonzero(values == validation), np.flatnonzero(values == fold)


def load_training_data(config: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    paths = new_data_paths(config); required = (paths["manifest"], paths["waves"], paths["maps"], paths["maps_done"], paths["status"])
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("请先完成数据、BCSC和重划分：" + "; ".join(missing))
    manifest, waves, maps, done, status = pd.read_csv(paths["manifest"]), np.load(paths["waves"], mmap_mode="r"), np.load(paths["maps"], mmap_mode="r"), np.load(paths["maps_done"]), pd.read_csv(paths["status"])
    if tuple(waves.shape) != (N_SAMPLES, WAVE_SAMPLES) or tuple(maps.shape) != (N_SAMPLES, 3, PATH_BINS) or not done.all(): raise RuntimeError("训练缓存合同失败")
    return manifest, waves, maps, status


def train(config: Config, resume: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest, waves, maps, _ = load_training_data(config); base = base_v4()
    # 新实验主终点与checkpoint选择均为全[-5,15] dB AUROC；网络与优化协议来自v4。
    base.split = split
    base.low_snr_mask = lambda frame: np.ones(len(frame), dtype=bool)
    jobs = [(seed, condition, fold) for seed in SEEDS for condition in CONDITIONS for fold in range(N_FOLDS)]
    rows, predictions, diagnostics = [], [], []
    bar = tqdm(total=len(jobs), desc="总体进度：C0/C1/C2三种子训练", unit="模型折")
    for position, (seed, condition, fold) in enumerate(jobs, 1):
        base.SEED = seed
        row, prediction, diagnostic = base.train_one(condition, fold, waves, maps, manifest, config, resume)
        row["checkpoint_selection_endpoint"] = "all_snr_auroc"
        row["validation_all_snr_auroc"] = row.pop("best_validation_low_snr_auroc")
        row["all_snr_auroc_duplicate"] = row.pop("low_snr_auroc")
        rows.append(row); predictions.append(prediction); diagnostics.append(diagnostic)
        bar.update(1); bar.set_postfix_str(f"{position}/45 {condition} seed={seed} fold={fold} AUROC={row['auroc']:.3f}")
    bar.close()
    metrics, oof = pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)
    meta = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "label", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]]
    oof = oof.merge(meta, on=["sample_index", "test_fold"], how="left", validate="many_to_one")
    if len(metrics) != 45 or len(oof) != 63000 or oof.duplicated(["seed", "condition", "sample_index"]).any(): raise RuntimeError("45折训练OOF合同失败")
    metrics.to_csv(config.output_dir / "full_fold_metrics.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(config.output_dir / "full_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(diagnostics, ignore_index=True).to_csv(config.output_dir / "full_epoch_diagnostics.csv", index=False, encoding="utf-8-sig")
    return metrics, oof


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    fixed = ["label", "sample_id", "ori_file_num", "noise_type", "target_snr_db", "test_fold"]
    aggregation = {name: "first" for name in fixed}; aggregation.update({"p_click": "mean", "logit": "mean"})
    result = predictions.groupby(["condition", "sample_index"], as_index=False).agg(aggregation)
    if len(result) != 21000 or result.duplicated(["condition", "sample_index"]).any(): raise RuntimeError("三种子平均OOF合同失败")
    return result


def bootstrap_delta(frame: pd.DataFrame, left: str, right: str, seed: int, runs: int) -> dict[str, Any]:
    a, b = frame[frame.condition.eq(left)].sort_values("sample_index"), frame[frame.condition.eq(right)].sort_values("sample_index")
    if not np.array_equal(a.sample_index.to_numpy(), b.sample_index.to_numpy()): raise RuntimeError("配对预测样本不一致")
    labels, pa, pb = a.label.to_numpy(int), a.p_click.to_numpy(float), b.p_click.to_numpy(float)
    groups = np.sort(a.ori_file_num.unique()); group_ids = a.ori_file_num.to_numpy(int); lookup = {group: i for i, group in enumerate(groups)}
    codes = np.asarray([lookup[value] for value in group_ids], dtype=int)
    point = float(roc_auc_score(labels, pa) - roc_auc_score(labels, pb)); rng, values = np.random.default_rng(seed), np.empty(runs, dtype=float)
    for index in range(runs):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        weight = counts[codes]
        values[index] = roc_auc_score(labels, pa, sample_weight=weight) - roc_auc_score(labels, pb, sample_weight=weight)
    low, high = float(np.quantile(values, .025)), float(np.quantile(values, .975))
    return {"left": left, "right": right, "point_difference": point, "ci_low": low, "ci_high": high,
            "bootstrap_runs_requested": runs, "bootstrap_runs_valid": runs, "supports_positive_gain": bool(low > 0)}


def snr_metrics(predictions: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        all_item = ensemble[ensemble.condition.eq(condition)]
        for level in SNR_LEVELS:
            mask = all_item.label.eq(0) | (all_item.label.eq(1) & np.isclose(all_item.target_snr_db, level))
            item = all_item[mask]; labels, probability = item.label.to_numpy(int), item.p_click.to_numpy(float)
            recalls = []
            for seed in SEEDS:
                seed_item = predictions[(predictions.condition.eq(condition)) & (predictions.seed.eq(seed))]
                seed_mask = seed_item.label.eq(0) | (seed_item.label.eq(1) & np.isclose(seed_item.target_snr_db, level))
                view = seed_item[seed_mask]; recalls.append(recall_score(view.label, view.predicted_label, zero_division=0))
            rows.append({"condition": condition, "snr_db": float(level), "positive_count": int(item.label.eq(1).sum()), "negative_count": int(item.label.eq(0).sum()),
                         "auroc": float(roc_auc_score(labels, probability)), "auprc": float(average_precision_score(labels, probability)),
                         "recall_seed_mean": float(np.mean(recalls)), "recall_seed_std": float(np.std(recalls, ddof=1))})
    return pd.DataFrame(rows)


def plot_snr(metrics: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    columns = (("auroc", "AUROC"), ("auprc", "AUPRC"), ("recall_seed_mean", "Recall（三种子均值）"))
    for condition in CONDITIONS:
        item = metrics[metrics.condition.eq(condition)].sort_values("snr_db")
        for axis, (column, label) in zip(axes, columns, strict=True): axis.plot(item.snr_db, item[column], marker="o", label=LABELS[condition])
    for axis, (_, label) in zip(axes, columns, strict=True):
        axis.set(xlabel="正样本目标SNR (dB)", ylabel=label, xticks=SNR_LEVELS, ylim=(0, 1.02)); axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.suptitle("重划分五折：离散SNR检测结果（三种子平均OOF）", x=.03, ha="left")
    fig.tight_layout(); fig.savefig(destination, dpi=200); plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    columns, values = list(frame.columns), []
    for row in frame.itertuples(index=False, name=None): values.append([f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) and math.isfinite(x) else str(x) for x in row])
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |", *["| " + " | ".join(row) + " |" for row in values]])


def report(config: Config) -> None:
    metric_path, prediction_path = config.output_dir / "full_fold_metrics.csv", config.output_dir / "full_oof_predictions.csv"
    if not metric_path.exists() or not prediction_path.exists(): raise FileNotFoundError("请先完成train")
    metrics, predictions = pd.read_csv(metric_path), pd.read_csv(prediction_path)
    ensemble = ensemble_predictions(predictions); ensemble.to_csv(config.output_dir / "three_seed_mean_probability_oof.csv", index=False, encoding="utf-8-sig")
    full, per_level = [], []
    for number, right in enumerate(("C0_waveform", "C1_full_zero")):
        full.append({**bootstrap_delta(ensemble, "C2_v2_scalar", right, 20261000 + number, config.bootstrap_runs), "endpoint": "all_snr_auroc"})
        for level_index, level in enumerate(SNR_LEVELS):
            mask = ensemble.label.eq(0) | (ensemble.label.eq(1) & np.isclose(ensemble.target_snr_db, level))
            result = bootstrap_delta(ensemble[mask], "C2_v2_scalar", right, 20262000 + number * 100 + level_index, config.bootstrap_runs)
            result.update({"endpoint": "snr_point_auroc", "snr_db": float(level)}); per_level.append(result)
    full_frame, level_frame = pd.DataFrame(full), pd.DataFrame(per_level)
    full_frame.to_csv(config.output_dir / "full_range_structural_gain_bootstrap.csv", index=False, encoding="utf-8-sig")
    level_frame.to_csv(config.output_dir / "snr_structural_gain_bootstrap.csv", index=False, encoding="utf-8-sig")
    detection = snr_metrics(predictions, ensemble); detection.to_csv(config.output_dir / "snr_detection_metrics.csv", index=False, encoding="utf-8-sig")
    plot_snr(detection, config.output_dir / "snr_detection_curves.png")
    seed_summary = metrics.groupby(["condition", "seed"], as_index=False).agg(auroc=("auroc", "mean"), auprc=("auprc", "mean"))
    summary = seed_summary.groupby("condition", as_index=False).agg(seeds=("seed", "nunique"), auroc_mean=("auroc", "mean"), auroc_seed_std=("auroc", "std"), auprc_mean=("auprc", "mean"), auprc_seed_std=("auprc", "std"))
    summary.to_csv(config.output_dir / "three_seed_summary.csv", index=False, encoding="utf-8-sig")
    supports = bool(full_frame.supports_positive_gain.all())
    text = ["# 实验23-v4：重划分五折与离散SNR检测曲线", "", "## 设计", "", "正样本在每个录音组内均衡分配到-5,-3,...,15 dB；C0/C1/C2均完成新五折和三个训练种子，共45个模型折。三种子概率均值用于34个录音组配对Bootstrap。", "", "## 三种子五折汇总", "", markdown_table(summary), "", "## 全范围结构增益主检验", "", markdown_table(full_frame), "", "## 结论", "", "C2相对C0与C1的全范围AUROC配对Bootstrap下限均大于0；在此独立划分与离散SNR数据下，结构增益成立。" if supports else "至少一个C2主比较的全范围AUROC配对Bootstrap下限不大于0；只能报告结构增益趋势，不能声称结构增益成立。", "", "## 分SNR检测曲线", "", "11个SNR点的AUROC、AUPRC和Recall为描述性检测结果；逐点C2增量区间见`snr_structural_gain_bootstrap.csv`，不作为经多重比较校正的主结论。", "", markdown_table(detection)]
    (config.output_dir / "research_report_zh.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def verify(config: Config) -> dict[str, Any]:
    manifest, waves, maps, status = load_training_data(config); paths = new_data_paths(config); base = base_v4()
    positive, negative = manifest.label.to_numpy(int) == 1, manifest.label.to_numpy(int) == 0
    targets, realized = manifest.loc[positive, "target_snr_db"].to_numpy(float), manifest.loc[positive, "realized_snr_db"].to_numpy(float)
    levels_ok = bool(np.isin(targets, SNR_LEVELS).all())
    group_counts = manifest.loc[positive].groupby(["ori_file_num", "target_snr_db"]).size().unstack(fill_value=0)
    snr_balance = bool((group_counts.max(axis=1) - group_counts.min(axis=1) <= 1).all())
    coverage = {fold: sorted(manifest.loc[(manifest.test_fold == fold) & positive, "target_snr_db"].unique().tolist()) for fold in range(N_FOLDS)}
    reuse = json.loads(paths["reuse"].read_text(encoding="utf-8"))
    device = base.device_for(config.device); ids = np.arange(4); model = base.FusionModel("C2_v2_scalar", config.dropout, config.structure_dropout).to(device)
    logits = model(torch.as_tensor(np.asarray(waves[ids]), dtype=torch.float32, device=device), torch.as_tensor(np.asarray(maps[ids]), dtype=torch.float32, device=device))
    result = {"sample_shape": list(waves.shape), "structure_shape": list(maps.shape), "classes": manifest.label.value_counts().to_dict(), "recording_groups": int(manifest.ori_file_num.nunique()),
              "snr_levels": SNR_LEVELS.tolist(), "only_requested_snr_levels": levels_ok, "max_snr_realization_error_db": float(np.max(np.abs(realized - targets))),
              "snr_balanced_within_recording": snr_balance, "negative_target_snr_missing": bool(manifest.loc[negative, "target_snr_db"].isna().all()),
              "negative_cache_byte_identical": bool(reuse["byte_identical"]), "structure_completed": bool(len(status) == N_SAMPLES and np.isfinite(maps).all()),
              "all_bcsc_status_ok": bool(status.status.astype(str).str.startswith("ok").all()),
              "no_group_leakage": int(manifest.groupby("ori_file_num").test_fold.nunique().max()) == 1, "fold_group_counts": manifest.groupby("test_fold").ori_file_num.nunique().to_dict(),
              "snr_coverage_by_fold": coverage, "all_folds_cover_all_snr_levels": all(np.array_equal(np.asarray(value), SNR_LEVELS) for value in coverage.values()),
              "transformer_layers": len(model.waveform.encoder.layers), "finite_logits": bool(torch.isfinite(logits).all())}
    result["passed"] = bool(result["sample_shape"] == [N_SAMPLES, WAVE_SAMPLES] and result["structure_shape"] == [N_SAMPLES, 3, PATH_BINS] and result["classes"] == {1: 3500, 0: 3500} and result["recording_groups"] == 34 and levels_ok and result["max_snr_realization_error_db"] < 1e-8 and snr_balance and result["negative_target_snr_missing"] and result["negative_cache_byte_identical"] and result["structure_completed"] and result["all_bcsc_status_ok"] and result["no_group_leakage"] and sorted(result["fold_group_counts"].values()) == [6, 7, 7, 7, 7] and result["all_folds_cover_all_snr_levels"] and result["transformer_layers"] == 1 and result["finite_logits"])
    dump_json(output_paths(config)["tests"] / "verification.json", result)
    if not result["passed"]: raise RuntimeError(f"验证失败：{result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-data", "extract-structure", "make-folds", "verify", "train", "report", "all"))
    parser.add_argument("--device", default="auto"); parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--bootstrap-runs", type=int, default=2000); parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(); config = Config(device=args.device, workers=args.workers, bootstrap_runs=args.bootstrap_runs)
    output_paths(config); dump_json(config.output_dir / "run_config.json", {**asdict(config), "output_dir": str(config.output_dir), "snr_levels_db": SNR_LEVELS.tolist(), "snr_assignment_seed": SNR_ASSIGNMENT_SEED, "fold_search_seed": FOLD_SEARCH_SEED, "training_seeds": list(SEEDS), "conditions": list(CONDITIONS), "unique_model_folds": 45, "primary_endpoint": "all_snr_auroc"})
    if args.command in ("prepare-data", "all"): print("[总体进度 0%] 构造离散SNR数据集", flush=True); prepare_data(config)
    if args.command in ("extract-structure", "all"): print("[总体进度 10%] 提取新正样本BCSC并复用已验证负样本缓存", flush=True); extract_structure(config)
    if args.command in ("make-folds", "all"): print("[总体进度 40%] 搜索组级平衡的新五折", flush=True); make_folds(config)
    if args.command in ("verify", "all"): print("[总体进度 45%] 验证数据、BCSC和折划分合同", flush=True); verify(config)
    if args.command in ("train", "all"): print("[总体进度 50%] 训练45个C0/C1/C2模型折", flush=True); train(config, resume=not args.no_resume)
    if args.command in ("report", "all"): print("[总体进度 95%] 汇总Bootstrap、离散SNR曲线和报告", flush=True); report(config); print(f"[总体进度 100%] 完成：{config.output_dir}", flush=True)


if __name__ == "__main__":
    main()
