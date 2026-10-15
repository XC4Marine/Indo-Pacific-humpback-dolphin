#!/usr/bin/env python3
"""Run truth-free 5-fold OOF extraction for Experiment 20 v8."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from extractors_v8 import classical_prediction, prepare_segments
from model_v8 import SparseGateNet, gate_candidates, gate_loss
from protocol_v8 import (CONDITIONS, OUTPUT_ROOT, add_awgn, atomic_json, condition_name, discover_inputs, fold_rows,
                         parse_conditions, prediction_path, save_prediction, source_group, write_csv)
from truth_guard_v8 import assert_extraction_truth_isolation

SEEDS = (20260908, 20260909, 20260910)


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def _tensor(path: Path, snr_db: int | None) -> tuple[torch.Tensor, np.ndarray]:
    raw = add_awgn(np.load(path, allow_pickle=False), path.parent.name, snr_db)
    prepared = prepare_segments(raw)
    return torch.from_numpy(prepared.segments.astype(np.float32)), prepared.support_samples


def _validation_groups(train_paths: list[Path], fold: int) -> set[str]:
    groups = sorted({source_group(path) for path in train_paths},
                    key=lambda group: hashlib.sha256(f"v8-validation:{fold}:{group}".encode()).hexdigest())
    return set(groups[:max(1, round(.15 * len(groups)))])


def _checkpoint(fold: int, seed: int) -> Path:
    return OUTPUT_ROOT / "checkpoints" / f"sparse_gate_fold{fold}_seed{seed}.pt"


def _train_one(fold: int, seed: int, train_paths: list[Path], device: torch.device, epochs: int, batch_size: int, resume: bool) -> Path:
    checkpoint = _checkpoint(fold, seed)
    if resume and checkpoint.exists():
        return checkpoint
    validation_groups = _validation_groups(train_paths, fold)
    fitting = [path for path in train_paths if source_group(path) not in validation_groups]
    validation = [path for path in train_paths if source_group(path) in validation_groups]
    if not fitting or not validation:
        raise RuntimeError(f"invalid fold {fold} fitting/validation split")
    training = [_tensor(path, None)[0] for path in fitting]
    validating = [_tensor(path, None)[0] for path in validation]
    _seed(seed); model = SparseGateNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best = float("inf"); best_state = None; stale = 0
    for epoch in range(1, epochs + 1):
        model.train(); order = np.random.default_rng(seed + epoch).permutation(len(training)); losses = []
        for begin in range(0, len(order), batch_size):
            batch = torch.stack([training[int(index)] for index in order[begin:begin + batch_size]]).to(device)
            optimizer.zero_grad(set_to_none=True); result = model(batch); loss, _ = gate_loss(result, batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); validation_losses = []
        with torch.inference_mode():
            for values in validating:
                batch = values[None].to(device); result = model(batch); loss, _ = gate_loss(result, batch); validation_losses.append(float(loss))
        score = float(np.mean(validation_losses))
        if score < best - 1e-7:
            best, stale = score, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= 20:
            print(f"[总体进度：门控训练] fold {fold + 1}/5 seed {seed} epoch {epoch}/{epochs} | train={np.mean(losses):.5f} val={score:.5f}", flush=True)
        if stale >= 20:
            break
    if best_state is None:
        raise RuntimeError("sparse gate training did not produce a checkpoint")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "fold": fold, "seed": seed, "validation_loss": best,
                "validation_groups": sorted(validation_groups), "epochs_completed": epoch}, checkpoint)
    return checkpoint


def train_models(fold: int, paths: list[Path], device: torch.device, epochs: int, batch_size: int, resume: bool) -> list[Path]:
    rows = fold_rows(paths); test_groups = {row["source_group"] for row in rows if row["fold"] == fold}
    train_paths = [path for path in paths if source_group(path) not in test_groups]
    return [_train_one(fold, seed, train_paths, device, epochs, batch_size, resume) for seed in SEEDS]


def _load_models(checkpoints: list[Path], device: torch.device) -> list[SparseGateNet]:
    models = []
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model = SparseGateNet().to(device); model.load_state_dict(saved["state_dict"]); model.eval(); models.append(model)
    return models


def _gate_prediction(models: list[SparseGateNet], values: torch.Tensor, support: int, device: torch.device) -> tuple[list[list[float]], list[float], np.ndarray, float]:
    reconstructions, masks = [], []
    with torch.inference_mode():
        for model in models:
            result = model(values[None].to(device)); masks.append(result["mask"][0].cpu().numpy()); reconstructions.append(result["reconstruction"][0].cpu().numpy())
    mask = np.mean(masks, axis=0)
    delays = gate_candidates(mask, support)
    reconstruction = np.mean(reconstructions, axis=0)
    nrmse = float(np.linalg.norm(reconstruction - values.numpy()) / (np.linalg.norm(values.numpy()) + 1e-12))
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    for delay in delays:
        index = int(round(delay * 1e-3 * 576_000))
        if index < len(binary_mask): binary_mask[index] = 1
    return [list(delays) for _ in range(10)], list(delays), binary_mask, nrmse


def extract(folds: list[int], conditions: tuple[int | None, ...], device: torch.device, epochs: int, batch_size: int, resume: bool, limit: int | None) -> None:
    paths = discover_inputs(); rows = fold_rows(paths); fold_by_id = {row["train_id"]: int(row["fold"]) for row in rows}
    manifest: list[dict[str, object]] = []
    total = sum(sum(1 for path in paths if fold_by_id[path.parent.name] == fold) for fold in folds) * len(conditions) * 3
    done = 0
    for fold in folds:
        checkpoints = train_models(fold, paths, device, epochs, batch_size, resume); models = _load_models(checkpoints, device)
        held_out = [path for path in paths if fold_by_id[path.parent.name] == fold]
        if limit is not None: held_out = held_out[:limit]
        for snr_db in conditions:
            condition = condition_name(snr_db)
            for path in held_out:
                train_id = path.parent.name; tensor, supports = _tensor(path, snr_db)
                prepared = type("PreparedProxy", (), {"segments": tensor.numpy(), "support_samples": supports})()
                for method in ("cepstrum_cfar", "med", "sparse_gate"):
                    destination = prediction_path(condition, method, train_id)
                    if resume and destination.exists():
                        manifest.append({"train_id": train_id, "fold": fold, "condition": condition, "method": method, "status": "resumed", "artifact": str(destination)})
                        done += 1; continue
                    started = time.perf_counter()
                    try:
                        if method == "sparse_gate":
                            event, stable, mask, nrmse = _gate_prediction(models, tensor, int(np.median(supports)), device)
                        else:
                            event, stable, mask = classical_prediction(prepared, method); nrmse = float("nan")
                        elapsed = (time.perf_counter() - started) * 1e3
                        save_prediction(destination, event_delays=event, stable_delays=stable, mask=mask,
                                        reconstruction_nrmse=nrmse, runtime_ms=elapsed, fold=fold)
                        manifest.append({"train_id": train_id, "fold": fold, "condition": condition, "method": method, "status": "success", "artifact": str(destination), "runtime_ms": elapsed})
                    except Exception as exc:
                        manifest.append({"train_id": train_id, "fold": fold, "condition": condition, "method": method, "status": "failed", "error": repr(exc)})
                    done += 1
                    print(f"[总体进度：盲提取] {done}/{max(total, 1)} | fold {fold + 1}/5 | {condition} | {method} | {train_id}", flush=True)
                    write_csv(OUTPUT_ROOT / "extraction_manifest.csv", manifest)
    write_csv(OUTPUT_ROOT / "folds.csv", rows); write_csv(OUTPUT_ROOT / "extraction_manifest.csv", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "extract", "evaluate", "all"), default="all")
    parser.add_argument("--conditions", default="clean,5,-5")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="limit held-out paths per fold for a smoke run")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(); assert_extraction_truth_isolation(Path(__file__).resolve().parent)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    folds = list(range(5)) if args.folds == "all" else sorted({int(value) for value in args.folds.split(",")})
    if not folds or any(fold not in range(5) for fold in folds): raise ValueError("--folds must be all or comma-separated values 0..4")
    conditions = parse_conditions(args.conditions)
    atomic_json(OUTPUT_ROOT / "protocol.json", {"name": "experiment_v8_automatic_multipath_topology", "fold_seed": 20260910,
                "folds": folds, "conditions": ["clean" if value is None else value for value in conditions], "gate_seeds": list(SEEDS),
                "gate_training_input": "clean", "truth_isolation": True, "device": str(device)})
    if args.phase in ("train", "extract", "all"):
        if args.phase == "train":
            for fold in folds: train_models(fold, discover_inputs(), device, args.epochs, args.batch_size, args.resume)
        else: extract(folds, conditions, device, args.epochs, args.batch_size, args.resume, args.limit)
    if args.phase in ("evaluate", "all"):
        evaluator = importlib.import_module("evaluate_v8"); evaluator.main_from_runner(conditions)


if __name__ == "__main__":
    main()
