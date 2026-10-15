#!/usr/bin/env python3
"""Experiment 20 v6: BCSC-guided neural residual refiner for channel response shape.

Training reads synthetic propagation truth only for train/validation groups.  Test
inference consumes precomputed blind BCSC artifacts; evaluation reads truth later.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
V5_ROOT = HERE.parent / "experiment_v5"
V4_ROOT = HERE.parent / "experiment_v4"
sys.path[:0] = [str(V5_ROOT), str(V4_ROOT)]
from protocol_v5 import DATA_ROOT, OUTPUT_ROOT as V5_OUTPUT, discover_inputs, group_split  # noqa: E402
from evaluate_bcsc_v4 import transfer_from_arrivals  # noqa: E402


OUT = HERE / "outputs"
FREQ = np.arange(20_000, 190_001, 5_000, dtype=float)
CONDITIONS: tuple[str, ...] = ("clean", "snr_m05", "snr_p05")


@dataclass(frozen=True)
class Config:
    epochs: int = 450
    batch_size: int = 64
    lr: float = 1e-3
    seeds: tuple[int, ...] = (20260918, 20260919, 20260920)
    patience: int = 80


def atomic_json(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def response_shape(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(np.asarray(channel, dtype=float), n=16_384)
    bins = FREQ / (576_000 / 16_384); lo = np.floor(bins).astype(int); weight = bins - lo
    values = spectrum[lo] * (1 - weight) + spectrum[lo + 1] * weight
    amplitude = 20 * np.log10(np.abs(values) + 1e-15); amplitude -= np.median(amplitude)
    phase = np.unwrap(np.angle(values)); slope, intercept = np.polyfit(FREQ, phase, 1)
    phase = np.angle(np.exp(1j * (phase - slope * FREQ - intercept)))
    return amplitude.astype(np.float32), phase.astype(np.float32)


def truth_shape(directory: Path) -> np.ndarray:
    with np.load(directory / "event_channels.npz", allow_pickle=True) as item:
        arrivals = item["event_synthesis_arrivals"]; first = float(item["first_arrival_s"])
    values = []
    for event in arrivals:
        transfer = transfer_from_arrivals(event, first)
        amplitude = 20 * np.log10(np.abs(transfer) + 1e-15); values.append(amplitude - np.median(amplitude))
    return np.asarray(values, dtype=np.float32)


def source_shape(source: np.ndarray) -> np.ndarray:
    return response_shape(source)[0]


def artifact_path(condition: str, train_id: str) -> Path:
    return V5_OUTPUT / "predictions" / condition / "v4_independent" / f"{train_id}.npz"


def features_from_artifact(path: Path) -> tuple[np.ndarray, np.ndarray]:
    item = np.load(path); source = source_shape(item["source"])
    x = []
    for event, channel in enumerate(item["channels"]):
        amplitude, phase = response_shape(channel)
        x.append(np.concatenate((amplitude, phase, source, np.asarray([event / 9], dtype=np.float32))))
    return np.asarray(x, dtype=np.float32), np.asarray(item["channels"], dtype=np.float32)


class ResidualRefiner(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_count, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.05),
                                 nn.Linear(192, 128), nn.GELU(), nn.Linear(128, len(FREQ)))

    def forward(self, values: torch.Tensor) -> torch.Tensor: return self.net(values)


def make_training_data(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = group_split(discover_inputs()); ids = [row["train_id"] for row in rows if row["split"] == split]
    features: list[np.ndarray] = []; targets: list[np.ndarray] = []; owners: list[str] = []
    for train_id in ids:
        x, channels = features_from_artifact(V4_ROOT / "outputs" / "independent" / f"{train_id}.npz")
        baseline = np.stack([response_shape(channel)[0] for channel in channels])
        features.append(x); targets.append(truth_shape(DATA_ROOT / train_id) - baseline); owners.extend([train_id] * len(x))
    return np.concatenate(features), np.concatenate(targets), owners


def train(config: Config, device: torch.device) -> None:
    train_x, train_y, _ = make_training_data("train"); val_x, val_y, _ = make_training_data("validation")
    mean = train_x.mean(0); std = train_x.std(0).clip(1e-5)
    atomic_json(OUT / "normalisation.json", {"mean": mean.tolist(), "std": std.tolist(), "feature_count": int(len(mean))})
    train_set = TensorDataset(torch.from_numpy((train_x - mean) / std), torch.from_numpy(train_y))
    val_x_t = torch.from_numpy((val_x - mean) / std).to(device); val_y_t = torch.from_numpy(val_y).to(device)
    history: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(config.seeds, 1):
        seed_all(seed); model = ResidualRefiner(train_x.shape[1]).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
        loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True); best = float("inf"); stale = 0
        checkpoint = OUT / "checkpoints" / f"refiner_seed{seed}.pt"
        for epoch in range(1, config.epochs + 1):
            model.train(); total = 0.0; count = 0
            for features, target in loader:
                features, target = features.to(device), target.to(device); prediction = model(features)
                loss = torch.nn.functional.smooth_l1_loss(prediction, target, beta=1.0)
                optimiser.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
                total += float(loss.detach()) * len(features); count += len(features)
            model.eval()
            with torch.inference_mode(): val_loss = float(torch.nn.functional.smooth_l1_loss(model(val_x_t), val_y_t, beta=1.0))
            history.append({"seed": seed, "epoch": epoch, "train_huber": total / count, "validation_huber": val_loss})
            if val_loss < best - 1e-6:
                best = val_loss; stale = 0; checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": epoch, "validation_huber": best,
                            "feature_count": train_x.shape[1], "config": asdict(config)}, checkpoint)
            else: stale += 1
            if epoch == 1 or epoch % 25 == 0 or stale >= config.patience:
                print(f"[总体进度：v6训练] seed {seed_index}/{len(config.seeds)} epoch {epoch}/{config.epochs} val={val_loss:.5f}", flush=True)
            if stale >= config.patience: break
        write_csv(OUT / "training_history.csv", history)


def infer_and_evaluate(device: torch.device) -> list[dict[str, Any]]:
    normalisation = json.loads((OUT / "normalisation.json").read_text(encoding="utf-8")); mean = np.asarray(normalisation["mean"], dtype=np.float32); std = np.asarray(normalisation["std"], dtype=np.float32)
    models = []
    for path in sorted((OUT / "checkpoints").glob("refiner_seed*.pt")):
        saved = torch.load(path, map_location=device, weights_only=True); model = ResidualRefiner(int(saved["feature_count"])).to(device)
        model.load_state_dict(saved["state_dict"]); model.eval(); models.append((path.stem, model, saved))
    if len(models) != 3: raise RuntimeError(f"expected 3 refiner checkpoints, found {len(models)}")
    test_ids = [row["train_id"] for row in group_split(discover_inputs()) if row["split"] == "test"]
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for index, train_id in enumerate(test_ids, 1):
            features, channels = features_from_artifact(artifact_path(condition, train_id)); baseline = np.stack([response_shape(channel)[0] for channel in channels])
            target = truth_shape(DATA_ROOT / train_id)
            values = torch.from_numpy((features - mean) / std).to(device)
            predictions = []
            with torch.inference_mode():
                for _, model, _ in models: predictions.append(model(values).cpu().numpy())
            for name, prediction, saved in [(name, prediction, saved) for (name, _, saved), prediction in zip(models, predictions)]:
                shape = baseline + prediction; shape -= np.median(shape, axis=1, keepdims=True)
                errors = np.sqrt(np.mean((shape - target) ** 2, axis=1))
                for event, error in enumerate(errors, 1): rows.append({"condition": condition, "train_id": train_id, "event_index": event, "method": name, "channel_amplitude_rmse_db": float(error)})
            mean_shape = baseline + np.mean(predictions, axis=0); mean_shape -= np.median(mean_shape, axis=1, keepdims=True)
            mean_error = np.sqrt(np.mean((mean_shape - target) ** 2, axis=1))
            baseline_error = np.sqrt(np.mean((baseline - target) ** 2, axis=1))
            for event, (neural, classical) in enumerate(zip(mean_error, baseline_error), 1):
                rows.append({"condition": condition, "train_id": train_id, "event_index": event, "method": "neural_ensemble", "channel_amplitude_rmse_db": float(neural)})
                rows.append({"condition": condition, "train_id": train_id, "event_index": event, "method": "v4_independent", "channel_amplitude_rmse_db": float(classical)})
            print(f"[总体进度：v6评估] {condition} {index}/{len(test_ids)}", flush=True)
    return rows


def bootstrap(values: np.ndarray, token: str) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % 2**32; rng = np.random.default_rng(seed)
    points = [np.median(rng.choice(values, len(values), replace=True)) for _ in range(2000)]
    return float(np.quantile(points, .025)), float(np.quantile(points, .975))


def report(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows); per_train = frame.groupby(["condition", "method", "train_id"], as_index=False).median(numeric_only=True)
    summary = []; paired = []
    for condition in CONDITIONS:
        for method in sorted(per_train.method.unique()):
            values = per_train[(per_train.condition == condition) & (per_train.method == method)].channel_amplitude_rmse_db.to_numpy()
            low, high = bootstrap(values, f"summary|{condition}|{method}")
            summary.append({"condition": condition, "method": method, "n_clicktrains": len(values), "median_rmse_db": float(np.median(values)), "ci_low": low, "ci_high": high})
        pivot = per_train[per_train.condition == condition].pivot(index="train_id", columns="method", values="channel_amplitude_rmse_db")
        delta = (pivot["neural_ensemble"] - pivot["v4_independent"]).to_numpy(); low, high = bootstrap(delta, f"paired|{condition}")
        paired.append({"condition": condition, "comparison": "neural_ensemble_minus_v4", "n_clicktrains": len(delta), "median_delta_rmse_db": float(np.median(delta)), "ci_low": low, "ci_high": high})
    write_csv(OUT / "event_metrics.csv", rows); write_csv(OUT / "summary_metrics.csv", summary); write_csv(OUT / "paired_deltas.csv", paired)
    lines = ["# 第20实验 v6：BCSC引导的神经频响残差精修", "", "## 协议", "",
             "网络以盲 BCSC 的事件频响幅度、相位残差、共享源频响和事件序号为输入，预测35个频率点的幅度形状残差。"
             "源波形、路径候选和时域重构仍保留 BCSC 输出，因此该实验只检验神经精修对信道幅度形状的增益。",
             "训练与验证使用48/10个来源组的合成真值；测试使用11个未见来源组、16个 clicktrain，且测试特征只来自盲 BCSC 工件。", "",
             "## 主结果：信道幅度RMSE", "", "| 条件 | 方法 | 中位数 (dB) | 95% bootstrap CI |", "| --- | --- | ---: | --- |"]
    labels = {"clean": "无噪声", "snr_m05": "−5 dB", "snr_p05": "5 dB"}
    for item in summary:
        lines.append(f"| {labels[item['condition']]} | {item['method']} | {item['median_rmse_db']:.4f} | [{item['ci_low']:.4f}, {item['ci_high']:.4f}] |")
    lines += ["", "## 配对比较：神经集成 − v4", "", "| 条件 | 中位数差 (dB) | 95% bootstrap CI |", "| --- | ---: | --- |"]
    for item in paired: lines.append(f"| {labels[item['condition']]} | {item['median_delta_rmse_db']:.4f} | [{item['ci_low']:.4f}, {item['ci_high']:.4f}] |")
    lines += ["", "若配对区间的上界小于0，则该条件下神经精修相对 BCSC 的信道幅度RMSE有可验证降低；否则仅报告点估计，不宣称稳定改进。",
             "本实验不以频响精修结果声称源恢复或路径F1优于 BCSC，因为这两项保持 BCSC 原输出。"]
    (OUT / "experiment_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--epochs", type=int, default=450); parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda")); args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    config = Config(epochs=args.epochs); atomic_json(OUT / "config.json", {**asdict(config), "device": str(device)})
    train(config, device); rows = infer_and_evaluate(device); report(rows); print(f"v6完成：{OUT / 'experiment_report_zh.md'}", flush=True)


if __name__ == "__main__": main()
