#!/usr/bin/env python3
"""Training-only Experiment 20 v7: time-domain BCSC residual refinement.

This entry point is allowed to read synthetic supervision.  It is deliberately
separate from `infer_v7.py`, which consumes only raw clicktrain input and BCSC
artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate

HERE = Path(__file__).resolve().parent; ROOT20 = HERE.parent; V4 = ROOT20 / "experiment_v4"; V5 = ROOT20 / "experiment_v5"; V2 = ROOT20 / "experiment_v2"
sys.path[:0] = [str(V5), str(V4), str(V2)]
from protocol_v5 import DATA_ROOT, discover_inputs, group_split  # noqa: E402
from bcsc_v4 import FS, bandpass, canonicalise  # noqa: E402
from run_experiment_v2 import FREQUENCIES_HZ, subband_synthesis  # noqa: E402
from model_v7 import TimeDomainRefiner, causal_convolution, unit_norm  # noqa: E402

OUT = HERE / "outputs"; SOURCE_LENGTH = 116; CHANNEL_LENGTH = 2304; SEGMENT_LENGTH = 2419


@dataclass(frozen=True)
class Config:
    epochs: int = 700
    lr: float = 3e-4
    patience: int = 80
    seeds: tuple[int, ...] = (20260921, 20260922, 20260923)
    source_weight: float = 1.0
    channel_weight: float = 1.2
    reconstruction_weight: float = 5.0


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def truth_source(directory: Path) -> np.ndarray:
    _, values = wavfile.read(directory / "original_click.wav")
    source, _ = canonicalise(bandpass(values.astype(float)), np.zeros((1, CHANNEL_LENGTH)))
    return source.astype(np.float32)


def shift_right(values: np.ndarray, amount: int) -> np.ndarray:
    result = np.roll(values, amount)
    if amount > 0: result[:amount] = 0
    if amount < 0: result[amount:] = 0
    return result


def effective_channels(directory: Path, starts: np.ndarray, source: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Recover the source-conditioned effective time channel of each bandpassed event.

    The propagation files store only 35 frequency responses, so they cannot by
    themselves define every sample of a 4 ms channel.  This Tikhonov
    deconvolution instead defines the unique, reproducible channel that maps the
    canonical source to the actual preprocessed event window.  It is used only
    as training/evaluation truth, never by inference.
    """
    _ = directory, starts
    autocorrelation = np.zeros(CHANNEL_LENGTH, dtype=float)
    autocorrelation[:len(source)] = np.correlate(source, source, mode="full")[len(source) - 1:]
    autocorrelation[0] += 1e-4 * autocorrelation[0]
    targets = [solve_toeplitz((autocorrelation, autocorrelation), correlate(event, source, mode="valid")) for event in observed]
    return np.asarray(targets, dtype=np.float32)


def sample(train_id: str) -> dict[str, np.ndarray]:
    directory = DATA_ROOT / train_id; baseline = np.load(V4 / "outputs" / "independent" / f"{train_id}.npz")
    source = np.asarray(baseline["source"], dtype=np.float32); channels = np.asarray(baseline["channels"], dtype=np.float32); starts = np.asarray(baseline["detected_window_starts"], dtype=int)
    raw = bandpass(np.load(directory / "clicktrain_raw.npy", allow_pickle=False).astype(float))
    observed = np.stack([np.pad(raw[int(start):int(start) + SEGMENT_LENGTH], (0, max(0, int(start) + SEGMENT_LENGTH - len(raw))))[:SEGMENT_LENGTH] for start in starts]).astype(np.float32)
    proxy = np.stack([correlate(y, source, mode="valid") for y in observed]).astype(np.float32)
    target_source = truth_source(directory); target_channels = effective_channels(directory, starts, target_source, observed)
    return {"source": source, "channels": channels, "proxy": proxy, "observed": observed, "target_source": target_source, "target_channels": target_channels}


def pack(ids: list[str], name: str) -> dict[str, torch.Tensor]:
    records = []
    for index, train_id in enumerate(ids, 1):
        records.append(sample(train_id))
        if index == 1 or index % 10 == 0 or index == len(ids):
            print(f"[总体进度：v7 数据准备] {name} {index}/{len(ids)}", flush=True)
    return {key: torch.from_numpy(np.stack([record[key] for record in records])) for key in records[0]}


def rel_mse(estimate: torch.Tensor, target: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return (((estimate - target).square().sum(dim=dims) / target.square().sum(dim=dims).clamp_min(1e-12)).mean())


def losses(model: TimeDomainRefiner, batch: dict[str, torch.Tensor], config: Config) -> tuple[torch.Tensor, dict[str, float]]:
    result = model(batch["source"], batch["channels"], batch["proxy"])
    source_loss = rel_mse(result["source"], batch["target_source"], (1,))
    channel_loss = rel_mse(result["channels"], batch["target_channels"], (2,))
    reconstruction_loss = rel_mse(result["reconstructions"], batch["observed"], (2,))
    total = config.source_weight * source_loss + config.channel_weight * channel_loss + config.reconstruction_weight * reconstruction_loss
    return total, {"total": float(total.detach()), "source": float(source_loss.detach()), "channel": float(channel_loss.detach()), "reconstruction": float(reconstruction_loss.detach())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=500); parser.add_argument("--seeds", type=int, nargs="*", default=[20260921, 20260922, 20260923]); args = parser.parse_args()
    config = Config(epochs=args.epochs, seeds=tuple(args.seeds)); rows = group_split(discover_inputs()); ids = {name: [str(row["train_id"]) for row in rows if row["split"] == name] for name in ("train", "validation")}
    print("[总体进度：v7 数据准备] 0/2 train", flush=True); train = pack(ids["train"], "train"); print("[总体进度：v7 数据准备] 1/2 validation", flush=True); validation = pack(ids["validation"], "validation"); print("[总体进度：v7 数据准备] 2/2 complete", flush=True)
    atomic_json(OUT / "config.json", asdict(config)); history: list[dict[str, object]] = []
    for index, seed in enumerate(config.seeds, 1):
        seed_all(seed); model = TimeDomainRefiner().cpu(); optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
        model.eval()
        with torch.no_grad(): _, initial = losses(model, validation, config)
        best, best_epoch, stale = initial["total"], 0, 0
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f"[总体进度：v7 训练] seed {index}/{len(config.seeds)} | epoch 0/{config.epochs} | baseline-val={best:.4f}", flush=True)
        for epoch in range(1, config.epochs + 1):
            model.train(); optimiser.zero_grad(); value, metric = losses(model, train, config); value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); optimiser.step()
            model.eval()
            with torch.no_grad(): _, val = losses(model, validation, config)
            history.append({"seed": seed, "epoch": epoch, **{f"train_{k}": v for k, v in metric.items()}, **{f"val_{k}": v for k, v in val.items()}})
            if val["total"] < best - 1e-6:
                best, best_epoch, stale = val["total"], epoch, 0; best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else: stale += 1
            if epoch == 1 or epoch % 25 == 0 or stale >= config.patience:
                print(f"[总体进度：v7 训练] seed {index}/{len(config.seeds)} | epoch {epoch}/{config.epochs} | val={val['total']:.4f} | best={best:.4f}", flush=True)
            if stale >= config.patience: break
        path = OUT / "checkpoints" / f"time_refiner_seed{seed}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"seed": seed, "epoch": best_epoch, "validation_total": best, "config": asdict(config), "state_dict": best_state}, path)
        print(f"[总体进度：v7 训练] seed {index}/{len(config.seeds)} complete; best epoch={best_epoch}", flush=True)
    write_csv(OUT / "training_history.csv", history)


if __name__ == "__main__": main()
