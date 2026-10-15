#!/usr/bin/env python3
"""Train three clean-data deep-unfolded BCSC models on the locked source-group split."""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from protocol_v5 import OUTPUT_ROOT, config_hash, discover_inputs, group_split, write_csv, atomic_json
from training_data_v5 import ClicktrainDataset, FREQUENCIES_HZ
from unrolled_bcsc import UnrolledBCSC


@dataclass(frozen=True)
class TrainConfig:
    stages: int = 6
    epochs: int = 30
    batch_size: int = 2
    learning_rate: float = 3e-3
    reconstruction_weight: float = 0.10
    sparsity_weight: float = 1e-4
    seeds: tuple[int, ...] = (20260908, 20260909, 20260910)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transfer_prediction(channels: torch.Tensor) -> torch.Tensor:
    """Interpolate predicted channel spectra to the v4 20--190 kHz analysis grid."""
    n_fft = 16_384
    spectrum = torch.fft.rfft(channels, n=n_fft)
    bins = torch.as_tensor(FREQUENCIES_HZ / (576_000 / n_fft), device=channels.device, dtype=channels.dtype)
    low = torch.floor(bins).to(torch.long); high = low + 1
    weight = (bins - low.to(channels.dtype))[None, None]
    sampled = spectrum[..., low] * (1 - weight) + spectrum[..., high] * weight
    return sampled / sampled.abs().median(dim=-1, keepdim=True).values.clamp_min(1e-8)


def losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor | list[str]], config: TrainConfig) -> dict[str, torch.Tensor]:
    source = outputs["source"]
    target_source = batch["source"]  # type: ignore[assignment]
    transfer = batch["transfer"]  # type: ignore[assignment]
    segments = batch["segments"]  # type: ignore[assignment]
    source_loss = 1 - F.cosine_similarity(source, target_source, dim=-1).abs().mean()
    transfer_loss = torch.mean(torch.abs(transfer_prediction(outputs["channels"]) - transfer) ** 2)
    segment_scale = torch.linalg.vector_norm(segments, dim=-1, keepdim=True).clamp_min(1e-8)
    reconstruction_loss = torch.mean(((outputs["reconstructions"] - segments) / segment_scale) ** 2)
    sparsity_loss = torch.mean(torch.abs(outputs["channels"]) / segment_scale)
    total = source_loss + transfer_loss + config.reconstruction_weight * reconstruction_loss + config.sparsity_weight * sparsity_loss
    return {"total": total, "source": source_loss, "transfer": transfer_loss,
            "reconstruction": reconstruction_loss, "sparsity": sparsity_loss}


def run_epoch(model: UnrolledBCSC, loader: DataLoader, device: torch.device, config: TrainConfig,
              optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    totals: dict[str, float] = {}; count = 0
    model.train(optimizer is not None)
    for raw_batch in loader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw_batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            values = losses(model(batch["segments"]), batch, config)  # type: ignore[arg-type]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); values["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        count += 1
        for key, value in values.items(): totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260908, 20260909, 20260910])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="continue a compatible incomplete seed from its best checkpoint")
    args = parser.parse_args()
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seeds=tuple(args.seeds))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    paths = discover_inputs(); split_rows = group_split(paths); write_csv(OUTPUT_ROOT / "group_split.csv", split_rows)
    split_ids = {name: [row["train_id"] for row in split_rows if row["split"] == name] for name in ("train", "validation", "test")}
    train_loader = DataLoader(ClicktrainDataset(split_ids["train"]), batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(ClicktrainDataset(split_ids["validation"]), batch_size=config.batch_size, shuffle=False, num_workers=0)
    run_config = {**asdict(config), "device": str(device), "split_seed": 20260908,
                  "split_counts": {name: len(ids) for name, ids in split_ids.items()}}
    run_config["hash"] = config_hash(run_config); atomic_json(OUTPUT_ROOT / "train_config.json", run_config)
    history_path = OUTPUT_ROOT / "training_history.csv"
    history_rows: list[dict[str, float | int]] = []
    if history_path.is_file():
        import csv
        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            history_rows = list(csv.DictReader(handle))
    for seed_index, seed in enumerate(config.seeds, 1):
        seed_everything(seed); model = UnrolledBCSC(config.stages, learnable=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        best = float("inf"); start_epoch = 1; checkpoint = OUTPUT_ROOT / "checkpoints" / f"learned_unroll_seed{seed}.pt"
        expected = asdict(config)
        if args.resume and checkpoint.is_file():
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            saved_config = saved.get("config", {})
            compatible = all(saved_config.get(key) == expected[key] for key in (
                "stages", "epochs", "batch_size", "learning_rate", "reconstruction_weight", "sparsity_weight"))
            if compatible:
                model.load_state_dict(saved["state_dict"]); best = float(saved["validation_loss"])
                start_epoch = int(saved["epoch"]) + 1
                if start_epoch > config.epochs:
                    print(f"[总体进度：训练] seed {seed_index}/{len(config.seeds)} 已由兼容检查点完成", flush=True)
                    continue
                print(f"[总体进度：训练] seed {seed_index}/{len(config.seeds)} 从 epoch {start_epoch}/{config.epochs} 续训", flush=True)
            else:
                history_rows = [row for row in history_rows if int(row["seed"]) != seed]
        started = time.monotonic()
        for epoch in range(start_epoch, config.epochs + 1):
            train_values = run_epoch(model, train_loader, device, config, optimizer)
            validation_values = run_epoch(model, validation_loader, device, config, None)
            row: dict[str, float | int] = {"seed": seed, "epoch": epoch, **{f"train_{key}": value for key, value in train_values.items()},
                                            **{f"validation_{key}": value for key, value in validation_values.items()}}
            history_rows.append(row); write_csv(history_path, history_rows)
            if validation_values["total"] < best:
                best = validation_values["total"]; checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "seed": seed, "epoch": epoch, "validation_loss": best,
                            "config": asdict(config)}, checkpoint)
            elapsed = time.monotonic() - started
            print(f"[总体进度：训练] seed {seed_index}/{len(config.seeds)} | epoch {epoch:02d}/{config.epochs} | "
                  f"val={validation_values['total']:.5f} | elapsed={elapsed:.1f}s", flush=True)
    print(f"训练完成：{OUTPUT_ROOT / 'checkpoints'}", flush=True)


if __name__ == "__main__":
    main()
