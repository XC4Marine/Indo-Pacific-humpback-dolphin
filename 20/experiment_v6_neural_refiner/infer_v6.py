#!/usr/bin/env python3
"""Truth-free v6 inference: refine blind BCSC channel-response shapes only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent; V5 = HERE.parent / "experiment_v5"
sys.path.insert(0, str(V5))
from protocol_v5 import OUTPUT_ROOT as V5_OUTPUT, discover_inputs, group_split  # noqa: E402

OUT = HERE / "outputs"; FREQ = np.arange(20_000, 190_001, 5_000, dtype=float)
CONDITIONS = ("clean", "snr_m05", "snr_p05")


def response_shape(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(np.asarray(channel, dtype=float), n=16_384); bins = FREQ / (576_000 / 16_384)
    low = np.floor(bins).astype(int); weight = bins - low; values = spectrum[low] * (1 - weight) + spectrum[low + 1] * weight
    amplitude = 20 * np.log10(np.abs(values) + 1e-15); amplitude -= np.median(amplitude)
    phase = np.unwrap(np.angle(values)); slope, intercept = np.polyfit(FREQ, phase, 1)
    return amplitude.astype(np.float32), np.angle(np.exp(1j * (phase - slope * FREQ - intercept))).astype(np.float32)


class ResidualRefiner(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(feature_count, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(.05), nn.Linear(192, 128), nn.GELU(), nn.Linear(128, len(FREQ)))
    def forward(self, values: torch.Tensor) -> torch.Tensor: return self.net(values)


def features(item: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    source, _ = response_shape(item["source"]); result = []; base = []
    for event, channel in enumerate(item["channels"]):
        amplitude, phase = response_shape(channel); result.append(np.concatenate((amplitude, phase, source, np.asarray([event / 9], dtype=np.float32)))); base.append(amplitude)
    return np.asarray(result, dtype=np.float32), np.asarray(base, dtype=np.float32)


def assert_truth_isolation() -> None:
    forbidden = ("event_" + "channels.npz", "original_" + "click.wav", "original_" + "channel.npz")
    text = Path(__file__).read_text(encoding="utf-8")
    leaked = [name for name in forbidden if name in text]
    if leaked: raise RuntimeError(f"truth-isolation violation: {leaked}")


def main() -> None:
    assert_truth_isolation(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normal = json.loads((OUT / "normalisation.json").read_text(encoding="utf-8")); mean = np.asarray(normal["mean"], dtype=np.float32); std = np.asarray(normal["std"], dtype=np.float32)
    models = []
    for path in sorted((OUT / "checkpoints").glob("refiner_seed*.pt")):
        saved = torch.load(path, map_location=device, weights_only=True); model = ResidualRefiner(int(saved["feature_count"])).to(device); model.load_state_dict(saved["state_dict"]); model.eval(); models.append(model)
    if len(models) != 3: raise RuntimeError(f"expected three checkpoints, found {len(models)}")
    test_ids = [row["train_id"] for row in group_split(discover_inputs()) if row["split"] == "test"]
    for condition in CONDITIONS:
        for index, train_id in enumerate(test_ids, 1):
            input_path = V5_OUTPUT / "predictions" / condition / "v4_independent" / f"{train_id}.npz"
            with np.load(input_path) as item: x, baseline = features(item); source = item["source"]; channels = item["channels"]
            tensor = torch.from_numpy((x - mean) / std).to(device)
            with torch.inference_mode(): predictions = np.mean([model(tensor).cpu().numpy() for model in models], axis=0)
            corrected = baseline + predictions; corrected -= np.median(corrected, axis=1, keepdims=True)
            destination = OUT / "blind_predictions" / condition / f"{train_id}.npz"; destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, baseline_shape=baseline, corrected_shape=corrected, source=source, channels=channels)
            print(f"[总体进度：v6盲推理] {condition} {index}/{len(test_ids)}", flush=True)


if __name__ == "__main__": main()
