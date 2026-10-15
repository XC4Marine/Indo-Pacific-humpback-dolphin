#!/usr/bin/env python3
"""Truth-free v7 inference on the locked clean test split."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import correlate, fftconvolve

HERE = Path(__file__).resolve().parent; ROOT20 = HERE.parent; V4 = ROOT20 / "experiment_v4"; V5 = ROOT20 / "experiment_v5"
sys.path[:0] = [str(V5), str(V4)]
from protocol_v5 import discover_inputs, group_split  # noqa: E402
from bcsc_v4 import bandpass  # noqa: E402
from model_v7 import TimeDomainRefiner, causal_convolution, unit_norm  # noqa: E402
from truth_guard_v7 import assert_inference_truth_isolation  # noqa: E402

OUT = HERE / "outputs"; SEGMENT_LENGTH = 2419


def observed_segments(raw: np.ndarray, starts: np.ndarray) -> np.ndarray:
    filtered = bandpass(raw.astype(float)); result = []
    for start in starts:
        values = filtered[int(start):int(start) + SEGMENT_LENGTH]
        result.append(np.pad(values, (0, max(0, SEGMENT_LENGTH - len(values))))[:SEGMENT_LENGTH])
    return np.asarray(result, dtype=np.float32)


def load_models() -> list[TimeDomainRefiner]:
    paths = sorted((OUT / "checkpoints").glob("time_refiner_seed*.pt"))
    if not paths: raise RuntimeError("no v7 checkpoints found")
    models = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True); model = TimeDomainRefiner(); model.load_state_dict(payload["state_dict"]); model.eval(); models.append(model)
    return models


def infer_one(raw: np.ndarray, baseline: np.lib.npyio.NpzFile, models: list[TimeDomainRefiner]) -> dict[str, np.ndarray]:
    source = np.asarray(baseline["source"], dtype=np.float32); channels = np.asarray(baseline["channels"], dtype=np.float32); starts = np.asarray(baseline["detected_window_starts"], dtype=int)
    observed = observed_segments(raw, starts); proxy = np.stack([correlate(y, source, mode="valid") for y in observed]).astype(np.float32)
    results = []
    with torch.inference_mode():
        for model in models:
            item = model(torch.from_numpy(source[None]), torch.from_numpy(channels[None]), torch.from_numpy(proxy[None]))
            results.append((item["source"][0], item["channels"][0]))
    # The trained source head is retained for its supervised loss, but the final
    # reconstruction uses the canonical BCSC source.  This lets the analytic
    # data-consistency layer make a monotone per-event improvement relative to
    # the BCSC reconstruction instead of creating a new scale/shift ambiguity.
    _ = unit_norm(torch.stack([item[0] for item in results]).mean(0, keepdim=True))[0]
    proposal = torch.stack([item[1] for item in results]).mean(0).numpy()
    refined, reconstruction, coefficients = [], [], []
    for event in range(channels.shape[0]):
        base_reconstruction = np.asarray(baseline["reconstructions"][event], dtype=float)
        proposed_reconstruction = fftconvolve(source, proposal[event], mode="full")[:SEGMENT_LENGTH]
        direction = proposed_reconstruction - base_reconstruction
        alpha = float(np.clip(np.dot(observed[event] - base_reconstruction, direction) / (np.dot(direction, direction) + 1e-18), 0.0, 1.0))
        refined.append(channels[event] + alpha * (proposal[event] - channels[event]))
        reconstruction.append(base_reconstruction + alpha * direction); coefficients.append(alpha)
    return {"source": source, "channels": np.asarray(refined, dtype=np.float32), "reconstructions": np.asarray(reconstruction, dtype=np.float32), "detected_window_starts": starts, "data_consistency_alpha": np.asarray(coefficients, dtype=np.float32)}


def main() -> None:
    assert_inference_truth_isolation(HERE); models = load_models(); rows = [row for row in group_split(discover_inputs()) if row["split"] == "test"]
    for index, row in enumerate(rows, 1):
        train_id = str(row["train_id"]); raw_path = next(path for path in discover_inputs() if path.parent.name == train_id)
        baseline = np.load(V5 / "outputs" / "predictions" / "clean" / "v4_independent" / f"{train_id}.npz")
        output = OUT / "blind_predictions" / "clean" / f"{train_id}.npz"; output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **infer_one(np.load(raw_path, allow_pickle=False), baseline, models))
        print(f"[总体进度：v7 盲推理] {index}/{len(rows)} | {train_id}", flush=True)
    print(f"v7 盲推理完成：{OUT / 'blind_predictions'}", flush=True)


if __name__ == "__main__": main()
