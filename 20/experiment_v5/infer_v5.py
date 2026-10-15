#!/usr/bin/env python3
"""Truth-free v5 extraction for the locked test split and three AWGN conditions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from protocol_v5 import OUTPUT_ROOT, discover_inputs, group_split, write_csv
from truth_guard_v5 import assert_inference_truth_isolation
from unrolled_bcsc import UnrolledBCSC, causal_fft_convolution

V4_ROOT = Path(__file__).resolve().parent.parent / "experiment_v4"
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import FitConfig, canonicalise, detect_event_segments, fit_clicktrain  # noqa: E402


CONDITIONS: tuple[int | None, ...] = (None, -5, 5)


def condition_name(snr_db: int | None) -> str:
    return "clean" if snr_db is None else f"snr_{snr_db:+03d}".replace("+", "p").replace("-", "m")


def add_awgn(clicktrain: np.ndarray, train_id: str, snr_db: int | None) -> np.ndarray:
    if snr_db is None:
        return np.asarray(clicktrain, dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(f"v4-awgn:{train_id}:{snr_db}".encode()).digest()[:8], "little") % (2**32)
    rng = np.random.default_rng(seed); signal = np.asarray(clicktrain, dtype=np.float64)
    power = float(np.mean(signal * signal)); noise = rng.normal(size=len(signal))
    noise *= np.sqrt(power / (10 ** (snr_db / 10)) / (np.mean(noise * noise) + 1e-18))
    return signal + noise


def constrained_result(source: np.ndarray, channels: np.ndarray) -> dict[str, np.ndarray]:
    source, channels = canonicalise(np.asarray(source, dtype=np.float64), np.asarray(channels, dtype=np.float64))
    source_t = torch.from_numpy(source[None].astype(np.float32)); channel_t = torch.from_numpy(channels[None].astype(np.float32))
    reconstruction = causal_fft_convolution(source_t, channel_t).squeeze(0).numpy()
    return {"source": source, "channels": channels, "reconstructions": reconstruction}


def model_result(model: UnrolledBCSC, raw: np.ndarray, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    segments, starts = detect_event_segments(raw)
    with torch.inference_mode():
        values = model(torch.from_numpy(segments[None].astype(np.float32)).to(device))
    return constrained_result(values["source"][0].cpu().numpy(), values["channels"][0].cpu().numpy()), starts


def v4_config() -> FitConfig:
    selected = V4_ROOT / "outputs" / "calibration" / "independent.json"
    if selected.is_file():
        item = json.loads(selected.read_text(encoding="utf-8"))["selected"]
        return FitConfig(lambda_ratio=float(item["lambda_ratio"]), group_ratio=0.0)
    return FitConfig()


def save_artifact(path: Path, result: dict[str, np.ndarray], starts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result, detected_window_starts=starts)


def load_learned_models(device: torch.device) -> list[tuple[str, UnrolledBCSC]]:
    checkpoints = sorted((OUTPUT_ROOT / "checkpoints").glob("learned_unroll_seed*.pt"))
    if len(checkpoints) != 3:
        raise RuntimeError(f"expected exactly 3 learned checkpoints, found {len(checkpoints)}")
    loaded: list[tuple[str, UnrolledBCSC]] = []
    for path in checkpoints:
        saved = torch.load(path, map_location=device, weights_only=True)
        model = UnrolledBCSC(int(saved["config"]["stages"]), learnable=True).to(device)
        model.load_state_dict(saved["state_dict"]); model.eval()
        loaded.append((path.stem.replace("learned_", ""), model))
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="limit test clicktrains for a smoke run")
    parser.add_argument("--include-shared-support", action="store_true")
    args = parser.parse_args()
    assert_inference_truth_isolation(Path(__file__).resolve().parent)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    test_paths = [path for path, row in zip(discover_inputs(), group_split(discover_inputs())) if row["split"] == "test"]
    if args.limit is not None: test_paths = test_paths[:args.limit]
    learned = load_learned_models(device); fixed = UnrolledBCSC(6, learnable=False).to(device).eval()
    methods: list[str] = ["v4_independent", "fixed_unroll", *[name for name, _ in learned]]
    if args.include_shared_support: methods.append("v4_shared_support")
    rows = []; total = len(test_paths) * len(CONDITIONS)
    for condition_index, snr_db in enumerate(CONDITIONS, 1):
        name = condition_name(snr_db)
        for item_index, path in enumerate(test_paths, 1):
            train_id = path.parent.name; raw = add_awgn(np.load(path, allow_pickle=False), train_id, snr_db)
            for method in methods:
                artifact = OUTPUT_ROOT / "predictions" / name / method / f"{train_id}.npz"
                if artifact.exists():
                    rows.append({"train_id": train_id, "condition": name, "snr_db": "clean" if snr_db is None else snr_db,
                                 "method": method, "status": "resumed", "artifact": str(artifact)})
                    continue
                try:
                    if method == "v4_independent":
                        temporary = OUTPUT_ROOT / "_blind_input" / "clicktrain_raw.npy"; temporary.parent.mkdir(parents=True, exist_ok=True); np.save(temporary, raw)
                        result = fit_clicktrain(temporary, v4_config()); temporary.unlink(missing_ok=True)
                        values = {key: np.asarray(result[key]) for key in ("source", "channels", "reconstructions")}; starts = result["detected_window_starts"]
                    elif method == "v4_shared_support":
                        temporary = OUTPUT_ROOT / "_blind_input" / "clicktrain_raw.npy"; temporary.parent.mkdir(parents=True, exist_ok=True); np.save(temporary, raw)
                        result = fit_clicktrain(temporary, FitConfig(group_ratio=0.5)); temporary.unlink(missing_ok=True)
                        values = {key: np.asarray(result[key]) for key in ("source", "channels", "reconstructions")}; starts = result["detected_window_starts"]
                    elif method == "fixed_unroll":
                        values, starts = model_result(fixed, raw, device)
                    else:
                        values, starts = model_result(dict(learned)[method], raw, device)
                    save_artifact(artifact, values, starts)
                    rows.append({"train_id": train_id, "condition": name, "snr_db": "clean" if snr_db is None else snr_db,
                                 "method": method, "status": "success", "artifact": str(artifact)})
                except Exception as exc:
                    rows.append({"train_id": train_id, "condition": name, "snr_db": "clean" if snr_db is None else snr_db,
                                 "method": method, "status": "failed", "error": repr(exc)})
            write_csv(OUTPUT_ROOT / "prediction_manifest.csv", rows)
            print(f"[总体进度：盲推理] condition {condition_index}/{len(CONDITIONS)} | {item_index}/{len(test_paths)} | {train_id}", flush=True)
    write_csv(OUTPUT_ROOT / "prediction_manifest.csv", rows)
    print(f"盲推理完成：{OUTPUT_ROOT / 'predictions'}", flush=True)


if __name__ == "__main__":
    main()
