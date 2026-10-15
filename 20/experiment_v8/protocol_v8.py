"""Truth-free data, split, noise, and artifact utilities for Experiment v8."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "experiment_v2" / "clicktrains_10click_perturbed" / "clicktrains"
OUTPUT_ROOT = ROOT / "outputs"
FS = 576_000
EVENT_COUNT = 10
CHANNEL_SAMPLES = int(0.004 * FS)
SOURCE_SAMPLES = 116
SEGMENT_SAMPLES = SOURCE_SAMPLES + CHANNEL_SAMPLES - 1
FOLD_SEED = 20260910
CONDITIONS: tuple[int | None, ...] = (None, 5, -5)
GROUP_RE = re.compile(r"^click_\d+_train_(\d+)_pulse_\d+$")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def discover_inputs() -> list[Path]:
    paths = sorted(DATA_ROOT.glob("*/clicktrain_raw.npy"))
    if len(paths) != 110:
        raise RuntimeError(f"expected 110 clicktrains, found {len(paths)}")
    return paths


def source_group(path: Path) -> str:
    match = GROUP_RE.match(path.parent.name)
    if not match:
        raise ValueError(f"invalid clicktrain directory name: {path.parent.name}")
    return f"train_{int(match.group(1)):03d}"


def fold_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    paths = list(paths)
    groups = sorted({source_group(path) for path in paths})
    if len(groups) != 69:
        raise RuntimeError(f"expected 69 source groups, found {len(groups)}")
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{FOLD_SEED}:{group}".encode()).hexdigest())
    assignment = {group: index % 5 for index, group in enumerate(ranked)}
    rows = [{"train_id": path.parent.name, "source_group": source_group(path), "fold": assignment[source_group(path)]}
            for path in paths]
    if {row["fold"] for row in rows} != set(range(5)):
        raise RuntimeError("five non-empty OOF folds were not generated")
    return rows


def condition_name(snr_db: int | None) -> str:
    return "clean" if snr_db is None else f"snr_{snr_db:+03d}".replace("+", "p").replace("-", "m")


def parse_conditions(value: str) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for token in value.split(","):
        token = token.strip().lower()
        item = None if token == "clean" else int(token)
        if item not in CONDITIONS:
            raise ValueError(f"unsupported condition {token!r}; choose clean, 5, or -5")
        if item not in result:
            result.append(item)
    return tuple(result)


def add_awgn(clicktrain: np.ndarray, train_id: str, snr_db: int | None) -> np.ndarray:
    signal = np.asarray(clicktrain, dtype=np.float64)
    if snr_db is None:
        return signal.copy()
    seed = int.from_bytes(hashlib.sha256(f"v8-awgn:{train_id}:{snr_db}".encode()).digest()[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=len(signal))
    power = float(np.mean(signal * signal))
    noise *= np.sqrt(power / (10 ** (snr_db / 10)) / (float(np.mean(noise * noise)) + 1e-18))
    return signal + noise


def prediction_path(condition: str, method: str, train_id: str) -> Path:
    return OUTPUT_ROOT / "predictions" / condition / method / f"{train_id}.npz"


def save_prediction(path: Path, *, event_delays: list[list[float]], stable_delays: list[float], mask: np.ndarray,
                    reconstruction_nrmse: float, runtime_ms: float, fold: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([len(values) for values in event_delays], dtype=np.int32)
    flattened = np.asarray([item for values in event_delays for item in values], dtype=np.float64)
    np.savez_compressed(path, event_delay_ms=flattened, event_counts=counts,
                        stable_delay_ms=np.asarray(stable_delays, dtype=np.float64), mask=np.asarray(mask, dtype=np.uint8),
                        reconstruction_nrmse=float(reconstruction_nrmse), runtime_ms=float(runtime_ms), fold=int(fold))


def load_prediction(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as item:
        flat = np.asarray(item["event_delay_ms"], dtype=float); counts = np.asarray(item["event_counts"], dtype=int)
        if len(counts) != EVENT_COUNT or int(np.sum(counts)) != len(flat):
            raise ValueError(f"invalid event candidate schema: {path}")
        cursor = 0; event_delays: list[list[float]] = []
        for count in counts:
            event_delays.append(flat[cursor:cursor + int(count)].tolist()); cursor += int(count)
        return {"event_delays": event_delays, "stable_delays": np.asarray(item["stable_delay_ms"], dtype=float).tolist(),
                "mask": np.asarray(item["mask"], dtype=np.uint8), "reconstruction_nrmse": float(item["reconstruction_nrmse"]),
                "runtime_ms": float(item["runtime_ms"]), "fold": int(item["fold"])}
