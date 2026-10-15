"""Training-only supervised data access for Experiment 20 v5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from protocol_v5 import DATA_ROOT

V4_ROOT = Path(__file__).resolve().parent.parent / "experiment_v4"
if str(V4_ROOT) not in sys.path:
    sys.path.insert(0, str(V4_ROOT))
from bcsc_v4 import FS, bandpass, canonicalise, detect_event_segments  # noqa: E402

FREQUENCIES_HZ = np.arange(20_000, 190_001, 5_000, dtype=float)


def truth_source(directory: Path) -> np.ndarray:
    _, value = wavfile.read(directory / "original_click.wav")
    source = bandpass(value.astype(np.float64))
    source, _ = canonicalise(source, np.zeros((1, 2304)))
    return source.astype(np.float32)


def transfer_targets(directory: Path) -> np.ndarray:
    with np.load(directory / "event_channels.npz", allow_pickle=True) as content:
        arrivals = content["event_synthesis_arrivals"]
        first_arrival = float(content["first_arrival_s"])
    targets = []
    for event in arrivals:
        values = []
        for frequency in FREQUENCIES_HZ:
            rows = event[np.isclose(event["frequency_hz"], frequency)]
            values.append(np.sum(rows["complex_gain"] * np.exp(-2j * np.pi * frequency * (rows["delay_s"] - first_arrival))))
        values = np.asarray(values, dtype=np.complex64)
        targets.append(values / max(float(np.median(np.abs(values))), 1e-8))
    return np.asarray(targets, dtype=np.complex64)


class ClicktrainDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, train_ids: list[str]) -> None:
        self.directories = [DATA_ROOT / name for name in train_ids]

    def __len__(self) -> int:
        return len(self.directories)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        directory = self.directories[index]
        raw = np.load(directory / "clicktrain_raw.npy", allow_pickle=False)
        segments, _ = detect_event_segments(raw)
        return {"train_id": directory.name, "segments": torch.from_numpy(segments.astype(np.float32)),
                "source": torch.from_numpy(truth_source(directory)),
                "transfer": torch.from_numpy(transfer_targets(directory))}
