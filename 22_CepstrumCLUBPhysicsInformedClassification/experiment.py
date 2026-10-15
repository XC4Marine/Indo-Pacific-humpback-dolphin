#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Positive-only CLUB physics-informed binary click classification experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt, windows
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm


PROJECT_ROOT = Path(r"D:\Project_Github\Indo-Pacific-humpback-dolphin")
EXPERIMENT_DIR = PROJECT_ROOT / "22_CepstrumCLUBPhysicsInformedClassification"
REFERENCE_MANIFEST = (
    PROJECT_ROOT
    / "21_PhysicsInformedClickClassification"
    / "outputs_snr_m15_p15_full_5seeds"
    / "dataset_manifest.csv"
)
SAMPLE_RATE = 576_000
SAMPLE_COUNT = 5_761
CHANNEL_COUNT = 2_881
EXPECTED_DATA_HASH = "224ec22c07267873dc7a045fd4a339bfc2164f129d144acc702828ac61506615"
BASE_CACHE_DIR = EXPERIMENT_DIR / "outputs"
FULL_OUTPUT_DIR = EXPERIMENT_DIR / "outputs_full_no_cepstrum_positive_mim_5seeds_4conditions"
CONDITIONS = (
    "full",
    "without_recon",
    "without_mim",
    "bce_only",
)
CONDITION_LABELS = {
    "full": "Full",
    "without_recon": "w/o Recon",
    "without_mim": "w/o MIM",
    "bce_only": "BCE-only",
}


@dataclass(frozen=True)
class Config:
    implementation_version: str = "experiment22_lightweight_full_v3"
    project_root: Path = PROJECT_ROOT
    experiment_dir: Path = EXPERIMENT_DIR
    output_dir: Path = EXPERIMENT_DIR / "outputs"
    manifest_source: Path = REFERENCE_MANIFEST
    sample_rate: int = SAMPLE_RATE
    sample_count: int = SAMPLE_COUNT
    channel_count: int = CHANNEL_COUNT
    lowcut_hz: float = 20_000.0
    highcut_hz: float = 190_000.0
    filter_order: int = 5
    tukey_alpha: float = 0.04
    snr_min_db: float = -15.0
    snr_max_db: float = 15.0
    recon_weight: float = 1.0
    bce_weight: float = 1.0
    mim_weight: float = 1.0
    learning_rate: float = 3e-4
    q_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 2
    random_seed: int = 20_260_901
    num_workers: int = 0
    epsilon: float = 1e-8
    device: str = "auto"
    max_epochs: int = 25
    patience: int = 5
    min_delta: float = 1e-4
    bootstrap_runs: int = 2_000

    @property
    def config_hash(self) -> str:
        payload = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
            if key != "output_dir"
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but CUDA is unavailable")
    return device


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def output_paths(config: Config) -> dict[str, Path]:
    paths = {
        "root": config.output_dir,
        "cache": config.output_dir / "cache",
        "models": config.output_dir / "models",
        "logs": config.output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_reference_manifest(config: Config, force: bool = False) -> pd.DataFrame:
    """Copy and validate experiment 21's fixed low-SNR manifest without modifying it."""
    destination = output_paths(config)["root"] / "dataset_manifest.csv"
    if destination.exists() and not force:
        frame = pd.read_csv(destination)
        if (
            len(frame) == 7_000
            and frame["label"].value_counts().to_dict() == {1: 3_500, 0: 3_500}
            and frame.get("experiment22_config_hash", pd.Series(dtype=str))
            .astype(str)
            .eq(config.config_hash)
            .all()
        ):
            return frame
    if not config.manifest_source.exists():
        raise FileNotFoundError(f"Reference manifest not found: {config.manifest_source}")
    frame = pd.read_csv(config.manifest_source)
    required = {
        "sample_id", "label", "ori_file_num", "wav_path", "source_kind", "timestamp_s",
        "test_fold", "mix_noise_wav_path", "mix_noise_timestamp_s", "target_snr_db",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Reference manifest is missing columns: {missing}")
    if len(frame) != 7_000 or frame["label"].value_counts().to_dict() != {1: 3_500, 0: 3_500}:
        raise RuntimeError("Reference manifest must contain 3500 positives and 3500 negatives")
    positives = frame["label"].eq(1)
    if not frame.loc[positives, "target_snr_db"].between(
        config.snr_min_db, config.snr_max_db, inclusive="both"
    ).all():
        raise RuntimeError("Reference target SNR values are outside the configured range")
    if frame.groupby("ori_file_num")["test_fold"].nunique().max() != 1:
        raise RuntimeError("Reference manifest contains recording-source fold leakage")
    frame = frame.copy()
    frame["reference_manifest"] = str(config.manifest_source.resolve())
    frame["experiment22_config_hash"] = config.config_hash
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return frame


def read_centered_waveform(path: Path, timestamp_s: float, sample_count: int) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    center = int(round(timestamp_s * info.samplerate))
    start = center - sample_count // 2
    with sf.SoundFile(str(path)) as handle:
        handle.seek(start)
        signal = handle.read(sample_count, dtype="float64", always_2d=False)
    return np.asarray(signal).squeeze(), int(info.samplerate)


def read_manifest_waveform(row: pd.Series, config: Config) -> np.ndarray:
    path = Path(str(row["wav_path"]))
    if row["source_kind"] == "positive_wav":
        signal, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
        signal = np.asarray(signal).squeeze()
    else:
        signal, sample_rate = read_centered_waveform(path, float(row["timestamp_s"]), config.sample_count)
    if sample_rate != config.sample_rate or signal.shape != (config.sample_count,):
        raise ValueError(f"Invalid waveform: {path}, fs={sample_rate}, shape={signal.shape}")
    return signal


def preprocess_component(signal: np.ndarray, config: Config) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    sos = butter(
        config.filter_order,
        [config.lowcut_hz / (config.sample_rate / 2), config.highcut_hz / (config.sample_rate / 2)],
        btype="bandpass",
        output="sos",
    )
    filtered = sosfiltfilt(sos, signal)
    filtered *= windows.tukey(config.sample_count, alpha=config.tukey_alpha)
    if not np.isfinite(filtered).all():
        raise RuntimeError("Non-finite waveform after preprocessing")
    return filtered


def zero_padded_shift_to_peak(signal: np.ndarray, target_index: int) -> tuple[np.ndarray, int, int]:
    """Translate without circular wrap so the absolute peak lands at target_index."""
    signal = np.asarray(signal)
    original_peak = int(np.argmax(np.abs(signal)))
    shift = int(target_index - original_peak)
    aligned = np.zeros_like(signal)
    if shift >= 0:
        aligned[shift:] = signal[: len(signal) - shift]
    else:
        aligned[:shift] = signal[-shift:]
    aligned_peak = int(np.argmax(np.abs(aligned)))
    if aligned_peak != target_index:
        raise RuntimeError(
            f"Peak alignment failed: original={original_peak}, shift={shift}, aligned={aligned_peak}"
        )
    return aligned, original_peak, shift


def rms(signal: np.ndarray, epsilon: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + epsilon))


def peak_normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if not np.isfinite(peak) or peak < 1e-12:
        raise RuntimeError("Waveform has no finite nonzero amplitude")
    return (signal / peak).astype(np.float32)


def materialize_waveforms(
    manifest: pd.DataFrame, config: Config, force: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    paths = output_paths(config)
    array_path = paths["cache"] / "aligned_mixed_waveforms.npy"
    audit_path = paths["root"] / "waveform_audit.csv"
    if array_path.exists() and audit_path.exists() and not force:
        values = np.load(array_path, mmap_mode="r")
        audit = pd.read_csv(audit_path)
        if values.shape == (7_000, config.sample_count) and len(audit) == 7_000:
            return values, audit

    values = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float32, shape=(len(manifest), config.sample_count)
    )
    rows: list[dict[str, Any]] = []
    iterator = tqdm(
        manifest.iterrows(), total=len(manifest), desc="总体进度 1/4：构建第22个独立波形缓存", unit="条"
    )
    for index, (_, row) in enumerate(iterator):
        label = int(row["label"])
        component = preprocess_component(read_manifest_waveform(row, config), config)
        original_peak = -1
        peak_shift = 0
        aligned_peak = -1
        target_snr = math.nan
        realized_snr = math.nan
        mix_alpha = math.nan
        if label == 1:
            component, original_peak, peak_shift = zero_padded_shift_to_peak(
                component, config.sample_count // 2
            )
            aligned_peak = int(np.argmax(np.abs(component)))
            noise_path = Path(str(row["mix_noise_wav_path"]))
            noise, noise_rate = read_centered_waveform(
                noise_path, float(row["mix_noise_timestamp_s"]), config.sample_count
            )
            if noise_rate != config.sample_rate or noise.shape != (config.sample_count,):
                raise ValueError(f"Invalid paired noise: {noise_path}")
            noise = preprocess_component(noise, config)
            target_snr = float(row["target_snr_db"])
            click_rms = rms(component)
            noise_rms = rms(noise)
            mix_alpha = click_rms / (noise_rms * 10.0 ** (target_snr / 20.0))
            combined = component + mix_alpha * noise
            realized_snr = 20.0 * math.log10(click_rms / (mix_alpha * noise_rms))
        else:
            combined = component
        values[index] = peak_normalize(combined)
        rows.append(
            {
                "sample_index": index,
                "sample_id": row["sample_id"],
                "label": label,
                "original_click_peak_index": original_peak,
                "peak_shift_samples": peak_shift,
                "aligned_click_peak_index": aligned_peak,
                "target_snr_db": target_snr,
                "realized_snr_db": realized_snr,
                "snr_error_db": abs(realized_snr - target_snr) if label == 1 else math.nan,
                "mix_alpha": mix_alpha,
                "output_peak": float(np.max(np.abs(values[index]))),
                "finite": bool(np.isfinite(values[index]).all()),
            }
        )
    values.flush()
    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return np.load(array_path, mmap_mode="r"), audit


def compute_data_hash(manifest: pd.DataFrame, waveforms: np.ndarray, config: Config) -> dict[str, Any]:
    digest = hashlib.sha256()
    columns = [
        "sample_id", "label", "ori_file_num", "test_fold", "wav_path", "timestamp_s",
        "mix_noise_wav_path", "mix_noise_timestamp_s", "target_snr_db",
    ]
    digest.update(manifest[columns].to_csv(index=False).encode())
    digest.update(np.asarray(waveforms).tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "peak_target": config.sample_count // 2,
                "snr": [config.snr_min_db, config.snr_max_db],
                "sample_rate": config.sample_rate,
                "filter": [config.lowcut_hz, config.highcut_hz, config.filter_order],
            },
            sort_keys=True,
        ).encode()
    )
    result = {
        "sha256": digest.hexdigest(),
        "samples": len(manifest),
        "shape": list(waveforms.shape),
        "reference_manifest": str(config.manifest_source.resolve()),
    }
    json_dump(config.output_dir / "data_hash.json", result)
    return result


def validate_data(
    manifest: pd.DataFrame, waveforms: np.ndarray, audit: pd.DataFrame, config: Config
) -> dict[str, Any]:
    positive_audit = audit[audit["label"].eq(1)]
    checks = {
        "sample_count": len(manifest),
        "positive_count": int(manifest["label"].eq(1).sum()),
        "negative_count": int(manifest["label"].eq(0).sum()),
        "waveform_shape": list(waveforms.shape),
        "all_finite": bool(np.isfinite(waveforms).all() and audit["finite"].all()),
        "all_click_peaks_aligned_to_2880": bool(
            positive_audit["aligned_click_peak_index"].eq(config.sample_count // 2).all()
        ),
        "original_click_peaks_at_2880": int(
            positive_audit["original_click_peak_index"].eq(config.sample_count // 2).sum()
        ),
        "shifted_click_count": int(positive_audit["peak_shift_samples"].ne(0).sum()),
        "maximum_snr_error_db": float(positive_audit["snr_error_db"].max()),
        "target_snr_in_bounds": bool(
            positive_audit["target_snr_db"].between(
                config.snr_min_db, config.snr_max_db, inclusive="both"
            ).all()
        ),
        "source_group_leakage_count": int(
            manifest.groupby("ori_file_num")["test_fold"].nunique().ne(1).sum()
        ),
        "fold_values": sorted(manifest["test_fold"].astype(int).unique().tolist()),
    }
    checks["passed"] = bool(
        checks["sample_count"] == 7_000
        and checks["positive_count"] == checks["negative_count"] == 3_500
        and checks["waveform_shape"] == [7_000, config.sample_count]
        and checks["all_finite"]
        and checks["all_click_peaks_aligned_to_2880"]
        and checks["maximum_snr_error_db"] <= 1e-6
        and checks["target_snr_in_bounds"]
        and checks["source_group_leakage_count"] == 0
        and checks["fold_values"] == [0, 1, 2, 3, 4]
    )
    json_dump(config.output_dir / "data_audit.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"Data validation failed: {checks}")
    return checks


class WaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None):
        self.waveforms = waveforms
        self.labels = labels
        self.indices = np.arange(len(labels)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (
            torch.from_numpy(np.asarray(self.waveforms[index], dtype=np.float32).copy()),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class FixedIndexSampler(Sampler[int]):
    def __init__(self, indices: list[int]):
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class ConvBlock(nn.Module):
    """Single lightweight same-padded strided convolution block."""

    def __init__(self, input_channels: int, output_channels: int, stride: int):
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 4
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=9, stride=stride, padding=4),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size, padding=kernel_size // 2,
                      groups=input_channels),
            nn.Conv1d(input_channels, output_channels, 1),
            nn.GroupNorm(8 if output_channels % 8 == 0 else 4, output_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class BranchEncoder(nn.Module):
    def __init__(self, channels: int = 32, latent_count: int = 64):
        super().__init__()
        self.temporal = DepthwiseSeparableConv(channels, channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(channels, latent_count), nn.LayerNorm(latent_count))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal(features)
        latent = self.project(self.pool(temporal))
        return latent, temporal


class ProgressiveDecoder(nn.Module):
    def __init__(self, latent_count: int, seed_channels: int, target_lengths: tuple[int, ...]):
        super().__init__()
        self.seed_channels = seed_channels
        self.target_lengths = target_lengths
        self.seed = nn.Linear(latent_count, seed_channels * 181)
        channels = [seed_channels]
        while len(channels) < len(target_lengths):
            channels.append(max(8, channels[-1] // 2))
        self.blocks = nn.ModuleList(
            DepthwiseSeparableConv(channels[index], channels[index + 1])
            for index in range(len(target_lengths) - 1)
        )
        self.output = nn.Conv1d(channels[-1], 1, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        value = self.seed(latent).view(latent.shape[0], self.seed_channels, 181)
        for length, block in zip(self.target_lengths[1:], self.blocks):
            value = nn.functional.interpolate(value, size=length, mode="linear", align_corners=False)
            value = block(value)
        return self.output(value).squeeze(1)


class ClickDecoder(nn.Module):
    def __init__(self, latent_count: int = 64, sample_count: int = SAMPLE_COUNT):
        super().__init__()
        if sample_count != SAMPLE_COUNT:
            raise ValueError(f"Lightweight decoder requires {SAMPLE_COUNT} samples")
        self.waveform = ProgressiveDecoder(
            latent_count, 32, (181, 361, 721, 1_441, 2_881, 5_761)
        )
        self.classifier = nn.Linear(latent_count, 1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self.waveform(latent)), self.classifier(latent).squeeze(-1)


class CausalChannelDecoder(nn.Module):
    """Strictly causal 0--5 ms decoder with a fixed unit direct path."""

    def __init__(self, latent_count: int = 64, channel_count: int = CHANNEL_COUNT):
        super().__init__()
        self.channel_count = channel_count
        if channel_count != CHANNEL_COUNT:
            raise ValueError(f"Lightweight channel decoder requires {CHANNEL_COUNT} samples")
        self.tail_decoder = ProgressiveDecoder(latent_count, 16, (181, 360, 720, 1_440, 2_880))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tail = 0.25 * torch.tanh(self.tail_decoder(latent))
        direct = torch.ones(latent.shape[0], 1, dtype=latent.dtype, device=latent.device)
        return torch.cat([direct, tail], dim=1)


def causal_fft_convolution(source: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
    output_count = source.shape[-1]
    linear_count = source.shape[-1] + channel.shape[-1] - 1
    fft_count = 1 << (linear_count - 1).bit_length()
    source_fft = torch.fft.rfft(source, n=fft_count)
    channel_fft = torch.fft.rfft(channel, n=fft_count)
    return torch.fft.irfft(source_fft * channel_fft, n=fft_count)[..., :output_count]


class CepstrumCLUBModel(nn.Module):
    def __init__(self, sample_count: int = SAMPLE_COUNT, channel_count: int = CHANNEL_COUNT):
        super().__init__()
        self.shared_frontend = nn.Sequential(
            ConvBlock(1, 16, stride=4),
            ConvBlock(16, 24, stride=4),
            ConvBlock(24, 32, stride=2),
        )
        self.E_click = BranchEncoder()
        self.E_channel = BranchEncoder()
        self.D_click = ClickDecoder(sample_count=sample_count)
        self.D_channel = CausalChannelDecoder(channel_count=channel_count)

    def encode(self, received: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_frontend(received.unsqueeze(1))
        z_click, click_temporal = self.E_click(shared)
        z_channel, channel_temporal = self.E_channel(shared)
        return z_click, z_channel, click_temporal, channel_temporal

    def forward(self, received: torch.Tensor) -> dict[str, torch.Tensor]:
        z_click, z_channel, click_temporal, channel_temporal = self.encode(received)
        x_hat, logit = self.D_click(z_click)
        h_hat = self.D_channel(z_channel)
        r_hat = causal_fft_convolution(x_hat, h_hat)
        return {
            "z_click": z_click,
            "z_channel": z_channel,
            "click_temporal": click_temporal,
            "channel_temporal": channel_temporal,
            "x_hat": x_hat,
            "h_hat": h_hat,
            "r_hat": r_hat,
            "logit": logit,
        }


class CLUBConditional(nn.Module):
    def __init__(self, latent_count: int = 64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(latent_count, 128), nn.GELU())
        self.mu_head = nn.Linear(128, latent_count)
        self.logvar_head = nn.Linear(128, latent_count)

    def forward(self, z_click: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(z_click)
        return self.mu_head(hidden), self.logvar_head(hidden).clamp(-5.0, 5.0)


def reconstruction_loss(target: torch.Tensor, prediction: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(prediction - target, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(epsilon)
    return (numerator / denominator).mean()


def random_derangement(batch_size: int, device: torch.device) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("Derangement requires batch_size >= 2")
    identity = torch.arange(batch_size, device=device)
    while True:
        candidate = torch.randperm(batch_size, device=device)
        if bool(torch.all(candidate != identity)):
            return candidate


def gaussian_log_likelihood(
    target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    return -0.5 * ((target - mu).square() * torch.exp(-logvar) + logvar)


def club_learning_loss(q_network: CLUBConditional, z_click: torch.Tensor, z_channel: torch.Tensor) -> torch.Tensor:
    mu, logvar = q_network(z_click.detach())
    return -gaussian_log_likelihood(z_channel.detach(), mu, logvar).mean()


def club_mi_loss(q_network: CLUBConditional, z_click: torch.Tensor, z_channel: torch.Tensor) -> torch.Tensor:
    if z_click.shape[0] < 2:
        return (z_click.sum() + z_channel.sum()) * 0.0
    mu, logvar = q_network(z_click)
    positive = gaussian_log_likelihood(z_channel, mu, logvar).mean()
    negative_index = random_derangement(z_click.shape[0], z_click.device)
    negative = gaussian_log_likelihood(z_channel[negative_index], mu, logvar).mean()
    return positive - negative


def positive_latents(
    z_click: torch.Tensor, z_channel: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Select real click examples; classification negatives never form CLUB pairs."""
    mask = labels > 0.5
    count = int(mask.sum().item())
    return z_click[mask], z_channel[mask], count


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    states = [parameter.requires_grad for parameter in module.parameters()]
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    try:
        yield
    finally:
        for parameter, state in zip(module.parameters(), states):
            parameter.requires_grad_(state)


def gradient_audit(module: nn.Module) -> dict[str, Any]:
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    norms = [float(g.detach().norm().cpu()) for g in gradients]
    return {
        "parameter_tensors_with_gradient": len(gradients),
        "all_finite": bool(all(math.isfinite(value) for value in norms)),
        "total_norm": float(sum(norms)),
        "nonzero": bool(any(value > 0.0 for value in norms)),
    }


def train_two_optimizer_batch(
    model: CepstrumCLUBModel,
    q_network: CLUBConditional,
    main_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    received: torch.Tensor,
    labels: torch.Tensor,
    config: Config,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.train()
    q_network.train()
    zero = received.sum() * 0.0

    with torch.no_grad():
        z_click_detached, z_channel_detached, _, _ = model.encode(received)
        q_z_click, q_z_channel, mim_positive_count = positive_latents(
            z_click_detached, z_channel_detached, labels
        )
    main_optimizer.zero_grad(set_to_none=True)
    q_optimizer.zero_grad(set_to_none=True)
    mim_applied = mim_positive_count >= 2
    if mim_applied:
        q_loss = club_learning_loss(q_network, q_z_click, q_z_channel)
        q_loss.backward()
        q_audit_after_q = gradient_audit(q_network)
        encoder_audit_after_q = gradient_audit(model)
        q_optimizer.step()
    else:
        q_loss = received.sum() * 0.0
        q_audit_after_q = gradient_audit(q_network)
        encoder_audit_after_q = gradient_audit(model)

    main_optimizer.zero_grad(set_to_none=True)
    q_optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(q_network):
        outputs = model(received)
        outputs["z_click"].retain_grad()
        outputs["z_channel"].retain_grad()
        loss_recon = reconstruction_loss(received, outputs["r_hat"], config.epsilon)
        loss_bce = nn.functional.binary_cross_entropy_with_logits(outputs["logit"], labels)
        mim_z_click, mim_z_channel, _ = positive_latents(
            outputs["z_click"], outputs["z_channel"], labels
        )
        loss_mim = (club_mi_loss(q_network, mim_z_click, mim_z_channel)
                    if mim_applied else (outputs["z_click"].sum() + outputs["z_channel"].sum()) * 0.0)
        mim_z_click_grad, mim_z_channel_grad = torch.autograd.grad(
            loss_mim,
            (outputs["z_click"], outputs["z_channel"]),
            retain_graph=True,
            allow_unused=False,
        )
        total = (
            config.recon_weight * loss_recon
            + config.bce_weight * loss_bce
            + config.mim_weight * loss_mim
        )
        total.backward()
        z_click_grad = outputs["z_click"].grad
        z_channel_grad = outputs["z_channel"].grad
        main_audit = gradient_audit(model)
        q_audit_after_main = gradient_audit(q_network)
        latent_gradients = {
            "z_click_total": {
                "finite": bool(z_click_grad is not None and torch.isfinite(z_click_grad).all()),
                "norm": float(z_click_grad.detach().norm().cpu()) if z_click_grad is not None else 0.0,
            },
            "z_channel_total": {
                "finite": bool(z_channel_grad is not None and torch.isfinite(z_channel_grad).all()),
                "norm": float(z_channel_grad.detach().norm().cpu()) if z_channel_grad is not None else 0.0,
            },
            "z_click_from_mim": {
                "finite": bool(torch.isfinite(mim_z_click_grad).all()),
                "norm": float(mim_z_click_grad.detach().norm().cpu()),
            },
            "z_channel_from_mim": {
                "finite": bool(torch.isfinite(mim_z_channel_grad).all()),
                "norm": float(mim_z_channel_grad.detach().norm().cpu()),
            },
            "E_click": gradient_audit(model.E_click),
            "E_channel": gradient_audit(model.E_channel),
        }
        main_optimizer.step()

    losses = {
        "loss_q": float(q_loss.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_bce": float(loss_bce.detach().cpu()),
        "loss_mim": float(loss_mim.detach().cpu()),
        "loss_total": float(total.detach().cpu()),
        "mim_positive_count": float(mim_positive_count),
        "mim_applied": float(mim_applied),
    }
    audits = {
        "q_after_q_update": q_audit_after_q,
        "encoder_after_q_backward": encoder_audit_after_q,
        "main_after_main_backward": main_audit,
        "q_after_main_backward": q_audit_after_main,
        "latent_encoder_gradients": latent_gradients,
        "outputs_finite": bool(all(torch.isfinite(value).all() for value in outputs.values())),
        "losses_finite": bool(all(math.isfinite(value) for value in losses.values())),
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "h0_exact": bool(torch.equal(outputs["h_hat"][:, 0], torch.ones_like(outputs["h_hat"][:, 0]))),
    }
    return losses, audits


def balanced_batch_loader(
    waveforms: np.ndarray, labels: np.ndarray, batch_size: int, seed: int
) -> DataLoader:
    if batch_size < 2:
        raise ValueError("Smoke mode requires batch_size >= 2")
    rng = np.random.default_rng(seed)
    positive_count = batch_size // 2
    negative_count = batch_size - positive_count
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == 1), positive_count, replace=False),
            rng.choice(np.flatnonzero(labels == 0), negative_count, replace=False),
        ]
    )
    rng.shuffle(selected)
    dataset = WaveformDataset(waveforms, labels)
    return DataLoader(dataset, batch_size=batch_size, sampler=FixedIndexSampler(selected.tolist()))


def make_optimizers(
    model: nn.Module, q_network: nn.Module, config: Config
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    main_optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    q_optimizer = torch.optim.AdamW(
        q_network.parameters(), lr=config.q_learning_rate, weight_decay=config.weight_decay
    )
    return main_optimizer, q_optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    q_network: nn.Module,
    main_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    config: Config,
    extra: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "q_network": q_network.state_dict(),
            "main_optimizer": main_optimizer.state_dict(),
            "q_optimizer": q_optimizer.state_dict(),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            **extra,
        },
        path,
    )


def smoke_run(
    waveforms: np.ndarray, manifest: pd.DataFrame, config: Config, device: torch.device
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Acceptance smoke test requires a CUDA GPU; pass --device cuda or auto")
    labels = manifest["label"].to_numpy(np.float32)
    loader = balanced_batch_loader(waveforms, labels, config.batch_size, config.random_seed)
    received, batch_labels, indices = next(iter(loader))
    received = received.to(device)
    batch_labels = batch_labels.to(device)
    model = CepstrumCLUBModel().to(device)
    q_network = CLUBConditional().to(device)
    main_optimizer, q_optimizer = make_optimizers(model, q_network, config)
    started = time.perf_counter()
    losses, gradients = train_two_optimizer_batch(
        model, q_network, main_optimizer, q_optimizer, received, batch_labels, config
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    passed = bool(
        gradients["outputs_finite"]
        and gradients["losses_finite"]
        and gradients["h0_exact"]
        and gradients["output_shapes"]["h_hat"] == [config.batch_size, config.channel_count]
        and gradients["encoder_after_q_backward"]["parameter_tensors_with_gradient"] == 0
        and gradients["q_after_main_backward"]["parameter_tensors_with_gradient"] == 0
        and gradients["latent_encoder_gradients"]["E_click"]["nonzero"]
        and gradients["latent_encoder_gradients"]["E_channel"]["nonzero"]
        and gradients["latent_encoder_gradients"]["z_click_from_mim"]["finite"]
        and gradients["latent_encoder_gradients"]["z_click_from_mim"]["norm"] > 0.0
        and gradients["latent_encoder_gradients"]["z_channel_from_mim"]["finite"]
        and gradients["latent_encoder_gradients"]["z_channel_from_mim"]["norm"] > 0.0
        and gradients["main_after_main_backward"]["all_finite"]
        and gradients["q_after_q_update"]["nonzero"]
    )
    summary = {
        "mode": "smoke",
        "passed": passed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "batch_indices": indices.tolist(),
        "batch_labels": batch_labels.detach().cpu().tolist(),
        "elapsed_seconds": elapsed,
        "losses": losses,
        "gradient_audit": gradients,
    }
    json_dump(config.output_dir / "smoke_summary.json", summary)
    save_checkpoint(
        output_paths(config)["models"] / "smoke_checkpoint.pt",
        model,
        q_network,
        main_optimizer,
        q_optimizer,
        config,
        {"mode": "smoke", "summary": summary},
    )
    if not passed:
        raise RuntimeError(f"Smoke acceptance failed: {summary}")
    return summary


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for received, batch_labels, _ in loader:
        outputs = model(received.to(device))
        labels.append(batch_labels.numpy())
        probabilities.append(torch.sigmoid(outputs["logit"]).cpu().numpy())
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probabilities)
    prediction = y_score >= 0.5
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, prediction)),
    }


def quick_run(
    waveforms: np.ndarray, manifest: pd.DataFrame, config: Config, device: torch.device
) -> dict[str, Any]:
    labels = manifest["label"].to_numpy(np.float32)
    folds = manifest["test_fold"].to_numpy(int)
    train_indices = np.flatnonzero(~np.isin(folds, [0, 1]))
    validation_indices = np.flatnonzero(folds == 1)
    test_indices = np.flatnonzero(folds == 0)
    generator = torch.Generator().manual_seed(config.random_seed)
    train_loader = DataLoader(
        WaveformDataset(waveforms, labels, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        WaveformDataset(waveforms, labels, validation_indices), batch_size=config.batch_size
    )
    test_loader = DataLoader(WaveformDataset(waveforms, labels, test_indices), batch_size=config.batch_size)
    model = CepstrumCLUBModel().to(device)
    q_network = CLUBConditional().to(device)
    main_optimizer, q_optimizer = make_optimizers(model, q_network, config)
    history: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    started = time.perf_counter()
    epoch_bar = tqdm(range(config.epochs), desc="总体进度 3/4：quick 单折短训练", unit="epoch")
    for epoch in epoch_bar:
        epoch_losses: list[dict[str, float]] = []
        batch_bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}", leave=False, unit="batch")
        for batch_number, (received, batch_labels, _) in enumerate(batch_bar):
            loss_values, gradient_values = train_two_optimizer_batch(
                model,
                q_network,
                main_optimizer,
                q_optimizer,
                received.to(device),
                batch_labels.to(device),
                config,
            )
            epoch_losses.append(loss_values)
            batches.append(
                {
                    "epoch": epoch + 1,
                    "batch": batch_number + 1,
                    **loss_values,
                    "main_gradient_norm": gradient_values["main_after_main_backward"]["total_norm"],
                    "q_gradient_norm": gradient_values["q_after_q_update"]["total_norm"],
                }
            )
            batch_bar.set_postfix(
                total=f"{loss_values['loss_total']:.4f}",
                recon=f"{loss_values['loss_recon']:.4f}",
                bce=f"{loss_values['loss_bce']:.4f}",
                mim=f"{loss_values['loss_mim']:.4f}",
            )
        validation = evaluate(model, validation_loader, device)
        means = {
            key: float(np.mean([row[key] for row in epoch_losses]))
            for key in epoch_losses[0]
        }
        history.append({"epoch": epoch + 1, **means, **{f"validation_{k}": v for k, v in validation.items()}})
        epoch_bar.set_postfix(validation_auroc=f"{validation['auroc']:.4f}")
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(output_paths(config)["logs"] / "quick_history.csv", index=False)
    pd.DataFrame(batches).to_csv(output_paths(config)["logs"] / "quick_batches.csv", index=False)
    summary = {
        "mode": "quick",
        "scope": "single source-isolated split; folds 2-4 train, fold 1 validation, fold 0 test",
        "seed": config.random_seed,
        "epochs": config.epochs,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "validation_last": {key: value for key, value in history[-1].items() if key.startswith("validation_")},
        "test_metrics": test_metrics,
        "boundary": "Short engineering run only; not a five-fold, multi-seed, or paper-performance result.",
    }
    json_dump(config.output_dir / "quick_summary.json", summary)
    save_checkpoint(
        output_paths(config)["models"] / "quick_checkpoint.pt",
        model,
        q_network,
        main_optimizer,
        q_optimizer,
        config,
        {"mode": "quick", "summary": summary},
    )
    return summary


def condition_weights(condition: str) -> dict[str, float]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return {
        "recon": float(condition in ("full", "without_mim")),
        "bce": 1.0,
        "mim": float(condition in ("full", "without_recon")),
    }


def active_main_parameters(model: nn.Module, condition: str) -> list[nn.Parameter]:
    assert isinstance(model, CepstrumCLUBModel)
    weights = condition_weights(condition)
    modules: list[nn.Module] = [model.shared_frontend, model.E_click, model.D_click.classifier]
    if weights["recon"] or weights["mim"]:
        modules.append(model.E_channel)
    if weights["recon"]:
        modules.append(model.D_click.waveform)
    if weights["recon"]:
        modules.append(model.D_channel)
    selected: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                selected.append(parameter)
                seen.add(id(parameter))
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in seen)
    return selected


def condition_forward(model: nn.Module, received: torch.Tensor, condition: str) -> dict[str, torch.Tensor]:
    assert isinstance(model, CepstrumCLUBModel)
    weights = condition_weights(condition)
    shared = model.shared_frontend(received.unsqueeze(1))
    z_click, click_temporal = model.E_click(shared)
    outputs: dict[str, torch.Tensor] = {
        "z_click": z_click,
        "click_temporal": click_temporal,
        "logit": model.D_click.classifier(z_click).squeeze(-1),
    }
    if weights["recon"] or weights["mim"]:
        z_channel, channel_temporal = model.E_channel(shared)
        outputs.update({"z_channel": z_channel, "channel_temporal": channel_temporal})
    if weights["recon"]:
        outputs["x_hat"] = torch.tanh(model.D_click.waveform(z_click))
    if weights["recon"]:
        outputs["h_hat"] = model.D_channel(outputs["z_channel"])
    if weights["recon"]:
        outputs["r_hat"] = causal_fft_convolution(outputs["x_hat"], outputs["h_hat"])
    return outputs


def train_condition_batch(
    model: nn.Module,
    q_network: CLUBConditional | None,
    main_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer | None,
    received: torch.Tensor,
    labels: torch.Tensor,
    condition: str,
    config: Config,
) -> dict[str, float]:
    weights = condition_weights(condition)
    model.train()
    zero = received.sum() * 0.0
    q_loss = zero
    mim_positive_count = int((labels > 0.5).sum().item())
    mim_applied = bool(weights["mim"] and mim_positive_count >= 2)
    if weights["mim"]:
        assert q_network is not None and q_optimizer is not None
        q_network.train()
        with torch.no_grad():
            assert isinstance(model, CepstrumCLUBModel)
            shared = model.shared_frontend(received.unsqueeze(1))
            z_click, _ = model.E_click(shared)
            z_channel, _ = model.E_channel(shared)
            q_z_click, q_z_channel, _ = positive_latents(z_click, z_channel, labels)
        q_optimizer.zero_grad(set_to_none=True)
        if mim_applied:
            q_loss = club_learning_loss(q_network, q_z_click, q_z_channel)
            q_loss.backward()
            q_optimizer.step()
    main_optimizer.zero_grad(set_to_none=True)
    outputs = condition_forward(model, received, condition)
    loss_recon = reconstruction_loss(received, outputs["r_hat"], config.epsilon) if weights["recon"] else zero
    loss_bce = nn.functional.binary_cross_entropy_with_logits(outputs["logit"], labels)
    if mim_applied:
        assert q_network is not None
        with frozen_parameters(q_network):
            mim_z_click, mim_z_channel, _ = positive_latents(outputs["z_click"], outputs["z_channel"], labels)
            loss_mim = club_mi_loss(q_network, mim_z_click, mim_z_channel)
    else:
        loss_mim = zero
    total = loss_recon + loss_bce + loss_mim
    total.backward()
    torch.nn.utils.clip_grad_norm_([p for group in main_optimizer.param_groups for p in group["params"]], 5.0)
    main_optimizer.step()
    return {"loss_q": float(q_loss.detach().cpu()), "loss_recon": float(loss_recon.detach().cpu()),
            "loss_bce": float(loss_bce.detach().cpu()), "loss_mim": float(loss_mim.detach().cpu()),
            "loss_total": float(total.detach().cpu()), "mim_positive_count": float(mim_positive_count),
            "mim_applied": float(mim_applied)}


def choose_validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = np.asarray([balanced_accuracy_score(labels, probabilities >= value) for value in candidates])
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - 0.5))]])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "threshold": float(threshold),
    }


@torch.no_grad()
def predict_condition(model: nn.Module, q_network: CLUBConditional | None, loader: DataLoader,
                      device: torch.device, condition: str) -> pd.DataFrame:
    model.eval()
    if q_network is not None:
        q_network.eval()
    frames: list[pd.DataFrame] = []
    for received, labels, indices in loader:
        received_device = received.to(device, non_blocking=True)
        outputs = condition_forward(model, received_device, condition)
        count = len(labels)
        ncc = np.full(count, np.nan)
        tail_energy = np.full(count, np.nan)
        diversity_proxy = np.full(count, np.nan)
        mim = np.full(count, np.nan)
        if "x_hat" in outputs:
            centered_left = outputs["x_hat"] - outputs["x_hat"].mean(1, keepdim=True)
            centered_right = received_device - received_device.mean(1, keepdim=True)
            ncc = (torch.sum(centered_left * centered_right, 1) / (
                torch.linalg.vector_norm(centered_left, dim=1) *
                torch.linalg.vector_norm(centered_right, dim=1) + 1e-8)).cpu().numpy()
        if "h_hat" in outputs:
            tails = outputs["h_hat"][:, 1:]
            tail_energy = tails.square().sum(1).cpu().numpy()
            diversity_proxy = (tails - tails.mean(0, keepdim=True)).square().mean(1).sqrt().cpu().numpy()
        positive = labels.to(device=received_device.device) > 0.5
        if q_network is not None and int(positive.sum().item()) >= 2:
            value = float(club_mi_loss(
                q_network, outputs["z_click"][positive], outputs["z_channel"][positive]
            ).cpu())
            mim[positive.cpu().numpy()] = value
        frames.append(pd.DataFrame({
            "sample_index": indices.numpy(), "label": labels.numpy().astype(int),
            "probability": torch.sigmoid(outputs["logit"]).cpu().numpy(),
            "mim": mim, "ncc": ncc, "channel_tail_energy": tail_energy,
            "response_diversity": diversity_proxy,
        }))
    return pd.concat(frames, ignore_index=True)


def split_indices(manifest: pd.DataFrame, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = manifest["test_fold"].to_numpy(int)
    validation_fold = (test_fold + 1) % 5
    test = np.flatnonzero(folds == test_fold)
    validation = np.flatnonzero(folds == validation_fold)
    train = np.flatnonzero((folds != test_fold) & (folds != validation_fold))
    groups = [set(manifest.iloc[index]["ori_file_num"]) for index in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("Recording-source leakage across train/validation/test")
    return train, validation, test


def epoch_train_loader(waveforms: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                       batch_size: int, seed: int, epoch: int, workers: int) -> DataLoader:
    rng = np.random.default_rng(seed + epoch * 1_000_003)
    order = rng.permutation(len(indices)).tolist()
    return DataLoader(WaveformDataset(waveforms, labels, indices), batch_size=batch_size,
                      sampler=FixedIndexSampler(order), num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def evaluation_loader(waveforms: np.ndarray, labels: np.ndarray, indices: np.ndarray,
                      batch_size: int, workers: int) -> DataLoader:
    return DataLoader(WaveformDataset(waveforms, labels, indices), batch_size=batch_size * 2,
                      shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


def full_output_paths(config: Config) -> dict[str, Path]:
    paths = {"root": config.output_dir, "runs": config.output_dir / "runs",
             "metrics": config.output_dir / "metrics", "predictions": config.output_dir / "predictions"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def model_run_dir(config: Config, condition: str, seed: int, fold: int) -> Path:
    path = full_output_paths(config)["runs"] / condition / f"seed_{seed}" / f"fold_{fold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def parameter_audit(model: nn.Module, active: list[nn.Parameter], q_network: nn.Module | None) -> dict[str, int]:
    main_total = sum(parameter.numel() for parameter in model.parameters())
    q_total = sum(parameter.numel() for parameter in q_network.parameters()) if q_network is not None else 0
    return {"main_total_parameters": int(main_total), "q_parameters": int(q_total),
            "total_parameters": int(main_total + q_total),
            "trainable_parameters": int(sum(parameter.numel() for parameter in active) + q_total)}


def save_training_state(path: Path, model: nn.Module, q_network: nn.Module | None,
                        main_optimizer: torch.optim.Optimizer, q_optimizer: torch.optim.Optimizer | None,
                        epoch: int, best_score: float, best_epoch: int, stale: int,
                        condition: str, seed: int, fold: int, config: Config) -> None:
    torch.save({"model": model.state_dict(), "q_network": q_network.state_dict() if q_network else None,
                "main_optimizer": main_optimizer.state_dict(),
                "q_optimizer": q_optimizer.state_dict() if q_optimizer else None,
                "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                "condition": condition, "seed": seed, "fold": fold, "config_hash": config.config_hash,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def restore_training_state(path: Path, model: nn.Module, q_network: nn.Module | None,
                           main_optimizer: torch.optim.Optimizer,
                           q_optimizer: torch.optim.Optimizer | None, config: Config,
                           device: torch.device) -> tuple[int, float, int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("config_hash") != config.config_hash:
        raise RuntimeError(f"Resume config mismatch: {path}")
    model.load_state_dict(state["model"])
    main_optimizer.load_state_dict(state["main_optimizer"])
    if q_network is not None:
        q_network.load_state_dict(state["q_network"])
        assert q_optimizer is not None
        q_optimizer.load_state_dict(state["q_optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["epoch"]) + 1, float(state["best_score"]), int(state["best_epoch"]), int(state["stale"])


def restore_best_checkpoint(path: Path, model: nn.Module, q_network: nn.Module | None,
                            device: torch.device) -> dict[str, Any]:
    saved = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    if q_network is not None and saved.get("q_network") is not None:
        q_network.load_state_dict(saved["q_network"])
    return saved


def fit_full_model(condition: str, seed: int, fold: int, waveforms: np.ndarray,
                   manifest: pd.DataFrame, config: Config, device: torch.device,
                   resume: bool, overall: tqdm | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = model_run_dir(config, condition, seed, fold)
    complete_path = run_dir / "COMPLETE.json"
    best_path = run_dir / "best_checkpoint.pt"
    state_path = run_dir / "training_state.pt"
    history_path = run_dir / "history.csv"
    batch_path = run_dir / "batches.csv"
    prediction_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.json"
    required = (best_path, history_path, batch_path, prediction_path, metrics_path)
    complete_valid = False
    if resume and complete_path.exists() and all(path.exists() for path in required):
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        complete_valid = marker.get("config_hash") == config.config_hash
        if complete_valid:
            if overall is not None:
                overall.update(1)
            return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(prediction_path)

    initialization_seed = seed + fold
    seed_everything(initialization_seed)
    model: nn.Module = CepstrumCLUBModel()
    active = active_main_parameters(model, condition)
    q_network = CLUBConditional() if condition_weights(condition)["mim"] else None
    audit = parameter_audit(model, active, q_network)
    if audit["main_total_parameters"] > 900_000 or audit["total_parameters"] > 950_000:
        raise RuntimeError(f"Parameter limit exceeded: {audit}")
    model.to(device)
    if q_network is not None:
        q_network.to(device)
    main_optimizer = torch.optim.AdamW(active, lr=config.learning_rate, weight_decay=config.weight_decay)
    q_optimizer = (torch.optim.AdamW(q_network.parameters(), lr=config.q_learning_rate,
                                     weight_decay=config.weight_decay) if q_network is not None else None)
    start_epoch, best_score, best_epoch, stale = 1, -np.inf, 0, 0
    state_valid = False
    if resume and state_path.exists() and not complete_valid:
        state_header = torch.load(state_path, map_location="cpu", weights_only=False)
        state_valid = state_header.get("config_hash") == config.config_hash
        del state_header
    resume_incomplete = resume and state_valid and not complete_valid
    history_rows = (pd.read_csv(history_path).to_dict("records")
                    if resume_incomplete and history_path.exists() else [])
    batch_rows = (pd.read_csv(batch_path).to_dict("records")
                  if resume_incomplete and batch_path.exists() else [])
    if resume_incomplete:
        start_epoch, best_score, best_epoch, stale = restore_training_state(
            state_path, model, q_network, main_optimizer, q_optimizer, config, device)

    labels = manifest["label"].to_numpy(np.float32)
    train_indices, validation_indices, test_indices = split_indices(manifest, fold)
    validation_loader = evaluation_loader(waveforms, labels, validation_indices, config.batch_size,
                                          config.num_workers)
    test_loader = evaluation_loader(waveforms, labels, test_indices, config.batch_size, config.num_workers)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batch_seconds: list[float] = []
    epoch_bar = tqdm(range(start_epoch, config.max_epochs + 1),
                     desc=f"{CONDITION_LABELS[condition]} seed {seed} fold {fold}", leave=False, unit="epoch")
    for epoch in epoch_bar:
        loader = epoch_train_loader(waveforms, labels, train_indices, config.batch_size,
                                    seed + fold, epoch, config.num_workers)
        epoch_losses: list[dict[str, float]] = []
        for batch_number, (received, batch_labels, _) in enumerate(loader, start=1):
            received = received.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_started = time.perf_counter()
            losses = train_condition_batch(model, q_network, main_optimizer, q_optimizer,
                                             received, batch_labels, condition, config)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds.append(time.perf_counter() - batch_started)
            epoch_losses.append(losses)
            batch_rows.append({"condition": condition, "seed": seed, "fold": fold, "epoch": epoch,
                               "batch": batch_number, **losses, "batch_seconds": batch_seconds[-1]})
        validation = predict_condition(model, q_network, validation_loader, device, condition)
        score = float(roc_auc_score(validation["label"], validation["probability"]))
        means = {name: float(np.mean([row[name] for row in epoch_losses])) for name in epoch_losses[0]}
        history_rows.append({"condition": condition, "seed": seed, "fold": fold, "epoch": epoch,
                             **means, "validation_auroc": score})
        if score > best_score + config.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "q_network": q_network.state_dict() if q_network else None,
                        "condition": condition, "seed": seed, "fold": fold, "best_epoch": best_epoch,
                        "validation_auroc": best_score, "config_hash": config.config_hash,
                        "parameter_audit": audit}, best_path)
        else:
            stale += 1
        pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(batch_rows).to_csv(batch_path, index=False, encoding="utf-8-sig")
        save_training_state(state_path, model, q_network, main_optimizer, q_optimizer, epoch,
                            best_score, best_epoch, stale, condition, seed, fold, config)
        epoch_bar.set_postfix(val_auroc=f"{score:.4f}", best=f"{best_score:.4f}", stale=stale)
        if stale >= config.patience:
            break

    saved = restore_best_checkpoint(best_path, model, q_network, device)
    validation = predict_condition(model, q_network, validation_loader, device, condition)
    threshold = choose_validation_threshold(validation["label"].to_numpy(),
                                            validation["probability"].to_numpy())
    test = predict_condition(model, q_network, test_loader, device, condition)
    metrics: dict[str, Any] = binary_metrics(test["label"].to_numpy(), test["probability"].to_numpy(), threshold)
    elapsed = time.perf_counter() - started
    metrics.update({"condition": condition, "condition_label": CONDITION_LABELS[condition], "seed": seed,
                    "test_fold": fold, "validation_fold": (fold + 1) % 5,
                    "best_epoch": int(saved["best_epoch"]), "validation_auroc": float(saved["validation_auroc"]),
                    **audit, "training_seconds": elapsed,
                    "peak_memory_mb": (float(torch.cuda.max_memory_allocated(device) / 2**20)
                                       if device.type == "cuda" else 0.0),
                    "mean_batch_seconds": float(np.mean(batch_seconds)) if batch_seconds else float("nan"),
                    "mean_mim": float(test.loc[test["label"] == 1, "mim"].mean()),
                    "mean_ncc": float(test["ncc"].mean()),
                    "mean_channel_tail_energy": float(test["channel_tail_energy"].mean()),
                    "mean_response_diversity": float(test["response_diversity"].mean())})
    test["condition"] = condition
    test["seed"] = seed
    test["test_fold"] = fold
    test["threshold"] = threshold
    metadata = manifest.reset_index(names="sample_index")[["sample_index", "sample_id", "ori_file_num",
                                                            "target_snr_db"]]
    test = test.merge(metadata, on="sample_index", how="left", validate="one_to_one")
    test.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    json_dump(metrics_path, metrics)
    json_dump(complete_path, {"training_complete": True, "config_hash": config.config_hash,
                              "best_checkpoint": str(best_path.resolve())})
    if overall is not None:
        overall.update(1)
        overall.set_postfix(condition=condition, seed=seed, fold=fold, auroc=f"{metrics['auroc']:.4f}")
    return metrics, test


METRIC_NAMES = ("auroc", "auprc", "balanced_accuracy", "f1", "sensitivity", "specificity")


def source_resample(frame: pd.DataFrame, sampled_sources: np.ndarray) -> pd.DataFrame:
    pieces = []
    for draw, source in enumerate(sampled_sources):
        piece = frame[frame["ori_file_num"] == source].copy()
        piece["bootstrap_source_draw"] = draw
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def frame_metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].to_numpy(int)
    probabilities = frame["probability"].to_numpy(float)
    predicted = probabilities >= frame["threshold"].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            "auprc": float(average_precision_score(labels, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1))}


def summarize_full_results(metrics: pd.DataFrame, predictions: pd.DataFrame,
                           config: Config, seeds: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    paths = full_output_paths(config)
    seed_rows = []
    for (condition, seed), frame in predictions.groupby(["condition", "seed"], sort=False):
        row = frame_metrics(frame)
        row.update({"condition": condition, "seed": int(seed)})
        seed_rows.append(row)
    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(paths["metrics"] / "seed_metrics.csv", index=False, encoding="utf-8-sig")

    main_rows = []
    for condition, group in metrics.groupby("condition", sort=False):
        seed_group = seed_metrics[seed_metrics["condition"] == condition]
        row: dict[str, Any] = {"condition": condition, "condition_label": CONDITION_LABELS[condition],
                               "total_parameters": int(group["total_parameters"].max()),
                               "trainable_parameters": int(group["trainable_parameters"].max())}
        for metric in METRIC_NAMES:
            row[metric] = float(seed_group[metric].mean())
            row[f"{metric}_seed_sd"] = float(seed_group[metric].std(ddof=1))
        row["training_seconds"] = float(group["training_seconds"].sum())
        row["peak_memory_mb"] = float(group["peak_memory_mb"].max())
        row["mean_batch_seconds"] = float(group["mean_batch_seconds"].mean())
        main_rows.append(row)
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(paths["metrics"] / "main_table.csv", index=False, encoding="utf-8-sig")

    snr_rows = []
    bins = np.arange(-15.0, 15.0 + 5.0, 5.0)
    for condition in predictions["condition"].unique():
        for seed in seeds:
            frame = predictions[(predictions["condition"] == condition) & (predictions["seed"] == seed)]
            negatives = frame[frame["label"] == 0]
            for lower, upper in zip(bins[:-1], bins[1:]):
                positives = frame[(frame["label"] == 1) & (frame["target_snr_db"] >= lower) &
                                  (frame["target_snr_db"] < upper if upper < 15 else
                                   frame["target_snr_db"].le(upper))]
                combined = pd.concat([positives, negatives], ignore_index=True)
                snr_rows.append({"condition": condition, "seed": seed, "snr_lower_db": lower,
                                 "snr_upper_db": upper, "positive_count": len(positives),
                                 "auroc": float(roc_auc_score(combined["label"], combined["probability"])),
                                 "sensitivity": float((positives["probability"] >=
                                                       positives["threshold"]).mean())})
    snr_detailed = pd.DataFrame(snr_rows)
    snr_summary = snr_detailed.groupby(["condition", "snr_lower_db", "snr_upper_db"], as_index=False).agg(
        positive_count=("positive_count", "first"), auroc=("auroc", "mean"),
        auroc_seed_sd=("auroc", "std"), sensitivity=("sensitivity", "mean"),
        sensitivity_seed_sd=("sensitivity", "std"))
    snr_detailed.to_csv(paths["metrics"] / "snr_table_by_seed.csv", index=False, encoding="utf-8-sig")
    snr_summary.to_csv(paths["metrics"] / "snr_table.csv", index=False, encoding="utf-8-sig")

    diagnostic_rows = []
    for condition, group in metrics.groupby("condition", sort=False):
        for name in ("mean_mim", "mean_ncc", "mean_channel_tail_energy", "mean_response_diversity"):
            values = group.groupby("seed")[name].mean()
            diagnostic_rows.append({"condition": condition, "diagnostic": name,
                                    "mean": float(values.mean()), "seed_sd": float(values.std(ddof=1)),
                                    "finite_count": int(np.isfinite(values).sum())})
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(paths["metrics"] / "disentanglement_diagnostics.csv", index=False,
                       encoding="utf-8-sig")

    rng = np.random.default_rng(20_260_901)
    sources = predictions["ori_file_num"].unique()
    draws = [(rng.choice(np.asarray(seeds), len(seeds), replace=True),
              rng.choice(sources, len(sources), replace=True)) for _ in range(config.bootstrap_runs)]
    paired_rows = []
    full_by_seed = {seed: predictions[(predictions["condition"] == "full") &
                                      (predictions["seed"] == seed)] for seed in seeds}
    for ablation in [value for value in CONDITIONS if value != "full" and value in predictions["condition"].unique()]:
        ablation_by_seed = {seed: predictions[(predictions["condition"] == ablation) &
                                              (predictions["seed"] == seed)] for seed in seeds}
        seed_diffs = {metric: [] for metric in METRIC_NAMES}
        for seed in seeds:
            left, right = frame_metrics(full_by_seed[seed]), frame_metrics(ablation_by_seed[seed])
            for metric in METRIC_NAMES:
                seed_diffs[metric].append(left[metric] - right[metric])
        bootstrap_values = {metric: [] for metric in METRIC_NAMES}
        for sampled_seeds, sampled_sources in tqdm(draws, desc=f"Full-{ablation} 两级配对Bootstrap",
                                                   leave=False):
            replicate = {metric: [] for metric in METRIC_NAMES}
            for sampled_seed in sampled_seeds:
                left_frame = source_resample(full_by_seed[int(sampled_seed)], sampled_sources)
                right_frame = source_resample(ablation_by_seed[int(sampled_seed)], sampled_sources)
                if left_frame["label"].nunique() < 2 or right_frame["label"].nunique() < 2:
                    continue
                left, right = frame_metrics(left_frame), frame_metrics(right_frame)
                for metric in METRIC_NAMES:
                    replicate[metric].append(left[metric] - right[metric])
            for metric in METRIC_NAMES:
                if replicate[metric]:
                    bootstrap_values[metric].append(float(np.mean(replicate[metric])))
        for metric in METRIC_NAMES:
            values = bootstrap_values[metric]
            low, high = np.percentile(values, [2.5, 97.5])
            paired_rows.append({"variant": ablation, "removed_mechanism": CONDITION_LABELS[ablation],
                                "metric": metric, "full_minus_ablation": float(np.mean(seed_diffs[metric])),
                                "ci_low": float(low), "ci_high": float(high),
                                "evidence": "证据不充分" if low <= 0 <= high else "CI不跨0",
                                "full_seed_wins": int(np.sum(np.asarray(seed_diffs[metric]) > 0)),
                                "bootstrap_runs_valid": len(values)})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(paths["metrics"] / "ablation_table.csv", index=False, encoding="utf-8-sig")
    return {"seed_metrics": seed_metrics, "main": main_table, "snr": snr_summary,
            "diagnostics": diagnostics, "paired": paired}


def verify_full_artifacts(config: Config, conditions: tuple[str, ...], seeds: tuple[int, ...],
                          folds: tuple[int, ...], metrics: pd.DataFrame,
                          predictions: pd.DataFrame) -> dict[str, Any]:
    expected_models = len(conditions) * len(seeds) * len(folds)
    expected_predictions = len(conditions) * len(seeds) * 7_000 if set(folds) == set(range(5)) else None
    run_dirs = [model_run_dir(config, condition, seed, fold) for condition in conditions
                for seed in seeds for fold in folds]
    complete_headers = [json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
                        for path in run_dirs if (path / "COMPLETE.json").exists()]
    required_prediction_columns = ["label", "probability", "threshold"]
    checks = {"expected_models": expected_models,
              "best_checkpoint_count": sum((path / "best_checkpoint.pt").exists() for path in run_dirs),
              "complete_marker_count": sum((path / "COMPLETE.json").exists() for path in run_dirs),
              "history_count": sum((path / "history.csv").exists() for path in run_dirs),
              "batch_log_count": sum((path / "batches.csv").exists() for path in run_dirs),
              "fold_metric_rows": len(metrics), "prediction_rows": len(predictions),
              "expected_prediction_rows": expected_predictions,
              "prediction_keys_unique": not predictions.duplicated(["condition", "seed", "sample_id"]).any(),
              "all_numeric_finite": bool(np.isfinite(metrics[list(METRIC_NAMES)].to_numpy()).all()),
              "prediction_numeric_finite_except_mim": bool(
                  np.isfinite(predictions[required_prediction_columns].to_numpy()).all()),
              "mim_missing_only_when_expected": bool(
                  predictions.loc[predictions["label"] == 0, "mim"].isna().all()),
              "complete_config_hashes_match": bool(
                  len(complete_headers) == expected_models and
                  all(header.get("config_hash") == config.config_hash for header in complete_headers)),
              "run_config_hash_matches": bool(
                  json.loads((config.output_dir / "run_config.json").read_text(encoding="utf-8"))
                  .get("config_hash") == config.config_hash)}
    checks["passed"] = bool(checks["best_checkpoint_count"] == expected_models and
                             checks["complete_marker_count"] == expected_models and
                             checks["history_count"] == expected_models and
                             checks["batch_log_count"] == expected_models and
                             checks["fold_metric_rows"] == expected_models and
                             (expected_predictions is None or len(predictions) == expected_predictions) and
                             checks["prediction_keys_unique"] and checks["all_numeric_finite"])
    checks["passed"] = bool(checks["passed"] and checks["prediction_numeric_finite_except_mim"] and
                             checks["mim_missing_only_when_expected"] and
                             checks["complete_config_hashes_match"] and checks["run_config_hash_matches"])
    json_dump(config.output_dir / "artifact_validation.json", checks)
    return checks


def load_locked_cache(config: Config) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    manifest_path = BASE_CACHE_DIR / "dataset_manifest.csv"
    cache_path = BASE_CACHE_DIR / "cache" / "aligned_mixed_waveforms.npy"
    audit_path = BASE_CACHE_DIR / "waveform_audit.csv"
    for path in (manifest_path, cache_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(f"Locked experiment-22 cache artifact is missing: {path}")
    manifest = pd.read_csv(manifest_path)
    waveforms = np.load(cache_path, mmap_mode="r")
    audit = pd.read_csv(audit_path)
    if len(manifest) != 7_000 or waveforms.shape != (7_000, SAMPLE_COUNT) or len(audit) != 7_000:
        raise RuntimeError("Locked experiment-22 cache has an invalid shape")
    if manifest["label"].value_counts().to_dict() != {1: 3_500, 0: 3_500}:
        raise RuntimeError("Locked experiment-22 manifest class counts changed")
    if manifest.groupby("ori_file_num")["test_fold"].nunique().max() != 1:
        raise RuntimeError("Locked experiment-22 manifest has source leakage")
    positive_audit = audit[audit["label"] == 1]
    if not positive_audit["aligned_click_peak_index"].eq(SAMPLE_COUNT // 2).all():
        raise RuntimeError("Locked experiment-22 peak alignment audit failed")
    contract = {"sha256": EXPECTED_DATA_HASH, "samples": 7_000, "shape": list(waveforms.shape),
                "manifest_path": str(manifest_path.resolve()), "cache_path": str(cache_path.resolve()),
                "cache_bytes": cache_path.stat().st_size, "cache_mtime_ns": cache_path.stat().st_mtime_ns,
                "note": "Hash is the locked accepted data-contract hash recorded on 2026-09-01; cache is read-only."}
    return manifest, waveforms, contract


def run_full_experiment(config: Config, seeds: tuple[int, ...], folds: tuple[int, ...],
                        conditions: tuple[str, ...], resume: bool) -> dict[str, Any]:
    started = time.perf_counter()
    paths = full_output_paths(config)
    manifest, waveforms, data_contract = load_locked_cache(config)
    manifest.to_csv(paths["root"] / "dataset_manifest.csv", index=False, encoding="utf-8-sig")
    json_dump(paths["root"] / "data_contract.json", data_contract)
    json_dump(paths["root"] / "run_config.json", {**serializable_config(config, "full"),
              "seeds": list(seeds), "folds": list(folds), "conditions": list(conditions),
              "resume": resume, "data_hash": EXPECTED_DATA_HASH})
    device = resolve_device(config.device)
    overall = tqdm(total=len(seeds) * len(folds) * len(conditions),
                   desc="总体进度：第22个完整实验", unit="模型")
    all_metrics, all_predictions = [], []
    for seed in seeds:
        for fold in folds:
            for condition in conditions:
                metric, prediction = fit_full_model(condition, seed, fold, waveforms, manifest,
                                                    config, device, resume, overall)
                all_metrics.append(metric)
                all_predictions.append(prediction)
    overall.close()
    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.to_csv(paths["metrics"] / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(paths["predictions"] / "sample_predictions.csv", index=False, encoding="utf-8-sig")
    summarize_full_results(metrics, predictions, config, seeds)
    artifacts = verify_full_artifacts(config, conditions, seeds, folds, metrics, predictions)
    result = {"mode": "full", "data_hash": EXPECTED_DATA_HASH, "config_hash": config.config_hash,
              "models": len(metrics), "predictions": len(predictions), "device": str(device),
              "artifact_validation": artifacts,
              "elapsed_seconds": time.perf_counter() - started}
    json_dump(paths["root"] / "run_summary.json", result)
    if not artifacts["passed"]:
        raise RuntimeError(f"Full artifact acceptance failed: {artifacts}")
    return result


def serializable_config(config: Config, mode: str) -> dict[str, Any]:
    result = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    result.update(
        {
            "mode": mode,
            "config_hash": config.config_hash,
            "label_definition": {"1": "Indo-Pacific humpback dolphin click", "0": "hard noise"},
            "method_boundary": "Binary click-versus-noise classification; CLUB pairs are restricted to real positive clicks.",
        }
    )
    return result


def run(config: Config, mode: str, force: bool = False) -> dict[str, Any]:
    seed_everything(config.random_seed)
    paths = output_paths(config)
    json_dump(paths["root"] / "run_config.json", serializable_config(config, mode))
    print("[总体进度  5%] 读取并复制第21个固定低SNR清单", flush=True)
    manifest = load_reference_manifest(config, force=force)
    print("[总体进度 15%] 构建/读取第22个独立峰对齐混合缓存", flush=True)
    waveforms, audit = materialize_waveforms(manifest, config, force=force)
    print("[总体进度 45%] 执行数据哈希与数据契约审计", flush=True)
    data_hash = compute_data_hash(manifest, waveforms, config)
    data_audit = validate_data(manifest, waveforms, audit, config)
    device = resolve_device(config.device)
    print(f"[总体进度 60%] 在 {device} 执行 {mode} 双优化器流程", flush=True)
    result = smoke_run(waveforms, manifest, config, device) if mode == "smoke" else quick_run(
        waveforms, manifest, config, device
    )
    print("[总体进度 95%] 写入汇总与检查点", flush=True)
    run_summary = {
        "mode": mode,
        "passed": bool(result.get("passed", True)),
        "data_hash": data_hash,
        "data_audit": data_audit,
        "result": result,
    }
    json_dump(paths["root"] / "run_summary.json", run_summary)
    print("[总体进度 100%] 完成", flush=True)
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quick", "full"), default="smoke")
    parser.add_argument("--manifest-source", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--mim-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--q-learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20_260_826, 20_260_827, 20_260_828, 20_260_829, 20_260_830])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bootstrap-runs", type=int, default=2_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.batch_size < 2:
        raise ValueError("batch-size must be >= 2 for CLUB training")
    weights = [
        arguments.recon_weight,
        arguments.bce_weight,
        arguments.mim_weight,
    ]
    if any(value < 0 or not math.isfinite(value) for value in weights):
        raise ValueError("Loss weights must be finite and nonnegative")
    if arguments.max_epochs < 1 or arguments.patience < 1 or arguments.bootstrap_runs < 1:
        raise ValueError("max-epochs, patience, and bootstrap-runs must be positive")
    if any(fold not in range(5) for fold in arguments.folds) or len(set(arguments.folds)) != len(arguments.folds):
        raise ValueError("folds must be unique values from 0 through 4")
    if len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("seeds must be unique")
    default_outputs = {"smoke": EXPERIMENT_DIR / "outputs_smoke",
                       "quick": EXPERIMENT_DIR / "outputs_quick", "full": FULL_OUTPUT_DIR}
    output_dir = (arguments.output_dir or default_outputs[arguments.mode]).resolve()
    config = Config(
        output_dir=output_dir,
        manifest_source=arguments.manifest_source.resolve(),
        recon_weight=arguments.recon_weight,
        bce_weight=arguments.bce_weight,
        mim_weight=arguments.mim_weight,
        learning_rate=arguments.learning_rate,
        q_learning_rate=arguments.q_learning_rate,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        random_seed=arguments.seed,
        num_workers=arguments.num_workers,
        device=arguments.device,
        max_epochs=arguments.max_epochs,
        patience=arguments.patience,
        bootstrap_runs=arguments.bootstrap_runs,
    )
    if arguments.mode == "full":
        result = run_full_experiment(config, tuple(arguments.seeds), tuple(arguments.folds),
                                     tuple(arguments.conditions), arguments.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    else:
        run(config, arguments.mode, force=arguments.force)


if __name__ == "__main__":
    main()
