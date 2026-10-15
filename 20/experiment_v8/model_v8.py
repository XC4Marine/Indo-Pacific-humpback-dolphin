"""Shared self-supervised sparse-gating network used by Experiment v8."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from protocol_v8 import CHANNEL_SAMPLES, EVENT_COUNT, FS, SEGMENT_SAMPLES, SOURCE_SAMPLES


class SparseGateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(EVENT_COUNT, 32, 9, padding=4), nn.GELU(), nn.MaxPool1d(4),
                                     nn.Conv1d(32, 48, 7, padding=3), nn.GELU(), nn.AdaptiveAvgPool1d(32), nn.Flatten(),
                                     nn.Linear(48 * 32, 256), nn.GELU())
        self.source_head = nn.Linear(256, SOURCE_SAMPLES)
        self.mask_head = nn.Linear(256, CHANNEL_SAMPLES - 1)
        self.amplitude_head = nn.Linear(256, EVENT_COUNT * CHANNEL_SAMPLES)

    def forward(self, segments: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(segments)
        source = self.source_head(hidden)
        source = source / (source.abs().amax(dim=1, keepdim=True) + 1e-8)
        mask_reflections = torch.sigmoid(self.mask_head(hidden))
        mask = torch.cat((torch.ones_like(mask_reflections[:, :1]), mask_reflections), dim=1)
        amplitudes = 2.0 * torch.tanh(self.amplitude_head(hidden).view(-1, EVENT_COUNT, CHANNEL_SAMPLES))
        channels = amplitudes * mask[:, None, :]
        nfft = SEGMENT_SAMPLES
        reconstruction = torch.fft.irfft(torch.fft.rfft(channels, n=nfft) * torch.fft.rfft(source, n=nfft)[:, None, :], n=nfft)
        return {"source": source, "mask": mask, "amplitudes": amplitudes, "channels": channels, "reconstruction": reconstruction}


def gate_loss(result: dict[str, torch.Tensor], segments: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    energy = torch.mean(segments * segments, dim=(1, 2), keepdim=True) + 1e-8
    reconstruction = torch.mean((result["reconstruction"] - segments) ** 2 / energy)
    # Universal threshold scale; it is fixed from dimensionality and never selected with truth.
    lambda_mask = math.sqrt(2.0 * math.log(CHANNEL_SAMPLES) / (EVENT_COUNT * SEGMENT_SAMPLES))
    sparsity = torch.mean(result["mask"][:, 1:])
    amplitude_stability = torch.mean(result["amplitudes"] ** 2)
    loss = reconstruction + lambda_mask * sparsity + 1e-4 * amplitude_stability
    return loss, {"reconstruction": float(reconstruction.detach()), "sparsity": float(sparsity.detach()), "loss": float(loss.detach())}


def gate_candidates(mask: np.ndarray, support: int) -> list[float]:
    from scipy.signal import find_peaks
    mask = np.asarray(mask, dtype=float).reshape(-1)
    minimum = max(int(round(30e-6 * FS)), int(support))
    peaks, _ = find_peaks(mask[minimum:], height=.5, distance=max(1, int(support)))
    return [float((minimum + peak) / FS * 1e3) for peak in peaks]
