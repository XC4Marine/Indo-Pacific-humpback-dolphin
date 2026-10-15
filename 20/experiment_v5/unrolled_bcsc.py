"""Truth-free six-stage differentiable BCSC core used by v5 inference and training."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


FS = 576_000
SOURCE_LENGTH = 116
CHANNEL_LENGTH = int(round(0.004 * FS))
SEGMENT_LENGTH = SOURCE_LENGTH + CHANNEL_LENGTH - 1
EVENT_COUNT = 10


def causal_fft_convolution(source: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """Convolve B shared sources with B-by-event causal channel responses."""
    if source.ndim != 2 or channels.ndim != 3:
        raise ValueError("expected source=(batch,116), channels=(batch,event,2304)")
    length = source.shape[-1] + channels.shape[-1] - 1
    return torch.fft.irfft(torch.fft.rfft(source[:, None], n=length) * torch.fft.rfft(channels, n=length), n=length)


def soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.relu(values.abs() - threshold)


def normalise_source(values: torch.Tensor) -> torch.Tensor:
    taper = torch.hann_window(SOURCE_LENGTH, periodic=False, device=values.device, dtype=values.dtype).clamp_min(0.05)
    result = values * taper
    result = result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-8)
    signs = torch.where(result.gather(1, result.abs().argmax(dim=1, keepdim=True)) < 0, -1.0, 1.0)
    return result * signs


def initialise_source(segments: torch.Tensor) -> torch.Tensor:
    """v4-style, waveform-only central burst initialization."""
    peaks = segments.abs().argmax(dim=-1)
    centres = peaks.median(dim=1).values
    mean_segment = segments.median(dim=1).values
    padded = F.pad(mean_segment, (SOURCE_LENGTH // 2, SOURCE_LENGTH // 2))
    offsets = centres[:, None] + torch.arange(SOURCE_LENGTH, device=segments.device)[None]
    atoms = padded.gather(1, offsets)
    return normalise_source(atoms)


class UnrolledBCSC(nn.Module):
    """Six unrolled BCSC stages; only positive step sizes and thresholds are learned."""

    def __init__(self, stages: int = 6, learnable: bool = True) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be positive")
        self.stages = stages
        self.learnable = learnable
        # Values are intentionally small because inputs are normalised per segment.
        code_step = torch.full((stages,), math.log(math.expm1(0.20)), dtype=torch.float32)
        source_step = torch.full((stages,), math.log(math.expm1(0.08)), dtype=torch.float32)
        threshold = torch.full((stages,), math.log(math.expm1(0.012)), dtype=torch.float32)
        if learnable:
            self.log_code_step = nn.Parameter(code_step)
            self.log_source_step = nn.Parameter(source_step)
            self.log_threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("log_code_step", code_step)
            self.register_buffer("log_source_step", source_step)
            self.register_buffer("log_threshold", threshold)

    def positive_parameters(self) -> dict[str, torch.Tensor]:
        return {"code_step": F.softplus(self.log_code_step), "source_step": F.softplus(self.log_source_step),
                "threshold": F.softplus(self.log_threshold)}

    def forward(self, segments: torch.Tensor) -> dict[str, torch.Tensor]:
        if segments.ndim != 3 or segments.shape[1:] != (EVENT_COUNT, SEGMENT_LENGTH):
            raise ValueError(f"expected (batch,{EVENT_COUNT},{SEGMENT_LENGTH}), got {tuple(segments.shape)}")
        scales = torch.linalg.vector_norm(segments, dim=-1, keepdim=True).clamp_min(1e-8)
        y = segments / scales
        source = initialise_source(y)
        channels = torch.zeros((len(y), EVENT_COUNT, CHANNEL_LENGTH), dtype=y.dtype, device=y.device)
        positive = self.positive_parameters()
        for stage in range(self.stages):
            reconstruction = causal_fft_convolution(source, channels)
            residual = reconstruction - y
            source_for_codes = source[:, None].expand(-1, EVENT_COUNT, -1).reshape(-1, 1, SOURCE_LENGTH)
            group_count = len(y) * EVENT_COUNT
            gradient_h = F.conv1d(residual.reshape(1, group_count, SEGMENT_LENGTH), source_for_codes,
                                  groups=group_count).reshape_as(channels)
            channels = soft_threshold(channels - positive["code_step"][stage] * gradient_h,
                                      positive["threshold"][stage])
            reconstruction = causal_fft_convolution(source, channels)
            residual = reconstruction - y
            gradient_x = F.conv1d(residual.reshape(1, group_count, SEGMENT_LENGTH), channels.reshape(-1, 1, CHANNEL_LENGTH),
                                  groups=group_count)
            gradient_x = gradient_x.reshape(len(y), EVENT_COUNT, SOURCE_LENGTH).mean(dim=1)
            source = normalise_source(source - positive["source_step"][stage] * gradient_x)
        reconstruction = causal_fft_convolution(source, channels)
        return {"source": source, "channels": channels * scales, "reconstructions": reconstruction * scales,
                "normalised_reconstructions": reconstruction}
