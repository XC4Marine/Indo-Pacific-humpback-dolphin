"""Truth-free v7 residual model: joint source and time-domain channel refinement."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def unit_norm(values: torch.Tensor) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


def causal_convolution(source: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
    """Batchwise causal convolution for [B,S] sources and [B,E,H] channels."""
    output_length = source.shape[-1] + channels.shape[-1] - 1
    source_spectrum = torch.fft.rfft(source[:, None, :], n=output_length)
    channel_spectrum = torch.fft.rfft(channels, n=output_length)
    return torch.fft.irfft(source_spectrum * channel_spectrum, n=output_length)


class TimeDomainRefiner(nn.Module):
    """Small residual network whose channel branch consumes only BCSC and raw-data adjoints."""
    def __init__(self, source_length: int = 116) -> None:
        super().__init__()
        self.source_net = nn.Sequential(nn.Linear(source_length, 192), nn.GELU(), nn.Linear(192, source_length))
        # The two inputs already encode a channel estimate and the source-adjoint
        # evidence from the raw event window.  A pointwise residual mapping keeps
        # the model small enough for the 480 supervised event instances while
        # retaining an event-specific, time-domain correction.
        self.channel_net = nn.Sequential(nn.Conv1d(2, 12, 1), nn.GELU(), nn.Conv1d(12, 1, 1))
        nn.init.zeros_(self.source_net[-1].weight); nn.init.zeros_(self.source_net[-1].bias)
        nn.init.zeros_(self.channel_net[-1].weight); nn.init.zeros_(self.channel_net[-1].bias)

    def forward(self, source: torch.Tensor, channels: torch.Tensor, proxy: torch.Tensor) -> dict[str, torch.Tensor]:
        # Bounded residual gates preserve BCSC as a stable fixed point.  A model
        # must earn a deviation through validation loss; it cannot erase the
        # sparse BCSC channel merely because of an early large gradient.
        refined_source = unit_norm(source + 0.12 * torch.tanh(self.source_net(source)))
        scale = channels.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        baseline = channels / scale
        matched = proxy / scale
        residual = self.channel_net(torch.stack((baseline, matched), dim=2).reshape(-1, 2, channels.shape[-1]))
        refined_channels = (baseline.reshape(-1, 1, channels.shape[-1]) + 0.30 * torch.tanh(residual)).reshape_as(channels) * scale
        reconstruction = causal_convolution(refined_source, refined_channels)
        return {"source": refined_source, "channels": refined_channels, "reconstructions": reconstruction}
