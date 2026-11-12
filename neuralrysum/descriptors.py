"""Explicit timbre descriptors, measured once per slow frame.

They are fed to the decoder next to the learned latent ``z`` so that three of
the plugin's knobs have a fixed, known meaning. All three are scaled to land
roughly in [-1, 1]; silent frames are 0.

    0  brightness  spectral centroid       (0 = 1 kHz, +-1 = three octaves)
    1  noisiness   spectral flatness       (+1 = white noise, -1 = pure tones)
    2  spread      spectral bandwidth      (0 = 1 kHz, +-1 = three octaves)
"""
from __future__ import annotations

import math

import torch

NAMES = ["brightness", "noisiness", "spread"]


def descriptors(x: torch.Tensor, slow_hop: int, sample_rate: int) -> torch.Tensor:
    """x: (B, T) or (B, 1, T) with T a multiple of slow_hop -> (B, 3, T / slow_hop)."""
    if x.dim() == 3:
        x = x.squeeze(1)
    b, t = x.shape
    assert t % slow_hop == 0
    frames = x.view(b, t // slow_hop, slow_hop)
    win = torch.hann_window(slow_hop, device=x.device, dtype=x.dtype)
    mag = torch.fft.rfft(frames * win, dim=-1).abs()[..., 1:]  # drop DC
    freqs = torch.fft.rfftfreq(slow_hop, 1.0 / sample_rate).to(x)[1:]
    power = mag**2
    total = mag.sum(-1)
    silent = total < 1e-4 * math.sqrt(slow_hop)

    centroid = (mag * freqs).sum(-1) / total.clamp_min(1e-9)
    spread = torch.sqrt(
        (mag * (freqs - centroid[..., None]) ** 2).sum(-1) / total.clamp_min(1e-9)
    )
    flat = torch.exp(torch.log(power + 1e-12).mean(-1)) / (power.mean(-1) + 1e-12)
    flat_db = 10.0 * torch.log10(flat.clamp_min(1e-12))

    brightness = torch.log2(centroid.clamp_min(20.0) / 1000.0) / 3.0
    noisiness = flat_db / 40.0 + 1.0
    spread_n = torch.log2(spread.clamp_min(20.0) / 1000.0) / 3.0
    d = torch.stack([brightness, noisiness, spread_n], dim=1).clamp(-1.5, 1.5)
    return d.masked_fill(silent[:, None, :], 0.0)
