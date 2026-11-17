"""Synthetic non-target input for the transfer objective.

Reconstruction alone never asks the model to turn a *foreign* sound into the
trained one: every example is organ in, organ out, and for anything else the
cheapest solution is to copy the input. With ``--foreign_weight`` the training
loop also sends the signals made here through the fast path, with the timbre
of a real clip, and asks for an output that

- the discriminator accepts as the trained sound,
- has its partials where the input has them (``fine_structure_loss``),
- is as loud as the input at every moment (``loudness_loss``).

The signals are random on purpose: one to three harmonic tones with random
pitch, glide, vibrato, spectrum, onset, decay and level, a random EQ on top,
and a little noise.
"""
from __future__ import annotations

import math

import torch

from .augment import random_eq


def _u(b: int, lo: float, hi: float, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(b, 1, device=device)


@torch.no_grad()
def synth_foreign(b: int, t: int, sample_rate: int, device, harmonics: int = 48) -> torch.Tensor:
    """-> (b, 1, t), peak between 0.1 and 0.6."""
    time = torch.arange(t, device=device)[None] / sample_rate  # (1, t)
    dur = t / sample_rate
    out = torch.zeros(b, t, device=device)
    for voice in range(3):
        on = torch.ones(b, 1, device=device) if voice == 0 else (torch.rand(b, 1, device=device) < 0.4).float()
        f0 = 55.0 * (1200.0 / 55.0) ** torch.rand(b, 1, device=device)
        glide = _u(b, -3.0, 3.0, device) / 12.0 * (torch.rand(b, 1, device=device) < 0.3)
        vib = _u(b, 0.0, 0.01, device) * (torch.rand(b, 1, device=device) < 0.3)
        freq = f0 * 2.0 ** (glide * time / dur) * (1.0 + vib * torch.sin(2 * math.pi * _u(b, 4.0, 7.0, device) * time))
        phase = 2 * math.pi * torch.cumsum(freq, dim=-1) / sample_rate + 2 * math.pi * torch.rand(b, 1, device=device)

        tilt = _u(b, 0.5, 3.0, device)
        odd_only = (torch.rand(b, 1, device=device) < 0.2).float()
        tone = torch.zeros(b, t, device=device)
        for k in range(1, harmonics + 1):
            amp = k ** (-tilt) * torch.exp(0.7 * torch.randn(b, 1, device=device))
            if k % 2 == 0:
                amp = amp * (1.0 - 0.9 * odd_only)
            tone += amp * (k * freq < 16000.0) * torch.sin(k * phase)

        start = _u(b, 0.0, 0.6, device) * dur * (torch.rand(b, 1, device=device) < 0.7)
        stop = start + _u(b, 0.1, 1.3, device) * dur
        attack, release = _u(b, 0.002, 0.05, device), _u(b, 0.02, 0.2, device)
        env = ((time - start) / attack).clamp(0.0, 1.0) * ((stop - time) / release).clamp(0.0, 1.0)
        decays = (torch.rand(b, 1, device=device) < 0.4).float()
        env = env * torch.exp(-decays * (time - start).clamp_min(0.0) / _u(b, 0.1, 1.0, device))
        gain = 10.0 ** (_u(b, -12.0, 0.0, device) / 20.0)
        out += on * gain * env * tone / tone.abs().amax(-1, keepdim=True).clamp_min(1e-6)

    x = out[:, None]
    shaped = (torch.rand(b, 1, 1, device=device) < 0.7)
    x = torch.where(shaped, random_eq(x, sample_rate, max_db=24.0, points=16), x)
    peak = x.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    x = x / peak * (0.1 + 0.5 * torch.rand(b, 1, 1, device=device))
    noise = 10.0 ** (-(40.0 + 30.0 * torch.rand(b, 1, 1, device=device)) / 20.0)
    return x + noise * x.abs().amax(-1, keepdim=True) * torch.randn_like(x)
