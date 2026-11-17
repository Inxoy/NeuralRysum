"""Timbre perturbation for the fast-path input.

The fast code should carry *what is played* (pitch, phase, envelope) and not
*what it sounds like*. During training the fast encoder therefore hears a copy
of the target whose spectrum has been bent at random, while the decoder still
has to reproduce the clean target. Whatever survives the bending is what the
fast code learns to rely on. This is also what lets a guitar or the plugin's
plain MIDI oscillators drive a model trained only on organ.

The EQ is zero-phase (applied in the FFT domain), so it does not move onsets.
"""
from __future__ import annotations

import math

import torch


def random_eq(x: torch.Tensor, sample_rate: int, max_db: float = 12.0, points: int = 8):
    """x: (B, 1, T). Random smooth magnitude response, per example."""
    b, _, t = x.shape
    spec = torch.fft.rfft(x, dim=-1)
    n = spec.shape[-1]
    freqs = torch.linspace(0.0, sample_rate / 2, n, device=x.device)
    # control points evenly spaced on a log-frequency axis, 40 Hz .. Nyquist
    pos = (torch.log2(freqs.clamp_min(40.0) / 40.0) / math.log2(sample_rate / 2 / 40.0)) * (
        points - 1
    )
    lo = pos.floor().clamp(0, points - 2).long()
    frac = (pos - lo).clamp(0.0, 1.0)
    gains = (torch.rand(b, points, device=x.device) * 2 - 1) * max_db
    db = gains[:, lo] * (1 - frac) + gains[:, lo + 1] * frac
    db = db - db.mean(-1, keepdim=True)  # keep overall level roughly unchanged
    return torch.fft.irfft(spec * (10.0 ** (db / 20.0))[:, None, :], n=t, dim=-1)


def random_drive(x: torch.Tensor, max_drive: float = 4.0):
    """Soft saturation with level compensation, per example."""
    drive = 1.0 + torch.rand(x.shape[0], 1, 1, device=x.device) * (max_drive - 1.0)
    y = torch.tanh(x * drive)
    rms_in = x.pow(2).mean(-1, keepdim=True).sqrt()
    rms_out = y.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return y * rms_in / rms_out


def random_shaper(x: torch.Tensor) -> torch.Tensor:
    """A different waveshaper per example: none, saturation, asymmetric, power law.

    Each keeps the period and roughly the envelope but redistributes the
    harmonics (the asymmetric one adds even harmonics, the others odd ones).
    """
    b = x.shape[0]
    dev = x.device
    rms_in = x.pow(2).mean(-1, keepdim=True).sqrt()
    xn = x / x.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    drive = 10.0 ** torch.rand(b, 1, 1, device=dev)  # 1 .. 10
    sat = torch.tanh(xn * drive)
    asym = torch.maximum(xn, torch.rand(b, 1, 1, device=dev) * 0.5 * xn)
    power = xn.sign() * xn.abs().pow(0.3 + 0.7 * torch.rand(b, 1, 1, device=dev))
    kind = torch.randint(0, 4, (b, 1, 1), device=dev)
    y = torch.where(kind == 1, sat, torch.where(kind == 2, asym, torch.where(kind == 3, power, xn)))
    y = y - y.mean(-1, keepdim=True)
    return y * rms_in / y.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)


def perturb_strong(x: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """EQ, waveshaper, EQ again: twice the range and resolution of the mild version."""
    y = random_eq(x, sample_rate, max_db=24.0, points=16)
    y = random_shaper(y)
    y = random_eq(y, sample_rate, max_db=24.0, points=16)
    noise_level = 10.0 ** (-(40.0 + 30.0 * torch.rand(x.shape[0], 1, 1, device=x.device)) / 20)
    return y + noise_level * torch.randn_like(y)


def perturb_timbre(x: torch.Tensor, sample_rate: int, strength: str = "mild") -> torch.Tensor:
    if strength == "strong":
        return perturb_strong(x, sample_rate)
    assert strength == "mild", strength
    y = random_eq(x, sample_rate)
    half = torch.rand(x.shape[0], 1, 1, device=x.device) < 0.5
    y = torch.where(half, random_drive(y), y)
    noise_level = 10.0 ** (-(40.0 + 30.0 * torch.rand(x.shape[0], 1, 1, device=x.device)) / 20)
    return y + noise_level * torch.randn_like(y)
