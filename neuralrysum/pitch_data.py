"""Synthetic, exactly labelled training material for the pitch estimator.

Random harmonic sounds of every kind (sustained, plucked, sung, with missing
fundamental, odd harmonics only, slightly inharmonic), up to six at once, with
glides, vibrato, repeated notes, background noise, room and EQ. Because the
timbre is random in every example, the estimator learns pitch and nothing
about timbre, which is the whole point of the redesign.

Generated on the GPU, batch by batch, so there is no dataset on disk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .augment import random_eq
from .pitch import PitchConfig


def _u(shape, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(shape, device=dev)


def _p(shape, p, dev):
    return (torch.rand(shape, device=dev) < p).float()


@dataclass
class Batch:
    audio: torch.Tensor  # (B, 1, T)
    salience: torch.Tensor  # (B, n_bins, F) targets in [0, 1]
    onset: torch.Tensor  # (B, n_bins, F)
    midi: torch.Tensor  # (B, V, F) pitch of each voice at frame end
    active: torch.Tensor  # (B, V, F) bool
    started: torch.Tensor  # (B, V, F) bool, first frame of a note


@torch.no_grad()
def make_batch(cfg: PitchConfig, b: int, t: int, device, voices: int = 6, harmonics: int = 64) -> Batch:
    sr, hop = cfg.sample_rate, cfg.hop
    fr = t // hop
    dur = t / sr
    dev = device
    tt = torch.arange(t, device=dev) / sr  # (T,)
    V = voices

    # ---- how many voices sound in each example ---------------------------
    probs = torch.tensor([0.05, 0.35, 0.2, 0.15, 0.12, 0.08, 0.05], device=dev)[: V + 1]
    n_voices = torch.multinomial(probs / probs.sum(), b, replacement=True)  # (B,)
    vmask = (torch.arange(V, device=dev)[None] < n_voices[:, None]).float()  # (B, V)

    # ---- two notes per voice at most -------------------------------------
    sh = (b, V)
    on1 = _u(sh, -0.3, 0.8, dev) * dur
    off1 = on1 + _u(sh, 0.08, 1.5, dev)
    has2 = _p(sh, 0.4, dev)
    on2 = off1 + _u(sh, 0.01, 0.4, dev)
    off2 = on2 + _u(sh, 0.08, 1.0, dev)
    on2 = torch.where(has2 > 0, on2, torch.full_like(on2, 1e9))
    off2 = torch.where(has2 > 0, off2, torch.full_like(off2, 1e9))

    def note_pitch():
        base = _u(sh, 24.0, 96.0, dev)
        snapped = base.round() + 0.15 * torch.randn(sh, device=dev)
        return torch.where(_p(sh, 0.7, dev) > 0, snapped, base)

    p1 = note_pitch()
    p2 = torch.where(_p(sh, 0.4, dev) > 0, p1, note_pitch())

    second = (tt[None, None] >= on2[..., None])  # (B, V, T)
    on = torch.where(second, on2[..., None], on1[..., None])
    off = torch.where(second, off2[..., None], off1[..., None])
    pitch0 = torch.where(second, p2[..., None], p1[..., None])
    age = tt[None, None] - on  # seconds since this note's onset (negative before)
    length = (off - on).clamp_min(1e-3)

    glide = _u(sh, -5.0, 5.0, dev) * _p(sh, 0.15, dev)
    vib_depth = _u(sh, 0.1, 0.8, dev) * _p(sh, 0.25, dev)
    vib_rate = _u(sh, 4.0, 7.0, dev)
    midi = (pitch0 + glide[..., None] * (age / length).clamp(0, 1)
            + vib_depth[..., None] * torch.sin(2 * math.pi * vib_rate[..., None] * tt[None, None]))

    # ---- envelope ---------------------------------------------------------
    attack = _u(sh, 0.001, 0.08, dev)[..., None]
    release = _u(sh, 0.01, 0.3, dev)[..., None]
    decays = _p(sh, 0.5, dev)[..., None]
    tau = _u(sh, 0.1, 2.0, dev)[..., None]
    rise = (age / attack).clamp(0, 1)
    sustain = torch.where(decays > 0, torch.exp(-age.clamp_min(0) / tau), torch.ones_like(age))
    fall = torch.where(tt[None, None] < off, torch.ones_like(age),
                       torch.exp(-(tt[None, None] - off) / release * 4.6))
    env = rise * sustain * fall * (age >= 0)
    level = 10.0 ** (_u(sh, -18.0, 0.0, dev) / 20.0) * vmask
    env = env * level[..., None]

    # ---- timbre: random harmonic amplitudes ---------------------------------
    tilt = _u(sh, 0.3, 2.5, dev)
    odd = _p(sh, 0.15, dev)
    nofund = _p(sh, 0.1, dev)
    formant = _p(sh, 0.25, dev)
    fc = 10.0 ** _u((b, V, 3), math.log10(300), math.log10(3500), dev)
    inharm = 10.0 ** _u(sh, -5.0, -3.3, dev) * _p(sh, 0.15, dev)
    hdecay = _u(sh, 0.0, 0.1, dev) * decays[..., 0]
    f0 = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    phase = 2 * math.pi * torch.cumsum(f0, dim=-1) / sr + 2 * math.pi * torch.rand(b, V, 1, device=dev)
    f_nom = 440.0 * 2.0 ** ((p1 - 69.0) / 12.0)  # formants sit at fixed frequencies
    y = torch.zeros(b, V, t, device=dev)
    for k in range(1, harmonics + 1):
        a = k ** (-tilt) * torch.exp(0.5 * torch.randn(sh, device=dev))
        if k % 2 == 0:
            a = a * (1 - 0.95 * odd)
        if k == 1:
            a = a * (1 - 0.97 * nofund)
        res = sum(1.0 / (1.0 + ((k * f_nom - fc[..., j]) / (0.15 * fc[..., j])) ** 2) for j in range(3))
        a = torch.where(formant > 0, a * (0.1 + res), a)
        kf = k * torch.sqrt(1 + inharm * k * k)
        alias = (kf[..., None] * f0 < 0.45 * sr).float()
        hd = torch.exp(-hdecay[..., None] * k * age.clamp_min(0) / tau)
        y += a[..., None] * alias * hd * torch.sin(kf[..., None] * phase)
    y = y / y.abs().amax(-1, keepdim=True).clamp_min(1e-6)  # per voice, before the envelope
    x = (y * env).sum(1, keepdim=True)  # (B, 1, T)

    # ---- labels at frame ends ------------------------------------------------
    idx = torch.arange(fr, device=dev) * hop + hop - 1
    m_f = midi[..., idx]
    note_peak = level[..., None]
    active = (env[..., idx] > 0.01 * note_peak) & (tt[idx][None, None] < off[..., idx]) & (vmask[..., None] > 0)
    first = (age[..., idx] >= 0) & (age[..., idx] < hop / sr * 2.0)
    started = first & active & (on[..., idx] >= 0)

    nb = cfg.n_bins
    pos = (m_f - cfg.midi_lo) * cfg.bins_per_semitone  # (B, V, F)
    grid = torch.arange(nb, device=dev, dtype=torch.float32)[None, None, :, None]
    g = torch.exp(-0.5 * (grid - pos[:, :, None]) ** 2)  # (B, V, nb, F), sigma 1 bin
    in_range = (pos >= 0) & (pos <= nb - 1)
    sal = (g * (active & in_range)[:, :, None]).amax(1)
    ons = (g * (started & in_range)[:, :, None]).amax(1)

    # ---- the world around it ---------------------------------------------------
    peak = x.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    x = x / peak
    room = _p((b, 1, 1), 0.3, dev)
    if room.any():
        n_ir = int(1.5 * sr)
        ti = torch.arange(n_ir, device=dev) / sr
        t60 = _u((b, 1, 1), 0.2, 2.5, dev)
        ir = torch.randn(b, 1, n_ir, device=dev) * torch.exp(-6.9 * ti / t60)
        ir = ir / ir.pow(2).sum(-1, keepdim=True).sqrt()
        n = 1 << math.ceil(math.log2(t + n_ir))
        wet = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(ir, n), n)[..., :t]
        wet = wet / wet.abs().amax(-1, keepdim=True).clamp_min(1e-6)
        mix = _u((b, 1, 1), 0.1, 0.5, dev)
        x = torch.where(room > 0, (1 - mix) * x + mix * wet, x)
    eq = _p((b, 1, 1), 0.5, dev)
    x = torch.where(eq > 0, random_eq(x, sr, max_db=12.0), x)
    noise = random_eq(torch.randn_like(x), sr, max_db=12.0)
    noise = noise / noise.pow(2).mean(-1, keepdim=True).sqrt()
    n_lvl = 10.0 ** (_u((b, 1, 1), -70.0, -30.0, dev) / 20.0) * _p((b, 1, 1), 0.5, dev)
    x = x + n_lvl * noise
    burst = _p((b, 1, 1), 0.1, dev)
    bt = _u((b, 1, 1), 0.0, dur, dev)
    bursts = torch.randn_like(x) * torch.exp(-(tt[None, None] - bt).clamp_min(0) / 0.02) * (tt[None, None] >= bt)
    x = x + burst * _u((b, 1, 1), 0.05, 0.5, dev) * bursts
    gain = 10.0 ** (_u((b, 1, 1), -35.0, -1.0, dev) / 20.0)
    x = x / x.abs().amax(-1, keepdim=True).clamp_min(1e-6) * gain
    return Batch(x, sal, ons, m_f, active, started)
