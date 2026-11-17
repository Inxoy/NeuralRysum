"""Polyphonic pitch estimator: audio in, "which pitches sound, how loud" out.

This replaces the learned fast code of v1. Its output carries no waveform and
almost no timbre, so the synthesis model behind it cannot copy the input.

    audio --> causal multi-window spectrum on a log-frequency grid (fixed)
          --> harmonic stacking (h = 1/2, 1, 2 ... 7)                (fixed)
          --> small causal 2-D conv net                              (learned)
          --> salience + onset, 3 bins per semitone, every 64 samples

Each frame only uses samples up to its own last sample. Low notes need longer
windows to be told apart, so their detection takes longer (by design: real
bass pipes are slow to speak too); onsets and loudness react within a frame.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PitchConfig:
    sample_rate: int = 48000
    hop: int = 64
    bins_per_semitone: int = 3
    midi_lo: int = 21  # lowest output pitch (A0)
    midi_hi: int = 108  # highest output pitch (C8)
    windows: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    harmonics: List[float] = field(default_factory=lambda: [0.5, 1, 2, 3, 4, 5, 6, 7])
    channels: int = 16

    @property
    def n_bins(self) -> int:
        return (self.midi_hi - self.midi_lo) * self.bins_per_semitone + 1

    def bin_midi(self) -> np.ndarray:
        return self.midi_lo + np.arange(self.n_bins) / self.bins_per_semitone

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "PitchConfig":
        return PitchConfig(**{k: v for k, v in d.items() if k in PitchConfig.__dataclass_fields__})


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


class LogSpectrum(nn.Module):
    """Causal log-frequency magnitude spectrum, one frame per hop. No parameters.

    Every log bin takes its value from the shortest window that still resolves
    a semitone there (linear interpolation between FFT bins).
    """

    def __init__(self, cfg: PitchConfig):
        super().__init__()
        self.cfg = cfg
        bps = cfg.bins_per_semitone
        shifts = [round(12 * bps * math.log2(h)) for h in cfg.harmonics]
        self.lo_pad = -min(shifts)
        self.shifts = shifts
        n_in = cfg.n_bins + self.lo_pad + max(shifts)
        midi = cfg.midi_lo + (np.arange(n_in) - self.lo_pad) / bps
        f = midi_to_hz(midi)
        sr = cfg.sample_rate
        need = sr / (0.5 * f * (2 ** (1 / 12) - 1))  # window length that resolves a semitone
        choice = np.full(n_in, len(cfg.windows) - 1)
        for i in range(n_in):
            for j, n in enumerate(cfg.windows):
                if n >= need[i]:
                    choice[i] = j
                    break
        self.n_in = n_in
        valid = f < 0.47 * sr
        for j, n in enumerate(cfg.windows):
            pos = f * n / sr
            lo = np.clip(np.floor(pos), 0, n // 2 - 1).astype(np.int64)
            frac = np.clip(pos - lo, 0.0, 1.0)
            sel = (choice == j) & valid
            self.register_buffer(f"idx{j}", torch.from_numpy(lo[sel]))
            self.register_buffer(f"frac{j}", torch.from_numpy(frac[sel]).float())
            self.register_buffer(f"dst{j}", torch.from_numpy(np.nonzero(sel)[0]))
            self.register_buffer(f"win{j}", torch.hann_window(n, periodic=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T), T a multiple of hop -> (B, n_in, T / hop), level in [0, 1]."""
        b, _, t = x.shape
        hop = self.cfg.hop
        out = x.new_zeros(b, self.n_in, t // hop)
        for j, n in enumerate(self.cfg.windows):
            if len(getattr(self, f"dst{j}")) == 0:
                continue
            win = getattr(self, f"win{j}")
            xp = F.pad(x[:, 0], (n - hop, 0))  # frame k ends at sample (k + 1) * hop - 1
            s = torch.stft(xp, n, hop_length=hop, window=win, center=False, return_complex=True)
            mag = s.abs()[:, :, : t // hop] * (2.0 / win.sum())  # full-scale sine -> 1
            idx, frac = getattr(self, f"idx{j}"), getattr(self, f"frac{j}")
            v = mag[:, idx] * (1 - frac)[None, :, None] + mag[:, idx + 1] * frac[None, :, None]
            out[:, getattr(self, f"dst{j}")] = v
        db = 20.0 * torch.log10(out + 1e-5)
        return ((db + 100.0) / 100.0).clamp(0.0, 1.0)  # -100 dBFS .. 0 dBFS -> 0 .. 1


class CausalConv2d(nn.Module):
    """(B, C, freq, time): causal in time, centred in frequency."""

    def __init__(self, cin, cout, kt, kf, dt=1, df=1):
        super().__init__()
        self.pt = (kt - 1) * dt
        self.pf = (kf - 1) * df // 2
        self.conv = nn.Conv2d(cin, cout, (kf, kt), dilation=(df, dt))

    def forward(self, x):
        return self.conv(F.pad(x, (self.pt, 0, self.pf, self.pf)))


class PitchEstimator(nn.Module):
    def __init__(self, cfg: PitchConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or PitchConfig()
        self.spec = LogSpectrum(cfg)
        c, h = cfg.channels, len(cfg.harmonics)
        self.l1 = CausalConv2d(h, c, 3, 5)
        self.l2 = CausalConv2d(c, c, 3, 3, dt=2)
        self.l3 = CausalConv2d(c, c, 3, 3, dt=4)
        self.l4 = CausalConv2d(c, c // 2, 1, 13, df=3)  # one octave in semitone steps
        self.out = nn.Conv2d(c // 2 + c, 2, 1)

    def stack(self, s: torch.Tensor) -> torch.Tensor:
        """(B, n_in, F) -> (B, harmonics, n_bins, F)."""
        n, lo = self.cfg.n_bins, self.spec.lo_pad
        return torch.stack([s[:, lo + sh: lo + sh + n] for sh in self.spec.shifts], dim=1)

    def forward(self, x: torch.Tensor):
        """x: (B, 1, T) -> salience logits, onset logits, each (B, n_bins, T / hop)."""
        h = self.stack(self.spec(x))
        a = F.leaky_relu(self.l1(h), 0.2)
        a = a + F.leaky_relu(self.l2(a), 0.2)
        a = a + F.leaky_relu(self.l3(a), 0.2)
        b = F.leaky_relu(self.l4(a), 0.2)
        o = self.out(torch.cat([a, b], 1))
        return o[:, 0], o[:, 1]


# ---- decoding ---------------------------------------------------------------
def peaks(sal: np.ndarray, bin_midi: np.ndarray, thresh: float):
    """sal: (n_bins,) probabilities -> list of (midi, strength) at local maxima above thresh."""
    out = []
    for i in range(1, len(sal) - 1):
        if sal[i] >= thresh and sal[i] >= sal[i - 1] and sal[i] > sal[i + 1]:
            w = sal[i - 1: i + 2]
            m = float((w * bin_midi[i - 1: i + 2]).sum() / w.sum())
            out.append((m, float(sal[i])))
    return out


class VoiceTracker:
    """Turns salience / onset frames into up to ``voices`` continuous voices.

    Same logic as the plugin will run: a peak above ``on`` that lasts
    ``confirm`` frames starts a voice, a voice continues while a peak above
    ``off`` lies within ``match`` semitones, an onset at its pitch restarts it,
    and it is released after ``hold`` frames without a peak.
    """

    def __init__(self, bin_midi, voices=8, on=0.6, off=0.3, match=0.7, hold=24, onset=0.5, confirm=4):
        self.bin_midi = bin_midi
        self.n = voices
        self.on, self.off, self.match, self.hold, self.onset_th = on, off, match, hold, onset
        self.confirm = confirm  # frames a new peak must last before it starts a voice
        self.midi = np.zeros(voices)
        self.active = np.zeros(voices, bool)
        self.missing = np.zeros(voices, int)
        self.age = np.zeros(voices, int)  # frames since (re)start
        self.pending = []  # (midi, frames seen) of peaks that have not started a voice yet

    def step(self, sal: np.ndarray, ons: np.ndarray):
        """-> (midi, gate, started) arrays for this frame."""
        started = np.zeros(self.n, bool)
        cand = peaks(sal, self.bin_midi, self.off)
        used = set()
        for v in np.argsort(-self.age):  # oldest voices claim their peak first
            if not self.active[v]:
                continue
            best, bd = None, self.match
            for j, (m, _) in enumerate(cand):
                if j not in used and abs(m - self.midi[v]) < bd:
                    best, bd = j, abs(m - self.midi[v])
            if best is None:
                self.missing[v] += 1
                if self.missing[v] > self.hold:
                    self.active[v] = False
                continue
            used.add(best)
            m = cand[best][0]
            self.midi[v], self.missing[v] = m, 0
            k = int(round((m - self.bin_midi[0]) * 3))
            if 0 <= k < len(ons) and ons[k] > self.onset_th and self.age[v] > 6:
                self.age[v] = 0
                started[v] = True
            else:
                self.age[v] += 1
        pending = []
        for j, (m, s) in enumerate(cand):
            if j in used or s < self.on:
                continue
            seen = 1 + max([c for pm, c in self.pending if abs(pm - m) < self.match], default=0)
            free = np.nonzero(~self.active)[0]
            if seen < self.confirm or len(free) == 0:
                pending.append((m, seen))
                continue
            v = free[0]
            self.active[v], self.midi[v], self.missing[v], self.age[v] = True, m, 0, 0
            started[v] = True
        self.pending = pending
        # a voice stays gated through short gaps; it is released only once ``hold`` runs out
        return self.midi.copy(), self.active.astype(np.float32), started


@torch.no_grad()
def track(est: PitchEstimator, x: torch.Tensor, voices: int = 8, **kw):
    """x: (1, 1, T) -> midi (V, F), gate (V, F), started (V, F) at the estimator's hop."""
    sal, ons = est(x)
    sal, ons = torch.sigmoid(sal)[0].cpu().numpy(), torch.sigmoid(ons)[0].cpu().numpy()
    tr = VoiceTracker(est.cfg.bin_midi(), voices, **kw)
    out = [tr.step(sal[:, f], ons[:, f]) for f in range(sal.shape[1])]
    midi, gate, st = (np.stack(a, axis=1) for a in zip(*out))
    return midi, gate, st
