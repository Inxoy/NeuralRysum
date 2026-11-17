"""NeuralRysum v2: organ synthesis from voices and timbre.

The model never sees the input waveform. Each *voice* is just a key (any
fractional MIDI number, so any tuning works), a gate and the time since its
onset / release. From that plus the timbre conditioning a shared network
predicts, for every voice and every frame,

- the amplitudes of up to ``table // 2 - 1`` harmonics of key / 2, which
  covers 16', 8', 4', 2 2/3', 2', 1 3/5', mixtures: all organ ranks sit on
  harmonics of the 16' pitch;
- the gains of ``noise_bands`` bands of noise (wind, chiff).

The harmonics are played from a wavetable per voice and frame, so the cost
per sample does not depend on how many harmonics there are. A learned room
impulse response follows. Every step is causal: frame ``f`` only uses the
voices as they are at the start of the frame, and the sound crossfades from
frame ``f - 1`` to frame ``f`` over one hop.

The timbre path is v1's slow encoder: latent ``z`` plus three descriptors,
one slow frame late. Now it is the *only* place timbre can come from.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import FastEncoder, NeuralRysum, SlowEncoder


@dataclass
class SynthConfig:
    sample_rate: int = 48000
    hop: int = 128  # control rate of the synth (2.7 ms)
    table: int = 1024  # wavetable length: up to 511 harmonics of key / 2
    voices: int = 8
    key_lo: int = 21
    key_hi: int = 108
    key_emb: int = 16  # learned per-key features (every pipe is its own)
    hidden: int = 192
    noise_bands: int = 32
    ir_seconds: float = 2.0
    z_dim: int = 8
    n_desc: int = 3
    slow_hop: int = 2048
    release_hold: float = 0.5  # a voice keeps sounding this long after its key is released

    @property
    def n_harm(self) -> int:
        return self.table // 2 - 1

    @property
    def cond_dim(self) -> int:
        return self.z_dim + self.n_desc

    @property
    def slow_ratio(self) -> int:
        return self.slow_hop // self.hop

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SynthConfig":
        return SynthConfig(**{k: v for k, v in d.items() if k in SynthConfig.__dataclass_fields__})


ON_TAUS = (0.01, 0.03, 0.1, 0.3, 1.0)
OFF_TAUS = (0.01, 0.03, 0.1, 0.3)


class VoiceNet(nn.Module):
    """Per voice and frame: features -> harmonic log-amplitudes and noise log-gains."""

    def __init__(self, cfg: SynthConfig):
        super().__init__()
        self.cfg = cfg
        n_keys = cfg.key_hi - cfg.key_lo + 1
        self.emb = nn.Parameter(torch.zeros(n_keys, cfg.key_emb))
        d_in = cfg.key_emb + 1 + 8 + 1 + len(ON_TAUS) + len(OFF_TAUS) + 1 + cfg.cond_dim
        self.l1 = nn.Linear(d_in, cfg.hidden)
        self.l2 = nn.Linear(cfg.hidden, cfg.hidden)
        self.l3 = nn.Linear(cfg.hidden, cfg.hidden)
        self.out = nn.Linear(cfg.hidden, cfg.n_harm + cfg.noise_bands)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.constant_(self.out.bias, -9.0)  # start quiet

    def features(self, key, gate, on_age, off_age, cond):
        """key, gate, on_age, off_age: (B, V, F); cond: (B, C, F) -> (B, V, F, D)."""
        cfg = self.cfg
        pos = (key - cfg.key_lo).clamp(0, cfg.key_hi - cfg.key_lo)
        lo = pos.floor().clamp(max=cfg.key_hi - cfg.key_lo - 1).long()
        fr = (pos - lo)[..., None]
        emb = self.emb[lo] * (1 - fr) + self.emb[lo + 1] * fr
        p = ((key - 60.0) / 24.0)[..., None]
        four = torch.cat([torch.sin(math.pi * p * 2 ** i) for i in range(4)]
                         + [torch.cos(math.pi * p * 2 ** i) for i in range(4)], -1)
        on_f = torch.stack([torch.exp(-on_age.clamp_min(0) / t) for t in ON_TAUS], -1)
        released = (off_age >= 0).float()
        off_f = torch.stack([torch.exp(-off_age.clamp_min(0) / t) * released for t in OFF_TAUS], -1)
        c = cond.transpose(1, 2)[:, None].expand(-1, key.shape[1], -1, -1)
        return torch.cat([emb, p, four, gate[..., None], on_f, off_f, released[..., None], c], -1)

    def forward(self, feats):
        h = F.leaky_relu(self.l1(feats), 0.2)
        h = h + F.leaky_relu(self.l2(h), 0.2)
        h = h + F.leaky_relu(self.l3(h), 0.2)
        o = self.out(h)
        return o[..., : self.cfg.n_harm], o[..., self.cfg.n_harm:]


def band_noise(b: int, t: int, sr: int, bands: int, device, generator=None) -> torch.Tensor:
    """White noise split into log-spaced bands (50 Hz .. 22 kHz) -> (B, bands, T).

    Zero-phase FFT filtering is fine here: the noise is random, so a causal
    filter bank (as in the plugin) gives the same statistics.
    """
    w = torch.randn(b, t, device=device, generator=generator)
    s = torch.fft.rfft(w)
    f = torch.fft.rfftfreq(t, 1 / sr).to(device)
    edges = 50.0 * (22000.0 / 50.0) ** (torch.arange(bands + 1, device=device) / bands)
    out = []
    for i in range(bands):
        m = ((f >= edges[i]) & (f < edges[i + 1])).float()
        out.append(torch.fft.irfft(s * m, n=t))
    out = torch.stack(out, 1)
    return out / out.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8) / math.sqrt(bands)


class OrganSynth(nn.Module):
    def __init__(self, cfg: Optional[SynthConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or SynthConfig()
        enc_cfg = ModelConfig(z_dim=cfg.z_dim, n_desc=cfg.n_desc)
        assert enc_cfg.slow_hop == cfg.slow_hop and cfg.slow_hop % cfg.hop == 0
        self.enc_cfg = enc_cfg
        self.trunk = FastEncoder(enc_cfg)
        self.slow = SlowEncoder(enc_cfg)
        self.voice = VoiceNet(cfg)
        n_ir = int(cfg.ir_seconds * cfg.sample_rate)
        t = torch.arange(n_ir) / cfg.sample_rate
        g = torch.Generator().manual_seed(0)
        self.ir = nn.Parameter(0.01 * torch.randn(n_ir, generator=g) * torch.exp(-t / 0.5))
        self.register_buffer("phases", 2 * math.pi * torch.rand(cfg.n_harm, generator=g))
        # NeuralRysum's encoder init (variance preserving) for the trunk / slow encoder
        tmp = NeuralRysum(enc_cfg)
        self.trunk.load_state_dict(tmp.fast.state_dict())
        self.slow.load_state_dict(tmp.slow.state_dict())

    # ---- timbre --------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        """x: (B, 1, T) -> mean, logvar, desc_pred at the slow rate."""
        return self.slow(self.trunk.trunk(x))

    def hold(self, c_slow: torch.Tensor, frames: int) -> torch.Tensor:
        c = c_slow.repeat_interleave(self.cfg.slow_ratio, dim=-1)
        return c[..., :frames]

    # ---- sound -----------------------------------------------------------------
    def render(self, key, gate, on_age, off_age, cond, noise=None, dry_only=False):
        """key etc.: (B, V, F) at hop, cond: (B, C, F) -> audio (B, 1, F * hop).

        ``key <= 0`` marks an unused voice. ``noise`` (B, bands, T) or None.
        """
        cfg = self.cfg
        b, v, fr = key.shape
        hop, n = cfg.hop, cfg.table
        t = fr * hop
        sr = cfg.sample_rate
        used = key > 0
        key = torch.where(used, key, torch.full_like(key, 60.0))
        log_a, log_n = self.voice(self.voice.features(key, gate, on_age, off_age, cond))
        alive = (used & ((gate > 0) | ((off_age >= 0) & (off_age < cfg.release_hold)))).float()

        f_base = 440.0 * 2.0 ** ((key - 69.0) / 12.0) / 2.0  # 16' pitch
        k = torch.arange(1, cfg.n_harm + 1, device=key.device, dtype=key.dtype)
        no_alias = (k * f_base[..., None] < 0.45 * sr).float()
        amp = torch.exp(log_a.clamp(-30, 4)) * no_alias * alias_fade(k * f_base[..., None], sr) \
            * alive[..., None]
        spec = torch.zeros(b, v, fr, n // 2 + 1, dtype=torch.complex64, device=key.device)
        spec[..., 1: cfg.n_harm + 1] = torch.polar(amp * (n / 2), self.phases - math.pi / 2)
        tables = torch.fft.irfft(spec, n=n)  # (B, V, F, N)
        tables = F.pad(tables, (0, 0, 1, 0))  # frame -1 is silent

        # frame f: crossfade table f-1 -> f; pitch glides f-1 -> f
        fb = F.pad(f_base, (1, 0), value=0.0)
        fb[..., 0] = fb[..., 1]
        w = (torch.arange(hop, device=key.device, dtype=key.dtype) / hop)  # (hop,)
        f_smp = (fb[..., :-1, None] * (1 - w) + fb[..., 1:, None] * w).reshape(b, v, t)
        cyc = torch.cumsum(f_smp.double() / sr, -1)
        cyc = (cyc - cyc.floor()).float() * n
        i0 = cyc.floor().long() % n
        a = cyc - cyc.floor()
        i1 = (i0 + 1) % n
        frame = torch.arange(fr, device=key.device).repeat_interleave(hop)[None, None].expand(b, v, -1)
        flat = tables.reshape(b, v, (fr + 1) * n)
        def look(fidx, idx):
            return flat.gather(-1, fidx * n + idx)
        prev = look(frame, i0) * (1 - a) + look(frame, i1) * a
        cur = look(frame + 1, i0) * (1 - a) + look(frame + 1, i1) * a
        ww = w.repeat(fr)[None, None]
        y = (prev * (1 - ww) + cur * ww).sum(1, keepdim=True)

        # noise
        if noise is None:
            noise = band_noise(b, t, sr, cfg.noise_bands, key.device)
        g = (torch.exp(log_n.clamp(-30, 4)) * alive[..., None]).sum(1).transpose(1, 2)  # (B, bands, F)
        g = F.pad(g, (1, 0))
        g_smp = (g[..., :-1, None] * (1 - w) + g[..., 1:, None] * w).reshape(b, cfg.noise_bands, t)
        y = y + (noise[..., :t] * g_smp).sum(1, keepdim=True)
        if dry_only:
            return y
        return self.reverb(y)

    def reverb(self, y: torch.Tensor) -> torch.Tensor:
        t = y.shape[-1]
        n = 1 << math.ceil(math.log2(t + len(self.ir)))
        wet = torch.fft.irfft(torch.fft.rfft(y, n) * torch.fft.rfft(self.ir, n), n)[..., :t]
        return y + wet

    # ---- training ---------------------------------------------------------------
    def forward(self, x, desc, key, gate, on_age, off_age, noise=None):
        """x: (B, 1, T) target, desc: (B, n_desc, T / slow_hop), voices at hop."""
        mean, logvar, desc_pred = self.encode(x)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        c = self.hold(NeuralRysum.delay_one(torch.cat([z, desc], 1)), key.shape[-1])
        y = self.render(key, gate, on_age, off_age, c, noise)
        kl = 0.5 * (mean ** 2 + logvar.exp() - 1.0 - logvar).mean()
        return {"y": y, "kl": kl, "desc_pred": desc_pred, "mean": mean}


def alias_fade(f: torch.Tensor, sr: int) -> torch.Tensor:
    """Fade harmonics out between 0.4 and 0.45 of the sample rate."""
    return ((0.45 * sr - f) / (0.05 * sr)).clamp(0.0, 1.0)
