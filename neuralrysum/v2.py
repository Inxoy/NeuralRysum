"""NeuralRysum v2 end to end: audio or MIDI in, organ out.

    audio --> PitchEstimator --> VoiceTracker --> voices --> OrganSynth --> organ
    MIDI  ------------------------------------> voices --/

    python -m neuralrysum.v2 --pitch runs/pitch/last.pt --synth runs/synth/last.pt \
        --in voice.wav --out organ.wav [--follow] [--ref organ_clip.wav]

Without ``--follow`` the timbre is fixed (Manual): the mean timbre of ``--ref``
or, without it, the centre of the latent space.
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .data import load_mono
from .descriptors import descriptors
from .model import NeuralRysum
from .pitch import PitchConfig, PitchEstimator, track
from .synth import OrganSynth, SynthConfig


def load_pitch(path: str, device="cpu") -> PitchEstimator:
    ck = torch.load(path, map_location=device)
    m = PitchEstimator(PitchConfig.from_dict(ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval().to(device)


def load_synth(path: str, device="cpu") -> OrganSynth:
    ck = torch.load(path, map_location=device)
    m = OrganSynth(SynthConfig.from_dict(ck["config"]))
    m.load_state_dict(ck["model"])
    return m.eval().to(device)


def fold(key: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Move pitches outside [lo, hi) into it by whole octaves."""
    k = key.copy()
    while (m := (k > 0) & (k < lo)).any():
        k[m] += 12
    while (m := k >= hi).any():
        k[m] -= 12
    return k


def tracked_voices(midi, gate, started, est_hop: int, synth_hop: int, frames: int):
    """Tracker output at the estimator hop -> synth voices (V, frames) at the synth hop.

    Synth frame f starts at sample f * synth_hop and may only use estimator
    frames that ended before it.
    """
    v = midi.shape[0]
    ratio = synth_hop // est_hop
    key = np.zeros((v, frames), np.float32)
    g = np.zeros_like(key)
    on_age = np.zeros_like(key)
    off_age = np.full_like(key, -1.0)
    dt = synth_hop / 48000.0
    for i in range(v):
        cur_key, on_t, off_t, have = 0.0, 0.0, -1.0, False
        for f in range(frames):
            k = f * ratio - 1  # last estimator frame that ended before this synth frame
            t = f * dt
            if k >= 0:
                lo = max(0, k - ratio + 1)
                if started[i, lo: k + 1].any():
                    on_t, off_t, have = t, -1.0, True
                if gate[i, k] > 0:
                    cur_key = float(midi[i, k])
                    off_t = -1.0 if have else off_t
                elif have and off_t < 0:
                    off_t = t
            if have:
                key[i, f] = cur_key
                g[i, f] = 1.0 if off_t < 0 else 0.0
                on_age[i, f] = t - on_t
                off_age[i, f] = -1.0 if off_t < 0 else t - off_t
    return key, g, on_age, off_age


class V2:
    """Same calls as v1 where the leak test needs them."""

    def __init__(self, est: PitchEstimator, synth: OrganSynth, key_range=(36.0, 84.0), **track_kw):
        self.est, self.synth = est, synth
        self.cfg = synth.cfg
        self.key_range = key_range
        self.track_kw = track_kw

    @torch.no_grad()
    def timbre(self, x: torch.Tensor):
        """x: (B, 1, T) -> (z, desc) means over the clips, the Manual default."""
        mean, _, _ = self.synth.encode(x)
        d = descriptors(x, self.cfg.slow_hop, self.cfg.sample_rate)
        return mean[:, :, 8:].mean((0, 2)), d[:, :, 8:].mean((0, 2))

    @torch.no_grad()
    def voices(self, x: torch.Tensor):
        midi, gate, started = track(self.est, x, voices=self.cfg.voices, **self.track_kw)
        midi = fold(midi, *self.key_range)
        frames = x.shape[-1] // self.cfg.hop
        return tracked_voices(midi, gate, started, self.est.cfg.hop, self.cfg.hop, frames)

    @torch.no_grad()
    def render_voices(self, key, gate, on_age, off_age, x_follow: Optional[torch.Tensor] = None,
                      z=None, desc=None):
        cfg = self.cfg
        dev = next(self.synth.parameters()).device
        key, gate, on_age, off_age = (torch.as_tensor(a, device=dev)[None] for a in (key, gate, on_age, off_age))
        frames = key.shape[-1]
        if z is not None and desc is not None:
            c = torch.cat([z, desc]).to(dev).view(1, -1, 1).expand(1, -1, frames)
        else:
            mean, _, dp = self.synth.encode(x_follow)
            c_slow = NeuralRysum.delay_one(torch.cat([mean, dp], 1))
            if z is not None:
                c_slow[:, : cfg.z_dim] = z.view(1, -1, 1)
            if desc is not None:
                c_slow[:, cfg.z_dim:] = desc.view(1, -1, 1)
            c = self.synth.hold(c_slow, frames)
        return self.synth.render(key, gate, on_age, off_age, c)

    @torch.no_grad()
    def transfer(self, x: torch.Tensor, z=None, desc=None, noise=None):
        """x: (1, 1, T), T a multiple of the slow hop. z / desc None = Follow."""
        return self.render_voices(*self.voices(x), x_follow=x, z=z, desc=desc)

    @torch.no_grad()
    def midi(self, notes, seconds: float, z, desc):
        """notes: [{on, off, key}] -> organ audio, voices assigned like the dataset."""
        from .organ_data import assign_voices, voice_frames
        frames = int(seconds * self.cfg.sample_rate) // self.cfg.slow_hop * self.cfg.slow_ratio
        ns = assign_voices([{**n, "midi": round(n["key"])} for n in notes], self.cfg.voices,
                           self.cfg.release_hold)
        return self.render_voices(*voice_frames(ns, 0, frames, self.cfg.hop, self.cfg.voices), z=z, desc=desc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--ref", help="organ clip whose mean timbre Manual uses")
    a = ap.parse_args()
    m = V2(load_pitch(a.pitch), load_synth(a.synth))
    sr, hop = m.cfg.sample_rate, m.cfg.slow_hop
    x = load_mono(a.inp, sr)
    x = torch.from_numpy(x[: len(x) // hop * hop])[None, None]
    z = d = None
    if not a.follow:
        if a.ref:
            r = load_mono(a.ref, sr)
            z, d = m.timbre(torch.from_numpy(r[: len(r) // hop * hop])[None, None])
        else:
            z, d = torch.zeros(m.cfg.z_dim), torch.zeros(m.cfg.n_desc)
    y = m.transfer(x, z, d)[0, 0].numpy()
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * 0.99 / peak
    sf.write(a.out, y, sr, subtype="PCM_24")
    print(f"wrote {a.out} ({len(y) / sr:.1f} s, peak {20 * np.log10(peak + 1e-12):.1f} dBFS)")


if __name__ == "__main__":
    main()
