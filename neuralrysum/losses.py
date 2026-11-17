"""Reconstruction and adversarial losses."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Floor of the log-magnitude term. Far below this, "silence" stops counting.
# With a much smaller floor, noise-free material (synthetic tones, digital
# silence) makes any broadband output so expensive that the model learns to
# output nothing at all.
LOG_EPS = 1e-3


def _stft_mag(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=x.device)
    s = torch.stft(x, n_fft, hop_length=n_fft // 4, window=win, return_complex=True)
    return s.abs()


def multires_stft_loss(y: torch.Tensor, x: torch.Tensor, ffts=(2048, 1024, 512, 256, 128)):
    """Spectral convergence + log-magnitude L1, summed over resolutions.

    The short windows (128, 256) are what punish smeared or late onsets.
    """
    y, x = y.squeeze(1), x.squeeze(1)
    loss = 0.0
    for n in ffts:
        my, mx = _stft_mag(y, n), _stft_mag(x, n)
        sc = torch.linalg.norm(mx - my) / torch.linalg.norm(mx).clamp_min(1e-6)
        lm = F.l1_loss(torch.log(my + LOG_EPS), torch.log(mx + LOG_EPS))
        loss = loss + sc + lm
    return loss / len(ffts)


def fine_structure_loss(y: torch.Tensor, x: torch.Tensor, sample_rate: int, n_fft: int = 2048,
                        fmax: float = 4000.0, smooth_hz: float = 400.0, eps: float = 0.1):
    """Where the partials are, regardless of how strong each one is.

    The log spectrum minus its own average over ``smooth_hz`` keeps the peaks
    and valleys (pitch, chords) and drops the spectral envelope (timbre), so
    this ties the notes of ``y`` to those of ``x`` and leaves the timbre free.
    """
    def whitened(s: torch.Tensor) -> torch.Tensor:
        m = torch.log(_stft_mag(s.squeeze(1), n_fft) + eps)[:, : int(fmax * n_fft / sample_rate)]
        k = int(smooth_hz * n_fft / sample_rate) // 2 * 2 + 1
        m = m.transpose(1, 2)  # (B, frames, bins)
        avg = F.avg_pool1d(F.pad(m, (k // 2, k // 2), mode="replicate"), k, stride=1)
        return m - avg

    return F.l1_loss(whitened(y), whitened(x))


def loudness_loss(y: torch.Tensor, x: torch.Tensor, frame: int = 1024, eps: float = 1e-3):
    """L1 between log RMS envelopes (hop = frame / 2)."""
    def env(s: torch.Tensor) -> torch.Tensor:
        return torch.log(F.avg_pool1d(s.pow(2), frame, stride=frame // 2).sqrt() + eps)

    return F.l1_loss(env(y), env(x))


class SpecDiscriminator(nn.Module):
    """Patch discriminator on one complex STFT resolution."""

    def __init__(self, n_fft: int, ch: int = 32):
        super().__init__()
        self.n_fft = n_fft
        wn = nn.utils.parametrizations.weight_norm
        self.layers = nn.ModuleList(
            [
                wn(nn.Conv2d(2, ch, (3, 9), padding=(1, 4))),
                wn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
                wn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
                wn(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
                wn(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1))),
            ]
        )
        self.out = wn(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor):
        win = torch.hann_window(self.n_fft, device=x.device)
        s = torch.stft(
            x.squeeze(1), self.n_fft, hop_length=self.n_fft // 4, window=win, return_complex=True
        )
        h = torch.view_as_real(s).permute(0, 3, 2, 1)  # (B, 2, frames, bins)
        feats = []
        for layer in self.layers:
            h = F.leaky_relu(layer(h), 0.2)
            feats.append(h)
        return self.out(h), feats


class MultiResDiscriminator(nn.Module):
    def __init__(self, ffts=(2048, 1024, 512, 128)):
        super().__init__()
        self.ds = nn.ModuleList(SpecDiscriminator(n) for n in ffts)

    def forward(self, x: torch.Tensor):
        return [d(x) for d in self.ds]


def d_loss(real: List, fake: List) -> torch.Tensor:
    loss = 0.0
    for (r, _), (f, _) in zip(real, fake):
        loss = loss + F.relu(1 - r).mean() + F.relu(1 + f).mean()
    return loss / len(real)


def g_adv_loss(fake: List) -> torch.Tensor:
    return sum(-f.mean() for f, _ in fake) / len(fake)


def feature_matching(real: List, fake: List) -> torch.Tensor:
    loss, n = 0.0, 0
    for (_, fr), (_, ff) in zip(real, fake):
        for a, b in zip(fr, ff):
            loss = loss + F.l1_loss(b, a.detach())
            n += 1
    return loss / n
