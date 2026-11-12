"""Measure the onset latency of a *trained* model.

    python -m neuralrysum.latency --ckpt runs/rysum/last.pt

The network has zero look-ahead by construction, but a trained network can
still respond late (it may have learned to wait before committing to a note).
This plays tone bursts with sharp onsets through the model and reports how
many milliseconds pass until the output envelope reaches 10 % and 50 % of its
level, relative to the same thresholds on the input. The measurement is good
to about half a millisecond. Add your audio interface's buffer on top.
"""
from __future__ import annotations

import argparse

import torch

from .export import load_checkpoint


def envelope(x: torch.Tensor, win: int = 32) -> torch.Tensor:
    return torch.nn.functional.avg_pool1d(x.abs()[None, None], win, 1, win // 2)[0, 0][: len(x)]


def first_above(env: torch.Tensor, start: int, level: float) -> int:
    idx = (env[start:] >= level).nonzero()
    return int(idx[0]) if len(idx) else -1


@torch.no_grad()
def measure(model, freqs=(110.0, 220.0, 440.0, 880.0), z=None, desc=None):
    cfg = model.cfg
    sr = cfg.sample_rate
    onset, total = cfg.slow_hop * 12, cfg.slow_hop * 24
    z = torch.zeros(cfg.z_dim) if z is None else z
    desc = torch.zeros(cfg.n_desc) if desc is None else desc
    rows = []
    for f in freqs:
        t = torch.arange(total - onset) / sr
        x = torch.zeros(total)
        x[onset:] = 0.3 * torch.sin(2 * torch.pi * f * t)
        y = model.transfer(x[None, None], z, desc)[0, 0]
        ex, ey = envelope(x), envelope(y)
        steady_x, steady_y = ex[-sr // 10 :].mean(), ey[-sr // 10 :].mean()
        floor = ey[onset - sr // 20 : onset].mean()
        row = {"freq": f}
        for frac in (0.1, 0.5):
            ix = first_above(ex, onset, frac * steady_x)
            iy = first_above(ey, onset, floor + frac * (steady_y - floor))
            row[f"t{int(frac * 100)}_ms"] = float("nan") if iy < 0 else 1000.0 * (iy - ix) / sr
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", required=True)
    a = p.parse_args()
    model = load_checkpoint(a.ckpt)
    cfg = model.cfg
    print(f"frame: {cfg.hop} samples = {1000 * cfg.hop / cfg.sample_rate:.2f} ms (upper bound of the buffering delay)")
    print("onset delay through the model, output vs input:")
    for r in measure(model):
        print(f"  {r['freq']:7.1f} Hz   10 %: {r['t10_ms']:6.2f} ms   50 %: {r['t50_ms']:6.2f} ms")


if __name__ == "__main__":
    main()
