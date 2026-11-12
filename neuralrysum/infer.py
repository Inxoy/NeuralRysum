"""Audition a checkpoint offline, without the plugin.

    # timbre transfer, timbre follows the input file
    python -m neuralrysum.infer --ckpt runs/rysum/last.pt --input guitar.wav --out out.wav

    # timbre transfer with a fixed latent and descriptors
    python -m neuralrysum.infer --ckpt ... --input guitar.wav --out out.wav \
        --z 1.5 0 0 0 0 0 0 0 --desc 0.3 -0.5 0

    # no input: 20 s of free generation from the prior
    python -m neuralrysum.infer --ckpt ... --free 20 --out free.wav --temperature 0.8
"""
from __future__ import annotations

import argparse

import soundfile as sf
import torch

from .data import load_mono
from .export import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input", default=None)
    p.add_argument("--free", type=float, default=0.0, help="seconds of free generation")
    p.add_argument("--z", type=float, nargs="*", default=None)
    p.add_argument("--desc", type=float, nargs="*", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    a = p.parse_args()

    model = load_checkpoint(a.ckpt)
    cfg = model.cfg
    z = torch.tensor(a.z) if a.z else None
    desc = torch.tensor(a.desc) if a.desc else None
    if z is not None:
        assert len(z) == cfg.z_dim, f"--z needs {cfg.z_dim} values"
    if desc is not None:
        assert len(desc) == cfg.n_desc, f"--desc needs {cfg.n_desc} values"

    if a.free > 0:
        frames = int(a.free * cfg.sample_rate / cfg.hop)
        c = torch.cat([z if z is not None else torch.zeros(cfg.z_dim),
                       desc if desc is not None else torch.zeros(cfg.n_desc)])
        y = model.free(c.view(1, -1, 1).expand(1, -1, frames).contiguous(), a.temperature)
    else:
        assert a.input, "give --input or --free"
        x = torch.from_numpy(load_mono(a.input, cfg.sample_rate))
        x = x[: len(x) // cfg.slow_hop * cfg.slow_hop]
        y = model.transfer(x[None, None], z, desc)
    sf.write(a.out, y[0, 0].numpy(), cfg.sample_rate)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
