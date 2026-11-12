"""Export a checkpoint to the ``.nrys`` file the plugin loads.

    python -m neuralrysum.export --ckpt runs/rysum/last.pt --out rysum.nrys --data path/to/wavs

With ``--data`` the timbre latent of the whole dataset is analysed and its
principal axes are stored, so the plugin's XY pad moves along the directions
in which your recordings actually differ (PC1 / PC2), in units of standard
deviations. Without it the pad moves along raw latent dimensions.

File layout (little endian)::

    "NRYS0001"            8 bytes
    header length         uint32
    header                ASCII lines:  cfg <key> <ints...>
                                        tensor <name> <offset> <ndim> <dims...>
    data                  float32, tensors back to back; offsets count floats
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import ModelConfig
from .data import find_audio, load_mono
from .model import NeuralRysum

MAGIC = b"NRYS0001"


def load_checkpoint(path: str, device: str = "cpu") -> NeuralRysum:
    ck = torch.load(path, map_location=device)
    model = NeuralRysum(ModelConfig.from_dict(ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval().to(device)


@torch.no_grad()
def latent_map(model: NeuralRysum, data: str, max_minutes: float = 30.0) -> Dict[str, torch.Tensor]:
    """Mean and scaled principal axes of the slow latent over a dataset."""
    cfg = model.cfg
    chunk = cfg.slow_hop * 256  # ~11 s
    budget = int(max_minutes * 60 * cfg.sample_rate)
    zs = []
    for f in find_audio(data):
        audio = torch.from_numpy(load_mono(f, cfg.sample_rate))
        for s in range(0, len(audio), chunk):
            x = audio[s : s + chunk]
            x = x[: len(x) // cfg.slow_hop * cfg.slow_hop]  # last chunk may be short
            if len(x) < 16 * cfg.slow_hop or x.pow(2).mean().sqrt() < 1e-3:
                continue
            _, h = model.fast(x[None, None])
            mean, _, _ = model.slow(h)
            zs.append(mean[0, :, 8:].T)  # skip the GRU warm-up frames
            budget -= len(x)
            if budget <= 0:
                break
        if budget <= 0:
            break
    if not zs:
        raise RuntimeError("no usable audio found for the latent map")
    z = torch.cat(zs)
    mu = z.mean(0)
    _, s, vt = torch.linalg.svd(z - mu, full_matrices=False)
    std = s / max(1, len(z) - 1) ** 0.5
    return {"z_mean": mu, "z_basis": vt * std[:, None]}  # row i = std_i * pc_i


def export(model: NeuralRysum, out: str, lmap: Optional[Dict[str, torch.Tensor]] = None) -> None:
    cfg = model.cfg
    if lmap is None:
        lmap = {"z_mean": torch.zeros(cfg.z_dim), "z_basis": torch.eye(cfg.z_dim)}
    tensors = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    tensors.update({k: v.detach().cpu().float() for k, v in lmap.items()})

    lines = []
    for key, val in cfg.to_dict().items():
        vals = val if isinstance(val, list) else [val]
        lines.append(f"cfg {key} " + " ".join(str(int(v)) for v in vals))
    blobs, offset = [], 0
    for name, t in tensors.items():
        dims = list(t.shape) or [1]
        lines.append(f"tensor {name} {offset} {len(dims)} " + " ".join(map(str, dims)))
        blobs.append(t.numpy().astype("<f4").tobytes())
        offset += t.numel()
    header = ("\n".join(lines) + "\n").encode("ascii")
    with open(out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(header)))
        fh.write(header)
        for b in blobs:
            fh.write(b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None, help="dataset folder for the latent map (PCA)")
    a = p.parse_args()
    model = load_checkpoint(a.ckpt)
    lmap = latent_map(model, a.data) if a.data else None
    export(model, a.out, lmap)
    size = Path(a.out).stat().st_size / 1e6
    print(f"wrote {a.out} ({size:.1f} MB)" + ("" if a.data else " - no latent map, raw axes"))


if __name__ == "__main__":
    main()
