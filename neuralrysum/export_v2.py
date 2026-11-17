"""Export v2 (pitch estimator + organ synth) to one ``.nrv2`` file for the C++ engine.

    python -m neuralrysum.export_v2 --pitch runs/pitch/last.pt --synth runs/synth/last.pt \
        --data data/rysum_organ_v2 --out vox_humana.nrv2

Same layout as v1's ``.nrys`` (see export.py), with magic ``NRV20001``. Besides the
weights it stores everything the engine would otherwise have to recompute, so the
C++ side matches PyTorch exactly:

- ``pitch.spec.idx<j> / frac<j> / dst<j> / win<j>``: log-bin tables of the front end
- ``synth.phases``: the fixed harmonic phases of the wavetables
- ``timbre.z_manual``, ``timbre.d_manual``: the default (Manual) timbre, the mean over
  the organ dataset's validation clips, as in ``V2.timbre``
- ``timbre.z_basis``: principal axes of z over those clips, row i = std_i * pc_i
- ``consts``: release_hold, fold_lo, fold_hi and the tracker defaults
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from .descriptors import descriptors
from .v2 import V2, load_pitch, load_synth

MAGIC = b"NRV20001"

# tracker defaults (VoxTracker in C++ uses these unless the plugin overrides them)
TRACK = {"on": 0.6, "off": 0.3, "match": 0.7, "hold": 24, "onset": 0.5, "confirm": 4}


@torch.no_grad()
def timbre_map(m: V2, data: str, clips: int = 144):
    """Mean timbre (z, descriptors) and the principal axes of z over organ clips."""
    audio = np.load(Path(data) / "audio.npy", mmap_mode="r")
    import json
    idx = json.loads((Path(data) / "index.json").read_text())
    rows = [c["row"] for c in idx["clips"] if c["split"] == "valid"][:clips]
    hop = m.cfg.slow_hop
    zs, ds = [], []
    for r in rows:
        x = torch.from_numpy(audio[r].astype(np.float32) / 32767)
        x = x[: len(x) // hop * hop][None, None]
        mean, _, _ = m.synth.encode(x)
        d = descriptors(x, hop, m.cfg.sample_rate)
        keep = d[0].abs().sum(0) > 0  # skip silent frames (descriptors 0)
        keep[:8] = False  # and the GRU warm-up
        zs.append(mean[0][:, keep].T)
        ds.append(d[0][:, keep].T)
    z, d = torch.cat(zs), torch.cat(ds)
    mu = z.mean(0)
    _, s, vt = torch.linalg.svd(z - mu, full_matrices=False)
    std = s / max(1, len(z) - 1) ** 0.5
    return mu, d.mean(0), vt * std[:, None], len(z)


def write(out: str, cfg: dict, tensors: dict) -> None:
    lines = []
    for key, val in cfg.items():
        vals = val if isinstance(val, (list, tuple)) else [val]
        lines.append(f"cfg {key} " + " ".join(str(int(v)) for v in vals))
    blobs, offset = [], 0
    for name, t in tensors.items():
        t = torch.as_tensor(t).detach().cpu().float()
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


def export(pitch: str, synth: str, data: str, out: str) -> None:
    est, syn = load_pitch(pitch), load_synth(synth)
    m = V2(est, syn)
    pc, sc = est.cfg, syn.cfg
    ec = syn.enc_cfg
    cfg = {
        "sample_rate": pc.sample_rate,
        "est_hop": pc.hop, "bins_per_semitone": pc.bins_per_semitone,
        "midi_lo": pc.midi_lo, "midi_hi": pc.midi_hi, "n_bins": pc.n_bins,
        "windows": pc.windows, "n_in": est.spec.n_in, "lo_pad": est.spec.lo_pad,
        "shifts": est.spec.shifts, "channels": pc.channels,
        "hop": sc.hop, "table": sc.table, "voices": sc.voices, "key_lo": sc.key_lo, "key_hi": sc.key_hi,
        "key_emb": sc.key_emb, "hidden": sc.hidden, "noise_bands": sc.noise_bands,
        "ir_len": syn.ir.numel(), "z_dim": sc.z_dim, "n_desc": sc.n_desc, "slow_hop": sc.slow_hop,
        "enc_strides": ec.enc_strides, "slow_strides": ec.slow_strides,
        "track_hold": TRACK["hold"], "track_confirm": TRACK["confirm"],
    }
    tensors = {}
    for k, v in est.state_dict().items():
        tensors["pitch." + k] = v
    for k, v in syn.state_dict().items():
        if k.startswith("trunk.to_e"):
            continue  # v1's fast code: not used by v2
        tensors["synth." + k] = v
    zm, dm, basis, n = timbre_map(m, data)
    tensors["timbre.z_manual"] = zm
    tensors["timbre.d_manual"] = dm
    tensors["timbre.z_basis"] = basis
    tensors["consts"] = torch.tensor([sc.release_hold, m.key_range[0], m.key_range[1],
                                      TRACK["on"], TRACK["off"], TRACK["match"], TRACK["onset"]])
    write(out, cfg, tensors)
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.1f} MB), timbre map from {n} frames; "
          f"z std {basis.norm(dim=1).numpy().round(3)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--data", required=True, help="prepared organ data (for the Manual timbre)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a.pitch, a.synth, a.data, a.out)


if __name__ == "__main__":
    main()
