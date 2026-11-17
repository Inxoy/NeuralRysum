"""Render one recording through v2 (Manual and Follow) and v1 for comparison.

    python scripts/render_compare.py IN.wav OUTDIR --pitch runs/pitch/last.pt --synth runs/synth/last.pt
        [--v1 runs/rysum/last.pt] [--organ data/rysum_organ_v2]

Manual uses the mean timbre of the organ dataset's validation clips.
Prints what the voice tracker found.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralrysum.data import load_mono  # noqa: E402
from neuralrysum.export import load_checkpoint  # noqa: E402
from neuralrysum.v2 import V2, load_pitch, load_synth  # noqa: E402

SR = 48000


def save(path: Path, y: np.ndarray, ref_rms: float) -> None:
    """Match the input's loudness, then keep the peak below 0 dBFS."""
    rms = np.sqrt(np.mean(y ** 2)) + 1e-12
    y = y * ref_rms / rms
    peak = np.abs(y).max()
    if peak > 0.98:
        y = y * 0.98 / peak
    sf.write(path, y.astype(np.float32), SR, subtype="PCM_24")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--v1")
    ap.add_argument("--organ", default="data/rysum_organ_v2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    x = load_mono(a.inp, SR)
    hop = 2048
    x = x[: len(x) // hop * hop].astype(np.float32)
    ref_rms = float(np.sqrt(np.mean(x ** 2)))
    sf.write(out / "input_48k_mono.wav", x, SR, subtype="PCM_24")
    xt = torch.from_numpy(x)[None, None]

    m = V2(load_pitch(a.pitch), load_synth(a.synth))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from leak_test import organ_clips  # noqa: E402
    organ = organ_clips(a.organ, 16, 65536)
    z0, d0 = m.timbre(organ)

    t0 = time.time()
    key, gate, on_age, off_age = m.voices(xt)
    print(f"tracking: {time.time() - t0:.1f} s")
    act = gate > 0
    n_act = act.sum(0)
    print(f"voices sounding: mean {n_act.mean():.2f}, max {int(n_act.max())}, "
          f"silent {100 * (n_act == 0).mean():.0f} % of the time")
    if act.any():
        k = key[act]
        print(f"pitches: {k.min():.1f} .. {k.max():.1f} (MIDI, after octave folding), median {np.median(k):.1f}")
        starts = int(((on_age == 0) & act).sum())
        print(f"note starts: {starts} ({starts / (len(x) / SR):.1f} per second)")

    for mode, (z, d) in (("manual", (z0, d0)), ("follow", (None, None))):
        t0 = time.time()
        y = m.render_voices(key, gate, on_age, off_age, x_follow=xt, z=z, desc=d)[0, 0].numpy()
        save(out / f"v2_{mode}.wav", y, ref_rms)
        print(f"v2 {mode}: {time.time() - t0:.1f} s")
    # stricter tracking: fewer, longer notes (6 frames to confirm, higher threshold)
    m.track_kw = {"confirm": 6, "on": 0.7, "match": 1.0}
    v = m.voices(xt)
    starts = int(((v[2] == 0) & (v[1] > 0)).sum())
    print(f"strict tracking: {starts / (len(x) / SR):.1f} note starts per second, "
          f"mean voices {(v[1] > 0).sum(0).mean():.2f}")
    y = m.render_voices(*v, x_follow=xt, z=z0, desc=d0)[0, 0].numpy()
    save(out / "v2_manual_strict.wav", y, ref_rms)

    if a.v1:
        v1 = load_checkpoint(a.v1)
        _, h = v1.fast(organ)
        mean, _, _ = v1.slow(h)
        from neuralrysum.descriptors import descriptors
        z1 = mean[:, :, 8:].mean((0, 2))
        d1 = descriptors(organ, v1.cfg.slow_hop, SR)[:, :, 8:].mean((0, 2))
        with torch.no_grad():
            save(out / "v1_manual.wav", v1.transfer(xt, z1, d1)[0, 0].numpy(), ref_rms)
            save(out / "v1_follow.wav", v1.transfer(xt)[0, 0].numpy(), ref_rms)
        print("v1 rendered")


if __name__ == "__main__":
    main()
