"""Prepare a folder of recordings for training.

    python scripts/prepare_data.py SRC DST [--channels split|mix|first]

For every audio file under SRC this writes 48 kHz mono 24-bit WAV to DST:

- resamples once, offline (the dataset would otherwise resample every crop)
- channels: ``split`` writes each channel as its own mono file (no phase
  cancellation from summing spaced microphones, and twice the material),
  ``mix`` averages them, ``first`` keeps channel 0 (use for Ambisonics W)
- trims leading / trailing silence below -60 dBFS, keeping 0.25 s of padding
- lowers the gain only if resampling pushed the peak above -0.1 dBFS

Levels are otherwise left alone: differences between pieces are musical.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg"}
SR = 48000


def band_db(mono: np.ndarray, lo: float, hi: float) -> float:
    """Mean level of a band relative to the whole spectrum, in dB."""
    win = 8192
    n = len(mono) // win * win
    step = max(1, (n // win) // 400)  # at most ~400 frames
    frames = mono[:n].reshape(-1, win)[::step] * np.hanning(win)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(win, 1 / SR)
    sel = (freqs >= lo) & (freqs < hi)
    return 10 * math.log10(power[sel].mean() / power.mean() + 1e-20)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--channels", choices=["split", "mix", "first"], default="split")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    total = 0.0
    for p in files:
        x, sr = sf.read(str(p), dtype="float64", always_2d=True)
        if sr != SR:
            g = math.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g, axis=0)

        # trim silence at the edges, the same cut for every channel
        n = SR // 20
        env = np.abs(x).max(axis=1)
        env = env[: len(env) // n * n].reshape(-1, n).max(axis=1)
        loud = np.where(env > 1e-3)[0]
        if len(loud) == 0:
            print(f"skip (silent): {p.name}")
            continue
        a = max(0, loud[0] * n - SR // 4)
        b = min(len(x), (loud[-1] + 1) * n + SR // 4)
        trimmed = (len(x) - (b - a)) / SR
        x = x[a:b]

        peak = float(np.abs(x).max())
        gain = min(1.0, 0.988 / peak)
        x = x * gain

        if args.channels == "mix":
            outs = {"": x.mean(axis=1)}
        elif args.channels == "first" or x.shape[1] == 1:
            outs = {"": x[:, 0]}
        else:
            names = "LR" if x.shape[1] == 2 else [str(i + 1) for i in range(x.shape[1])]
            outs = {f"_{names[i]}": x[:, i] for i in range(x.shape[1])}

        mono = x.mean(axis=1)
        hf = " ".join(f"{band_db(mono, lo, hi):+6.1f}" for lo, hi in [(4e3, 8e3), (8e3, 12e3), (12e3, 16e3), (16e3, 20e3)])
        for suffix, y in outs.items():
            sf.write(str(dst / f"{p.stem}{suffix}.wav"), y.astype(np.float32), SR, subtype="PCM_24")
            total += len(y) / SR
        print(
            f"{p.name[:34]:34s} {sr}->{SR} {len(x)/SR/60:6.2f} min  trimmed {trimmed:4.1f} s  "
            f"gain {20*math.log10(gain):+5.2f} dB  bands 4-8/8-12/12-16/16-20k: {hf} dB"
        )
    print(f"wrote {total/3600:.2f} h of mono audio to {dst}")


if __name__ == "__main__":
    main()
