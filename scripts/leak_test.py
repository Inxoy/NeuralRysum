"""Leak test: does a model re-render foreign input as the trained sound, or copy it?

    python scripts/leak_test.py --data data/organ --ckpt runs/a/last.pt runs/b/last.pt
                                [--wav-dir out] [--json out.json]

Three synthetic non-organ signals (saw, sung vowels, plucked string; four notes
of known pitch each) are sent through every checkpoint, once with the timbre
fixed to the dataset mean (Manual, knobs centred) and once in Follow.

    transfer  0 % = output has the input's spectrum, 100 % = the dataset's
              (third-octave band levels, 50 Hz - 16 kHz, overall level removed)
    HF        output level above 5 kHz relative to the input's, in dB.
              The more negative, the less of the input's top end gets through.
    pitch     share of the output's energy (50 Hz - 5 kHz) that sits on
              harmonics of the played note. Low = wrong notes or noise.
    env       correlation of input and output loudness envelopes (10 ms frames)
    organ     multi-resolution spectral loss when reconstructing clips of the
              dataset itself (Follow). Lower is better.
    silence   output level for silent input in Manual, dBFS. Lower is better.

A model that just copies scores transfer near 0 and HF near 0 dB with high
pitch and env. The aim is high transfer and strongly negative HF *without*
losing pitch, env and organ.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralrysum.data import find_audio, load_mono  # noqa: E402
from neuralrysum.descriptors import descriptors  # noqa: E402
from neuralrysum.export import load_checkpoint  # noqa: E402
from neuralrysum.losses import multires_stft_loss  # noqa: E402

SR = 48000
NOTE = 2 * SR  # samples per note
VOWELS = [(800, 1150, 2900), (350, 2000, 2800), (325, 700, 2700), (400, 1700, 2600)]


def _note(f0: float, amp_of_k, vibrato: float = 0.0) -> np.ndarray:
    t = np.arange(NOTE) / SR
    phase = 2 * np.pi * f0 * (t - vibrato * np.cos(2 * np.pi * 5 * t) / (2 * np.pi * 5))
    y = np.zeros(NOTE)
    for k in range(1, int(16000 / f0) + 1):
        y += amp_of_k(k, t) * np.sin(k * phase)
    fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
    return y * fade


def test_signals() -> dict:
    """name -> (signal, [f0 of each note])."""
    rng = np.random.default_rng(0)
    out = {}
    f = [110.0, 220.0, 329.63, 440.0]
    out["saw"] = (np.concatenate([_note(x, lambda k, t: 1.0 / k) for x in f]), f)

    f = [130.81, 196.0, 261.63, 164.81]
    notes = []
    for x, formants in zip(f, VOWELS):
        def amp(k, t, x=x, formants=formants):
            g = sum(1.0 / np.sqrt(1.0 + ((k * x - fc) / 60.0) ** 2) for fc in formants)
            return g / k  # glottal -12 dB/oct plus lip radiation +6 dB/oct
        notes.append(_note(x, amp, vibrato=0.004))
    v = np.concatenate(notes)
    out["voice"] = (v + rng.standard_normal(len(v)) * np.abs(v).max() * 10 ** (-40 / 20), f)

    f = [82.41, 146.83, 196.0, 329.63]
    out["pluck"] = (np.concatenate(
        [_note(x, lambda k, t: np.exp(-t * (1.5 + 0.4 * k)) / k) for x in f]), f)
    return {n: ((0.3 * s / np.abs(s).max()).astype(np.float32), f0) for n, (s, f0) in out.items()}


# ---- measurements ----------------------------------------------------------
def power_spectrum(y: np.ndarray, nfft: int = 8192):
    w = np.hanning(nfft)
    frames = [y[i:i + nfft] * w for i in range(0, len(y) - nfft + 1, nfft // 2)]
    return np.fft.rfftfreq(nfft, 1 / SR), np.mean([np.abs(np.fft.rfft(fr)) ** 2 for fr in frames], axis=0)


def band_levels(y: np.ndarray) -> np.ndarray:
    """Third-octave band levels in dB relative to the total, floored at -80."""
    f, p = power_spectrum(y)
    edges = 50.0 * 2 ** (np.arange(0, 26) / 3)  # 50 Hz .. 16 kHz
    bands = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    return np.maximum(10 * np.log10(bands / bands.sum() + 1e-30), -80.0)


def hf_power(y: np.ndarray) -> float:
    f, p = power_spectrum(y)
    return float(p[(f >= 5000) & (f < 16000)].sum())


def pitch_share(y: np.ndarray, f0s) -> float:
    shares = []
    for i, f0 in enumerate(f0s):
        seg = y[i * NOTE + int(0.3 * SR): (i + 1) * NOTE - int(0.1 * SR)]
        p = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        k = np.maximum(1, np.round(f / f0))
        # window around each harmonic: wide enough for the vibrato, never more than 0.2 f0
        on = np.abs(f - k * f0) < np.clip(0.012 * k * f0, 3.0, 0.2 * f0)
        rng = (f >= 50) & (f < 5000)
        shares.append(p[rng & on].sum() / max(p[rng].sum(), 1e-30))
    return float(np.mean(shares))


def envelope(y: np.ndarray) -> np.ndarray:
    n = len(y) // 480
    return np.sqrt((y[: n * 480].reshape(n, 480) ** 2).mean(1))


def organ_clips(data: str, n: int, length: int) -> torch.Tensor:
    files = find_audio(data)
    picks = [files[i * len(files) // n] for i in range(n)]
    clips = []
    for p in picks:
        start = sf.info(str(p)).frames // 3
        a = load_mono(p, SR, start, length)[:length]
        clips.append(np.pad(a, (0, length - len(a))))
    return torch.from_numpy(np.stack(clips))[:, None]


@torch.no_grad()
def evaluate(ckpt: str, organ: torch.Tensor, organ_bands: np.ndarray, signals: dict, wav_dir):
    model = load_checkpoint(ckpt)
    cfg = model.cfg
    assert cfg.sample_rate == SR
    torch.manual_seed(0)

    # centre of the timbre space = where the plugin's knobs sit at zero
    _, h = model.fast(organ)
    mean, _, _ = model.slow(h)
    z0 = mean[:, :, 8:].mean((0, 2))
    d0 = descriptors(organ, cfg.slow_hop, SR)[:, :, 8:].mean((0, 2))

    res = {"organ": float(np.mean([multires_stft_loss(model.transfer(c[None]), c[None]).item()
                                   for c in organ]))}
    sil = model.transfer(torch.zeros(1, 1, cfg.slow_hop * 96), z0, d0)[0, 0, SR:].numpy()
    res["silence"] = float(20 * np.log10(np.sqrt(np.mean(sil**2)) + 1e-12))

    for name, (x, f0s) in signals.items():
        n = len(x) // cfg.slow_hop * cfg.slow_hop
        xt = torch.from_numpy(x[:n])[None, None]
        bx = band_levels(x)
        for mode in ("manual", "follow"):
            y = (model.transfer(xt, z0, d0) if mode == "manual" else model.transfer(xt))[0, 0].numpy()
            by = band_levels(y)
            d_in = np.sqrt(np.mean((by - bx) ** 2))
            d_org = np.sqrt(np.mean((by - organ_bands) ** 2))
            res[f"{name}/{mode}"] = {
                "transfer": float(100 * d_in / (d_in + d_org + 1e-9)),
                "hf": float(10 * np.log10(hf_power(y) / hf_power(x) + 1e-30)),
                "pitch": 100 * pitch_share(y, f0s),
                "env": float(np.corrcoef(envelope(x[:n]), envelope(y))[0, 1]),
                "level": float(20 * np.log10(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(x**2)) + 1e-12)),
            }
            if wav_dir:
                wav_dir.mkdir(parents=True, exist_ok=True)
                sf.write(wav_dir / f"{name}_{mode}.wav", y, SR, subtype="PCM_24")
    for mode in ("manual", "follow"):
        res[mode] = {k: float(np.mean([res[f"{s}/{mode}"][k] for s in signals]))
                     for k in ("transfer", "hf", "pitch", "env")}
    return res


def label(ckpt: str) -> str:
    p = Path(ckpt)
    return p.parent.name if p.stem == "last" else f"{p.parent.name}/{p.stem}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="the training data, as the reference sound")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--wav-dir", help="write the rendered signals here, one folder per checkpoint")
    ap.add_argument("--json")
    ap.add_argument("--detail", action="store_true", help="also print every signal on its own")
    a = ap.parse_args()

    signals = test_signals()
    organ = organ_clips(a.data, 8, 65536)
    organ_bands = band_levels(organ.numpy().reshape(-1))
    if a.wav_dir:
        Path(a.wav_dir, "input").mkdir(parents=True, exist_ok=True)
        for n, (x, _) in signals.items():
            sf.write(Path(a.wav_dir, "input", f"{n}.wav"), x, SR, subtype="PCM_24")
    print("input signals:  pitch " + "  ".join(f"{n} {100 * pitch_share(x, f):.0f} %"
                                               for n, (x, f) in signals.items()))
    hdr = (f"{'checkpoint':26s} | {'Manual: transfer':>16s} {'HF':>7s} {'pitch':>6s} {'env':>5s} |"
           f" {'Follow: transfer':>16s} {'HF':>7s} {'pitch':>6s} {'env':>5s} | {'organ':>6s} {'silence':>8s}")
    print(hdr + "\n" + "-" * len(hdr))
    results = {}
    for ck in a.ckpt:
        name = label(ck)
        r = results[name] = evaluate(ck, organ, organ_bands, signals,
                                     Path(a.wav_dir, name.replace("/", "_")) if a.wav_dir else None)
        cells = " |".join(f" {r[m]['transfer']:14.0f} % {r[m]['hf']:6.1f}  {r[m]['pitch']:4.0f} % {r[m]['env']:5.2f}"
                          for m in ("manual", "follow"))
        print(f"{name:26s} |{cells} | {r['organ']:6.3f} {r['silence']:7.1f}", flush=True)
        if a.detail:
            for k, v in r.items():
                if "/" in k:
                    print(f"    {k:14s} transfer {v['transfer']:3.0f} %  HF {v['hf']:6.1f} dB  "
                          f"pitch {v['pitch']:3.0f} %  env {v['env']:.2f}  level {v['level']:+.1f} dB")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
