"""Reference vectors for the C++ port of v2 (same file format as the model).

    python scripts/make_v2_reference.py --pitch P.pt --synth S.pt --model M.nrv2 --out ref.nrv2 [--music x.wav]

Contains the input and, for it, everything the C++ engine has to reproduce:
estimator logits, tracker output (from the reference salience), synth voices,
Manual / Follow conditioning and audio (noise switched off, so it is exact), and
a MIDI render.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralrysum.data import load_mono  # noqa: E402
from neuralrysum.export_v2 import write  # noqa: E402
from neuralrysum.model import NeuralRysum  # noqa: E402
from neuralrysum.organ_data import assign_voices, voice_frames  # noqa: E402
from neuralrysum.pitch import VoiceTracker  # noqa: E402
from neuralrysum.v2 import V2, fold, load_pitch, load_synth, tracked_voices  # noqa: E402

SR = 48000


def chord_input(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    x = np.zeros_like(t)
    for on, off, midi in ((0.2, 1.6, 48), (0.5, 1.6, 55), (0.8, 2.4, 64), (1.8, 2.9, 31), (2.2, 2.9, 79)):
        f0 = 440 * 2 ** ((midi - 69) / 12)
        env = np.clip((t - on) / 0.02, 0, 1) * np.clip((off - t) / 0.05, 0, 1)
        for k in range(1, 16):
            if k * f0 < 18000:
                x += env * (0.6 / k) * np.sin(2 * np.pi * k * f0 * t + k)
    return 0.25 * x / np.abs(x).max()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--model", required=True, help="the exported .nrv2 (for the Manual timbre)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--music", help="a recording; 3 s of it are appended")
    a = ap.parse_args()
    torch.manual_seed(0)
    est, syn = load_pitch(a.pitch), load_synth(a.synth)
    m = V2(est, syn)
    cfg = syn.cfg

    parts = [chord_input(3.0)]
    if a.music:
        mus = load_mono(a.music, SR)
        parts.append(mus[30 * SR: 33 * SR] * 0.5)
    parts.append(np.zeros(SR // 2))
    x = np.concatenate(parts).astype(np.float32)
    x = x[: len(x) // cfg.slow_hop * cfg.slow_hop]
    xt = torch.from_numpy(x)[None, None]

    with torch.no_grad():
        s_log, o_log = est(xt)
    sal = torch.sigmoid(s_log)[0].numpy()
    ons = torch.sigmoid(o_log)[0].numpy()
    tr = VoiceTracker(est.cfg.bin_midi(), cfg.voices)
    out = [tr.step(sal[:, f], ons[:, f]) for f in range(sal.shape[1])]
    t_midi, t_gate, t_started = (np.stack(v, axis=1) for v in zip(*out))
    frames = len(x) // cfg.hop
    key, gate, on_age, off_age = tracked_voices(fold(t_midi, *m.key_range), t_gate, t_started,
                                                est.cfg.hop, cfg.hop, frames)

    # Manual timbre: read back from the exported file (identical to the plugin's default)
    import struct
    raw = Path(a.model).read_bytes()
    hlen = struct.unpack("<I", raw[8:12])[0]
    header = raw[12: 12 + hlen].decode().splitlines()
    data = np.frombuffer(raw[12 + hlen:], dtype="<f4")
    tens = {}
    for line in header:
        p = line.split()
        if p[0] == "tensor":
            off, nd = int(p[2]), int(p[3])
            dims = [int(d) for d in p[4: 4 + nd]]
            tens[p[1]] = data[off: off + int(np.prod(dims))].reshape(dims)
    z0 = torch.from_numpy(tens["timbre.z_manual"].copy())
    d0 = torch.from_numpy(tens["timbre.d_manual"].copy())

    V = lambda v: torch.as_tensor(v)[None]  # noqa: E731
    zeros = torch.zeros(1, cfg.noise_bands, frames * cfg.hop)
    with torch.no_grad():
        c_man = torch.cat([z0, d0]).view(1, -1, 1).expand(1, -1, frames)
        y_man_dry = syn.render(V(key), V(gate), V(on_age), V(off_age), c_man, zeros, dry_only=True)
        y_man = syn.render(V(key), V(gate), V(on_age), V(off_age), c_man, zeros)
        mean, _, dp = syn.encode(xt)
        c_slow = NeuralRysum.delay_one(torch.cat([mean, dp], 1))
        c_fol = syn.hold(c_slow, frames)
        y_fol_dry = syn.render(V(key), V(gate), V(on_age), V(off_age), c_fol, zeros, dry_only=True)

        # MIDI: a few notes, voices assigned like the dataset / the plugin
        notes = [{"on": 0.10, "off": 0.90, "key": 60.0}, {"on": 0.30, "off": 1.40, "key": 64.0},
                 {"on": 0.30, "off": 0.50, "key": 67.0}, {"on": 0.55, "off": 1.10, "key": 67.0},
                 {"on": 1.00, "off": 1.80, "key": 43.5}]
        ns = assign_voices([{**n, "midi": round(n["key"])} for n in notes], cfg.voices, cfg.release_hold)
        mframes = 2 * SR // cfg.hop
        mk, mg, mon, moff = voice_frames(ns, 0, mframes, cfg.hop, cfg.voices)
        c_mid = torch.cat([z0, d0]).view(1, -1, 1).expand(1, -1, mframes)
        y_midi = syn.render(V(mk), V(mg), V(mon), V(moff), c_mid,
                            torch.zeros(1, cfg.noise_bands, mframes * cfg.hop), dry_only=True)

    tensors = {
        "input": x,
        "sal_logit": s_log[0].numpy(), "ons_logit": o_log[0].numpy(),
        "trk_midi": t_midi, "trk_gate": t_gate, "trk_started": t_started.astype(np.float32),
        "v_key": key, "v_gate": gate, "v_on_age": on_age, "v_off_age": off_age,
        "manual_dry": y_man_dry[0, 0].numpy(), "manual_wet": y_man[0, 0].numpy(),
        "follow_cond": c_slow[0].numpy(), "follow_dry": y_fol_dry[0, 0].numpy(),
        "midi_notes": np.array([[n["on"], n["off"], n["key"]] for n in notes], np.float32),
        "midi_key": mk, "midi_gate": mg, "midi_dry": y_midi[0, 0].numpy(),
    }
    write(a.out, {"frames": frames, "est_frames": sal.shape[1]}, tensors)
    st = int(t_started.sum())
    print(f"wrote {a.out}: {len(x) / SR:.2f} s, {st} tracked note starts, "
          f"manual peak {float(y_man.abs().max()):.3f}, follow peak {float(y_fol_dry.abs().max()):.3f}")


if __name__ == "__main__":
    main()
