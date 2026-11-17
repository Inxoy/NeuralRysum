"""Organ clips with MIDI notes, for training the v2 synth.

    python -m neuralrysum.organ_data SRC DST [--kinds pipe,registration]

SRC is a folder with ``metadata.jsonl`` (one clip per line: audio, notes with
on / off in seconds and midi, kind, split) as rendered by the Rysum plugin.
``prepare`` mixes every clip to mono into one int16 file (memory-mapped by the
data loader), measures the real pitch of every note from the audio (temperaments
and detune differ between clips, so the nominal MIDI key can be off by tens of
cents), and assigns notes to voices the same way the plugin will.
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 48000


# ---- preparation --------------------------------------------------------------
def note_offset_cents(x: np.ndarray, note: dict, others: list, sr: int = SR) -> float:
    """Pitch deviation of a note from equal temperament (A = 440), from its partials."""
    a = int((note["on"] + 0.15) * sr)
    b = int(min(note["off"], note["on"] + 2.0) * sr)
    if b - a < int(0.15 * sr):
        return 0.0
    seg = x[a:b] * np.hanning(b - a)
    n = 1 << 18
    sp = np.abs(np.fft.rfft(seg, n))
    f_et = 440.0 * 2 ** ((note["midi"] - 69) / 12)
    devs, wts = [], []
    other_f = [440.0 * 2 ** ((o["midi"] - 69) / 12) for o in others]
    for h in (0.5, 1, 2, 3, 4, 5, 6, 8):
        fc = h * f_et
        if fc > 8000:
            break
        # skip partials that another sounding note could also own
        if any(abs(1200 * np.log2(fc / (k * fo))) < 80 for fo in other_f for k in (0.5, 1, 2, 3, 4, 5, 6, 8)):
            continue
        lo, hi = int(fc * 2 ** (-60 / 1200) * n / sr), int(fc * 2 ** (60 / 1200) * n / sr) + 1
        k = lo + int(np.argmax(sp[lo:hi]))
        if k <= lo or k >= hi - 1:
            continue
        y0, y1, y2 = np.log(sp[k - 1: k + 2] + 1e-12)
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)  # parabolic peak interpolation
        f = (k + d) * sr / n
        devs.append(1200 * np.log2(f / fc))
        wts.append(sp[k])
    if not devs:
        return 0.0
    o = np.argsort(devs)
    c = np.cumsum(np.array(wts)[o])
    return float(np.array(devs)[o][np.searchsorted(c, c[-1] / 2)])  # weighted median


def fill_offsets(clips: list, limit: float = 45.0) -> None:
    """Notes without a usable measurement get the typical offset of their pitch class.

    The organ is in an unequal temperament, so the offset depends mostly on the
    pitch class. Measurements of exactly 0 (nothing measurable) or beyond
    ``limit`` cents (the search window edge) are replaced.
    """
    by_pc = {}
    for c in clips:
        for n in c["notes"]:
            d = (n["key"] - n["midi"]) * 100
            if d != 0.0 and abs(d) <= limit:
                by_pc.setdefault(n["midi"] % 12, []).append(d)
    med = {k: float(np.median(v)) for k, v in by_pc.items()}
    for c in clips:
        for n in c["notes"]:
            d = (n["key"] - n["midi"]) * 100
            if d == 0.0 or abs(d) > limit:
                n["key"] = n["midi"] + med.get(n["midi"] % 12, 0.0) / 100


def assign_voices(notes: list, voices: int, hold: float) -> list:
    free_at = [-1e9] * voices
    out = []
    for n in sorted(notes, key=lambda n: n["on"]):
        cand = [v for v in range(voices) if free_at[v] <= n["on"]]
        v = cand[0] if cand else int(np.argmin(free_at))  # steal the one that frees first
        free_at[v] = n["off"] + hold
        out.append({**n, "voice": v})
    return out


def _prepare_clip(args):
    src, m, voices, hold = args
    a, sr = sf.read(src / m["audio"], dtype="float32", always_2d=True)
    assert sr == SR, f"{m['audio']}: {sr} Hz"
    x = a.mean(1)
    notes = []
    for i, n in enumerate(m["notes"]):
        others = [o for j, o in enumerate(m["notes"]) if j != i and o["on"] < n["off"] and o["off"] > n["on"]]
        notes.append({"on": n["on"], "off": n["off"], "midi": n["midi"],
                      "key": n["midi"] + note_offset_cents(x, n, others) / 100.0})
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    return pcm, assign_voices(notes, voices, hold)


def prepare(src: str, dst: str, kinds=("pipe", "registration"), voices: int = 8, hold: float = 0.5,
            workers: int = 8) -> None:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in (src / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = [m for m in meta if m["kind"] in kinds]
    length = sf.info(str(src / meta[0]["audio"])).frames
    mm = np.lib.format.open_memmap(dst / "audio.npy", mode="w+", dtype=np.int16, shape=(len(meta), length))
    clips = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (pcm, notes) in enumerate(ex.map(_prepare_clip, [(src, m, voices, hold) for m in meta], chunksize=8)):
            mm[i, : len(pcm)] = pcm[:length]
            m = meta[i]
            clips.append({"row": i, "audio": m["audio"], "kind": m["kind"], "split": m["split"],
                          "stops": m["stops"], "notes": notes})
            if (i + 1) % 400 == 0:
                print(f"{i + 1}/{len(meta)}", flush=True)
    mm.flush()
    fill_offsets(clips)
    offs = np.array([n["key"] - n["midi"] for c in clips for n in c["notes"]]) * 100
    print(f"{len(clips)} clips, {sum(len(c['notes']) for c in clips)} notes; pitch offset from ET: "
          f"median {np.median(offs):+.1f} cents, 90 % within {np.percentile(np.abs(offs), 90):.1f} cents, "
          f"max {np.abs(offs).max():.1f}")
    (dst / "index.json").write_text(json.dumps({"sample_rate": SR, "length": length, "voices": voices,
                                                "hold": hold, "kinds": list(kinds), "clips": clips}))


# ---- voices as frame features -------------------------------------------------------
def voice_frames(notes: list, start: int, frames: int, hop: int, voices: int, sr: int = SR):
    """State of every voice at the start of each frame -> key, gate, on_age, off_age (V, F)."""
    t = (start + np.arange(frames) * hop) / sr
    key = np.zeros((voices, frames), np.float32)
    gate = np.zeros_like(key)
    on_age = np.zeros_like(key)
    off_age = np.full_like(key, -1.0)
    for v in range(voices):
        ns = sorted((n for n in notes if n["voice"] == v), key=lambda n: n["on"])
        if not ns:
            continue
        ons = np.array([n["on"] for n in ns])
        j = np.searchsorted(ons, t, side="right") - 1  # latest note started by t
        ok = j >= 0
        jj = np.clip(j, 0, None)
        on = ons[jj]
        off = np.array([n["off"] for n in ns])[jj]
        k = np.array([n["key"] for n in ns])[jj]
        key[v] = np.where(ok, k, 0.0)
        gate[v] = (ok & (t < off)).astype(np.float32)
        on_age[v] = np.where(ok, t - on, 0.0)
        off_age[v] = np.where(ok & (t >= off), t - off, -1.0)
    return key, gate, on_age, off_age


class OrganMidiDataset(Dataset):
    """Random windows: ``preroll`` samples to let the room fill, then ``crop`` samples that count."""

    def __init__(self, root: str, split: str, preroll: int, crop: int, hop: int, voices: int,
                 epoch_size: int = 10_000, gain_db=(-6.0, 0.0)):
        self.root = Path(root)
        idx = json.loads((self.root / "index.json").read_text())
        assert idx["voices"] >= voices
        self.clips = [c for c in idx["clips"] if c["split"] == split]
        self.length = idx["length"]
        self.preroll, self.crop, self.hop, self.voices = preroll, crop, hop, voices
        self.epoch_size, self.gain_db = epoch_size, gain_db
        self.audio = None
        self.hours = len(self.clips) * self.length / SR / 3600

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        if self.audio is None:
            self.audio = np.load(self.root / "audio.npy", mmap_mode="r")
        for _try in range(8):
            c = random.choice(self.clips)
            s = random.randint(0, self.length - self.crop)  # first counted sample
            a = s - self.preroll
            x = np.zeros(self.preroll + self.crop, np.float32)
            lo = max(a, 0)
            x[lo - a:] = self.audio[c["row"], lo: s + self.crop].astype(np.float32) / 32767
            if np.sqrt(np.mean(x[self.preroll:] ** 2)) > 1e-3:
                break
        g = 10 ** (random.uniform(*self.gain_db) / 20)
        key, gate, on_age, off_age = voice_frames(c["notes"], a, len(x) // self.hop, self.hop, self.voices)
        return (torch.from_numpy(x * g)[None], torch.from_numpy(key), torch.from_numpy(gate),
                torch.from_numpy(on_age), torch.from_numpy(off_age))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--kinds", default="pipe,registration")
    ap.add_argument("--voices", type=int, default=8)
    a = ap.parse_args()
    prepare(a.src, a.dst, tuple(a.kinds.split(",")), a.voices)


if __name__ == "__main__":
    main()
