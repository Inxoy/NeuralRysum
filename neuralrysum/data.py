"""Folder-of-audio-files dataset with random crops."""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


def find_audio(root: str) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)


def load_mono(path, sample_rate: int, start: int = 0, frames: int = -1) -> np.ndarray:
    """Read (part of) a file as mono float32 at ``sample_rate``.

    start / frames are in the file's own sample rate.
    """
    audio, sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        from scipy.signal import resample_poly

        g = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    return audio


class CropDataset(Dataset):
    """Random crops, chosen with probability proportional to file length.

    Multichannel files (e.g. Ambisonics or multi-mic organ recordings) are
    mixed down to mono. Near-silent crops are re-drawn a few times.
    """

    def __init__(self, root: str, crop: int, sample_rate: int, epoch_size: int = 10_000):
        self.files = find_audio(root)
        if not self.files:
            raise FileNotFoundError(f"no audio files ({', '.join(sorted(EXTS))}) under {root}")
        self.crop, self.sr, self.epoch_size = crop, sample_rate, epoch_size
        self.info = [sf.info(str(f)) for f in self.files]
        self.weights = [max(i.frames / i.samplerate, 1e-3) for i in self.info]
        self.hours = sum(self.weights) / 3600.0

    def __len__(self) -> int:
        return self.epoch_size

    def _draw(self) -> np.ndarray:
        i = random.choices(range(len(self.files)), weights=self.weights)[0]
        info = self.info[i]
        need = int(math.ceil(self.crop * info.samplerate / self.sr)) + 8
        start = random.randint(0, max(0, info.frames - need))
        audio = load_mono(self.files[i], self.sr, start, need)[: self.crop]
        if len(audio) < self.crop:
            audio = np.pad(audio, (0, self.crop - len(audio)))
        return audio

    def __getitem__(self, _):
        for _try in range(8):
            audio = self._draw()
            if float(np.sqrt(np.mean(audio**2))) > 1e-3:
                break
        gain = 10.0 ** (random.uniform(-6.0, 0.0) / 20.0)
        peak = float(np.abs(audio).max())
        if peak * gain > 0.99:
            gain = 0.99 / peak
        return torch.from_numpy(audio * gain)[None]
