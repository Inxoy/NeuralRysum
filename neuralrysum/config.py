"""Model / training configuration.

Everything that the C++ engine needs to rebuild the network lives in
``ModelConfig`` and is written into the exported ``.nrys`` file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class ModelConfig:
    sample_rate: int = 48000

    # ---- fast path (sets the latency) ------------------------------------
    # product of enc_strides = hop of the fast code in samples.
    # [4, 4, 4] -> 64 samples -> 1.33 ms at 48 kHz.
    enc_strides: List[int] = field(default_factory=lambda: [4, 4, 4])
    enc_channels: List[int] = field(default_factory=lambda: [32, 64, 96])
    fast_dim: int = 12  # size of the fast code e_t (pitch / phase / envelope)

    # ---- slow path (timbre, may lag) -------------------------------------
    # product of slow_strides = how many fast frames per slow frame.
    # [4, 4, 2] -> 32 fast frames -> 2048 samples -> 42.7 ms.
    slow_strides: List[int] = field(default_factory=lambda: [4, 4, 2])
    slow_channels: int = 128
    z_dim: int = 8  # navigable timbre latent
    n_desc: int = 3  # brightness, noisiness, spread (explicit knobs)

    # ---- decoder ----------------------------------------------------------
    dec_channels: List[int] = field(default_factory=lambda: [96, 64, 32, 16])
    dec_strides: List[int] = field(default_factory=lambda: [4, 4, 4])
    dec_dilations: List[int] = field(default_factory=lambda: [1, 3])
    cond_hidden: int = 128  # width of the conditioning MLP that feeds FiLM

    # ---- prior (free latent navigation, no input) -------------------------
    prior_hidden: int = 128

    @property
    def hop(self) -> int:
        h = 1
        for s in self.enc_strides:
            h *= s
        return h

    @property
    def slow_ratio(self) -> int:
        r = 1
        for s in self.slow_strides:
            r *= s
        return r

    @property
    def slow_hop(self) -> int:
        return self.hop * self.slow_ratio

    @property
    def cond_dim(self) -> int:
        return self.z_dim + self.n_desc

    def validate(self) -> None:
        up = 1
        for s in self.dec_strides:
            up *= s
        assert up == self.hop, "decoder upsampling must equal encoder hop"
        assert len(self.dec_channels) == len(self.dec_strides) + 1
        assert len(self.enc_channels) == len(self.enc_strides)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hop"] = self.hop
        d["slow_ratio"] = self.slow_ratio
        return d

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        keys = ModelConfig.__dataclass_fields__.keys()
        return ModelConfig(**{k: v for k, v in d.items() if k in keys})


def large_config() -> ModelConfig:
    """More capacity, same latency. Roughly 4x the CPU cost of the default."""
    return ModelConfig(
        enc_channels=[48, 96, 192],
        dec_channels=[192, 128, 64, 32],
        slow_channels=192,
        cond_hidden=192,
        prior_hidden=256,
        fast_dim=16,
    )


@dataclass
class TrainConfig:
    data: str = "data"  # folder with wav/flac files
    out: str = "runs/default"
    crop: int = 65536  # samples per training example (multiple of slow hop)
    batch: int = 8
    lr: float = 2e-4
    steps: int = 500_000
    warmup_adv: int = 100_000  # spectral-only steps before the discriminator
    kl_weight: float = 1e-2
    kl_warmup: int = 20_000
    desc_weight: float = 1.0
    prior_weight: float = 1.0
    adv_weight: float = 1.0
    fm_weight: float = 10.0
    augment_prob: float = 0.8  # chance to timbre-perturb the fast-path input
    workers: int = 4
    save_every: int = 5000
    log_every: int = 100
    seed: int = 0
    size: str = "default"  # or "large"
    # ---- levers against timbre leaking through the fast path ---------------
    fast_dim: int = 0  # size of the fast code; 0 = keep the default of --size
    code_noise: float = 0.0  # std of noise added to the fast code on its way to the decoder
    perturb: str = "mild"  # or "strong": how hard the fast-path input is bent
    # transfer objective on synthetic foreign input (see foreign.py); adversarial phase only
    foreign_weight: float = 0.0  # 0 = off
    foreign_batch: int = 4
    fine_weight: float = 2.0  # keep the input's partials
    loud_weight: float = 1.0  # keep the input's loudness envelope
