"""Structural guarantees the plugin relies on. Run with ``pytest``."""
import numpy as np
import soundfile as sf
import torch

from neuralrysum import ModelConfig, NeuralRysum, TrainConfig
from neuralrysum.augment import perturb_timbre
from neuralrysum.descriptors import descriptors
from neuralrysum.train import train


def make():
    torch.manual_seed(0)
    m = NeuralRysum().eval()
    # the FiLM head starts at zero; randomise it so the tests exercise it
    torch.nn.init.normal_(m.decoder.cond.head.weight, std=0.05)
    torch.nn.init.normal_(m.decoder.noise_w, std=0.1)
    return m


def test_shapes_and_hop():
    m = make()
    cfg = m.cfg
    assert cfg.hop == 64 and cfg.slow_hop == 2048
    x = torch.randn(2, 1, cfg.slow_hop * 4)
    o = m(x, descriptors(x, cfg.slow_hop, cfg.sample_rate))
    assert o["y"].shape == x.shape
    assert o["e"].shape == (2, cfg.fast_dim, x.shape[-1] // cfg.hop)


def test_fast_path_is_causal():
    """Changing the input at sample n must not change any output before the
    start of n's frame: zero look-ahead, latency = at most one hop."""
    m = make()
    cfg = m.cfg
    t = cfg.slow_hop * 4
    x = torch.randn(1, 1, t)
    z, d = torch.zeros(cfg.z_dim), torch.zeros(cfg.n_desc)
    noise = torch.zeros(1, 1, t)
    y1 = m.transfer(x, z, d, noise)
    n = 3000
    x2 = x.clone()
    x2[..., n:] = torch.randn(t - n)
    y2 = m.transfer(x2, z, d, noise)
    frame_start = n // cfg.hop * cfg.hop
    assert torch.equal(y1[..., :frame_start], y2[..., :frame_start])
    # ...and it does react within the very frame the change happens in
    assert not torch.allclose(y1[..., frame_start : frame_start + cfg.hop],
                              y2[..., frame_start : frame_start + cfg.hop])


def test_slow_path_never_sees_the_current_slow_frame():
    """In follow mode the timbre used for slow frame j comes from frames < j."""
    m = make()
    cfg = m.cfg
    t = cfg.slow_hop * 6
    x = torch.randn(1, 1, t)
    noise = torch.zeros(1, 1, t)
    y1 = m.transfer(x, noise=noise)
    n = cfg.slow_hop * 3 + 100
    x2 = x.clone()
    x2[..., n:] = torch.randn(t - n)
    y2 = m.transfer(x2, noise=noise)
    frame_start = n // cfg.hop * cfg.hop
    assert torch.equal(y1[..., :frame_start], y2[..., :frame_start])


def test_timbre_controls_change_the_output():
    m = make()
    cfg = m.cfg
    x = torch.randn(1, 1, cfg.slow_hop * 2)
    noise = torch.zeros_like(x)
    d = torch.zeros(cfg.n_desc)
    y1 = m.transfer(x, torch.zeros(cfg.z_dim), d, noise)
    y2 = m.transfer(x, torch.ones(cfg.z_dim), d, noise)
    assert not torch.allclose(y1, y2)


def test_free_generation_runs():
    m = make()
    cfg = m.cfg
    y = m.free(torch.zeros(1, cfg.cond_dim, 40), temperature=1.0)
    assert y.shape == (1, 1, 40 * cfg.hop) and torch.isfinite(y).all()


def test_descriptors_order_sounds_sensibly():
    sr, hop = 48000, 2048
    t = torch.arange(hop * 4) / sr
    low = torch.sin(2 * torch.pi * 110 * t)[None]
    high = torch.sin(2 * torch.pi * 4000 * t)[None]
    noise = torch.randn(1, hop * 4) * 0.3
    dl, dh, dn = (descriptors(s, hop, sr).mean(-1)[0] for s in (low, high, noise))
    assert dh[0] > dl[0]  # brightness
    assert dn[1] > dl[1] and dn[1] > dh[1]  # noisiness
    assert descriptors(torch.zeros(1, hop * 2), hop, sr).abs().sum() == 0


def test_augmentation_keeps_shape_and_level():
    x = torch.randn(4, 1, 16384) * 0.1
    y = perturb_timbre(x, 48000)
    assert y.shape == x.shape and torch.isfinite(y).all()
    ratio = y.pow(2).mean().sqrt() / x.pow(2).mean().sqrt()
    assert 0.3 < ratio < 3.0


def test_training_smoke(tmp_path):
    """A few optimisation steps incl. the adversarial phase, then resume."""
    sr = 48000
    t = np.arange(sr * 3) / sr
    for i, f0 in enumerate([110.0, 220.0, 330.0]):
        sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * 0.1
        sf.write(tmp_path / f"tone{i}.wav", sig.astype(np.float32), sr)
    tc = TrainConfig(data=str(tmp_path), out=str(tmp_path / "run"), crop=8192, batch=2,
                     steps=4, warmup_adv=2, workers=0, save_every=2, log_every=1, kl_warmup=2)
    ck = train(tc)
    assert ck.exists()
    tc.steps = 5
    train(tc)
    assert torch.load(ck)["step"] == 5
