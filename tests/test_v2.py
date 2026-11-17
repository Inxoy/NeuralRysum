"""v2: pitch estimator, voice tracking and organ synth."""
import numpy as np
import torch

from neuralrysum.organ_data import assign_voices, voice_frames
from neuralrysum.pitch import PitchConfig, PitchEstimator, VoiceTracker
from neuralrysum.pitch_data import make_batch
from neuralrysum.synth import OrganSynth, SynthConfig, band_noise
from neuralrysum.v2 import fold, tracked_voices


def test_pitch_estimator_is_causal():
    torch.manual_seed(0)
    est = PitchEstimator().eval()
    hop = est.cfg.hop
    x = torch.randn(1, 1, hop * 200) * 0.1
    y = x.clone()
    cut = hop * 120
    y[..., cut:] = torch.randn_like(y[..., cut:])
    with torch.no_grad():
        s1, o1 = est(x)
        s2, o2 = est(y)
    n = cut // hop  # frames 0 .. n-1 end before the cut
    assert torch.allclose(s1[..., :n], s2[..., :n], atol=1e-5)
    assert torch.allclose(o1[..., :n], o2[..., :n], atol=1e-5)
    assert not torch.allclose(s1[..., n:], s2[..., n:])


def test_synthetic_batch_has_labels():
    cfg = PitchConfig()
    b = make_batch(cfg, 4, 32768, "cpu")
    assert b.audio.shape == (4, 1, 32768)
    assert b.salience.shape == (4, cfg.n_bins, 32768 // cfg.hop)
    assert torch.isfinite(b.audio).all() and b.audio.abs().max() <= 1.0
    assert b.salience.max() > 0.9


def test_synth_is_causal_and_quiet_without_voices():
    torch.manual_seed(0)
    cfg = SynthConfig(ir_seconds=0.1)
    m = OrganSynth(cfg).eval()
    torch.nn.init.normal_(m.voice.out.weight, std=0.05)
    torch.nn.init.constant_(m.voice.out.bias, -3.0)
    fr = 64
    v = cfg.voices
    key = torch.zeros(1, v, fr)
    gate = torch.zeros(1, v, fr)
    on_age = torch.zeros(1, v, fr)
    off_age = -torch.ones(1, v, fr)
    key[0, 0, 10:] = 60.25
    gate[0, 0, 10:] = 1.0
    on_age[0, 0, 10:] = torch.arange(fr - 10) * cfg.hop / cfg.sample_rate
    cond = torch.zeros(1, cfg.cond_dim, fr)
    noise = band_noise(1, fr * cfg.hop, cfg.sample_rate, cfg.noise_bands, "cpu")
    with torch.no_grad():
        y = m.render(key, gate, on_age, off_age, cond, noise)
        key2 = key.clone()
        key2[0, 1, 40:] = 72.0
        gate2 = gate.clone()
        gate2[0, 1, 40:] = 1.0
        y2 = m.render(key2, gate2, on_age, off_age, cond, noise)
    # the reverb is an FFT convolution: its round-off reaches ~1e-7 everywhere
    assert y[..., : 10 * cfg.hop].abs().max() < 1e-6  # nothing before the first note
    assert y[..., 10 * cfg.hop:].abs().max() > 1e-3
    assert (y[..., : 40 * cfg.hop] - y2[..., : 40 * cfg.hop]).abs().max() < 1e-6  # no look-ahead
    assert (y - y2).abs().max() > 1e-3


def test_voice_frames_follow_the_notes():
    notes = assign_voices([{"on": 0.1, "off": 0.5, "midi": 60, "key": 60.2},
                           {"on": 0.2, "off": 0.4, "midi": 64, "key": 64.0}], 4, 0.5)
    hop = 480  # 10 ms
    key, gate, on_age, off_age = voice_frames(notes, 0, 100, hop, 4)
    assert key[0, 5] == 0 and gate[0, 5] == 0
    assert abs(key[0, 20] - 60.2) < 1e-6 and gate[0, 20] == 1 and abs(on_age[0, 20] - 0.1) < 1e-6
    assert gate[0, 60] == 0 and abs(off_age[0, 60] - 0.1) < 1e-6
    assert gate[1, 30] == 1 and key[1, 30] == 64.0


def test_tracker_starts_holds_and_releases_a_voice():
    cfg = PitchConfig()
    bm = cfg.bin_midi()
    tr = VoiceTracker(bm, voices=4, on=0.5, hold=12, confirm=2)
    sal = np.zeros(len(bm))
    i = int((60.0 - cfg.midi_lo) * 3)
    sal[i - 1: i + 2] = [0.4, 0.9, 0.4]
    ons = np.zeros(len(bm))
    m, g, s = tr.step(sal, ons)
    assert g.sum() == 0  # a peak has to last two frames
    m, g, s = tr.step(sal, ons)
    assert g.sum() == 1 and s.sum() == 1 and abs(m[g > 0][0] - 60.0) < 0.05
    m, g, s = tr.step(sal, ons)
    assert g.sum() == 1 and s.sum() == 0
    for _ in range(5):  # a short gap keeps the voice gated
        m, g, s = tr.step(np.zeros(len(bm)), ons)
    assert g.sum() == 1
    for _ in range(10):
        m, g, s = tr.step(np.zeros(len(bm)), ons)
    assert g.sum() == 0


def test_tracked_voices_are_causal_and_fold_octaves():
    midi = np.zeros((2, 40))
    gate = np.zeros((2, 40))
    started = np.zeros((2, 40), bool)
    midi[0, 10:30], gate[0, 10:30], started[0, 10] = 30.0, 1.0, True
    key, g, on_age, off_age = tracked_voices(fold(midi, 36, 84), gate, started, 64, 128, 20)
    assert g[0, 5] == 0 and g[0, 6] == 1  # estimator frame 10 ends at 703, synth frame 6 starts at 768
    assert key[0, 6] == 42.0  # folded up an octave
    assert g[0, 16] == 0 and off_age[0, 16] >= 0
