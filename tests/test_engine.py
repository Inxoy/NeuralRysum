"""The C++ engine must produce the same audio as the PyTorch model.

Compiles engine/tools/nrys_cli.cpp with whatever C++ compiler is on PATH and
compares outputs sample by sample. Skipped when no compiler is found.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from neuralrysum import NeuralRysum
from neuralrysum.export import export

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++") or shutil.which("clang++")
pytestmark = pytest.mark.skipif(CXX is None, reason="no C++ compiler on PATH")
TOL = 2e-4


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("engine")
    cli = d / "nrys_cli"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", "-I", str(ROOT / "engine/include"),
         str(ROOT / "engine/tools/nrys_cli.cpp"), "-o", str(cli)],
        check=True,
    )
    torch.manual_seed(1)
    model = NeuralRysum().eval()
    # untrained weights, but with every path active
    torch.nn.init.normal_(model.decoder.cond.head.weight, std=0.05)
    torch.nn.init.normal_(model.decoder.noise_w, std=0.1)
    path = d / "model.nrys"
    export(model, str(path))
    return d, cli, model, path


def run(d, cli, path, x, mode, cond, block):
    (d / "in.f32").write_bytes(x.astype("<f4").tobytes())
    (d / "cond.f32").write_bytes(cond.astype("<f4").tobytes())
    r = subprocess.run([str(cli), "parity", str(path), str(d / "in.f32"), str(d / "out.f32"),
                        mode, str(d / "cond.f32"), str(block)], check=True, capture_output=True, text=True)
    latency = int(r.stdout.split()[-1])
    return np.frombuffer((d / "out.f32").read_bytes(), dtype="<f4"), latency


@pytest.mark.parametrize("block", [64, 512])
def test_manual_matches_pytorch_with_zero_added_latency(setup, block):
    d, cli, model, path = setup
    cfg = model.cfg
    x = torch.randn(cfg.slow_hop * 8) * 0.2
    c = torch.randn(cfg.cond_dim)
    ref = model.transfer(x[None, None], c[: cfg.z_dim], c[cfg.z_dim :], torch.zeros(1, 1, len(x)))[0, 0]
    y, latency = run(d, cli, path, x.numpy(), "manual", c.numpy(), block)
    assert latency == 0
    assert np.abs(y - ref.numpy()).max() < TOL


@pytest.mark.parametrize("block", [1, 37, 480])
def test_any_block_size_gives_exactly_one_hop_of_latency(setup, block):
    d, cli, model, path = setup
    cfg = model.cfg
    x = torch.randn(cfg.slow_hop * 4) * 0.2
    c = torch.randn(cfg.cond_dim)
    ref = model.transfer(x[None, None], c[: cfg.z_dim], c[cfg.z_dim :], torch.zeros(1, 1, len(x)))[0, 0]
    y, latency = run(d, cli, path, x.numpy(), "manual", c.numpy(), block)
    assert latency == cfg.hop
    assert np.abs(y[cfg.hop :] - ref.numpy()[: -cfg.hop]).max() < TOL
    assert np.abs(y[: cfg.hop]).max() == 0


def test_follow_mode_matches_pytorch(setup):
    d, cli, model, path = setup
    cfg = model.cfg
    x = torch.randn(cfg.slow_hop * 8) * 0.2
    ref = model.transfer(x[None, None], noise=torch.zeros(1, 1, len(x)))[0, 0]
    y, _ = run(d, cli, path, x.numpy(), "follow", np.zeros(cfg.cond_dim), 256)
    assert np.abs(y - ref.numpy()).max() < TOL


def test_free_mode_matches_pytorch_at_temperature_zero(setup):
    d, cli, model, path = setup
    cfg = model.cfg
    frames = 200
    c = torch.randn(cfg.cond_dim)
    cf = c.view(1, -1, 1).expand(1, -1, frames).contiguous()
    ref = model.free(cf, temperature=0.0, noise=torch.zeros(1, 1, frames * cfg.hop))[0, 0]
    y, _ = run(d, cli, path, np.zeros(frames * cfg.hop), "free", c.numpy(), 64)
    assert np.abs(y - ref.numpy()).max() < TOL


def test_meantone_tuning(setup):
    d, cli, _, path = setup
    out = subprocess.run([str(cli), "tuning", str(path)], check=True, capture_output=True, text=True).stdout
    rows = {int(a): (float(b), float(c)) for a, b, c in (l.split() for l in out.strip().splitlines())}
    eq, mt = rows[60][0], rows[60][1]
    assert abs(rows[69][1] - 440.0) < 1e-3  # A4 pinned
    assert abs(rows[64][1] / mt - 5 / 4) < 1e-5  # C-E is a pure major third
    assert abs(rows[67][1] / mt - 5 ** 0.25) < 1e-5  # C-G is a quarter-comma fifth
    assert abs(rows[72][1] / mt - 2.0) < 1e-5
    assert abs(rows[64][0] / eq - 2 ** (4 / 12)) < 1e-5  # equal temperament for comparison


def test_midi_render_is_finite_and_sounds(setup):
    d, cli, _, path = setup
    subprocess.run([str(cli), "midi", str(path), str(d / "midi.f32"), "meantone"], check=True)
    y = np.frombuffer((d / "midi.f32").read_bytes(), dtype="<f4")
    assert np.isfinite(y).all() and np.abs(y).max() <= 1.0
    assert np.abs(y[: len(y) // 2]).max() > 0
