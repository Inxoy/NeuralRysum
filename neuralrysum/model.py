"""NeuralRysum split-rate model.

Signal flow (all causal)::

    audio / MIDI exciter ──► FastEncoder ──► e_t  (hop 64, ~1.3 ms)  ─┐
                                 │ trunk features                      ▼
                                 ▼                                  Decoder ──► audio
                             SlowEncoder ──► z (hop 2048)  ──► FiLM ──┘
    knobs (z, descriptors) ─────────────────────────────────►

    Prior: autoregressive model over e_t given the conditioning, used when
    there is no input at all (free latent navigation).

Latency is set by the fast path only. The slow path decides *what it sounds
like* and may lag by a slow frame or two without being heard as latency.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import CausalConv1d, FiLMResBlock, ResBlock, SubPixelUp, act


class FastEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.convs = nn.ModuleList()
        self.res = nn.ModuleList()
        cin = 1
        for s, c in zip(cfg.enc_strides, cfg.enc_channels):
            self.convs.append(CausalConv1d(cin, c, 2 * s, stride=s))
            self.res.append(ResBlock(c, 1))
            cin = c
        self.to_e = CausalConv1d(cin, cfg.fast_dim, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) -> trunk features (B, C, T / hop)."""
        for i, (conv, res) in enumerate(zip(self.convs, self.res)):
            x = conv(x if i == 0 else act(x))
            x = res(x)
        return x

    def bottleneck(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.to_e(act(h)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.bottleneck(h), h


class SlowEncoder(nn.Module):
    """Trunk features -> timbre latent + predicted descriptors, at the slow rate."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList()
        cin = cfg.enc_channels[-1]
        for s in cfg.slow_strides:
            self.convs.append(CausalConv1d(cin, cfg.slow_channels, 2 * s, stride=s))
            cin = cfg.slow_channels
        self.gru = nn.GRU(cin, cin, batch_first=True)
        self.head = nn.Linear(cin, 2 * cfg.z_dim + cfg.n_desc)

    def forward(self, h: torch.Tensor):
        """h: (B, C, F) -> mean, logvar (B, z, S), desc (B, n_desc, S)."""
        x = h
        for conv in self.convs:
            x = conv(act(x))
        x, _ = self.gru(act(x).transpose(1, 2))
        out = self.head(x).transpose(1, 2)
        z = self.cfg.z_dim
        mean, logvar, desc = out[:, :z], out[:, z : 2 * z], out[:, 2 * z :]
        return mean, logvar.clamp(-10.0, 4.0), desc


class Conditioner(nn.Module):
    """[z, descriptors] -> FiLM (gamma, beta) for every decoder block."""

    def __init__(self, cfg: ModelConfig, block_channels: List[int]):
        super().__init__()
        self.block_channels = block_channels
        self.l1 = nn.Linear(cfg.cond_dim, cfg.cond_hidden)
        self.l2 = nn.Linear(cfg.cond_hidden, cfg.cond_hidden)
        self.head = nn.Linear(cfg.cond_hidden, 2 * sum(block_channels))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, c: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """c: (B, cond_dim, F) -> list of (gamma, beta), each (B, ch, F)."""
        h = act(self.l1(c.transpose(1, 2)))
        h = act(self.l2(h))
        p = self.head(h).transpose(1, 2)
        out, o = [], 0
        for ch in self.block_channels:
            out.append((1.0 + p[:, o : o + ch], p[:, o + ch : o + 2 * ch]))
            o += 2 * ch
        return out


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        chs, nd = cfg.dec_channels, len(cfg.dec_dilations)
        block_channels = [c for c in chs for _ in range(nd)]
        self.cond = Conditioner(cfg, block_channels)
        self.in_conv = CausalConv1d(cfg.fast_dim, chs[0], 3)
        self.ups = nn.ModuleList(
            SubPixelUp(chs[i], chs[i + 1], r) for i, r in enumerate(cfg.dec_strides)
        )
        self.blocks = nn.ModuleList(
            FiLMResBlock(c, d) for c in chs for d in cfg.dec_dilations
        )
        # white noise injected at the output rate, before the last blocks, so
        # the network can shape it into wind / chiff
        self.noise_w = nn.Parameter(torch.zeros(chs[-1]))
        self.out_conv = CausalConv1d(chs[-1], 1, 7)

    def forward(
        self, e: torch.Tensor, c: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """e: (B, fast_dim, F), c: (B, cond_dim, F), noise: (B, 1, F * hop) or None."""
        film = self.cond(c)
        nd = len(self.cfg.dec_dilations)
        x = self.in_conv(e)
        rate, bi = 1, 0
        for stage in range(len(self.cfg.dec_channels)):
            if stage > 0:
                x = self.ups[stage - 1](act(x))
                rate *= self.cfg.dec_strides[stage - 1]
            if stage == len(self.cfg.dec_channels) - 1:
                if noise is None:
                    noise = torch.randn(x.shape[0], 1, x.shape[-1], device=x.device)
                x = x + self.noise_w[None, :, None] * noise
            for _ in range(nd):
                g, b = film[bi]
                if rate > 1:
                    g = g.repeat_interleave(rate, dim=-1)
                    b = b.repeat_interleave(rate, dim=-1)
                x = self.blocks[bi](x, g, b)
                bi += 1
        return torch.tanh(self.out_conv(act(x)))


class Prior(nn.Module):
    """p(e_t | e_<t, c_t): a small GRU with a diagonal Gaussian output."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(cfg.fast_dim + cfg.cond_dim, cfg.prior_hidden, batch_first=True)
        self.head = nn.Linear(cfg.prior_hidden, 2 * cfg.fast_dim)

    def forward(self, e_prev: torch.Tensor, c: torch.Tensor, h=None):
        x = torch.cat([e_prev, c], dim=1).transpose(1, 2)
        y, h = self.gru(x, h)
        out = self.head(y).transpose(1, 2)
        d = self.cfg.fast_dim
        return out[:, :d], out[:, d:].clamp(-7.0, 2.0), h

    def nll(self, e: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Teacher-forced negative log-likelihood of a code sequence."""
        e_prev = torch.cat([torch.zeros_like(e[..., :1]), e[..., :-1]], dim=-1)
        mu, logsig, _ = self.forward(e_prev, c)
        return (0.5 * ((e - mu) / logsig.exp()) ** 2 + logsig).mean()

    @torch.no_grad()
    def generate(self, c: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """c: (B, cond_dim, F) -> sampled codes (B, fast_dim, F)."""
        b, _, f = c.shape
        e_prev = torch.zeros(b, self.cfg.fast_dim, 1, device=c.device)
        h, out = None, []
        for t in range(f):
            mu, logsig, h = self.forward(e_prev, c[..., t : t + 1], h)
            e_prev = mu + temperature * logsig.exp() * torch.randn_like(mu)
            e_prev = e_prev.clamp(-1.0, 1.0)
            out.append(e_prev)
        return torch.cat(out, dim=-1)


class NeuralRysum(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.cfg.validate()
        self.fast = FastEncoder(self.cfg)
        self.slow = SlowEncoder(self.cfg)
        self.decoder = Decoder(self.cfg)
        self.prior = Prior(self.cfg)
        self._init_weights()

    def _init_weights(self) -> None:
        """Variance-preserving init.

        There is no normalisation layer anywhere (they cost latency or CPU),
        so with PyTorch's default init the signal shrinks at every layer until
        the fast code is practically constant and nothing is learned. He init
        for the leaky ReLU keeps its level; the second conv of each residual
        block starts small so that blocks begin close to identity.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, (ResBlock, FiLMResBlock)):
                with torch.no_grad():
                    m.conv2.conv.weight.mul_(0.1)
        # Start from (almost) silence. Otherwise the first thing training does
        # is turn a loud random output down, and it does so by shrinking every
        # layer on the way, which can leave the fast path dead for good.
        with torch.no_grad():
            self.decoder.out_conv.conv.weight.mul_(0.01)

    # ---- helpers -----------------------------------------------------------
    def hold(self, c_slow: torch.Tensor) -> torch.Tensor:
        """Slow-rate conditioning -> one value per fast frame (sample & hold)."""
        return c_slow.repeat_interleave(self.cfg.slow_ratio, dim=-1)

    @staticmethod
    def delay_one(z: torch.Tensor) -> torch.Tensor:
        """Shift by one slow frame.

        A slow frame is only known once its 2048 samples have arrived, so the
        decoder may only ever use the *previous* one. Training has to see the
        same delay the plugin has.
        """
        return torch.cat([torch.zeros_like(z[..., :1]), z[..., :-1]], dim=-1)

    # ---- training forward --------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        desc: torch.Tensor,
        x_fast: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        x:      (B, 1, T) clean target audio, T a multiple of ``slow_hop``.
        desc:   (B, n_desc, T / slow_hop) descriptors measured on ``x``.
        x_fast: timbre-perturbed copy of x fed to the fast path (None = x).
        """
        e, h = self.fast(x if x_fast is None else x_fast)
        h_clean = h if x_fast is None else self.fast.trunk(x)
        mean, logvar, desc_pred = self.slow(h_clean)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        c = self.hold(torch.cat([self.delay_one(z), desc], dim=1))
        y = self.decoder(e, c, noise)
        kl = 0.5 * (mean**2 + logvar.exp() - 1.0 - logvar).mean()
        return {
            "y": y,
            "e": e,
            "c": c,
            "mean": mean,
            "kl": kl,
            "desc_pred": desc_pred,
        }

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def transfer(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        desc: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Audio in, audio out.

        z / desc: (cond dims,) fixed manual timbre. If None, follow the input
        (slow encoder, delayed by one slow frame, predicted descriptors).
        """
        e, h = self.fast(x)
        if z is None or desc is None:
            mean, _, dp = self.slow(h)
            c_slow = self.delay_one(torch.cat([mean, dp], dim=1))
            if z is not None:
                c_slow[:, : self.cfg.z_dim] = z.view(1, -1, 1)
            if desc is not None:
                c_slow[:, self.cfg.z_dim :] = desc.view(1, -1, 1)
            c = self.hold(c_slow)
        else:
            c = torch.cat([z, desc]).view(1, -1, 1).expand(x.shape[0], -1, e.shape[-1])
        return self.decoder(e, c, noise)

    @torch.no_grad()
    def free(self, c: torch.Tensor, temperature: float = 1.0, noise=None) -> torch.Tensor:
        """No input: the prior invents the fast code. c: (B, cond_dim, F)."""
        return self.decoder(self.prior.generate(c, temperature), c, noise)
