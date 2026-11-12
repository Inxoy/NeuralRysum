"""Strictly causal building blocks.

Every layer here has zero look-ahead: output frame ``i`` depends only on input
samples ``<= (i + 1) * stride - 1``. That is what makes the offline (training)
graph identical to the streaming graph in the C++ engine, and it is checked in
``tests/test_causality.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAK = 0.2


def act(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, LEAK)


class CausalConv1d(nn.Module):
    """Conv1d with left zero padding only.

    padding = (k - 1) * dilation + 1 - stride, so that for an input whose
    length is a multiple of ``stride`` the output has length ``T // stride``.
    """

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        assert stride == 1 or dilation == 1
        self.pad = (k - 1) * dilation + 1 - stride
        assert self.pad >= 0, "kernel must be >= stride"
        self.conv = nn.Conv1d(cin, cout, k, stride=stride, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class SubPixelUp(nn.Module):
    """Causal upsampling by ``r``: conv to ``cout * r`` channels, then interleave.

    Output channel ``c`` at time ``t * r + j`` is conv channel ``c * r + j`` at
    frame ``t``. No transposed convolutions, so no overlap-add state to carry.
    """

    def __init__(self, cin: int, cout: int, r: int, k: int = 3):
        super().__init__()
        self.r = r
        self.cout = cout
        self.conv = CausalConv1d(cin, cout * r, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)  # (B, cout * r, T)
        b, _, t = y.shape
        y = y.view(b, self.cout, self.r, t).permute(0, 1, 3, 2)
        return y.reshape(b, self.cout, t * self.r)


class FiLMResBlock(nn.Module):
    """Residual block whose inner activation is modulated by the timbre path.

    ``x + conv2(act(gamma * conv1(act(x)) + beta))``

    gamma / beta come from the slow path, one pair per fast frame, and are
    held constant across the samples of that frame.
    """

    def __init__(self, ch: int, dilation: int, k: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, k)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.conv1(act(x))
        h = gamma * h + beta
        return x + self.conv2(act(h))


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(act(self.conv1(act(x))))
