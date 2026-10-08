"""Neural network building blocks. Milestone M6, yours to build.

Reference: "The 37 Implementation Details of PPO", core details 2 and 13.
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def layer_init(layer: nn.Module, std: float = 2**0.5, bias_const: float = 0.0) -> nn.Module:
    """Orthogonal weights scaled by ``std``, constant bias. Returns the layer.

    Conventions: hidden layers ``std=sqrt(2)``, policy output ``std=0.01``
    (near-uniform initial policy), value output ``std=1``.
    """
    # LEARN(M6): use torch.nn.init.orthogonal_ and torch.nn.init.constant_.
    # Test: tests/learn/test_networks.py
    raise NotImplementedError("LEARN(M6): layer_init")


def mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int, out_std: float) -> nn.Sequential:
    """Linear -> Tanh -> ... -> Linear, every layer through ``layer_init``.

    The last layer uses ``out_std`` (0.01 for a policy head, 1.0 for a value head).
    """
    # LEARN(M6)
    raise NotImplementedError("LEARN(M6): mlp")


class GridCNN(nn.Module):
    """Small CNN for ``GridEncoder`` observations ``(C, H, W)`` -> features. (M8)

    Two or three conv layers are plenty for a 16 x 8 grid. Flatten, then a
    Linear to ``out_dim``.
    """

    def __init__(self, in_channels: int, height: int, width: int, out_dim: int = 128):
        super().__init__()
        # LEARN(M8): build the layers.
        raise NotImplementedError("LEARN(M8): GridCNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LEARN(M8): GridCNN.forward")
