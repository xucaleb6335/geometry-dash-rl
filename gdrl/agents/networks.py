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
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int, out_std: float) -> nn.Sequential:
    """Linear -> Tanh -> ... -> Linear, every layer through ``layer_init``.

    The last layer uses ``out_std`` (0.01 for a policy head, 1.0 for a value head).
    """
    sizes = [in_dim, *hidden_sizes]
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [layer_init(nn.Linear(a, b)), nn.Tanh()]
    layers.append(layer_init(nn.Linear(sizes[-1], out_dim), std=out_std))
    return nn.Sequential(*layers)


class GridCNN(nn.Module):
    """Small CNN for ``GridEncoder`` observations ``(C, H, W)`` -> features. (M8)

    Two or three conv layers are plenty for a 16 x 8 grid. Flatten, then a
    Linear to ``out_dim``.
    """

    def __init__(self, in_channels: int, height: int, width: int, out_dim: int = 128):
        super().__init__()
        # Stride 1 with padding keeps the grid's resolution: one cell is one
        # block, and downsampling a 16 x 8 grid would blur where a spike is.
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * height * width, out_dim)),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
