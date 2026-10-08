"""M6: layer_init and mlp."""

import torch
from torch import nn

from gdrl.agents.networks import GridCNN, layer_init, mlp


def test_layer_init_is_orthogonal_with_constant_bias():
    layer = nn.Linear(64, 64)
    out = layer_init(layer, std=2**0.5, bias_const=0.0)
    assert out is layer
    w = layer.weight.detach()
    torch.testing.assert_close(w @ w.T, 2.0 * torch.eye(64), atol=1e-4, rtol=1e-4)
    assert torch.all(layer.bias == 0)


def test_mlp_shapes_and_head_scale():
    net = mlp(8, (64, 64), 2, out_std=0.01)
    assert net(torch.zeros(5, 8)).shape == (5, 2)
    assert any(isinstance(m, nn.Tanh) for m in net.modules())
    last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
    assert last.weight.abs().max() < 0.05  # policy head starts near-uniform


def test_grid_cnn_shape():
    net = GridCNN(in_channels=3, height=8, width=16, out_dim=128)
    assert net(torch.zeros(4, 3, 8, 16)).shape == (4, 128)
