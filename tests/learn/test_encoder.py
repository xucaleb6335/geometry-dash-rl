"""M5: encoder properties. These test what any good encoder must do, not one
particular layout, so they stay valid as you change your design."""

import numpy as np
import pytest

from gdrl.env.protocol import GameState, ObjectKind
from gdrl.features.encoder import GridEncoder, ObjectListEncoder
from tests.conftest import make_objects, make_state

ENCODERS = [ObjectListEncoder, GridEncoder]


def shifted(state: GameState, dx: float) -> GameState:
    objs = state.objects.copy()
    objs["x"] += dx
    return GameState(**{**state.__dict__, "player_x": state.player_x + dx, "objects": objs})


@pytest.mark.parametrize("cls", ENCODERS)
def test_shape_dtype_bounds(cls):
    enc = cls()
    obs = enc.encode(make_state())
    assert obs.shape == enc.observation_space.shape
    assert obs.dtype == np.float32
    assert enc.observation_space.contains(obs)


@pytest.mark.parametrize("cls", ENCODERS)
def test_translation_invariant(cls):
    """Same situation further along the level must encode identically: no absolute x."""
    enc = cls()
    state = make_state()
    np.testing.assert_allclose(enc.encode(state), enc.encode(shifted(state, 1234.0)), atol=1e-5)


@pytest.mark.parametrize("cls", ENCODERS)
def test_sees_obstacles(cls):
    enc = cls()
    empty = enc.encode(make_state(objects=make_objects()))
    one_spike = enc.encode(make_state(objects=make_objects((560.0, 9.0, 9.0, 18.0, ObjectKind.HAZARD))))
    assert not np.allclose(empty, one_spike)


@pytest.mark.parametrize("cls", ENCODERS)
def test_distance_matters(cls):
    enc = cls()
    near = enc.encode(make_state(objects=make_objects((560.0, 9.0, 9.0, 18.0, ObjectKind.HAZARD))))
    far = enc.encode(make_state(objects=make_objects((800.0, 9.0, 9.0, 18.0, ObjectKind.HAZARD))))
    assert not np.allclose(near, far)


@pytest.mark.parametrize("cls", ENCODERS)
def test_kind_matters(cls):
    enc = cls()
    spike = enc.encode(make_state(objects=make_objects((575.0, 15.0, 30.0, 30.0, ObjectKind.HAZARD))))
    block = enc.encode(make_state(objects=make_objects((575.0, 15.0, 30.0, 30.0, ObjectKind.SOLID))))
    assert not np.allclose(spike, block)


def test_object_list_ignores_input_order():
    enc = ObjectListEncoder()
    state = make_state()
    reversed_objs = GameState(**{**state.__dict__, "objects": state.objects[::-1].copy()})
    np.testing.assert_allclose(enc.encode(state), enc.encode(reversed_objs))
