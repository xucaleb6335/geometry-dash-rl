import numpy as np
import pytest

from gdrl.env.mock_game import MockGame, generate_level
from gdrl.env.protocol import MAX_OBJECTS, GameState
from tests.conftest import scripted_jumper


def run(game: MockGame, policy, ticks_per_step: int = 1) -> GameState:
    state = game.reset()
    while not state.terminal:
        state = game.step(policy(state), ticks_per_step)
    return state


def test_same_actions_same_bytes():
    rng = np.random.default_rng(0)
    actions = rng.integers(2, size=400).astype(bool)
    streams = []
    for _ in range(2):
        game = MockGame(generate_level(5))
        game.reset()
        streams.append(b"".join(game.step(bool(a), 4).to_bytes() for a in actions))
    assert streams[0] == streams[1]


def test_never_jumping_dies_on_first_obstacle():
    game = MockGame(generate_level(3))
    state = run(game, lambda s: False, 4)
    assert state.dead and not state.completed
    assert state.player_x < game.level.left[0] + 30


@pytest.mark.parametrize("ticks", [1, 4])
@pytest.mark.parametrize("seed", range(1, 21))
def test_every_level_is_beatable(seed, ticks):
    state = run(MockGame(generate_level(seed)), scripted_jumper, ticks)
    assert state.completed, f"seed {seed} died at x={state.player_x:.1f}"


def test_terminal_state_is_sticky():
    game = MockGame(generate_level(3))
    dead = run(game, lambda s: False, 4)
    again = game.step(True, 10)
    assert again.tick == dead.tick and again.dead


def test_objects_sorted_and_capped():
    level = generate_level(7, n_obstacles=60)
    state = MockGame(level).reset()
    assert len(state.objects) == MAX_OBJECTS
    left = state.objects["x"] - state.objects["w"] / 2
    assert np.all(np.diff(left) >= 0)


def test_holding_jumps_and_lands():
    game = MockGame(generate_level(1))
    game.reset()
    up = game.step(True, 20)
    assert up.player_y > 15 and not up.on_ground
    down = game.step(False, 200)
    assert down.on_ground or down.terminal
