"""M5: ProgressReward with its default coefficients."""

from gdrl.env.protocol import StateFlag
from gdrl.rewards.reward import ProgressReward
from tests.conftest import make_state


def test_forward_progress_is_positive():
    reward = ProgressReward()
    prev, now = make_state(player_x=500.0), make_state(player_x=505.2)
    reward.reset(prev)
    assert reward(prev, 0, now) > 0


def test_death_is_negative():
    reward = ProgressReward()
    prev = make_state(player_x=500.0)
    dead = make_state(player_x=505.2, flags=StateFlag.DEAD)
    reward.reset(prev)
    assert reward(prev, 1, dead) < 0


def test_completion_beats_plain_progress():
    reward = ProgressReward()
    prev = make_state(player_x=2994.8)
    plain = make_state(player_x=3000.0)
    done = make_state(player_x=3000.0, flags=StateFlag.COMPLETED)
    reward.reset(prev)
    assert reward(prev, 0, done) > reward(prev, 0, plain)


def test_action_alone_does_not_change_reward():
    reward = ProgressReward()
    prev, now = make_state(player_x=500.0), make_state(player_x=505.2)
    reward.reset(prev)
    assert reward(prev, 0, now) == reward(prev, 1, now)
