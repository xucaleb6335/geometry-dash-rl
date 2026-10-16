"""Start-position curriculum wrapper (gdrl/curriculum.py) on the mock game."""

import csv
import dataclasses

import numpy as np

from gdrl.curriculum import StartCurriculum
from gdrl.env.backends import MockBackend
from gdrl.env.gd_env import GDEnv
from gdrl.env.mock_game import StartPos
from gdrl.features.encoder import ObjectListEncoder
from gdrl.rewards.reward import ProgressReward


def make_env(xs):
    backend = MockBackend(level_seed=1)
    level = backend.game.level
    backend.game.level = dataclasses.replace(level, start_positions=tuple(StartPos(x=x) for x in xs))
    return GDEnv(backend, ObjectListEncoder(), ProgressReward(), max_steps=50)


def test_forced_start_and_logging(tmp_path):
    xs = [300.0, 600.0]
    env = StartCurriculum(make_env(xs), [{"x": 0.0}, {"start_pos": 0, "x": xs[0]}, {"start_pos": 1, "x": xs[1]}],
                          log_path=tmp_path / "ep.csv")
    for i, x in enumerate([None, *xs]):
        _, info = env.reset(options={"start": i})
        assert info["start"] == i
        if x is not None:
            assert env.unwrapped.state.player_x == x
        done = False
        while not done:
            _, _, term, trunc, info = env.step(0)
            done = term or trunc
    rows = list(csv.DictReader(open(tmp_path / "ep.csv")))
    assert [int(r["start"]) for r in rows] == [0, 1, 2]


def test_weights_favour_unsolved_starts():
    env = StartCurriculum(make_env([300.0]), [{"x": 0.0}, {"start_pos": 0, "x": 300.0}], p_first=0.2, min_weight=0.1)
    assert np.allclose(env.weights(), [0.6, 0.4])
    env.success[:] = [0.0, 1.0]  # start 1 always reaches the end: sampled at min_weight
    p = env.weights()
    assert np.isclose(p.sum(), 1.0) and p[0] > 0.9


def test_frontier_gating_opens_starts_as_the_agent_advances():
    starts = [{"x": 0.0}] + [{"start_pos": i, "x": x} for i, x in enumerate([300.0, 600.0, 900.0, 1200.0])]
    env = StartCurriculum(make_env([300.0, 600.0, 900.0, 1200.0]), starts, frontier_extra=1)
    assert env.num_open() == 3  # start 0, the first unsolved start past reach 0 (x=300), one more
    assert env.weights()[3:].sum() == 0
    env.recent_reach = [650.0]  # the level start now gets past x=600
    assert env.num_open() == 5  # first unsolved past 650 is x=900 (index 3), plus one more
    env.success[3] = 0.9  # x=900 solved: first unsolved past the reach is x=1200
    assert env.num_open() == 5


def test_snapshot_starts_are_added_and_exact():
    backend = MockBackend(level_seed=1)
    env = StartCurriculum(GDEnv(backend, ObjectListEncoder(), ProgressReward(), max_steps=400),
                          [{"start_pos": -1, "x": 15.0}], snapshot_every=100.0, snapshot_confirm=5, seed=0)
    from gdrl.agents.baselines import RuleBasedAgent  # survives the mock levels
    agent = RuleBasedAgent(env.observation_space, env.action_space)
    for _ in range(3):
        obs, _ = env.reset(options={"start": 0})
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(int(agent.act(obs[None], deterministic=True).action[0]))
            done = term or trunc
    snaps = [s for s in env.starts if "slot" in s]
    assert snaps, "no snapshot start was added"
    xs = sorted(s["x"] for s in snaps)
    assert all(b - a >= 50.0 for a, b in zip(xs, xs[1:]))
    k = env.starts.index(snaps[0])
    env.reset(options={"start": k})
    assert abs(env.unwrapped.state.player_x - snaps[0]["x"]) < 1e-3
    assert len(env.success) == len(env.starts)


def test_unpassed_snapshot_start_is_retired():
    backend = MockBackend(level_seed=1)
    env = StartCurriculum(GDEnv(backend, ObjectListEncoder(), ProgressReward(), max_steps=400),
                          [{"start_pos": -1, "x": 15.0}], snapshot_every=100.0, snapshot_confirm=5,
                          snapshot_retire_after=3, seed=0)
    from gdrl.agents.baselines import RuleBasedAgent
    agent = RuleBasedAgent(env.observation_space, env.action_space)
    obs, _ = env.reset(options={"start": 0})
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(int(agent.act(obs[None], deterministic=True).action[0]))
        done = term or trunc
    # the furthest snapshot: nothing ahead of it, so only completing the level counts
    k = max((i for i, s in enumerate(env.starts) if "slot" in s), key=lambda i: env.starts[i]["x"])
    env.snapshot_every = None  # no new snapshots ahead of it during the test
    for _ in range(3):  # hold the button: dies before the end
        env.reset(options={"start": k})
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(1)
            done = term or trunc
    assert env.starts[k].get("retired")
    assert env.weights()[k] == 0.0
