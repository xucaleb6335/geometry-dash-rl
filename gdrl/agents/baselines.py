"""Baselines: what your PPO has to beat (M8). Every row in the results table
needs one of these next to it, or a high score means nothing."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from gdrl.agents.base import ActOutput, Agent, Rollout


class RandomAgent(Agent):
    """Uniform random actions. The floor. Also the agent used by smoke tests."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 rollout_steps: int = 128, seed: int | None = None):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("RandomAgent expects a Discrete action space")
        self.n_actions = int(action_space.n)
        self.rollout_steps = rollout_steps
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        n = obs.shape[0]
        return ActOutput(action=self.rng.integers(self.n_actions, size=n, dtype=np.int64))

    def update(self, rollout: Rollout) -> dict[str, float]:
        return {}


class RuleBasedAgent(Agent):
    """Jump when the nearest obstacle ahead is within a distance band. (M8)

    Works on ``ObjectListEncoder`` observations, so it doubles as a check that
    your encoder carries enough information: if a 10-line rule can beat the
    level from the observation, a network can too. Tune the bands per object
    kind: a block needs an earlier jump than a spike.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, rollout_steps: int = 128):
        self.rollout_steps = rollout_steps
        # LEARN(M8): store whatever you need to decode the observation layout.
        raise NotImplementedError("LEARN(M8): RuleBasedAgent")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        raise NotImplementedError("LEARN(M8): RuleBasedAgent.act")

    def update(self, rollout: Rollout) -> dict[str, float]:
        return {}


class GeneticTimingAgent(Agent):
    """Evolve a fixed sequence of button presses for one level, ignoring the
    observation entirely. (M8)

    Pure memorization: it can beat a deterministic level and nothing else,
    which is exactly why it belongs in the comparison. Sketch: a population of
    press/release sequences, fitness = progress reached, keep the best, mutate,
    repeat. Decide whether this fits the act/update loop or wants its own
    script.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, rollout_steps: int = 128):
        raise NotImplementedError("LEARN(M8): GeneticTimingAgent")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        raise NotImplementedError

    def update(self, rollout: Rollout) -> dict[str, float]:
        raise NotImplementedError
