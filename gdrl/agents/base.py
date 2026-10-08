"""The brain's contract. ``train.py`` and ``evaluate.py`` only ever call this.

Anything that implements ``Agent`` (your PPO, a DQN, a hand-written rule, a
genetic algorithm) plugs into the same loop with a config change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ActOutput:
    """What ``act`` returns for a batch of N observations.

    ``extras`` carries anything the agent needs again at update time, one
    array of shape ``(N, ...)`` per key, e.g. PPO's ``logprob`` and ``value``.
    The train loop stores them in the rollout untouched.
    """

    action: np.ndarray  # (N,) int64
    extras: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Rollout:
    """``T`` steps from ``N`` vector envs, collected by the train loop.

    Indexing: row ``t`` is the transition ``obs[t] --actions[t]--> (rewards[t], ...)``.

    - ``terminated[t]``: the episode really ended at step t (death, completion,
      CartPole falling). The value after it is 0.
    - ``truncated[t]``: cut off by the step limit. Not a real end.
    - ``final_obs[t]``: the last observation of the episode that ended at step t,
      valid where ``terminated[t] | truncated[t]``. Needed to bootstrap
      truncated episodes, because with same-step autoreset ``obs[t + 1]``
      already belongs to the next episode.
    - ``next_obs``: the observation after the last step, to bootstrap step T-1.

    The value of the state after step t is therefore:
      0                    if terminated[t]
      V(final_obs[t])      if truncated[t]
      V(obs[t + 1])        otherwise, for t < T - 1
      V(next_obs)          otherwise, for t = T - 1
    """

    obs: np.ndarray  # (T, N, *obs_shape)
    actions: np.ndarray  # (T, N)
    rewards: np.ndarray  # (T, N) float32
    terminated: np.ndarray  # (T, N) bool
    truncated: np.ndarray  # (T, N) bool
    final_obs: np.ndarray  # (T, N, *obs_shape), zero where not done
    next_obs: np.ndarray  # (N, *obs_shape)
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # each (T, N, ...)

    @property
    def num_steps(self) -> int:
        return self.rewards.shape[0]

    @property
    def num_envs(self) -> int:
        return self.rewards.shape[1]

    @property
    def dones(self) -> np.ndarray:
        return self.terminated | self.truncated


class Agent(ABC):
    """Base class for every brain."""

    #: Steps per env collected before each ``update`` call.
    rollout_steps: int = 128

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> ActOutput:
        """Choose actions for a batch of observations ``(N, *obs_shape)``.

        ``deterministic=True`` is for evaluation: take the most likely action.
        """

    @abstractmethod
    def update(self, rollout: Rollout) -> dict[str, float]:
        """Learn from one rollout. Return scalar metrics to log."""

    def save(self, path: str | Path) -> None:
        """Write a checkpoint. Agents without parameters can skip this."""

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint written by ``save``."""
