"""Reward functions. Milestone M5, and yours to build.

Start simple: progress per step, a penalty for dying, a bonus for finishing.
Only add shaping when a training curve tells you to, and when you do, watch
for the agent exploiting it instead of playing the level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gdrl.env.protocol import GameState


class RewardFn(ABC):
    def reset(self, state: GameState) -> None:
        """Called at the start of each episode with the first state."""

    @abstractmethod
    def __call__(self, prev: GameState, action: int, state: GameState) -> float:
        """Reward for the transition prev --action--> state."""


class ConstantReward(RewardFn):
    """Always 0. Plumbing only, for tests and smoke runs."""

    def __call__(self, prev: GameState, action: int, state: GameState) -> float:
        return 0.0


class ProgressReward(RewardFn):
    """``progress_scale`` per level-length travelled, ``death_penalty`` on death,
    ``completion_bonus`` on completion. (M5)"""

    def __init__(self, progress_scale: float = 1.0, death_penalty: float = -1.0, completion_bonus: float = 1.0):
        self.progress_scale = progress_scale
        self.death_penalty = death_penalty
        self.completion_bonus = completion_bonus

    def __call__(self, prev: GameState, action: int, state: GameState) -> float:
        # LEARN(M5): implement the three terms in the docstring.
        # - Measure progress in a unit that does not depend on level length?
        #   Or as a fraction of it? Think about levels of different lengths.
        # - Should the step you die on also get its progress?
        # Test: tests/learn/test_reward.py
        raise NotImplementedError("LEARN(M5): ProgressReward.__call__")


REWARDS: dict[str, type[RewardFn]] = {
    "constant": ConstantReward,
    "progress": ProgressReward,
}


def make_reward(name: str, **kwargs) -> RewardFn:
    try:
        return REWARDS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown reward {name!r}; choose from {sorted(REWARDS)}") from exc
