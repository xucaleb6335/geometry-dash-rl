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
    ``completion_bonus`` on completion. (M5)

    Progress is the change in ``GameState.progress`` (fraction of the level,
    clipped to [0, 1]), so a whole level is worth ``progress_scale`` whatever
    its length, and the three terms stay on the same scale. The step you die
    on still gets the progress it made; the penalty is added on top. The
    action is ignored: no reward for pressing or not pressing.
    """

    def __init__(self, progress_scale: float = 1.0, death_penalty: float = -1.0, completion_bonus: float = 1.0):
        self.progress_scale = progress_scale
        self.death_penalty = death_penalty
        self.completion_bonus = completion_bonus

    def __call__(self, prev: GameState, action: int, state: GameState) -> float:
        reward = self.progress_scale * (state.progress - prev.progress)
        if state.dead:
            reward += self.death_penalty
        if state.completed:
            reward += self.completion_bonus
        return reward


REWARDS: dict[str, type[RewardFn]] = {
    "constant": ConstantReward,
    "progress": ProgressReward,
}


def make_reward(name: str, **kwargs) -> RewardFn:
    try:
        return REWARDS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown reward {name!r}; choose from {sorted(REWARDS)}") from exc
