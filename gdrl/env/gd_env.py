"""The Gymnasium environment. Composes a Backend, an Encoder and a RewardFn.

Swap any of the three without touching the others:

    env = GDEnv(MockBackend(), ObjectListEncoder(), ProgressReward())
    env = GDEnv(SocketBackend(port=5555), ObjectListEncoder(), ProgressReward())
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gdrl.env.backends import Backend
from gdrl.env.protocol import GameState
from gdrl.features.encoder import Encoder
from gdrl.rewards.reward import RewardFn


class GDEnv(gym.Env):
    """One Geometry Dash game as an RL environment.

    Action: ``Discrete(2)``, 0 = button released, 1 = held, kept for
    ``frame_skip`` physics ticks (4 ticks at 240 TPS = 60 decisions a second).

    Episode rules:
      - ``terminated`` when the player dies or completes the level. These are
        real ends: the value of what comes after is 0.
      - ``truncated`` when ``max_steps`` decisions pass without either. Not a
        real end: PPO should bootstrap from the value of the final state.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        backend: Backend,
        encoder: Encoder,
        reward_fn: RewardFn,
        *,
        frame_skip: int = 4,
        max_steps: int = 5000,
        level_ids: list[int] | None = None,
        start_pos: int = -1,
        render_mode: str | None = None,
    ):
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.backend = backend
        self.encoder = encoder
        self.reward_fn = reward_fn
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.level_ids = list(level_ids) if level_ids else [0]
        self.start_pos = start_pos
        self.render_mode = render_mode

        self.observation_space = encoder.observation_space
        self.action_space = spaces.Discrete(2)

        self._state: GameState | None = None
        self._steps = 0
        self._level_id = self.level_ids[0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "level_id" in options:
            self._level_id = int(options["level_id"])
        else:
            self._level_id = int(self.np_random.choice(self.level_ids))
        start_pos = int(options.get("start_pos", self.start_pos))

        if "load_slot" in options:  # a saved snapshot as the start (protocol v2)
            self._state = self.backend.load_state(int(options["load_slot"]))
        else:
            self._state = self.backend.reset(self._level_id, start_pos)
        self._steps = 0
        self.encoder.reset(self._state)
        self.reward_fn.reset(self._state)
        return self.encoder.encode(self._state), self._info()

    def step(self, action):
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        prev = self._state
        self._state = self.backend.step(bool(action), self.frame_skip)
        self._steps += 1

        reward = float(self.reward_fn(prev, int(action), self._state))
        terminated = self._state.terminal
        truncated = (not terminated) and self._steps >= self.max_steps
        return self.encoder.encode(self._state), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        s = self._state
        return {
            "level_id": self._level_id,
            "tick": s.tick,
            "progress": s.progress,
            "dead": s.dead,
            "completed": s.completed,
        }

    def render(self):
        if self.render_mode != "ansi" or self._state is None:
            return None
        s = self._state
        return (
            f"tick={s.tick:6d} x={s.player_x:8.1f} y={s.player_y:6.1f} vy={s.vel_y:7.1f} "
            f"ground={int(s.on_ground)} hold={int(s.holding)} objs={len(s.objects)} "
            f"progress={s.progress:5.1%}{' DEAD' if s.dead else ''}{' DONE' if s.completed else ''}"
        )

    def close(self):
        self.backend.close()

    @property
    def state(self) -> GameState | None:
        """The raw GameState behind the last observation (for debugging and baselines)."""
        return self._state
