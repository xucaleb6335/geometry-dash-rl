"""Raw GameState -> observation vector. Milestone M5, and yours to build.

The encoder decides what the agent can see, which decides whether it
generalizes or memorizes. Two rules for every encoder here:

1. Everything is relative to the player. No absolute x, no progress, no tick
   count: with those the policy can learn "at x = 1234, jump" and nothing else.
2. Every value lands in [-1, 1] (or [0, 1] for the grid), so the network
   never sees raw game units in the hundreds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces

from gdrl.env.mock_game import JUMP_VELOCITY
from gdrl.env.protocol import BLOCK, GameState


class Encoder(ABC):
    """Maps a GameState to a fixed-shape float32 array inside ``observation_space``."""

    observation_space: spaces.Box

    def reset(self, state: GameState) -> None:
        """Called at the start of each episode. Override if you keep history."""

    @abstractmethod
    def encode(self, state: GameState) -> np.ndarray: ...


class DebugEncoder(Encoder):
    """Player-only features: y, vertical speed, on-ground.

    Plumbing only. It cannot see obstacles, so nothing trained on it will beat
    a level. It exists so the env, tests and train loop run before your real
    encoder does.
    """

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def encode(self, state: GameState) -> np.ndarray:
        obs = np.array(
            [state.player_y / (10 * BLOCK), state.vel_y / (2 * JUMP_VELOCITY), float(state.on_ground)],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)


class ObjectListEncoder(Encoder):
    """Observation v1: player features plus the next ``k`` obstacles. (M5)

    Layout, length ``4 + 5 * k``:

    - ``[0:4]`` player: y, vel_y, on_ground, upside_down
    - then ``k`` slots of ``(dx, dy, w, h, kind)`` for the nearest objects
      ahead, sorted by dx. ``dx``/``dy`` are relative to the player. Empty
      slots are all zero, so pick an encoding where "no object" cannot be
      confused with a real one (hint: kind).

    Normalize with fixed constants (e.g. ``max_dx_blocks``), not per-batch
    statistics, so the same state always encodes the same way.
    """

    def __init__(self, k: int = 8, max_dx_blocks: float = 20.0, max_dy_blocks: float = 10.0) -> None:
        self.k = k
        self.max_dx = max_dx_blocks * BLOCK
        self.max_dy = max_dy_blocks * BLOCK
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4 + 5 * k,), dtype=np.float32)

    def encode(self, state: GameState) -> np.ndarray:
        # LEARN(M5): build the vector described in the class docstring.
        # - Which objects count as "ahead"? Keep the one you are passing over.
        # - How do you encode kind as a number in [-1, 1]? Or use one-hot and
        #   change the layout (update observation_space and the docstring too).
        # Test: tests/learn/test_encoder.py
        raise NotImplementedError("LEARN(M5): ObjectListEncoder.encode")


class GridEncoder(Encoder):
    """Observation v2: hitboxes rasterized into a grid ahead of the player. (M5, for a CNN)

    Shape ``(channels, height, width)`` with one channel per kind
    (solid, hazard, orb/pad). Cell size is one block. Column 0 starts one block
    behind the player; row 0 is ``rows_below`` blocks under the player.
    """

    def __init__(self, width_blocks: int = 16, height_blocks: int = 8, rows_below: int = 2, channels: int = 3):
        self.width = width_blocks
        self.height = height_blocks
        self.rows_below = rows_below
        self.channels = channels
        self.observation_space = spaces.Box(0.0, 1.0, shape=(channels, height_blocks, width_blocks), dtype=np.float32)

    def encode(self, state: GameState) -> np.ndarray:
        # LEARN(M5 v2): mark every cell a hitbox overlaps. Partial overlap can
        # be 1.0 or the covered fraction; try both in M8.
        raise NotImplementedError("LEARN(M5): GridEncoder.encode")


ENCODERS: dict[str, type[Encoder]] = {
    "debug": DebugEncoder,
    "object_list": ObjectListEncoder,
    "grid": GridEncoder,
}


def make_encoder(name: str, **kwargs) -> Encoder:
    try:
        return ENCODERS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}") from exc
