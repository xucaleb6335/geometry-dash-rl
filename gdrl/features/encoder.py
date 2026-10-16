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
from gdrl.env.protocol import BLOCK, GameState, ObjectKind


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

    Layout, length ``4 + (4 + len(KINDS)) * k``:

    - ``[0:4]`` player: y, vel_y, on_ground, upside_down
    - then ``k`` slots of ``(dx, dy, w, h, *one_hot(kind))`` for the nearest
      objects ahead, sorted by dx. ``dx``/``dy`` are the object's hitbox centre
      relative to the player's. Kind is one-hot over ``KINDS`` (an unknown kind
      counts as OTHER), so every real object has exactly one kind bit set and
      an empty slot, all zero, cannot be confused with one.

    Scales are fixed: x lengths (dx, w) are divided by ``max_dx``, y lengths
    (y, dy, h) by ``max_dy``, so ``dx - w / 2`` is still the left edge in the
    same units. Velocity is divided by twice the cube's jump velocity. Then
    everything is clipped to [-1, 1].

    "Ahead" means the hitbox's right edge is past the player's left edge (so
    the object being passed over stays visible) and its left edge is within
    ``max_dx``; objects more than ``max_dy`` away vertically are out of view.
    """

    KINDS = (ObjectKind.SOLID, ObjectKind.HAZARD, ObjectKind.ORB, ObjectKind.PAD, ObjectKind.PORTAL, ObjectKind.OTHER)
    PLAYER_FEATURES = 4
    SLOT_SIZE = 4 + len(KINDS)

    def __init__(self, k: int = 8, max_dx_blocks: float = 20.0, max_dy_blocks: float = 10.0) -> None:
        self.k = k
        self.max_dx = max_dx_blocks * BLOCK
        self.max_dy = max_dy_blocks * BLOCK
        size = self.PLAYER_FEATURES + self.SLOT_SIZE * k
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(size,), dtype=np.float32)
        # ObjectKind value -> one-hot column; anything unknown is OTHER.
        self._kind_column = np.full(256, self.KINDS.index(ObjectKind.OTHER), dtype=np.int64)
        for i, kind in enumerate(self.KINDS):
            self._kind_column[int(kind)] = i

    def encode(self, state: GameState) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float64)
        obs[0] = state.player_y / self.max_dy
        obs[1] = state.vel_y / (2 * JUMP_VELOCITY)
        obs[2] = float(state.on_ground)
        obs[3] = float(state.upside_down)

        objs = state.objects
        x, y = objs["x"].astype(np.float64), objs["y"].astype(np.float64)
        w, h = objs["w"].astype(np.float64), objs["h"].astype(np.float64)
        dx, dy = x - state.player_x, y - state.player_y
        visible = (
            (dx + w / 2 > -BLOCK / 2)  # right edge past the player's left edge
            & (dx - w / 2 <= self.max_dx)  # left edge within view
            & (np.abs(dy) <= self.max_dy)
            & (objs["kind"] != ObjectKind.NONE)
        )
        idx = np.nonzero(visible)[0]
        # Sort by dx, ties broken by every other field, so the input order of
        # the objects never changes the encoding.
        kinds = objs["kind"][idx].astype(np.int64)
        order = np.lexsort((h[idx], w[idx], kinds, dy[idx], dx[idx]))
        idx = idx[order][: self.k]

        slots = obs[self.PLAYER_FEATURES :].reshape(self.k, self.SLOT_SIZE)
        n = len(idx)
        slots[:n, 0] = dx[idx] / self.max_dx
        slots[:n, 1] = dy[idx] / self.max_dy
        slots[:n, 2] = w[idx] / self.max_dx
        slots[:n, 3] = h[idx] / self.max_dy
        slots[np.arange(n), 4 + self._kind_column[objs["kind"][idx]]] = 1.0
        return np.clip(obs, -1.0, 1.0).astype(np.float32)


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
