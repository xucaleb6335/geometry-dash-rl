"""Observation for mixed-mode (demon) levels: ``name: demon`` in an encoder config.

Like ``ObjectListEncoder`` (relative, normalized, nearest objects first, a
fine distance code), plus what cube-only play did not need:

- player: mode one-hot, gravity, mini, button held, speed, hitbox size, and
  the distance to the floor and the roof when the protocol gives them
- objects: kind one-hot plus a subtype group (spike, saw, block, orb, pad,
  mode portal, gravity portal, speed portal), and the gaps between the
  player's hitbox and the object's top and bottom (what a ship or wave
  threads through)
- Gaussian distance codes in x (left edge) and y (centre) for the nearest
  objects, since timing and height both need precision of a few units
- optionally the previous observation stacked on (``stack: 2``), so moving
  objects show their direction

Registered in ``gdrl.features.encoder.ENCODERS`` on import (the factory imports it).
"""

from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces

from gdrl.env.protocol import BLOCK, GameState, ObjectKind, ObjectSubtype, PlayerMode
from gdrl.features.encoder import ENCODERS, Encoder

KINDS = (ObjectKind.SOLID, ObjectKind.HAZARD, ObjectKind.ORB, ObjectKind.PAD, ObjectKind.PORTAL, ObjectKind.OTHER)
GROUPS = ("spike", "saw", "block", "orb", "pad", "mode_portal", "gravity_portal", "speed_portal", "other_portal")


def _group(subtype: int) -> int:
    s = int(subtype)
    if s == ObjectSubtype.SPIKE:
        return 0
    if s == ObjectSubtype.SAW:
        return 1
    if s in (ObjectSubtype.BLOCK, ObjectSubtype.SLOPE):
        return 2
    if 10 <= s <= 20:
        return 3
    if 30 <= s <= 34:
        return 4
    if 40 <= s <= 47:
        return 5
    if 50 <= s <= 52:
        return 6
    if 60 <= s <= 64:
        return 7
    if 53 <= s <= 59:
        return 8
    return -1


class DemonEncoder(Encoder):
    MAX_X_SPEED = 600.0
    VEL_SCALE = 1200.0

    def __init__(self, k: int = 12, max_dx_blocks: float = 20.0, max_dy_blocks: float = 10.0, near_slots: int = 4,
                 rbf_x_step: float = 10.0, rbf_x_range: tuple[float, float] = (-30.0, 240.0),
                 rbf_y_step: float = 15.0, rbf_y_range: tuple[float, float] = (-150.0, 150.0), stack: int = 1):
        self.k = k
        self.max_dx = max_dx_blocks * BLOCK
        self.max_dy = max_dy_blocks * BLOCK
        self.near = min(near_slots, k)
        self.rbf_x = np.arange(rbf_x_range[0], rbf_x_range[1] + 1e-9, rbf_x_step)
        self.rbf_x_w = rbf_x_step
        self.rbf_y = np.arange(rbf_y_range[0], rbf_y_range[1] + 1e-9, rbf_y_step)
        self.rbf_y_w = rbf_y_step
        self.stack = max(1, int(stack))
        self.player_size = 7 + len(PlayerMode) + 4
        self.slot_size = 6 + len(KINDS) + len(GROUPS)
        self.frame = self.player_size + self.k * self.slot_size + self.near * (len(self.rbf_x) + len(self.rbf_y))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.frame * self.stack,), dtype=np.float32)
        self._kind_col = np.full(256, KINDS.index(ObjectKind.OTHER), dtype=np.int64)
        for i, kd in enumerate(KINDS):
            self._kind_col[int(kd)] = i
        self._prev: list[np.ndarray] = []

    def reset(self, state: GameState) -> None:
        self._prev = []

    def encode_frame(self, state: GameState) -> np.ndarray:
        obs = np.zeros(self.frame, dtype=np.float64)
        size = state.player_size if state.player_size > 0 else 30.0
        half = size / 2
        obs[0] = state.player_y / self.max_dy
        obs[1] = state.vel_y / self.VEL_SCALE
        obs[2] = float(state.on_ground)
        obs[3] = float(state.upside_down)
        obs[4] = float(state.mini)
        obs[5] = float(state.holding)
        obs[6] = state.x_speed / self.MAX_X_SPEED
        mode = int(state.mode)
        if 0 <= mode < len(PlayerMode):
            obs[7 + mode] = 1.0
        j = 7 + len(PlayerMode)
        obs[j] = size / 30.0
        if math.isfinite(state.floor_y):
            obs[j + 1] = (state.player_y - half - state.floor_y) / self.max_dy
        if math.isfinite(state.ceil_y):
            obs[j + 2] = (state.ceil_y - state.player_y - half) / self.max_dy
            obs[j + 3] = 1.0

        objs = state.objects
        if len(objs):
            x, y = objs["x"].astype(np.float64), objs["y"].astype(np.float64)
            w, h = objs["w"].astype(np.float64), objs["h"].astype(np.float64)
            dx, dy = x - state.player_x, y - state.player_y
            visible = ((dx + w / 2 > -half) & (dx - w / 2 <= self.max_dx)
                       & (np.abs(dy) <= self.max_dy + h / 2) & (objs["kind"] != ObjectKind.NONE))
            idx = np.nonzero(visible)[0]
            left = dx - w / 2
            order = np.lexsort((dy[idx], left[idx]))
            idx = idx[order][: self.k]
            base = self.player_size
            for n, i in enumerate(idx):
                o = base + n * self.slot_size
                obs[o] = left[i] / self.max_dx
                obs[o + 1] = dy[i] / self.max_dy
                obs[o + 2] = w[i] / self.max_dx
                obs[o + 3] = h[i] / self.max_dy
                obs[o + 4] = (y[i] + h[i] / 2 - (state.player_y - half)) / self.max_dy  # top - player bottom
                obs[o + 5] = (y[i] - h[i] / 2 - (state.player_y + half)) / self.max_dy  # bottom - player top
                obs[o + 6 + self._kind_col[int(objs["kind"][i])]] = 1.0
                g = _group(int(objs["subtype"][i]))
                if g >= 0:
                    obs[o + 6 + len(KINDS) + g] = 1.0
            r = base + self.k * self.slot_size
            for n, i in enumerate(idx[: self.near]):
                bx = np.exp(-(((left[i] - self.rbf_x) / self.rbf_x_w) ** 2))
                by = np.exp(-(((dy[i] - self.rbf_y) / self.rbf_y_w) ** 2))
                o = r + n * (len(self.rbf_x) + len(self.rbf_y))
                obs[o : o + len(self.rbf_x)] = bx
                obs[o + len(self.rbf_x) : o + len(self.rbf_x) + len(self.rbf_y)] = by
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def encode(self, state: GameState) -> np.ndarray:
        frame = self.encode_frame(state)
        if self.stack == 1:
            return frame
        if not self._prev:
            self._prev = [frame] * (self.stack - 1)
        out = np.concatenate([frame, *self._prev[::-1]]) if self.stack > 1 else frame
        self._prev = (self._prev + [frame])[-(self.stack - 1):]
        return out


ENCODERS["demon"] = DemonEncoder
