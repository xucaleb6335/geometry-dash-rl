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
from gdrl.env.protocol import BLOCK, GameState, ObjectKind, PlayerMode


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

    Layout, length ``PLAYER_FEATURES + (4 + len(KINDS)) * k + near_slots * len(RBF_CENTERS)``:

    - ``[0:13]`` player: y, vel_y, on_ground, upside_down, x_speed, then the
      game mode one-hot over ``PlayerMode`` (cube, ship, ball, UFO, wave,
      robot, spider, swing). Mode and speed matter on real levels: the same
      gap needs a different input in a ship than in a cube, and at 2x speed
      the same distance arrives sooner.
    - then ``k`` slots of ``(dx, dy, w, h, *one_hot(kind))`` for the nearest
      objects ahead, sorted by dx. ``dx``/``dy`` are the object's hitbox centre
      relative to the player's. Kind is one-hot over ``KINDS`` (an unknown kind
      counts as OTHER), so every real object has exactly one kind bit set and
      an empty slot, all zero, cannot be confused with one.

    Scales are fixed: x lengths (dx, w) are divided by ``max_dx``, y lengths
    (y, dy, h) by ``max_dy``, so ``dx - w / 2`` is still the left edge in the
    same units. Velocity is divided by twice the cube's jump velocity and
    x_speed by ``MAX_X_SPEED``. Then everything is clipped to [-1, 1].

    "Ahead" means the hitbox's right edge is past the player's left edge (so
    the object being passed over stays visible) and its left edge is within
    ``max_dx``; objects more than ``max_dy`` away vertically are out of view.

    Last, a fine distance code for the ``near_slots`` nearest objects: the
    distance from the player's centre to the object's left edge, as Gaussian
    bumps (width ``RBF_WIDTH``) centred every ``RBF_WIDTH`` units over
    ``RBF_RANGE``. Jump timing needs a precision of a few units (one decision
    at frame skip 4 is ~5 units), which is ~0.01 of ``dx`` above: a small MLP
    could not learn the jump band from ``dx`` alone (behaviour cloning the
    rule agent got 7% of its presses), but learns it from this code (100%).
    """

    KINDS = (ObjectKind.SOLID, ObjectKind.HAZARD, ObjectKind.ORB, ObjectKind.PAD, ObjectKind.PORTAL, ObjectKind.OTHER)
    MODES = tuple(PlayerMode)
    MAX_X_SPEED = 600.0  # a bit above GD's fastest (4x) speed, in units per second
    PLAYER_FEATURES = 5 + len(MODES)
    SLOT_SIZE = 4 + len(KINDS)
    RBF_WIDTH = 10.0
    RBF_RANGE = (-30.0, 180.0)
    RBF_CENTERS = np.arange(RBF_RANGE[0], RBF_RANGE[1] + 1e-9, RBF_WIDTH)
    PROFILE_DX = np.arange(0.0, 240.0 + 1e-9, 15.0)  # column centres ahead of the player
    POSITION_PERIODS = 30.0 * 2.0 ** np.arange(12)    # 30 .. 61,440 units

    def __init__(self, k: int = 8, max_dx_blocks: float = 20.0, max_dy_blocks: float = 10.0, near_slots: int = 3,
                 focus: str = "nearest", profile: bool = False, position: bool = False) -> None:
        if focus not in ("nearest", "obstacles"):
            raise ValueError(f"focus must be 'nearest' or 'obstacles', not {focus!r}")
        self.k = k
        self.focus = focus
        self.near_slots = min(near_slots, k)
        self.max_dx = max_dx_blocks * BLOCK
        self.max_dy = max_dy_blocks * BLOCK
        self.rbf_start = self.PLAYER_FEATURES + self.SLOT_SIZE * k
        size = self.rbf_start + self.near_slots * len(self.RBF_CENTERS)
        # Optional terrain profile: per column ahead, the surface the player
        # would stand on, whether a hazard sits on it, and the nearest thing
        # overhead (all relative to the feet). Why: on Stereo Madness the
        # object slots fill up with the pit spikes under a slab path, and the
        # policy never sees where the slab ends or the next one starts (six
        # starts stuck at 0% for 2M steps, even with random exploration).
        self.profile = profile
        self.profile_start = size
        if profile:
            size += 3 * len(self.PROFILE_DX)
        # Optional position in the level (sin/cos of x at periods 30 .. 61,440
        # units). Why: on extreme demons many hazard-typed objects are
        # decoration (unreachable, inside walls), so what the object list
        # shows is not always what kills. With its position, the policy can
        # also learn a level's timing from deaths alone, like a TAS.
        self.position = position
        self.position_start = size
        if position:
            size += 2 * len(self.POSITION_PERIODS)
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
        obs[4] = state.x_speed / self.MAX_X_SPEED
        mode = int(state.mode)
        if 0 <= mode < len(self.MODES):
            obs[5 + mode] = 1.0

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
        idx = idx[order]
        if self.focus == "obstacles":
            # Obstacles first: hazards, and solids the player could run into
            # (top above its feet), not the floor it runs on. On real levels a
            # row of floor blocks otherwise fills the slots and the fine
            # distance code, and the spike on it is never timed (Stereo
            # Madness: every attempt died at the spike on the raised floor at
            # x = 3075). Stable sort, so each group stays ordered by dx.
            feet = state.player_y - BLOCK / 2
            ym = 2 * state.player_y - y if state.upside_down else y  # gravity flipped: mirror, "up" is down
            obstacle = (objs["kind"][idx] == ObjectKind.HAZARD) | (
                (objs["kind"][idx] == ObjectKind.SOLID) & (ym[idx] + h[idx] / 2 > feet + 2.0))
            # Slots (and so the distance code) hold the obstacles by dx, then the rest by dx.
            idx = idx[np.argsort(~obstacle, kind="stable")][: self.k]
        else:
            idx = idx[: self.k]

        slots = obs[self.PLAYER_FEATURES : self.rbf_start].reshape(self.k, self.SLOT_SIZE)
        n = len(idx)
        m = min(n, self.near_slots)
        if m:
            left = (dx[idx[:m]] - w[idx[:m]] / 2)[:, None]
            rbf = np.exp(-(((left - self.RBF_CENTERS) / self.RBF_WIDTH) ** 2))
            obs[self.rbf_start : self.rbf_start + m * len(self.RBF_CENTERS)] = rbf.reshape(-1)
        slots[:n, 0] = dx[idx] / self.max_dx
        slots[:n, 1] = dy[idx] / self.max_dy
        slots[:n, 2] = w[idx] / self.max_dx
        slots[:n, 3] = h[idx] / self.max_dy
        slots[np.arange(n), 4 + self._kind_column[objs["kind"][idx]]] = 1.0
        if self.profile:
            end = self.profile_start + 3 * len(self.PROFILE_DX)
            obs[self.profile_start:end] = self._profile(state, x, y, w, h, objs["kind"])
        if self.position:
            phase = 2 * np.pi * state.player_x / self.POSITION_PERIODS
            obs[self.position_start:] = np.concatenate([np.sin(phase), np.cos(phase)])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _profile(self, state: GameState, x, y, w, h, kind) -> np.ndarray:
        feet = state.player_y - BLOCK / 2
        # Gravity flipped: mirror around the player, so "support" is what it
        # stands on either way. The ground then is not a floor (y = 0 is
        # overhead), so there is no default support.
        flipped = bool(state.upside_down)
        if flipped:
            y = 2 * state.player_y - y
        ground = feet - self.max_dy if flipped else 0.0
        solid = kind == ObjectKind.SOLID
        hazard = kind == ObjectKind.HAZARD
        # Spike hitboxes are narrow (6-9 units); count them over their tile.
        w = np.where(hazard, np.maximum(w, BLOCK), w)
        cols = state.player_x + self.PROFILE_DX                     # (C,)
        over = np.abs(cols[:, None] - x[None, :]) <= w[None, :] / 2  # (C, N): object covers the column
        top, bottom = y + h / 2, y - h / 2
        # Support: the highest solid top within a jump of the feet (> 0 means
        # a step up), else the ground (y = 0).
        can_stand = over & solid[None, :] & (top[None, :] <= feet + 2.5 * BLOCK)
        support = np.maximum(np.where(can_stand, top[None, :], -np.inf).max(axis=1, initial=-np.inf), ground)
        # A hazard on that surface (where the player would run or land).
        on_surface = over & hazard[None, :] & (top[None, :] >= support[:, None] - 1.0)             & (bottom[None, :] <= support[:, None] + BLOCK)
        # Overhead: lowest bottom of anything above the support.
        above = over & (solid | hazard)[None, :] & (bottom[None, :] >= support[:, None] + 5.0)
        ceiling = np.where(above, bottom[None, :], np.inf).min(axis=1, initial=np.inf)
        ceiling = np.minimum(ceiling - feet, self.max_dy)
        return np.concatenate([(support - feet) / self.max_dy, on_surface.any(axis=1).astype(np.float64),
                               ceiling / self.max_dy])


class GridEncoder(Encoder):
    """Observation v2: hitboxes rasterized into a grid ahead of the player. (M5, for a CNN)

    Shape ``(channels, height, width)`` with one channel per kind
    (solid, hazard, orb/pad). Cell size is one block. Column 0 starts one block
    behind the player; row 0 is ``rows_below`` blocks under the player.

    A cell is 1.0 when a hitbox overlaps it at all (touching an edge does not
    count). The ground (y < 0) is drawn as solid, since a grid has no other
    way to show where the floor is. Portals and other kinds are not drawn.
    """

    CHANNEL_OF_KIND = {ObjectKind.SOLID: 0, ObjectKind.HAZARD: 1, ObjectKind.ORB: 2, ObjectKind.PAD: 2}

    def __init__(self, width_blocks: int = 16, height_blocks: int = 8, rows_below: int = 2, channels: int = 3):
        self.width = width_blocks
        self.height = height_blocks
        self.rows_below = rows_below
        self.channels = channels
        self.observation_space = spaces.Box(0.0, 1.0, shape=(channels, height_blocks, width_blocks), dtype=np.float32)

    def encode(self, state: GameState) -> np.ndarray:
        grid = np.zeros(self.observation_space.shape, dtype=np.float32)
        x0 = state.player_x - BLOCK  # left edge of column 0
        y0 = state.player_y - self.rows_below * BLOCK  # bottom edge of row 0

        def mark(channel: int, left: float, right: float, bottom: float, top: float) -> None:
            # Cells [c0, c1) x [r0, r1) that the open box overlaps.
            c0 = max(0, int(np.floor((left - x0) / BLOCK)))
            c1 = min(self.width, int(np.ceil((right - x0) / BLOCK)))
            r0 = max(0, int(np.floor((bottom - y0) / BLOCK)))
            r1 = min(self.height, int(np.ceil((top - y0) / BLOCK)))
            if c0 < c1 and r0 < r1:
                grid[channel, r0:r1, c0:c1] = 1.0

        mark(0, x0, x0 + self.width * BLOCK, y0, 0.0)  # the ground
        for obj in state.objects:
            channel = self.CHANNEL_OF_KIND.get(int(obj["kind"]))  # IntEnum keys match ints
            if channel is None or channel >= self.channels:
                continue
            x, y, w, h = (float(obj[f]) for f in ("x", "y", "w", "h"))
            mark(channel, x - w / 2, x + w / 2, y - h / 2, y + h / 2)
        return grid


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
