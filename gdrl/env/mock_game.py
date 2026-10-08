"""A tiny Geometry Dash look-alike for developing without the real game.

Cube mode only: constant horizontal speed, gravity, jump while on the ground,
solid blocks you can land on, and spikes that kill. The numbers are close to
GD's but not exact, so a policy trained here will not transfer as-is. The
point is a fast, deterministic stand-in that produces the same ``GameState``
the mod sends, so the env, encoders, rewards and agents can be built and tested
before the real bridge works.

Plumbing, not a learning target: read it, don't rewrite it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gdrl.env.protocol import (
    BLOCK,
    DEFAULT_TICK_RATE,
    MAX_OBJECTS,
    OBJECT_DTYPE,
    GameState,
    ObjectKind,
    PlayerMode,
    StateFlag,
)

# Physics, in game units and seconds. Approximate 1x-speed cube values.
X_SPEED = 311.58
GRAVITY = 2794.0
JUMP_VELOCITY = 603.0
PLAYER_HALF = 15.0  # player hitbox is 30 x 30
LAND_TOLERANCE = 6.0  # how far below a block's top you may be and still land on it
SPIN_DEG_PER_S = 410.0

SPIKE_W, SPIKE_H = 9.0, 18.0  # spike hazard hitbox, smaller than its 30 x 30 sprite

PATTERNS = ("spike", "double_spike", "triple_spike", "block")


@dataclass(frozen=True)
class MockLevel:
    seed: int
    length: float
    objects: np.ndarray  # OBJECT_DTYPE, sorted by left edge

    @property
    def left(self) -> np.ndarray:
        return self.objects["x"] - self.objects["w"] / 2

    @property
    def right(self) -> np.ndarray:
        return self.objects["x"] + self.objects["w"] / 2


def generate_level(
    seed: int,
    n_obstacles: int = 12,
    min_gap_blocks: int = 7,
    max_gap_blocks: int = 12,
    patterns: tuple[str, ...] = PATTERNS,
) -> MockLevel:
    """Procedurally build a beatable cube level. Same seed, same level."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, float, float, float, int]] = []
    cursor = 10 * BLOCK  # run-up before the first obstacle

    def spike(cell_left: float) -> None:
        rows.append((cell_left + BLOCK / 2, SPIKE_H / 2, SPIKE_W, SPIKE_H, ObjectKind.HAZARD))

    for _ in range(n_obstacles):
        pattern = patterns[int(rng.integers(len(patterns)))]
        if pattern == "spike":
            spike(cursor)
            width = 1
        elif pattern == "double_spike":
            spike(cursor)
            spike(cursor + BLOCK)
            width = 2
        elif pattern == "triple_spike":
            for i in range(3):
                spike(cursor + i * BLOCK)
            width = 3
        elif pattern == "block":
            rows.append((cursor + BLOCK / 2, BLOCK / 2, BLOCK, BLOCK, ObjectKind.SOLID))
            width = 1
        else:
            raise ValueError(f"unknown pattern {pattern!r}")
        gap = int(rng.integers(min_gap_blocks, max_gap_blocks + 1))
        cursor += (width + gap) * BLOCK

    objects = np.zeros(len(rows), dtype=OBJECT_DTYPE)
    for i, (x, y, w, h, kind) in enumerate(rows):
        objects[i]["x"], objects[i]["y"], objects[i]["w"], objects[i]["h"] = x, y, w, h
        objects[i]["kind"] = int(kind)
    order = np.argsort(objects["x"] - objects["w"] / 2, kind="stable")
    return MockLevel(seed=seed, length=cursor + 10 * BLOCK, objects=objects[order])


class MockGame:
    """Deterministic cube physics over a ``MockLevel``. Same API shape as the mod."""

    tick_rate = DEFAULT_TICK_RATE

    def __init__(self, level: MockLevel | None = None):
        self.level = level if level is not None else generate_level(0)
        self._left = self.level.left.astype(np.float64)
        self._right = self.level.right.astype(np.float64)
        self._top = (self.level.objects["y"] + self.level.objects["h"] / 2).astype(np.float64)
        self._bottom = (self.level.objects["y"] - self.level.objects["h"] / 2).astype(np.float64)
        self._kind = self.level.objects["kind"].astype(np.int64)
        self.reset()

    # ----------------------------------------------------------------- API

    def reset(self) -> GameState:
        self.tick = 0
        self.x = PLAYER_HALF  # hitbox centre
        self.bottom = 0.0  # hitbox bottom edge
        self.vy = 0.0
        self.rotation = 0.0
        self.on_ground = True
        self.holding = False
        self.dead = False
        self.completed = False
        return self.state()

    def step(self, button: bool, ticks: int = 1) -> GameState:
        """Hold or release the button for ``ticks`` physics ticks."""
        self.holding = bool(button)
        for _ in range(ticks):
            if self.dead or self.completed:
                break
            self._tick()
        return self.state()

    def state(self) -> GameState:
        flags = StateFlag.NONE
        if self.dead:
            flags |= StateFlag.DEAD
        if self.completed:
            flags |= StateFlag.COMPLETED
        if self.on_ground:
            flags |= StateFlag.ON_GROUND
        if self.holding:
            flags |= StateFlag.HOLDING
        visible = np.nonzero(self._right >= self.x - PLAYER_HALF - BLOCK)[0][:MAX_OBJECTS]
        return GameState(
            tick=self.tick,
            flags=flags,
            player_x=self.x,
            player_y=self.bottom + PLAYER_HALF,
            vel_y=self.vy,
            rotation=self.rotation,
            x_speed=X_SPEED,
            level_length=self.level.length,
            mode=PlayerMode.CUBE,
            objects=self.level.objects[visible].copy(),
        )

    # ------------------------------------------------------------- physics

    def _tick(self) -> None:
        dt = 1.0 / self.tick_rate
        self.tick += 1

        if self.holding and self.on_ground:
            self.vy = JUMP_VELOCITY
        prev_bottom = self.bottom
        self.vy -= GRAVITY * dt
        self.bottom += self.vy * dt
        self.x += X_SPEED * dt
        self.on_ground = False

        if self.bottom <= 0.0:
            self.bottom = 0.0
            self.vy = 0.0
            self.on_ground = True

        p_left, p_right = self.x - PLAYER_HALF, self.x + PLAYER_HALF
        # Objects are sorted by left edge; only those overlapping in x matter.
        hi = int(np.searchsorted(self._left, p_right, side="left"))
        for i in range(hi):
            if self._right[i] <= p_left:
                continue
            p_top = self.bottom + 2 * PLAYER_HALF
            if not (self.bottom < self._top[i] and p_top > self._bottom[i]):
                continue
            if self._kind[i] == ObjectKind.HAZARD:
                self.dead = True
            elif self._kind[i] == ObjectKind.SOLID:
                if prev_bottom >= self._top[i] - LAND_TOLERANCE and self.vy <= 0.0:
                    self.bottom = self._top[i]
                    self.vy = 0.0
                    self.on_ground = True
                else:
                    self.dead = True  # hit the side or the underside
            if self.dead:
                return

        if self.on_ground:
            self.rotation = 90.0 * math.floor(self.rotation / 90.0 + 0.5) % 360.0
        else:
            self.rotation = (self.rotation + SPIN_DEG_PER_S * dt) % 360.0

        if self.x >= self.level.length:
            self.completed = True
