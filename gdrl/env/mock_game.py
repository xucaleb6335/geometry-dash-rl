"""A tiny Geometry Dash look-alike for developing without the real game.

Constant horizontal speed, gravity, solid blocks you can land on, and spikes
that kill. The numbers are close to GD's but not exact, so a policy trained
here will not transfer as-is. The point is a fast, deterministic stand-in that
produces the same ``GameState`` the mod sends, so the env, encoders, rewards
and agents can be built and tested before the real bridge works.

Beyond the original cube-only game (whose physics are unchanged, bit for bit),
it has what the demon work needs to develop against:

- modes: cube, ship, wave; gravity flips; speed changes (all via portals)
- yellow jump orbs (a press, or a press still held when you enter), yellow pads
- hazards moving up and down (sine), read live in every STATE
- an optional roof (``MockLevel.ceiling``), like GD's ship/wave corridor bounds
- Start Pos objects (``reset(start_pos=k)``); like GD's, they do not replay
  what came before (moving hazards restart their cycle)
- snapshots: ``snapshot()`` / ``restore()``, exact unless ``restore_error``
  is set (to test code that must cope with an inexact restore)

Plumbing, not a learning target: read it, don't rewrite it.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

from gdrl.env.protocol import (
    BLOCK,
    DEFAULT_TICK_RATE,
    MAX_OBJECTS,
    OBJECT_DTYPE,
    GameState,
    ObjectFlag,
    ObjectKind,
    ObjectSubtype,
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

# Ship and wave (mock values, GD-like in feel only).
SHIP_UP_ACCEL = 1800.0  # net upward acceleration while holding
SHIP_DOWN_ACCEL = 1400.0  # net downward acceleration while released
SHIP_MAX_UP = 400.0
SHIP_MAX_DOWN = 500.0
WAVE_HALF = 5.0  # the wave's hitbox is small
ORB_VELOCITY = 590.0
PAD_VELOCITY = 1000.0

SPEEDS = {  # units per second for each speed portal (the mod's table)
    ObjectSubtype.SPEED_SLOW: 251.16,
    ObjectSubtype.SPEED_NORMAL: 311.58,
    ObjectSubtype.SPEED_2X: 387.42,
    ObjectSubtype.SPEED_3X: 468.0,
    ObjectSubtype.SPEED_4X: 576.0,
}
MODE_PORTALS = {
    ObjectSubtype.PORTAL_CUBE: PlayerMode.CUBE,
    ObjectSubtype.PORTAL_SHIP: PlayerMode.SHIP,
    ObjectSubtype.PORTAL_WAVE: PlayerMode.WAVE,
}

PATTERNS = ("spike", "double_spike", "triple_spike", "block")


@dataclass(frozen=True)
class StartPos:
    """A Start Pos object: where and how an attempt begins. ``y`` is the hitbox bottom."""

    x: float
    y: float = 0.0
    mode: PlayerMode = PlayerMode.CUBE
    gravity: int = 1  # 1 normal, -1 upside down
    x_speed: float = X_SPEED


@dataclass(frozen=True)
class MockLevel:
    seed: int
    length: float
    objects: np.ndarray  # OBJECT_DTYPE, sorted by left edge
    # Per object (amplitude, period in ticks, phase in radians): y moves by
    # amplitude * sin(2 pi tick / period + phase). None = nothing moves.
    motion: np.ndarray | None = None
    ceiling: float | None = None  # a roof at this y for every mode; None = open sky
    start_positions: tuple[StartPos, ...] = ()
    name: str = ""
    max_objects: int = MAX_OBJECTS  # how many objects a STATE carries (64 for protocol v2 levels)

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


def make_mock_level(level_id: int, kind: str = "simple", **kwargs) -> MockLevel:
    """Level ``level_id`` of a generator: ``simple`` (cube, ``generate_level``)
    or ``demon`` (long, mixed modes, ``gdrl.env.mock_demon``)."""
    if kind == "simple":
        return generate_level(level_id, **kwargs)
    if kind == "demon":
        from gdrl.env.mock_demon import generate_demon_level

        return generate_demon_level(level_id, **kwargs)
    raise ValueError(f"unknown mock level kind {kind!r}; use 'simple' or 'demon'")


class MockGame:
    """Deterministic physics over a ``MockLevel``. Same API shape as the mod."""

    tick_rate = DEFAULT_TICK_RATE

    def __init__(self, level: MockLevel | None = None, restore_error: float = 0.0):
        self.level = level if level is not None else generate_level(0)
        self.restore_error = restore_error
        objs = self.level.objects
        n = len(objs)
        # Plain lists: scalar indexing is much faster than on numpy arrays.
        self._left_np = self.level.left.astype(np.float64)
        self._right_np = self.level.right.astype(np.float64)
        self._left = self._left_np.tolist()
        self._right = self._right_np.tolist()
        self._y0 = objs["y"].astype(np.float64).tolist()
        self._half_h = (objs["h"].astype(np.float64) / 2).tolist()
        # Static edges computed in float32 first, exactly as the original cube game did.
        self._top = (objs["y"] + objs["h"] / 2).astype(np.float64).tolist()
        self._bottom = (objs["y"] - objs["h"] / 2).astype(np.float64).tolist()
        self._kind = objs["kind"].astype(np.int64).tolist()
        self._subtype = objs["subtype"].astype(np.int64).tolist()
        self._max_w = float(objs["w"].max()) if n else 0.0
        motion = self.level.motion
        if motion is not None and np.any(motion[:, 0] != 0):
            self._amp = motion[:, 0].astype(np.float64).tolist()
            self._omega = (2 * math.pi / np.maximum(motion[:, 1], 1.0)).tolist()
            self._phase = motion[:, 2].astype(np.float64).tolist()
            self._movers = np.nonzero(motion[:, 0] != 0)[0]
        else:
            self._amp = None
            self._movers = np.zeros(0, dtype=np.int64)
        self._ceiling = self.level.ceiling
        self.reset()

    # ----------------------------------------------------------------- API

    def reset(self, start_pos: int = -1) -> GameState:
        """Restart the level; ``start_pos = k >= 0`` starts at the k-th Start Pos (by x)."""
        self.tick = 0
        self.x = PLAYER_HALF  # hitbox centre
        self.bottom = 0.0  # hitbox bottom edge
        self.vy = 0.0
        self.rotation = 0.0
        self.on_ground = True
        self.holding = False
        self.dead = False
        self.completed = False
        self.mode = PlayerMode.CUBE
        self.gravity = 1
        self.x_speed = X_SPEED
        self.half = PLAYER_HALF
        self.prev_holding = False
        self.buffered = False  # a press not yet used by an orb
        self.used = bytearray(len(self.level.objects))  # orbs, pads, portals already taken
        if start_pos >= 0:
            starts = sorted(self.level.start_positions, key=lambda s: s.x)
            if start_pos >= len(starts):
                raise ValueError(f"start_pos {start_pos}: the level has {len(starts)} Start Pos objects")
            sp = starts[start_pos]
            self.mode = PlayerMode(sp.mode)
            self.half = WAVE_HALF if self.mode == PlayerMode.WAVE else PLAYER_HALF
            self.x = sp.x
            self.bottom = sp.y
            self.gravity = sp.gravity
            self.x_speed = sp.x_speed
            self.on_ground = sp.y <= 0.0 and sp.gravity == 1
        return self.state()

    def step(self, button: bool, ticks: int = 1) -> GameState:
        """Hold or release the button for ``ticks`` physics ticks."""
        self.run(button, ticks)
        return self.state()

    def run(self, button: bool, ticks: int) -> None:
        """Like ``step`` but without building a STATE (for search loops)."""
        self.holding = bool(button)
        for _ in range(ticks):
            if self.dead or self.completed:
                break
            self._tick()

    @property
    def terminal(self) -> bool:
        return self.dead or self.completed

    def snapshot(self) -> tuple:
        """Everything that changes during an attempt. Restore with ``restore``."""
        return (self.tick, self.x, self.bottom, self.vy, self.rotation, self.on_ground, self.holding,
                self.dead, self.completed, self.mode, self.gravity, self.x_speed, self.half,
                self.prev_holding, self.buffered, bytes(self.used))

    def restore(self, snap: tuple) -> GameState:
        (self.tick, self.x, self.bottom, self.vy, self.rotation, self.on_ground, self.holding,
         self.dead, self.completed, self.mode, self.gravity, self.x_speed, self.half,
         self.prev_holding, self.buffered, used) = snap
        self.used = bytearray(used)
        if self.restore_error:
            self.bottom += self.restore_error  # simulate an inexact restore
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
        if self.gravity < 0:
            flags |= StateFlag.UPSIDE_DOWN
        visible = np.nonzero(self._right_np >= self.x - PLAYER_HALF - BLOCK)[0][: self.level.max_objects]
        objects = self.level.objects[visible].copy()
        if len(self._movers) and len(visible):
            for j, i in enumerate(visible.tolist()):
                if self._amp[i]:
                    objects[j]["y"] = self._object_y(i)
        if len(visible):
            used = np.frombuffer(bytes(self.used), dtype=np.uint8)[visible]
            objects["oflags"] |= np.where(used > 0, int(ObjectFlag.ACTIVATED), 0).astype(np.uint8)
        return GameState(
            tick=self.tick,
            flags=flags,
            player_x=self.x,
            player_y=self.bottom + self.half,
            vel_y=self.vy,
            rotation=self.rotation,
            x_speed=self.x_speed,
            level_length=self.level.length,
            mode=PlayerMode(self.mode),
            objects=objects,
            player_size=2 * self.half,
            floor_y=0.0,
            ceil_y=float(self._ceiling) if self._ceiling is not None else float("nan"),
        )

    # ------------------------------------------------------------- physics

    def _object_y(self, i: int) -> float:
        if self._amp is None or not self._amp[i]:
            return self._y0[i]
        return self._y0[i] + self._amp[i] * math.sin(self._omega[i] * self.tick + self._phase[i])

    def _set_mode(self, mode: PlayerMode) -> None:
        if mode == self.mode:
            return
        centre = self.bottom + self.half
        self.mode = mode
        self.half = WAVE_HALF if mode == PlayerMode.WAVE else PLAYER_HALF
        self.bottom = centre - self.half
        self.vy *= 0.5

    def _tick(self) -> None:
        dt = 1.0 / self.tick_rate
        self.tick += 1
        g = self.gravity
        mode = self.mode

        if self.holding and not self.prev_holding:
            self.buffered = True
        if not self.holding:
            self.buffered = False
        self.prev_holding = self.holding

        if mode == PlayerMode.CUBE:
            if self.holding and self.on_ground:
                self.vy = g * JUMP_VELOCITY
                self.buffered = False
            prev_bottom = self.bottom
            self.vy -= g * GRAVITY * dt
        elif mode == PlayerMode.SHIP:
            prev_bottom = self.bottom
            if self.holding:
                self.vy += g * SHIP_UP_ACCEL * dt
            else:
                self.vy -= g * SHIP_DOWN_ACCEL * dt
            up = self.vy * g  # velocity in the gravity frame, + = away from the floor
            if up > SHIP_MAX_UP:
                self.vy = g * SHIP_MAX_UP
            elif up < -SHIP_MAX_DOWN:
                self.vy = -g * SHIP_MAX_DOWN
        else:  # wave: 45 degrees up while held, down while released
            prev_bottom = self.bottom
            self.vy = g * (1.0 if self.holding else -1.0) * self.x_speed
        self.bottom += self.vy * dt
        self.x += self.x_speed * dt
        self.on_ground = False

        size = 2 * self.half
        if self.bottom <= 0.0:
            self.bottom = 0.0
            if self.vy < 0.0:
                self.vy = 0.0
            elif self.vy == 0.0:
                self.vy = 0.0
            self.on_ground = g == 1
        if self._ceiling is not None and self.bottom + size >= self._ceiling:
            self.bottom = self._ceiling - size
            if self.vy > 0.0:
                self.vy = 0.0
            self.on_ground = g == -1

        p_left, p_right = self.x - self.half, self.x + self.half
        # Objects are sorted by left edge; only those overlapping in x matter.
        hi = bisect_left(self._left, p_right)
        lo = bisect_left(self._left, p_left - self._max_w)
        for i in range(lo, hi):
            if self._right[i] <= p_left:
                continue
            p_top = self.bottom + size
            if self._amp is not None and self._amp[i]:
                y = self._object_y(i)
                o_top = y + self._half_h[i]
                o_bottom = y - self._half_h[i]
            else:
                o_top = self._top[i]
                o_bottom = self._bottom[i]
            if not (self.bottom < o_top and p_top > o_bottom):
                continue
            kind = self._kind[i]
            if kind == ObjectKind.HAZARD:
                self.dead = True
            elif kind == ObjectKind.SOLID:
                if mode == PlayerMode.WAVE:
                    self.dead = True
                elif g == 1 and prev_bottom >= o_top - LAND_TOLERANCE and self.vy <= 0.0:
                    self.bottom = o_top
                    self.vy = 0.0
                    self.on_ground = True
                elif g == -1 and prev_bottom + size <= o_bottom + LAND_TOLERANCE and self.vy >= 0.0:
                    self.bottom = o_bottom - size
                    self.vy = 0.0
                    self.on_ground = True
                elif mode == PlayerMode.SHIP and g == 1 and prev_bottom + size <= o_bottom + LAND_TOLERANCE \
                        and self.vy >= 0.0:
                    self.bottom = o_bottom - size  # a ship slides along a block's underside
                    self.vy = 0.0
                elif mode == PlayerMode.SHIP and g == -1 and prev_bottom >= o_top - LAND_TOLERANCE \
                        and self.vy <= 0.0:
                    self.bottom = o_top
                    self.vy = 0.0
                else:
                    self.dead = True  # hit the side or the underside
            elif kind == ObjectKind.PAD and not self.used[i]:
                self.used[i] = 1
                if mode != PlayerMode.WAVE:
                    self.vy = g * PAD_VELOCITY
            elif kind == ObjectKind.ORB and not self.used[i] and self.buffered and mode != PlayerMode.WAVE:
                self.used[i] = 1
                self.buffered = False
                self.vy = g * ORB_VELOCITY
            elif kind == ObjectKind.PORTAL and not self.used[i]:
                self.used[i] = 1
                self._portal(self._subtype[i])
                g, mode, size = self.gravity, self.mode, 2 * self.half
            if self.dead:
                return

        if self.mode == PlayerMode.CUBE:
            if self.on_ground:
                self.rotation = 90.0 * math.floor(self.rotation / 90.0 + 0.5) % 360.0
            else:
                self.rotation = (self.rotation + SPIN_DEG_PER_S * dt) % 360.0
        else:
            self.rotation = math.degrees(math.atan2(-self.vy, self.x_speed)) % 360.0

        if self.x >= self.level.length:
            self.completed = True

    def _portal(self, subtype: int) -> None:
        if subtype in MODE_PORTALS:
            self._set_mode(MODE_PORTALS[subtype])
        elif subtype in SPEEDS:
            self.x_speed = SPEEDS[subtype]
        elif subtype in (ObjectSubtype.PORTAL_GRAVITY_UP, ObjectSubtype.PORTAL_GRAVITY_DOWN,
                         ObjectSubtype.PORTAL_GRAVITY_TOGGLE):
            if subtype == ObjectSubtype.PORTAL_GRAVITY_UP:
                new = -1
            elif subtype == ObjectSubtype.PORTAL_GRAVITY_DOWN:
                new = 1
            else:
                new = -self.gravity
            if new != self.gravity:
                self.gravity = new
                self.vy *= 0.5
