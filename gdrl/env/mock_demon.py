"""Long, mixed-mode mock levels: a stand-in for an extreme demon.

A level is a row of sections, each entered through portals:

- ``cube``:    tight spike groups, blocks, jump pads; gaps sized to the speed
- ``orb``:     spike pits too wide to jump, with a yellow orb to click mid-air
- ``gravity``: a gravity portal sends the cube to the roof; spikes hang from it
- ``ship``:    gates (hazard walls with a gap) at changing heights, moving saws
- ``wave``:    a zigzag of narrow gates

Speed portals (slow to 3x) change the speed between sections. Every section
start has a Start Pos object, like the ones a player would place to practise.

These levels are *probably* beatable, not provably: the gaps are sized from
the physics with some margin, and the tests run the checkpoint search over
them. ``level_id`` is the seed.
"""

from __future__ import annotations

import math

import numpy as np

from gdrl.env.mock_game import (
    GRAVITY,
    JUMP_VELOCITY,
    PAD_VELOCITY,
    SPIKE_H,
    SPIKE_W,
    X_SPEED,
    MockLevel,
    StartPos,
    SPEEDS,
)
from gdrl.env.protocol import BLOCK, MAX_OBJECTS_V2, OBJECT_DTYPE, ObjectKind, ObjectSubtype, PlayerMode

CEILING = 10 * BLOCK  # the roof of every demon level
SECTION_TYPES = ("cube", "orb", "gravity", "ship", "wave")
SPEED_CHOICES = (ObjectSubtype.SPEED_SLOW, ObjectSubtype.SPEED_NORMAL, ObjectSubtype.SPEED_2X,
                 ObjectSubtype.SPEED_3X)
AIRTIME = 2 * JUMP_VELOCITY / GRAVITY  # seconds a cube jump lasts on flat ground
# Seconds a jump keeps the cube's bottom above a spike's top (18 units).
_disc = math.sqrt(JUMP_VELOCITY ** 2 - 4 * (GRAVITY / 2) * SPIKE_H)
CLEAR_TIME = _disc / (GRAVITY / 2)  # t2 - t1 of 18 = v t - g t^2 / 2


def max_spikes(speed: float, margin: float = 8.0) -> int:
    """Largest row of adjacent spikes one jump clears at this speed."""
    clear = CLEAR_TIME * speed - 2 * 15.0  # the player is 30 wide
    n = 1
    while n < 3 and ((n + 1) * BLOCK - (BLOCK - SPIKE_W)) + margin <= clear:
        n += 1
    return n


class _Builder:
    def __init__(self) -> None:
        self.rows: list[tuple] = []  # x, y, w, h, kind, subtype, amp, period, phase

    def add(self, x, y, w, h, kind, subtype, amp=0.0, period=0.0, phase=0.0) -> None:
        self.rows.append((x, y, w, h, int(kind), int(subtype), amp, period, phase))

    def spike(self, cell_left: float, upside_down: bool = False) -> None:
        y = CEILING - SPIKE_H / 2 if upside_down else SPIKE_H / 2
        self.add(cell_left + BLOCK / 2, y, SPIKE_W, SPIKE_H, ObjectKind.HAZARD, ObjectSubtype.SPIKE)

    def portal(self, x: float, subtype: ObjectSubtype) -> None:
        # Full height, so it is taken at any y.
        self.add(x, CEILING / 2, 10.0, CEILING, ObjectKind.PORTAL, subtype)

    def wall(self, x: float, y0: float, y1: float, w: float = 20.0) -> None:
        if y1 - y0 > 0.5:
            self.add(x, (y0 + y1) / 2, w, y1 - y0, ObjectKind.HAZARD, ObjectSubtype.SPIKE)


def generate_demon_level(
    seed: int,
    n_sections: int = 14,
    difficulty: float = 0.6,
    section_types: tuple[str, ...] = SECTION_TYPES,
    speeds: bool = True,
) -> MockLevel:
    """Build a long mixed-mode level. ``difficulty`` in [0, 1] tightens gaps."""
    rng = np.random.default_rng(10_000 + seed)
    b = _Builder()
    starts: list[StartPos] = []
    cursor = 8 * BLOCK
    mode = PlayerMode.CUBE
    speed_sub = ObjectSubtype.SPEED_NORMAL
    d = float(np.clip(difficulty, 0.0, 1.0))
    prev = None

    for k in range(n_sections):
        if k == 0:
            kind = "cube"
        else:
            options = [t for t in section_types if t != prev] or list(section_types)
            kind = options[int(rng.integers(len(options)))]
        prev = kind

        # Section entry: speed portal, mode portal, a Start Pos, then a run-up.
        if speeds and k > 0:
            speed_sub = SPEED_CHOICES[int(rng.integers(len(SPEED_CHOICES)))]
            if kind == "wave" and speed_sub == ObjectSubtype.SPEED_SLOW:
                speed_sub = ObjectSubtype.SPEED_NORMAL
        speed = SPEEDS[speed_sub]
        if k > 0:
            b.portal(cursor + BLOCK / 2, speed_sub)
        new_mode = {"ship": PlayerMode.SHIP, "wave": PlayerMode.WAVE}.get(kind, PlayerMode.CUBE)
        if new_mode != mode or k == 0:
            sub = {PlayerMode.CUBE: ObjectSubtype.PORTAL_CUBE, PlayerMode.SHIP: ObjectSubtype.PORTAL_SHIP,
                   PlayerMode.WAVE: ObjectSubtype.PORTAL_WAVE}[new_mode]
            if k > 0:
                b.portal(cursor + BLOCK / 2, sub)
            mode = new_mode
        sp_x = cursor + 1.5 * BLOCK
        mid = CEILING / 2 - (5.0 if mode == PlayerMode.WAVE else 15.0)
        starts.append(StartPos(x=sp_x, y=0.0 if mode == PlayerMode.CUBE else mid, mode=mode, gravity=1,
                               x_speed=speed))
        # Room to settle: a cube falls from the roof; a ship or wave climbs from
        # the floor to mid-height (about 0.7 s), whatever the speed.
        settle = 6 if mode == PlayerMode.CUBE else max(5, math.ceil(0.7 * speed / BLOCK))
        cursor += settle * BLOCK

        jump_blocks = AIRTIME * speed / BLOCK  # how far one cube jump carries
        if kind == "cube":
            cursor = _cube_section(b, rng, cursor, jump_blocks, d)
        elif kind == "orb":
            cursor = _orb_section(b, rng, cursor, jump_blocks, d)
        elif kind == "gravity":
            cursor = _gravity_section(b, rng, cursor, jump_blocks, speed, d)
        elif kind == "ship":
            cursor = _ship_section(b, rng, cursor, speed, d)
        elif kind == "wave":
            cursor = _wave_section(b, rng, cursor, d)
        cursor += 4 * BLOCK

    length = cursor + 6 * BLOCK
    rows = b.rows
    objects = np.zeros(len(rows), dtype=OBJECT_DTYPE)
    motion = np.zeros((len(rows), 3), dtype=np.float64)
    for i, (x, y, w, h, kind, sub, amp, period, phase) in enumerate(rows):
        objects[i]["x"], objects[i]["y"], objects[i]["w"], objects[i]["h"] = x, y, w, h
        objects[i]["kind"], objects[i]["subtype"] = kind, sub
        motion[i] = (amp, period, phase)
    order = np.argsort(objects["x"] - objects["w"] / 2, kind="stable")
    return MockLevel(seed=seed, length=length, objects=objects[order], motion=motion[order], ceiling=CEILING,
                     start_positions=tuple(starts), name=f"mock-demon-{seed}", max_objects=MAX_OBJECTS_V2)


def _cube_section(b: _Builder, rng, cursor: float, jump_blocks: float, d: float) -> float:
    most = max_spikes(jump_blocks * BLOCK / AIRTIME)
    for _ in range(int(rng.integers(6, 10))):
        pattern = ("spike", "double", "triple", "block", "pad")[int(rng.integers(5))]
        if pattern == "double" and most < 2 or pattern == "triple" and most < 3:
            pattern = "spike"
        if pattern == "spike":
            b.spike(cursor)
            width = 1
        elif pattern == "double":
            b.spike(cursor)
            b.spike(cursor + BLOCK)
            width = 2
        elif pattern == "triple":
            for i in range(3):
                b.spike(cursor + i * BLOCK)
            width = 3
        elif pattern == "block":
            b.add(cursor + BLOCK / 2, BLOCK / 2, BLOCK, BLOCK, ObjectKind.SOLID, ObjectSubtype.BLOCK)
            b.spike(cursor + BLOCK)  # land on the block, then clear the spike behind it
            width = 2
        else:  # a jump pad, then a spike row as long as the pad's flight allows
            b.add(cursor + BLOCK / 2, 4.0, 24.0, 8.0, ObjectKind.PAD, ObjectSubtype.PAD_YELLOW)
            pad_blocks = jump_blocks * PAD_VELOCITY / JUMP_VELOCITY
            n = max(2, int(pad_blocks - 3.5))
            for i in range(n):
                b.spike(cursor + (2 + i) * BLOCK)
            width = 2 + n + 1
        # After landing there must be room to land and jump again.
        slack = 3.0 - 2.0 * d
        gap = math.ceil(jump_blocks + 0.5 + rng.uniform(0.0, slack))
        cursor += (width + gap) * BLOCK
    return cursor


def _orb_section(b: _Builder, rng, cursor: float, jump_blocks: float, d: float) -> float:
    for _ in range(int(rng.integers(3, 5))):
        pit = max(int(jump_blocks * 1.3) + 1, int(jump_blocks) + 2)
        for i in range(pit):
            b.spike(cursor + i * BLOCK)
        # The orb sits near the apex of a jump taken at the pit's edge.
        apex_x = cursor - BLOCK + jump_blocks * BLOCK / 2
        b.add(apex_x, 75.0, 36.0, 36.0, ObjectKind.ORB, ObjectSubtype.ORB_YELLOW)
        gap = math.ceil(jump_blocks + 1 + rng.uniform(0.0, 3.0 - 2.0 * d))
        cursor += (pit + gap) * BLOCK
    return cursor


def _gravity_section(b: _Builder, rng, cursor: float, jump_blocks: float, speed: float, d: float) -> float:
    b.portal(cursor, ObjectSubtype.PORTAL_GRAVITY_UP)
    fall_t = math.sqrt(2 * (CEILING - 30) / GRAVITY)
    cursor += (math.ceil(fall_t * speed / BLOCK) + 3) * BLOCK
    for _ in range(int(rng.integers(4, 7))):
        n = int(rng.integers(1, max_spikes(speed) + 1))
        for i in range(n):
            b.spike(cursor + i * BLOCK, upside_down=True)
        gap = math.ceil(jump_blocks + 0.5 + rng.uniform(0.0, 3.0 - 2.0 * d))
        cursor += (n + gap) * BLOCK
    b.portal(cursor, ObjectSubtype.PORTAL_GRAVITY_DOWN)
    cursor += (math.ceil(fall_t * speed / BLOCK) + 2) * BLOCK
    return cursor


def _ship_section(b: _Builder, rng, cursor: float, speed: float, d: float) -> float:
    gap_h = 105.0 - 30.0 * d  # 3.5 blocks easy, 2.5 blocks hard
    spacing_blocks = 6 if speed < 400 else 7
    centre = CEILING / 2
    for i in range(int(rng.integers(6, 10))):
        max_shift = 2.0 * BLOCK * (speed / X_SPEED) ** 0.5 if i else 0.0  # the first gate is centred
        centre = float(np.clip(centre + rng.uniform(-max_shift, max_shift), gap_h / 2 + 15, CEILING - gap_h / 2 - 15))
        x = cursor + BLOCK / 2
        b.wall(x, 0.0, centre - gap_h / 2)
        b.wall(x, centre + gap_h / 2, CEILING)
        if d >= 0.5 and i % 3 == 2:
            # A saw bobbing above or below the gate band between two gates: it
            # never enters the band, it punishes wide swings.
            amp = 25.0
            off = gap_h / 2 + 15 + 12 + amp + 6
            sy = centre + off if centre + off + amp + 12 < CEILING else centre - off
            if amp + 12 < sy < CEILING - amp - 12:
                b.add(cursor + spacing_blocks * BLOCK / 2, sy, 24.0, 24.0, ObjectKind.HAZARD, ObjectSubtype.SAW,
                      amp=amp, period=float(rng.uniform(200, 320)), phase=float(rng.uniform(0, 2 * math.pi)))
        cursor += spacing_blocks * BLOCK
    return cursor


def _wave_section(b: _Builder, rng, cursor: float, d: float) -> float:
    gap_h = 70.0 - 20.0 * d  # the wave is 10 units tall
    centre = CEILING / 2
    up = bool(rng.integers(2))
    for _ in range(int(rng.integers(8, 14))):
        shift = rng.uniform(0.8, 1.8) * BLOCK
        centre = centre + (shift if up else -shift)
        if not gap_h / 2 + 10 < centre < CEILING - gap_h / 2 - 10:
            up = not up
            centre = float(np.clip(centre, gap_h / 2 + 10, CEILING - gap_h / 2 - 10))
        up = not up if rng.random() < 0.7 else up
        x = cursor + BLOCK / 2
        b.wall(x, 0.0, centre - gap_h / 2, w=12.0)
        b.wall(x, centre + gap_h / 2, CEILING, w=12.0)
        cursor += 3 * BLOCK
    return cursor
