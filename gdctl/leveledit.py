"""Edit raw GD level strings: StartPos objects, level length, portal state.

Coordinates are editor units (30 per block); the ground surface is y = 0, so a
cube standing on the ground has its centre at y = 15. In game (and in the RL
bridge's raw positions) everything is 90 units higher.

A StartPos (object id 31) carries its own start settings, as GD's editor
writes them::

    1,31,2,<x>,3,<y>,kA2,<mode>,kA3,<mini>,kA8,<dual>,kA4,<speed>,kA9,1,kA10,0,kA11,<flip>

GD starts the level from a StartPos when one exists. To make the start
unambiguous, ``set_only_startpos`` removes every other StartPos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from gdctl import levelstring

STARTPOS_ID = 31

MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider", "swing"]
# kA4 speed codes, GD's Speed enum.
SPEEDS = {"normal": 0, "1x": 0, "slow": 1, "0.5x": 1, "fast": 2, "2x": 2, "faster": 3, "3x": 3,
          "fastest": 4, "4x": 4}
SPEED_NAMES = {0: "1x", 1: "0.5x", 2: "2x", 3: "3x", 4: "4x"}

MODE_PORTALS = {12: 0, 13: 1, 47: 2, 111: 3, 660: 4, 745: 5, 1331: 6, 1933: 7}
SPEED_PORTALS = {201: 0, 200: 1, 202: 2, 203: 3, 1334: 4}
GRAVITY_PORTALS = {10: False, 11: True}  # blue = normal, yellow = flipped
GRAVITY_TOGGLE = 2926                     # green portal
SIZE_PORTALS = {99: False, 101: True}     # normal, mini
DUAL_PORTALS = {286: True, 287: False}


@dataclass
class StartState:
    mode: int = 0       # index into MODES
    speed: int = 0      # kA4 code
    flipped: bool = False
    mini: bool = False
    dual: bool = False

    def describe(self) -> dict:
        d = asdict(self)
        d["mode_name"] = MODES[self.mode] if 0 <= self.mode < len(MODES) else str(self.mode)
        d["speed_name"] = SPEED_NAMES.get(self.speed, str(self.speed))
        return d


def split(raw: str) -> tuple[str, list[str]]:
    """Raw (or compressed) level string -> (header, object strings)."""
    raw = levelstring.decompress(raw)
    parts = raw.split(";")
    return parts[0], [p for p in parts[1:] if p]


def join(header: str, objects: list[str]) -> str:
    return header + ";" + "".join(o + ";" for o in objects)


def fields(obj: str) -> dict[str, str]:
    f = obj.split(",")
    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}


def header_fields(header: str) -> dict[str, str]:
    return fields(header)


def _f(d: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except ValueError:
        return default


def header_state(header: str) -> StartState:
    h = header_fields(header)
    return StartState(mode=int(_f(h, "kA2")), speed=int(_f(h, "kA4")), flipped=_f(h, "kA11") == 1,
                      mini=_f(h, "kA3") == 1, dual=_f(h, "kA8") == 1)


def infer_state(raw: str, x: float) -> StartState:
    """Mode/speed/gravity/size/dual a player would have at ``x``, from the
    level start settings and the portals before ``x``. Approximate: it assumes
    every portal left of ``x`` is passed (no alternate paths, no triggers)."""
    header, objects = split(raw)
    state = header_state(header)
    portals = []
    for o in objects:
        f = fields(o)
        oid = int(_f(f, "1", -1))
        ox = _f(f, "2")
        if ox < x and (oid in MODE_PORTALS or oid in SPEED_PORTALS or oid in GRAVITY_PORTALS
                       or oid == GRAVITY_TOGGLE or oid in SIZE_PORTALS or oid in DUAL_PORTALS):
            portals.append((ox, oid))
    for _, oid in sorted(portals):
        if oid in MODE_PORTALS:
            state.mode = MODE_PORTALS[oid]
        elif oid in SPEED_PORTALS:
            state.speed = SPEED_PORTALS[oid]
        elif oid in GRAVITY_PORTALS:
            state.flipped = GRAVITY_PORTALS[oid]
        elif oid == GRAVITY_TOGGLE:
            state.flipped = not state.flipped
        elif oid in SIZE_PORTALS:
            state.mini = SIZE_PORTALS[oid]
        elif oid in DUAL_PORTALS:
            state.dual = DUAL_PORTALS[oid]
    return state


def startpos_string(x: float, y: float, state: StartState) -> str:
    n = levelstring._num
    return (f"1,{STARTPOS_ID},2,{n(x)},3,{n(y)},kA2,{state.mode},kA3,{int(state.mini)},kA8,{int(state.dual)},"
            f"kA4,{state.speed},kA9,1,kA10,0,kA11,{int(state.flipped)}")


def list_startpos(raw: str) -> list[dict]:
    _, objects = split(raw)
    out = []
    for i, o in enumerate(objects):
        f = fields(o)
        if f.get("1") != str(STARTPOS_ID):
            continue
        st = StartState(mode=int(_f(f, "kA2")), speed=int(_f(f, "kA4")), flipped=_f(f, "kA11") == 1,
                        mini=_f(f, "kA3") == 1, dual=_f(f, "kA8") == 1)
        out.append({"index": i, "x": _f(f, "2"), "y": _f(f, "3"), **st.describe()})
    return sorted(out, key=lambda s: s["x"])


def remove_startpos(raw: str, xs: list[float] | None = None, tolerance: float = 15.0) -> tuple[str, int]:
    """Remove StartPos objects (all when ``xs`` is None, else those within
    ``tolerance`` of one of ``xs``). Returns (raw, removed count)."""
    header, objects = split(raw)
    kept, removed = [], 0
    for o in objects:
        f = fields(o)
        if f.get("1") == str(STARTPOS_ID) and (xs is None or any(abs(_f(f, "2") - x) <= tolerance for x in xs)):
            removed += 1
            continue
        kept.append(o)
    return join(header, kept), removed


def add_startpos(raw: str, x: float, y: float = 15.0, state: StartState | None = None) -> tuple[str, StartState]:
    """Append a StartPos at (x, y). ``state`` None = infer from portals."""
    state = state or infer_state(raw, x)
    header, objects = split(raw)
    objects.append(startpos_string(x, y, state))
    return join(header, objects), state


def set_only_startpos(raw: str, x: float, y: float = 15.0, state: StartState | None = None) -> tuple[str, StartState, int]:
    """Remove every StartPos, then add one at (x, y): the level starts there."""
    raw, removed = remove_startpos(raw)
    raw, state = add_startpos(raw, x, y, state)
    return raw, state, removed


def info(raw: str) -> dict:
    header, objects = split(raw)
    last_x = 0.0
    counts = {"startpos_count": 0, "portals": 0}
    for o in objects:
        f = fields(o)
        oid = int(_f(f, "1", -1))
        last_x = max(last_x, _f(f, "2"))
        if oid == STARTPOS_ID:
            counts["startpos_count"] += 1
        elif (oid in MODE_PORTALS or oid in SPEED_PORTALS or oid in GRAVITY_PORTALS or oid == GRAVITY_TOGGLE
              or oid in SIZE_PORTALS or oid in DUAL_PORTALS):
            counts["portals"] += 1
    return {"objects": len(objects), "last_object_x": last_x, "last_object_block": round(last_x / 30, 1),
            "start": header_state(header).describe(), "startpos": list_startpos(raw), **counts}


def make_state(mode: str | int | None = None, speed: str | int | None = None, flipped: bool | None = None,
               mini: bool | None = None, dual: bool | None = None, base: StartState | None = None) -> StartState:
    """Override fields of ``base`` (default: cube, 1x)."""
    st = StartState(**asdict(base)) if base else StartState()
    if mode is not None:
        st.mode = MODES.index(mode) if isinstance(mode, str) and not mode.isdigit() else int(mode)
    if speed is not None:
        st.speed = SPEEDS[speed] if isinstance(speed, str) and speed in SPEEDS else int(speed)
    if flipped is not None:
        st.flipped = flipped
    if mini is not None:
        st.mini = mini
    if dual is not None:
        st.dual = dual
    return st
