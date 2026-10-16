"""Binary wire format shared with the Geode mod.

Mirrors ``protocol/PROTOCOL.md`` and ``mod/src/Protocol.hpp``. Change all
three together and bump ``PROTOCOL_VERSION``.

Two versions are spoken. Version 1 is the original (STATE 676 bytes, 32
objects, RESET/STEP only). Version 2 adds snapshots (SAVE_STATE /
LOAD_STATE / DROP_STATE), player 2, hitbox size, floor/ceiling bounds, 64
objects, and an object subtype and flags. The client asks for 2 and falls
back to 1 when the server rejects it, so it still talks to an unpatched mod.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = 0x4C524447  # b"GDRL" read as little-endian u32
PROTOCOL_VERSION = 2  # the newest version this client speaks
SUPPORTED_VERSIONS = (1, 2)
MAX_OBJECTS = 32  # version 1
MAX_OBJECTS_V2 = 64
DEFAULT_PORT = 5555
DEFAULT_TICK_RATE = 240
BLOCK = 30.0  # game units per block


class MsgType(enum.IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    RESET = 3
    STEP = 4
    STATE = 5
    ERROR = 6
    BYE = 7
    SAVE_STATE = 8  # v2
    LOAD_STATE = 9  # v2
    DROP_STATE = 10  # v2


class StateFlag(enum.IntFlag):
    NONE = 0
    DEAD = 1 << 0
    COMPLETED = 1 << 1
    ON_GROUND = 1 << 2
    UPSIDE_DOWN = 1 << 3
    HOLDING = 1 << 4
    MINI = 1 << 5  # v2
    DUAL = 1 << 6  # v2: dual mode, the player 2 fields are valid


class ObjectKind(enum.IntEnum):
    NONE = 0
    SOLID = 1
    HAZARD = 2
    ORB = 3
    PAD = 4
    PORTAL = 5
    OTHER = 255


class ObjectSubtype(enum.IntEnum):
    """v2: what exactly an object is (v1 servers send 0 = UNKNOWN)."""

    UNKNOWN = 0
    BLOCK = 1
    SLOPE = 2
    SPIKE = 3
    SAW = 4  # animated hazard
    ORB_YELLOW = 10
    ORB_PINK = 11
    ORB_GRAVITY = 12  # blue
    ORB_GREEN = 13
    ORB_DROP = 14  # black
    ORB_RED = 15
    ORB_DASH = 16
    ORB_GRAVITY_DASH = 17
    ORB_SPIDER = 18
    ORB_TELEPORT = 19
    ORB_CUSTOM = 20
    PAD_YELLOW = 30
    PAD_PINK = 31
    PAD_GRAVITY = 32
    PAD_RED = 33
    PAD_SPIDER = 34
    PORTAL_CUBE = 40
    PORTAL_SHIP = 41
    PORTAL_BALL = 42
    PORTAL_UFO = 43
    PORTAL_WAVE = 44
    PORTAL_ROBOT = 45
    PORTAL_SPIDER = 46
    PORTAL_SWING = 47
    PORTAL_GRAVITY_DOWN = 50
    PORTAL_GRAVITY_UP = 51
    PORTAL_GRAVITY_TOGGLE = 52
    PORTAL_MINI = 53
    PORTAL_NORMAL_SIZE = 54
    PORTAL_DUAL = 55
    PORTAL_SOLO = 56
    PORTAL_MIRROR = 57
    PORTAL_MIRROR_OFF = 58
    PORTAL_TELEPORT = 59
    SPEED_SLOW = 60  # 0.5x
    SPEED_NORMAL = 61
    SPEED_2X = 62
    SPEED_3X = 63
    SPEED_4X = 64


class ObjectFlag(enum.IntFlag):
    """v2 per-object flags."""

    NONE = 0
    MOVED = 1 << 0  # not at its spawn position (moved by triggers)
    ACTIVATED = 1 << 1  # orb/pad/portal already used this attempt
    HIDDEN = 1 << 2  # invisible (alpha 0 or toggled off) but still collides


class PlayerMode(enum.IntEnum):
    CUBE = 0
    SHIP = 1
    BALL = 2
    UFO = 3
    WAVE = 4
    ROBOT = 5
    SPIDER = 6
    SWING = 7


class ErrorCode(enum.IntEnum):
    VERSION_MISMATCH = 1
    BAD_MESSAGE = 2
    NOT_IN_LEVEL = 3
    UNSUPPORTED = 4
    NO_SNAPSHOT = 5  # v2: LOAD_STATE of an empty slot, or a slot out of range


class ProtocolError(RuntimeError):
    """The bytes on the wire do not follow PROTOCOL.md."""


class ServerError(RuntimeError):
    """The server replied with an ERROR message."""

    def __init__(self, code: int, message: str):
        try:
            self.code: ErrorCode | int = ErrorCode(code)
        except ValueError:
            self.code = code
        super().__init__(f"{getattr(self.code, 'name', self.code)}: {message}")


HEADER = struct.Struct("<IHHI")
HELLO = struct.Struct("<HH")
HELLO_ACK = struct.Struct("<HHHH")  # v1: the last field is reserved; v2: max_snapshots
RESET = struct.Struct("<ii")
STEP = struct.Struct("<BBH")
ERROR = struct.Struct("<HH64s")
SNAPSHOT = struct.Struct("<iI")  # v2: slot, reserved

OBJECT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("kind", "u1"),
        ("subtype", "u1"),  # v2 ObjectSubtype; v1 sends 0
        ("oflags", "u1"),  # v2 ObjectFlag; v1 sends 0
        ("_reserved", "u1"),
    ]
)

STATE_DTYPE = np.dtype(
    [
        ("tick", "<u4"),
        ("flags", "<u4"),
        ("player_x", "<f4"),
        ("player_y", "<f4"),
        ("vel_y", "<f4"),
        ("rotation", "<f4"),
        ("x_speed", "<f4"),
        ("level_length", "<f4"),
        ("mode", "u1"),
        ("_reserved", "u1"),
        ("n_objects", "<u2"),
        ("objects", OBJECT_DTYPE, (MAX_OBJECTS,)),
    ]
)

STATE_DTYPE_V2 = np.dtype(
    [
        ("tick", "<u4"),
        ("flags", "<u4"),
        ("player_x", "<f4"),
        ("player_y", "<f4"),
        ("vel_y", "<f4"),
        ("rotation", "<f4"),
        ("x_speed", "<f4"),
        ("level_length", "<f4"),
        ("mode", "u1"),
        ("_reserved", "u1"),
        ("n_objects", "<u2"),
        ("player_size", "<f4"),  # side of player 1's hitbox (30 normal; smaller when mini or wave)
        ("floor_y", "<f4"),  # lowest y the player can stand on here (0 = the ground); NaN = none
        ("ceil_y", "<f4"),  # highest y (the roof of a ship/ball/wave corridor); NaN = none
        ("p2_x", "<f4"),
        ("p2_y", "<f4"),
        ("p2_vel_y", "<f4"),
        ("p2_mode", "u1"),
        ("p2_flags", "u1"),  # StateFlag bits DEAD, ON_GROUND, UPSIDE_DOWN, MINI
        ("_reserved2", "<u2"),
        ("objects", OBJECT_DTYPE, (MAX_OBJECTS_V2,)),
    ]
)

PAYLOAD_SIZES = {
    MsgType.HELLO: HELLO.size,
    MsgType.HELLO_ACK: HELLO_ACK.size,
    MsgType.RESET: RESET.size,
    MsgType.STEP: STEP.size,
    MsgType.STATE: STATE_DTYPE.itemsize,
    MsgType.ERROR: ERROR.size,
    MsgType.BYE: 0,
}
PAYLOAD_SIZES_V2 = {
    **PAYLOAD_SIZES,
    MsgType.STATE: STATE_DTYPE_V2.itemsize,
    MsgType.SAVE_STATE: SNAPSHOT.size,
    MsgType.LOAD_STATE: SNAPSHOT.size,
    MsgType.DROP_STATE: SNAPSHOT.size,
}

assert HEADER.size == 12
assert OBJECT_DTYPE.itemsize == 20
assert STATE_DTYPE.itemsize == 676
assert STATE_DTYPE_V2.itemsize == 1344


def payload_sizes(version: int) -> dict[MsgType, int]:
    return PAYLOAD_SIZES if version <= 1 else PAYLOAD_SIZES_V2


def state_dtype(version: int) -> np.dtype:
    return STATE_DTYPE if version <= 1 else STATE_DTYPE_V2


def max_objects(version: int) -> int:
    return MAX_OBJECTS if version <= 1 else MAX_OBJECTS_V2


def empty_objects(n: int = 0) -> np.ndarray:
    return np.zeros(n, dtype=OBJECT_DTYPE)


@dataclass(frozen=True)
class GameState:
    """One STATE message, decoded. This is the raw truth the encoder sees.

    Coordinates are game units (30 per block). ``player_x``/``player_y`` are the
    centre of the player's hitbox; the ground surface is ``y = 0``. ``objects``
    holds only the valid entries (``len == n_objects``), nearest first.
    The fields after ``objects`` come from protocol v2; a v1 server leaves
    them at their defaults.
    """

    tick: int
    flags: StateFlag
    player_x: float
    player_y: float
    vel_y: float
    rotation: float
    x_speed: float
    level_length: float
    mode: PlayerMode
    objects: np.ndarray = field(default_factory=empty_objects)
    player_size: float = 0.0  # 0 = unknown (v1)
    floor_y: float = float("nan")
    ceil_y: float = float("nan")
    p2_x: float = 0.0
    p2_y: float = 0.0
    p2_vel_y: float = 0.0
    p2_mode: PlayerMode = PlayerMode.CUBE
    p2_flags: StateFlag = StateFlag.NONE

    @property
    def dead(self) -> bool:
        return bool(self.flags & StateFlag.DEAD)

    @property
    def completed(self) -> bool:
        return bool(self.flags & StateFlag.COMPLETED)

    @property
    def terminal(self) -> bool:
        return self.dead or self.completed

    @property
    def on_ground(self) -> bool:
        return bool(self.flags & StateFlag.ON_GROUND)

    @property
    def upside_down(self) -> bool:
        return bool(self.flags & StateFlag.UPSIDE_DOWN)

    @property
    def holding(self) -> bool:
        return bool(self.flags & StateFlag.HOLDING)

    @property
    def mini(self) -> bool:
        return bool(self.flags & StateFlag.MINI)

    @property
    def dual(self) -> bool:
        return bool(self.flags & StateFlag.DUAL)

    @property
    def progress(self) -> float:
        """Fraction of the level travelled, in [0, 1]. Do not feed this to the policy."""
        if self.level_length <= 0:
            return 0.0
        return float(np.clip(self.player_x / self.level_length, 0.0, 1.0))

    def physics_bytes(self) -> bytes:
        """``to_bytes(2)`` without the objects' HIDDEN flag, which follows
        visual opacity (fade effects) and does not affect physics: a snapshot
        LOAD restores physics exactly but not every object's alpha."""
        import dataclasses

        objs = self.objects.copy()
        objs["oflags"] &= np.uint8(~int(ObjectFlag.HIDDEN) & 0xFF)
        return dataclasses.replace(self, objects=objs).to_bytes(2)

    def to_bytes(self, version: int = 1) -> bytes:
        """Serialize as a STATE payload of protocol ``version`` (default 1)."""
        cap = max_objects(version)
        if len(self.objects) > cap:
            raise ValueError(f"at most {cap} objects, got {len(self.objects)}")
        rec = np.zeros((), dtype=state_dtype(version))
        rec["tick"] = self.tick
        rec["flags"] = int(self.flags)
        rec["player_x"] = self.player_x
        rec["player_y"] = self.player_y
        rec["vel_y"] = self.vel_y
        rec["rotation"] = self.rotation
        rec["x_speed"] = self.x_speed
        rec["level_length"] = self.level_length
        rec["mode"] = int(self.mode)
        rec["n_objects"] = len(self.objects)
        rec["objects"][: len(self.objects)] = self.objects
        if version >= 2:
            rec["player_size"] = self.player_size
            rec["floor_y"] = self.floor_y
            rec["ceil_y"] = self.ceil_y
            rec["p2_x"] = self.p2_x
            rec["p2_y"] = self.p2_y
            rec["p2_vel_y"] = self.p2_vel_y
            rec["p2_mode"] = int(self.p2_mode)
            rec["p2_flags"] = int(self.p2_flags)
        return rec.tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> GameState:
        """Decode a STATE payload. The version is told apart by the size."""
        if len(payload) == STATE_DTYPE.itemsize:
            version = 1
        elif len(payload) == STATE_DTYPE_V2.itemsize:
            version = 2
        else:
            raise ProtocolError(
                f"STATE payload must be {STATE_DTYPE.itemsize} (v1) or {STATE_DTYPE_V2.itemsize} (v2) "
                f"bytes, got {len(payload)}")
        rec = np.frombuffer(payload, dtype=state_dtype(version), count=1)[0]
        n = int(rec["n_objects"])
        if n > max_objects(version):
            raise ProtocolError(f"n_objects={n} exceeds {max_objects(version)}")
        extra = {}
        if version >= 2:
            extra = dict(
                player_size=float(rec["player_size"]),
                floor_y=float(rec["floor_y"]),
                ceil_y=float(rec["ceil_y"]),
                p2_x=float(rec["p2_x"]),
                p2_y=float(rec["p2_y"]),
                p2_vel_y=float(rec["p2_vel_y"]),
                p2_mode=PlayerMode(int(rec["p2_mode"])),
                p2_flags=StateFlag(int(rec["p2_flags"])),
            )
        return cls(
            tick=int(rec["tick"]),
            flags=StateFlag(int(rec["flags"])),
            player_x=float(rec["player_x"]),
            player_y=float(rec["player_y"]),
            vel_y=float(rec["vel_y"]),
            rotation=float(rec["rotation"]),
            x_speed=float(rec["x_speed"]),
            level_length=float(rec["level_length"]),
            mode=PlayerMode(int(rec["mode"])),
            objects=rec["objects"][:n].copy(),
            **extra,
        )


def encode_frame(msg_type: MsgType, payload: bytes = b"", version: int = 1) -> bytes:
    """One framed message. ``version`` goes in the header and picks the valid sizes."""
    expected = payload_sizes(version).get(msg_type)
    if expected is None:
        raise ProtocolError(f"{msg_type.name} does not exist in protocol version {version}")
    if len(payload) != expected:
        raise ProtocolError(f"{msg_type.name} payload must be {expected} bytes, got {len(payload)}")
    return HEADER.pack(MAGIC, version, int(msg_type), len(payload)) + payload


def decode_header(header: bytes) -> tuple[MsgType, int]:
    """Validate a 12-byte header and return (type, payload length).

    The header's version field decides which types and sizes are valid.
    """
    magic, version, raw_type, length = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08X}")
    try:
        msg_type = MsgType(raw_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown message type {raw_type}") from exc
    sizes = payload_sizes(version)
    if msg_type not in sizes:
        raise ProtocolError(f"{msg_type.name} needs protocol version 2 (header says {version})")
    if length != sizes[msg_type]:
        raise ProtocolError(f"{msg_type.name} length {length} != {sizes[msg_type]}")
    return msg_type, length


def error_payload(code: ErrorCode, message: str) -> bytes:
    return ERROR.pack(int(code), 0, message.encode("utf-8")[:63])


def decode_error(payload: bytes) -> ServerError:
    code, _reserved, raw = ERROR.unpack(payload)
    return ServerError(code, raw.split(b"\0", 1)[0].decode("utf-8", errors="replace"))
