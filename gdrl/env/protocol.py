"""Binary wire format shared with the Geode mod.

Mirrors ``protocol/PROTOCOL.md`` and ``mod/src/Protocol.hpp``. Change all
three together and bump ``PROTOCOL_VERSION``.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = 0x4C524447  # b"GDRL" read as little-endian u32
PROTOCOL_VERSION = 1
MAX_OBJECTS = 32
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


class StateFlag(enum.IntFlag):
    NONE = 0
    DEAD = 1 << 0
    COMPLETED = 1 << 1
    ON_GROUND = 1 << 2
    UPSIDE_DOWN = 1 << 3
    HOLDING = 1 << 4


class ObjectKind(enum.IntEnum):
    NONE = 0
    SOLID = 1
    HAZARD = 2
    ORB = 3
    PAD = 4
    PORTAL = 5
    OTHER = 255


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
HELLO_ACK = struct.Struct("<HHHH")
RESET = struct.Struct("<ii")
STEP = struct.Struct("<BBH")
ERROR = struct.Struct("<HH64s")

OBJECT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("kind", "u1"),
        ("_reserved", "u1", (3,)),
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

PAYLOAD_SIZES = {
    MsgType.HELLO: HELLO.size,
    MsgType.HELLO_ACK: HELLO_ACK.size,
    MsgType.RESET: RESET.size,
    MsgType.STEP: STEP.size,
    MsgType.STATE: STATE_DTYPE.itemsize,
    MsgType.ERROR: ERROR.size,
    MsgType.BYE: 0,
}

assert HEADER.size == 12
assert OBJECT_DTYPE.itemsize == 20
assert STATE_DTYPE.itemsize == 676


def empty_objects(n: int = 0) -> np.ndarray:
    return np.zeros(n, dtype=OBJECT_DTYPE)


@dataclass(frozen=True)
class GameState:
    """One STATE message, decoded. This is the raw truth the encoder sees.

    Coordinates are game units (30 per block). ``player_x``/``player_y`` are the
    centre of the player's hitbox; the ground surface is ``y = 0``. ``objects``
    holds only the valid entries (``len == n_objects``), nearest first.
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
    def progress(self) -> float:
        """Fraction of the level travelled, in [0, 1]. Do not feed this to the policy."""
        if self.level_length <= 0:
            return 0.0
        return float(np.clip(self.player_x / self.level_length, 0.0, 1.0))

    def to_bytes(self) -> bytes:
        if len(self.objects) > MAX_OBJECTS:
            raise ValueError(f"at most {MAX_OBJECTS} objects, got {len(self.objects)}")
        rec = np.zeros((), dtype=STATE_DTYPE)
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
        return rec.tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> GameState:
        if len(payload) != STATE_DTYPE.itemsize:
            raise ProtocolError(f"STATE payload must be {STATE_DTYPE.itemsize} bytes, got {len(payload)}")
        rec = np.frombuffer(payload, dtype=STATE_DTYPE, count=1)[0]
        n = int(rec["n_objects"])
        if n > MAX_OBJECTS:
            raise ProtocolError(f"n_objects={n} exceeds {MAX_OBJECTS}")
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
        )


def encode_frame(msg_type: MsgType, payload: bytes = b"") -> bytes:
    expected = PAYLOAD_SIZES[msg_type]
    if len(payload) != expected:
        raise ProtocolError(f"{msg_type.name} payload must be {expected} bytes, got {len(payload)}")
    return HEADER.pack(MAGIC, PROTOCOL_VERSION, int(msg_type), len(payload)) + payload


def decode_header(header: bytes) -> tuple[MsgType, int]:
    """Validate a 12-byte header and return (type, payload length)."""
    magic, _version, raw_type, length = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08X}")
    try:
        msg_type = MsgType(raw_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown message type {raw_type}") from exc
    if length != PAYLOAD_SIZES[msg_type]:
        raise ProtocolError(f"{msg_type.name} length {length} != {PAYLOAD_SIZES[msg_type]}")
    return msg_type, length


def error_payload(code: ErrorCode, message: str) -> bytes:
    return ERROR.pack(int(code), 0, message.encode("utf-8")[:63])


def decode_error(payload: bytes) -> ServerError:
    code, _reserved, raw = ERROR.unpack(payload)
    return ServerError(code, raw.split(b"\0", 1)[0].decode("utf-8", errors="replace"))
