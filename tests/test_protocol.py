import numpy as np
import pytest

from gdrl.env.protocol import (
    HEADER,
    MAX_OBJECTS,
    OBJECT_DTYPE,
    STATE_DTYPE,
    ErrorCode,
    GameState,
    MsgType,
    ObjectKind,
    ProtocolError,
    StateFlag,
    decode_error,
    decode_header,
    encode_frame,
    error_payload,
)
from tests.conftest import make_objects, make_state


def test_sizes_match_protocol_md():
    assert HEADER.size == 12
    assert OBJECT_DTYPE.itemsize == 20
    assert STATE_DTYPE.itemsize == 676


def test_state_round_trip():
    state = make_state(flags=StateFlag.DEAD | StateFlag.HOLDING, vel_y=-123.5)
    decoded = GameState.from_bytes(state.to_bytes())
    assert decoded.tick == state.tick
    assert decoded.flags == state.flags
    assert decoded.dead and decoded.holding and not decoded.completed
    assert decoded.vel_y == pytest.approx(-123.5)
    assert len(decoded.objects) == len(state.objects)
    np.testing.assert_array_equal(decoded.objects, state.objects)


def test_state_field_offsets():
    """Spot-check byte offsets against the table in PROTOCOL.md."""
    raw = make_state(tick=0x01020304, player_x=1.5).to_bytes()
    assert raw[0:4] == bytes([4, 3, 2, 1])  # little-endian u32 tick at offset 0
    assert np.frombuffer(raw[8:12], "<f4")[0] == 1.5  # player_x at offset 8
    assert np.frombuffer(raw[34:36], "<u2")[0] == 2  # n_objects at offset 34
    assert raw[36 + 16] == ObjectKind.HAZARD  # first object's kind at 36 + 16


def test_too_many_objects_rejected():
    objs = make_objects(*[(float(i * 30), 9.0, 9.0, 18.0, ObjectKind.HAZARD) for i in range(MAX_OBJECTS + 1)])
    with pytest.raises(ValueError):
        make_state(objects=objs).to_bytes()


def test_frame_header_round_trip():
    frame = encode_frame(MsgType.STEP, b"\x01\x00\x04\x00")
    msg_type, length = decode_header(frame[:12])
    assert msg_type == MsgType.STEP and length == 4


@pytest.mark.parametrize(
    "header",
    [
        b"XXXX" + b"\x01\x00\x04\x00\x04\x00\x00\x00",  # bad magic
        HEADER.pack(0x4C524447, 1, 99, 0),  # unknown type
        HEADER.pack(0x4C524447, 1, int(MsgType.STEP), 5),  # wrong length
    ],
)
def test_bad_headers_raise(header):
    with pytest.raises(ProtocolError):
        decode_header(header)


def test_error_round_trip():
    err = decode_error(error_payload(ErrorCode.NOT_IN_LEVEL, "open a level first"))
    assert err.code == ErrorCode.NOT_IN_LEVEL
    assert "open a level first" in str(err)
