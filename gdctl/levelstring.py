"""GD level strings: compress/decompress and a tiny builder.

A raw level string is ``<header>;<object>;<object>;...``. Each object is
comma-separated key,value pairs: 1 = object id, 2 = x, 3 = y (game units,
30 per block; the ground surface is y = 0 in editor coordinates, so a block
resting on the ground has its centre at y = 15). The header holds level
settings (``kA..``/``kS..`` keys) and may be minimal.

GD stores level data gzip-compressed and URL-safe base64 encoded ("H4sI...").
"""

from __future__ import annotations

import base64
import gzip
import zlib

# Common object ids.
BLOCK = 1      # default solid block
SPIKE = 8      # spike
YELLOW_ORB = 36
YELLOW_PAD = 35

# Header in the format of GD's bundled main levels (Resources/levels/1.txt):
# kS29/kS30 = background/ground colour, kA2 = start mode (0 cube), kA4 =
# start speed (0 normal), the rest default.
DEFAULT_HEADER = ("kS29,1_40_2_125_3_255_4_0_5_0,kS30,1_0_2_102_3_255_4_0_5_0,"
                  "kA13,0,kA15,0,kA16,0,kA14,,kA6,0,kA7,0,kA2,0,kA3,0,kA8,0,kA4,0,kA9,0,kA10,0,kA11,0")


def is_compressed(data: str) -> bool:
    return data.startswith("H4sI")


def compress(raw: str) -> str:
    """Raw level string -> GD's stored form (gzip, URL-safe base64)."""
    packed = gzip.compress(raw.encode("utf-8"), compresslevel=9, mtime=0)
    return base64.urlsafe_b64encode(packed).decode("ascii")


def decompress(data: str) -> str:
    """GD's stored form -> raw level string. Raw input is returned unchanged."""
    if not is_compressed(data):
        return data
    padded = data + "=" * (-len(data) % 4)
    packed = base64.urlsafe_b64decode(padded.encode("ascii"))
    # zlib with wbits=47 accepts gzip or zlib headers.
    return zlib.decompress(packed, 47).decode("utf-8")


def obj(object_id: int, x: float, y: float, **extra: object) -> str:
    """One object: ``obj(BLOCK, 165, 15)``. ``extra`` keys are GD keys like k6="90"."""
    parts = [f"1,{object_id}", f"2,{_num(x)}", f"3,{_num(y)}"]
    for key, value in extra.items():
        parts.append(f"{key.lstrip('k')},{value}")
    return ",".join(parts)


def build(objects: list[str], header: str = DEFAULT_HEADER) -> str:
    """Raw level string from a header and object strings."""
    return header + ";" + "".join(o + ";" for o in objects)


def parse_objects(raw: str) -> list[dict[str, str]]:
    """Split a raw level string into object dicts (header skipped)."""
    raw = decompress(raw)
    out = []
    for chunk in raw.split(";")[1:]:
        if not chunk:
            continue
        fields = chunk.split(",")
        out.append({fields[i]: fields[i + 1] for i in range(0, len(fields) - 1, 2)})
    return out


def test_level(length_blocks: int = 40, spikes_every: int = 8) -> str:
    """A flat cube level with single spikes every ``spikes_every`` blocks."""
    objects = []
    for b in range(10, length_blocks, spikes_every):
        objects.append(obj(SPIKE, b * 30 + 15, 15))
    objects.append(obj(BLOCK, length_blocks * 30 + 15, 15))
    return build(objects)


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"
