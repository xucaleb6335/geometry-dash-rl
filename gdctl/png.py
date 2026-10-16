"""Minimal PNG writer (stdlib zlib), for host-side window captures."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_rgb(width: int, height: int, rgb: bytes) -> bytes:
    """``rgb`` is top-down rows of width*3 bytes."""
    if len(rgb) != width * height * 3:
        raise ValueError(f"expected {width * height * 3} bytes, got {len(rgb)}")
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: none
        raw.extend(rgb[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _chunk(b"IEND", b""))


def bgra_to_rgb(width: int, height: int, bgra: bytes, bottom_up: bool = False) -> bytes:
    """Windows DIB pixels (BGRA) to top-down RGB."""
    out = bytearray(width * height * 3)
    stride = width * 4
    for y in range(height):
        src_y = height - 1 - y if bottom_up else y
        row = bgra[src_y * stride:(src_y + 1) * stride]
        dst = y * width * 3
        out[dst:dst + width * 3:3] = row[2::4]
        out[dst + 1:dst + width * 3:3] = row[1::4]
        out[dst + 2:dst + width * 3:3] = row[0::4]
    return bytes(out)


def downscale_rgb(width: int, height: int, rgb: bytes, target_width: int) -> tuple[int, int, bytes]:
    """Nearest-neighbour downscale to ``target_width`` (keeps aspect ratio)."""
    if target_width <= 0 or target_width >= width:
        return width, height, rgb
    out_w = target_width
    out_h = max(1, round(height * out_w / width))
    out = bytearray(out_w * out_h * 3)
    xs = [min(width - 1, (x * width) // out_w) * 3 for x in range(out_w)]
    for y in range(out_h):
        sy = min(height - 1, (y * height) // out_h)
        row = rgb[sy * width * 3:(sy + 1) * width * 3]
        base = y * out_w * 3
        for i, sx in enumerate(xs):
            out[base + i * 3:base + i * 3 + 3] = row[sx:sx + 3]
    return out_w, out_h, bytes(out)


def write_png(path: str | Path, width: int, height: int, rgb: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_rgb(width, height, rgb))
    return path
