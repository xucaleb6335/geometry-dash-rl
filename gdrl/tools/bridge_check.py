"""M3 gate checks for any server that speaks PROTOCOL.md: the mod, the mock
server or the C++ standalone server.

    python -m gdrl.tools.bridge_check --port 5555                 # all checks
    python -m gdrl.tools.bridge_check --port 5555 --soak 10000    # the full soak

Checks:
  determinism  the same RESET + STEP sequence twice gives byte-identical STATEs
  speed        STEP round trips per second and simulated ticks per wall second
  soak         many short episodes in a row with no hang, error or desync
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gdrl.env.client import GDClient
from gdrl.env.protocol import DEFAULT_PORT, GameState


def describe_diff(x: bytes, y: bytes) -> str:
    """Which STATE fields differ between two encodings, for a failure message."""
    a, b = GameState.from_bytes(x), GameState.from_bytes(y)
    diffs = []
    for name in ("tick", "flags", "player_x", "player_y", "vel_y", "rotation", "x_speed", "level_length", "mode"):
        va, vb = getattr(a, name), getattr(b, name)
        if va != vb:
            diffs.append(f"{name} {va!r} vs {vb!r}")
    if len(a.objects) != len(b.objects):
        diffs.append(f"n_objects {len(a.objects)} vs {len(b.objects)}")
    elif a.objects.tobytes() != b.objects.tobytes():
        diffs.append("objects")
    return ", ".join(diffs) or "reserved bytes only"


def check_determinism(client: GDClient, steps: int, ticks: int, seed: int = 0) -> tuple[bool, str]:
    actions = np.random.default_rng(seed).integers(2, size=steps).astype(bool)
    streams = []
    for _ in range(2):
        chunks = [client.reset().to_bytes()]
        for a in actions:
            state = client.step(bool(a), ticks)
            chunks.append(state.to_bytes())
            if state.terminal:
                break
        streams.append(chunks)
    a, b = streams
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"streams differ at step {i}: {describe_diff(x, y)}"
    if len(a) != len(b):
        return False, f"streams have different lengths ({len(a)} vs {len(b)})"
    return True, f"{len(a)} states identical"


def check_speed(client: GDClient, steps: int, ticks: int) -> tuple[bool, str]:
    client.reset()
    done_steps, sim_ticks = 0, 0
    start = time.perf_counter()
    while done_steps < steps:
        state = client.step(done_steps % 7 == 0, ticks)
        done_steps += 1
        if state.terminal:
            sim_ticks += state.tick
            client.reset()
    elapsed = time.perf_counter() - start
    sim_ticks += state.tick
    rate = sim_ticks / client.info.tick_rate / elapsed
    msg = f"{done_steps / elapsed:,.0f} steps/s, {rate:.1f}x real time"
    return rate > 1.0, msg


def check_soak(client: GDClient, episodes: int, ticks: int, max_steps: int = 200) -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    start = time.perf_counter()
    for ep in range(episodes):
        state = client.reset()
        if state.tick != 0:
            return False, f"episode {ep}: RESET returned tick {state.tick}"
        for _ in range(max_steps):
            prev_tick = state.tick
            state = client.step(bool(rng.integers(2)), ticks)
            if state.terminal:
                break
            if state.tick != prev_tick + ticks:
                return False, f"episode {ep}: tick went {prev_tick} -> {state.tick}, expected +{ticks}"
    return True, f"{episodes} episodes in {time.perf_counter() - start:.1f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3 gate checks for a bridge server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ticks", type=int, default=4, help="ticks per STEP (frame skip)")
    parser.add_argument("--steps", type=int, default=500, help="steps for the determinism and speed checks")
    parser.add_argument("--soak", type=int, default=200, help="episodes for the soak check (gate: 10000)")
    args = parser.parse_args(argv)

    ok = True
    with GDClient(args.host, args.port, timeout=30.0) as client:
        print(f"connected: protocol v{client.info.protocol_version}, {client.info.tick_rate} TPS")
        for name, fn in (
            ("determinism", lambda: check_determinism(client, args.steps, args.ticks)),
            ("speed", lambda: check_speed(client, args.steps, args.ticks)),
            ("soak", lambda: check_soak(client, args.soak, args.ticks)),
        ):
            passed, msg = fn()
            ok &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {name:12s} {msg}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
