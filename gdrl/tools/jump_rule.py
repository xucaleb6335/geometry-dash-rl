"""M2 checks against the real game: a hard-coded jump rule plays the open level.

Open a cube level in GD (practice mode off), then:

    python -m gdrl.tools.jump_rule                    # 20 attempts, prints progress
    python -m gdrl.tools.jump_rule --attempts 500     # the unattended-reset gate

Each attempt sends RESET, then STEPs of a few ticks with the button decided by
a simple rule: jump when an obstacle starts within ``--jump-distance`` units
ahead. The rule is not meant to be good, only to prove that RESET, STEP, the
button and death detection all work. Works against the mock server too.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from gdrl.env.client import GDClient
from gdrl.env.protocol import DEFAULT_PORT, GameState, ObjectKind

PLAYER_HALF = 15.0  # cube hitbox half-size, game units


def should_jump(state: GameState, jump_distance: float) -> bool:
    """Jump if a hazard, or a solid we would run into, starts just ahead."""
    objs = state.objects
    if len(objs) == 0:
        return False
    left = objs["x"] - objs["w"] / 2
    top = objs["y"] + objs["h"] / 2
    gap = left - (state.player_x + PLAYER_HALF)
    ahead = (gap >= 0) & (gap <= jump_distance)
    feet = state.player_y - PLAYER_HALF
    hazard = objs["kind"] == ObjectKind.HAZARD
    wall = (objs["kind"] == ObjectKind.SOLID) & (top > feet + 5)
    return bool(np.any(ahead & (hazard | wall)))


def run_attempt(client: GDClient, ticks: int, jump_distance: float, max_steps: int) -> tuple[GameState, int]:
    state = client.reset()
    if state.tick != 0:
        raise RuntimeError(f"RESET returned tick {state.tick}, expected 0")
    steps = 0
    while not state.terminal and steps < max_steps:
        state = client.step(should_jump(state, jump_distance), ticks)
        steps += 1
    return state, steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play the open level with a hard-coded jump rule (M2).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=2, help="ticks per STEP")
    parser.add_argument("--jump-distance", type=float, default=20.0, help="units ahead of the player to jump at (tune per level)")
    parser.add_argument("--max-steps", type=int, default=20000, help="give up on an attempt after this many steps")
    args = parser.parse_args(argv)

    deaths = completions = 0
    best = 0.0
    start = time.perf_counter()
    with GDClient(args.host, args.port, timeout=30.0) as client:
        print(f"connected: protocol v{client.info.protocol_version}, {client.info.tick_rate} TPS")
        for i in range(1, args.attempts + 1):
            state, steps = run_attempt(client, args.ticks, args.jump_distance, args.max_steps)
            progress = 100.0 * state.player_x / state.level_length if state.level_length else 0.0
            best = max(best, progress)
            deaths += state.dead
            completions += state.completed
            outcome = "COMPLETED" if state.completed else "dead" if state.dead else "timeout"
            print(f"attempt {i:4d}: {outcome:9s} at tick {state.tick:6d}, x {state.player_x:8.1f} ({progress:5.1f}%)", flush=True)
    elapsed = time.perf_counter() - start
    print(f"\n{args.attempts} attempts in {elapsed:.0f}s: {deaths} deaths, {completions} completions, best {best:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
