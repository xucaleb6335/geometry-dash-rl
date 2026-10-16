"""Where GameStates come from. The env talks to a Backend and never cares which.

- ``SocketBackend``: TCP to the real mod (or to ``MockServer``). One game = one env.
- ``MockBackend``: the mock game in-process. No sockets, so it is fast and
  you can run many copies in a vector env while developing the agent.
"""

from __future__ import annotations

from typing import Protocol

from gdrl.env.client import GDClient
from gdrl.env.mock_game import MockGame, make_mock_level
from gdrl.env.protocol import DEFAULT_PORT, ErrorCode, GameState, ServerError


class Backend(Protocol):
    tick_rate: int

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState: ...

    def step(self, button: bool, ticks: int) -> GameState: ...

    def close(self) -> None: ...


class SnapshotBackend(Backend, Protocol):
    """A backend that can save and restore the whole game state (protocol v2).

    Slots are numbered ``0 .. max_snapshots - 1``; saving overwrites. Loading
    replies the restored state, with the tick it had when saved.
    """

    max_snapshots: int

    def save_state(self, slot: int) -> GameState: ...

    def load_state(self, slot: int) -> GameState: ...

    def drop_state(self, slot: int) -> GameState: ...


def wait_until_running(backend, start_pos: int = -1, timeout: float = 20.0) -> bool:
    """RESET and STEP until a STEP really moves the player (or kills it).

    A freshly opened level answers STEPs during its intro with 0 physics
    ticks, while the bridge still counts them. Inputs recorded then are off
    by those ticks when replayed later (the search's "restore was inexact"
    on Stereo Madness). Call this once after opening a level."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        s0 = backend.reset(0, start_pos)
        s1 = backend.step(False, 4)
        if s1.dead or s1.player_x != s0.player_x or s1.player_y != s0.player_y:
            backend.reset(0, start_pos)
            return True
        time.sleep(0.2)
    return False


def supports_snapshots(backend) -> bool:
    return int(getattr(backend, "max_snapshots", 0)) > 0


class SocketBackend:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float | None = 10.0):
        self.client = GDClient(host, port, timeout)
        info = self.client.connect()
        self.tick_rate = info.tick_rate
        self.protocol_version = info.protocol_version
        self.max_snapshots = info.max_snapshots if self.client.supports_snapshots else 0

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState:
        return self.client.reset(level_id, start_pos)

    def step(self, button: bool, ticks: int) -> GameState:
        return self.client.step(button, ticks)

    def save_state(self, slot: int) -> GameState:
        return self.client.save_state(slot)

    def load_state(self, slot: int) -> GameState:
        return self.client.load_state(slot)

    def drop_state(self, slot: int) -> GameState:
        return self.client.drop_state(slot)

    def close(self) -> None:
        self.client.close()


class MockBackend:
    """In-process mock. ``level_id`` 0 keeps the current level; k >= 1 loads level k
    of the generator (``kind: simple`` = seed k of the cube levels, ``kind: demon``
    = the long mixed-mode levels). Supports Start Pos and snapshots."""

    def __init__(self, level_seed: int = 1, restore_error: float = 0.0, max_snapshots: int = 100_000,
                 **level_kwargs):
        self._level_kwargs = level_kwargs
        self._restore_error = restore_error
        self.game = MockGame(make_mock_level(level_seed, **level_kwargs), restore_error=restore_error)
        self.tick_rate = self.game.tick_rate
        self.max_snapshots = max_snapshots
        self._slots: dict[int, tuple] = {}

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState:
        if level_id != 0 and level_id != self.game.level.seed:
            self.game = MockGame(make_mock_level(level_id, **self._level_kwargs), restore_error=self._restore_error)
            self._slots.clear()
        return self.game.reset(start_pos)

    def step(self, button: bool, ticks: int) -> GameState:
        return self.game.step(button, ticks)

    def _check(self, slot: int) -> None:
        if not 0 <= slot < self.max_snapshots:
            raise ServerError(ErrorCode.NO_SNAPSHOT, f"slot {slot} out of range")

    def save_state(self, slot: int) -> GameState:
        self._check(slot)
        self._slots[slot] = self.game.snapshot()
        return self.game.state()

    def load_state(self, slot: int) -> GameState:
        self._check(slot)
        if slot not in self._slots:
            raise ServerError(ErrorCode.NO_SNAPSHOT, f"slot {slot} is empty")
        return self.game.restore(self._slots[slot])

    def drop_state(self, slot: int) -> GameState:
        self._check(slot)
        self._slots.pop(slot, None)
        return self.game.state()

    def close(self) -> None:
        pass
