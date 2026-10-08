"""Where GameStates come from. The env talks to a Backend and never cares which.

- ``SocketBackend``: TCP to the real mod (or to ``MockServer``). One game = one env.
- ``MockBackend``: the mock game in-process. No sockets, so it is fast and
  you can run many copies in a vector env while developing the agent.
"""

from __future__ import annotations

from typing import Protocol

from gdrl.env.client import GDClient
from gdrl.env.mock_game import MockGame, generate_level
from gdrl.env.protocol import DEFAULT_PORT, GameState


class Backend(Protocol):
    tick_rate: int

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState: ...

    def step(self, button: bool, ticks: int) -> GameState: ...

    def close(self) -> None: ...


class SocketBackend:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float | None = 10.0):
        self.client = GDClient(host, port, timeout)
        info = self.client.connect()
        self.tick_rate = info.tick_rate

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState:
        return self.client.reset(level_id, start_pos)

    def step(self, button: bool, ticks: int) -> GameState:
        return self.client.step(button, ticks)

    def close(self) -> None:
        self.client.close()


class MockBackend:
    """In-process mock. ``level_id`` 0 keeps the current level; k >= 1 loads seed k."""

    def __init__(self, level_seed: int = 1, **level_kwargs):
        self._level_kwargs = level_kwargs
        self.game = MockGame(generate_level(level_seed, **level_kwargs))
        self.tick_rate = self.game.tick_rate

    def reset(self, level_id: int = 0, start_pos: int = -1) -> GameState:
        if start_pos != -1:
            raise ValueError("the mock has no Start Pos objects")
        if level_id != 0 and level_id != self.game.level.seed:
            self.game = MockGame(generate_level(level_id, **self._level_kwargs))
        return self.game.reset()

    def step(self, button: bool, ticks: int) -> GameState:
        return self.game.step(button, ticks)

    def close(self) -> None:
        pass
