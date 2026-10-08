"""Shared fixtures, plus one rule: a test that hits a ``LEARN(...)`` stub is
reported as *skipped* with the stub's name, not as a failure.

So ``pytest -rs`` doubles as your to-do list: every skip is something left to
build, and it turns green (or red) the moment you implement it.
"""

from __future__ import annotations

import pytest

from gdrl.env.protocol import BLOCK, GameState, ObjectKind, OBJECT_DTYPE, PlayerMode, StateFlag
import numpy as np


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    excinfo = call.excinfo
    if (
        call.when in ("setup", "call")
        and excinfo is not None
        and excinfo.errisinstance(NotImplementedError)
        and "LEARN(" in str(excinfo.value)
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"TODO {excinfo.value}")


def make_objects(*objs: tuple[float, float, float, float, ObjectKind]) -> np.ndarray:
    """Build an objects array from (x, y, w, h, kind) tuples, sorted by left edge."""
    arr = np.zeros(len(objs), dtype=OBJECT_DTYPE)
    for i, (x, y, w, h, kind) in enumerate(objs):
        arr[i]["x"], arr[i]["y"], arr[i]["w"], arr[i]["h"], arr[i]["kind"] = x, y, w, h, int(kind)
    order = np.argsort(arr["x"] - arr["w"] / 2, kind="stable")
    return arr[order]


def make_state(**overrides) -> GameState:
    """A plausible mid-level GameState; override any field."""
    fields = dict(
        tick=100,
        flags=StateFlag.ON_GROUND,
        player_x=500.0,
        player_y=15.0,
        vel_y=0.0,
        rotation=0.0,
        x_speed=311.58,
        level_length=3000.0,
        mode=PlayerMode.CUBE,
        objects=make_objects(
            (560.0, 9.0, 9.0, 18.0, ObjectKind.HAZARD),
            (700.0, 15.0, BLOCK, BLOCK, ObjectKind.SOLID),
        ),
    )
    fields.update(overrides)
    return GameState(**fields)


def scripted_jumper(state: GameState) -> bool:
    """Hold the button when the next obstacle is at the right distance.

    Bands are per kind (blocks need an earlier jump than spikes). Beats every
    mock level at frame_skip 1 to 4; used to prove the mock levels are beatable.
    """
    for obj in state.objects:
        dist = (obj["x"] - obj["w"] / 2) - state.player_x
        lo, hi = (28.0, 34.0) if obj["kind"] == ObjectKind.HAZARD else (36.0, 42.0)
        if lo <= dist <= hi:
            return True
    return False
