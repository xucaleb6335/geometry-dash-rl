"""Exclusive game lock: only one agent drives Geometry Dash at a time.

The lock is ``logs/game.lock`` (JSON: owner, pid, host, acquired/refreshed
times). It is advisory but gdctl enforces it: commands that change the game
refuse to run while another owner holds a fresh lock. A lock older than
``STALE_SECONDS`` since its last refresh can be taken over. Re-acquiring with
the same owner refreshes it (do that at least every hour during long jobs).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from gdctl import paths

STALE_SECONDS = 2 * 3600


def lock_path() -> Path:
    return Path(os.environ.get("GDCTL_LOCK", str(paths.REPO / "logs" / "game.lock")))


def read(path: Path | None = None) -> dict[str, Any] | None:
    path = path or lock_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return {"owner": "?", "corrupt": True, "refreshed": path.stat().st_mtime if path.exists() else 0}
    return data if isinstance(data, dict) else None


def age(data: dict[str, Any], now: float | None = None) -> float:
    return (now or time.time()) - float(data.get("refreshed", data.get("acquired", 0)))


def is_stale(data: dict[str, Any], now: float | None = None) -> bool:
    return age(data, now) > STALE_SECONDS


def describe(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {"held": False}
    return {"held": True, "owner": data.get("owner"), "pid": data.get("pid"), "host": data.get("host"),
            "note": data.get("note"), "acquired_at": data.get("acquired_at"),
            "age_seconds": round(age(data), 1), "stale": is_stale(data)}


def _record(owner: str, note: str | None, acquired: float | None = None) -> dict[str, Any]:
    now = time.time()
    acquired = acquired or now
    return {"owner": owner, "pid": os.getppid(), "host": socket.gethostname(), "note": note,
            "acquired": acquired, "acquired_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(acquired)),
            "refreshed": now}


def _create(path: Path, record: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return True


def acquire(owner: str, wait: float = 0.0, note: str | None = None, path: Path | None = None) -> dict[str, Any]:
    """Take (or refresh) the lock. Returns {"ok": bool, ...}."""
    if not owner:
        return {"ok": False, "code": "usage", "error": "owner is required"}
    path = path or lock_path()
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        cur = read(path)
        if cur is None:
            if _create(path, _record(owner, note)):
                return {"ok": True, "action": "acquired", **describe(read(path))}
            continue  # lost a race; look again
        if cur.get("owner") == owner:
            path.write_text(json.dumps(_record(owner, note or cur.get("note"), cur.get("acquired")), indent=2),
                            encoding="utf-8")
            return {"ok": True, "action": "refreshed", **describe(read(path))}
        if is_stale(cur):
            previous = describe(cur)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            if _create(path, _record(owner, note)):
                return {"ok": True, "action": "took_over_stale", "previous": previous, **describe(read(path))}
            continue
        if time.monotonic() >= deadline:
            return {"ok": False, "code": "locked", "error": f"game is locked by {cur.get('owner')!r}",
                    "lock": describe(cur)}
        time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))


def release(owner: str | None, force: bool = False, path: Path | None = None) -> dict[str, Any]:
    path = path or lock_path()
    cur = read(path)
    if cur is None:
        return {"ok": True, "action": "not_held"}
    if not force and cur.get("owner") != owner:
        return {"ok": False, "code": "not_owner",
                "error": f"lock is held by {cur.get('owner')!r}, not {owner!r} (use --force to break it)",
                "lock": describe(cur)}
    path.unlink(missing_ok=True)
    return {"ok": True, "action": "released", "previous": describe(cur)}


def check(owner: str | None, path: Path | None = None) -> dict[str, Any] | None:
    """None if ``owner`` may drive the game now, else an error dict."""
    cur = read(path or lock_path())
    if cur is None or is_stale(cur) or (owner and cur.get("owner") == owner):
        return None
    return {"ok": False, "code": "locked",
            "error": f"game is locked by {cur.get('owner')!r}; pass --owner <you> if that is you, "
                     "or wait: python -m gdctl lock acquire --owner <you> --wait 600",
            "lock": describe(cur)}
