"""Command-line interface: ``python -m gdctl <command> [options]``.

Every command prints exactly one JSON object to stdout. ``ok`` is true on
success; on failure ``code`` and ``error`` say why and the exit code is 1
(2 for usage errors, which are JSON too).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import os

from gdctl import __version__, game, leveledit, levelstring, lock, paths
from gdctl.client import ControlError, request
from gdctl.game import GdctlError


class UsageError(Exception):
    pass


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # noqa: D401 - argparse hook
        raise UsageError(f"{self.prog}: {message}")


# --------------------------------------------------------------------------
# Helpers

def proxy(cmd: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Send a command to the control mod; return its reply as the result."""
    try:
        return request(cmd, params, timeout=timeout)
    except ControlError as exc:
        return {"ok": False, "code": exc.code, "error": exc.message,
                "hint": "is the game running with the control mod? try: python -m gdctl status"}


def default_shot_path() -> Path:
    return paths.screens_dir() / f"shot-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.png"


def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# --------------------------------------------------------------------------
# Command handlers: (args) -> dict

def c_status(a):
    return {"ok": True, **game.status(deep=not a.shallow)}


def c_launch(a):
    return {"ok": True, **game.launch(via=a.via, timeout=a.timeout, close_popups=not a.keep_popups,
                                      wait_menu=not a.no_wait)}


def c_kill(a):
    return {"ok": True, **game.kill(graceful_timeout=a.grace, force=not a.no_force)}


def c_restart(a):
    return {"ok": True, **game.restart(via=a.via, timeout=a.timeout)}


def c_build(a):
    targets = a.targets or ["mod", "control-mod"]
    restart = True if a.relaunch else False if a.no_relaunch else None
    res = game.build(targets, stop_first=a.stop_first, restart_game=restart, timeout=a.timeout)
    if not res["ok"]:
        failed = [r["target"] for r in res["results"] if not r["ok"]]
        return {"ok": False, "code": "build_failed", "error": f"build failed: {', '.join(failed)}", **res}
    return res


def c_logs(a):
    return {"ok": True, **game.read_log(lines=a.lines, grep=a.grep, crash=a.crash, file=a.file,
                                        ignore_case=not a.case_sensitive)}


def c_screenshot(a):
    path = Path(a.path) if a.path else default_shot_path()
    path = path.resolve()
    if not a.host:
        reply = proxy("screenshot", {"path": str(path), "width": a.width}, timeout=20.0)
        if reply.get("ok") or a.mod:
            reply["source"] = "mod"
            return reply
        fallback_reason = reply.get("error")
    else:
        fallback_reason = None
    res = game.host_screenshot(path, width=a.width, method=a.method)
    if fallback_reason:
        res["fallback_reason"] = fallback_reason
    return {"ok": True, **res}


def c_focus(a):
    from gdctl import win32

    pids = game.game_pids()
    if not pids:
        raise GdctlError("not_running", "game is not running")
    win = win32.main_window(pids[0], paths.WINDOW_TITLE)
    if not win:
        raise GdctlError("no_window", "game has no visible window")
    ok = win32.focus(win["hwnd"])
    return {"ok": ok, "hwnd": win["hwnd"], "foreground": ok,
            **({} if ok else {"code": "focus_failed", "error": "Windows refused to change the foreground window"})}


def c_input(a):
    from gdctl import win32

    pids = game.game_pids()
    if not pids:
        raise GdctlError("not_running", "game is not running")
    win = win32.main_window(pids[0], paths.WINDOW_TITLE)
    if not win:
        raise GdctlError("no_window", "game has no visible window")
    if not a.no_focus:
        win32.focus(win["hwnd"])
    if a.action == "key":
        win32.send_key(win32.vk_code(a.key), hold=a.hold)
        return {"ok": True, "key": a.key, "hwnd": win["hwnd"]}
    if a.fx is not None and a.fy is not None:
        cx, cy = int(a.fx * win["client_w"]), int(a.fy * win["client_h"])
    elif a.px is not None and a.py is not None:
        cx, cy = a.px, a.py
    else:
        raise UsageError("input click needs --fx/--fy (fractions) or --px/--py (client pixels)")
    sx, sy = win32.send_click(win["hwnd"], cx, cy, hold=a.hold)
    return {"ok": True, "client": [cx, cy], "screen": [sx, sy], "hwnd": win["hwnd"]}


def c_mods(a):
    if a.action == "list":
        return {"ok": True, **game.mod_listing()}
    changes = {m: a.action == "enable" for m in a.mod_ids}
    if not changes:
        raise UsageError("mods enable/disable needs at least one mod id")
    return {"ok": True, **game.set_mods(changes, restart_game=a.restart)}


def c_prepare(a):
    return {"ok": True, **game.prepare_training(a.level, practice=a.practice, reopen=a.reopen,
                                                probe=not a.no_probe, timeout=a.timeout)}


def c_bridge_probe(a):
    from gdctl import bridge_probe

    return bridge_probe.probe(paths.BRIDGE_PORT, reset=not a.no_reset)


# ---- game-side (proxied to the control mod)

def c_ping(a):
    return proxy("ping")


def c_scene(a):
    return proxy("scene")


def c_tree(a):
    params = {"root": a.root, "depth": a.depth, "visible_only": not a.all, "clickable_only": a.clickable,
              "match": a.match, "max_nodes": a.max_nodes}
    return proxy("tree", params)


def c_click(a):
    params = {"path": a.path, "node_id": a.id, "force": a.force or None, "touch": a.touch or None}
    if a.x is not None:
        params.update(x=a.x, y=a.y)
    if a.fx is not None:
        params.update(fx=a.fx, fy=a.fy)
    if not any(params.get(k) is not None for k in ("path", "node_id", "x", "fx")):
        raise UsageError("click needs --path, --id, --x/--y or --fx/--fy")
    return proxy("click", params)


def c_key(a):
    return proxy("key", {"key": a.key, "action": a.action, "code": a.code})


def c_open_level(a):
    params = game.parse_level_spec(a.level)
    params["practice"] = a.practice
    return proxy("open_level", params, timeout=a.timeout)


def c_exit_level(a):
    return proxy("exit_level", {"menu": a.menu}, timeout=a.timeout)


def c_menu(a):
    return proxy("menu", {"save": a.save_editor}, timeout=a.timeout)


def c_close_popups(a):
    return proxy("close_popups")


def c_list_levels(a):
    return proxy("list_levels", {"types": a.types})


def c_create_level(a):
    if a.file:
        data = Path(a.file).read_text(encoding="utf-8").strip()
    elif a.string:
        data = a.string
    elif a.test:
        data = levelstring.test_level()
    else:
        raise UsageError("create-level needs --string, --file or --test")
    if not levelstring.is_compressed(data):
        data = levelstring.compress(data)
    params = {"name": a.name, "level_string": data, "replace": a.replace, "save": not a.no_save,
              "audio_track": a.audio_track, "song_id": a.song_id}
    return proxy("create_level", params)


def c_delete_level(a):
    params = game.parse_level_spec(a.level)
    if params["type"] != "local":
        raise UsageError("delete-level only deletes local levels (local:<name>, local-id:<m_id>)")
    params.update(force=a.force or None, save=not a.no_save)
    params.pop("type")
    return proxy("delete_level", params)


def c_quit(a):
    return proxy("quit_game", {"save": not a.no_save})


def c_prevent_pause(a):
    params = {}
    if a.state is not None:
        params["enabled"] = a.state == "on"
    if a.persist:
        params["persist"] = True
    return proxy("prevent_pause", params)


def c_option(a):
    if a.action == "get":
        return proxy("get_option", {"key": a.key})
    if a.key is None or a.value is None:
        raise UsageError("option set needs KEY and VALUE")
    return proxy("set_option", {"key": a.key, "value": parse_value(a.value)})


def c_wait_scene(a):
    deadline = time.monotonic() + a.timeout
    last = None
    while True:
        try:
            last = request("scene", timeout=5.0)
        except ControlError as exc:
            last = {"ok": False, "code": exc.code, "error": exc.message}
        if last.get("ok") and not last.get("transitioning"):
            layers = last.get("layers") or []
            if (a.layer is None or a.layer in layers) and (not a.in_level or last.get("in_level")) and (
                    not a.no_popups or not last.get("popups")):
                return last
        if time.monotonic() > deadline:
            return {"ok": False, "code": "timeout", "error": "scene condition not met in time", "last": last}
        time.sleep(0.3)


# ---- online levels and level editing

def c_search_levels(a):
    return proxy("search_levels", {"query": a.query, "page": a.page}, timeout=a.timeout)


def c_download_level(a):
    return proxy("download_level", {"id": a.id, "song": not a.no_song, "force": a.force or None,
                                    "wait": a.timeout - 5}, timeout=a.timeout)


def c_download_song(a):
    return proxy("download_song", {"id": a.id, "wait": a.timeout - 5}, timeout=a.timeout)


def c_copy_to_local(a):
    return proxy("copy_to_local", {"id": a.id, "name": a.name, "replace": a.replace,
                                   "strip_song": a.strip_song, "save": True})


def fetch_level(spec: str) -> tuple[dict[str, Any], str]:
    """(level info, raw level string) for any level spec."""
    params = game.parse_level_spec(spec)
    reply = proxy("get_level_string", params, timeout=30.0)
    if not reply.get("ok"):
        raise GdctlError(reply.get("code", "error"), reply.get("error", "get_level_string failed"))
    return reply["level"], levelstring.decompress(reply["level_string"])


def store_level(spec: str, raw: str, force: bool) -> dict[str, Any]:
    params = game.parse_level_spec(spec)
    if params["type"] != "local":
        raise UsageError("only local levels can be written (copy online levels with copy-to-local first)")
    params.pop("type")
    params.update(level_string=levelstring.compress(raw), force=force or None, save=True)
    reply = proxy("set_level_string", params, timeout=30.0)
    if not reply.get("ok"):
        raise GdctlError(reply.get("code", "error"), reply.get("error", "set_level_string failed"))
    return reply["level"]


def c_level_export(a):
    level, raw = fetch_level(a.level)
    path = Path(a.path or paths.REPO / "logs" / "levels" / f"{level['name'] or 'level'}-{level['id'] or level['m_id']}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(levelstring.compress(raw) if a.compressed else raw, encoding="utf-8")
    return {"ok": True, "level": level, "path": str(path.resolve()), "compressed": a.compressed,
            "chars": len(raw), **{k: v for k, v in leveledit.info(raw).items() if k != "startpos"}}


def c_level_import(a):
    raw = levelstring.decompress(Path(a.path).read_text(encoding="utf-8").strip())
    level = store_level(a.level, raw, a.force)
    return {"ok": True, "level": level, "objects": leveledit.info(raw)["objects"]}


def c_level_info(a):
    level, raw = fetch_level(a.level)
    return {"ok": True, "level": level, **leveledit.info(raw)}


def c_startpos(a):
    level, raw = fetch_level(a.level)
    if a.action == "list":
        return {"ok": True, "level": level, "startpos": leveledit.list_startpos(raw)}
    if a.action == "remove":
        if not a.all and a.x is None:
            raise UsageError("startpos remove needs --x or --all")
        raw, removed = leveledit.remove_startpos(raw, None if a.all else [a.x], tolerance=a.tolerance)
        level = store_level(a.level, raw, a.force)
        return {"ok": True, "level": level, "removed": removed, "startpos": leveledit.list_startpos(raw)}
    if a.x is None:
        raise UsageError(f"startpos {a.action} needs --x")
    base = leveledit.infer_state(raw, a.x) if not a.no_infer else None
    state = leveledit.make_state(a.mode, a.speed, {"normal": False, "flipped": True}.get(a.gravity),
                                 a.mini, a.dual, base=base)
    removed = 0
    if a.action == "set":
        raw, state, removed = leveledit.set_only_startpos(raw, a.x, a.y, state)
    else:
        raw, state = leveledit.add_startpos(raw, a.x, a.y, state)
    level = store_level(a.level, raw, a.force)
    return {"ok": True, "level": level, "added": {"x": a.x, "y": a.y, **state.describe()},
            "inferred": not a.no_infer, "removed": removed, "startpos": leveledit.list_startpos(raw)}


def c_lock(a):
    if a.action == "status":
        return {"ok": True, "file": str(lock.lock_path()), **lock.describe(lock.read())}
    if not a.owner and not (a.action == "release" and a.force):
        raise UsageError("lock acquire/release need --owner (or GDCTL_OWNER)")
    if a.action == "acquire":
        return lock.acquire(a.owner, wait=a.wait, note=a.note)
    return lock.release(a.owner, force=a.force)


def c_stop_training(a):
    """Ask overnight.ps1 to stop (logs/overnight.stop) and wait for it to exit."""
    from gdctl import win32

    logs = paths.REPO / "logs"
    pid_file = logs / "overnight.pid"
    if not pid_file.exists():
        return {"ok": True, "action": "not_running", "lock": lock.describe(lock.read())}
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid = None
    if pid and not win32.pid_alive(pid):
        pid_file.unlink(missing_ok=True)
        return {"ok": True, "action": "not_running", "note": "stale pid file removed",
                "lock": lock.describe(lock.read())}
    (logs / "overnight.stop").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    deadline = time.monotonic() + a.wait
    while time.monotonic() < deadline:
        if not pid_file.exists() or (pid and not win32.pid_alive(pid)):
            return {"ok": True, "action": "stopped", "pid": pid, "lock": lock.describe(lock.read())}
        time.sleep(1.0)
    return {"ok": False, "code": "timeout", "error": f"overnight.ps1 (pid {pid}) did not stop in {a.wait:.0f} s",
            "pid": pid}


def c_raw(a):
    try:
        msg = json.loads(a.json)
    except json.JSONDecodeError as exc:
        raise UsageError(f"raw: invalid JSON: {exc}") from exc
    if not isinstance(msg, dict) or "cmd" not in msg:
        raise UsageError('raw needs a JSON object with "cmd"')
    cmd = msg.pop("cmd")
    msg.pop("seq", None)
    timeout = float(msg.pop("timeout", a.timeout))
    return proxy(cmd, msg, timeout=timeout)


# --------------------------------------------------------------------------
# Parser

def build_parser() -> argparse.ArgumentParser:
    p = JsonArgumentParser(prog="python -m gdctl", description="Drive Geometry Dash (JSON output).")
    p.add_argument("--version", action="version", version=f"gdctl {__version__}")
    p.add_argument("--owner", default=os.environ.get("GDCTL_OWNER"),
                   help="who you are, for the game lock (default: GDCTL_OWNER env var)")
    sub = p.add_subparsers(dest="command", required=True, parser_class=JsonArgumentParser)

    def add(name, func, help_text, aliases=()):
        sp = sub.add_parser(name, help=help_text, description=help_text, aliases=list(aliases))
        sp.set_defaults(func=func)
        return sp

    # host side
    sp = add("status", c_status, "process, window, ports, control-mod health")
    sp.add_argument("--shallow", action="store_true", help="do not query the control mod")
    sp = add("launch", c_launch, "start the game and wait for the main menu")
    sp.add_argument("--via", choices=["auto", "exe", "steam"], default="auto")
    sp.add_argument("--timeout", type=float, default=120.0)
    sp.add_argument("--keep-popups", action="store_true", help="do not close startup popups")
    sp.add_argument("--no-wait", action="store_true", help="return once the process exists")
    sp = add("kill", c_kill, "quit the game (clean quit, then WM_CLOSE, then taskkill /F)")
    sp.add_argument("--grace", type=float, default=30.0, help="seconds to wait for a clean exit")
    sp.add_argument("--no-force", action="store_true", help="never taskkill /F")
    sp = add("restart", c_restart, "kill, then launch")
    sp.add_argument("--via", choices=["auto", "exe", "steam"], default="auto")
    sp.add_argument("--timeout", type=float, default=120.0)
    sp = add("build", c_build, "geode build mod and/or control-mod, install, relaunch if it was running")
    sp.add_argument("targets", nargs="*", choices=["mod", "bridge", "control-mod", "control"],
                    help="default: both")
    sp.add_argument("--stop-first", action="store_true", help="stop the game before building")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--relaunch", action="store_true", help="(re)launch the game after a good build")
    g.add_argument("--no-relaunch", action="store_true", help="leave the game as it is after the build")
    sp.add_argument("--timeout", type=float, default=1800.0)
    sp = add("logs", c_logs, "tail the newest Geode log")
    sp.add_argument("-n", "--lines", type=int, default=50, help="last N lines (0 = all)")
    sp.add_argument("-g", "--grep", help="regex filter (case-insensitive)")
    sp.add_argument("--case-sensitive", action="store_true")
    sp.add_argument("--crash", action="store_true", help="newest crash log instead")
    sp.add_argument("--file", help="a specific log file")
    sp = add("screenshot", c_screenshot, "PNG of the game (mod render; host capture fallback)")
    sp.add_argument("--path", help="output PNG (default logs/screens/shot-<time>.png)")
    sp.add_argument("--width", type=int, default=960, help="output width in pixels (0 = native)")
    sp.add_argument("--host", action="store_true", help="skip the mod; capture the window")
    sp.add_argument("--mod", action="store_true", help="mod only; no host fallback")
    sp.add_argument("--method", choices=["auto", "print", "screen"], default="auto",
                    help="host capture: screen (accurate, GD on top), print (PrintWindow, may be stale)")
    add("focus", c_focus, "bring the game window to the foreground")
    sp = add("input", c_input, "synthetic OS input (SendInput) fallback")
    isub = sp.add_subparsers(dest="action", required=True, parser_class=JsonArgumentParser)
    ic = isub.add_parser("click", help="left click in the client area")
    ic.add_argument("--fx", type=float, help="x as a fraction of the client width (0-1)")
    ic.add_argument("--fy", type=float, help="y as a fraction of the client height, from the top")
    ic.add_argument("--px", type=int, help="client x pixel")
    ic.add_argument("--py", type=int, help="client y pixel")
    ik = isub.add_parser("key", help="press and release a key")
    ik.add_argument("key", help="space, enter, escape, a-z, 0-9, f1-f12, arrows ...")
    for x in (ic, ik):
        x.add_argument("--hold", type=float, default=0.05)
        x.add_argument("--no-focus", action="store_true")
    sp = add("mods", c_mods, "list installed mods, or enable/disable them")
    sp.add_argument("action", nargs="?", choices=["list", "enable", "disable"], default="list")
    sp.add_argument("mod_ids", nargs="*")
    sp.add_argument("--restart", action="store_true", help="restart the game to apply")
    sp = add("prepare-training", c_prepare, "game up, training mods, level open, bridge ready")
    sp.add_argument("--level", required=True, help="main:1, local:<name>, local-id:<m_id>, saved:<id>")
    sp.add_argument("--practice", action="store_true")
    sp.add_argument("--reopen", action="store_true", help="reopen even if the level is already open")
    sp.add_argument("--no-probe", action="store_true", help="skip the HELLO/RESET check on 5555")
    sp.add_argument("--timeout", type=float, default=180.0)
    sp = add("bridge-probe", c_bridge_probe, "HELLO (+RESET) + BYE on the RL bridge port")
    sp.add_argument("--no-reset", action="store_true")

    # game side
    add("ping", c_ping, "control mod: versions, mods")
    add("scene", c_scene, "current scene, level, popups")
    sp = add("tree", c_tree, "node tree of the current scene")
    sp.add_argument("--root", help="start at this node path")
    sp.add_argument("--depth", type=int, default=12)
    sp.add_argument("--all", action="store_true", help="include invisible nodes")
    sp.add_argument("--clickable", action="store_true", help="only enabled, visible menu items")
    sp.add_argument("--match", help="substring filter on class/id/text/label/path")
    sp.add_argument("--max-nodes", type=int, default=3000)
    sp = add("click", c_click, "activate a button by path/id, or touch a point")
    sp.add_argument("--path")
    sp.add_argument("--id", help="node ID, searched recursively")
    sp.add_argument("--x", type=float, help="cocos points from the left")
    sp.add_argument("--y", type=float, help="cocos points from the bottom")
    sp.add_argument("--fx", type=float, help="fraction of window width")
    sp.add_argument("--fy", type=float, help="fraction of window height, from the top")
    sp.add_argument("--force", action="store_true", help="click even if invisible/disabled")
    sp.add_argument("--touch", action="store_true", help="simulate a touch at the node instead of activate()")
    sp = add("key", c_key, "keyboard key or the back button (ESC)")
    sp.add_argument("key", help="back, space, enter, escape, a-z, 0-9, up/down/left/right, f1-f12")
    sp.add_argument("--action", choices=["tap", "down", "up"], default="tap")
    sp.add_argument("--code", type=int, help="raw enumKeyCodes value")
    sp = add("open-level", c_open_level, "open a level (main:1, local:<name>, saved:<id>)")
    sp.add_argument("level")
    sp.add_argument("--practice", action="store_true")
    sp.add_argument("--timeout", type=float, default=60.0)
    sp = add("exit-level", c_exit_level, "quit the open level")
    sp.add_argument("--menu", action="store_true", help="then go to the main menu")
    sp.add_argument("--timeout", type=float, default=60.0)
    sp = add("menu", c_menu, "go to the main menu from anywhere")
    sp.add_argument("--save-editor", action="store_true", help="if in the editor, save and exit it")
    sp.add_argument("--timeout", type=float, default=60.0)
    add("close-popups", c_close_popups, "close open FLAlertLayer/dropdown popups")
    sp = add("list-levels", c_list_levels, "main, local and saved levels")
    sp.add_argument("--types", default="main,local,saved")
    sp = add("create-level", c_create_level, "create a local level from a level string")
    sp.add_argument("--name", required=True, help="level name (max 20 chars); tests use gdrl-test-*")
    sp.add_argument("--string", help="raw or compressed (H4sI...) level string")
    sp.add_argument("--file", help="file holding the level string")
    sp.add_argument("--test", action="store_true", help="a built-in flat level with a few spikes")
    sp.add_argument("--replace", action="store_true", help="replace a local level with the same name")
    sp.add_argument("--no-save", action="store_true", help="do not write CCLocalLevels.dat now")
    sp.add_argument("--audio-track", type=int)
    sp.add_argument("--song-id", type=int)
    sp = add("delete-level", c_delete_level, "delete a local level (gdrl-test-* unless --force)")
    sp.add_argument("level", help="local:<name> or local-id:<m_id>")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-save", action="store_true")
    sp = add("quit", c_quit, "clean exit through the control mod")
    sp.add_argument("--no-save", action="store_true")
    sp = add("prevent-pause", c_prevent_pause, "get/set never-pause-on-focus-loss")
    sp.add_argument("state", nargs="?", choices=["on", "off"])
    sp.add_argument("--persist", action="store_true", help="also store it in the mod setting")
    sp = add("option", c_option, "get/set game options (show_percent, fps, 0040 ...)")
    sp.add_argument("action", choices=["get", "set"])
    sp.add_argument("key", nargs="?")
    sp.add_argument("value", nargs="?")
    sp = add("wait-scene", c_wait_scene, "wait for a settled scene matching the conditions")
    sp.add_argument("--layer", help="a layer class that must be in the scene, e.g. MenuLayer")
    sp.add_argument("--in-level", action="store_true")
    sp.add_argument("--no-popups", action="store_true")
    sp.add_argument("--timeout", type=float, default=30.0)
    sp = add("search-levels", c_search_levels, "online search by name or id")
    sp.add_argument("query")
    sp.add_argument("--page", type=int, default=0)
    sp.add_argument("--timeout", type=float, default=30.0)
    sp = add("download-level", c_download_level, "download an online level (and its song)")
    sp.add_argument("id", type=int)
    sp.add_argument("--no-song", action="store_true")
    sp.add_argument("--force", action="store_true", help="download again even if saved")
    sp.add_argument("--timeout", type=float, default=120.0)
    sp = add("download-song", c_download_song, "download a custom song by id")
    sp.add_argument("id", type=int)
    sp.add_argument("--timeout", type=float, default=120.0)
    sp = add("copy-to-local", c_copy_to_local, "clone a downloaded online level into a local level")
    sp.add_argument("id", type=int)
    sp.add_argument("--name", required=True, help="local name, max 20 chars; use gdrl-<name> so it can be edited")
    sp.add_argument("--replace", action="store_true")
    sp.add_argument("--strip-song", action="store_true", help="no custom song in the copy")
    sp = add("level-export", c_level_export, "write a level's string to a file (raw by default)")
    sp.add_argument("level", help="main:1, local:<name>, local-id:<m_id>, saved:<id>")
    sp.add_argument("--path")
    sp.add_argument("--compressed", action="store_true", help="write GD's gzip+base64 form")
    sp = add("level-import", c_level_import, "replace a local level's string from a file")
    sp.add_argument("level", help="local:<name> or local-id:<m_id> (named gdrl-* unless --force)")
    sp.add_argument("--path", required=True)
    sp.add_argument("--force", action="store_true")
    sp = add("level-info", c_level_info, "object count, last object x, start settings, StartPos list")
    sp.add_argument("level")
    sp = add("startpos", c_startpos, "list/add/remove/set StartPos objects of a local level")
    sp.add_argument("level", help="local:<name> or local-id:<m_id>")
    sp.add_argument("action", choices=["list", "add", "remove", "set"],
                    help="set = remove all StartPos, add one (play from there)")
    sp.add_argument("--x", type=float, help="editor x (30 units per block)")
    sp.add_argument("--y", type=float, default=15.0, help="editor y; ground = 0, cube on ground = 15")
    sp.add_argument("--mode", choices=leveledit.MODES)
    sp.add_argument("--speed", choices=sorted(leveledit.SPEEDS))
    sp.add_argument("--gravity", choices=["normal", "flipped"])
    sp.add_argument("--mini", action="store_true", default=None)
    sp.add_argument("--dual", action="store_true", default=None)
    sp.add_argument("--no-infer", action="store_true", help="do not infer settings from portals before x")
    sp.add_argument("--all", action="store_true", help="remove: every StartPos")
    sp.add_argument("--tolerance", type=float, default=15.0, help="remove: match x within this")
    sp.add_argument("--force", action="store_true", help="edit a level not named gdrl-*")
    sp = add("lock", c_lock, "exclusive game lock (mandatory for agents; see README)")
    sp.add_argument("action", choices=["acquire", "release", "status"])
    sp.add_argument("--wait", type=float, default=0.0, help="acquire: seconds to wait for the lock")
    sp.add_argument("--note", help="acquire: what you are doing")
    sp.add_argument("--force", action="store_true", help="release: break someone else's lock")
    sp = add("stop-training", c_stop_training, "stop overnight.ps1 (releases its game lock)")
    sp.add_argument("--wait", type=float, default=120.0)
    sp = add("raw", c_raw, "send a raw JSON command to the control mod")
    sp.add_argument("json")
    sp.add_argument("--timeout", type=float, default=30.0)
    return p


# Commands that never change the game; everything else checks the lock.
READ_ONLY = {"status", "logs", "screenshot", "ping", "scene", "tree", "list-levels", "level-info",
             "level-export", "search-levels", "wait-scene", "lock", "focus", "stop-training"}


def run(argv: list[str] | None = None) -> tuple[int, dict[str, Any]]:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command not in READ_ONLY:
            denied = lock.check(args.owner)
            if denied:
                return 1, denied
        result = args.func(args)
    except UsageError as exc:
        return 2, {"ok": False, "code": "usage", "error": str(exc)}
    except ValueError as exc:
        return 2, {"ok": False, "code": "usage", "error": str(exc)}
    except GdctlError as exc:
        return 1, {"ok": False, "code": exc.code, "error": exc.message, **exc.extra}
    except ControlError as exc:
        return 1, {"ok": False, "code": exc.code, "error": exc.message}
    except SystemExit as exc:  # --help / --version
        return int(exc.code or 0), {}
    except Exception as exc:  # noqa: BLE001 - agents need JSON even for bugs
        return 1, {"ok": False, "code": "internal", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        result = {"ok": True, "result": result}
    result.setdefault("ok", True)
    return (0 if result.get("ok") else 1), result


def main(argv: list[str] | None = None) -> int:
    code, result = run(argv)
    if result:
        sys.stdout.write(json.dumps(result, indent=None if len(json.dumps(result)) > 4000 else 2,
                                    ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    return code
