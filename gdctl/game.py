"""Host-side game control: status, launch, kill, build, logs, mods, and the
prepare-training workflow. Works when the game is closed or hung.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from gdctl import bridge_probe, client, mods, paths, win32
from gdctl.client import ControlError


class GdctlError(Exception):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


# --------------------------------------------------------------------------
# Level specs: "main:1", "1", "local:My Level", "local-id:5", "local-index:0", "saved:128"

def parse_level_spec(spec: str) -> dict[str, Any]:
    spec = spec.strip()
    if not spec:
        raise ValueError("empty level spec")
    if spec.lstrip("-").isdigit():
        return {"type": "main", "id": int(spec)}
    kind, sep, value = spec.partition(":")
    if not sep or value == "":
        raise ValueError(f"bad level spec {spec!r}: use main:<id>, local:<name>, local-id:<m_id>, "
                         "local-index:<i> or saved:<id>")
    kind = kind.lower()
    if kind == "main":
        return {"type": "main", "id": _int(value, spec)}
    if kind in ("local", "editor", "created"):
        return {"type": "local", "name": value}
    if kind in ("local-id", "local_id", "mid"):
        return {"type": "local", "id": _int(value, spec)}
    if kind in ("local-index", "local_index"):
        return {"type": "local", "index": _int(value, spec)}
    if kind in ("saved", "online"):
        return {"type": "saved", "id": _int(value, spec)}
    raise ValueError(f"unknown level type {kind!r} in {spec!r}")


def _int(value: str, spec: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"bad number in level spec {spec!r}") from None


# --------------------------------------------------------------------------
# Status

def game_pids() -> list[int]:
    return [p["pid"] for p in win32.processes(paths.EXE_NAME)]


def status(deep: bool = True) -> dict[str, Any]:
    pids = game_pids()
    ports = win32.listening_ports()
    out: dict[str, Any] = {
        "running": bool(pids),
        "pid": pids[0] if pids else None,
        "pids": pids,
        "ports": {
            str(paths.BRIDGE_PORT): _port_info(ports, paths.BRIDGE_PORT, pids),
            str(paths.CONTROL_PORT): _port_info(ports, paths.CONTROL_PORT, pids),
        },
        "window": None,
        "control": None,
        # Small = someone is using the keyboard/mouse; think twice before
        # taking over the game.
        "user_idle_seconds": round(win32.user_idle_seconds(), 1),
    }
    if pids:
        win = win32.main_window(pids[0], paths.WINDOW_TITLE)
        if win:
            win["responsive"] = win32.responsive(win["hwnd"], 1500)
        out["window"] = win
    if deep and ports.get(paths.CONTROL_PORT):
        ctl: dict[str, Any] = {"net": client.reachable(timeout=2.0)}
        try:
            ping = client.request("ping", timeout=3.0)
            ctl["main_thread"] = bool(ping.get("ok"))
            ctl.update({k: ping.get(k) for k in ("mod_version", "geode_version", "gd_version", "menu_reached",
                                                  "prevent_pause", "frame")})
            scene = client.request("scene", timeout=3.0)
            if scene.get("ok"):
                ctl["scene"] = {k: scene.get(k) for k in ("main_layer", "transitioning", "in_level", "in_editor")}
                lvl = scene.get("level")
                if lvl:
                    ctl["scene"]["level"] = {k: lvl.get(k) for k in ("id", "name", "type", "practice", "paused",
                                                                      "percent", "attempt")}
                ctl["scene"]["popups"] = len(scene.get("popups") or [])
        except ControlError as exc:
            ctl["main_thread"] = False
            ctl["error"] = exc.message
        out["control"] = ctl
    out["responsive"] = bool(out["window"] and out["window"].get("responsive")) and (
        out["control"] is None or out["control"].get("main_thread", False))
    return out


def _port_info(ports: dict[int, int], port: int, pids: list[int]) -> dict[str, Any]:
    owner = ports.get(port)
    return {"listening": owner is not None, "pid": owner, "owned_by_game": owner in pids if owner else False}


# --------------------------------------------------------------------------
# Waiting helpers

def wait_for(pred: Callable[[], Any], timeout: float, interval: float = 0.5) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = pred()
        except ControlError:
            value = None
        if value:
            return value
        if time.monotonic() > deadline:
            return None
        time.sleep(interval)


def ctl(cmd: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Control-mod request that raises GdctlError on failure."""
    try:
        reply = client.request(cmd, params, timeout=timeout)
    except ControlError as exc:
        raise GdctlError(exc.code, exc.message) from exc
    if not reply.get("ok"):
        raise GdctlError(reply.get("code", "error"), reply.get("error", "command failed"), reply=reply)
    return reply


def wait_for_menu(timeout: float = 120.0, close_popups: bool = True) -> dict[str, Any]:
    """Wait until the control mod answers on the main thread and the main menu is up."""
    deadline = time.monotonic() + timeout

    def left() -> float:
        return max(0.5, deadline - time.monotonic())

    if not wait_for(lambda: client.request("ping", timeout=3.0).get("ok"), left()):
        raise GdctlError("timeout", f"control mod did not answer within {timeout:.0f} s "
                                    "(is xucaleb6335.gd-agent-control installed and enabled?)")
    closed: list[str] = []
    while True:
        scene = client.request("scene", timeout=5.0)
        popups = scene.get("popups") or []
        if close_popups and popups:
            res = client.request("close_popups", timeout=5.0)
            closed += res.get("closed", [])
        elif scene.get("main_layer") == "MenuLayer" and not scene.get("transitioning"):
            # Startup popups can appear a moment after the menu: settle briefly.
            time.sleep(1.0)
            scene = client.request("scene", timeout=5.0)
            if close_popups and scene.get("popups"):
                continue
            scene["closed_popups"] = closed
            return scene
        if time.monotonic() > deadline:
            raise GdctlError("timeout", "main menu did not come up", scene=scene)
        time.sleep(0.5)


# --------------------------------------------------------------------------
# Launch / kill

def steam_running() -> bool:
    return bool(win32.processes("steam.exe"))


def launch(via: str = "auto", timeout: float = 120.0, close_popups: bool = True,
           wait_menu: bool = True) -> dict[str, Any]:
    """Start GD. via: "exe" (needs Steam already running, else GD quits at
    once), "steam" (steam://rungameid, starts Steam if needed; slow on a cold
    Steam start), "auto" (exe if Steam runs, else steam)."""
    pids = game_pids()
    if pids:
        out = {"already_running": True, "pid": pids[0]}
        if wait_menu:
            out["scene"] = _scene_brief(wait_for_menu(timeout, close_popups))
        return out
    t0 = time.monotonic()
    if via == "auto":
        via = "exe" if steam_running() else "steam"
    if via == "steam":
        os.startfile(f"steam://rungameid/{paths.STEAM_APP_ID}")  # noqa: S606 - documented launch path
    else:
        exe = paths.gd_exe()
        if not exe.exists():
            raise GdctlError("not_found", f"game exe not found: {exe}")
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([str(exe)], cwd=str(exe.parent), creationflags=flags, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # A cold Steam start (updates, login) can take a minute or more.
    pid = wait_for(lambda: (game_pids() or [None])[0], 150.0 if via == "steam" else 30.0)
    if pid and via == "exe":
        # Without Steam, GD exits within a couple of seconds.
        time.sleep(3.0)
        if not game_pids():
            pid = None
    if not pid:
        raise GdctlError("launch_failed", f"{paths.EXE_NAME} did not start or exited at once (via {via}); "
                                          "is Steam running and logged in? try --via steam")
    out: dict[str, Any] = {"already_running": False, "pid": pid, "via": via}
    if wait_menu:
        scene = wait_for_menu(max(10.0, timeout - (time.monotonic() - t0)), close_popups)
        out["scene"] = _scene_brief(scene)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def _scene_brief(scene: dict[str, Any]) -> dict[str, Any]:
    keys = ("main_layer", "in_level", "in_editor", "transitioning", "closed_popups", "level", "popups")
    return {k: scene.get(k) for k in keys if k in scene}


def kill(graceful_timeout: float = 30.0, force: bool = True) -> dict[str, Any]:
    pids = game_pids()
    if not pids:
        return {"was_running": False}
    out: dict[str, Any] = {"was_running": True, "pid": pids[0], "method": None}
    # 1) Ask the control mod to exit cleanly (saves game data and Geode state).
    try:
        client.request("quit_game", {"save": True}, timeout=5.0)
        if wait_for(lambda: not game_pids(), graceful_timeout):
            out["method"] = "quit_game"
            return out
    except ControlError as exc:
        out["quit_game_error"] = exc.message
    # 2) WM_CLOSE (GD may show a confirm dialog; it then stays up).
    for pid in game_pids():
        win = win32.main_window(pid, paths.WINDOW_TITLE)
        if win:
            win32.close_window(win["hwnd"])
    if wait_for(lambda: not game_pids(), 5.0):
        out["method"] = "wm_close"
        return out
    if not force:
        raise GdctlError("still_running", "game did not exit; rerun with force", **out)
    # 3) Hard kill: unsaved progress since the last save is lost.
    for pid in game_pids():
        out["taskkill"] = win32.taskkill(pid, force=True)[1]
    if not wait_for(lambda: not game_pids(), 10.0):
        raise GdctlError("still_running", "taskkill /F did not stop the game", **out)
    out["method"] = "taskkill"
    return out


def restart(via: str = "auto", timeout: float = 120.0) -> dict[str, Any]:
    killed = kill()
    time.sleep(1.0)
    launched = launch(via=via, timeout=timeout)
    return {"kill": killed, "launch": launched}


# --------------------------------------------------------------------------
# Mods

def mod_listing() -> dict[str, Any]:
    out: dict[str, Any] = {"source": "offline", "mods": mods.offline_listing()}
    if game_pids():
        try:
            ping = client.request("ping", timeout=5.0)
            if ping.get("ok"):
                live = {m["id"]: m for m in ping.get("mods", [])}
                for m in out["mods"]:
                    lm = live.get(m["id"])
                    if lm:
                        m["enabled"] = lm["enabled"]
                        m["loaded"] = lm["loaded"]
                        m["name"] = m["name"] or lm["name"]
                out["source"] = "control-mod"
        except ControlError:
            out["note"] = "game running but control mod unreachable: enabled flags are from saved.json"
    return out


def set_mods(changes: dict[str, bool], restart_game: bool = False) -> dict[str, Any]:
    installed = {m["id"] for m in mods.installed_mods()}
    missing = [m for m in changes if m not in installed]
    if missing:
        raise GdctlError("not_found", f"not installed: {', '.join(missing)}")
    if not game_pids():
        res = mods.set_flags(changes)
        return {"method": "saved.json", "applied": res, "restart_required": False}
    results = {}
    try:
        for mod_id, enabled in changes.items():
            results[mod_id] = ctl("set_mod_enabled", {"mod": mod_id, "enabled": enabled})
    except GdctlError as exc:
        raise GdctlError(exc.code, f"{exc.message} (close the game to edit saved.json offline)") from exc
    out: dict[str, Any] = {"method": "control-mod", "results": results,
                           "restart_required": any(r.get("restart_required") for r in results.values())}
    if restart_game and out["restart_required"]:
        out["restart"] = restart()
        out["restart_required"] = False
    return out


def ensure_training_mods() -> dict[str, Any]:
    """CBF and QOLMod disabled, bridge + control enabled, restarting if needed."""
    installed = {m["id"] for m in mods.installed_mods()}
    want = {m: False for m in paths.TRAINING_BLOCKLIST if m in installed}
    want.update({m: True for m in paths.TRAINING_REQUIRED})
    missing = [m for m in paths.TRAINING_REQUIRED if m not in installed]
    if missing:
        raise GdctlError("not_installed", f"required mods not installed: {missing} (python -m gdctl build)")

    def current() -> dict[str, dict]:
        """{id: {"enabled":, "loaded":}}; loaded is None when the game is closed."""
        if game_pids():
            ping = ctl("ping", timeout=10.0)
            return {m["id"]: {"enabled": m["enabled"], "loaded": m["loaded"]} for m in ping["mods"]}
        flags = mods.enabled_flags(mods.read_saved(), list(want))
        return {m: {"enabled": v, "loaded": None} for m, v in flags.items()}

    state = current()
    actions: list[str] = []
    wrong_flag = {m: v for m, v in want.items() if state.get(m, {}).get("enabled") != v}
    if wrong_flag:
        res = set_mods(wrong_flag)
        actions.append(f"set {wrong_flag} via {res['method']}")
        state = current()
    # Loaded state must match too (a flag change needs a restart).
    if game_pids() and any(state.get(m, {}).get("loaded") != v for m, v in want.items()):
        actions.append("restart to apply mod changes")
        restart()
        state = current()
    bad = {m: state.get(m) for m, v in want.items()
           if state.get(m, {}).get("enabled") != v or (game_pids() and state.get(m, {}).get("loaded") != v)}
    if bad:
        raise GdctlError("mods_wrong", f"mods still not in the training state: {bad}", actions=actions)
    return {"actions": actions, "mods": {m: state.get(m) for m in want}}


# --------------------------------------------------------------------------
# Build

ERROR_RE = re.compile(r"(error( [A-Z]+\d+)?:|: error |fatal error|CMake Error|\| Error \||error MSB)", re.I)


def build(targets: list[str], stop_first: bool = False, restart_game: bool | None = None,
          timeout: float = 1800.0) -> dict[str, Any]:
    """geode build for each target (mod / control-mod), installing into the game."""
    dirs = []
    for t in targets:
        if t not in paths.MOD_DIRS:
            raise GdctlError("bad_request", f"unknown build target {t!r}; use mod or control-mod")
        d = paths.MOD_DIRS[t]
        if d not in dirs:
            dirs.append(d)
    geode = shutil.which("geode")
    if not geode:
        raise GdctlError("not_found", "geode CLI not on PATH")
    was_running = bool(game_pids())
    out: dict[str, Any] = {"was_running": was_running, "results": []}
    if stop_first and was_running:
        out["kill"] = kill()
    log_dir = paths.gdctl_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for d in dirs:
        res = _geode_build(geode, d, log_dir, timeout)
        if not res["ok"] and res.get("locked") and game_pids():
            # The installed .geode is in use: stop the game and install again.
            out["kill"] = kill()
            res = _geode_build(geode, d, log_dir, timeout)
            res["retried_after_kill"] = True
        out["results"].append(res)
        ok_all = ok_all and res["ok"]
    out["ok"] = ok_all
    if restart_game is None:
        restart_game = was_running
    if ok_all and restart_game:
        out["relaunch"] = restart() if game_pids() else launch()
    elif not ok_all and restart_game and was_running and not game_pids():
        out["relaunch"] = launch()  # put the game back even though the build failed
    return out


def _geode_build(geode: str, mod_dir: Path, log_dir: Path, timeout: float) -> dict[str, Any]:
    t0 = time.monotonic()
    log_path = log_dir / f"build-{mod_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    try:
        proc = subprocess.run([geode, "build"], cwd=str(mod_dir), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
        text = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
        code = -1
    log_path.write_text(text, encoding="utf-8")
    errors = summarize_build_errors(text)
    installed = "Installed" in text and "Successfully packaged" in text
    res = {"target": mod_dir.name, "ok": code == 0 and installed, "exit_code": code,
           "installed": installed, "seconds": round(time.monotonic() - t0, 1), "log": str(log_path),
           "errors": errors}
    if not res["ok"] and re.search(r"used by another process|Access is denied|permission denied", text, re.I):
        res["locked"] = True
    return res


def summarize_build_errors(text: str, limit: int = 20) -> list[str]:
    """Unique compiler/CMake error lines, shortened, first ``limit``."""
    seen: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not ERROR_RE.search(line) or "warning" in line.lower() and "error" not in line.lower():
            continue
        line = re.sub(r"\s*\[[^\]]*\.vcxproj\]$", "", line)  # MSBuild project suffix
        if len(line) > 300:
            line = line[:300] + "..."
        if line not in seen:
            seen.append(line)
        if len(seen) >= limit:
            break
    return seen


# --------------------------------------------------------------------------
# Logs

def newest_log(crash: bool = False) -> Path | None:
    d = paths.geode_crashlogs_dir() if crash else paths.geode_logs_dir()
    files = sorted(d.glob("*.log" if not crash else "*"), key=lambda p: p.stat().st_mtime, reverse=True)
    files = [f for f in files if f.is_file()]
    return files[0] if files else None


def read_log(lines: int = 50, grep: str | None = None, crash: bool = False, file: str | None = None,
             ignore_case: bool = True) -> dict[str, Any]:
    path = Path(file) if file else newest_log(crash)
    if not path or not path.exists():
        raise GdctlError("not_found", "no log file found")
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(text)
    if grep:
        rx = re.compile(grep, re.I if ignore_case else 0)
        text = [ln for ln in text if rx.search(ln)]
    selected = text[-lines:] if lines > 0 else text
    return {"file": str(path), "total_lines": total, "matched": len(text), "lines": selected,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))}


# --------------------------------------------------------------------------
# Host screenshot (fallback when the mod does not answer)

def host_screenshot(path: Path, width: int = 960, method: str = "auto") -> dict[str, Any]:
    """Window capture without the mod. "screen" copies the desktop (accurate,
    needs GD visible on top); "print" uses PrintWindow (works when covered, but
    for GD's GL window it can return a stale frame); "auto" picks "screen" when
    GD is the foreground window."""
    from gdctl import png

    pids = game_pids()
    if not pids:
        raise GdctlError("not_running", "game is not running")
    win = win32.main_window(pids[0], paths.WINDOW_TITLE)
    if not win:
        raise GdctlError("no_window", "game has no visible window")
    if method == "auto":
        method = "screen" if win["foreground"] and not win["minimized"] else "print"
    w, h, bgra = win32.capture_client(win["hwnd"], method)
    rgb = png.bgra_to_rgb(w, h, bgra)
    ow, oh, rgb = png.downscale_rgb(w, h, rgb, width)
    png.write_png(path, ow, oh, rgb)
    black = rgb.count(0) > len(rgb) * 0.98
    out = {"path": str(path), "width": ow, "height": oh, "client_w": w, "client_h": h, "source": f"host-{method}"}
    if black:
        out["warning"] = "image is almost all black (minimized, or the GL surface was not captured)"
    elif method == "print":
        out["warning"] = "PrintWindow can return a stale frame for GD's GL window; prefer the mod or --method screen"
    return out


# --------------------------------------------------------------------------
# prepare-training

def prepare_training(spec: str, practice: bool = False, reopen: bool = False, probe: bool = True,
                     timeout: float = 180.0) -> dict[str, Any]:
    level = parse_level_spec(spec)
    steps: list[dict[str, Any]] = []
    if not game_pids():
        steps.append({"launch": launch(timeout=timeout)})
    else:
        # A busy main thread (saving, loading) can miss one check: give it 20 s.
        if not wait_for(lambda: status()["responsive"], 20.0, interval=2.0):
            steps.append({"restart_unresponsive": restart(timeout=timeout)})
    steps.append({"mods": ensure_training_mods()})
    wait_for_menu(timeout=timeout) if not _in_level() else None
    steps.append({"close_popups": ctl("close_popups").get("closed")})
    steps.append({"prevent_pause": ctl("prevent_pause", {"enabled": True})["prevent_pause"]})
    # Training wants the bridge's normal view at full speed: undo a watch
    # session (gdrl.tools.watch) that left watch-mode / legacy-view on.
    for key in ("watch-mode", "legacy-view"):
        try:
            ctl("set_mod_setting", {"mod": "xucaleb6335.gd-rl-bridge", "key": key, "value": False})
        except (ControlError, GdctlError) as e:
            steps.append({key: f"not reset: {e}"})

    scene = ctl("scene")
    cur = scene.get("level") or {}
    same = (scene.get("in_level") and cur.get("type") == level["type"]
            and all(cur.get(k) == level[k] for k in ("id", "name") if k in level)
            and bool(cur.get("practice")) == practice and not cur.get("paused"))
    if same and not reopen:
        steps.append({"open_level": "already open"})
    else:
        if scene.get("in_editor"):
            raise GdctlError("in_editor", "the editor is open; close it first (gdctl menu --save-editor)")
        params = dict(level, practice=practice)
        opened = ctl("open_level", params, timeout=60.0)
        steps.append({"open_level": {k: opened.get("level", {}).get(k) for k in ("id", "name", "type", "practice")}})
        scene = opened

    lvl = (ctl("scene").get("level")) or {}
    if lvl.get("paused"):
        raise GdctlError("paused", "level is paused", steps=steps)
    st = status(deep=False)
    bridge = st["ports"][str(paths.BRIDGE_PORT)]
    if not bridge["listening"] or not bridge["owned_by_game"]:
        raise GdctlError("bridge_down", f"bridge port {paths.BRIDGE_PORT} not listening in the game", steps=steps)
    out: dict[str, Any] = {"level": {k: lvl.get(k) for k in ("id", "name", "type", "practice", "paused")},
                           "steps": steps, "pid": st["pid"]}
    if probe:
        res = bridge_probe.probe(paths.BRIDGE_PORT)
        out["bridge_probe"] = res
        if not res.get("ok"):
            raise GdctlError("bridge_probe_failed", f"bridge did not accept HELLO/RESET: {res.get('error')}",
                             **out)
    out["ready"] = True
    return out


def _in_level() -> bool:
    try:
        return bool(client.request("scene", timeout=5.0).get("in_level"))
    except ControlError:
        return False
