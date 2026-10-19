# gdctl: drive Geometry Dash from the shell

For AI agents. Every command prints **one JSON object** to stdout and exits
`0` when `"ok": true`, `1` on failure (`code` and `error` say why) and `2` on
usage errors (also JSON). Run everything from the repo root with the
original project's interpreter:

```powershell
cd $HOME\Desktop\geometry-dash-rl-autopilot
$py = "..\geometry-dash-rl\.venv\Scripts\python.exe"
& $py -m gdctl <command> [options]          # --help on any command
```

Two halves:

- **Host side** (Python + Win32; works with the game closed or hung):
  `status launch kill restart build logs screenshot focus input mods
  prepare-training bridge-probe lock`.
- **Game side**: proxied to the control mod `xucaleb6335.gd-agent-control`
  (TCP 5556, JSON lines, `control-mod/PROTOCOL.md`).

## 1. The game lock (mandatory)

Only one agent may drive the game at a time. Before doing anything that
changes the game, take the lock; release it when done:

```powershell
$env:GDCTL_OWNER = "train-agent-1"            # or pass --owner on every call
& $py -m gdctl lock acquire --note "overnight training" --wait 600
...
& $py -m gdctl lock release
& $py -m gdctl lock status
```

```json
{"ok": true, "action": "acquired", "held": true, "owner": "train-agent-1", "pid": 1234,
 "host": "LAPTOP", "note": "overnight training", "acquired_at": "2026-10-11 01:02:03",
 "age_seconds": 0.0, "stale": false}
```

- The lock is `logs/game.lock`. Re-acquiring with the same owner refreshes it;
  do that at least hourly during long jobs. A lock not refreshed for **2 h**
  is stale and the next `acquire` takes it over (`"action": "took_over_stale"`).
- gdctl enforces it: while another owner holds a fresh lock, every command
  that changes the game fails with `"code": "locked"`. Read-only commands
  (`status logs screenshot ping scene tree list-levels level-info level-export
  search-levels wait-scene focus lock`) always work.
- `lock release --force` breaks someone else's lock. Don't, unless its owner
  is gone.
- Also look at `status.user_idle_seconds`: a few seconds means a person is
  using the PC (possibly playing). Don't take the game from them.

## 2. Mods and what `build` installs

`build` runs `geode build` in `mod/` (the RL bridge, `xucaleb6335.gd-rl-bridge`,
port 5555) and/or `control-mod/` (`xucaleb6335.gd-agent-control`, port 5556).
Each build **installs its .geode into the one shared game**
(`<GD>\geode\mods\`). The original project at `..\geometry-dash-rl` installs the
same bridge mod id, so the last build from either copy wins. Never change the
bridge's protocol from here.

Geode stores enabled flags as `"should-load-<mod id>": bool` in
`%LOCALAPPDATA%\GeometryDash\geode\mods\geode.loader\saved.json` (missing key =
enabled). Geode reads it at startup and rewrites it on exit, so gdctl edits
it only while the game is closed; while the game runs it uses Geode's API via
the control mod (takes effect after a restart). Click Between Frames
(`syzzi.click_between_frames`) and QOLMod (`thesillydoggo.qolmod`) must be off
for training; `prepare-training` enforces that. Note: the Geode log prints
"Loading mod <id>" even for disabled mods; `mods` (`loaded`) is the truth.

## 3. Commands

Outputs below are trimmed (`...`).

### Host side

**status**: process, window, ports, control-mod health, user idle time.
```json
{"ok": true, "running": true, "pid": 23608,
 "ports": {"5555": {"listening": true, "pid": 23608, "owned_by_game": true},
           "5556": {"listening": true, "pid": 23608, "owned_by_game": true}},
 "window": {"hwnd": 657182, "title": "Geometry Dash", "minimized": false, "hung": false,
            "foreground": true, "client_w": 1920, "client_h": 1080, "responsive": true},
 "control": {"net": true, "main_thread": true, "mod_version": "v0.1.0", "geode_version": "v5.11.1",
             "scene": {"main_layer": "MenuLayer", "in_level": false, "popups": 0}},
 "user_idle_seconds": 42.0, "responsive": true}
```
`responsive` needs both the window message loop and the mod's main thread.
`--shallow` skips the mod.

**launch** `[--via auto|exe|steam] [--timeout 120] [--keep-popups] [--no-wait]`:
starts GD, waits for the control mod and the main menu, closes startup popups.
`auto` runs the exe if Steam is running (about 5 to 40 s) and otherwise
`steam://rungameid/322170` (GD quits at once without Steam; a cold Steam start
takes a minute or two).
```json
{"ok": true, "already_running": false, "pid": 26192, "via": "exe",
 "scene": {"main_layer": "MenuLayer", "closed_popups": [], ...}, "seconds": 5.4}
```

**kill** `[--grace 30] [--no-force]`: `quit_game` through the mod (saves
data), then WM_CLOSE, then `taskkill /F` (loses unsaved progress).
```json
{"ok": true, "was_running": true, "pid": 23608, "method": "quit_game"}
```

**restart** `[--via ...]`: kill, then launch. Output `{"kill": {...}, "launch": {...}}`.

**build** `[mod|control-mod ...] [--stop-first] [--relaunch|--no-relaunch]`:
builds while the game runs, installs, and restarts the game if it was running
(new mods load only at startup). Retries after stopping the game if the
install hits a locked file. Full logs in `logs/gdctl/build-*.log`. An
incremental build takes 30 to 75 s, a clean one about 5 min.
```json
{"ok": true, "was_running": true,
 "results": [{"target": "control-mod", "ok": true, "exit_code": 0, "installed": true,
              "seconds": 32.0, "log": "...\\logs\\gdctl\\build-control-mod-....log", "errors": []}],
 "relaunch": {"kill": {"method": "quit_game"}, "launch": {"seconds": 5.4, ...}}}
```
On failure: `"code": "build_failed"` and `errors` holds the first 20 unique
compiler/CMake error lines.

**logs** `[-n 50] [-g REGEX] [--crash] [--file F]`: tail (and grep) the newest
Geode log, or the newest crash log.
```json
{"ok": true, "file": "...\\geode\\logs\\Geode 2026-10-11 00.34.08.log", "total_lines": 558,
 "matched": 3, "lines": ["00:39:46 INFO  [Main] [GD-RL Bridge]: Level Opened: Stereo Madness", ...]}
```

**screenshot** `[--path P] [--width 960] [--host] [--mod] [--method auto|screen|print]`:
PNG you can Read. Default path `logs/screens/shot-<time>.png`. Uses the mod's
render of the scene; falls back to a window capture if the mod does not
answer (`"source": "host-screen"`, plus `fallback_reason`). Host `print`
(PrintWindow) can return a **stale** frame for GD's GL window; `screen`
copies the desktop and needs GD on top.
```json
{"ok": true, "path": "...\\logs\\screens\\menu.png", "width": 960, "height": 540,
 "win_w": 569, "win_h": 320, "points_per_pixel": 0.593, "source": "mod"}
```

**focus**: bring GD to the foreground. `{"ok": true, "hwnd": 657182, "foreground": true}`

**input** `click (--fx F --fy F | --px X --py Y)` / `input key NAME`: real OS
input (SendInput) after focusing GD. Last resort when the mod is down.
`{"ok": true, "client": [960, 540], "screen": [960, 540], "hwnd": 657182}`

**mods** `[list]` / `mods enable|disable ID... [--restart]`:
```json
{"ok": true, "source": "control-mod", "mods": [
  {"id": "syzzi.click_between_frames", "name": "Click Between Frames", "version": "v1.5.0",
   "enabled": false, "loaded": false}, ...]}
{"ok": true, "method": "control-mod", "results": {"syzzi.click_between_frames":
  {"enabled": true, "loaded": false, "restart_required": true}}, "restart_required": true}
{"ok": true, "method": "saved.json", "applied": {"before": {...}, "after": {...},
  "backup": "...\\saved.json.gdctl-20261011-004348.bak"}, "restart_required": false}
```

**prepare-training** `--level SPEC [--practice] [--reopen] [--no-probe]`:
game up (launch, or restart if unresponsive for 20 s), CBF + QOLMod off and
both of our mods on (restarting if needed), popups closed, `prevent_pause` on,
level open in normal mode, bridge port owned by GD, the bridge's watch-only
settings off (watch-mode, legacy-view, show-hitboxes), then a HELLO/RESET/BYE
probe on 5555. Then start the RL client.
```json
{"ok": true, "level": {"id": 1, "name": "Stereo Madness", "type": "main", "practice": false, "paused": false},
 "steps": [{"mods": {"actions": [], ...}}, {"close_popups": []}, {"prevent_pause": true},
           {"open_level": {"id": 1, "name": "Stereo Madness", "type": "main", "practice": false}}],
 "bridge_probe": {"ok": true, "protocol": 1, "tick_rate": 240,
                  "state": {"tick": 0, "player_x": 0.0, "player_y": 15.0, "mode": 0, ...}},
 "ready": true}
```

**bridge-probe** `[--no-reset]`: HELLO (+ RESET) + BYE on 5555 (resets the
level). `{"ok": true, "protocol": 1, "state": {"tick": 0, ...}}` or
`{"ok": false, "stage": "reset", "error": {"name": "NOT_IN_LEVEL", ...}}`.

**lock** `acquire|release|status`: see section 1.

### Game side (control mod)

**Level specs** (used by `open-level`, `level-*`, `startpos`, `prepare-training`):
`main:<id>` or just `<id>` (1 = Stereo Madness ... 22, 5001-5004, 3001),
`local:<name>`, `local-id:<m_id>`, `local-index:<i>` (0 = top of "My Levels"),
`saved:<online id>` (downloaded online level).
**GD renumbers local `m_id`s at every startup** (and creating a level shifts
indexes), so name training levels uniquely (`gdrl-<name>`) and refer to them
by `local:<name>`; use `local-id`/`local-index` only within one session.

**ping**: versions and the live mod list (`enabled` after restart, `loaded` now).

**scene**:
```json
{"ok": true, "scene": "cocos2d::CCScene", "transitioning": false, "layers": ["PlayLayer"],
 "main_layer": "PlayLayer", "in_level": true, "in_editor": false,
 "level": {"id": 1, "name": "Stereo Madness", "type": "main", "m_id": 0, "attempt": 1,
           "percent": 0, "practice": false, "paused": false, "dead": false, "completed": false,
           "level_length": 26724, "player_x": 0, "player_y": 105, ...},
 "popups": [], "prevent_pause": true}
```
Popups list `class`, `text` and their `buttons` with paths (click them).

**tree** `[--root PATH] [--depth 12] [--all] [--clickable] [--match S] [--max-nodes N]`:
```json
{"ok": true, "win_w": 569, "win_h": 320, "count": 17, "nodes": [
  {"path": "MenuLayer/main-menu/play-button", "depth": 3, "class": "CCMenuItemSpriteExtra",
   "id": "play-button", "visible": true, "x": 229.8, "y": 115.3, "w": 109.5, "h": 109.5,
   "label": "", "clickable": true, "children": 1}, ...]}
```

**click** `--path P | --id NODE_ID | --x X --y Y | --fx F --fy F [--force] [--touch]`:
menu items are activated (their callback runs, like a click); other nodes and
points get a simulated touch.
`{"ok": true, "method": "activate", "node": {"path": "MenuLayer/main-menu/play-button", ...}}`

**key** `NAME [--action tap|down|up]`: `back` is GD's ESC/back button.
`{"ok": true, "key": "back", "action": "tap", "handled": true}`

**open-level** `SPEC [--practice]`: quits a running level first; replies once
the level runs, with the `scene` result. Missing custom songs don't block it
(the level plays silent).

**exit-level** `[--menu]`, **menu** `[--save-editor]`, **close-popups**:
reply with the resulting scene / `{"closed": ["OptionsLayer"], "remaining": []}`.

**list-levels** `[--types main,local,saved]`:
`{"main": [{"id": 1, "name": "Stereo Madness", "best_normal": 18}, ...], "local": [{"index": 0, "name": "...", "m_id": 138, ...}], "saved": [...]}`

**create-level** `--name N (--string S | --file F | --test) [--replace] [--no-save] [--song-id N]`:
raw strings are compressed for you (`gdctl/levelstring.py` has a builder).
`{"ok": true, "level": {"name": "gdrl-test-flat", "type": "local", "m_id": 223, ...}, "index": 0}`

**delete-level** `local:<name>|local-id:<m_id> [--force]`: only `gdrl-test-*`
without `--force`.

**search-levels** `QUERY [--page N]`: online search by name or id.
```json
{"ok": true, "levels": [{"id": 146399247, "name": "rauchkammer", "creator": "Devilmine",
  "difficulty": "extreme demon", "stars": 10, "song_id": 433359, "downloads": 2365584, "objects": ...}, ...]}
```
Known ids: Bloodbath 10565740 (Riot), Cataclysm 3979721 (Ggb0y),
Rauchkammer 146399247 (Devilmine).

**download-level** `ID [--no-song] [--force]`: downloads the level (it becomes
a saved level) and its custom song.
`{"ok": true, "level": {"id": 10565740, "name": "Bloodbath", "string_length": 157404, "objects": 24746, ...}, "song_ready": true}`

**download-song** `ID`: `{"ok": true, "song_id": 467339, "path": "..."}`

**copy-to-local** `ID --name gdrl-<name> [--replace] [--strip-song]`: GD's
"clone level": a local, editable copy. Returns its `m_id`.
`{"ok": true, "level": {"name": "gdrl-bloodbath", "type": "local", "m_id": 222, ...}, "source": {...}}`

**level-export** `SPEC [--path F] [--compressed]` / **level-import** `SPEC --path F [--force]`:
the raw level string to/from a file (default `logs/levels/<name>-<id>.txt`).
Import (and every write) only touches local levels named `gdrl-*` unless `--force`.

**level-info** `SPEC`:
```json
{"ok": true, "objects": 24746, "last_object_x": 39029.2, "last_object_block": 1301.0,
 "start": {"mode_name": "ship", "speed_name": "1x", "flipped": false, "mini": false, "dual": false},
 "startpos_count": 0, "portals": 175, "startpos": []}
```

**startpos** `SPEC list|add|remove|set [--x X] [--y 15] [--mode M] [--speed S] [--gravity normal|flipped] [--mini] [--dual] [--no-infer] [--all]`:
editor units (30 per block, ground y = 0, cube on the ground y = 15; in game
and in the bridge everything is 90 higher). `set` removes every StartPos and
adds one, so the level starts there (in normal mode GD then shows "Testmode"
and saves no progress; the bridge's RESET respawns there). Settings are
inferred from the portals before x unless given. The inference is approximate:
it assumes every portal left of x is taken, so check `added` and override
when a level has alternate paths.
```json
{"ok": true, "added": {"x": 19500.0, "y": 105.0, "mode_name": "ship", "speed_name": "3x", "mini": true, ...},
 "removed": 0, "startpos": [{"x": 19500.0, ...}]}
```

**quit** `[--no-save]`, **prevent-pause** `[on|off] [--persist]` (the runtime
value resets on restart unless `--persist`), **option get [KEY]** /
**option set KEY VALUE** (`show_percent`, `auto_retry`, `auto_checkpoints`,
`show_cursor`, `fast_menu`, any `0xxx` variable, `fps`),
**wait-scene** `[--layer L] [--in-level] [--no-popups] [--timeout 30]`,
**raw** `'{"cmd": "...", ...}'` (any protocol command).

## 4. Workflows

**Rebuild the bridge mod and relaunch**
```powershell
& $py -m gdctl lock acquire --wait 600
& $py -m gdctl build mod                 # restarts GD if it was running
& $py -m gdctl logs -g "bridge: listening"
& $py -m gdctl lock release
```

**Prepare for training on level X**
```powershell
& $py -m gdctl lock acquire --note training
& $py -m gdctl prepare-training --level main:1        # or local:gdrl-bloodbath
& $py -m gdrl.train --config configs/gd_overnight.yaml
```

**Recover from a crash** (e.g. in a training restart loop)
```powershell
& $py -m gdctl status                  # running? responsive?
& $py -m gdctl logs --crash -n 80      # what happened
& $py -m gdctl prepare-training --level local:gdrl-bloodbath   # relaunches if needed
```
`prepare-training` launches a closed game and restarts one that stays
unresponsive for 20 s.

**Train from mid-level on an online level**
```powershell
& $py -m gdctl search-levels Bloodbath
& $py -m gdctl download-level 10565740
& $py -m gdctl copy-to-local 10565740 --name gdrl-bloodbath --replace
& $py -m gdctl level-info local:gdrl-bloodbath
& $py -m gdctl startpos local:gdrl-bloodbath set --x 19500 --y 105
& $py -m gdctl prepare-training --level local:gdrl-bloodbath --reopen
```

**Create a custom level from a string**
```powershell
& $py -m gdctl create-level --name gdrl-test-a --file my_level.txt --replace
& $py -m gdctl open-level local:gdrl-test-a
```
Python: `from gdctl import levelstring as ls; raw = ls.build([ls.obj(ls.SPIKE, 315, 15)])`.

**Find and click a button via tree**
```powershell
& $py -m gdctl tree --clickable --match settings     # copy a path from "nodes"
& $py -m gdctl click --path MenuLayer/bottom-menu/settings-button
& $py -m gdctl screenshot                            # then Read the PNG
& $py -m gdctl close-popups
```
If a node has no ID, its path uses `Class[i]`. Buttons inside popups are
listed in `scene.popups[].buttons`.

## 5. Notes and limits

- The game must have a window that is not minimized for main-thread commands
  to run quickly; `focus` restores it.
- Opening the editor or editor playtest is not supported (the bridge hooks
  only PlayLayer anyway). `menu --save-editor` exits an editor someone opened.
- `launch --via exe` needs Steam running and logged in.
- The person using this laptop may be playing GD. Check `user_idle_seconds`.
- Back up before bulk edits: `create-level`, `copy-to-local`,
  `set_level_string` and `delete-level` write `CCLocalLevels.dat` right away.
