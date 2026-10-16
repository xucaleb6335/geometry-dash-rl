# GD Agent Control protocol, version 1

The `xucaleb6335.gd-agent-control` mod lets an agent drive Geometry Dash. It
is separate from the RL bridge (`mod/`, port 5555, binary protocol) and does
not touch its behaviour. Most agents should use the `gdctl` CLI
(`gdctl/README.md`) rather than speak this protocol directly.

## Transport

- TCP on `127.0.0.1:5556` (mod setting `port`). Several clients may be
  connected at once; each connection is served by its own thread.
- **JSON lines**: every request is one UTF-8 JSON object on one line ending in
  `\n`; every reply is one JSON object on one line. Requests on a connection
  are answered in order.
- Requests run on the game's **main thread** (queued with
  `queueInMainThread`, run on the next frame). The connection thread waits up
  to `timeout` seconds (default 30, clamped to 0.5 to 600) and then replies
  `{"ok": false, "code": "timeout"}`. A late result is dropped.
- `net_ping` is answered on the network thread, so it works even when the main
  thread is stuck (it is how `gdctl status` tells "hung" from "closed").

## Envelope

Request: `{"cmd": "<name>", "seq": <any>, "timeout": <seconds>, ...parameters}`.
`seq` is optional and echoed back. Parameters sit at the top level.

Reply: `{"ok": true, "cmd": "<name>", "seq": <echo>, ...results}`, or
`{"ok": false, "cmd": ..., "seq": ..., "code": "<code>", "error": "<text>"}`.

Error codes: `bad_json`, `bad_request`, `unknown_command`, `timeout`,
`not_found`, `ambiguous`, `transitioning`, `no_scene`, `not_visible`,
`disabled`, `not_in_level`, `in_editor`, `in_use`, `exists`, `refused`,
`not_downloaded`, `search_failed`, `create_failed`, `open_failed`,
`render_failed`, `save_failed`, `geode_error`, `exception`.

## Coordinates and node paths

- Points: cocos coordinates, origin bottom-left, window size `win_w` x
  `win_h` (569 x 320 on this laptop). `tree` and `click` use them.
- Fractions `fx, fy`: 0 to 1 of the window, origin **top-left**, the same
  for any screenshot size (`fx = px / image_width`).
- Node path: `/`-separated segments from the running scene (the scene itself
  is not in the path). A segment is the node's ID (from geode.node-ids) when
  that ID is unique among its siblings, else `Class[i]` (the i-th sibling of
  that class). Example: `MenuLayer/main-menu/play-button`.

## Commands

| cmd | parameters | result |
| --- | --- | --- |
| `net_ping` | | `thread: "net"`, `frame` |
| `ping` | | `protocol`, `mod_version`, `geode_version`, `gd_version`, `pid`, `port`, `frame`, `menu_reached`, `prevent_pause`, `mods: [{id, name, version, enabled, loaded, internal}]` (`enabled` = enabled after restart) |
| `scene` | | `scene`, `layers` (classes of the scene's children), `main_layer`, `transitioning`, `in_level`, `in_editor`, `level` (below), `popups: [{class, path, text[], buttons[{path, label, id, enabled}]}]`, `prevent_pause`, `frame` |
| `tree` | `root` (path), `depth` (12), `visible_only` (true), `clickable_only`, `match` (case-insensitive substring of class/id/text/label/path), `max_nodes` (3000) | `nodes: [{path, depth, class, id, tag, visible, x, y, w, h, text, label, clickable, toggled, children}]` (box in points), `win_w`, `win_h`, `count`, `truncated` |
| `click` | `path` or `node_id` (recursive ID search); or `x, y` (points); or `fx, fy`; `force`, `touch` | `method`: `activate` (CCMenuItem callback, like a real click) or `touch` (began this frame, ended next frame at the node's centre or the point); `node`; `warning` if a popup would block real input |
| `key` | `key` (`back`, `space`, `enter`, `escape`, `tab`, arrows, `a`-`z`, `0`-`9`, `f1`-`f12`), `action` (`tap`, `down`, `up`), `code` (raw enumKeyCodes), `frames` (tap hold, 2) | `handled` for `back` |
| `screenshot` | `path` (absolute PNG path, folders created), `width` (960; 0 = native) | `path`, `width`, `height`, `win_w`, `win_h`, `points_per_pixel` |
| `open_level` | `type`: `main` (`id`, 1 = Stereo Madness), `local` (`name`, `id` = m_id, or `index`), `saved` (`id`, must be downloaded); `practice` (false); `wait` (20 s) | the `scene` result once the level is open. Quits an open level first. Refuses in the editor |
| `exit_level` | `menu` (false), `wait` | `scene` result (GD returns to where the level came from, or the main menu with `menu`) |
| `menu` | `save` (needed when the editor is open: save and exit), `wait` | `scene` result at MenuLayer |
| `close_popups` | `wait` (3 s) | `closed` (classes), `remaining` (popups). FLAlertLayers and drop-downs get `keyBackClicked` (like ESC) |
| `list_levels` | `types` (`main,local,saved`) | `main: [{id, name, best_normal}]`, `local: [level + index]`, `saved: [level]` |
| `create_level` | `name` (1-20 chars), `level_string` (raw or `H4sI...`), `replace`, `save` (true: writes CCLocalLevels.dat), `audio_track`, `song_id` | `level`, `index` |
| `delete_level` | local `name` / `id` (m_id) / `index`; `prefix` (`gdrl-test-`), `force`, `save` | `deleted` |
| `search_levels` | `query` (name or id), `page`, `wait` (20) | `levels: [level + creator, difficulty, stars, demon, downloads, likes, length, objects, original]` |
| `download_level` | `id`, `song` (true: also download its custom song), `force`, `wait` (60) | `level`, `song_ready` |
| `download_song` | `id`, `wait` | `song_id`, `path` |
| `copy_to_local` | `id` (downloaded online level), `name`, `replace`, `strip_song`, `save` | `level` (the new local level: `m_id`), `source` |
| `get_level_string` | any level selector (as `open_level`) | `level`, `level_string` (as stored, usually gzip+base64) |
| `set_level_string` | local selector, `level_string` (raw or compressed), `prefix` (`gdrl-`), `force`, `save` | `level` |
| `set_mod_enabled` | `mod`, `enabled` | `enabled`, `loaded`, `restart_required` |
| `set_mod_setting` | `mod`, `key`, `value` (bool) | `mod`, `key`, `value`. Runtime change of a loaded mod's bool setting (saved on exit) |
| `quit_game` | `save` (true), `restart` (false) | replies, then exits (Geode's `utils::game::exit`) 5 frames later |
| `prevent_pause` | `enabled`, `persist` | `prevent_pause`, `setting`. Runtime value resets to the mod setting on restart unless `persist` |
| `get_option` | `key` (alias or 4-digit GameManager variable) | `value`, `options` (aliases, `fps_target`, `fps_now`, `vsync`), `aliases` |
| `set_option` | `key` (alias, `0xxx`, or `fps`), `value` | `options` |

A `level` object: `id` (online/main id, 0 for local), `name`, `type` (`main`,
`local`, `saved`), `m_id` (local id; GD renumbers these at every startup, so
select local levels by `name` across restarts), `has_string`, `string_length`,
`best_normal`, `best_practice`, `audio_track`, `song_id`; inside a level also
`attempt`, `percent`, `practice`, `paused`, `dead`, `completed`,
`level_length`, `player_x`, `player_y` (game units; ground at y = 90).

Option aliases: `show_percent` (0040), `auto_retry` (0026),
`auto_checkpoints` (0027), `show_cursor` (0024), `fast_menu` (0168). These
keys come from community documentation; verify before relying on one.

## Settings (mod.json)

`port` (5556), `prevent-pause` (false), `log-commands` (true: each command
except `ping`/`scene` is written to the Geode log as `control: <cmd>`).
