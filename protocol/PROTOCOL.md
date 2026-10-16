# GD-RL Bridge Protocol, versions 1 and 2

This file is the contract between the Geode mod (C++, the server) and the Python
environment (the client). Three things implement it and must stay in sync:

| Implementation | File |
| --- | --- |
| Geode mod (server) | `mod/src/Protocol.hpp`, `mod/src/Bridge.cpp` |
| Python client | `gdrl/env/protocol.py`, `gdrl/env/client.py` |
| Mock server (fake game) | `gdrl/env/mock_server.py` |

If you change a struct here, bump `PROTOCOL_VERSION` and update all three. The
HELLO handshake rejects mismatched versions, so drift fails loudly instead of
silently decoding garbage.

**Version 2** (2026-10-11) adds snapshots (save and restore the whole game
state), RESET from a Start Pos, a bigger STATE (player 2, hitbox size,
floor/ceiling, 64 objects) and object subtypes and flags. Everything in
version 1 is unchanged. A server speaks both and answers in the version the
client's HELLO asked for; the Python client asks for 2 and, if the server
replies `VERSION_MISMATCH`, reconnects and speaks 1. Status: the Python client
and the mock server speak 2; the mod's v2 code is the patch in
`logs/mod-patch/` (DEMON_PLAN.md), applied once it is tested in game.

## Transport

- TCP, bound to `127.0.0.1` only. Default port `5555` (a mod setting).
- One client at a time. The client always speaks first; the server only replies.
- Both ends set `TCP_NODELAY`. Without it, Nagle's algorithm can delay each
  small request/reply pair.
- All integers and floats are **little-endian**. Structs are packed (no
  implicit padding); every padding byte is an explicit `reserved` field and is
  sent as zero.

## Framing

Every message is a 12-byte header followed by `length` payload bytes.

| Offset | Type | Field | Value |
| --- | --- | --- | --- |
| 0 | u32 | magic | `0x4C524447` (the bytes `G D R L`) |
| 4 | u16 | version | the agreed version (`1` or `2`; the HELLO carries the version asked for) |
| 6 | u16 | type | see message types |
| 8 | u32 | length | payload size in bytes |

A receiver that sees a bad magic, an unknown type or a payload length that does
not match the type's fixed size must reply `ERROR(BAD_MESSAGE)` (server) or
raise (client) and close the connection.

## Message types

| Type | Name | Direction | Payload size |
| --- | --- | --- | --- |
| 1 | HELLO | client → server | 4 |
| 2 | HELLO_ACK | server → client | 8 |
| 3 | RESET | client → server | 8 |
| 4 | STEP | client → server | 4 |
| 5 | STATE | server → client | 676 |
| 6 | ERROR | server → client | 68 |
| 7 | BYE | client → server | 0 |
| 8 | SAVE_STATE (v2) | client → server | 8 |
| 9 | LOAD_STATE (v2) | client → server | 8 |
| 10 | DROP_STATE (v2) | client → server | 8 |

In version 2, STATE is 1344 bytes (STATE v2 below). A receiver validates
sizes against the version in the frame header.

### HELLO (client → server)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u16 | client_version (must equal the server's version) |
| 2 | u16 | reserved |

Reply: `HELLO_ACK`, or `ERROR(VERSION_MISMATCH)` followed by closing the socket.

### HELLO_ACK (server → client)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u16 | protocol_version |
| 2 | u16 | tick_rate (physics ticks per second, 240 by default) |
| 4 | u16 | max_objects (length of the STATE object array: 32 in v1, 64 in v2) |
| 6 | u16 | v1: reserved (0). v2: max_snapshots (snapshot slots are `0 .. max_snapshots - 1`; 0 = none) |

### RESET (client → server)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | i32 | level_id. `0` = restart the currently loaded level. Other values are reserved for loading local levels by ID (curriculum, M7+). |
| 4 | i32 | start_pos. `-1` = where the game itself starts this level (the level start, or the level's active Start Pos if it has one: v1 behaviour). `-2` = the real level start, ignoring Start Pos objects (v2). `k >= 0` = start from the k-th Start Pos object, ordered by x (v2; a v1 mod replies UNSUPPORTED). A Start Pos does not replay what came before it (triggers, moving objects), so for exact mid-level starts use snapshots. |

Reply: `STATE` with `tick = 0`. The respawn delay and any end screen are skipped.

### STEP (client → server)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u8 | button (`0` released, `1` held) for player 1's jump |
| 1 | u8 | reserved |
| 2 | u16 | ticks (how many physics ticks to advance, `>= 0`) |

The server holds the button state for the whole step, advances exactly `ticks`
physics ticks at a fixed `1 / tick_rate` seconds each, then replies `STATE`.

- If the player dies or completes the level partway through, the server stops
  at that tick and replies immediately. `STATE.tick` tells you where it stopped.
- `ticks = 0` returns the current state without advancing (a peek).
- After a terminal state (DEAD or COMPLETED), further STEPs return the same
  terminal STATE unchanged. The client must send RESET.

### SAVE_STATE, LOAD_STATE, DROP_STATE (client → server, v2)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | i32 | slot (`0 .. max_snapshots - 1`) |
| 4 | u32 | reserved |

- `SAVE_STATE` stores the complete game state in `slot` (overwriting it): in
  the mod, GD's practice checkpoint (`PlayLayer::createCheckpoint`: game
  state, triggers, object states) plus player fields the checkpoint does not
  keep, the tick counter and the button state. Reply: `STATE` (unchanged).
  Saving a terminal state is an error (`UNSUPPORTED`).
- `LOAD_STATE` restores it. Reply: `STATE` of the restored state, whose
  `tick` is the tick it had when saved. The button is in the state it was in
  when saved, without a press or release event, so continuing with the same
  STEPs reproduces the same states as a continuous run. Empty or
  out-of-range slot: `ERROR(NO_SNAPSHOT)`.
- `DROP_STATE` frees the slot. Reply: `STATE` (unchanged).
- Leaving the level clears every slot.

The determinism requirement extends to snapshots: SAVE at tick t, a sequence
of STEPs, LOAD, the same STEPs must give byte-identical STATEs, equal to the
ones a run from RESET gives (checked by `python -m gdrl.tools.snapshot_check`).

### STATE (server → client), 676 bytes (v1)

| Offset | Type | Field | Notes |
| --- | --- | --- | --- |
| 0 | u32 | tick | physics ticks since the last RESET |
| 4 | u32 | flags | bit field, below |
| 8 | f32 | player_x | centre of player 1's hitbox, game units (30 units = 1 block) |
| 12 | f32 | player_y | centre of the hitbox; the ground surface is y = 0 |
| 16 | f32 | vel_y | units per second, positive = up |
| 20 | f32 | rotation | degrees, cosmetic |
| 24 | f32 | x_speed | horizontal speed, units per second |
| 28 | f32 | level_length | x of the end of the level |
| 32 | u8 | mode | player mode, below |
| 33 | u8 | reserved | |
| 34 | u16 | n_objects | how many entries of `objects` are valid (0 to 32) |
| 36 | Object[32] | objects | 20 bytes each; unused entries are zero |

Flags:

| Bit | Name | Meaning |
| --- | --- | --- |
| 0 | DEAD | player 1 (or player 2 in dual mode) died this episode (terminal) |
| 1 | COMPLETED | level completed (terminal) |
| 2 | ON_GROUND | standing on the ground or on top of a solid |
| 3 | UPSIDE_DOWN | gravity flipped |
| 4 | HOLDING | the jump button is currently held |
| 5 | MINI | (v2) player 1 is mini |
| 6 | DUAL | (v2) dual mode; the player 2 fields are valid |

Object (20 bytes):

| Offset | Type | Field |
| --- | --- | --- |
| 0 | f32 | x (hitbox centre) |
| 4 | f32 | y (hitbox centre) |
| 8 | f32 | w (hitbox width) |
| 12 | f32 | h (hitbox height) |
| 16 | u8 | kind |
| 17 | u8 | subtype (v2; 0 in v1), below |
| 18 | u8 | object flags (v2; 0 in v1): bit 0 MOVED (not at its spawn position), bit 1 ACTIVATED (orb/pad/portal used this attempt), bit 2 HIDDEN (invisible but collides) |
| 19 | u8 | reserved |

Which objects are sent: those whose hitbox right edge is at or past the
player's left edge minus one block (so the object being passed is still
visible), sorted by hitbox left edge, nearest first, capped at 32. Decorations
and triggers are never sent.

Object kinds:

| Value | Kind |
| --- | --- |
| 0 | NONE (unused slot) |
| 1 | SOLID (blocks, slabs: land on top, die on the side) |
| 2 | HAZARD (spikes, saws: die on contact) |
| 3 | ORB (jump rings) |
| 4 | PAD (jump pads) |
| 5 | PORTAL (mode, gravity, speed portals) |
| 255 | OTHER |

Player modes: 0 cube, 1 ship, 2 ball, 3 ufo, 4 wave, 5 robot, 6 spider, 7 swing.
Version 1 of the agent only handles cube.

Object subtypes (v2): 0 unknown, 1 block, 2 slope, 3 spike, 4 saw;
orbs 10 yellow, 11 pink, 12 gravity (blue), 13 green, 14 drop (black), 15 red,
16 dash, 17 gravity dash, 18 spider, 19 teleport, 20 custom; pads 30 yellow,
31 pink, 32 gravity, 33 red, 34 spider; portals 40 cube, 41 ship, 42 ball,
43 ufo, 44 wave, 45 robot, 46 spider, 47 swing, 50 gravity down, 51 gravity up,
52 gravity toggle, 53 mini, 54 normal size, 55 dual, 56 solo, 57 mirror,
58 mirror off, 59 teleport; speed portals (kind PORTAL) 60 slow, 61 normal,
62 2x, 63 3x, 64 4x. In v2 speed portals are sent (v1 dropped them).

### STATE v2 (server → client), 1344 bytes

Offsets 0 to 35 are as in v1. Then:

| Offset | Type | Field | Notes |
| --- | --- | --- | --- |
| 36 | f32 | player_size | side of player 1's hitbox (smaller when mini) |
| 40 | f32 | floor_y | lowest y the player can stand on here; NaN = not reported |
| 44 | f32 | ceil_y | roof of a ship/ball/wave corridor; NaN = not reported |
| 48 | f32 | p2_x | player 2 (valid when DUAL) |
| 52 | f32 | p2_y | |
| 56 | f32 | p2_vel_y | |
| 60 | u8 | p2_mode | |
| 61 | u8 | p2_flags | StateFlag bits DEAD, ON_GROUND, UPSIDE_DOWN, MINI |
| 62 | u16 | reserved | |
| 64 | Object[64] | objects | as in v1, 64 entries |

### ERROR (server → client)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | u16 | code |
| 2 | u16 | reserved |
| 4 | char[64] | message, UTF-8, NUL-padded |

| Code | Name | When |
| --- | --- | --- |
| 1 | VERSION_MISMATCH | HELLO version differs |
| 2 | BAD_MESSAGE | bad magic, unknown type, wrong payload length, message out of order |
| 3 | NOT_IN_LEVEL | RESET or STEP while no level is open |
| 4 | UNSUPPORTED | a valid request the server does not implement yet (e.g. level_id != 0) |
| 5 | NO_SNAPSHOT | (v2) LOAD_STATE of an empty slot, or a slot out of range |

### BYE (client → server)

No payload, no reply. The server closes the connection and waits for the next client.

## Session

```
client                         server
  | HELLO(version=1)  ------->   |
  |   <-------  HELLO_ACK(240, 32)
  | RESET(0, -1)      ------->   |   restart level, skip respawn delay
  |   <-------  STATE(tick=0)    |
  | STEP(button, k)   ------->   |   run exactly k ticks with button held/released
  |   <-------  STATE(tick=k)    |
  | ...                          |
  | STEP(...)         ------->   |   player dies partway through
  |   <-------  STATE(DEAD)      |
  | RESET(0, -1)      ------->   |
  |   ...                        |
  | BYE               ------->   |
```

## Determinism requirement

The same RESET followed by the same sequence of STEPs must produce
byte-identical STATE replies. The M3 gate tests this. Levels with random
triggers break it and are excluded from training.
