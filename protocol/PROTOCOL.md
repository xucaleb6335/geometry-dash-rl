# GD-RL Bridge Protocol, version 1

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
| 4 | u16 | version | `1` |
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
| 4 | u16 | max_objects (length of the STATE object array, 32) |
| 6 | u16 | reserved |

### RESET (client → server)

| Offset | Type | Field |
| --- | --- | --- |
| 0 | i32 | level_id. `0` = restart the currently loaded level. Other values are reserved for loading local levels by ID (curriculum, M7+). |
| 4 | i32 | start_pos. `-1` = level start. `k >= 0` = start from the k-th Start Pos object, ordered by x. |

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

### STATE (server → client), 676 bytes

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
| 0 | DEAD | player 1 died this episode (terminal) |
| 1 | COMPLETED | level completed (terminal) |
| 2 | ON_GROUND | standing on the ground or on top of a solid |
| 3 | UPSIDE_DOWN | gravity flipped |
| 4 | HOLDING | the jump button is currently held |

Object (20 bytes):

| Offset | Type | Field |
| --- | --- | --- |
| 0 | f32 | x (hitbox centre) |
| 4 | f32 | y (hitbox centre) |
| 8 | f32 | w (hitbox width) |
| 12 | f32 | h (hitbox height) |
| 16 | u8 | kind |
| 17 | u8[3] | reserved |

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
