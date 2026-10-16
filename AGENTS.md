# AGENTS.md

Notes for AI agents (and future me) working in this repo. Read README.md for
the architecture and LEARNING.md for the milestone map first.

## Ground rules

- This is a learning project. `LEARN(Mx)` stubs are for the owner to write.
  Agents explain, review, debug and check numbers by hand; they do not fill in
  stubs unless explicitly asked.
- Don't edit plumbing to make a learn test pass (see LEARNING.md).
- Protocol changes go in three places together: `protocol/PROTOCOL.md`,
  `mod/src/Protocol.hpp`, `gdrl/env/protocol.py`, plus a version bump.

## Environment (Windows laptop, set up 2026-10-09)

| Component | Version / location |
| --- | --- |
| Python | 3.14.2, venv at `.venv` (`.venv\Scripts\activate`) |
| PyTorch | 2.14.1+cu126, CUDA works on the RTX 4060 Laptop GPU |
| NVIDIA driver | 536.52: too old for cu130 wheels, so cu126 is used instead of the README's cu130 |
| Geometry Dash | 2.2081 (Steam), `<Steam library>\steamapps\common\Geometry Dash` |
| Geode (in game) | v5.10.1 (latest stable as of 2026-10-09) |
| Geode CLI | 3.9.0 (winget), profile `steam` points at the GD exe |
| Geode SDK | 5.10.1 + binaries at `%USERPROFILE%\Documents\Geode` (`GEODE_SDK` user env var) |
| Toolchain | CMake 4.4.4, LLVM/Clang 23.1.3 (LLVM's `bin` folder added to user PATH), Ninja 1.13.2, VS Build Tools 2022 (17.14) with Windows SDK 10.0.26100 |

Gotchas:

- `geode build` (from `mod/`) builds and installs the `.geode` into the game.
  Delete `mod/build` if the CMake generator or toolchain changes.
- `mod/CMakeLists.txt` builds `src/Bridge.cpp` with `SKIP_PRECOMPILE_HEADERS`:
  Geode's forced PCH pulls in `windows.h`/`winsock.h` before `winsock2.h`,
  which breaks `ws2tcpip.h`. Keep Bridge Geode-free.
- Disable the Click Between Frames and QOLMod mods while training; they alter
  input/physics timing and will break the lockstep tick loop.
- `tests/test_cpp_bridge.py` skips on Windows (needs a POSIX toolchain).
- Lockstep (M3): while a Python client is connected and a level is open, the
  level only moves when STEPs arrive, so it looks frozen when Python is idle.
  GD's respawn delay, end screen and focus-loss pause are skipped then. Don't
  open the pause menu under the agent: RESET/STEP get an error until it closes.

## Progress

Tracked in `PROGRESS.md`: a checklist per milestone plus a dated log. Tick
items and add a log line there when a test or gate passes. Once a milestone
is done, remove its `LEARN(Mx)` comments from the code (the checklist keeps
the record).
