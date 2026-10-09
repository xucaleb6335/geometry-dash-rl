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

## Progress

Status as of 2026-10-09. Update this table when a milestone's gate passes.

| Milestone | What | Status |
| --- | --- | --- |
| Setup | Python env, PyTorch, Geode SDK, toolchain | Done. `pytest`: 73 passed, 31 skipped (learn stubs + POSIX-only bridge tests) |
| M0 | Mod builds, loads, logs level name | Mod builds and installs. **Next:** add the log line in `RLPlayLayer::init`, confirm in game |
| M1 | Per-tick hook at 240/s; `readState` | Not started |
| M2 | Real deaths, reset, button input | Not started |
| M3 | Lockstep; `bridge_check --soak 10000` | Not started |
| M5 | `ObjectListEncoder`, `ProgressReward` | Not started |
| M6 | `layer_init`, `mlp`, `compute_gae`, `PPOAgent`; CartPole gate >= 475 on 3 seeds | Not started (**next** on the Python side) |
| M7 | Own GD levels, train on the real game | Not started |
| M8 | Baselines, `GridEncoder`, `GridCNN` | Not started |

## Log

- 2026-10-07: Project scaffolded (plumbing done, LEARN stubs in place).
- 2026-10-09: Dev environment installed; fixed the Bridge.cpp/PCH winsock
  clash in `mod/CMakeLists.txt`; first successful `geode build`.
