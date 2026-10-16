# Progress (autopilot copy)

> This is the **autopilot** copy of the project (branch `autopilot`), made
> 2026-10-10. Every `LEARN` stub here was filled in by Claude to test the full
> pipeline overnight. The learning copy is `..\geometry-dash-rl`. See the
> bottom of the log for what changed beyond the stubs.

What's done and what's left. LEARNING.md explains each milestone; this file
tracks where I am. Tick a box when its test or gate passes, and add a dated
line to the log at the bottom.

The C++ chain (M0 to M3) and PPO (M6) are independent. Week 1 is done when
M3 and M6 pass their gates.

## Setup

- [x] Python venv, PyTorch (cu126), Geode SDK + CLI, toolchain (2026-10-09)
- [x] `pytest`: 73 passed, 31 skipped (learn stubs + POSIX-only bridge tests)
- [x] Fix the Bridge.cpp / PCH winsock clash in `mod/CMakeLists.txt`

## Part 1: the Geode mod (C++)

### M0: mod builds, loads, logs the level name (done 2026-10-09)

- [x] `geode build` succeeds and installs the mod
- [x] "GD-RL Bridge" shows in Geode's mod list
- [x] `RLPlayLayer::init` logs the level name
  (`log::info("Level Opened: {}", level->m_levelName)` in `mod/src/main.cpp`)
- [x] Confirmed in game: `Level Opened: Nine Circles` in the Geode log

### M1: per-tick hook and reading the game state (done 2026-10-10)

- [x] Disable the Click Between Frames and QOLMod mods (2026-10-10 log:
  "is not enabled" for both)
- [x] `mod/src/main.cpp` `processCommands`: count calls per second, confirm 240
  (2026-10-09: 23 of 37 one-second readings were 239 to 242; the low ones
  were frame drops, pauses and respawns). Note: the player does not move
  inside `processCommands`; movement happens elsewhere in the tick
- [x] `mod/src/StateReader.cpp` `readState`: all player fields filled
  - [x] position: `getPosition()` (same as `getRealPosition()`; `m_position`
    can lag a tick)
  - [x] on ground, upside down flags look right
  - [x] modes: Jawbreaker 2 wave = 4; Hexagon Force cube → ship → ball → cube;
    Electroman UFO = 3
  - [x] rotation: `getRotation()` wrapped to [0, 360), checked in game
  - [x] `xSpeed`: lookup table from the `m_playerSpeed` multiplier; reads
    311.6 at normal speed and x advances ~312 units/s, matching
  - [x] ground offset (subtract 90): a cube on flat ground reads y = 15, on
    blocks 45, 75, 105... (39 + 25 readings, 2026-10-10)
  - [x] `velY`: `m_yVelocity * 60` is already + = up when upside down (the
    flip was wrong: it gave the opposite sign of the measured value). Every
    mode except the wave moves at 0.9x that; the wave matches exactly
    (387.4 = 387.4). Now `* 60`, times 0.9 unless wave
- [x] `readState` objects: nearby hitboxes, kind, sorted by left edge, capped
  at 32. Spikes kind 2 (6x12 hitbox), blocks kind 1 (30x30)
- [x] Deleted `logStateForDebug` and the `ticks/s` log
- **Gate:** calls/s match 240; 1 Hz log matches the screen. Passed.

### M2: deaths, reset, button input (code written 2026-10-10)

- [x] `destroyPlayer`: ignore the anticheat spike GD fires at level start,
  count every other call as a death (`Controller::onDeath`, logs once per death)
- [x] RESET: `handleReset` only sets `m_resetRequested`; a new `update` hook
  (`Controller::onFrameStart`) does `resetLevel()` before the next frame and
  replies with the tick-0 STATE. Practice mode and start_pos != -1 are refused
- [x] Respawn delay and end screen skipped while a client is connected
  (`delayedResetLevel` and `showEndLayer` hooks); the pause GD does when its
  window loses focus is skipped too (`pauseGame` hook)
- [x] STEP: button via `handleButton(down, 1, true)` only when it changes;
  `onTick` counts the ticks down and replies; stops early on death/completion;
  `ticks = 0` peeks; terminal state repeats until RESET
- [x] `gdrl/tools/jump_rule.py`: hard-coded jump rule driver (passes on the
  mock server: completes mock level 1 at `--jump-distance 20`)
- [x] Verify: no false deaths. 2026-10-10 log: the anticheat spike fires ~13
  times per attempt (all ignored, player never dead); every counted death had
  the player really dead. GD sometimes calls destroyPlayer several times for
  one death, so `onDeath` now only counts the first
- [x] Verify: `jump_rule.py` connects and plays (button 1 = jump); repeat
  attempts die at the same tick (Stereo Madness easy 3990, rauchkammer 2 ~253)
- [ ] Verify: `jump_rule.py --attempts 500` runs unattended without a crash
- [ ] Verify: the jump rule beats an easy level (tune `--jump-distance`)
- **Gate:** no false deaths in a session; 500 unattended attempts without a
  crash; a hard-coded jump rule plays a level

### M3: lockstep (code written 2026-10-10, not yet verified in game)

- [x] Stage 1: under the agent, one update call = one tick (`runOneTick`:
  empty `m_extraDelta`, no `m_resumeTimer`, feed 1/240 s); the Controller
  warns in the log if a call ever runs != 1 physics tick
- [x] Stage 2: messages are only handled at frame level: a `CCScheduler`
  hook (`pollIdle`) for menus/handshake, the level's `update` hook
  (`onFrame`) under the agent. Never inside a tick
- [x] Stage 3: STEP runs exactly N one-tick updates right away, stopping at
  the tick that ends the episode; no STEP, no ticks. RESET resets directly
- [x] Stage 4: `onFrame` keeps serving STEPs for up to 10 ms per frame, then
  lets cocos draw
- [x] Stage 5: client lost or BYE → button released, level restarted if the
  attempt had ended, game back to normal. Quitting the level → later
  requests get NOT_IN_LEVEL. A STEP while the level is not updating gets an
  error instead of hanging
- [x] `bridge_check` now names the differing fields when determinism fails
  (passes on the mock server: 73x real time)
- [ ] Verify: the game freezes while Python is idle and runs fast while it steps
- [ ] Verify: no "a one-tick update ran N physics ticks" warnings in the log
- [x] Verify: `jump_rule.py` repeat attempts die at exactly the same tick
  (2026-10-10, Cycles: 19 attempts in a row died at tick 1314)
- [ ] Verify: `bridge_check --soak 200` passes all three checks
- **Gate:** `python -m gdrl.tools.bridge_check --port 5555 --soak 10000` passes
  all three checks

## Part 2: environment and brain (Python)

### M4: environment (done 2026-10-10)

- [x] Read `gdrl/env/gd_env.py`, `mock_game.py`, `mock_server.py`: terminated vs.
  truncated, same-step autoreset, `final_obs`

### M5: features and reward (done 2026-10-10, written by Claude on request)

- [x] `gdrl/features/encoder.py` `ObjectListEncoder.encode` → `tests/learn/test_encoder.py`
- [x] `gdrl/rewards/reward.py` `ProgressReward.__call__` → `tests/learn/test_reward.py`

### M6: PPO

- [x] `gdrl/agents/networks.py` `layer_init`, `mlp` → `tests/learn/test_networks.py`
- [x] `gdrl/agents/ppo.py` `compute_gae` → `tests/learn/test_gae.py`
  (work the docstring numbers by hand first)
- [x] `gdrl/agents/ppo.py` `PPOAgent.__init__`, `act`, `update` → `tests/learn/test_ppo.py`
- [x] CartPole seed 1 >= 475
- [x] CartPole seed 2 >= 475
- [x] CartPole seed 3 >= 475
- **Gate:** `python -m gdrl.train --config configs/cartpole_ppo.yaml --set seed=N`
  then `python -m gdrl.evaluate --run runs/<run>`, `return_mean >= 475` on 3 seeds

### Mock game

- [x] Train on the mock levels with `configs/mock_ppo.yaml` (seeds 1 to 5)
- [x] Beats the held-out mock seeds in `levels.yaml` (101 to 103)

## Part 3: the real game

### M7: train on GD

- [ ] Build 3 to 5 short, cube-only, unpublished levels in the editor
- [ ] Add their ids to `levels.yaml` (`game.train` / `game.heldout`)
- [ ] `python -m gdrl.train --config configs/gd_ppo.yaml` with the level open
- **Gate:** greedy policy beats level 1; 18/20 with `--sticky 0.1`

### M8: baselines and a grid observation

- [x] `gdrl/agents/baselines.py` `RuleBasedAgent`
- [x] `gdrl/agents/baselines.py` `GeneticTimingAgent`
- [x] `gdrl/features/encoder.py` `GridEncoder` → `tests/learn/test_encoder.py`
- [x] `gdrl/agents/networks.py` `GridCNN` → `tests/learn/test_networks.py`
- **Gate:** a results table comparing PPO with the baselines

## Log

- 2026-10-07: Project scaffolded (plumbing done, LEARN stubs in place).
- 2026-10-09: Dev environment installed; fixed the Bridge.cpp/PCH winsock
  clash in `mod/CMakeLists.txt`; first successful `geode build`.
- 2026-10-09: M0 done: mod loads in game and logs `Level Opened: <name>`.
- 2026-10-09: M1 part A done: `processCommands` runs ~240 times/s in steady
  play, so it is the per-tick hook. Frame drops lower the count (the game
  slows down rather than catching up); lockstep in M3 makes that irrelevant.
- 2026-10-10: M1 player fields, modes, rotation, xSpeed and objects checked
  in game; y offset and velY still to confirm. Starting M2.
- 2026-10-10: M2 implemented (deaths, reset, button, `jump_rule.py`); needs
  the in-game checks.
- 2026-10-10: M1 done (y offset and velY confirmed, velY sign/scale fixed,
  debug log removed). M2 deaths and button verified in game; 500-attempt
  run and beating a level still to do.
- 2026-10-10: M3 lockstep implemented (stages 1-5); needs the in-game checks
  and the 10000-episode soak.
- 2026-10-10: M3 in game: jump_rule deterministic on Cycles; a ~10,000-episode
  soak on Blast Processing ran 9 min (~1,240 episodes/min) with no warnings,
  stalls or crash. An earlier Cycles session had 10 "ran 0 physics ticks"
  warnings on its first attempt (death at tick 12786); that run was stopped
  by hand, not investigated. Watch for it if first episodes look odd.
- 2026-10-10: M4 read through (terminated vs. truncated, same-step autoreset,
  `final_obs`). M5 implemented by Claude: object-list encoder with one-hot
  kinds (84 features at k = 8) and progress-fraction reward; learn tests pass,
  and a rule reading only the observation beats all 8 mock levels.
- 2026-10-10 (autopilot): all stubs filled in by Claude. CartPole PPO: 500.0
  on seeds 1, 2, 3 (20 greedy episodes each). Mock PPO (2M steps, default
  config): seeds 1-3 complete all 5 training and all 3 held-out levels
  greedily; held-out with sticky 0.1: 0.83 / 0.90 / 0.97.
- 2026-10-10 (autopilot): the M5 encoder alone did NOT learn on the mock
  (plateau at ~20% progress, greedy policy = hold the button forever).
  Behaviour cloning showed why: the jump band is ~6 units wide, ~0.01 of the
  normalized dx, too fine for a 64x64 tanh MLP. Fix: a Gaussian distance code
  (RBF every 10 units over -30..180) for the 3 nearest objects. Also added
  x_speed and the game-mode one-hot to the player features.
- 2026-10-10 (autopilot): other additions: `train.resume` (continue from a
  checkpoint), `configs/gd_overnight.yaml`, `overnight.ps1` (auto-resume on
  crash; tested against the mock server with a killed connection),
  `gdrl/tools/genetic.py` (beats mock level 1 in 8 generations).
- 2026-10-10: the GPU currently fails to create a CUDA context ("busy or
  unavailable"), and torch's Adam.step() touches CUDA even for CPU tensors.
  Run training with CUDA_VISIBLE_DEVICES=-1 (overnight.ps1 does) until a
  reboot fixes the GPU.
- 2026-10-11 (autopilot): M8 mock comparison done (`results/m8_mock.md`).
  Object-list PPO: 100% train and held-out greedy, 91% held-out with sticky
  0.1 (3/3 seeds). Binary grid+CNN learns on 3/3 seeds but generalizes worse
  (89% held-out greedy, 48% sticky) and is 2-5x slower. Rule and genetic are
  brittle under sticky actions; random 0%. A coverage-fill GridEncoder variant
  was worse (1/3 seeds learned) and was reverted (patch in `results/`).
- 2026-10-11 (autopilot): `gdctl` + `control-mod/` let agents launch, build
  and drive GD; `overnight.ps1` recovers from GD crashes. CUDA fixed with the
  user env var CUDA_MODULE_LOADING=LAZY. Level beats are logged in
  `holistic_progress/` (`python -m gdrl.tools.record_beat`).
- 2026-10-12 (autopilot): bridge v2 installed; community StartPos copy (SM Start Pos, 141625101) curriculum via RESET start_pos; fixed StartPos respawn (cached checkpoint) and completion detection under lockstep; training on GPU (`configs/gd_sm_community.yaml`).
- 2026-10-12 (autopilot): **Stereo Madness beaten** by the greedy PPO policy on the original level (3/3; 5/10 with sticky 0.1), checkpoint `results/sm_beat_update_001170.pt`, after 4.8M steps / 2.4 h on the community StartPos curriculum (encoder focus=obstacles + terrain profile, CPU acting, horizon 2, stuck-start exploration).
