# AUTOPILOT: read this first

This folder (`Desktop\geometry-dash-rl-autopilot`, git branch `autopilot`) is
a **copy** of the GD-RL project in which Claude filled in every `LEARN` stub,
so the whole pipeline could be tested end to end (and trained overnight).

**Hard rule: never edit, build from, or run tools against the original
project at `Desktop\geometry-dash-rl`.** It is the owner's learning copy;
its M6 to M8 stubs are theirs to write. The only thing borrowed from it is
its Python venv (see "Environment" below). Work only in this folder.

Then read AGENTS.md (ground rules, environment, gotchas), PROGRESS.md (the
checklist; the bottom of the log has the autopilot entries) and README.md.

## Status (2026-10-10)

The code is written up to M8. The gates for M7 and M8 are not done.

| Milestone | Code | Gate / verification |
|---|---|---|
| M0-M2 | done | M2 has 2 open checks: `jump_rule --attempts 500`, and the jump rule beating a level |
| M3 | done | the 10k soak looked clean in the Geode log, but `bridge_check`'s terminal result was never seen |
| M4-M5 | done | learn tests pass |
| M6 | done | **passed**: CartPole 500.0 on seeds 1, 2, 3 (20 greedy episodes each) |
| Mock game | - | **passed**: seeds 1-3 (2M steps, `configs/mock_ppo.yaml`) complete all 5 training and all 3 held-out levels greedily; held-out with sticky 0.1: 0.83 / 0.90 / 0.97 |
| M7 | config only | **not done**: the owner has not built custom levels; `level_ids: [0]` (the open level) only |
| M8 | done (rule agent, genetic agent, GridEncoder, GridCNN) | **not done**: no results table. The grid + CNN path is untrained (unit tests only). The rule agent beats all 8 mock levels; the genetic agent beats mock level 1 in 8 generations |

Tests: `pytest` gives 102 passed, 2 skipped (POSIX-only bridge tests).
Run them with `CUDA_VISIBLE_DEVICES=-1` until CUDA is fixed (see below).

## What changed beyond filling in the stubs

1. **Encoder distance code** (`gdrl/features/encoder.py`, `ObjectListEncoder`).
   The plain M5 encoder did not learn on the mock: PPO plateaued at ~20%
   progress and the greedy policy held the button forever. Behaviour cloning
   the rule agent into the same 64x64 tanh MLP showed why: it caught 7.5% of
   the rule's presses. The jump window is ~6 units, ~0.01 of the normalized
   `dx`, too fine for the MLP. Fix: Gaussian bumps of the left-edge distance
   (every 10 units over -30..180) for the 3 nearest objects. Behaviour
   cloning then caught 100% of presses, and PPO learned the levels.
   **The original project's M5 encoder still has this flaw.** Tell the owner
   when they reach mock training; do not fix it there yourself.
2. **Player features**: `x_speed` and a one-hot of the game mode (cube,
   ship, ball, UFO, wave, robot, spider, swing). Untested on the real game.
3. **`train.resume`** in `gdrl/train.py`: continue from a checkpoint
   (weights, optimizer, update count, so LR annealing continues).
4. **`configs/gd_overnight.yaml`**: real game, 30M steps, `max_steps: 15000`
   (250 s, so a full level is not truncated), checkpoints every 20,480 steps,
   CPU.
5. **`overnight.ps1`**: waits 10 s, trains, and on any non-zero exit waits
   60 s and resumes from the newest `runs\*gd-overnight*` checkpoint (up to
   20 restarts). Tested against the mock server with a killed connection:
   it resumed and finished.
6. **`gdrl/tools/genetic.py`**: drives `GeneticTimingAgent` (its own loop;
   fitness needs whole episodes from a fixed start).
7. `logs/` is gitignored (training output from the tests lives there).

## Environment

- No venv of its own. Use the original's interpreter **from this folder**:
  `..\geometry-dash-rl\.venv\Scripts\python.exe`. `gdrl` is pip-installed
  in editable mode pointing at the original, but `python -m ...` run from
  this folder puts this folder first on `sys.path`, so this copy's `gdrl`
  is the one imported (checked: `gdrl.__file__` is under the autopilot
  folder). Do not `pip install -e .` here; it would repoint the original's
  venv at this copy.
- The mod source in `mod/` is identical to the original's. Both copies share
  one installed mod (same mod id), so the last build wins. The owner now
  permits agents to build it from here and to relaunch GD (2026-10-11).
- Geode logs: `C:\Program Files (x86)\Steam\steamapps\common\Geometry Dash\geode\logs\`.
- **Driving the game:** use `python -m gdctl` (`gdctl/README.md`): launch,
  kill, build, screenshots, menus, levels, online download, StartPos editing,
  `prepare-training`. Take the game lock first (`gdctl lock acquire --owner
  <you>`); it is mandatory. The companion mod is in `control-mod/`.

## CUDA problem (open)

Symptom: any `optimizer.step()` crashes, even with every tensor on the CPU:

```
torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
```

What is known:

- `torch.cuda.is_available()` is True and `nvidia-smi` lists the RTX 4060
  Laptop GPU (compute mode Default, ~2% use, only desktop apps on it).
  Creating any CUDA tensor fails (`torch.zeros(1, device="cuda")`).
- It is not the Claude Code sandbox: the same failure happens with the
  sandbox off.
- Why CPU training hits it: `torch/optim/optimizer.py`
  `_accelerator_graph_capture_health_check` calls
  `torch.accelerator.current_stream()` whenever an accelerator is available,
  which needs a CUDA context.
- CUDA worked on 2026-10-09 (AGENTS.md). Driver 536.52 (CUDA 12.2) with the
  cu126 wheels relies on minor-version compatibility.
- Workaround in use: `CUDA_VISIBLE_DEVICES=-1` (hides the GPU; CPU training
  works). `overnight.ps1` sets it. The networks are tiny, so the CPU is the
  right device for them anyway.

The owner is rebooting to see if that fixes it. If it does not, things to
check, in order:

1. Repro: `..\geometry-dash-rl\.venv\Scripts\python -c "import torch; print(torch.zeros(1, device='cuda'))"`.
2. Windows Graphics settings / NVIDIA Control Panel: is python.exe forced
   onto the integrated GPU, or is the dGPU powered down (Optimus)?
3. Something holding the GPU exclusively (`nvidia-smi` process list), or a
   recent TDR (Event Viewer, "Display" source, nvlddmkm).
4. Driver: 536.52 is old for cu126. Updating the NVIDIA driver is the real
   fix if the context still cannot be created. Ask the owner before
   installing drivers or changing torch wheels.

## Running the overnight training (owner does this)

1. Plug in, disable sleep. Start GD; Click Between Frames and QOLMod must
   be disabled. Open a short **cube-only** level in normal mode.
2. In PowerShell:
   ```powershell
   cd $HOME\Desktop\geometry-dash-rl-autopilot
   powershell -ExecutionPolicy Bypass -File .\overnight.ps1
   ```
3. Within 10 s, click into GD and resume it if it paused (GD pauses on
   focus loss until a client connects; after that the mod blocks it).

Morning checks:

- Console output: `logs\gd_overnight.log`.
- Curves: `..\geometry-dash-rl\.venv\Scripts\tensorboard --logdir runs`.
- Score (level open in GD): `..\geometry-dash-rl\.venv\Scripts\python -m gdrl.evaluate --run runs\<newest gd-overnight run>`.
- Resume manually: `python -m gdrl.train --config configs/gd_overnight.yaml --set train.resume=runs/<run>/checkpoints/final.pt`.

## Start-position curriculum (2026-10-11)

Plain training from the level start (`gd_overnight.yaml`) plateaued on
Stereo Madness at 19.4% from ~0.7M to 3.8M steps (entropy 0.1, every episode
identical). The cause was exploration, not the encoder: the agent held the
button through a landing at x ~ 5050 and auto-jumped into a spiked slab, and
the low route needs ~40 releases in a row, which a collapsed policy never
tries. Fix: `gdrl/curriculum.py` (`env.curriculum` in the config) starts
episodes from the level start or from local copies of the level, each with
one StartPos (`gdrl.tools.make_starts` builds and checks them in game;
`configs/starts/sm.yaml`). It switches levels with `gdctl open-level`
(~2.3 s, so it stays on a start for `switch_steps`), and it waits after a
switch because a freshly opened level answers STEPs with 0 physics ticks
during its intro. That intro is the source of the "ran 0 physics ticks"
warnings in concern 3 below. `train.eval_every` logs one greedy episode from
the real level start (`eval/*`). The episode log is
`logs/curriculum_sm_episodes.csv` (one row per attempt).

```powershell
powershell -ExecutionPolicy Bypass -File .\overnight.ps1 -Level main:1 -Config configs/gd_curriculum.yaml -RunFilter "*gd-curriculum-sm*"
& $py -m gdrl.tools.make_starts --level main:2 --prefix gdrl-bot --every 1500 --out configs/starts/bot.yaml   # next level (game up, lock held, no client)
```

## Other useful commands (from this folder, CUDA hidden)

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
$py = "..\geometry-dash-rl\.venv\Scripts\python.exe"
& $py -m pytest -q                                              # all tests
& $py -m pytest -m slow tests/learn/test_ppo.py                 # CartPole gate, one seed
& $py -m gdrl.train --config configs/cartpole_ppo.yaml --set seed=1
& $py -m gdrl.train --config configs/mock_ppo.yaml --set seed=1
& $py -m gdrl.evaluate --run runs\<run> --levels 101 102 103 --sticky 0.1
& $py -m gdrl.evaluate --config configs/mock_ppo.yaml --set "agent={name: rule}" --levels 1 2 3
& $py -m gdrl.tools.genetic --config configs/mock_ppo.yaml --level 1
```

Existing runs: `runs\*cartpole-ppo_s{1,2,3}` and `runs\sweep\*` (the mock
sweep: `rbf_s1..s3`, `rbf_h128`, `rbf_lr1e3` all learn; `rbf_ent003`
(ent_coef 0.003) failed completely, so PPO is sensitive to the entropy bonus).

## Concerns and open items

1. **The real-game run is untested.** The ~1,000 steps/s used to size the
   30M-step budget is estimated from the M3 soak. Non-cube modes are
   untested.
2. **If GD itself crashes overnight, nothing can recover it**: the script
   retries for ~20 minutes, then stops (checkpoints are kept). A planned fix:
   a mod setting "open level N on startup" so `overnight.ps1` can relaunch
   GD (`start steam://rungameid/322170`), wait for port 5555, and resume.
   Building the mod changes the shared installed mod, so ask the owner first.
   A fuller fix (RESET with `level_id` loads that level) is a protocol change:
   PROTOCOL.md + Protocol.hpp + protocol.py + version bump, and overlaps the
   owner's M7.
3. **Unexplained "a one-tick update ran 0 physics ticks" warnings** once in
   the Geode log, on the first attempt after a client connected ~2 s after
   opening a level (first death at tick 12786 vs 1314 later). Not reproduced;
   the owner chose not to chase it. Watch first episodes after a connect.
4. **One level means memorization.** A good overnight score shows the
   pipeline works, not that the agent generalizes.
5. **Offered, not started:** the M8 comparison on the mock (PPO object-list,
   PPO grid + CNN, rule, genetic, random), ~30 minutes, no game needed. It
   would also be the first training run of the grid + CNN path.

## Community StartPos curriculum, bridge v2 in game (2026-10-12)

The self-made StartPos copies (`configs/starts/sm.yaml`, `gdrl-sm-sNN`) are
retired: the owner wants community "<level> StartPos" copies only, played as
downloaded. Stereo Madness uses `SM Start Pos` (online id 141625101, 29
StartPos objects; offline diff vs `main:1`: identical except the 3 coins).
`configs/starts/sm_community.yaml` lists them; `configs/gd_sm_community.yaml`
trains on them (all starts open, 20% of episodes from the real start,
`[128,128]` net, `device: cuda`).

```powershell
powershell -ExecutionPolicy Bypass -File .\overnight.ps1 -Level saved:141625101 -Config configs/gd_sm_community.yaml -RunFilter "*gd-sm-community*"
```

- The mod patch (protocol v2, `logs/mod-patch`) is now applied and installed
  (mod v0.2.0). RESET `start_pos k` = k-th StartPos by x, `-2` = real level
  start; switching starts costs one RESET, not a level reload.
- Fix: GD respawns from a cached Start Pos checkpoint (`m_currentCheckpoint`),
  so RESET k spawned at GD's default StartPos until that is cleared (done in
  `handleReset`). The curriculum now raises if a spawn is >5 units off.
- Fix: completion. GD's `levelComplete()` fires in real time after the end
  animation, i.e. after the agent had already reset; completion was never
  reported (episodes ran to `max_steps` at the end wall) and the late call
  leaked into the next attempt. Completion is now taken from
  `playEndAnimationToPos` / `m_levelEndAnimationStarted` inside the tick;
  stale `levelComplete()` calls are dropped. A client that disconnects on a
  completed attempt leaves GD's normal end screen (leave with
  `gdctl exit-level` or the menu button); the next client's RESET removes it.
- Saved (downloaded) levels lose their data when GD restarts;
  `overnight.ps1` re-downloads `saved:<id>` levels before each prepare.
- GPU: `overnight.ps1` no longer hides CUDA (`CUDA_MODULE_LOADING=LAZY`).
  On the mock, 1 env, cuda was 2,230 sps vs cpu 3,050; the real game runs
  ~300 sps, so the device costs ~3% there.

### Training changes from the data (2026-10-12 evening)

- **Encoder `focus: obstacles`**: rows of floor blocks filled the 3 fine
  distance slots, so the spike on a raised floor (x = 3075) was never timed;
  every greedy run died there (11.4%). Obstacles (hazards, solids above the
  feet) now come first.
- **`agent.act_device: cpu`** (updates stay on `device: cuda`): profiling
  showed act() on cuda at 1.8 ms/step vs 0.7 ms for the whole env step.
  Training went from ~300 to ~3,000 steps/s.
- **`curriculum.horizon_starts: 2`**: hard starts died ~26 steps after
  spawning while easy ones ran ~300 steps, so hard spots got ~1% of samples.
- **`ent_coef: 0.02`**: 9 starts sat at 0% with the same death spot for
  300k+ steps.
- **Minimized GD stalls the bridge** (trainer timeout). `gdctl unminimize`
  restores it behind other windows without focus; `overnight.ps1` calls it
  every 15 s.
- `python -m gdrl.tools.curriculum_report logs/sm_community_episodes.csv`:
  per-start success and top death spots per time window (is it learning, or
  repeating the same mistake?).
- Correction: the "3,000 steps/s" printed after a resume counted the resumed
  steps (fixed in train.py); the real rate is ~970 steps/s (~3x the old ~300).

### Checkpoint search on the real game (2026-10-12): not working yet

- `snapshot_check` PASSES on Stereo Madness (once the object HIDDEN flag,
  a visual opacity bit, is ignored: `GameState.physics_bytes()`). SAVE
  0.14 ms, LOAD 0.6 ms, ~15k ticks/s. Hand-written chained/interleaved
  save-load tests (`$TMPDIR` scripts, same pattern as the search) were exact.
- But `demon_search` on `main:1` reports "replay differs" from segment 4
  (x ~ 311; e.g. player_y 81 vs 37, or ON_GROUND missing) and then
  backtracks forever at x ~ 389 (even with `verify_every=1`). The search now
  logs the differing field and checks that the slot itself is unchanged
  (it is: the node state matches its slot, but not a clean replay of its
  inputs). Cause not found yet. Added `wait_until_running` (level intro
  answers STEPs with 0 physics ticks) to demon_search; not the cause here.

### More data-driven changes (2026-10-12, 20:10-20:40)

- **Epsilon exploration on stuck starts** (`curriculum.explore_eps: 0.3`,
  `train._explore`): starts passed < 5% of the time get 30% random actions,
  with the behaviour policy's log-prob stored so PPO stays consistent. The
  greedy policy at the stuck starts had p(hold) 0.9-1.0 into the same spike.
  Alone it did not unstick them (2M steps).
- **Terrain profile** (`encoder.profile: true`): per column ahead (every 15
  units to 240): support height (solid top within a jump, else ground),
  hazard on that surface, overhead clearance. The object slots were full of
  the pit spikes under slab paths, so the slab edge and the next slab were
  invisible. `gdrl.tools.widen_checkpoint` carried the 2.3M-step policy over
  (zero weights for the new inputs), so nothing was retrained from scratch.
  First 5 minutes: three of the 0% starts cleared for the first time.

## Stereo Madness beaten; Bloodbath started (2026-10-12 21:35)

- SM: greedy policy `results/sm_beat_update_001170.pt` beats `main:1` 3/3
  (5/10 sticky 0.1). Ledger + `progress_reports/progress_report_2026-10-12_2135.md`.
- Bloodbath: community copy 'Bloodbath Start Pos' (KJQ, 77986175) is the
  original object for object plus 14 StartPos (other copies remap 2.2
  decorations or add a size portal). `configs/gd_bb_community.yaml`,
  `configs/starts/bb_community.yaml`, initialised from the SM policy
  (`runs/init-gd-bb-community/checkpoints/init.pt`). Encoder obstacle focus
  and profile now mirror y when gravity is flipped.
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\overnight.ps1 -Level saved:77986175 -Config configs/gd_bb_community.yaml -RunFilter "*gd-bb-community*"
  ```
- Not handled yet: dual mode (no player 2 in the object-list encoder).

### Bloodbath (2026-10-13)

- Snapshot starts (`curriculum.snapshot_every: 300`): the run's own saved
  states become starts. Bloodbath from the level start: 5.4% -> 12.2% in
  1.5 h (StartPos are ~2,800 units apart; every segment had been 0%).
  Snapshots on Bloodbath are not always exact (ship y off by 0.02 in one
  check): fine for practice starts, not for the search.
- Plateau at 12.2% (3x ship corridor, x ~ 4,300-4,500): the bridge's 64
  nearest objects were all single wall blocks within ~150 units. The bridge
  now merges plain blocks into column/row rectangles and searches +-10
  blocks vertically (was 20); the view in that corridor went from ~150 to
  ~600 units. Snapshot check still passes.
- `overnight.ps1` keeps Windows awake (the PC slept 22 h on 10-12).
- **The 30% wall (2026-10-14)** is the UFO tunnel at x 11,715-12,225
  (floor and ceiling spikes, ~20 units of slack). The greedy run dies at
  x = 11,819 after flapping into a ceiling spike; the save point before it
  was 0/570 and was retired (`snapshot_retire_after: 150`). The tunnel IS
  passable: `logs/ufo_search.py` (checkpoint search from the agent's own
  state at the ball portal, x = 11,002) got through in 65 s / 6,008
  rollouts, and a continuous replay confirms it. Snapshots are exact in UFO
  mode (only the ball's visual rotation differs). The solution plus the
  policy's inputs from StartPos 4 is `configs/demos/bb_ufo_from_sp4.npy`;
  `curriculum.demos` replays it at startup and adds a start every 100 units
  on it (x 10,900-12,318), so the agent practises the tunnel from states
  that can be survived.

## Watching the policies at 1x (2026-10-15)

- Steam updated GD to 2.209 on 10-15; Geode does not support it yet and
  disabled every mod. GD was rolled back to 2.2081 (Steam console
  `download_depot 322170 322171 3816559102876907245`, files copied over the
  install) and `steamapps\appmanifest_322170.acf` is read-only to stop the
  update. To undo: clear read-only, Steam > Verify integrity.
- `python -m gdrl.tools.watch sm|bb [--attempts N] [--speed 1]` plays a
  policy from `configs/watch.yaml` in game (takes the lock, opens the level,
  keeps GD un-minimized). Policies: `results/watch/sm_beat.pt` (beats Stereo
  Madness, 84.7 s) and `results/watch/bb_30pct.pt` (Bloodbath, dies at 30.0%,
  x = 11,819, UFO tunnel: it presses again while falling and hits the ceiling).
- The dashboard (`python -m gdrl.dashboard`) has the same Play buttons on the
  level board and in the Stereo Madness / Bloodbath drawers. It now hides
  mock levels, mock runs, CartPole and the mock agent comparison (`--mock`
  shows them).
- Two bridge settings, read every frame and set at runtime through the
  control mod (`gdctl mod-setting <key> on|off`, new `set_mod_setting`):
  `watch-mode` (one STEP per rendered frame; without it lockstep batches
  ~6 paced STEPs into each 100 ms frame, ~10 fps) and `legacy-view` (the
  object view before 10-13: no block merging, no No-Touch/speed-portal
  handling, +-20 blocks vertically). The SM policy needs `legacy-view`: on
  the current view it dies at 19.4%. `prepare-training` turns both off, so
  training is never capped or fed the old view.
