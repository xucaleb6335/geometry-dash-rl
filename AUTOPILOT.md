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

### Bloodbath past the UFO tunnel (2026-10-15 night, loop agent)

- Why it stayed at 30%: the earlier tunnel demo (`bb_ufo_from_sp4.npy`) was
  searched with 2-tick/1-tick inputs (54 of its 481 4-tick blocks mix press
  and release), so the 4-tick policy could not reproduce it; and the run that
  had the demo starts was stopped after 40 minutes (~30 tries per start).
- A search at the policy's rate (`decision_ticks=4`) from the agent's own
  state at x 11,400 got through in ~70 s. Root at x 11,002 gave solutions
  whose replay died (save/load is not exact there). The full demo from
  StartPos 4 is `configs/demos/bb_ufo_4tick.npy` (valid to x ~12,680; used to
  12,550). Demo save points are now only on 4-tick boundaries (`align`) and
  up to `to_x`.
- `gdrl.tools.demo_bc` cloned it into the 30% policy (plus the policy's own
  run as retention data): the greedy policy now reaches x 12,688 (32.5%)
  from the level start, 3/3. Saved as `runs/bc-gd-bb-community/checkpoints/bc.pt`
  (and `results/watch/bb_ufo_bc.pt`, watch preset `bb_bc`); PPO resumed from it.
- Attempts are not independent: dynamics depend slightly on how the previous
  attempt ended (same inputs, y off by 0.4 at x ~1,690; greedy runs ended at
  10,772 or 11,819 depending on the previous attempt). The bridge now resets
  player rotation state and resets twice; rotation is now identical across
  attempts, but a smaller history effect remains (cause not found). demo_bc
  records several attempts so the clone is robust to it.

### Bloodbath wave at x ~15,000-15,200 (2026-10-16 06:30, loop agent)

- After the tunnel the eval rose 30% -> 38.5% by 03:30, then stayed flat for
  3 h; deaths clustered at x 15,080-15,165 (wave, mode 4).
- A 4-tick search from the policy's own state at x 14,900, reached from
  StartPos 5, got through in 22 s: `configs/demos/bb_15k_4tick.npy` (from
  StartPos 5, valid to x 15,707). Added as demo starts (14,800-15,600).
- Searching from states reached from the level start failed (roots 14,600,
  14,800: unsolved in 600 s; root 14,900: the solution's replay died), so
  save/load is probably not exact after a long run from the level start.
- `demo_bc --retain-until-x 14900` (new flag: keep the policy's own actions
  up to the search root, since the demo opens with the policy's actions from
  StartPos 5). Result `runs/bc2-gd-bb-community/checkpoints/bc.pt`: from
  StartPos 5 it follows the demo to 15,655; from the level start it reaches
  15,002 (the original policy reached 15,138), because it arrives at the wave
  in a slightly different state. PPO resumed from it with the demo starts.
- 10:30: flat again at ~40.8% for 2 h (deaths x 15,850 and 16,050, UFO).
  Searched from x 15,550 (StartPos 5 + current policy); roots 15,750 and
  15,650 gave solutions whose replay died. `configs/demos/bb_16k_4tick.npy`
  (StartPos 5, valid to x 16,761) replaces the 15k demo; demo starts now
  14,900-16,600 (the 14,800 save point killed the player on spawn). BC:
  `runs/bc3-gd-bb-community/checkpoints/bc.pt` (from StartPos 5 to 16,707;
  from the level start 15,588, down from 16,045, same arrival-state gap).
- 13:27: practice spots on the route pass ~100%, but runs from the level
  start stay at ~40.2% and die at x ~15,850 (100 of 281 deaths): they reach
  it in a slightly different state than the route from StartPos 5. Next
  step: search from the policy's own state when played from the level start
  (several roots between x 15,000 and 15,550, keep only a root whose
  solution replays alive), then demo_bc with that demo.
- Stopped 2026-10-16 ~13:50 for a PC restart. Resume with the overnight
  command above (it resumes from the newest `runs\*gd-bb-community*`
  checkpoint, `runs/20261016-110740_gd-bb-community_s1`). The best policy
  from the level start so far is `results/watch/bb_41pct.pt` (watch preset
  `bb_41`, 40.9%).
- 20:30: why the SP5 routes did not transfer: not a level/trigger difference
  (modes, speeds and the object view match on both routes, and save/load is
  exact there) and not the 4-tick decision phase (1-, 2- and 4-tick searches
  from the level-start state all stalled at x ~15,463). The search
  (`lookahead_ticks` 80) committed to a high UFO path that is a dead end
  ~150 units later. With `lookahead_ticks: 240`, `beam_width: 16`,
  `max_found: 24` it went from the policy's own level-start state at
  x 15,100 to x 16,788 in ~2 min. `configs/demos/bb_start_16k_4tick.npy` is
  that route from the real level start (one mixed 4-tick block set to
  release); demo starts now 15,000-16,700 on it. demo_bc ->
  `runs/bc4-gd-bb-community/checkpoints/bc.pt`: the greedy policy from the
  level start reaches x 16,833 (~42.8%), 3/3. Prefer level-start routes and
  a long lookahead for future searches.
- 2026-10-16 22:57: `anneal_lr: false` at the full 2.5e-4 wrecked the
  policy within an hour (eval 2-30%, runs from the start rarely past the
  tunnel). 00:30: restarted from bc4 (copied to `runs/bc5-gd-bb-community`)
  with a constant `learning_rate: 5e-5`. Best policy from the level start:
  `results/watch/bb_43pct.pt` (watch preset `bb_43`, 42.8%, x 16,833).
- 2026-10-17 night (lr 5e-5): eval median 41.4% for ~4 h, best single evals
  54.8%; runs from the start mostly died at x ~16,300 (p90 flat at 16,300
  for 3 h). Searches from level-start roots 15,900 / 15,500 / 16,100 gave
  solutions whose continuous replay died at x 16,285, so save/load is not
  exact somewhere before it (hidden state). Root 15,700 replayed alive to
  17,668: `configs/demos/bb_start_17k_4tick.npy`, demo starts 15,000-17,550.
  demo_bc -> `runs/bc6-gd-bb-community/checkpoints/bc.pt`, greedy from the
  level start x 17,628 (~44.8%), 3/3. Watch preset `bb_45`.
- 2026-10-17 10:30: PPO from bc6 drifted back to ~41% in ~1 h (deaths
  x 15,750-16,200, on the cloned route), the third time a cloned route
  eroded. Added a behaviour-cloning anchor to PPO (`agent.bc_data`,
  `bc_coef`, `bc_batch`): each minibatch adds weighted cross-entropy on the
  demo_bc pairs, which demo_bc now saves as `<out>.bcdata.npz`. bc7 = bc6
  plus its pairs (the route from the level start to x 17,600 plus the
  policy's own actions before the search root). Resumed from
  `runs/bc7-gd-bb-community` with `bc_coef: 1.0`; first evals 44.8%.
- 2026-10-17 12:27-16:05: the anchored policy held 45% but every run died at
  x ~17,710. The owner watched it and spotted the cause: the route flew past
  the **ship portal at x 16,695** (with a 3x speed portal) as a UFO; the gaps
  after 17,700 need the ship. The search had preferred the UFO path because
  it survives its 240-tick lookahead (it never sees the dead end), PPO
  because the UFO path reliably pays 45%, and the anchor locked it in. The
  bridge's object view never showed that portal on the route; the level
  string does (`gdctl level-export`). Fix: search from level-start roots
  16,400/16,500 with a gate that counts "past x 16,760 and not a ship" as
  death, and (ship-mode save/load is inexact after ~16,800) restore nodes
  by loading the exact pre-portal root and replaying their ticks. Both
  roots reached x ~19,000 and replay alive from the level start.
  `configs/demos/bb_start_19k_ship_4tick.npy` (cut at x 18,409 before a
  mixed 4-tick block that dies either way), demo starts 15,000-18,350,
  demo_bc -> `runs/bc8-gd-bb-community` (greedy from the level start
  x 18,507, ~47.2%, 3/3; watch preset `bb_47`); anchor now bc8's pairs.
- 2026-10-17 19:27: after the ship route, PPO (anchored) went 47% -> ~57%
  in 2 h, then stopped at x 22,413 (UFO, just after the normal-size portal
  at 22,365; a logged greedy run takes every portal correctly, so a hard
  spot, not a skipped portal). The root-replay search (exact restores) from
  level-start roots 21,950 / 22,150 / 22,250 solved all three in 7-10 min,
  each replaying alive to ~23,750. `configs/demos/bb_start_23k_4tick.npy`
  (root 21,950, to x 23,767), demo starts 15,000-23,600, demo_bc ->
  `runs/bc9-gd-bb-community` (greedy from the level start x 23,781, ~60.4%,
  3/3; watch preset `bb_60`); anchor now bc9's pairs.
- 2026-10-17 20:27: from bc9 every run from the start died at x 23,783,
  16 units past the cloned route's end (the search stops at its goal_x in a
  state that may be doomed just ahead; leave margin: clone to well before
  the demo's end). Searches with the policy as prefix died in the UFO
  tunnel on the first attempt after a level reopen (history effect), so the
  prefix is now the demo itself (`DEMO=` in scratchpad `seg_gate.py`, a
  configurable mode gate: `GATE_X/GATE_END/GATE_MODE`). Roots 23,000 /
  23,300 / 23,500 with ball mode required after 24,060 all solved in 5-8
  min to x ~25,050. `configs/demos/bb_start_25k_4tick.npy`, demo starts
  15,000-24,800, demo_bc -> `runs/bc10-gd-bb-community` (greedy from the
  level start x 25,061, ~63.7%, 3/3; watch preset `bb_64`); anchor = bc10.
- 2026-10-17 21:27: from bc10 (64.5-64.9% evals) runs died x 25,050-25,562,
  none past the gravity-up portal at 25,575. Searches from roots 24,500 /
  24,700 / 24,900 (demo prefix; gates: ball 24,060-26,520, cube
  26,600-27,800; `GATES=x0:x1:mode,...`) all solved in 7-9 min to x ~27,000.
  `configs/demos/bb_start_27k_4tick.npy` (root 24,700), demo starts
  15,000-26,800, demo_bc -> `runs/bc11-gd-bb-community`: greedy from the
  level start x 28,074 (~71.4%), 3/3, past the UFO portal at 28,005; watch
  preset `bb_71`; anchor = bc11.
- 2026-10-17 22:27: from bc11 (71.3% evals) runs died at x 28,050-28,150,
  just after the UFO + 3x portal at 28,005 (the never-cleared community
  StartPos 27,915 is there too). Search from root 26,900 (27k demo prefix,
  gates cube 26,600-27,800 / UFO 28,060-28,990) solved to x 28,943 in
  12 min. `configs/demos/bb_start_29k_4tick.npy`, demo starts
  15,000-28,700, demo_bc -> `runs/bc12-gd-bb-community` (greedy from the
  level start x 28,906, ~73.5%, 3/3; watch preset `bb_73`); anchor = bc12.
- 2026-10-17 23:27: from bc12 every run from the start died at x 28,906
  (155/218), ~60 units past the cloned route's end, before the gravity
  flips at 28,995. Search (29k demo prefix, gates UFO 28,060-29,400 / ship
  29,460-30,760) from root 28,500 solved to x 30,715 in 18 min (roots
  28,700 and 28,300 did not, 20 min each). `configs/demos/bb_start_31k_4tick.npy`,
  demo starts 15,000-30,500, demo_bc -> `runs/bc13-gd-bb-community`
  (greedy from the level start x 30,769, ~78.2%, 3/3, past the UFO portal
  at 30,765; watch preset `bb_78`); anchor = bc13.
- 2026-10-18 01:27: PPO from bc13 reached x 31,138 (79.1%) in ~1 h, then
  every run from the start died there (196/552), just after the ship portal
  at 31,065. Search (31k demo prefix, gates ship 31,120-31,690 / ball
  31,750-32,640) from roots 30,400 / 30,200 / 30,550 all solved in 12-18
  min to x ~32,600; root 30,400's full replay from the level start died at
  30,477 (history effect), the other two replayed alive.
  `configs/demos/bb_start_33k_4tick.npy` (root 30,200), demo starts
  15,000-32,400, demo_bc -> `runs/bc14-gd-bb-community` (greedy from the
  level start x 32,811, ~83.4%, 3/3; watch preset `bb_83`); anchor = bc14.
- 2026-10-18 03:27: PPO from bc14 fell back to 74.3% (every run from the
  start died at x 29,230-29,240, on the cloned route; the latest checkpoint
  played greedily died at 16,941). Found a real source of the "history
  effect": the bridge marked blocks HIDDEN when their **opacity** was 0, and
  hidden blocks were not merged. Opacity is visual state that alpha
  triggers leave behind from earlier attempts, so the same physics state
  gave a different object view depending on attempt history (fresh after a
  level reopen vs after the level had been played). Reproduced: after a
  reopen bc14 died at x 6,095 every time until any attempt had passed that
  area, then reached 32,811. Fix (mod, non-legacy view): HIDDEN = editor
  hide flag only, opacity ignored. Fresh and played views are now
  identical along the whole route (object-by-object check to x 32,625).
  bc14 re-cloned on the new view -> `runs/bc15-gd-bb-community` (x 32,811
  from the level start, 3/3, also right after a reopen); anchor = bc15.
- 2026-10-18 05:27: from bc15 PPO reached 84.9% (x 33,410) in 45 min, then
  the eval fell to 41-58% with deaths spread along the cloned route. The
  latest checkpoint got exactly one anchor decision wrong (decision 3,119 of
  5,434, x ~17,490, a top death spot) while the mean bc_loss stayed 0.0002:
  with 16k pairs and 256 sampled per minibatch, that pair was rarely in a
  BC batch. New `agent.bc_hard_p` (0.98): before each epoch, pairs where
  the demo action has p < 0.98 are added to every minibatch's BC batch
  (stat `train/bc_hard`). Restarted from bc15 (copied to
  `runs/bc16-gd-bb-community` so the resume picks it).
- 2026-10-18 06:27: with hard BC pairs the route held (evals 84-85%, a few
  dips to ~45% at x ~17,700), but the frontier sat at x ~33,410 for ~2 h
  across two runs. Search (33k demo prefix, gates UFO 32,700-33,800 / wave
  33,880-34,560) from roots 32,400 / 32,200 / 32,550: all solved in 14-19
  min to x ~34,500 and replay alive from the level start.
  `configs/demos/bb_start_35k_4tick.npy` (root 32,400), demo starts
  15,000-34,300, demo_bc -> `runs/bc17-gd-bb-community` (greedy from the
  level start x 34,908, ~88.7%, 3/3; watch preset `bb_89`); anchor = bc17.
- 2026-10-18 12:00-12:35: the owner watched the "91.5%" policy and it died
  at 89.1% (x 35,082), 3/3, also in fast greedy runs (so not watch mode,
  not the attempt history: a demo pass or a run from a late StartPos first
  changed nothing). Cause: the greedy run flips between consecutive PPO
  updates (evals alternate 91.5% / 89.1%), and the saved checkpoint was a
  89.1% one; update 20,770 reaches x 36,014 (the dual portal). Fix:
  `train.eval_ratchet` -- an eval that beats the best so far is added to
  the BC anchor (minus its last 50 decisions; weight 3), the anchor is
  saved as `checkpoints/bc_ratchet.npz` and the policy as
  `checkpoints/best_eval.pt` (watch that one). On a restart, point
  `agent.bc_data` at the newest `bc_ratchet.npz` to keep the added runs.
  Resumed from update 20,770 (`runs/bc18-gd-bb-community`); its first eval
  (91.5%) was added. Watch preset `bb_91` = update 20,770.
- Dual-mode search (2026-10-18 10:30-12:15): roots in the wave section
  (34,100 / 34,450) reached x 36,460-36,720 but their full replays died
  right after the root (save/load inexact in wave mode); root 33,700 (UFO,
  before the wave portal) replayed alive but stopped at x 35,947 in 30 min.
- 2026-10-18 13:27: with the ratchet every eval was exactly 91.5%, but 145
  of 182 runs from the start died at the cube+dual portal (x 36,010-36,020).
  Search with the 33,700 search's route as prefix (it replays alive to
  35,947), roots 35,700 / 35,500 / 35,300 in the ship section, gates ship
  to 35,990 / cube 36,075-36,700: all solved in 4-7 min; roots 35,500 and
  35,300 replay alive from the level start (35,700's died at 35,750). The
  route had 5 mixed 4-tick blocks (in the dual part); 15 of the 32
  uniform settings replay alive, one is `configs/demos/bb_start_37k_4tick.npy`
  (to x 36,730). Demo starts 15,000-36,500, demo_bc from bc18 ->
  `runs/bc19-gd-bb-community`: greedy from the level start x 36,967
  (~93.9%, 3/3) -- through dual although the encoder has no player 2;
  watch preset `bb_94`; anchor = bc19.
- 2026-10-18 14:27-18:03: **Bloodbath beaten.** From bc19 every run died at
  x 36,967. A search from 36,400 (37k route prefix) reached 38,259, then
  roots 37,900 / 38,050 both stalled at x ~38,990: the last obstacle is a
  ground spike row (x 38,952-39,018) and the player was a **mini** cube;
  testing a jump at every single tick before it (scratchpad `endjump.py`)
  showed no single mini jump clears it. Cause: in the dual section player 1
  had dropped to the bottom lane and taken the mini portal at x 36,075 (y 75,
  meant for the lower player; size is per player in dual), and there is no
  normal-size portal after it. A search with a size gate (player 1 not mini
  past 36,100; StateFlag MINI = 32) went through dual with player 1 on top at
  normal size, then searches from 36,400 (to 38,429) and 38,250 (70 s)
  **completed the level**. `configs/demos/bb_start_end_4tick.npy` (level
  start to the end, no mixed 4-tick blocks) completes both the community copy
  and the original (10565740). demo_bc -> `runs/bc20-gd-bb-community`
  (completes the community copy 3/3), but on the original it died at x 34,996:
  the object view differs slightly between the copies (333 of 6,710 decisions
  see different inputs; one decision flipped). New `demo_bc --extra-bcdata`
  clones the route recorded on the original together with bc20's pairs ->
  `runs/bc21-gd-bb-community`: greedy from the level start **completes the
  original Bloodbath 3/3 and the community copy 3/3** (gdrl.evaluate, sticky 0,
  fresh level open). Ledger updated; watch preset `bb_beat`
  (`results/watch/bb_beat.pt`, level `saved:10565740`: download it first with
  `gdctl download-level 10565740` if GD was restarted). Caveat: this is PPO
  plus 20 search segments cloned into the policy, not PPO alone; with sticky
  actions it would not complete. Next level: Rauchkammer.

## Rauchkammer (2026-10-18 18:20)

- Community copy: 'rauchkammer startpos' by 2dependent (147032152): every
  object of the original 146399247 at the same id and position (rotations
  differ by < 0.1 degree, re-save rounding), plus 15 StartPos and 6 id-1755
  objects near x 13,815. The most downloaded copy (mat0tas, 146683938) differs
  in ~8,000 object properties; KeksCool's lacks a yellow pad. The original has
  no random triggers (ids 1912/2068), but ~275 move triggers (901): check
  save/load exactness before relying on the search. Level length 14,635;
  modes cube/ship/wave/UFO, mini at 10,950 and 13,845.
- `configs/starts/rk_community.yaml` (spawns checked with RESET start_pos k),
  `configs/gd_rk_community.yaml` (the Bloodbath setup, no demos, no anchor
  data), initialised from the Bloodbath policy (`runs/init-gd-rk-community`).
  `PPOAgent.add_bc` now starts an anchor when there is none, so the eval
  ratchet works from scratch.
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\overnight.ps1 -Level saved:147032152 -Config configs/gd_rk_community.yaml -RunFilter "*gd-rk-community*"
  ```
- 2026-10-18 19:00 - 2026-10-19 00:30: PPO from the Bloodbath policy got 0%
  on every Rauchkammer start, so progress is search-driven (scratchpad
  `seg_gate.py` with the previous route as prefix, then keep only roots whose
  route replays alive from the level start). Valid routes (logs/, not
  committed): x 4,859 -> 6,726 (root 4,400; root 4,550 replayed dead) ->
  8,386 (root 6,250) -> 10,102 (root 8,250, `logs/rk_8250.npy`, wave at the
  end, 0 mixed 4-tick blocks). Ship-section roots 8,100 / 8,200 were inexact
  (8,100's route reached 10,190 but replays dead at 8,249 every time).
  Gates used: ship 7,000-9,200, wave 9,320-13,000, mini 11,000-11,150,
  normal 11,250-13,000.
- 2026-10-19 02:30: **Rauchkammer beaten.** Further roots 9,900 (-> 11,568),
  11,150 (-> 12,795) and 12,600 (-> the end, 0 mixed 4-tick blocks; roots
  9,800 / 11,300 / 12,350 replayed dead, 11,400 / 8,200 killed the player on
  spawn). `configs/demos/rk_start_end_4tick.npy` completes both the community
  copy and the original. demo_bc from the Bloodbath init (`--retain-until-x 0`)
  -> `runs/bc1-gd-rk-community` (reaches the end greedily), then on the
  original with `--extra-bcdata` -> `runs/bc2-gd-rk-community`: greedy
  **completes the original 146399247 3/3 and the community copy 3/3**
  (gdrl.evaluate, sticky 0). Ledger updated; watch preset `rk_beat`. Mostly
  search: PPO never got off 0% here. ~8.6 h, 8 chained searches of 20 min.
  Next level: GRIEF.

## GRIEF (2026-10-19)

- The original (149992717, length 56,065, ~190k objects) holds 33 StartPos of
  its own; RESET start_pos k spawns at each (`configs/starts/grief.yaml`,
  `configs/gd_grief.yaml`, init = the Rauchkammer policy). The popular
  community copies differ in 700-160,000 objects.
- Search bug fixed (`segment_search._candidates`): a plan cut at a death tick
  could end mid-block, so later inputs left the decision_ticks grid. That was
  where the mixed 4-tick blocks in the BB/RK routes came from.
- **GRIEF's opening needs 1-tick inputs.** From the level start, strict
  4-tick and 2-tick searches stalled at x ~460-500 (25 / 20 min; 3,000 random
  2-tick rollouts from x 377 never passed 456); a 1-tick search passed in a
  few segments. So the GRIEF policy will need `frame_skip: 1` (demo_bc reads
  it from the config). Searches run with `DT=1` (scratchpad `seg_gate.py`).
- 1-tick chain (logs/gr_dt1_<root>.npy, all replay alive from the level
  start): 845 (root 300) -> 1,270 (600; root 700 replayed dead) -> 2,071
  (1,150). ~800 units per hour.
