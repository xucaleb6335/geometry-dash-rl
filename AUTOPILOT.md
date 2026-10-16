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
