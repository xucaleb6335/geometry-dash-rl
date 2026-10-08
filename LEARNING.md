# Learning map

Every piece left for you is marked `LEARN(Mx)` in the code. List them with:

```bash
git grep -n "LEARN("
pytest -rs          # each SKIPPED line is a stub still to build
```

Work top to bottom inside each part. The C++ chain (M0 to M3) and your PPO
(M6) are independent, so you can switch between them. Week 1 is done when M3
and M6 pass their gates.

## Part 1: the Geode interface (C++)

| Milestone | Where | What you build | Done when |
| --- | --- | --- | --- |
| M0 | `mod/src/main.cpp` `RLPlayLayer::init` | Get the mod building with `geode build`, loading, and logging the level name. Fix any binding names that changed. | Mod in Geode's list; log line on level open |
| M1 | `mod/src/main.cpp` `processCommands` | Confirm it runs once per physics tick (count calls per second) | Calls/s match 240 |
| M1 | `mod/src/StateReader.cpp` `readState` | Player position, velocity, ground, gravity, mode; nearby object hitboxes and kinds | A 1 Hz log matches the screen |
| M2 | `mod/src/main.cpp` `destroyPlayer` | Decide which calls are real deaths, then call `Controller::onDeath` | No false deaths in a session |
| M2 | `mod/src/Controller.cpp` `onTick`, `handleReset` | Reset at the start of the next tick (never inside a hook), skip the respawn delay, reply with tick 0 | 500 unattended attempts, no crash |
| M2 | `mod/src/Controller.cpp` `handleStep` | Press/release with `handleButton(down, 1, true)` | A hard-coded jump rule plays the level |
| M3 | `mod/src/Controller.cpp` `pollClient`, `handleStep` | Lockstep: the game advances exactly `ticks` ticks per STEP and waits otherwise | `python -m gdrl.tools.bridge_check --soak 10000` passes all three checks |

Already done for you: `Bridge` (sockets, framing, Winsock), `Protocol.hpp`, the
HELLO handshake in `Controller`, and `mod/tools/standalone_server.cpp`, a
Geode-free server you can read as a reference for how replies should look.
`gdrl/env/mock_server.py` is the full reference behaviour, including terminal
states and errors.

## Part 2: environment and brain (Python)

| Milestone | Where | What you build | Turns green |
| --- | --- | --- | --- |
| M4 | (done) `gdrl/env/gd_env.py`, `mock_game.py`, `mock_server.py` | Read them: terminated vs. truncated, same-step autoreset, `final_obs` | `tests/test_env.py` (already passing) |
| M5 | `gdrl/features/encoder.py` `ObjectListEncoder.encode` | Player features + next k obstacles, relative and normalized | `tests/learn/test_encoder.py` |
| M5 | `gdrl/rewards/reward.py` `ProgressReward.__call__` | Progress, death penalty, completion bonus | `tests/learn/test_reward.py` |
| M6 | `gdrl/agents/networks.py` `layer_init`, `mlp` | Orthogonal init, Tanh MLP | `tests/learn/test_networks.py` |
| M6 | `gdrl/agents/ppo.py` `compute_gae` | GAE with terminated vs. truncated handled correctly | `tests/learn/test_gae.py` (hand-worked numbers in the docstring) |
| M6 | `gdrl/agents/ppo.py` `PPOAgent.__init__`, `act`, `update` | Separate policy/value nets, clipped surrogate, value loss, entropy, minibatches, grad clipping, LR annealing, debug metrics | `tests/learn/test_ppo.py`, then the gate below |
| M7 | `configs/gd_ppo.yaml`, `levels.yaml` | Build 3 to 5 short cube levels; train on the real game | Greedy policy beats level 1; 18/20 with `--sticky 0.1` |
| M8 | `gdrl/agents/baselines.py` `RuleBasedAgent`, `GeneticTimingAgent` | The baselines your PPO must beat | A results table |
| M8 | `gdrl/features/encoder.py` `GridEncoder`, `gdrl/agents/networks.py` `GridCNN` | Grid observation + CNN | `tests/learn/test_encoder.py`, `test_networks.py` |

### The M6 gate

```bash
python -m gdrl.train --config configs/cartpole_ppo.yaml --set seed=1   # then seeds 2 and 3
python -m gdrl.evaluate --run runs/<run-folder>                         # return_mean >= 475
pytest -m slow tests/learn/test_ppo.py                                  # the same gate, one seed
```

Debug signals to watch in TensorBoard: `train/approx_kl` (should stay under
about 0.02), `train/clipfrac`, `train/entropy` (falls slowly, not to 0 at
once), `train/explained_variance` (rises toward 1).

### Then the mock game

`configs/mock_ppo.yaml` trains your encoder + reward + PPO on the mock levels
(seeds 1 to 5) and `levels.yaml` lists held-out seeds. Every mock level is
beatable by a simple timing rule (see `scripted_jumper` in `tests/conftest.py`),
so a policy that cannot finish them has a bug or a blind spot in the encoder.

## Rules of thumb

- If a bug in it would teach you something about GD or RL, write it yourself.
  Otherwise generate it, read it once, move on.
- Don't edit the plumbing to make a learn test pass. If a test looks wrong,
  check the docstring numbers by hand first.
- One config change per experiment, and keep the run folder (it saves its
  config and metrics).
