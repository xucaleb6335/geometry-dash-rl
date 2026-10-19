# Geometry Dash Reinforcement Learning Agent: project summary

A reinforcement-learning agent that plays Geometry Dash through a custom game
mod, and has beaten two of the game's hardest levels from the level start.

**Stack:** Python, PyTorch, C++ (Geode mod SDK), TCP sockets, Gymnasium-style
API, vanilla JS (dashboard and network visualizer).

**One line:** Built a reinforcement-learning agent that beat two of Geometry
Dash's hardest levels from the start, using a C++ game mod interface, PPO with a
start-position curriculum, and search-guided behaviour cloning.

## Highlights

### Systems and infrastructure

- Built a lockstep training interface for Geometry Dash running 970 steps/s
  (~16x real time) on a laptop: a C++ game mod that freezes the game clock and
  advances physics only on agent commands, a Gymnasium-style Python environment
  over a versioned TCP protocol, and sub-millisecond state save/load used for
  checkpoint search.
- Tripled training throughput (300 to 970 steps/s) by profiling the agent-game
  loop and splitting per-step inference (CPU: batch-1 GPU launch and transfer
  overhead, 1.8 ms per step, exceeded the 0.7 ms game step) from batched PPO
  updates (GPU).
- Ran unattended training for 140+ hours on one level with a supervisor script
  that resumes from the latest checkpoint after crashes, re-downloads levels
  and keeps the game window active.

### Machine learning

- Implemented PPO from scratch and validated it with a perfect 500/500 CartPole
  score on 3/3 seeds, and 83-97% completion on unseen levels of a procedurally
  generated mock game (with sticky-action noise 0.1).
- Raised the agent's timing accuracy from 7.5% to 100% of required jumps by
  finding that an MLP could not resolve a ~6-unit jump window from raw
  distances, and adding Gaussian distance encodings to the observation encoder.
- Beat Stereo Madness in 2.4 hours (~14k attempts) after plain training stalled
  at 19%, with a start-position curriculum, a terrain-profile encoder and
  exploration targeted at practice starts stuck at 0%.
- Beat the extreme demons Bloodbath (about 299k attempts) and Rauchkammer
  (8.6 hours) from the level start, 3/3 greedy completions each, by combining
  PPO with a checkpoint beam search that finds frame-precise routes, then
  behaviour-cloning those routes into the policy with a cloning penalty during
  PPO so later training did not forget them.
- Stopped repeated regressions of cloned routes with an "eval ratchet" (each new
  best run joins the cloning anchor) and hard-example oversampling (a single
  mispredicted decision out of 16k was rarely sampled).

### Debugging and analysis

- Removed a run-to-run nondeterminism (identical inputs diverged between
  attempts): the observation included visual opacity left behind by earlier
  attempts; after excluding it, the agent's view matched object for object
  along the whole route.
- Got past multi-hour plateaus at 45% and 99% by tracing them to routes that
  skipped a mode or size portal, and adding mode and size gates to the search.
- Fixed a search bug that put mixed-timing input blocks into found routes, and
  showed with controlled 4-, 2- and 1-tick searches that GRIEF needs
  single-frame input timing.

### Tooling

- Built a training dashboard (per-level progress, learning curves, death
  heatmaps, success per practice start, one-click in-game replays): the main
  tool for diagnosing stalls.
- Built a neural-network visualizer that replays the policy decision by
  decision: the game scene, all 234 inputs, both 128-unit hidden layers and the
  outputs lighting up, with knock-out tests that name the input driving each
  jump.
- 173 automated tests.

## How the agent decides

234 inputs (player state, the 8 nearest obstacles, a fine timing code, a
17-column terrain profile, position) go through two hidden layers of 128 tanh
neurons to two outputs, Release and Jump. The agent takes the more likely
action every 4 physics ticks (60 decisions per second); during training it
samples from the probabilities instead, which is how it explores. A separate
network of the same shape (the critic) estimates the value of each state and is
used only during training.

## Results

| Level | Difficulty | Result | Method |
|---|---|---|---|
| Stereo Madness | easy | beaten, 3/3 greedy | PPO + start-position curriculum |
| Bloodbath | extreme demon | beaten, 3/3 greedy | PPO + search-guided behaviour cloning |
| Rauchkammer | extreme demon | beaten, 3/3 greedy | search-guided behaviour cloning |
| GRIEF | extreme demon | 8.7% route found (in progress) | 1-tick checkpoint search |

Caveats: each policy is trained on one level, so the results show the pipeline
can master a level, not general skill. On the extreme demons, plain PPO stalled
(Bloodbath) or stayed at 0% (Rauchkammer); the routes came from search and
were cloned into the policy, which does not complete them with sticky actions.

Details: `holistic_progress/level_beats.csv` (beat ledger), `python -m gdrl.dashboard` (dashboard, `/viz` for the network
visualizer).
