# Progress report: 2026-10-11 04:15 PDT

**Doing:**
- The PPO agent replaced the stalled Stereo Madness run with `gd-curriculum-sm`: a StartPos reverse curriculum (`gdrl/curriculum.py`, `configs/starts/`) plus greedy evaluations from the level start. It has trained 270k steps; the first completions come from late start points.
- The demon agent built protocol v2 snapshots, `gdrl/search/` and a demon-like mock level (`mock_demon.py`), with tests.

**M8:** the ledger has mock beats.
- Rule: first try on every level.
- Genetic: 160–992 attempts per level.
- PPO with the object list: 2.5k–3.7k episodes to its first completion.
- PPO with the grid: 1.3k–4.6k episodes.

**Next:** a full-level greedy beat of Stereo Madness.
