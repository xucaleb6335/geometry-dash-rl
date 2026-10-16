# Progress report: 2026-10-11 04:45 PDT

**Doing:**
- Stereo Madness curriculum PPO resumed in a new run at 04:36 (655k steps in total). The greedy evaluation from the level start is still weak (about 2%). The agent is tuning the start-point weighting so the curriculum works back to the start.
- The demon agent is running checkpoint search on the hard mock demon (`demon-search-mock`), then distilling the found input sequences into a policy (`demon-distill-mock`).

**Next:** M8 table; give the demon stack the game after the Stereo Madness beat; dashboard at about 05:53.
