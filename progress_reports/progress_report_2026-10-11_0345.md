# Progress report: 2026-10-11 03:45 PDT

**Doing:**
- Real-game PPO on Stereo Madness ran unattended for 2.5 h: 3.8M steps at about 450 steps/s, with crash recovery through gdctl.
- It **plateaued at 19.4%**: every episode was identical and entropy collapsed to 0.11. An agent is diagnosing the killer obstacle. Next, it fixes the encoder coverage or exploration, adds an entropy floor and a StartPos reverse curriculum, then restarts.

**Demon stack (resumed):** protocol v2 save/load state, for search segment by segment over press timings, then distilled into PPO.

**M8 (Sonnet):** the grid encoder gets sub-block x resolution and coverage fill; the comparison table is being finished.
