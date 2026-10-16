# Progress report: 2026-10-11 08:45 PDT

**Lost time:** the usage limit (05:00–08:40) stopped all agents, and the game sat idle from 05:01 because the PPO agent had paused training. Curriculum training was relaunched at 08:41.

**Doing:**
- Resumed the PPO agent (Stereo Madness: denser StartPos curriculum every 750 units, greedy evaluation from the level start).
- Resumed the demon agent (fixing the snapshot slot ring for the checkpoint search) and the M8 agent.
- Started the dashboard (Sonnet).

**Proposed:** `watchdog.ps1` restarts training whenever the game is idle, so a usage-limit cutoff doesn't leave it idle. It needs the owner's OK to run it as a scheduled task.
