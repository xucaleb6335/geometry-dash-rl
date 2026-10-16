# Progress report: 2026-10-12 21:35 PDT

**Stereo Madness is beaten.** The greedy PPO policy (checkpoint
`results/sm_beat_update_001170.pt`) completes the original level from the
start 3/3 times, and 5/10 with 10% sticky actions. 4.8M env steps, 14,445
attempts, 2.4 h of training on this setup. Ledger rows in
`holistic_progress/level_beats.md`.

What got it there (all driven by the episode data):
1. Community StartPos copy (141625101) + bridge v2 RESET start_pos; fixed
   StartPos respawn (cached checkpoint) and completion detection.
2. Encoder `focus: obstacles` (floor blocks hid spikes) and a terrain
   profile (pit spikes hid slab edges): greedy 11% -> 19% -> 91% -> beat.
3. CPU acting (3x faster), horizon truncation, stuck-start exploration.

Open: the checkpoint search still disagrees with clean replays on the real
game (needed for the extreme demons). Next: Bloodbath StartPos copy.
