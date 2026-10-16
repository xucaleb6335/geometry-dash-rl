# Progress report: 2026-10-11 00:53

**Doing:**
- Fixed CUDA (`CUDA_MODULE_LOADING=LAZY`).
- An agent is building `gdctl` and a control mod, so agents can launch GD, build mods, download levels and add StartPos objects.
- An M8 baseline comparison is running on the mock, the first training of the grid+CNN path.

**Model and regime:** an extreme-demon agent is designing a hybrid approach:
- deterministic checkpoint search (save/restore state, beam search over press timings, segment by segment)
- PPO with a reverse curriculum from StartPos segments, plus richer multi-mode object features
- distilling search solutions into the policy (behaviour cloning, DAgger)

**Next:** hand the game to the demon agent, then train on Bloodbath first. Dashboard deferred about 5 h.
