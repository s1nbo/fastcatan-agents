# fastcatan-agents

Scripts for an MSc thesis (TU Berlin, DAI-Labor) on multi-agent RL in 4-player Catan with
player-to-player trading. The simulator is the separate `fastCatan` repo at `../fastcatan`.

**Research question:** when does trading benefit competing agents, and how does
competitive threat change their willingness and terms of cooperation? The four parts are
benefit, willingness (is the leader frozen out?), terms (does the leader pay worse
rates?), and stability.

## Rules

- **Keep it simple.** Plain single-file scripts, no package or framework. Ask before adding
  structure.
- **Clean slate.** Don't copy code or results from earlier experiments.
- **Leave fastCatan alone.** Never modify it from here. Missing features go under "Requests
  for fastCatan" below.
- **Stack.** No RLlib, JAX/JaxMARL, MuZero or Gym wrappers. Drive `fastcatan.BatchedEnv`
  directly with NumPy and torch.
- **Reward.** +1 for the winner, -1 for the others, 0 for everyone when a game hits the
  turn limit. Any shaping must be potential-based with Φ = public VP.
- **Matches.** A plays seats 0 and 2, B plays seats 1 and 3. Report per-seat win rate with
  a Wilson 95% CI against the 25% chance level.
- **Alpha-Beta benchmark.** Player trades are disabled for all seats; bank and port trades
  stay on. AB never initiates trades.
- **Experiments.** Seed everything, keep every checkpoint, and evaluate fixed checkpoints
  in a separate run.

## Setup

```bash
uv venv --python 3.12
uv pip install ../fastcatan ../fastcatan/viz torch numpy
uv pip install --reinstall --no-cache ../fastcatan    # after changing fastCatan
```

Don't install with `-e`. On this Mac the editable install breaks `import fastcatan`: the
venv's `.pth` files get the macOS hidden flag, and Python skips hidden `.pth` files.

## Files

- `ppo_demo.py`: minimal masked PPO. One network self-plays all four seats, with per-seat
  GAE. `record()` plays games one at a time on a `fastcatan.Env` and writes `.jsonl.gz`
  with `fastcatan_viz.GameLogWriter`, including the value per PPO move. Games in one file
  must be sequential, so batched games can't share a file. It records 4 self-play games
  before training and 4 after (same boards), plus 4 against random.
  `win_rate_vs_random()` plays 100 batched games without recording. Weights are saved
  to `ppo.pt`. View recordings with `fastcatan-viz`.

Planned: AlphaZero (the main agent), Q-learning, and an Alpha-Beta port from Catanatron.

## fastCatan notes (v2.2.0)

- 286 actions (`fc.action`) and 1089 observation floats. Observations are seat-relative:
  the observer is always slot 0.
- Batched reads fill NumPy arrays in place: `write_obs`, `write_obs_pov`,
  `write_masks_dense`, `write_actors`, `write_status`.
- `step` keeps finished games, so pass `NO_ACTION` for them. `step_autoreset` restarts
  them; read the result from `last_winner(i)`, where `NO_PLAYER` means the game hit the
  turn limit.
- The acting seat is often not the turn owner, for example during trade responses and
  discards.
- For search agents, copy a game with `env.snapshot(i)` and `Env.load_snapshot`.
  `enumerate_outcomes` gives the chance outcomes (dice and similar) of an action.
- `fastcatan.tournament`, `fc.play` and `step_n` don't exist in 2.x, despite older notes.

## Requests for fastCatan

- Runtime build info (compiler, flags, commit) for run logs.
- Batched VP reads (public and true) for shaping and metrics.

Compute: SLURM cluster with CUDA.
