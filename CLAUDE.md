# fastcatan-agents

Scripts for an MSc thesis (TU Berlin, DAI-Labor) on multi-agent RL in 4-player Catan with
player-to-player trading. The simulator is the separate `fastCatan` repo at `../fastcatan`.

**Research question:** when does trading benefit competing agents, and how does
competitive threat change their willingness and terms of cooperation? The four parts are
benefit, willingness (is the leader frozen out?), terms (does the leader pay worse
rates?), and stability.

## Rules

- **Keep it simple.** Flat scripts plus `common.py` and `league.py`. No package, framework or
  SLURM scripts. Settings are constants at the top of each file. Trainers take only
  `--name --seed --hours --no-trades --config` (`--config` = JSON overrides of the constants,
  e.g. from `hpsearch.py`). Ask before adding structure.
- **Clean slate.** Don't copy code or results from earlier experiments. Catanatron is
  GPL-3.0: use only its ideas and numbers, never its code.
- **Leave fastCatan alone.** Never modify it from here. Missing features go under "Requests
  for fastCatan" below.
- **Stack.** No RLlib, JAX/JaxMARL, MuZero or Gym wrappers. Drive `fastcatan.BatchedEnv`
  directly with NumPy and torch.
- **Reward.** +1 for the winner, -1 for the others, 0 for everyone when a game hits the
  turn limit. Any shaping must be potential-based with Φ = public VP.
- **Trade rules** (`common.TradeRules`) apply to every agent, in training and evaluation.
  - Trading is on or off per seat. A trade-off seat never builds or opens an offer, and it
    declines every offer it gets.
  - Per seat per turn, counted on the agent side: at most 10 trade-building actions and 3
    offers.
  - Bank and port trades are always allowed.
- **Budget.** Training runs are compared at equal wall-clock hours (`--hours`).
  `metrics.jsonl` logs `env_steps`, `hours` and, for AZ, `sim_steps`.
- **Matches.** A plays seats 0 and 2, B plays seats 1 and 3, then the seats are swapped on
  the same boards. Report the per-seat win rate with a Wilson 95% CI against 25%. For
  strength (E0/E6), use `--seats 1v3`: A alone against 3 B, in each seat in turn.
- **Benefit** is measured in mixed games: trade-on seats against trade-off seats.
- **Alpha-Beta benchmark.**
  - Player trades are off for all seats, and the evaluation has at least 1000 games.
  - AB is PIMC: K = 8 determinizations per move (also report K = 1). It never searches
    the true state.
- **AlphaZero search** is ISMCTS. It re-determinizes for every simulation, and leaves are
  evaluated from `write_obs(acting seat)` only.
- **Experiments.** Seed everything and keep every checkpoint. Evaluate fixed checkpoints in
  a separate run (`evaluate.py`), never during training beyond light monitoring.

## Setup

```bash
uv venv --python 3.12
uv pip install ../fastcatan ../fastcatan/viz torch numpy pytest
uv pip install --reinstall --no-cache ../fastcatan    # after changing fastCatan
.venv/bin/python -m pytest -q test_core.py
```

Don't install with `-e`. On this Mac the editable install breaks `import fastcatan`: the
venv's `.pth` files get the macOS hidden flag, and Python skips hidden `.pth` files.

## Files

| File | What it is |
|---|---|
| `common.py` | `observe`, `TradeRules` / `trade_mask`, `play()` (batched matches; with `log_path`, viewer recordings one game at a time), `RandomAgent`, `wins`, `wilson`, `run_dir`, `log`, `Trunk` (the body of every net: `WIDTH` × `DEPTH`, residual layers with LayerNorm) |
| `league.py` | `League`: a pool of frozen past learners, at most 20 (random eviction, random agent kept as anchor). PFSP weights (1−p)²+0.05. Per-slot lineups: either pure self-play or the learner in {0,2}/{1,3} against one pool opponent. Only learner seats train. |
| `ppo.py` | masked PPO plus the league. Per-seat GAE: each seat's next state is its own next decision. |
| `alphazero.py` | Gumbel-root ISMCTS AlphaZero for one CUDA GPU: `ACTORS` self-play processes (one CPU core each, network copy on the GPU) feed one GPU learner, which sends weights every `SYNC_EVERY` s. Search generators batch leaves into one forward pass. Value head = one value per relative seat. |
| `dqn.py` | optional Double DQN with per-seat n-step returns, plus the league. |
| `alphabeta.py` | PIMC Alpha-Beta: depth 2, paranoid, with expectation over `enumerate_outcomes`. Our own value function, using Catanatron's feature list and weights. |
| `evaluate.py` | `vs A B` (`--seats 2v2` or `1v3`, Wilson CI) and `crossplay` (Bradley-Terry Elo, intransitive triples). Results go to `results/eval.jsonl`. `--events x.fcev` writes one trade log per seating (`x.A02.fcev`, ...). Each log's header names the agent in every seat, and `meta.trades` has the per-seat trade flags. |
| `hpsearch.py` | two stages per algorithm: random search over `SPACE` at the default net size, then `--sizes` trains the winner at every `WIDTH` × `DEPTH`. Short trials, ranked by per-seat win rate vs `ab:1` (trades off), then vs random. Best settings go to `results/hpsearch_<algo>_best.json`. |
| `probe.py` | Q1/Q3/Q4: scripted good vs bad offers at real states, plus edits of the proposer's VP in the responder's observation at three offer ratios. AZ's lead probe uses its network. |
| `counterfactual.py` | Q2: at trades about to be confirmed, 2×N rollouts (confirm vs cancel, fresh dice). Gives the change in win probability for proposer, partner and outsiders. AZ rolls out with its network. |
| `metrics.py` | Q3/Q4/Q6 from trade logs: acceptance, offer rate and terms by lead; the lead slope with a game-bootstrap CI; reciprocity; acceptance per agent pair. Benefit comes from `results/eval.jsonl`. |
| `test_core.py` | 11 tests: per-seat GAE, trade rules, league, `enumerate_outcomes`, `determinize`, "AB never reads the true hidden state", MCTS, per-seating logs, `probs()` |
| `ppo_demo.py` | the minimal standalone PPO example |

- **Agent interface:** `actions, values = agent.act(env, idx, obs, mask, seat, rules)` and
  `probs = agent.probs(...)` (action distribution, for probes). Search agents snapshot `env`
  at slot `i` into their own scratch `fc.Env`.
- **Trade logs are for analysis.** They keep trade, game and END_TURN rows only (END_TURN rows
  give per-turn VP for offer rates), so they can't be replayed.
- **Workflow:**
  1. `hpsearch.py <algo>`, then `hpsearch.py <algo> --sizes`, for PPO, DQN and AZ.
  2. Train each best config for the same hours (`--config`).
  3. `evaluate.py crossplay` across the algorithms.
  4. Train the most promising one much longer.
- **Run outputs:** `runs/<name>/` holds `config.json` (constants, seed, fastCatan version,
  commit, dirty flag, compiler), `metrics.jsonl`, `ckpt_*.pt` and `games/*.jsonl.gz`.
- **Viewing games:** `.venv/bin/fastcatan-viz`.

## fastCatan notes (v2.2.1 local build, OBS_VERSION 1)

- 286 actions (`fc.action`) and 1089 observation floats. Observations are seat-relative:
  the observer is always slot 0.
  - `nodes`: 54×4×2 (settlement, city per relative seat).
  - `edges`: 72×4.
  - Resource order: brick, lumber, wool, grain, ore.
  - Dev-card order: knight, VP, road building, year of plenty, monopoly.
- Batched reads fill NumPy arrays in place: `write_obs`, `write_obs_pov`,
  `write_masks_dense`, `write_actors`, `write_status`.
- `step` keeps finished games, so pass `NO_ACTION` for them. `step_autoreset` restarts
  them; read the result from `last_winner(i)`, where `NO_PLAYER` means the game hit the
  turn limit.
- The acting seat is often not the turn owner, for example during trade responses and
  discards.
- `Env.determinize(pov, seed)` works only on a scratch `Env` restored with
  `load_snapshot`. Its result depends only on what `pov` sees plus `seed`.
- `Env.enumerate_outcomes(action)` returns (probability, snapshot) pairs for dice, dev
  draws and steals.
- `fastcatan.tournament`, `fc.play` and `step_n` don't exist in 2.x, despite older notes.

## Requests for fastCatan

- Runtime build info (compiler, flags, commit) for run logs.
- Batched VP reads (public and true) for shaping and metrics.

Compute: SLURM cluster with CUDA (jobs are started by hand; no SLURM scripts in this repo).
