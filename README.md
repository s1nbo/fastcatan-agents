# fastcatan-agents

Agents for an MSc thesis (TU Berlin, DAI-Labor) on multi-agent RL in 4-player Catan with
player trading, built on [fastCatan](https://github.com/s1nbo/fastcatan).

## Setup

Clone fastCatan next to this repo, then install everything (building fastCatan needs
CMake and a C++23 compiler):

```bash
git clone git@github.com:s1nbo/fastcatan.git ../fastcatan
uv venv --python 3.12
uv pip install ../fastcatan torch numpy pytest matplotlib    # matplotlib: hpsearch figures
.venv/bin/python -m pytest -q test_core.py
```

## Train

Every trainer learns by self-play against a league of its own frozen past versions, and
stops after `--hours` of wall-clock time:

```bash
.venv/bin/python ppo.py       --name ppo_on_s1 --seed 1 --hours 2
.venv/bin/python alphazero.py --name az_on_s1  --seed 1 --hours 2
.venv/bin/python dqn.py       --name dqn_on_s1 --seed 1 --hours 2
```

- `--no-trades` blocks player trades for every seat.
- `--config file.json` overrides the constants at the top of the file.
- AlphaZero runs `ACTORS` self-play processes (one CPU core each) that share the GPU with
  the learner. Set `ACTORS` to about the number of free cores.
- `--resume` continues `runs/<name>/` after a crash from its `resume.pt` (written every 10
  minutes), with the run's own settings, or starts the run if it is new. With `--hours` it
  sets a new total, so a finished run can also be trained longer. For long runs:

  ```bash
  until .venv/bin/python ppo.py --name ppo_on_s1 --seed 1 --hours 72 --resume; do sleep 60; done
  ```

Each run writes `runs/<name>/`:

| File | Contents |
|---|---|
| `config.json` | settings, seed, fastCatan version and commit |
| `metrics.jsonl` | one row per update (AZ: per minute): steps, hours, losses, steps per turn, win rate vs random |
| `ckpt_<env steps>.pt` | weights every 100M env steps and at the end, all kept |
| `pool/<id>.pt` | the league's current snapshots (deleted when evicted) |
| `resume.pt` | everything `--resume` needs, every 10 minutes |
| `games/*.jsonl.gz` | recorded self-play games for the viewer |

## Pick and compare settings

```bash
until .venv/bin/python hpsearch.py ppo --search s1 --trials 16 --hours 1; do sleep 60; done   # also: dqn, az
until .venv/bin/python hpsearch.py ppo --search s2 --sizes --hours 2; do sleep 60; done       # width x depth
.venv/bin/python hpsearch.py ppo --search s1 --report     # redo the tables and figures
.venv/bin/python ppo.py --name ppo_best --hours 72 --resume --config results/hpsearch_ppo_best.json
.venv/bin/python evaluate.py crossplay runs/ppo_best/ckpt_X.pt runs/dqn_best/ckpt_X.pt runs/az_best/ckpt_X.pt
```

- **Stage 1** trains trial 00 with the hand-set defaults and 15 random draws from `SPACE`, at the
  default network size (1024 wide, 5 deep). **Stage 2** trains the stage-1 winner at every size in
  `SIZES` (1024 or 2048 wide, 3, 5 or 7 deep: only large nets, since the goal is to beat
  Alpha-Beta). `P_SELFPLAY` is fixed at 0.8 in every trainer and not searched.
- Every trial trains for the same hours with the same seed, one at a time. Its final checkpoint
  then plays 400 games (2v2, seats swapped, same boards) against trial 00 (the ranking score),
  Alpha-Beta K=1 with trades off, and random; 200 games against itself measure its trading.
- A search is resumable: rerun the same command after a crash (the `until` loop does it).
- Everything for the write-up is in `results/hpsearch_<algo>_<search>/`: `plan.json` (protocol,
  configs, versions, machine, `code.diff` if uncommitted), `trials.jsonl` (one row per trial),
  `results.csv`, `report.md` and `figures/` (ranking, sensitivity or size heatmaps, learning
  curves, trading; PDF and PNG). The `hpsearch.py` docstring has the full protocol.
- The winner's settings go to `results/hpsearch_<algo>_best.json`. Then train the most promising
  algorithm for much longer. The defaults are set for multi-day runs: league snapshots every 20k
  finished games in every trainer, kept checkpoints every 100M env steps.

## Evaluate and analyse

```bash
# strength: one agent against three Alpha-Beta, trades off (chance 25%)
.venv/bin/python evaluate.py vs runs/a/ckpt_X.pt ab --seats 1v3 --trades-a off --trades-b off
# benefit: traders against trade-masked clones of the same checkpoint
.venv/bin/python evaluate.py vs runs/a/ckpt_X.pt runs/a/ckpt_X.pt --trades-a on --trades-b off
# willingness, terms, reciprocity from a logged self-play match
.venv/bin/python evaluate.py vs runs/a/ckpt_X.pt runs/a/ckpt_X.pt --events results/a.fcev
.venv/bin/python metrics.py results/a.A*.fcev --name a
# good vs bad offers, and acceptance as only the proposer's VP changes
.venv/bin/python probe.py runs/a/ckpt_X.pt
# per-trade counterfactuals: who gains from a trade
.venv/bin/python counterfactual.py runs/a/ckpt_X.pt
```

`ab` is the Alpha-Beta baseline. It averages 8 resampled versions of the hidden cards per
move (`ab:1` uses one sample).

All results are appended to `results/*.jsonl`.

## Watch games

```bash
.venv/bin/fastcatan view
```

Drop a `.jsonl.gz` file from `runs/<name>/games/` onto the viewer.

## License

MIT
