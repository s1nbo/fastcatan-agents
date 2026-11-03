# fastcatan-agents

Agents for an MSc thesis (TU Berlin, DAI-Labor) on multi-agent RL in 4-player Catan with
player trading, built on [fastCatan](https://github.com/s1nbo/fastcatan).

## Setup

Clone fastCatan next to this repo, then install everything (building fastCatan needs
CMake and a C++23 compiler):

```bash
git clone git@github.com:s1nbo/fastcatan.git ../fastcatan
uv venv --python 3.12
uv pip install ../fastcatan ../fastcatan/viz torch numpy pytest
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

Each run writes `runs/<name>/`:

| File | Contents |
|---|---|
| `config.json` | settings, seed, fastCatan version and commit |
| `metrics.jsonl` | one row per update (AZ: per minute): steps, hours, losses, steps per turn, win rate vs random |
| `ckpt_*.pt` | every checkpoint |
| `games/*.jsonl.gz` | recorded self-play games for the viewer |

## Pick and compare settings

```bash
.venv/bin/python hpsearch.py ppo --trials 16 --hours 0.5 --parallel 2    # also: dqn, az
.venv/bin/python hpsearch.py ppo --sizes --hours 1 --parallel 2           # network width x depth
.venv/bin/python ppo.py --name ppo_best --hours 4 --config results/hpsearch_ppo_best.json
.venv/bin/python evaluate.py crossplay runs/ppo_best/ckpt_X.pt runs/dqn_best/ckpt_X.pt runs/az_best/ckpt_X.pt
```

Stage 1 searches the training settings at the default network size (256 wide, 2 deep). Stage 2
trains the winner at every size in `SIZES` (up to 1024 wide, 7 deep) and adds the best size to
`results/hpsearch_<algo>_best.json`. Then train the most promising algorithm for much longer.

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
.venv/bin/fastcatan-viz
```

Drop a `.jsonl.gz` file from `runs/<name>/games/` onto the viewer.

## License

MIT
