# fastcatan-agents

Agents for an MSc thesis (TU Berlin, DAI-Labor) on multi-agent RL in 4-player Catan with
player trading, built on [fastCatan](https://github.com/s1nbo/fastcatan).

## Setup

Clone fastCatan next to this repo, then install everything (building fastCatan needs
CMake and a C++23 compiler):

```bash
git clone git@github.com:s1nbo/fastcatan.git ../fastcatan
uv venv --python 3.12
uv pip install ../fastcatan ../fastcatan/viz torch numpy
```

## Run

```bash
.venv/bin/python ppo_demo.py
```

The script trains PPO by self-play: one network plays all four seats. The default 49M
steps take about 25 minutes on an M-series Mac CPU. The settings are constants at the top
of the script.

It writes these files:

| File | Contents |
|---|---|
| `selfplay_start.jsonl.gz` | 4 games of the untrained network against itself |
| `selfplay_end.jsonl.gz` | 4 games of the trained network against itself, on the same boards |
| `vs_random.jsonl.gz` | 4 games against a random player, with PPO in seats 0 and 2 |
| `ppo.pt` | the trained weights |

At the end it also prints the win rate over 100 games against random. Those games are not
recorded.

## Watch the games

```bash
.venv/bin/fastcatan-viz
```

Drop a `.jsonl.gz` file onto the viewer. Each PPO move shows the network's value estimate:
the expected result for that seat, from -1 to +1.

## Planned

AlphaZero, Q-learning, Alpha-Beta baseline.

## License

MIT
