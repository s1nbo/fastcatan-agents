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

The script trains PPO by self-play: one network plays all four seats. It then plays 100
games against a random player, with PPO in seats 0 and 2, and records them to
`eval_games.fcev`. The settings are constants at the top of the script.

## Watch the games

```bash
.venv/bin/fastcatan-viz-convert eval_games.fcev --finished --limit 20
.venv/bin/fastcatan-viz
```

Drop the converted `eval_games.jsonl.gz` onto the viewer. PPO plays P0 and P2.

## Planned

AlphaZero, Q-learning, Alpha-Beta baseline.

## License

MIT
