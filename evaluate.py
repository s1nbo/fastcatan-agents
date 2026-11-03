"""Evaluate fixed checkpoints (a separate run, never during training).

    python evaluate.py vs runs/a/ckpt_01000.pt random --games 1000
    python evaluate.py vs runs/a/ckpt_01000.pt ab --games 1000 --trades-a off --trades-b off
    python evaluate.py vs runs/a/ckpt_01000.pt runs/b/ckpt_01000.pt --trades-a on --trades-b off
    python evaluate.py crossplay runs/a/ckpt_*.pt --games 200

Agents: `random`, `ab` (K = 8 determinizations), `ab:<K>`, or a checkpoint path (PPO, AZ, DQN).
`vs` plays the same boards in every seating: --seats 2v2 (default) puts A in seats 0+2, then
1+3; --seats 1v3 puts A alone in each seat in turn (strength against 3 x B). It reports A's
per-seat win rate with a Wilson 95% interval (chance 25%). Truncated games count for nobody.
Rows are appended to results/eval.jsonl; --events saves one trade log per seating.
`crossplay` plays every pair and fits Elo ratings (Bradley-Terry). Ratings that rise with the
training step mean no cycling.
"""

import argparse
import importlib
import json
import time
from pathlib import Path

import fastcatan as fc
import numpy as np
import torch

from common import ROOT, RandomAgent, play, wilson, wins

AGENTS = {"ppo": ("ppo", "PPOAgent"), "az": ("alphazero", "AZAgent"), "dqn": ("dqn", "DQNAgent")}


def load(spec, seed=0):
    if spec == "random":
        return RandomAgent(seed)
    if spec.split(":")[0] == "ab":
        from alphabeta import K, AlphaBeta

        return AlphaBeta(k=int(spec.split(":")[1]) if ":" in spec else K, seed=seed)
    module, cls = AGENTS[torch.load(spec, map_location="cpu").get("algo", "ppo")]
    return getattr(importlib.import_module(module), cls).load(spec)


SEATINGS = {"2v2": ([0, 2], [1, 3]), "1v3": ([0], [1], [2], [3])}


def match(a, b, games, seed, trades_a=True, trades_b=True, seats="2v2", events=None, labels=("A", "B")):
    """A against B on the same boards in every seating of `seats`.

    2v2: A in seats 0+2, then 1+3. 1v3: A alone in each seat in turn, B in the other three.
    events: a .fcev path; each seating writes its own file (<path>.A02.fcev, ...) whose header
    names the agent in every seat. Returns (A wins, B wins, truncated, games played)."""
    a_wins = b_wins = truncated = played = 0
    n = games // len(SEATINGS[seats])
    for a_seats in SEATINGS[seats]:
        b_seats = [s for s in range(4) if s not in a_seats]
        agents = [a if s in a_seats else b for s in range(4)]
        on = [trades_a if s in a_seats else trades_b for s in range(4)]
        if events:
            path = Path(events).with_suffix(".A" + "".join(map(str, a_seats)) + ".fcev")
            names = [labels[0] if s in a_seats else labels[1] for s in range(4)]
            with fc.events.EventWriter(path, agents=names, meta=dict(trades=on)) as writer:
                status = play(agents, n, seed, on, events=writer)
        else:
            status = play(agents, n, seed, on)
        a_wins, b_wins = a_wins + wins(status, a_seats), b_wins + wins(status, b_seats)
        truncated, played = truncated + int(status[:, 4].sum()), played + n
    return a_wins, b_wins, truncated, played


def per_seat(a_wins, played, seats="2v2"):
    """A's per-seat win rate and Wilson 95% interval (one trial per game, divided by A's seats)."""
    k = len(SEATINGS[seats][0])
    lo, hi = wilson(a_wins, played)
    return a_wins / played / k, (lo / k, hi / k)


def elo(W, iters=1000):
    """Bradley-Terry strengths from W[i, j] = games i won against j, as Elo points."""
    W = W + 0.5 * (1 - np.eye(len(W)))  # half a win each way keeps every rating finite
    N = W + W.T
    p = np.ones(len(W))
    for _ in range(iters):
        p = W.sum(1) / (N / (p[:, None] + p[None, :])).sum(1)
        p /= np.exp(np.log(p).mean())
    return 400 * np.log10(p)


def save(row):
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "eval.jsonl", "a") as f:
        f.write(json.dumps(dict(row, fastcatan=fc.__version__, date=time.strftime("%Y-%m-%d %H:%M"))) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["vs", "crossplay"])
    p.add_argument("agents", nargs="+")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026, help="seed of the evaluation boards")
    p.add_argument("--trades-a", choices=["on", "off"], default="on", help="for crossplay: every seat")
    p.add_argument("--trades-b", choices=["on", "off"], default="on")
    p.add_argument("--seats", choices=list(SEATINGS), default="2v2", help="1v3: A alone against three B")
    p.add_argument("--events", help="save trade event logs (<path>.A02.fcev, ...) for metrics.py")
    args = p.parse_args()
    trades_a, trades_b = args.trades_a == "on", args.trades_b == "on"

    if args.mode == "vs":
        a, b = load(args.agents[0], seed=1), load(args.agents[1], seed=2)
        a_wins, b_wins, truncated, played = match(a, b, args.games, args.seed, trades_a, trades_b, args.seats,
                                                  args.events, labels=args.agents)
        rate, (lo, hi) = per_seat(a_wins, played, args.seats)
        print(f"A per-seat win rate {rate:.3f} [{lo:.3f}, {hi:.3f}] (chance 0.250, {args.seats}) | "
              f"A {a_wins} B {b_wins} truncated {truncated} of {played} games")
        save(dict(mode="vs", a=args.agents[0], b=args.agents[1], trades_a=trades_a, trades_b=trades_b,
                  seats=args.seats, games=played, seed=args.seed, a_wins=a_wins, b_wins=b_wins,
                  truncated=truncated, a_per_seat=rate, ci95=[lo, hi]))
    else:
        agents = [load(spec, seed=i) for i, spec in enumerate(args.agents)]
        n = len(agents)
        W = np.zeros((n, n))
        games = args.games
        for i in range(n):
            for j in range(i + 1, n):
                W[i, j], W[j, i], _, _ = match(agents[i], agents[j], games, args.seed, trades_a, trades_a)
        ratings = elo(W)
        share = W / np.maximum(W + W.T, 1)
        cycles = sum(1 for i in range(n) for j in range(n) for k in range(n) if len({i, j, k}) == 3
                     and share[i, j] > 0.5 and share[j, k] > 0.5 and share[k, i] > 0.5) // 3
        for spec, r, w in zip(args.agents, ratings, W.sum(1)):
            print(f"{r:7.0f}  {int(w):5d} wins  {spec}")
        print(f"intransitive triples (A>B>C>A): {cycles}")
        save(dict(mode="crossplay", agents=args.agents, games_per_pair=games, seed=args.seed, trades=trades_a,
                  wins=W.tolist(), elo=ratings.tolist(), intransitive_triples=cycles))


if __name__ == "__main__":
    main()
