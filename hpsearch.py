"""Hyperparameter search in two stages: short training runs, ranked by strength.

    python hpsearch.py ppo --trials 16 --hours 0.5 [--parallel 2]   # stage 1: SPACE at the default size
    python hpsearch.py ppo --sizes --hours 1 [--parallel 2]         # stage 2: every WIDTH x DEPTH in SIZES
    python ppo.py --name ppo_best --hours 8 --config results/hpsearch_ppo_best.json

Stage 1 trains the algorithm (ppo, dqn or az) with random draws from SPACE, every other constant
unchanged. Stage 2 trains the stage-1 winner once per network size in SIZES. Trades are on. Each
trial's last checkpoint then plays Alpha-Beta K=1 (2v2, trades off: the benchmark setting) and
random; trials are ranked by the per-seat win rate against Alpha-Beta, ties broken by the rate
against random. Rows go to results/hpsearch_<algo>.jsonl and the winning settings to
results/hpsearch_<algo>_best.json (after stage 2: the stage-1 settings plus the size).

- The trials of a stage share one wall-clock budget, so short trials favour small, fast nets:
  give stage 2 more hours than stage 1.
- Stage 1 runs at the default size; a much bigger net may want a lower LR.
- Every trial uses the same seed, so confirm the winner with more seeds before trusting small
  differences.
"""

import argparse
import importlib
import json
import multiprocessing as mp
import time

import numpy as np

from common import ROOT
from evaluate import load, match, per_seat

MODULES = {"ppo": "ppo", "dqn": "dqn", "az": "alphazero"}
SPACE = {  # ("log", lo, hi) = log-uniform, a tuple of values = one of them
    "ppo": dict(LR=("log", 1e-4, 1e-3), ENT_COEF=("log", 1e-3, 3e-2), GAMMA=(0.995, 0.999, 0.9995),
                LAMBDA=(0.9, 0.95, 0.98), CLIP=(0.1, 0.2, 0.3), EPOCHS=(2, 4, 8), NUM_ENVS=(128, 256, 512),
                P_SELFPLAY=(0.3, 0.5, 0.8)),
    "dqn": dict(LR=("log", 3e-5, 1e-3), N_STEP=(3, 5, 10), GAMMA=(0.995, 0.999),
                EPS_STEPS=(2_000_000, 5_000_000, 10_000_000), TARGET_EVERY=(500, 2_000, 5_000),
                BATCH=(256, 512, 1024), P_SELFPLAY=(0.3, 0.5, 0.8)),
    "az": dict(N_SIMS=(16, 32, 64), C_PUCT=(1.0, 1.5, 2.5), LR=("log", 3e-4, 3e-3), REPLAY_RATIO=(2, 4, 8),
               TOP_K=(8, 16), P_SELFPLAY=(0.5, 0.8, 1.0)),
}
SIZES = dict(WIDTH=(256, 512, 1024), DEPTH=(2, 3, 5, 7))  # stage 2: every combination
EVAL_GAMES, EVAL_SEED, TRIAL_SEED = 400, 7_000, 1


def draw(space, rng):
    return {k: float(np.exp(rng.uniform(np.log(v[1]), np.log(v[2])))) if v[0] == "log"
            else v[int(rng.integers(len(v)))] for k, v in space.items()}


def trial(algo, search, i, config, hours):
    """Train one setting, evaluate its last checkpoint, append the result row."""
    name = f"hp_{algo}_{search}_{i:02d}"
    importlib.import_module(MODULES[algo]).train(name, TRIAL_SEED, hours, True, config)
    ckpt = str(sorted((ROOT / "runs" / name).glob("ckpt_*.pt"))[-1])
    agent = load(ckpt)
    ab, rnd = match(agent, load("ab:1"), EVAL_GAMES, EVAL_SEED, False, False), match(agent, load("random"), EVAL_GAMES, EVAL_SEED)
    row = dict(algo=algo, search=search, trial=i, config=config, hours=hours, ckpt=ckpt,
               vs_ab1=per_seat(ab[0], ab[3])[0], vs_random=per_seat(rnd[0], rnd[3])[0])
    with open(ROOT / "results" / f"hpsearch_{algo}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"trial {i}: vs AB {row['vs_ab1']:.3f}  vs random {row['vs_random']:.3f}  {config}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("algo", choices=list(MODULES))
    p.add_argument("--trials", type=int, default=16, help="stage 1 only")
    p.add_argument("--sizes", action="store_true", help="stage 2: every network size on the stage-1 winner")
    p.add_argument("--hours", type=float, default=0.5, help="training budget per trial")
    p.add_argument("--parallel", type=int, default=1, help="trials at the same time (they share the GPU)")
    p.add_argument("--seed", type=int, default=0, help="seed of the random draws")
    args = p.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    search, rng = time.strftime("%Y%m%d-%H%M%S"), np.random.default_rng(args.seed)
    best = ROOT / "results" / f"hpsearch_{args.algo}_best.json"
    if args.sizes:
        if not best.exists():
            p.error(f"{best} is missing: run stage 1 first")
        base = json.loads(best.read_text())
        configs = [dict(base, WIDTH=w, DEPTH=d) for w in SIZES["WIDTH"] for d in SIZES["DEPTH"]]
    else:
        configs = [draw(SPACE[args.algo], rng) for _ in range(args.trials)]

    ctx = mp.get_context("spawn")  # each trial in its own process (AZ trials start their own actors)
    for first in range(0, len(configs), args.parallel):
        procs = [ctx.Process(target=trial, args=(args.algo, search, i, configs[i], args.hours))
                 for i in range(first, min(first + args.parallel, len(configs)))]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    rows = [r for r in map(json.loads, open(ROOT / "results" / f"hpsearch_{args.algo}.jsonl")) if r["search"] == search]
    rows.sort(key=lambda r: (r["vs_ab1"], r["vs_random"]), reverse=True)
    for r in rows:
        print(f"vs AB {r['vs_ab1']:.3f}  vs random {r['vs_random']:.3f}  {r['config']}")
    best.write_text(json.dumps(rows[0]["config"], indent=2) + "\n")
    print(f"best settings -> {best}")


if __name__ == "__main__":
    main()
