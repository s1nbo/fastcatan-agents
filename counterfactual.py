"""Per-trade counterfactuals (question 2): who gains from a trade?

    python counterfactual.py runs/a/ckpt_01000.pt --trades 200 --rollouts 200

From the agent's own self-play, take moments where a proposer is about to confirm a trade. Each
moment is copied 2 x ROLLOUTS times: half confirm the trade, half cancel it, and every copy is
played to the end with fresh dice and the same agent in every seat. The difference in win rate
is the trade's effect on the proposer, on the partner and on the two outsiders. Search agents
play the rollouts with their network alone (a full search per move would take days).
Rows go to results/counterfactual.jsonl: the summary and every trade with its context.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy/torch/fastcatan load (see common.py)

import argparse
import copy
import json

import fastcatan as fc
import numpy as np
import torch

from common import ROOT, A, TradeRules, run
from evaluate import load

rng = np.random.default_rng(0)


def collect(agent, n):
    """Trades about to be confirmed, with the trade-rule counters of that moment."""
    found, seed = [], 0
    while len(found) < n:
        rules = TradeRules(64, True)

        def grab(env, obs, mask, seat, status, actions):
            for i in np.flatnonzero((actions >= A.TRADE_CONFIRM_BASE) & (actions < A.TRADE_CONFIRM_BASE + 4)):
                if rng.random() < 0.25:  # spread over games
                    found.append(dict(snapshot=env.snapshot(int(i)), proposer=int(seat[i]),
                                      partner=int(actions[i] - A.TRADE_CONFIRM_BASE), adds=rules.adds[i].copy(),
                                      offers=rules.offers[i].copy(), turn=int(rules.turn[i])))
            return len(found) >= n

        env = fc.BatchedEnv(64, seed=5_000 + seed)
        env.reset()
        run([agent] * 4, env, rules, on_step=grab)
        seed += 1
    return found[:n]


def effect(agent, trade, k, seed):
    """Win rate per seat with the trade minus without it, over k rollouts each."""
    env = fc.BatchedEnv(2 * k, seed=seed)
    env.load_snapshots(np.repeat(np.frombuffer(trade["snapshot"], np.uint8)[None], 2 * k, axis=0))
    env.reseed(np.arange(2 * k, dtype=np.uint64) + np.uint64(seed) * np.uint64(1_000_003))  # fresh dice per copy
    first = np.r_[np.full(k, A.TRADE_CONFIRM_BASE + trade["partner"]), np.full(k, A.TRADE_CANCEL)].astype(np.uint32)
    env.step(first, np.zeros(2 * k, np.uint8))
    rules = TradeRules(2 * k, True)
    rules.adds[:], rules.offers[:], rules.turn[:] = trade["adds"], trade["offers"], trade["turn"]
    won = run([agent] * 4, env, rules)[:, 1][:, None] == np.arange(4)
    return won[:k].mean(0) - won[k:].mean(0)


def context(trade):
    env = fc.Env()
    env.load_snapshot(trade["snapshot"])
    vp = [env.player_vp_public(s) for s in range(4)]
    lead = lambda s: vp[s] - max(v for t, v in enumerate(vp) if t != s)
    return dict(turn=env.turn_count, proposer_lead=lead(trade["proposer"]), partner_lead=lead(trade["partner"]),
                gives=sum(env.trade_give(r) for r in range(5)), gets=sum(env.trade_want(r) for r in range(5)))


def ci(x):
    boot = x[rng.integers(len(x), size=(1000, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent")
    p.add_argument("--trades", type=int, default=200)
    p.add_argument("--rollouts", type=int, default=200, help="per branch")
    args = p.parse_args()
    torch.manual_seed(0)  # PPO samples its moves with torch's RNG
    agent = load(args.agent)
    if getattr(agent, "n_sims", 0):  # AlphaZero: roll out with the network alone
        agent = copy.copy(agent)
        agent.n_sims = 0

    rows = []
    for t, trade in enumerate(collect(agent, args.trades)):
        d = effect(agent, trade, args.rollouts, seed=t)
        outsiders = [s for s in range(4) if s not in (trade["proposer"], trade["partner"])]
        rows.append(dict(context(trade), proposer=float(d[trade["proposer"]]), partner=float(d[trade["partner"]]),
                         outsiders=float(d[outsiders].mean())))
        if (t + 1) % 20 == 0:
            print(f"{t + 1} trades done", flush=True)
    summary = {k: ci(np.array([r[k] for r in rows])) for k in ("proposer", "partner", "outsiders")}
    summary["both_gain"] = round(float(np.mean([r["proposer"] > 0 and r["partner"] > 0 for r in rows])), 3)
    print("change in win probability from a trade [mean, 95% CI]:")
    for k in ("proposer", "partner", "outsiders"):
        print(f"  {k:>9}: {summary[k]}")
    print(f"  trades where proposer and partner both gain: {summary['both_gain']}")
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "counterfactual.jsonl", "a") as f:
        f.write(json.dumps(dict(agent=args.agent, rollouts=args.rollouts, summary=summary, trades=rows)) + "\n")


if __name__ == "__main__":
    main()
