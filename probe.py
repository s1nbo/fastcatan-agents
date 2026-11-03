"""Offer probes (questions 1, 3 and 4): scripted offers at real game states.

    python probe.py runs/a/ckpt_01000.pt --states 300

States come from the agent's own self-play: the turn owner P could start an offer, some opponent
R holds 2+ cards of one resource and P holds 2+ of another. On a copy of each state, P offers R
a trade (the other responders decline first) and the agent's probability that R accepts is read.
1. Sanity: a good offer for R (R gets 2 cards for 1) against a bad one (R gives 2 for 1).
2. Lead: only P's VP in R's observation is edited, from 3 behind to 3 ahead of the best other
   seat, and the same offers are shown at three ratios. Rising demands as P's lead grows mean
   leaders pay worse terms. An edited observation does not reach a search, so AlphaZero answers
   this part with its network alone. The edit can make the board look inconsistent (VP without
   matching buildings); keep that in mind when reading the numbers.
Intervals are 95% bootstrap intervals over states. Rows go to results/probe.jsonl.
"""

import argparse
import copy
import json

import fastcatan as fc
import numpy as np

from common import ADD, ROOT, A, TradeRules, observe, run
from evaluate import load

OFFERS = {"R gets 2 for 1": (2, 1), "1 for 1": (1, 1), "R gives 2 for 1": (1, 2)}  # (P gives, P asks)
LEADS = range(-3, 4)
rng = np.random.default_rng(0)


def collect(agent, n):
    """Snapshots where the turn owner may start an offer, spread over many games and turns."""
    states, seed = [], 0
    while len(states) < n:
        def grab(env, obs, mask, seat, status, actions):
            for i in np.flatnonzero((status[:, 0] == 0) & (status[:, 3] == seat) & mask[:, ADD].any(1)):
                if rng.random() < 0.02:
                    states.append(env.snapshot(int(i)))
            return len(states) >= n

        env = fc.BatchedEnv(64, seed=1_000 + seed)
        env.reset()
        run([agent] * 4, env, TradeRules(64, True), on_step=grab)
        seed += 1
    return states[:n]


def pick(snapshot):
    """(P, R, resource P gives, resource P asks for), with 2+ cards of each, or None."""
    env = fc.Env()
    env.load_snapshot(snapshot)
    P = env.actor_to_act
    options = []
    for R in (s for s in range(4) if s != P):
        asks = max(range(5), key=lambda r: env.player_resource(R, r))
        gives = max((r for r in range(5) if r != asks), key=lambda r: env.player_resource(P, r))
        if env.player_resource(R, asks) >= 2 and env.player_resource(P, gives) >= 2:
            options.append((P, R, gives, asks))
    return options[rng.integers(len(options))] if options else None


def offer(snapshot, P, R, gives, asks, n_give, n_ask):
    """The game right after P offers n_give x `gives` for n_ask x `asks`, at R's answer."""
    env, mask = fc.Env(), np.zeros(fc.NUM_ACTIONS, bool)
    env.load_snapshot(snapshot)
    moves = [A.TRADE_ADD_GIVE_BASE + gives] * n_give + [A.TRADE_ADD_WANT_BASE + asks] * n_ask + [A.TRADE_OPEN]
    for a in moves:
        env.action_mask_dense(mask)
        if not mask[a]:
            return None
        env.step(a)
    while env.actor_to_act != R:  # responders before R decline
        env.action_mask_dense(mask)
        if env.actor_to_act == P or not mask[A.TRADE_DECLINE]:
            return None
        env.step(A.TRADE_DECLINE)
    env.action_mask_dense(mask)
    return env.snapshot() if mask[A.TRADE_ACCEPT] else None


def p_accept(agent, snapshots, lead=None, proposers=None):
    """The agent's P(accept) at each answer state; `lead` edits the proposer's VP in R's view."""
    n = len(snapshots)
    env = fc.BatchedEnv(n, seed=0)
    env.load_snapshots(np.stack([np.frombuffer(s, np.uint8) for s in snapshots]))
    obs, mask, seat, status = observe(env, n)
    rules = TradeRules(n, True)
    rules.apply(mask, seat, status[:, 2])
    keep = np.ones(n, bool)
    if lead is not None:
        vp = np.rint(obs[:, 0:68:17] * 10)  # VP per relative seat, as R sees it
        rel = (np.asarray(proposers) - seat) % 4
        others = vp.copy()
        others[np.arange(n), rel] = -1
        new = others.max(1) + lead
        keep = (new >= 0) & (new <= 9)
        obs[np.arange(n), rel * 17] = np.clip(new, 0, 9) / 10
    return agent.probs(env, np.arange(n), obs, mask, seat, rules)[:, A.TRADE_ACCEPT], keep


def ci(x):
    boot = x[rng.integers(len(x), size=(1000, len(x)))].mean(1)
    return [round(float(x.mean()), 3), round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent")
    p.add_argument("--states", type=int, default=300)
    args = p.parse_args()
    agent = load(args.agent)
    if getattr(agent, "n_sims", 0):  # AlphaZero: lead edits only reach the network
        raw = copy.copy(agent)
        raw.n_sims = 0
    else:
        raw = agent

    probes = []  # one (P, R, snapshot per offer) per usable state
    for s in collect(agent, args.states):
        choice = pick(s)
        if choice:
            snaps = {name: offer(s, *choice, *size) for name, size in OFFERS.items()}
            if all(snaps.values()):
                probes.append((choice[0], snaps))
    proposers = [P for P, _ in probes]
    print(f"{len(probes)} usable states of {args.states}")

    good = p_accept(agent, [snaps["R gets 2 for 1"] for _, snaps in probes])[0]
    bad = p_accept(agent, [snaps["R gives 2 for 1"] for _, snaps in probes])[0]
    sanity = dict(good=ci(good), bad=ci(bad), gap=ci(good - bad))
    print(f"P(accept) good offer {sanity['good']}  bad offer {sanity['bad']}  gap {sanity['gap']}  [mean, 95% CI]")

    table = {}
    print("P(accept) by the proposer's lead:", "  ".join(f"{d:+d}" for d in LEADS))
    for name in OFFERS:
        snaps = [s[name] for _, s in probes]
        table[name] = {}
        for d in LEADS:
            pa, keep = p_accept(raw, snaps, lead=d, proposers=proposers)
            table[name][d] = ci(pa[keep]) if keep.sum() >= 20 else None
        print(f"  {name:>16}:", "  ".join(f"{v[0]:.2f}" if v else "  - " for v in table[name].values()))

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "probe.jsonl", "a") as f:
        f.write(json.dumps(dict(agent=args.agent, states=len(probes), sanity=sanity, by_lead=table,
                                lead_probe_uses="network" if raw is not agent else "agent")) + "\n")


if __name__ == "__main__":
    main()
