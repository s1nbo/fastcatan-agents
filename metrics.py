"""Trade metrics for the research question, from evaluation event logs.

    python evaluate.py vs A.pt A.pt --games 1000 --events results/a.fcev   # a.A02.fcev, a.A13.fcev
    python metrics.py results/a.A*.fcev --name a

All files given form one dataset (run once per checkpoint / seed / algorithm for stability).
Threat is the proposer's lead: its public VP minus the best other seat's at that moment,
clipped to -3..+3. Answers of trade-off seats are left out (they have to decline).
- willingness: acceptance by lead (Wilson 95%); a logistic fit accept ~ lead + log(cards R gets /
  cards R gives) with a game-bootstrap 95% interval for the lead slope; offers per turn by the
  turn owner's lead;
- terms: cards the proposer gives per card it asks for, by lead, for offers and for trades;
- reciprocity: acceptance when the proposer accepted an offer from this responder earlier in
  the same game, vs not;
- by agent: acceptance per (responder, proposer) pair, from the seat names in the logs.
Benefit (trade-on vs trade-off seats) is printed from results/eval.jsonl. Rows go to
results/metrics.jsonl.
"""

import argparse
import json
from collections import defaultdict

import fastcatan as fc
import numpy as np
from fastcatan.events import read_events

from common import ROOT, A, wilson

F = {k: v[0] for k, v in fc.EVENT_FIELDS.items()}
E = fc.log
LEADS = range(-3, 4)


def lead(row, seat):
    vp = row[F["public_vp"]: F["public_vp"] + 4].astype(int)
    return int(np.clip(vp[seat] - np.delete(vp, seat).max(), -3, 3))


def bundle(row, field):
    return int(row[F[field]: F[field] + 5].sum())


def read(paths):
    """Answers, offers, trades and turns over every game of every file."""
    answers, offers, trades, turns, games = [], [], [], [], 0
    for path in paths:
        for game in read_events(path).games():
            g, names, on = games, game.agents or ["?"] * 4, game.meta.get("trades", [True] * 4)
            games += 1
            accepted = set()  # (i, j): i accepted an offer from j earlier in this game
            for row in game.rows:
                kind, proposer, actor = row[F["event_type"]], row[F["proposer"]], row[F["actor"]]
                if kind == E.ACTION and row[F["action"]] == A.END_TURN:
                    turns.append((g, lead(row, actor)))
                elif kind == E.TRADE_OPEN:
                    offers.append((g, lead(row, proposer), bundle(row, "give"), bundle(row, "want")))
                elif kind in (E.TRADE_ACCEPT, E.TRADE_DECLINE) and on[actor]:
                    yes = kind == E.TRADE_ACCEPT
                    answers.append(dict(game=g, lead=lead(row, proposer), yes=yes, gets=bundle(row, "give"),
                                        gives=bundle(row, "want"), reciprocal=(proposer, actor) in accepted,
                                        pair=(names[actor], names[proposer])))
                    if yes:
                        accepted.add((actor, proposer))
                elif kind == E.TRADE_CONFIRM:
                    trades.append((g, lead(row, proposer), bundle(row, "give"), bundle(row, "want")))
    return answers, offers, trades, turns, games


def logistic(X, y, iters=25):
    """Logistic regression weights [bias, ...] by Newton's method."""
    X, w = np.column_stack([np.ones(len(X)), X]), np.zeros(X.shape[1] + 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        w += np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w)), X.T @ (y - p))
    return w


def rate(k, n):
    return [round(k / n, 3), *(round(x, 3) for x in wilson(k, n))] if n else None


def summarize(paths, name):
    answers, offers, trades, turns, games = read(paths)
    yes = np.array([a["yes"] for a in answers], float)
    L = np.array([a["lead"] for a in answers])
    out = dict(name=name, logs=[str(p) for p in paths], games=games, offers_per_game=round(len(offers) / games, 2),
               trades_per_game=round(len(trades) / games, 2), accept_rate=rate(int(yes.sum()), len(yes)))
    out["accept_by_lead"] = {d: rate(int(yes[L == d].sum()), int((L == d).sum())) for d in LEADS}

    # Lead slope, controlling for how generous the offer is; 95% interval by resampling games.
    X = np.column_stack([L, np.log((np.array([a["gets"] for a in answers]) + 0.5) /
                                   (np.array([a["gives"] for a in answers]) + 0.5))])
    game_of = np.array([a["game"] for a in answers])
    rows_of = defaultdict(list)
    for i, g in enumerate(game_of):
        rows_of[g].append(i)
    ids, rng = list(rows_of), np.random.default_rng(0)
    slope = logistic(X, yes)[1]
    boot = [logistic(X[j], yes[j])[1] for j in
            (np.concatenate([rows_of[g] for g in rng.choice(ids, len(ids))]) for _ in range(200))]
    out["lead_slope"] = [round(float(slope), 4), *(round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5]))]

    turn_lead, offer_lead = np.array([t[1] for t in turns]), np.array([o[1] for o in offers])
    out["offers_per_turn_by_lead"] = {d: round((offer_lead == d).sum() / max(1, (turn_lead == d).sum()), 3)
                                      for d in LEADS}
    for key, rows in (("offer_terms_by_lead", offers), ("trade_terms_by_lead", trades)):
        out[key] = {d: round(sum(r[2] for r in rows if r[1] == d) / max(1, sum(r[3] for r in rows if r[1] == d)), 3)
                    for d in LEADS}  # cards the proposer gives per card it asks for
    rec = np.array([a["reciprocal"] for a in answers])
    out["reciprocity"] = dict(after_partner_accepted=rate(int(yes[rec].sum()), int(rec.sum())),
                              otherwise=rate(int(yes[~rec].sum()), int((~rec).sum())))
    pairs = defaultdict(lambda: [0, 0])
    for a in answers:
        pairs[a["pair"]][0] += a["yes"]
        pairs[a["pair"]][1] += 1
    out["accept_by_pair"] = {f"{r} <- {p}": rate(k, n) for (r, p), (k, n) in pairs.items()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--name", default="")
    args = p.parse_args()
    out = summarize(args.logs, args.name)
    print(f"{out['games']} games: {out['offers_per_game']} offers, {out['trades_per_game']} trades per game, "
          f"acceptance {out['accept_rate']}  [rate, 95% CI]")
    print("lead:                  " + "  ".join(f"{d:+6d}" for d in LEADS))
    for key, label in (("accept_by_lead", "acceptance"), ("offers_per_turn_by_lead", "offers per turn"),
                       ("offer_terms_by_lead", "give/ask, offers"), ("trade_terms_by_lead", "give/ask, trades")):
        cells = [v[0] if isinstance(v, list) else v for v in out[key].values()]
        print(f"{label:>22} " + "  ".join(f"{c:6.3f}" if c is not None else "     -" for c in cells))
    print(f"lead slope of acceptance (log-odds per VP of lead): {out['lead_slope']}")
    print(f"reciprocity: {out['reciprocity']}")
    if len(out["accept_by_pair"]) > 1:
        print(f"acceptance by (responder <- proposer): {out['accept_by_pair']}")
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "metrics.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")

    evals = ROOT / "results" / "eval.jsonl"
    for r in map(json.loads, evals.read_text().splitlines()) if evals.exists() else []:
        if r.get("trades_a") and r.get("trades_b") is False:  # benefit: A may trade, B may not
            print(f"benefit  {r['a']} (trades) vs {r['b']} (no trades): A per-seat win rate {r['a_per_seat']:.3f} "
                  f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] over {r['games']} games")


if __name__ == "__main__":
    main()
