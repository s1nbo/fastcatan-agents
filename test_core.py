"""Hand-case checks for the tricky parts. Run: .venv/bin/python -m pytest -q"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy/torch/fastcatan load (see common.py)

import fastcatan as fc
import numpy as np
import pytest

from alphabeta import AlphaBeta
from alphazero import AZAgent
from common import ADD, MAX_ADDS, MAX_OFFERS, A, RandomAgent, TradeRules, observe, trade_mask
from league import League


def midgame(seed):
    """A random game at turn 30+ where the acting seat has 3+ moves and opponents hold cards."""
    env, mask, rng = fc.Env(), np.zeros(fc.NUM_ACTIONS, bool), np.random.default_rng(seed)
    env.reset(seed)
    while True:
        env.action_mask_dense(mask)
        legal, me = np.flatnonzero(trade_mask(mask, False, 0, 0)), env.actor_to_act
        if env.turn_count >= 30 and len(legal) >= 3 and sum(env.player_handsize(s) for s in range(4) if s != me) >= 4:
            return env, me, legal
        env.step(int(rng.choice(legal)))
        assert not env.done


def hidden(env, me):
    """Everything about the opponents' hands that `me` cannot see."""
    return [[env.player_resource(s, r) for r in range(5)] + [env.player_dev(s, d) for d in range(5)]
            for s in range(4) if s != me]


def test_gae_hand_cases(monkeypatch):
    import ppo

    monkeypatch.setattr(ppo, "GAMMA", 1.0)
    monkeypatch.setattr(ppo, "LAMBDA", 1.0)
    # Seat 1 wins with the action at t=4; seats 0 and 2 still get their -1 on their last move.
    seats, dones, rew = np.array([[0], [1], [0], [2], [1]]), np.zeros((5, 1), bool), np.zeros((5, 1, 4), np.float32)
    dones[4], rew[4, 0] = True, [-1, 1, -1, -1]
    adv = ppo.advantages(np.zeros((5, 1), np.float32), seats, dones, rew, np.full((1, 4), 9, np.float32))
    np.testing.assert_allclose(adv[:, 0], [-1, 1, -1, -1, 1])
    # A new game starts after t=1; its moves are bootstrapped with the value 0.5.
    seats, dones, rew = np.array([[0], [0], [0], [1]]), np.zeros((4, 1), bool), np.zeros((4, 1, 4), np.float32)
    dones[1], rew[1, 0] = True, [1, -1, -1, -1]
    adv = ppo.advantages(np.zeros((4, 1), np.float32), seats, dones, rew, np.full((1, 4), 0.5, np.float32))
    np.testing.assert_allclose(adv[:, 0], [1, 1, 0.5, 0.5])


@pytest.mark.parametrize("trades", [True, False, [True, False, True, False]])
def test_trade_rules(trades):
    n, rnd = 32, RandomAgent(0)
    env, rules, done = fc.BatchedEnv(n, seed=3), TradeRules(n, trades), np.zeros(n, np.uint8)
    env.reset()
    while True:
        obs, mask, seat, status = observe(env, n)
        active = status[:, 0] == 0
        if not active.any():
            break
        rules.apply(mask, seat, status[:, 2])
        assert mask[active].any(1).all()
        act = np.where(active, rnd.act(env, None, obs, mask, seat, rules)[0], fc.NO_ACTION).astype(np.uint32)
        trade = ((act >= ADD.start) & (act < ADD.stop)) | (act == A.TRADE_OPEN) | (act == A.TRADE_ACCEPT)
        assert not trade[active & ~rules.on[np.arange(n), seat]].any()  # trade-off seats never trade
        env.step(act, done)
        rules.count(act, seat)
        assert rules.adds.max() <= MAX_ADDS and rules.offers.max() <= MAX_OFFERS


def test_league():
    league = League(64, 0.5, size=5, seed=0)
    anchor = league.pool[0]
    for _ in range(12):
        league.add(RandomAgent(1))
    assert len(league.pool) == 5 and league.pool[0] is anchor
    assert abs(league.pfsp().sum() - 1) < 1e-9
    for i in range(64):
        if league.opponent[i] < 0:
            assert league.learner[i].all()
        else:
            assert league.learner[i].tolist() in ([True, False, True, False], [False, True, False, True])


def test_enumerate_outcomes_sum_to_one():
    env, rng, seen = fc.Env(), np.random.default_rng(0), set()
    for game in range(30):
        env.reset(game)
        while not env.done and len(seen) < 3:
            legal = env.legal_actions()
            for a in legal:
                kind = ("roll" if a == A.ROLL_DICE else "dev" if a == A.BUY_DEV
                        else "steal" if A.STEAL_BASE <= a < A.TRADE_BASE else None)
                if kind:
                    assert abs(sum(p for p, _ in env.enumerate_outcomes(a)) - 1) < 1e-9
                    seen.add(kind)
            env.step(int(rng.choice(legal)))
    assert seen == {"roll", "dev", "steal"}


def test_determinize_keeps_what_the_seat_sees():
    env, me, _ = midgame(1)
    before, after = np.zeros(fc.OBS_SIZE, np.float32), np.zeros(fc.OBS_SIZE, np.float32)
    env.write_obs(me, before)
    scratch = fc.Env()
    scratch.load_snapshot(env.snapshot())
    scratch.determinize(me, 5)
    scratch.write_obs(me, after)
    np.testing.assert_array_equal(before, after)
    for r in range(5):
        assert scratch.bank(r) + sum(scratch.player_resource(s, r) for s in range(4)) == 19


def test_alphabeta_never_reads_the_true_hidden_state():
    for seed in range(3):
        env, me, legal = midgame(seed)
        true_a, other = env.snapshot(), fc.Env()
        for s in range(100, 200):  # a second true state: same public view, other hidden cards
            other.load_snapshot(true_a)
            other.determinize(me, s)
            if hidden(other, me) != hidden(env, me):
                break
        assert hidden(other, me) != hidden(env, me)
        moves = [AlphaBeta(k=4, seed=7).choose(snap, me, legal) for snap in (true_a, other.snapshot())]
        assert moves[0] == moves[1]


def test_mcts_returns_a_legal_move_and_samples_from_the_root_view():
    env, me, legal = midgame(2)
    root_view = np.zeros(fc.OBS_SIZE, np.float32)
    env.write_obs(me, root_view)

    class Spy:  # records what the root seat would see after every determinization
        def __init__(self, inner):
            self.inner, self.views = inner, []

        def determinize(self, pov, seed):
            self.inner.determinize(pov, seed)
            view = np.zeros(fc.OBS_SIZE, np.float32)
            self.inner.write_obs(me, view)
            self.views.append(view)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    az = AZAgent(seed=0, n_sims=16)
    az.env = Spy(az.env)
    rules = TradeRules(1, False)
    search = az.search(env.snapshot(), me, legal, rules.on[0], rules.adds[0], rules.offers[0], env.turn_count)
    (action, target, _), = az.run([search])
    assert action in legal and abs(target.sum() - 1) < 1e-5 and target[legal].sum() == pytest.approx(1, abs=1e-5)
    assert len(az.env.views) == az.sims > 0 and all((v == root_view).all() for v in az.env.views)


def test_eval_logs_one_file_per_seating(tmp_path):
    from fastcatan.events import read_events

    from evaluate import match

    a, b = RandomAgent(1), RandomAgent(2)
    a_wins, b_wins, truncated, played = match(a, b, 8, 0, True, False, "2v2", tmp_path / "x.fcev", ("a", "b"))
    assert played == 8 and a_wins + b_wins + truncated == 8
    for name, names, trades in (("x.A02.fcev", ["a", "b", "a", "b"], [True, False, True, False]),
                                ("x.A13.fcev", ["b", "a", "b", "a"], [False, True, False, True])):
        log = read_events(tmp_path / name)
        assert len(log.games()) == 4 and log.header["agents"] == names and log.header["meta"]["trades"] == trades


def test_probs_are_distributions_over_legal_actions():
    from dqn import DQNAgent
    from ppo import PPOAgent

    n = 8
    env = fc.BatchedEnv(n, seed=4)
    env.reset()
    obs, mask, seat, status = observe(env, n)
    rules = TradeRules(n, True)
    rules.apply(mask, seat, status[:, 2])
    for agent in (RandomAgent(0), PPOAgent(), DQNAgent(eps=0.1), AZAgent(n_sims=0), AZAgent(n_sims=4), AlphaBeta(k=1)):
        p = agent.probs(env, np.arange(n), obs, mask, seat, rules)
        assert p.shape == (n, fc.NUM_ACTIONS) and np.allclose(p.sum(1), 1, atol=1e-5) and (p[~mask] < 1e-6).all()


def test_league_state_round_trip():
    league = League(8, 0.5, size=4, seed=0)
    for i in range(1, 7):  # 6 snapshots into 3 slots: random evictions
        league.add(RandomAgent(i), id=i)
    league.winrate[1], league.games = 0.9, 42
    copy_ = League(8, 0.5, size=4, seed=1)
    copy_.load_state(league.state(), lambda id: RandomAgent(id))
    assert copy_.ids == league.ids and copy_.winrate == league.winrate and copy_.games == 42
    partial = League(8, 0.5, size=4, seed=1)  # a missing file stops the sync; the rest waits for the next one
    partial.sync(league.ids, lambda id: None if id == league.ids[2] else RandomAgent(id))
    assert partial.ids == league.ids[:2]


def test_ppo_resume_after_a_crash(tmp_path, monkeypatch):
    import json

    import torch

    import common
    import ppo

    monkeypatch.setattr(common, "ROOT", tmp_path)
    for k in [k for k in vars(ppo) if k.isupper()]:  # train() overrides the module constants
        monkeypatch.setattr(ppo, k, getattr(ppo, k))
    cfg = dict(NUM_ENVS=128, STEPS=16, WIDTH=32, DEPTH=1, SNAPSHOT_EVERY=5, RESUME_EVERY=1, CKPT_EVERY=50_000)
    ppo.train("r", 3, 0.002, True, cfg)
    run = tmp_path / "runs" / "r"
    with open(run / "metrics.jsonl", "a") as f:  # the crash: a row after the last save, a line cut off
        f.write(json.dumps(dict(update=10**6, hours=99.0)) + '\n{"upd')
    (run / "pool" / "999.pt").write_bytes(b"")  # a snapshot the saved league no longer holds

    ppo.train("r", hours=0.004, resume=True)
    rows = [json.loads(line) for line in open(run / "metrics.jsonl")]
    hours = [r["hours"] for r in rows]
    assert hours == sorted(hours) and max(hours) < 99 and sum("resumed" in r for r in rows) == 1
    state = torch.load(run / "resume.pt", weights_only=True)
    assert state["resumes"] == 1 and state["elapsed"] >= 0.004 * 3600 and state["snapshots"] > 0
    assert json.loads((run / "config.json").read_text())["seed"] == 3
    on_disk = {int(p.stem) for p in (run / "pool").glob("*.pt")}
    assert on_disk == {i for i in state["league"]["ids"] if i is not None}
    ckpts = sorted(p.name for p in run.glob("ckpt_*.pt"))
    assert ckpts[0] == f"ckpt_{0:012d}.pt" and ckpts[-1] == f"ckpt_{state['env_steps']:012d}.pt" and len(ckpts) > 2
