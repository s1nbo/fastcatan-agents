"""Q-learning (Double DQN) self-play with a league.

    python dqn.py --name dqn_on_s1 --seed 1 --hours 2 [--no-trades] [--config best.json] [--resume]

Each seat's transitions run from one of its decisions to its own N_STEP-th next decision
(the seat-stream idea of the PPO advantages); the game result reaches every seat's last
decisions. Epsilon-greedy over legal actions, a target network, and a replay buffer.
Run files and --resume as in ppo.py (the replay buffer is not saved: it refills in seconds).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy/torch/fastcatan load (see common.py)

import argparse
import copy
import json
import time

import fastcatan as fc
import numpy as np
import torch
from torch import nn

from common import RandomAgent, TradeRules, Trunk, log, observe, one_hot, play, resume_run, run_dir, save_resume, wins
from league import League

NUM_ENVS, N_STEP, GAMMA = 512, 5, 0.999
LR, BATCH, REPLAY, LEARN_START, TRAIN_EVERY, TARGET_EVERY = 3e-5, 512, 1_000_000, 100_000, 2, 2_000
EPS_END, EPS_DECAY = 0.05, 0.02  # epsilon falls linearly to EPS_END over the first EPS_DECAY of --hours
P_SELFPLAY = 0.8  # share of pure self-play games (the same in every trainer)
WIDTH, DEPTH = 1024, 5  # hidden units per layer, hidden layers
SNAPSHOT_EVERY = 20_000  # finished games between league snapshots (the same in every trainer)
POOL_EPS = 0.02  # epsilon of the league's frozen copies
LOG_EVERY = 500  # in batched steps
CKPT_EVERY = 100_000_000  # env steps between kept checkpoints (the same in every trainer)
MONITOR_EVERY, RESUME_EVERY = 3600, 600  # seconds (the same in every trainer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def qnet(width, depth):
    return nn.Sequential(Trunk(width, depth), nn.Linear(width, fc.NUM_ACTIONS)).to(device)


class DQNAgent:
    name = "dqn"

    def __init__(self, eps=0.0, seed=0, width=None, depth=None):
        self.width, self.depth = width or WIDTH, depth or DEPTH
        self.q, self.eps, self.rng = qnet(self.width, self.depth), eps, np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, env, idx, obs, mask, seat, rules):
        """Epsilon-greedy over the legal actions."""
        q = self.q(torch.from_numpy(obs).to(device)).masked_fill(~torch.from_numpy(mask).to(device), -1e9)
        value, best = (x.cpu().numpy() for x in q.max(1))
        explore = np.flatnonzero(self.rng.random(len(obs)) < self.eps)
        best[explore] = np.where(mask[explore], self.rng.random((len(explore), mask.shape[1])), -1).argmax(1)
        return best.astype(np.uint32), value

    @torch.no_grad()
    def probs(self, env, idx, obs, mask, seat, rules):
        """The epsilon-greedy policy: the greedy action, plus epsilon spread over the legal ones."""
        q = self.q(torch.from_numpy(obs).to(device)).masked_fill(~torch.from_numpy(mask).to(device), -1e9)
        return (1 - self.eps) * one_hot(q.argmax(1).cpu().numpy()) + self.eps * mask / mask.sum(1, keepdims=True)

    def save(self, path, **extra):
        torch.save({"algo": "dqn", "width": self.width, "depth": self.depth, "q": self.q.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent = cls(width=ckpt["width"], depth=ckpt["depth"])
        agent.q.load_state_dict(ckpt["q"])
        return agent


class Replay:
    """Ring buffer of n-step transitions; masks are bit-packed."""

    def __init__(self, size):
        self.obs = np.zeros((size, fc.OBS_SIZE), np.float16)
        self.next = np.zeros((size, fc.OBS_SIZE), np.float16)
        self.mask = np.zeros((size, (fc.NUM_ACTIONS + 7) // 8), np.uint8)  # legal actions at `next`
        self.act, self.ret, self.disc = np.zeros(size, np.int64), np.zeros(size, np.float32), np.zeros(size, np.float32)
        self.size, self.n, self.i = size, 0, 0

    def add(self, obs, act, ret, disc, next_obs, next_mask):
        i = self.i
        self.obs[i], self.act[i], self.ret[i], self.disc[i] = obs, act, ret, disc
        self.next[i], self.mask[i] = next_obs, np.packbits(next_mask)
        self.i, self.n = (i + 1) % self.size, min(self.n + 1, self.size)

    def sample(self, k, rng):
        j = rng.integers(self.n, size=k)
        mask = np.unpackbits(self.mask[j], axis=1, count=fc.NUM_ACTIONS).astype(bool)
        return [torch.from_numpy(x).to(device) for x in (self.obs[j].astype(np.float32), self.act[j], self.ret[j],
                                                         self.disc[j], self.next[j].astype(np.float32), mask)]


def pool_agent(out, id):
    """League snapshot `id` from <run>/pool/, exploring with POOL_EPS."""
    agent = DQNAgent.load(out / "pool" / f"{id}.pt")
    agent.eps, agent.rng = POOL_EPS, np.random.default_rng(id)
    return agent


def train(name, seed=1, hours=None, trades=True, config=None, resume=False):
    """resume: as in ppo.train."""
    resumed = resume_run(name, hours) if resume else None
    if resumed:
        out, saved, state = resumed
        seed, trades, hours = saved["seed"], saved["trades"], saved["hours"]
        config = {k: v for k, v in saved.items() if k.isupper()}
    else:
        state, hours = None, 1.0 if hours is None else hours
    if unknown := set(config or {}) - {k for k in globals() if k.isupper()}:
        raise ValueError(f"--config has unknown constants: {sorted(unknown)}")
    globals().update(config or {})  # overrides of the constants above, e.g. from hpsearch.py
    resumes = state["resumes"] + 1 if state else 0
    rs = seed + 10_007 * resumes  # fresh random streams after each resume
    torch.manual_seed(rs)
    np.random.seed(rs)
    if not state:
        constants = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
        out = run_dir(name, dict(algo="dqn", seed=seed, hours=hours, trades=trades, **constants))
    rng = np.random.default_rng(rs)

    agent, rnd = DQNAgent(eps=1.0, seed=rs), RandomAgent(rs)
    target, opt = copy.deepcopy(agent.q), torch.optim.Adam(agent.q.parameters(), lr=LR)
    env = fc.BatchedEnv(NUM_ENVS, seed=rs)
    env.reset()
    rules = TradeRules(NUM_ENVS, trades)
    league = League(NUM_ENVS, P_SELFPLAY, seed=rs)
    replay = Replay(REPLAY)
    history = [[[] for _ in range(4)] for _ in range(NUM_ENVS)]  # the last N_STEP (obs, action) per seat
    done, no_mask = np.zeros(NUM_ENVS, np.uint8), np.ones(fc.NUM_ACTIONS, bool)
    start, env_steps, step, losses, games = time.time(), 0, 0, [], 0
    snapshots = last_snapshot = 0  # snapshots taken; league.games at the last one
    if state:
        agent.q.load_state_dict(state["q"])
        target.load_state_dict(state["target"])
        opt.load_state_dict(state["opt"])
        league.load_state(state["league"], lambda id: pool_agent(out, id))
        start -= state["elapsed"]
        env_steps, step, snapshots, last_snapshot = (state[k] for k in ("env_steps", "step", "snapshots",
                                                                        "last_snapshot"))
        log(out, dict(resumed=resumes, hours=state["elapsed"] / 3600, env_steps=env_steps))
    else:
        agent.save(out / f"ckpt_{0:012d}.pt", step=0, env_steps=0)

    def save_state():
        save_resume(out, dict(resumes=resumes, elapsed=time.time() - start, env_steps=env_steps, step=step,
                              snapshots=snapshots, last_snapshot=last_snapshot, q=agent.q.state_dict(),
                              target=target.state_dict(), opt=opt.state_dict(), league=league.state()), league.ids)

    save_state()
    next_ckpt = (env_steps // CKPT_EVERY + 1) * CKPT_EVERY
    last_monitor = last_resume = time.time()

    while time.time() - start < hours * 3600:
        step += 1
        agent.eps = max(EPS_END, 1 - (time.time() - start) / (EPS_DECAY * hours * 3600))
        obs, mask, seat, status = observe(env, NUM_ENVS)
        rules.apply(mask, seat, status[:, 2])
        mine = league.learner_rows(seat)
        actions, _ = agent.act(env, None, obs, mask, seat, rules)
        league.opponents_act(env, obs, mask, seat, rules, actions)
        for i in np.flatnonzero(mine):  # this decision closes the seat's transition from N_STEP ago
            h = history[i][seat[i]]
            if len(h) == N_STEP:
                o, a = h.pop(0)
                replay.add(o, a, 0.0, GAMMA ** N_STEP, obs[i], mask[i])
            h.append((obs[i], actions[i]))
        env.step_autoreset(actions, done)
        rules.count(actions, seat)
        env_steps += NUM_ENVS
        for i in np.flatnonzero(done):  # the result reaches each seat's last decisions
            winner = env.last_winner(int(i))
            result = np.zeros(4) if winner == fc.NO_PLAYER else np.where(np.arange(4) == winner, 1.0, -1.0)
            for s in range(4):
                for k, (o, a) in enumerate(reversed(history[i][s])):
                    replay.add(o, a, GAMMA ** k * result[s], 0.0, o, no_mask)
                history[i][s] = []
            league.game_over(i, winner)
            games += 1

        if replay.n >= LEARN_START and step % TRAIN_EVERY == 0:  # Double DQN step
            o, a, ret, disc, nxt, nmask = replay.sample(BATCH, rng)
            with torch.no_grad():
                best = agent.q(nxt).masked_fill(~nmask, -1e9).argmax(1, keepdim=True)
                y = ret + disc * target(nxt).gather(1, best).squeeze(1)
            loss = nn.functional.smooth_l1_loss(agent.q(o).gather(1, a[:, None]).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if step % TARGET_EVERY == 0:
            target.load_state_dict(agent.q.state_dict())
        if league.games - last_snapshot >= SNAPSHOT_EVERY:
            snapshots, last_snapshot = snapshots + 1, league.games
            agent.save(out / "pool" / f"{snapshots}.pt", step=step, env_steps=env_steps)
            frozen = copy.copy(agent)  # the league makes the deep copy
            frozen.eps = POOL_EPS
            league.add(frozen, snapshots)
        if step % LOG_EVERY == 0:
            row = dict(step=step, env_steps=env_steps, hours=(time.time() - start) / 3600, games=games,
                       total_games=league.games,
                       eps=agent.eps, replay=replay.n, loss=float(np.mean(losses)) if losses else None,
                       pool=len(league.pool), pool_winrate=float(np.mean(league.winrate)))
            losses, games = [], 0
            if time.time() - last_monitor > MONITOR_EVERY:
                greedy = copy.deepcopy(agent)
                greedy.eps = 0.0
                row["vs_random"] = wins(play([greedy, rnd, greedy, rnd], 50, 10_000, trades), [0, 2]) / 50
                play([greedy] * 4, 2, 20_000, trades, log_path=out / "games" / f"step_{step:07d}.jsonl.gz")
                last_monitor = time.time()
            log(out, row)
        if env_steps >= next_ckpt:  # kept for good, unlike the pool snapshots
            agent.save(out / f"ckpt_{next_ckpt:012d}.pt", step=step, env_steps=env_steps)
            next_ckpt = (env_steps // CKPT_EVERY + 1) * CKPT_EVERY
        if time.time() - last_resume > RESUME_EVERY:
            save_state()
            last_resume = time.time()

    agent.save(out / f"ckpt_{env_steps:012d}.pt", step=step, env_steps=env_steps)
    save_state()  # --resume --hours <more> trains on


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hours", type=float, help="wall-clock training budget (default 1); with --resume: a new total")
    p.add_argument("--no-trades", action="store_true", help="block player trades for every seat")
    p.add_argument("--config", help="JSON file overriding the constants above (e.g. hpsearch's best)")
    p.add_argument("--resume", action="store_true",
                   help="continue runs/<name>/ from resume.pt with its own settings, or start it if it is new")
    args = p.parse_args()
    train(args.name, args.seed, args.hours, not args.no_trades,
          json.load(open(args.config)) if args.config else None, args.resume)
