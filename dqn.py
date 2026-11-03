"""Q-learning (Double DQN) self-play with a league.

    python dqn.py --name dqn_on_s1 --seed 1 --hours 2 [--no-trades] [--config best.json]

Each seat's transitions run from one of its decisions to its own N_STEP-th next decision
(the seat-stream idea of the PPO advantages); the game result reaches every seat's last
decisions. Epsilon-greedy over legal actions, a target network, and a replay buffer.
"""

import argparse
import copy
import json
import time

import fastcatan as fc
import numpy as np
import torch
from torch import nn

from common import RandomAgent, TradeRules, Trunk, log, observe, one_hot, play, run_dir, wins
from league import League

NUM_ENVS, N_STEP, GAMMA = 256, 5, 0.999
LR, BATCH, REPLAY, LEARN_START, TRAIN_EVERY, TARGET_EVERY = 1e-4, 512, 100_000, 20_000, 4, 2_000
EPS_END, EPS_STEPS = 0.05, 5_000_000  # epsilon decays linearly to EPS_END over EPS_STEPS env steps
P_SELFPLAY = 0.5
WIDTH, DEPTH = 256, 2  # hidden units per layer, hidden layers
SNAPSHOT_EVERY, CKPT_EVERY, MONITOR_EVERY, LOG_EVERY = 4_000, 4_000, 8_000, 500  # in batched steps

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
        best, value = q.argmax(1).cpu().numpy(), q.max(1).values.cpu().numpy()
        explore = self.rng.random(len(obs)) < self.eps
        best[explore] = np.where(mask, self.rng.random(mask.shape), -1).argmax(1)[explore]
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
        ckpt = torch.load(path, map_location=device)
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


def train(name, seed, hours, trades, config=None):
    globals().update(config or {})  # overrides of the constants above, e.g. from hpsearch.py
    torch.manual_seed(seed)
    np.random.seed(seed)
    constants = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    out = run_dir(name, dict(algo="dqn", seed=seed, hours=hours, trades=trades, **constants))
    rng = np.random.default_rng(seed)

    agent, rnd = DQNAgent(eps=1.0, seed=seed), RandomAgent(seed)
    target, opt = copy.deepcopy(agent.q), torch.optim.Adam(agent.q.parameters(), lr=LR)
    env = fc.BatchedEnv(NUM_ENVS, seed=seed)
    env.reset()
    rules = TradeRules(NUM_ENVS, trades)
    league = League(NUM_ENVS, P_SELFPLAY, seed=seed)
    replay = Replay(REPLAY)
    history = [[[] for _ in range(4)] for _ in range(NUM_ENVS)]  # the last N_STEP (obs, action) per seat
    done, no_mask = np.zeros(NUM_ENVS, np.uint8), np.ones(fc.NUM_ACTIONS, bool)
    start, env_steps, step, losses, games = time.time(), 0, 0, [], 0
    agent.save(out / "ckpt_0000000.pt", step=0, env_steps=0)

    while time.time() - start < hours * 3600:
        step += 1
        agent.eps = max(EPS_END, 1 - env_steps / EPS_STEPS)
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
        if step % SNAPSHOT_EVERY == 0:
            frozen = copy.deepcopy(agent)
            frozen.eps = 0.02
            league.add(frozen)
        if step % LOG_EVERY == 0:
            row = dict(step=step, env_steps=env_steps, hours=(time.time() - start) / 3600, games=games,
                       eps=agent.eps, replay=replay.n, loss=float(np.mean(losses)) if losses else None,
                       pool=len(league.pool), pool_winrate=float(np.mean(league.winrate)))
            losses, games = [], 0
            if step % MONITOR_EVERY == 0:
                greedy = copy.deepcopy(agent)
                greedy.eps = 0.0
                row["vs_random"] = wins(play([greedy, rnd, greedy, rnd], 50, 10_000, trades), [0, 2]) / 50
                play([greedy] * 4, 2, 20_000, trades, log_path=out / "games" / f"step_{step:07d}.jsonl.gz")
            log(out, row)
        if step % CKPT_EVERY == 0:
            agent.save(out / f"ckpt_{step:07d}.pt", step=step, env_steps=env_steps)

    agent.save(out / f"ckpt_{step:07d}.pt", step=step, env_steps=env_steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hours", type=float, default=1.0, help="wall-clock training budget")
    p.add_argument("--no-trades", action="store_true", help="block player trades for every seat")
    p.add_argument("--config", help="JSON file overriding the constants above (e.g. hpsearch's best)")
    args = p.parse_args()
    train(args.name, args.seed, args.hours, not args.no_trades, json.load(open(args.config)) if args.config else None)
