"""PPO self-play with a league: one network learns, frozen past copies of it are opponents.

    python ppo.py --name ppo_on_s1 --seed 1 --hours 2 [--no-trades] [--config best.json]

Writes runs/<name>/: config.json, metrics.jsonl, ckpt_XXXXX.pt (all kept), games/*.jsonl.gz.
"""

import argparse
import json
import time

import fastcatan as fc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from common import RandomAgent, TradeRules, Trunk, log, observe, play, run_dir, wins
from league import League

NUM_ENVS, STEPS = 256, 128
LR, GAMMA, LAMBDA, CLIP, ENT_COEF, VF_COEF, MAX_GRAD_NORM = 2.5e-4, 0.999, 0.95, 0.2, 0.01, 0.5, 0.5
EPOCHS, MINIBATCHES = 4, 8
P_SELFPLAY = 0.5
WIDTH, DEPTH = 256, 2  # hidden units per layer, hidden layers (policy and value net each)
SNAPSHOT_EVERY, CKPT_EVERY, MONITOR_EVERY = 25, 25, 50  # in updates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(out, width, depth):
    return nn.Sequential(Trunk(width, depth), nn.Linear(width, out)).to(device)


class PPOAgent:
    name = "ppo"

    def __init__(self, width=None, depth=None):
        self.width, self.depth = width or WIDTH, depth or DEPTH
        self.policy, self.value = mlp(fc.NUM_ACTIONS, self.width, self.depth), mlp(1, self.width, self.depth)

    def dist(self, obs, mask):
        return Categorical(logits=self.policy(obs).masked_fill(~mask, -1e8))  # illegal actions masked

    @torch.no_grad()
    def act(self, env, idx, obs, mask, seat, rules):
        obs = torch.from_numpy(obs).to(device)
        action = self.dist(obs, torch.from_numpy(mask).to(device)).sample()
        return action.cpu().numpy().astype(np.uint32), self.value(obs).squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def probs(self, env, idx, obs, mask, seat, rules):
        return self.dist(torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device)).probs.cpu().numpy()

    def save(self, path, **extra):
        torch.save({"algo": "ppo", "width": self.width, "depth": self.depth, "policy": self.policy.state_dict(),
                    "value": self.value.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location=device)
        agent = cls(ckpt["width"], ckpt["depth"])
        agent.policy.load_state_dict(ckpt["policy"])
        agent.value.load_state_dict(ckpt["value"])
        return agent


def advantages(values, seats, dones, rewards, bootstrap):
    """GAE when four seats take turns in one game.

    Each seat's "next state" is its own next decision, and the end-of-game reward
    reaches every seat's last decision, not just the final step's.
    """
    T, N = values.shape
    rows = np.arange(N)
    next_v = bootstrap.copy()
    next_a = np.zeros((N, 4), np.float32)
    pending = np.zeros((N, 4), np.float32)  # game reward not yet given to that seat
    adv = np.zeros((T, N), np.float32)
    for t in reversed(range(T)):
        d = dones[t]
        next_v[d], next_a[d], pending[d] = 0, 0, rewards[t, d]
        s = seats[t]
        delta = pending[rows, s] + GAMMA * next_v[rows, s] - values[t]
        pending[rows, s] = 0
        adv[t] = delta + GAMMA * LAMBDA * next_a[rows, s]
        next_v[rows, s], next_a[rows, s] = values[t], adv[t]
    return adv


def train(name, seed, hours, trades, config=None):
    globals().update(config or {})  # overrides of the constants above, e.g. from hpsearch.py
    torch.manual_seed(seed)
    np.random.seed(seed)
    constants = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))}
    out = run_dir(name, dict(algo="ppo", seed=seed, hours=hours, trades=trades, **constants))

    agent, rnd = PPOAgent(), RandomAgent(seed)
    params = [*agent.policy.parameters(), *agent.value.parameters()]
    opt = torch.optim.Adam(params, lr=LR, eps=1e-5)
    env = fc.BatchedEnv(NUM_ENVS, seed=seed)
    env.reset()
    rules = TradeRules(NUM_ENVS, trades)
    league = League(NUM_ENVS, P_SELFPLAY, seed=seed)
    done = np.zeros(NUM_ENVS, np.uint8)
    steps_in_game = np.zeros(NUM_ENVS, np.int64)
    start, env_steps, update = time.time(), 0, 0
    agent.save(out / "ckpt_00000.pt", update=0, env_steps=0)

    while time.time() - start < hours * 3600:
        update += 1
        # Collect STEPS steps in every game. Pool opponents move in their seats.
        buf = {k: [] for k in ("obs", "mask", "act", "logp", "val", "seat", "done", "rew", "mine")}
        finished = []  # (steps, turns) of the games that ended
        with torch.no_grad():
            for _ in range(STEPS):
                obs, mask, seat, status = observe(env, NUM_ENVS)
                rules.apply(mask, seat, status[:, 2])
                mine = league.learner_rows(seat)
                obs_t, mask_t = torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device)
                d = agent.dist(obs_t, mask_t)
                actions = d.sample().cpu().numpy().astype(np.uint32)
                league.opponents_act(env, obs, mask, seat, rules, actions)
                env.step_autoreset(actions, done)
                rules.count(actions, seat)
                steps_in_game += 1

                rew = np.zeros((NUM_ENVS, 4), np.float32)  # +1 winner, -1 others
                for i in np.flatnonzero(done):
                    winner = env.last_winner(int(i))
                    if winner != fc.NO_PLAYER:  # NO_PLAYER = hit the turn limit: 0 for all
                        rew[i] = -1
                        rew[i, winner] = 1
                    league.game_over(i, winner)
                    finished.append((steps_in_game[i], status[i, 2]))
                    steps_in_game[i] = 0

                act_t = torch.from_numpy(actions.astype(np.int64)).to(device)
                for k, v in zip(buf, (obs_t, mask_t, act_t, d.log_prob(act_t), agent.value(obs_t).squeeze(-1),
                                      seat, done.astype(bool), rew, mine)):
                    buf[k].append(v)

            # Games still running: estimate each seat's outcome with the value net.
            bootstrap = np.zeros((NUM_ENVS, 4), np.float32)
            pov_obs = np.zeros((NUM_ENVS, fc.OBS_SIZE), np.float32)
            for s in range(4):
                env.write_obs_pov(np.full(NUM_ENVS, s, np.uint8), pov_obs)
                bootstrap[:, s] = agent.value(torch.from_numpy(pov_obs).to(device)).squeeze(-1).cpu().numpy()

        # Advantages per seat, then train on the learner's rows only.
        vals = torch.stack(buf["val"])
        adv = advantages(vals.cpu().numpy(), np.stack(buf["seat"]), np.stack(buf["done"]),
                         np.stack(buf["rew"]), bootstrap)
        mine = torch.from_numpy(np.stack(buf["mine"]).reshape(-1)).to(device)
        adv = torch.from_numpy(adv).to(device).flatten()[mine]
        ret = adv + vals.flatten()[mine]
        obs, mask = torch.cat(buf["obs"])[mine], torch.cat(buf["mask"])[mine]
        act, logp_old = torch.cat(buf["act"])[mine], torch.cat(buf["logp"])[mine]

        for _ in range(EPOCHS):
            for mb in torch.randperm(len(act), device=device).chunk(MINIBATCHES):
                d = agent.dist(obs[mb], mask[mb])
                ratio = (d.log_prob(act[mb]) - logp_old[mb]).exp()
                a = (adv[mb] - adv[mb].mean()) / (adv[mb].std() + 1e-8)
                pg_loss = -torch.min(a * ratio, a * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
                v_loss = (agent.value(obs[mb]).squeeze(-1) - ret[mb]).pow(2).mean()
                entropy = d.entropy().mean()
                opt.zero_grad()
                (pg_loss + VF_COEF * v_loss - ENT_COEF * entropy).backward()
                nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                opt.step()

        env_steps += NUM_ENVS * STEPS
        if update % SNAPSHOT_EVERY == 0:
            league.add(agent)
        row = dict(update=update, env_steps=env_steps, hours=(time.time() - start) / 3600, games=len(finished),
                   steps_per_turn=float(sum(s for s, _ in finished) / max(1, sum(t for _, t in finished))),
                   pg_loss=pg_loss.item(), v_loss=v_loss.item(), entropy=entropy.item(),
                   pool=len(league.pool), pool_winrate=float(np.mean(league.winrate)))
        if update % MONITOR_EVERY == 0:
            row["vs_random"] = wins(play([agent, rnd, agent, rnd], 50, 10_000, trades), [0, 2]) / 50
            play([agent] * 4, 2, 20_000, trades, log_path=out / "games" / f"update_{update:05d}.jsonl.gz")
        if update % CKPT_EVERY == 0:
            agent.save(out / f"ckpt_{update:05d}.pt", update=update, env_steps=env_steps)
        log(out, row)

    agent.save(out / f"ckpt_{update:05d}.pt", update=update, env_steps=env_steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hours", type=float, default=1.0, help="wall-clock training budget")
    p.add_argument("--no-trades", action="store_true", help="block player trades for every seat")
    p.add_argument("--config", help="JSON file overriding the constants above (e.g. hpsearch's best)")
    args = p.parse_args()
    train(args.name, args.seed, args.hours, not args.no_trades, json.load(open(args.config)) if args.config else None)
