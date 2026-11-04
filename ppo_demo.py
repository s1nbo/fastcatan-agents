"""Minimal PPO self-play on fastCatan: one network plays all four seats."""

import time

import fastcatan as fc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from fastcatan.viz import GameLogWriter

NUM_ENVS, STEPS, UPDATES = 256, 128, 1500  # 1500 x 256 x 128 = 49M steps
LR, GAMMA, LAMBDA, CLIP, ENT_COEF = 2.5e-4, 0.999, 0.95, 0.2, 0.01
EPOCHS, MINIBATCHES = 4, 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)


def mlp(out):
    return nn.Sequential(
        nn.Linear(fc.OBS_SIZE, 256),
          nn.Tanh(), 
          nn.Linear(256, 256), 
          nn.Tanh(), 
          nn.Linear(256, out)
    ).to(device)

policy, value = mlp(fc.NUM_ACTIONS), mlp(1)
opt = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=LR)


def dist(obs, mask):
    return Categorical(logits=policy(obs).masked_fill(~mask, -1e8))  # illegal actions masked


def observe(env, n):
    """Observation and legal actions from each game's acting seat, plus that seat."""
    obs = np.zeros((n, fc.OBS_SIZE), np.float32)
    mask = np.zeros((n, fc.NUM_ACTIONS), bool)
    seat = np.zeros(n, np.uint8)
    env.write_obs(obs)
    env.write_masks_dense(mask)
    env.write_actors(seat)
    return torch.from_numpy(obs).to(device), torch.from_numpy(mask).to(device), seat


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


def train():
    env = fc.BatchedEnv(NUM_ENVS, seed=0)
    env.reset()
    done = np.zeros(NUM_ENVS, np.uint8)
    total_games, start = 0, time.time()

    for update in range(1, UPDATES + 1):
        # Collect STEPS steps in every game.
        buf = {k: [] for k in ("obs", "mask", "act", "logp", "val", "seat", "done", "rew")}
        with torch.no_grad():
            for _ in range(STEPS):
                obs, mask, seat = observe(env, NUM_ENVS)
                d = dist(obs, mask)
                act = d.sample()
                env.step_autoreset(act.cpu().numpy().astype(np.uint32), done)

                rew = np.zeros((NUM_ENVS, 4), np.float32)  # +1 winner, -1 others
                for i in np.flatnonzero(done):
                    winner = env.last_winner(int(i))
                    if winner != fc.NO_PLAYER:  # NO_PLAYER = hit the turn limit: 0 for all
                        rew[i] = -1
                        rew[i, winner] = 1

                for k, v in zip(buf, (obs, mask, act, d.log_prob(act), value(obs).squeeze(-1),
                                      seat, done.astype(bool), rew)):
                    buf[k].append(v)

            # Games still running: estimate each seat's outcome with the value net.
            bootstrap = np.zeros((NUM_ENVS, 4), np.float32)
            pov_obs = np.zeros((NUM_ENVS, fc.OBS_SIZE), np.float32)
            for s in range(4):
                env.write_obs_pov(np.full(NUM_ENVS, s, np.uint8), pov_obs)
                bootstrap[:, s] = value(torch.from_numpy(pov_obs).to(device)).squeeze(-1).cpu().numpy()

        vals = torch.stack(buf["val"])
        adv = advantages(vals.cpu().numpy(), np.stack(buf["seat"]), np.stack(buf["done"]),
                         np.stack(buf["rew"]), bootstrap)
        adv = torch.from_numpy(adv).to(device).flatten()
        ret = adv + vals.flatten()
        obs, mask = torch.cat(buf["obs"]), torch.cat(buf["mask"])
        act, logp_old = torch.cat(buf["act"]), torch.cat(buf["logp"])

        # PPO update.
        for _ in range(EPOCHS):
            for mb in torch.randperm(len(act), device=device).chunk(MINIBATCHES):
                d = dist(obs[mb], mask[mb])
                ratio = (d.log_prob(act[mb]) - logp_old[mb]).exp()
                a = (adv[mb] - adv[mb].mean()) / (adv[mb].std() + 1e-8)
                pg_loss = -torch.min(a * ratio, a * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
                v_loss = (value(obs[mb]).squeeze(-1) - ret[mb]).pow(2).mean()
                loss = pg_loss + 0.5 * v_loss - ENT_COEF * d.entropy().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        total_games += int(np.stack(buf["done"]).sum())
        print(f"update {update}/{UPDATES}  games finished {total_games}  "
              f"entropy {d.entropy().mean():.2f}  {time.time() - start:.0f}s")


@torch.no_grad()
def record(path, games=4, ppo_seats=(0, 1, 2, 3), seed=1):
    """Play games one at a time, PPO in ppo_seats and random elsewhere, saved for the viewer."""
    agents = ["ppo" if s in ppo_seats else "random" for s in range(4)]
    obs = np.zeros(fc.OBS_SIZE, np.float32)
    mask = np.zeros(fc.NUM_ACTIONS, bool)
    with GameLogWriter(path) as log:
        for g in range(games):
            env = fc.Env()
            env.reset(seed + g)
            rec = log.start_game(env, seed=seed + g, agents=agents)
            while not env.done:
                seat = env.actor_to_act
                env.write_obs(seat, obs)
                env.action_mask_dense(mask)
                if seat in ppo_seats:
                    o = torch.from_numpy(obs).to(device)[None]
                    m = torch.from_numpy(mask).to(device)[None]
                    action = int(dist(o, m).sample())
                    info = {"value": round(value(o).item(), 3)}  # shown in the viewer
                else:
                    action = int(np.random.choice(np.flatnonzero(mask)))
                    info = None
                env.step(action)
                rec.record(action, info=info)
    print(f"saved {games} games to {path}")


@torch.no_grad()
def win_rate_vs_random(games=100, seed=1000):
    """PPO in seats 0 and 2, random in seats 1 and 3, all games at once. Chance is 50%."""
    env = fc.BatchedEnv(games, seed=seed)
    env.reset()
    status = np.zeros((games, fc.STATUS_INTS), np.int32)
    done = np.zeros(games, np.uint8)
    while True:
        env.write_status(status)  # columns: done, winner, turn_count, current_player, truncated
        if status[:, 0].all():
            break
        obs, mask, seat = observe(env, games)
        ppo = dist(obs, mask).sample().cpu().numpy()
        rnd = np.where(mask.cpu().numpy(), np.random.rand(games, fc.NUM_ACTIONS), -1).argmax(1)
        actions = np.where(seat % 2 == 0, ppo, rnd).astype(np.uint32)
        actions[status[:, 0] == 1] = fc.NO_ACTION  # leave finished games alone
        env.step(actions, done)
    ppo_wins = int(((status[:, 4] == 0) & (status[:, 1] % 2 == 0)).sum())
    print(f"PPO won {ppo_wins}/{games} games vs random (chance: 50%)")


if __name__ == "__main__":
    record("selfplay_start.jsonl.gz")  # untrained network against itself
    train()
    torch.save({"policy": policy.state_dict(), "value": value.state_dict()}, "ppo.pt")
    record("selfplay_end.jsonl.gz")  # trained network against itself, same 4 boards
    record("vs_random.jsonl.gz", ppo_seats=(0, 2))
    win_rate_vs_random()
