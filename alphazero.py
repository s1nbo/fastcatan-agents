"""AlphaZero self-play with information-set search and a league, built for one CUDA GPU.

    python alphazero.py --name az_on_s1 --seed 1 --hours 2 [--no-trades] [--config best.json] [--resume]

The tree search is Python on the CPU, so ACTORS self-play processes (one CPU core each) run
searches in parallel, each with its own copy of the network on the GPU. One learner process
trains on the GPU from their finished games and sends new weights every SYNC_EVERY seconds.
The learner also decides the league: it writes each snapshot to <run>/pool/ and every weights
message names the pool, so all actors share one pool. Run files and --resume as in ppo.py (the
replay buffer is not saved). If an actor dies, the learner stops too, so --resume can restart all.
Set ACTORS to about the number of free CPU cores.

Search, for each decision of the acting seat (SO-ISMCTS: one shared tree):
- every simulation re-samples the hidden cards from the root seat's view (Env.determinize)
  and walks the tree in that sampled world;
- leaves are evaluated from write_obs(acting seat), so the network never sees the sampled
  cards of other seats. Values are kept per seat; each seat picks what is best for itself;
- the tree branches on dice results; dev draws and steals are sampled by stepping;
- the root uses Gumbel top-k with sequential halving down to 2 candidates (N_SIMS
  simulations, the mctx schedule); interior nodes use PUCT.
Training targets: the improved Gumbel policy and the final result of the game per seat.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # before numpy/torch/fastcatan load (see common.py)

import argparse
import copy
import json
import queue
import time

import fastcatan as fc
import numpy as np
import torch
from torch import nn

from common import (ADD, A, RandomAgent, TradeRules, Trunk, log, observe, one_hot, play, resume_run, run_dir,
                    save_resume, trade_mask, wins)
from league import League

ACTORS, NUM_ENVS, STEPS = 4, 64, 16  # self-play processes; games in each; decisions between messages
N_SIMS, TOP_K, C_PUCT, C_VISIT, C_SCALE = 32, 16, 1.5, 50, 0.1
LR, BATCH, REPLAY, REPLAY_RATIO = 3e-4, 1024, 2_000_000, 4  # each sample is trained on ~REPLAY_RATIO times
P_SELFPLAY = 0.8  # share of pure self-play games (the same in every trainer)
WIDTH, DEPTH = 1024, 5  # hidden units per layer, hidden layers of the shared trunk
SNAPSHOT_EVERY = 20_000  # finished games (all actors) between league snapshots (the same in every trainer)
CKPT_EVERY = 100_000_000  # env steps between kept checkpoints (the same in every trainer)
SYNC_EVERY, LOG_EVERY = 60, 60  # seconds
MONITOR_EVERY, RESUME_EVERY = 3600, 600  # seconds (the same in every trainer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Policy logits over all actions and one value per seat, relative to the observer."""

    def __init__(self, width=None, depth=None):
        super().__init__()
        self.width, self.depth = width or WIDTH, depth or DEPTH
        self.trunk = Trunk(self.width, self.depth)
        self.policy, self.value = nn.Linear(self.width, fc.NUM_ACTIONS), nn.Linear(self.width, 4)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.policy(h), torch.tanh(self.value(h))


class Node:
    __slots__ = ("logits", "N", "W", "children")

    def __init__(self):
        self.logits, self.N, self.W, self.children = None, 0, np.zeros(4), {}

    def select(self, seat, legal):
        """PUCT among the actions legal in this sampled world."""
        p = np.exp(self.logits[legal] - self.logits[legal].max())
        p /= p.sum()
        kids = [self.children.get(a) for a in legal]
        n = np.array([k.N if k else 0 for k in kids])
        fpu = self.W[seat] / max(self.N, 1)  # unvisited actions start at this node's value
        q = np.array([k.W[seat] / k.N if k and k.N else fpu for k in kids])
        return legal[np.argmax(q + C_PUCT * p * np.sqrt(max(n.sum(), 1)) / (1 + n))]


class AZAgent:
    name = "alphazero"

    def __init__(self, net=None, seed=0, n_sims=None):
        self.net = net if net is not None else Net().to(device)
        self.n_sims = N_SIMS if n_sims is None else n_sims
        self.rng, self.sims = np.random.default_rng(seed), 0
        self.env = fc.Env()  # scratch copy for the search; the real game is never searched
        self.obs = np.zeros(fc.OBS_SIZE, np.float32)
        self.mask = np.zeros(fc.NUM_ACTIONS, bool)

    def __deepcopy__(self, memo):  # for the league pool (an Env cannot be copied)
        return AZAgent(copy.deepcopy(self.net), int(self.rng.integers(2**31)), self.n_sims)

    def act(self, env, idx, obs, mask, seat, rules):
        """Search every row; the policy targets of searched rows are left in self.last."""
        if self.n_sims == 0:  # raw policy, no search
            with torch.no_grad():
                logits, v = self.net(torch.from_numpy(obs).to(device))
            logits = logits.masked_fill(~torch.from_numpy(mask).to(device), -1e8)
            return torch.distributions.Categorical(logits=logits).sample().cpu().numpy().astype(np.uint32), \
                v[:, 0].cpu().numpy()
        actions, values = np.zeros(len(idx), np.uint32), np.full(len(idx), np.nan, np.float32)
        targets, searched = np.zeros((len(idx), fc.NUM_ACTIONS), np.float32), np.zeros(len(idx), bool)
        rows, searches = [], []
        for r, i in enumerate(idx):
            legal = np.flatnonzero(mask[r])
            if len(legal) == 1:  # forced move: no search, no training sample
                actions[r] = legal[0]
                continue
            rows.append(r)
            searches.append(self.search(env.snapshot(int(i)), int(seat[r]), legal, rules.on[i],
                                        rules.adds[i], rules.offers[i], rules.turn[i]))
        for r, (action, target, value) in zip(rows, self.run(searches)):
            actions[r], targets[r], values[r], searched[r] = action, target, value, True
        self.last = targets, searched
        return actions, values

    def probs(self, env, idx, obs, mask, seat, rules):
        """The search policy (improved Gumbel policy); the network's policy when n_sims = 0."""
        if self.n_sims == 0:
            with torch.no_grad():
                logits, _ = self.net(torch.from_numpy(obs).to(device))
            return torch.softmax(logits.masked_fill(~torch.from_numpy(mask).to(device), -1e8), -1).cpu().numpy()
        actions, _ = self.act(env, idx, obs, mask, seat, rules)
        targets, searched = self.last
        return np.where(searched[:, None], targets, one_hot(actions))

    def run(self, searches):
        """Drive the search generators together, batching their leaves into one forward pass."""
        results, pending = [None] * len(searches), {i: next(g) for i, g in enumerate(searches)}
        while pending:
            ids = list(pending)
            with torch.no_grad():
                logits, v = self.net(torch.from_numpy(np.stack([pending[i] for i in ids])).to(device))
            for i, lg, vv in zip(ids, logits.cpu().numpy(), v.cpu().numpy()):
                try:
                    pending[i] = searches[i].send((lg, vv))
                except StopIteration as stop:
                    results[i] = stop.value
                    del pending[i]
        return results

    def search(self, snapshot, seat, legal, on, adds, offers, turn):
        """Generator: yields leaf observations, receives (logits, values).

        Returns (action, improved policy over all actions, root value)."""
        root = Node()
        self.env.load_snapshot(snapshot)
        self.env.write_obs(seat, self.obs)
        root.logits, v = yield self.obs.copy()
        prior, gumbel = root.logits[legal], self.rng.gumbel(size=len(legal))
        cand = np.argsort(-(gumbel + prior))[:min(TOP_K, len(legal))]
        rounds, used = max(1, int(np.ceil(np.log2(len(cand))))), 0
        while used < self.n_sims:  # sequential halving down to 2 candidates, n_sims in all (as in mctx)
            visits = max(1, min(self.n_sims // (rounds * len(cand)), (self.n_sims - used) // len(cand)))
            for c in cand:
                for _ in range(visits):
                    yield from self.simulate(root, snapshot, seat, legal[c], on, adds.copy(), offers.copy(), turn)
            used += visits * len(cand)
            score = (gumbel + prior + self.sigma(root, legal, seat, 0.0))[cand]
            cand = cand[np.argsort(-score)][: max(2, len(cand) // 2)]
        target = np.zeros(fc.NUM_ACTIONS, np.float32)
        logits = prior + self.sigma(root, legal, seat, (v[0] + 1) / 2)
        target[legal] = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        return legal[cand[0]], target, v[0]

    def sigma(self, root, legal, seat, default):
        """Gumbel's monotone transform of the root Q values (unvisited actions get `default`)."""
        kids = [root.children.get(a) for a in legal]
        q = np.array([(k.W[seat] / k.N + 1) / 2 if k and k.N else default for k in kids])
        return (C_VISIT + max((k.N for k in kids if k), default=0)) * C_SCALE * q

    def simulate(self, root, snapshot, root_seat, first, on, adds, offers, turn):
        """One simulation in a fresh determinization, starting with root action `first`."""
        env = self.env
        env.load_snapshot(snapshot)
        env.determinize(root_seat, int(self.rng.integers(2**63)))
        self.sims += 1
        path, node, a, seat = [root], root, first, root_seat
        while True:
            env.step(int(a))
            adds[seat] += ADD.start <= a < ADD.stop
            offers[seat] += a == A.TRADE_OPEN
            node = node.children.setdefault(int(a), Node())
            if a == A.ROLL_DICE:  # chance node: branch on the public dice result
                path.append(node)
                node = node.children.setdefault(env.dice_roll, Node())
            path.append(node)
            if env.done:
                values = np.zeros(4) if env.truncated else np.where(np.arange(4) == env.winner, 1.0, -1.0)
                break
            seat = env.actor_to_act
            if node.logits is None:  # new leaf: evaluated from the acting seat's own view
                env.write_obs(seat, self.obs)
                node.logits, v = yield self.obs.copy()
                values = np.roll(v, seat)  # relative seats -> absolute seats
                break
            if env.turn_count != turn:
                adds[:], offers[:], turn = 0, 0, env.turn_count
            env.action_mask_dense(self.mask)
            a = node.select(seat, np.flatnonzero(trade_mask(self.mask, on[seat], adds[seat], offers[seat])))
        for n in path:
            n.N += 1
            n.W += values

    def save(self, path, **extra):
        torch.save({"algo": "az", "width": self.net.width, "depth": self.net.depth, "net": self.net.state_dict(),
                    "n_sims": self.n_sims, **extra}, path)

    @classmethod
    def load(cls, path, n_sims=None):
        """n_sims: the budget the checkpoint was trained with, unless given."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net = Net(ckpt["width"], ckpt["depth"]).to(device)
        net.load_state_dict(ckpt["net"])
        return cls(net, n_sims=ckpt.get("n_sims", N_SIMS) if n_sims is None else n_sims)


class Replay:
    """Ring buffer of (obs, legal mask, policy target, result per relative seat)."""

    def __init__(self, size):
        self.obs = np.zeros((size, fc.OBS_SIZE), np.float16)
        self.mask = np.zeros((size, fc.NUM_ACTIONS), bool)
        self.pi = np.zeros((size, fc.NUM_ACTIONS), np.float16)
        self.z = np.zeros((size, 4), np.float32)
        self.size, self.n, self.i = size, 0, 0

    def add(self, obs, mask, pi, z):
        """Add k samples (arrays with k rows)."""
        j = (self.i + np.arange(len(obs))) % self.size
        self.obs[j], self.mask[j], self.pi[j], self.z[j] = obs, mask, pi, z
        self.i, self.n = (self.i + len(obs)) % self.size, min(self.n + len(obs), self.size)

    def sample(self, k, rng):
        j = rng.integers(self.n, size=k)
        return [torch.from_numpy(x[j].astype(np.float32) if x.dtype == np.float16 else x[j]).to(device)
                for x in (self.obs, self.mask, self.pi, self.z)]


def actor(rank, seed, trades, config, out, samples, weights, stop):
    """Self-play process: NUM_ENVS games with search; sends every finished game to the learner.
    Each weights message is (network weights, ids of the pool's snapshots in <run>/pool/)."""
    globals().update(config or {})
    torch.manual_seed(seed * 1000 + rank)  # seed * 1000 + rank: no stream shared with another run's actors
    torch.set_num_threads(1)
    agent = AZAgent(seed=seed * 1000 + rank)
    env = fc.BatchedEnv(NUM_ENVS, seed=seed * 1000 + rank)
    env.reset()
    rules = TradeRules(NUM_ENVS, trades)
    league = League(NUM_ENVS, P_SELFPLAY, seed=seed * 1000 + rank)

    def load(id):
        try:
            return AZAgent.load(out / "pool" / f"{id}.pt")
        except FileNotFoundError:  # just dropped from the pool: the next message has the new one
            return None

    def sync():
        state, ids = weights.get()
        agent.net.load_state_dict(state)
        league.sync(ids, load)

    sync()
    pending = [[] for _ in range(NUM_ENVS)]  # samples of unfinished games, per slot
    done, steps_in_game = np.zeros(NUM_ENVS, np.uint8), np.zeros(NUM_ENVS, np.int64)
    learner = os.getppid()
    running = lambda: not stop.is_set() and os.getppid() == learner  # a killed learner leaves no orphans
    while running():
        if not weights.empty():  # newer weights and pool from the learner
            sync()
        sims, games, lengths = agent.sims, [], []
        for _ in range(STEPS):  # the learner searches, pool opponents move in their seats
            obs, mask, seat, status = observe(env, NUM_ENVS)
            rules.apply(mask, seat, status[:, 2])
            idx = np.flatnonzero(league.learner_rows(seat))
            actions = np.zeros(NUM_ENVS, np.uint32)
            if len(idx):
                actions[idx], _ = agent.act(env, idx, obs[idx], mask[idx], seat[idx], rules)
                for r in np.flatnonzero(agent.last[1]):  # copies: a row view would keep the whole step alive
                    i = idx[r]
                    pending[i].append((obs[i].astype(np.float16), mask[i].copy(), agent.last[0][r].astype(np.float16),
                                       int(seat[i])))
            league.opponents_act(env, obs, mask, seat, rules, actions)
            env.step_autoreset(actions, done)
            rules.count(actions, seat)
            steps_in_game += 1
            for i in np.flatnonzero(done):
                winner = env.last_winner(int(i))
                z = np.zeros(4) if winner == fc.NO_PLAYER else np.where(np.arange(4) == winner, 1.0, -1.0)
                if pending[i]:
                    o, m, pi, s = zip(*pending[i])  # result per seat, relative to each sample's seat
                    games.append((np.array(o, np.float16), np.array(m), np.array(pi, np.float16),
                                  np.array([np.roll(z, -s_) for s_ in s], np.float32)))
                pending[i] = []
                league.game_over(i, winner)
                lengths.append((int(steps_in_game[i]), int(status[i, 2])))
                steps_in_game[i] = 0
        msg = dict(games=games, lengths=lengths, env_steps=STEPS * NUM_ENVS, sims=agent.sims - sims,
                   pool_winrate=float(np.mean(league.winrate)))
        while running():
            try:
                samples.put(msg, timeout=1)
                break
            except queue.Full:
                pass
    samples.cancel_join_thread()  # exit without waiting for unread messages


def train(name, seed=1, hours=None, trades=True, config=None, resume=False):
    """Learner: trains on the GPU from the games of ACTORS self-play processes. resume: as in ppo.train."""
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
        out = run_dir(name, dict(algo="az", seed=seed, hours=hours, trades=trades, **constants))
    rng = np.random.default_rng(rs)
    agent, rnd = AZAgent(seed=rs), RandomAgent(rs)
    opt = torch.optim.Adam(agent.net.parameters(), lr=LR, weight_decay=1e-4)
    replay = Replay(REPLAY)
    pool = League(0, P_SELFPLAY, seed=rs)  # which snapshots are in the pool; the actors hold the agents
    elapsed, env_steps, sim_steps, steps, total_games = 0.0, 0, 0, 0, 0
    snapshots = last_snapshot = 0  # snapshots taken; total_games at the last one
    if state:
        agent.net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        pool.load_state(state["league"], lambda id: id)  # the ids stand in for agents: the learner plays no games
        elapsed = state["elapsed"]
        env_steps, sim_steps, steps, total_games, snapshots, last_snapshot = (
            state[k] for k in ("env_steps", "sim_steps", "train_steps", "total_games", "snapshots", "last_snapshot"))
        log(out, dict(resumed=resumes, hours=elapsed / 3600, env_steps=env_steps))
    else:
        agent.save(out / f"ckpt_{0:012d}.pt", train_steps=0, env_steps=0)

    def message():
        return {k: v.detach().cpu() for k, v in agent.net.state_dict().items()}, list(pool.ids)

    ctx = torch.multiprocessing.get_context("spawn")
    samples, stop = ctx.Queue(maxsize=4 * ACTORS), ctx.Event()
    weights = [ctx.Queue() for _ in range(ACTORS)]
    for q in weights:
        q.put(message())
    actors = [ctx.Process(target=actor, args=(r, rs, trades, config, out, samples, weights[r], stop), daemon=True)
              for r in range(ACTORS)]
    for a in actors:
        a.start()

    start = time.time() - elapsed

    def save_state():
        save_resume(out, dict(resumes=resumes, elapsed=time.time() - start, env_steps=env_steps, sim_steps=sim_steps,
                              train_steps=steps, total_games=total_games, snapshots=snapshots,
                              last_snapshot=last_snapshot, net=agent.net.state_dict(), opt=opt.state_dict(),
                              league=pool.state()), pool.ids)

    save_state()
    next_ckpt = (env_steps // CKPT_EVERY + 1) * CKPT_EVERY
    now = time.time()
    last = dict(sync=now, log=now, monitor=now, resume=now)
    owed, losses, lengths, winrates = 0.0, [], [], []
    try:
        while time.time() - start < hours * 3600:
            if not all(a.is_alive() for a in actors):
                raise RuntimeError("an actor process died (its traceback is above); continue with --resume")
            try:
                msg = samples.get(timeout=1)
            except queue.Empty:
                msg = None
            if msg:
                for g in msg["games"]:
                    replay.add(*g)
                    owed = min(owed + len(g[0]) * REPLAY_RATIO / BATCH, 1000)
                env_steps, sim_steps = env_steps + msg["env_steps"], sim_steps + msg["sims"]
                lengths += msg["lengths"]
                total_games += len(msg["lengths"])
                winrates.append(msg["pool_winrate"])
            if total_games - last_snapshot >= SNAPSHOT_EVERY:  # the actors load it with the next weights
                snapshots, last_snapshot = snapshots + 1, total_games
                agent.save(out / "pool" / f"{snapshots}.pt", train_steps=steps, env_steps=env_steps)
                pool.add(None, snapshots)
            while owed >= 1 and replay.n >= 4 * BATCH:  # each sample is used about REPLAY_RATIO times
                obs, mask, pi, z = replay.sample(BATCH, rng)
                logits, v = agent.net(obs)
                logp = torch.log_softmax(logits.masked_fill(~mask, -1e8), -1)
                p_loss, v_loss = -(pi * logp).sum(-1).mean(), (v - z).pow(2).mean()
                opt.zero_grad()
                (p_loss + v_loss).backward()
                opt.step()
                losses.append((p_loss.item(), v_loss.item()))
                owed, steps = owed - 1, steps + 1

            now = time.time()
            if now - last["sync"] > SYNC_EVERY:
                update = message()
                for q in weights:
                    if q.empty():
                        q.put(update)
                last["sync"] = now
            if env_steps >= next_ckpt:  # kept for good, unlike the pool snapshots
                agent.save(out / f"ckpt_{next_ckpt:012d}.pt", train_steps=steps, env_steps=env_steps)
                next_ckpt = (env_steps // CKPT_EVERY + 1) * CKPT_EVERY
            if now - last["resume"] > RESUME_EVERY:
                save_state()
                last["resume"] = time.time()
            if now - last["log"] > LOG_EVERY:
                row = dict(train_steps=steps, env_steps=env_steps, sim_steps=sim_steps, hours=(now - start) / 3600,
                           games=len(lengths), total_games=total_games, replay=replay.n,
                           steps_per_turn=sum(s for s, _ in lengths) / max(1, sum(t for _, t in lengths)),
                           p_loss=float(np.mean([p for p, _ in losses])) if losses else None,
                           v_loss=float(np.mean([v for _, v in losses])) if losses else None,
                           pool_winrate=float(np.mean(winrates)) if winrates else None)
                if now - last["monitor"] > MONITOR_EVERY:
                    raw = AZAgent(agent.net, rs, n_sims=0)  # raw policy: cheap to play many games
                    row["vs_random_raw"] = wins(play([raw, rnd, raw, rnd], 50, 10_000, trades), [0, 2]) / 50
                    play([agent] * 4, 1, 20_000, trades, log_path=out / "games" / f"step_{steps:07d}.jsonl.gz")
                    last["monitor"] = time.time()
                log(out, row)
                losses, lengths, winrates, last["log"] = [], [], [], time.time()
    finally:
        stop.set()
        for a in actors:
            a.join(timeout=30)
            if a.is_alive():
                a.terminate()
    agent.save(out / f"ckpt_{env_steps:012d}.pt", train_steps=steps, env_steps=env_steps)
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
