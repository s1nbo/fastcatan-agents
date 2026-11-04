"""Shared pieces for every agent: observations, trade rules, playing games, small helpers.

Agent interface (every agent has a `name` and this method):

    actions, values = agent.act(env, idx, obs, mask, seat, rules)
    probs = agent.probs(env, idx, obs, mask, seat, rules)   # [k, NUM_ACTIONS], for probes

`env` is the BatchedEnv, `idx` the game slots this agent moves in, `obs` / `mask` / `seat`
the rows of those slots, `rules` the TradeRules. `values` (or None) are shown in the viewer.

Every entry script sets OMP_NUM_THREADS=1 (unless it is already set) before it imports
numpy, torch or fastcatan. Each brings its own OpenMP pool with one thread per core, and
together they made fastcatan's batched reads 10-25x slower (PPO ran 1.5x slower overall).
"""

import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import fastcatan as fc
import numpy as np
import torch
from torch import nn

torch.set_float32_matmul_precision("high")  # TF32 matmuls (Ampere+): PPO 1024 x 5 ran 1.2x faster

A = fc.action
ADD = slice(A.TRADE_ADD_GIVE_BASE, A.TRADE_ADD_WANT_BASE + 5)  # add one card to an offer
PLAYER_TRADES = slice(A.TRADE_ADD_GIVE_BASE, A.TRADE_CANCEL + 1)
MAX_ADDS, MAX_OFFERS = 10, 3  # per seat per turn
ROOT = Path(__file__).resolve().parent
FASTCATAN = ROOT.parent / "fastcatan"


def observe(env, n):
    """Obs and legal mask from each game's acting seat, that seat, and the status rows."""
    obs = np.zeros((n, fc.OBS_SIZE), np.float32)
    mask = np.zeros((n, fc.NUM_ACTIONS), bool)
    seat = np.zeros(n, np.uint8)
    status = np.zeros((n, fc.STATUS_INTS), np.int32)  # done, winner, turn_count, current_player, truncated
    env.write_obs(obs)
    env.write_masks_dense(mask)
    env.write_actors(seat)
    env.write_status(status)
    return obs, mask, seat, status


def trade_mask(mask, on, adds, offers):
    """The agent-side trade rules, in place, for one mask row or a batch of rows.

    on: the acting seat may trade with players. adds / offers: what it used this turn.
    A trade-off seat never builds or opens an offer and declines every offer it gets.
    Bank and port trades are never touched.
    """
    rows = mask.reshape(-1, fc.NUM_ACTIONS)  # a view: writes go to `mask`
    on, adds, offers = (np.broadcast_to(x, len(rows)) for x in (on, adds, offers))
    rows[~on | (adds >= MAX_ADDS) | (offers >= MAX_OFFERS), ADD] = False
    rows[~on | (offers >= MAX_OFFERS), A.TRADE_OPEN] = False
    rows[~on, A.TRADE_ACCEPT] = False
    return mask


class TradeRules:
    """Trade on/off per (game, seat) and the per-turn caps, counted on the agent side."""

    def __init__(self, n, trades_on=True):
        self.on = np.zeros((n, 4), bool)
        self.on[:] = trades_on  # True / False, one flag per seat, or one per (game, seat)
        self.adds = np.zeros((n, 4), np.int32)
        self.offers = np.zeros((n, 4), np.int32)
        self.turn = np.full(n, -1)

    def apply(self, mask, seat, turn):
        """Before acting: reset the counters of games on a new turn, then mask."""
        new = turn != self.turn
        self.adds[new] = 0
        self.offers[new] = 0
        self.turn = turn.copy()
        rows = np.arange(len(seat))
        return trade_mask(mask, self.on[rows, seat], self.adds[rows, seat], self.offers[rows, seat])

    def count(self, actions, seat):
        """After acting: count the trade-building actions and offers."""
        rows = np.arange(len(seat))
        self.adds[rows, seat] += (actions >= ADD.start) & (actions < ADD.stop)
        self.offers[rows, seat] += actions == A.TRADE_OPEN


class RandomAgent:
    """Uniformly random legal action."""

    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, idx, obs, mask, seat, rules):
        return np.where(mask, self.rng.random(mask.shape), -1).argmax(1).astype(np.uint32), None

    def probs(self, env, idx, obs, mask, seat, rules):
        return mask / mask.sum(1, keepdims=True)


def one_hot(actions):
    out = np.zeros((len(actions), fc.NUM_ACTIONS), np.float32)
    out[np.arange(len(actions)), actions] = 1
    return out


def play(agents, games, seed, trades_on=True, log_path=None, events=None):
    """Play full games with agents[s] in seat s. Returns the final status rows.

    log_path: play the games one at a time and save them for fastcatan-viz.
    events: an open fastcatan.events.EventWriter that receives the trade event log.
    """
    if log_path is None:
        env = fc.BatchedEnv(games, seed=seed)
        if events:
            env.enable_log(fc.log.LOG_ALL, capacity_per_env=256)
        env.reset()
        return run(agents, env, TradeRules(games, trades_on), events=events)
    from fastcatan.viz import GameLogWriter

    status = []
    with GameLogWriter(log_path) as log:
        for g in range(games):
            env = fc.BatchedEnv(1, seed=seed + g)
            env.reset()
            rec = log.start_game(env, env_idx=0, agents=[a.name for a in agents])
            status.append(run(agents, env, TradeRules(1, trades_on), rec=rec))
    return np.concatenate(status)


def run(agents, env, rules, rec=None, events=None, on_step=None):
    """Play every game in `env` to the end. Returns the final status rows.

    on_step(env, obs, mask, seat, status, actions) is called before each step; returning
    True stops early.
    """
    n = env.num_envs
    done = np.zeros(n, np.uint8)
    while True:
        obs, mask, seat, status = observe(env, n)
        active = status[:, 0] == 0
        if not active.any():
            return status
        rules.apply(mask, seat, status[:, 2])
        assert mask[active].any(1).all(), "a game has no legal action"
        actions = np.full(n, fc.NO_ACTION, np.uint32)  # finished games stay as they are
        values = np.full(n, np.nan, np.float32)
        for agent in dict.fromkeys(agents):  # each distinct agent moves once, for all its rows
            idx = np.flatnonzero(active & np.isin(seat, [s for s in range(4) if agents[s] is agent]))
            if len(idx):
                actions[idx], v = agent.act(env, idx, obs[idx], mask[idx], seat[idx], rules)
                if v is not None:
                    values[idx] = v
        if on_step and on_step(env, obs, mask, seat, status, actions):
            return status
        env.step(actions, done)
        rules.count(actions, seat)
        if events:
            drain_trades(env, events)
        if rec:
            rec.record(int(actions[0]), info=None if np.isnan(values[0]) else {"value": round(float(values[0]), 3)})


EV_TYPE, EV_ACTION = fc.EVENT_FIELDS["event_type"][0], fc.EVENT_FIELDS["action"][0]


def drain_trades(env, writer):
    """Write trade and game rows, plus END_TURN rows (the VP at each turn, for offer rates).

    The other moves are dropped, so these logs are for analysis, not for replay."""
    rows = np.empty((env.pending_events(), fc.EVENT_INTS), np.int32)
    rows = rows[: env.drain_events(rows)]
    assert env.log_overflow == 0, "event log overflowed"
    writer.write(rows[(rows[:, EV_TYPE] != fc.log.ACTION) | (rows[:, EV_ACTION] == A.END_TURN)])


def wins(status, seats):
    """Games won by any of `seats` (a truncated game counts for nobody)."""
    return int(np.isin(status[:, 1], seats).sum())


def wilson(k, n, z=1.96):
    """Wilson 95% interval for k successes in n trials."""
    p, z2 = k / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return float(center - half), float(center + half)


def run_dir(name, config):
    """Create runs/<name>/ (never reused) and write config.json with the fastCatan build."""
    path = ROOT / "runs" / name
    (path / "games").mkdir(parents=True)
    (path / "pool").mkdir()

    def git(*args):
        out = subprocess.run(["git", "-C", str(FASTCATAN), *args], capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        print(f"warning: {FASTCATAN} is not a fastCatan checkout; config.json gets no commit", flush=True)
    cache = sorted(FASTCATAN.glob("build/*/CMakeCache.txt"), key=lambda p: p.stat().st_mtime)
    compiler = re.search(r"CMAKE_CXX_COMPILER:\w+=(.*)", cache[-1].read_text()) if cache else None
    config = dict(config, fastcatan=fc.__version__, fastcatan_commit=commit,
                  fastcatan_dirty=None if commit is None else bool(git("status", "--porcelain")),
                  compiler=compiler.group(1) if compiler else None, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  host=platform.node(), omp_threads=os.environ.get("OMP_NUM_THREADS"),
                  started=time.strftime("%Y-%m-%d %H:%M"))
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return path


def resume_run(name, hours=None):
    """For --resume: (run dir, its config.json, its resume.pt), or None if runs/<name>/ is new.

    `hours`, if given, becomes the run's new total budget. Metrics rows logged after resume.pt was
    written (work the crash threw away) are dropped, so metrics.jsonl keeps one timeline."""
    path = ROOT / "runs" / name
    if not path.exists():
        return None
    if not (path / "resume.pt").exists():
        raise SystemExit(f"{path} has no resume.pt to continue from; delete it to start over")
    config = json.loads((path / "config.json").read_text())
    if hours is not None and hours != config["hours"]:
        config["hours"] = hours
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    state = torch.load(path / "resume.pt", map_location="cpu", weights_only=True)

    def before_save(line):
        try:
            return json.loads(line).get("hours", 0) <= state["elapsed"] / 3600
        except json.JSONDecodeError:  # a line cut off by the crash
            return False

    metrics = path / "metrics.jsonl"
    rows = metrics.read_text().splitlines() if metrics.exists() else []
    metrics.write_text("".join(r + "\n" for r in rows if before_save(r)))
    print(f"resuming {path} at {state['elapsed'] / 3600:.2f} of {config['hours']} hours, with its config.json",
          flush=True)
    return path, config, state


def save_resume(path, state, pool_ids):
    """Write <run>/resume.pt atomically (a crash mid-write keeps the previous one), then delete the
    pool snapshots the league no longer holds. Snapshots evicted since the previous save survive
    until this one, so every file that resume.pt names exists."""
    torch.save(state, path / "resume.tmp")
    os.replace(path / "resume.tmp", path / "resume.pt")
    for f in (path / "pool").glob("*.pt"):
        if int(f.stem) not in pool_ids:
            f.unlink()


class Trunk(nn.Module):
    """Obs -> `depth` hidden layers of `width` units, the body of every network.

    Every layer after the first is residual, h + Linear(ReLU(LayerNorm(h))), so deep nets still
    train. The output is ReLU(LayerNorm(h)); heads are plain Linear layers on top.
    """

    def __init__(self, width, depth):
        super().__init__()
        self.first = nn.Linear(fc.OBS_SIZE, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, width))
                                    for _ in range(depth - 1))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.ReLU())

    def forward(self, x):
        h = self.first(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def log(path, row):
    """Print one metrics row and append it to <path>/metrics.jsonl."""
    print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    with open(path / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
