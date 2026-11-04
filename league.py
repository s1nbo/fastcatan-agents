"""League: frozen past copies of the learner as opponents, picked by PFSP (against cycling).

Each game slot gets a lineup when its game starts. With probability p_selfplay the learner
plays all four seats; otherwise it plays seats {0,2} or {1,3} against one pool opponent.
Only the learner's seats are trained on. Every trainer adds a snapshot each SNAPSHOT_EVERY
finished games (`games`), so the pools of the algorithms are comparable per game played.
Each snapshot has an id; its weights are in <run>/pool/<id>.pt while it is in the pool, so a
resumed run (and every AlphaZero actor) can rebuild the pool from `ids`.
"""

import copy

import fastcatan as fc
import numpy as np

from common import RandomAgent


class League:
    def __init__(self, n, p_selfplay, size=20, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n, self.p_selfplay, self.size = n, p_selfplay, size
        self.pool = [RandomAgent(seed)]  # entry 0 is the random anchor and is never evicted
        self.ids = [None]  # snapshot id of each entry (None: the random anchor)
        self.winrate = [0.5]  # EMA of the learner's win share against each pool entry
        self.opponent = np.full(n, -1)  # pool entry per game slot, -1 = pure self-play
        self.learner = np.ones((n, 4), bool)  # the learner's seats per game slot
        self.games = 0  # finished games so far, every lineup
        for i in range(n):
            self.assign(i)

    def add(self, agent, id=None):
        """Freeze a copy of the learner into the pool as snapshot `id`; a random entry goes when it is full."""
        j = len(self.pool) if len(self.pool) < self.size else int(self.rng.integers(1, self.size))
        self.put(j, copy.deepcopy(agent), id)

    def put(self, j, agent, id):
        """Entry j (or a new last entry) becomes `agent`, with fresh PFSP statistics."""
        if j == len(self.pool):
            self.pool.append(None)
            self.ids.append(None)
            self.winrate.append(0.5)
        self.pool[j], self.ids[j], self.winrate[j] = agent, id, 0.5

    def sync(self, ids, load):
        """Hold the snapshots `ids`, entry by entry: after a resume, and in AlphaZero actors (the
        learner picks their pool). load(id) gives the agent of snapshot `id`, or None if its file
        is gone; the remaining entries then wait for the next sync."""
        for j, id in enumerate(ids):
            if j < len(self.ids) and self.ids[j] == id:
                continue
            agent = load(id)
            if agent is None:
                return
            self.put(j, agent, id)

    def state(self):
        return dict(ids=list(self.ids), winrate=list(self.winrate), games=self.games)

    def load_state(self, state, load):
        """Restore a state() (load as in sync)."""
        self.sync(state["ids"], load)
        self.winrate = list(state["winrate"][: len(self.pool)])
        self.games = state["games"]

    def pfsp(self):
        """Opponent probabilities: favour the opponents the learner still loses to."""
        w = (1 - np.array(self.winrate)) ** 2 + 0.05
        return w / w.sum()

    def assign(self, i):
        """Draw the lineup of game slot i."""
        if self.rng.random() < self.p_selfplay:
            self.opponent[i], self.learner[i] = -1, True
        else:
            self.opponent[i] = self.rng.choice(len(self.pool), p=self.pfsp())
            self.learner[i] = np.arange(4) % 2 == self.rng.integers(2)

    def learner_rows(self, seat):
        """For every game slot: does the learner make the current move?"""
        return self.learner[np.arange(self.n), seat]

    def opponents_act(self, env, obs, mask, seat, rules, actions):
        """Overwrite `actions` in the rows where a pool opponent moves."""
        theirs = ~self.learner_rows(seat)
        for j in np.unique(self.opponent[theirs]):
            idx = np.flatnonzero(theirs & (self.opponent == j))
            actions[idx] = self.pool[j].act(env, idx, obs[idx], mask[idx], seat[idx], rules)[0]

    def game_over(self, i, winner):
        """Update the PFSP statistics with slot i's finished game and draw its next lineup."""
        self.games += 1
        j = self.opponent[i]
        if j >= 0:
            won = 0.5 if winner == fc.NO_PLAYER else float(self.learner[i, winner])
            self.winrate[j] = 0.95 * self.winrate[j] + 0.05 * won
        self.assign(i)
