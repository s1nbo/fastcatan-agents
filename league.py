"""League: frozen past copies of the learner as opponents, picked by PFSP (against cycling).

Each game slot gets a lineup when its game starts. With probability p_selfplay the learner
plays all four seats; otherwise it plays seats {0,2} or {1,3} against one pool opponent.
Only the learner's seats are trained on.
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
        self.winrate = [0.5]  # EMA of the learner's win share against each pool entry
        self.opponent = np.full(n, -1)  # pool entry per game slot, -1 = pure self-play
        self.learner = np.ones((n, 4), bool)  # the learner's seats per game slot
        for i in range(n):
            self.assign(i)

    def add(self, agent):
        """Freeze a copy of the learner into the pool; a random entry goes when it is full."""
        if len(self.pool) < self.size:
            self.pool.append(None)
            self.winrate.append(0.5)
            j = len(self.pool) - 1
        else:
            j = int(self.rng.integers(1, self.size))
        self.pool[j], self.winrate[j] = copy.deepcopy(agent), 0.5

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
        j = self.opponent[i]
        if j >= 0:
            won = 0.5 if winner == fc.NO_PLAYER else float(self.learner[i, winner])
            self.winrate[j] = 0.95 * self.winrate[j] + 0.05 * won
        self.assign(i)
