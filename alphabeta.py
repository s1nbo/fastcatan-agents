"""Alpha-Beta baseline with Perfect Information Monte Carlo (PIMC) over hidden cards.

For each move: K times, resample everything this seat cannot see (Env.determinize), run a
depth-limited alpha-beta search on that sample, and add up each legal action's score. The
best total wins. The true hidden state is never searched.

Search: paranoid (this seat maximizes its value, every other seat minimizes it), expectation
over chance outcomes with Env.enumerate_outcomes, player trades masked for every seat (the
benchmark is played with player trades off). The value function is our own code over the
feature list and weights of Catanatron's AlphaBetaPlayer (GPL-3.0, so no code is copied).
"""

import fastcatan as fc
import numpy as np

from common import one_hot, trade_mask

K, DEPTH = 8, 2
WIN = 1e20

# Static board structure (the same for every game).
_topo = fc.topology()
HEX_NODES = np.zeros((fc.NUM_NODES, fc.NUM_HEXES))  # node touches hex
for _h, _nodes in enumerate(_topo["hex_nodes"]):
    HEX_NODES[_nodes, _h] = 1
NODE_EDGES = np.zeros((fc.NUM_NODES, fc.NUM_EDGES))  # node is an end of edge
for _e, (_a, _b) in enumerate(_topo["edge_nodes"]):
    NODE_EDGES[[_a, _b], _e] = 1
NEIGHBORS = (NODE_EDGES @ NODE_EDGES.T > 0) & ~np.eye(fc.NUM_NODES, dtype=bool)
DICE = np.array([0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36  # P(dice sum = n)


def value(obs):
    """Catanatron-style value of a position, from the observing seat's own observation."""
    players = obs[:68].reshape(4, 17)  # per seat: vp/10, hand/25, devs/10, knights/10, road/15, ...
    hand = obs[68:73] * 19  # own brick, lumber, wool, grain, ore
    level = obs[83:515].reshape(54, 4, 2) @ np.array([1.0, 2.0])  # [node, seat]: 1 settlement, 2 city
    roads = obs[515:803].reshape(72, 4)
    number = np.rint(obs[917:936] * 12).astype(int)
    hex_yield = obs[803:917].reshape(19, 6)[:, :5] * (DICE[number] * (1 - obs[990:1009]))[:, None]
    node_yield = HEX_NODES @ hex_yield  # expected cards per roll for a settlement on each node
    prod = level.T @ node_yield  # [seat, resource]
    production = prod.sum(1) + (prod > 0).sum(1) / 36  # small bonus for resource variety

    occupied = level.sum(1) > 0
    free = ~occupied & ~(NEIGHBORS @ occupied)  # empty and the distance rule holds
    my_roads = NODE_EDGES @ roads[:, 0] > 0
    buildable = free & my_roads
    open_edges = roads.sum(1) == 0
    my_net = my_roads | (level[:, 0] > 0)
    reach1 = free & ~buildable & (NODE_EDGES @ (open_edges & (NODE_EDGES.T @ my_net > 0)) > 0)

    handsize = players[0, 1] * 25
    city_gap = (max(2 - hand[3], 0) + max(3 - hand[4], 0)) / 5
    settle_gap = sum(max(1 - hand[r], 0) for r in range(4)) / 4
    public_vp = players[0, 0] * 10 - obs[74] * 10  # own VP minus own hidden VP cards
    return (3e14 * public_vp
            + 1e8 * production[0] - 1e8 * production[1:].mean()
            + 1 * (HEX_NODES.T @ (level[:, 0] > 0) > 0).sum()
            + 1e4 * node_yield[reach1].sum()
            + 1e3 * buildable.sum()
            + (10 if not buildable.any() else 0.1) * players[0, 4] * 15
            + 1e2 * (2 - city_gap - settle_gap) / 2
            + 1 * handsize - 5 * (handsize > 7)
            + 10 * players[0, 2] * 10 + 10.1 * players[0, 3] * 10)


class AlphaBeta:
    def __init__(self, k=K, seed=0):
        self.k, self.name = k, f"alphabeta-k{k}"
        self.rng = np.random.default_rng(seed)
        self.env = fc.Env()  # scratch copy; the real game is never searched
        self.obs = np.zeros(fc.OBS_SIZE, np.float32)
        self.mask = np.zeros(fc.NUM_ACTIONS, bool)

    def act(self, env, idx, obs, mask, seat, rules):
        actions = np.zeros(len(idx), np.uint32)
        for r, i in enumerate(idx):  # AB never trades with players, whatever the seat's rules say
            legal = np.flatnonzero(trade_mask(mask[r].copy(), False, 0, 0))
            actions[r] = self.choose(env.snapshot(int(i)), int(seat[r]), legal)
        return actions, None

    def probs(self, env, idx, obs, mask, seat, rules):
        return one_hot(self.act(env, idx, obs, mask, seat, rules)[0])

    def choose(self, snapshot, me, legal):
        """Best action on average over K determinizations of the real game."""
        if len(legal) == 1:
            return legal[0]
        score = np.zeros(len(legal))
        for _ in range(self.k):
            self.env.load_snapshot(snapshot)
            self.env.determinize(me, int(self.rng.integers(2**63)))
            sample = self.env.snapshot()
            score += [self.action_value(sample, a, me, DEPTH, -np.inf, np.inf) for a in legal]
        return legal[np.argmax(score)]

    def action_value(self, snapshot, action, me, depth, alpha, beta):
        """Expected search value of `action`, over its chance outcomes."""
        self.env.load_snapshot(snapshot)
        outcomes = self.env.enumerate_outcomes(int(action))
        if len(outcomes) == 1:
            return self.search(outcomes[0][1], me, depth - 1, alpha, beta)
        return sum(p * self.search(s, me, depth - 1, -np.inf, np.inf) for p, s in outcomes)

    def search(self, snapshot, me, depth, alpha, beta):
        env = self.env
        env.load_snapshot(snapshot)
        if env.done:
            return 0.0 if env.truncated else (WIN if env.winner == me else -WIN)
        if depth == 0:
            env.write_obs(me, self.obs)
            return value(self.obs)
        env.action_mask_dense(self.mask)
        legal = np.flatnonzero(trade_mask(self.mask, False, 0, 0))
        maximize = env.actor_to_act == me
        best = -np.inf if maximize else np.inf
        for a in legal:
            v = self.action_value(snapshot, a, me, depth, alpha, beta)
            if maximize:
                best = max(best, v)
                alpha = max(alpha, best)
            else:
                best = min(best, v)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best
