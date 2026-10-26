"""Tabular Q-learning agent for the irrigation environment.

The observation is continuous (moisture, budget), so it is cut into bins to
get a small discrete state space:
  moisture bins (10) x weather (3) x time of day (4) x budget-per-day bins (4)
  = 480 states, 3 actions.
"""
import numpy as np

MOISTURE_EDGES = np.array([20, 30, 35, 40, 45, 50, 60, 70, 80])  # finer near the dry edge of the target
BUDGET_EDGES = np.array([2.5, 5.0, 7.5])                  # water units per remaining day

N_MOISTURE = len(MOISTURE_EDGES) + 1
N_WEATHER = 3
N_TOD = 4
N_BUDGET = len(BUDGET_EDGES) + 1
N_STATES = N_MOISTURE * N_WEATHER * N_TOD * N_BUDGET
N_ACTIONS = 3


def discretize(obs):
    m = int(np.digitize(obs["moisture"], MOISTURE_EDGES))
    b = int(np.digitize(obs["budget_per_day"], BUDGET_EDGES))
    return np.ravel_multi_index((m, obs["weather"], obs["time_of_day"], b),
                                (N_MOISTURE, N_WEATHER, N_TOD, N_BUDGET))


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros((N_STATES, N_ACTIONS))
        self.rng = np.random.default_rng(seed)

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        # Break ties randomly so unvisited states don't always pick "off".
        best = np.flatnonzero(self.q[state] == self.q[state].max())
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def policy(self, obs):
        """Greedy policy on a raw observation, for evaluation."""
        return self.act(discretize(obs), epsilon=0.0)

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
