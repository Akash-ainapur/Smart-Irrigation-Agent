"""Sanity checks for the simulator and agent. Run: python test_env.py"""
import numpy as np

from irrigation_env import IrrigationEnv
from q_agent import N_STATES, QLearningAgent, discretize


def test_moisture_stays_in_bounds():
    env = IrrigationEnv(seed=1)
    obs, done = env.reset(), False
    rng = np.random.default_rng(1)
    while not done:
        obs, _, done, info = env.step(int(rng.integers(3)))
        assert 0.0 <= info["true_moisture"] <= 100.0


def test_episode_length():
    env = IrrigationEnv(days=5)
    env.reset(seed=0)
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    assert steps == 5 * env.steps_per_day


def test_budget_is_never_exceeded():
    env = IrrigationEnv(water_budget=30.0)
    env.reset(seed=0)
    used, done = 0.0, False
    while not done:
        _, _, done, info = env.step(2)   # always heavy
        used += info["water_used"]
    assert abs(used - 30.0) < 1e-9
    assert env.budget_left == 0.0


def test_same_seed_same_season():
    a, b = IrrigationEnv(), IrrigationEnv()
    a.reset(seed=42), b.reset(seed=42)
    for _ in range(40):
        ra = a.step(1)[1]
        rb = b.step(1)[1]
        assert ra == rb


def test_reward_shape():
    env = IrrigationEnv()
    assert env._moisture_reward(55.0) == 1.0
    assert env._moisture_reward(30.0) == -1.0
    assert env._moisture_reward(80.0) == -1.0


def test_discretize_covers_all_states():
    env = IrrigationEnv()
    states = set()
    for seed in range(50):
        obs, done = env.reset(seed=seed), False
        while not done:
            s = discretize(obs)
            assert 0 <= s < N_STATES
            states.add(s)
            obs, _, done, _ = env.step(seed % 3)
    assert len(states) > 50


def test_q_update_moves_toward_target():
    agent = QLearningAgent(alpha=0.5, gamma=0.0)
    agent.update(0, 1, reward=2.0, next_state=1, done=False)
    assert agent.q[0, 1] == 1.0


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
