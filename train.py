"""Train the Q-learning irrigation agent.

    python train.py               # 50,000 episodes, writes results/q_table.npy
    python train.py --episodes 2000
"""
import argparse
import csv
import os
import time

import numpy as np

from irrigation_env import IrrigationEnv
from q_agent import QLearningAgent, discretize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train(episodes=50000, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05,
          eps_decay_frac=0.7, seed=0):
    env = IrrigationEnv(seed=seed)
    agent = QLearningAgent(alpha=alpha, gamma=gamma, seed=seed)
    decay_episodes = int(episodes * eps_decay_frac)
    log = []

    for ep in range(episodes):
        # Linear epsilon decay, then hold at eps_end.
        frac = min(ep / decay_episodes, 1.0)
        epsilon = eps_start + frac * (eps_end - eps_start)

        # Training seeds start at 0; evaluate.py uses seeds from 1,000,000.
        obs = env.reset(seed=seed + ep)
        state = discretize(obs)
        total_reward, water, done = 0.0, 0.0, False
        while not done:
            action = agent.act(state, epsilon)
            obs, reward, done, info = env.step(action)
            next_state = discretize(obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            water += info["water_used"]
        log.append((ep, epsilon, total_reward, water))

        if (ep + 1) % 5000 == 0:
            recent = np.mean([r for _, _, r, _ in log[-2000:]])
            print(f"episode {ep + 1:>6}  epsilon {epsilon:.3f}  avg return (last 2000) {recent:7.2f}")

    return agent, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    start = time.time()
    agent, log = train(episodes=args.episodes, seed=args.seed)
    print(f"trained in {time.time() - start:.1f}s")

    agent.save(os.path.join(RESULTS, "q_table.npy"))
    with open(os.path.join(RESULTS, "training_log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "epsilon", "return", "water_used"])
        writer.writerows(log)
    visited = np.count_nonzero(np.any(agent.q != 0, axis=1))
    print(f"states visited: {visited}/{agent.q.shape[0]}")


if __name__ == "__main__":
    main()
