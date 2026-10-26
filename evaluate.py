"""Compare the trained agent with the baseline policies on unseen seasons.

    python evaluate.py            # needs results/q_table.npy from train.py

Writes results/metrics.json and the plots.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines import fixed_schedule, make_random_policy, never_water, threshold_policy
from irrigation_env import ACTIONS, TIME_OF_DAY, WEATHER, IrrigationEnv
from q_agent import MOISTURE_EDGES, N_MOISTURE, QLearningAgent

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EVAL_SEED_START = 1_000_000   # far away from the training seeds


def run_episode(env, policy, seed):
    obs = env.reset(seed=seed)
    done = False
    trace = {"moisture": [], "action": [], "water": [], "rain": [], "reward": []}
    totals = {"return": 0.0, "water_used": 0.0, "lost_on_application": 0.0,
              "drained": 0.0, "in_range": 0, "too_dry": 0, "too_wet": 0,
              "budget_ran_out": False}
    low, high = env.target
    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        m = info["true_moisture"]
        totals["return"] += reward
        for k in ("water_used", "lost_on_application", "drained"):
            totals[k] += info[k]
        totals["in_range"] += low <= m <= high
        totals["too_dry"] += m < low
        totals["too_wet"] += m > high
        trace["moisture"].append(m)
        trace["action"].append(action)
        trace["water"].append(info["water_used"])
        trace["rain"].append(info["rain"])
        trace["reward"].append(reward)
    totals["budget_ran_out"] = env.budget_left < 1e-9
    return totals, trace


def evaluate(policy, episodes=500):
    env = IrrigationEnv()
    rows = [run_episode(env, policy, EVAL_SEED_START + i)[0] for i in range(episodes)]
    steps = env.max_steps
    return {
        "return_mean": float(np.mean([r["return"] for r in rows])),
        "return_std": float(np.std([r["return"] for r in rows])),
        "in_range_pct": 100 * float(np.mean([r["in_range"] for r in rows])) / steps,
        "too_dry_pct": 100 * float(np.mean([r["too_dry"] for r in rows])) / steps,
        "too_wet_pct": 100 * float(np.mean([r["too_wet"] for r in rows])) / steps,
        "water_used": float(np.mean([r["water_used"] for r in rows])),
        "lost_on_application": float(np.mean([r["lost_on_application"] for r in rows])),
        "drained": float(np.mean([r["drained"] for r in rows])),
        "budget_ran_out_pct": 100 * float(np.mean([r["budget_ran_out"] for r in rows])),
    }


def tune_threshold(candidates=range(35, 65, 5), episodes=200):
    """Best dry threshold for the simple rule, chosen on training seeds."""
    env = IrrigationEnv()
    best, best_return = None, -np.inf
    for thr in candidates:
        policy = lambda obs, thr=thr: threshold_policy(obs, dry_below=thr)
        ret = np.mean([run_episode(env, policy, seed)[0]["return"] for seed in range(episodes)])
        if ret > best_return:
            best, best_return = thr, ret
    return best


def plot_learning_curve(path_csv, path_png, window=500):
    with open(path_csv) as f:
        returns = np.array([float(row["return"]) for row in csv.DictReader(f)])
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(returns, alpha=0.15, color="tab:blue", label="episode return")
    plt.plot(np.arange(window - 1, len(returns)), smooth, color="tab:blue",
             label=f"{window}-episode average")
    plt.xlabel("training episode")
    plt.ylabel("return")
    plt.title("Q-learning training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=130)
    plt.close()


def typical_seed(policy, n=100):
    """Eval seed with the median return for this policy (used for the example plot)."""
    env = IrrigationEnv()
    seeds = [EVAL_SEED_START + i for i in range(n)]
    returns = [run_episode(env, policy, s)[0]["return"] for s in seeds]
    return seeds[int(np.argsort(returns)[n // 2])]


def plot_episode(policies, path_png, seed):
    env = IrrigationEnv()
    fig, axes = plt.subplots(len(policies), 1, figsize=(10, 2.6 * len(policies)), sharex=True)
    for ax, (name, policy) in zip(axes, policies.items()):
        totals, trace = run_episode(env, policy, seed)
        days = np.arange(len(trace["moisture"])) / env.steps_per_day
        ax.axhspan(*env.target, color="tab:green", alpha=0.12, label="target range")
        ax.plot(days, trace["moisture"], color="tab:brown", label="soil moisture")
        # Only mark steps where water actually flowed (not after the budget ran out).
        acts = np.where(np.array(trace["water"]) > 0, np.array(trace["action"]), 0)
        for a, marker, label in ((1, "v", "light watering"), (2, "s", "heavy watering")):
            idx = np.flatnonzero(acts == a)
            ax.scatter(days[idx], np.full(len(idx), 95), marker=marker, s=18,
                       color="tab:blue", label=label)
        rain = np.array(trace["rain"]) > 0
        ax.scatter(days[rain], np.full(rain.sum(), 5), marker="|", s=40,
                   color="tab:gray", label="rain")
        ax.set_ylim(0, 100)
        ax.set_ylabel("moisture")
        ax.set_title(f"{name}: return {totals['return']:.1f}, water used "
                     f"{totals['water_used']:.0f}, in range "
                     f"{100 * totals['in_range'] / env.max_steps:.0f}%", fontsize=10)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.18), fontsize=8, ncol=5)
    axes[-1].set_xlabel("day")
    fig.suptitle(f"One typical unseen season (seed {seed})", y=1.0)
    fig.tight_layout()
    fig.savefig(path_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_policy(agent, path_png, budget_bin=2):
    """Greedy action for every moisture bin x time of day, one panel per weather."""
    q = agent.q.reshape(N_MOISTURE, 3, 4, -1, 3)
    labels = [f"<{MOISTURE_EDGES[0]}"] + \
             [f"{a}-{b}" for a, b in zip(MOISTURE_EDGES[:-1], MOISTURE_EDGES[1:])] + \
             [f">{MOISTURE_EDGES[-1]}"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    cmap = matplotlib.colors.ListedColormap(["#eeeeee", "#9ecae1", "#2171b5"])
    for w, ax in enumerate(axes):
        best = q[:, w, :, budget_bin, :].argmax(axis=-1)
        visited = np.any(q[:, w, :, budget_bin, :] != 0, axis=-1)
        ax.imshow(best, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")
        for i in range(best.shape[0]):
            for j in range(best.shape[1]):
                text = ACTIONS[best[i, j]] if visited[i, j] else "-"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
        ax.set_xticks(range(4), TIME_OF_DAY, rotation=30)
        ax.set_title(f"weather: {WEATHER[w]}")
    axes[0].set_yticks(range(N_MOISTURE), labels)
    axes[0].set_ylabel("moisture reading")
    fig.suptitle("Learned policy (budget 5-7.5 units per remaining day; '-' = never visited)")
    fig.tight_layout()
    fig.savefig(path_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    agent = QLearningAgent()
    agent.load(os.path.join(RESULTS, "q_table.npy"))

    best_thr = tune_threshold()
    print(f"best threshold on training seeds: {best_thr}")

    policies = {
        "Q-learning agent": agent.policy,
        "Threshold rule, dry < 40 (robot smart mode)": threshold_policy,
        f"Threshold rule, tuned (dry < {best_thr})":
            lambda obs: threshold_policy(obs, dry_below=best_thr),
        "Fixed schedule (robot fixed mode)": fixed_schedule,
        "Never water": never_water,
        "Random": make_random_policy(seed=0),
    }
    metrics = {name: evaluate(p, args.episodes) for name, p in policies.items()}

    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    header = ("| Policy | Return (mean ± std) | In range % | Too dry % | Too wet % "
              "| Water used | Lost on application | Drained | Budget ran out % |")
    lines = [f"Evaluated on {args.episodes} unseen 30-day seasons (seeds "
             f"{EVAL_SEED_START}+). Water budget = {IrrigationEnv().water_budget:.0f} "
             "units per season.", "",
             header, "|" + "---|" * 9]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['return_mean']:.1f} ± {m['return_std']:.1f} "
                     f"| {m['in_range_pct']:.1f} | {m['too_dry_pct']:.1f} "
                     f"| {m['too_wet_pct']:.1f} | {m['water_used']:.1f} "
                     f"| {m['lost_on_application']:.1f} | {m['drained']:.1f} "
                     f"| {m['budget_ran_out_pct']:.1f} |")
    table = "\n".join(lines)
    print(table)

    plot_learning_curve(os.path.join(RESULTS, "training_log.csv"),
                        os.path.join(RESULTS, "learning_curve.png"))
    plot_episode({k: policies[k] for k in list(policies)[:3]},
                 os.path.join(RESULTS, "example_season.png"),
                 seed=typical_seed(agent.policy))
    plot_policy(agent, os.path.join(RESULTS, "learned_policy.png"))
    print("plots written to results/")


if __name__ == "__main__":
    main()
