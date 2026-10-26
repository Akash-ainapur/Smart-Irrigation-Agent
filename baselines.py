"""Hand-written policies to compare the learned agent against.

threshold_policy mirrors the KRISHIBODH robot's smart mode: read the moisture
sensor and give one watering pulse only if the soil is dry. fixed_schedule
mirrors the robot's fixed mode: water on a timer whatever the soil says.
"""
import numpy as np

OFF, LIGHT, HEAVY = 0, 1, 2


def threshold_policy(obs, dry_below=40.0):
    return LIGHT if obs["moisture"] < dry_below else OFF


def fixed_schedule(obs):
    # One light watering every morning.
    return LIGHT if obs["time_of_day"] == 1 else OFF


def never_water(obs):
    return OFF


def make_random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(3))
