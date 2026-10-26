"""Soil moisture simulator for the irrigation agent.

One step is 6 hours, one episode is 30 days. Each step the agent picks off,
light or heavy watering. The water budget is for the whole season; once it is
used up, watering does nothing.
"""
import numpy as np

WEATHER = ("hot", "mild", "rainy")
TIME_OF_DAY = ("night", "morning", "afternoon", "evening")
ACTIONS = ("off", "light", "heavy")

# Water units per action. One unit raises soil moisture by about one point.
WATER_PER_ACTION = (0.0, 5.0, 10.0)

# Fraction of applied water that reaches the roots, by time of day.
IRRIGATION_EFFICIENCY = (0.95, 0.85, 0.60, 0.90)

# Moisture lost per step to evaporation: weather base rate x time-of-day factor.
EVAP_BASE = (2.5, 1.5, 0.5)
EVAP_TOD_FACTOR = (0.3, 1.0, 1.6, 0.7)
PLANT_UPTAKE = 0.3

# Weather changes once a day. Rows: today, columns: tomorrow.
WEATHER_TRANSITION = np.array([
    [0.6, 0.3, 0.1],   # hot
    [0.3, 0.5, 0.2],   # mild
    [0.2, 0.4, 0.4],   # rainy
])
RAIN_PROB_PER_STEP = 0.6
RAIN_AMOUNT = (2.0, 6.0)

FIELD_CAPACITY = 80.0
DRAIN_RATE = 0.5


class IrrigationEnv:
    steps_per_day = 4

    def __init__(self, days=30, water_budget=180.0, target=(40.0, 70.0),
                 water_cost=0.03, sensor_noise=1.5, seed=None):
        self.days = days
        self.max_steps = days * self.steps_per_day
        self.water_budget = water_budget
        self.target = target
        self.water_cost = water_cost
        self.sensor_noise = sensor_noise
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.moisture = self.rng.uniform(35.0, 65.0)
        self.weather = self.rng.choice(3, p=[0.4, 0.4, 0.2])
        self.budget_left = self.water_budget
        return self._observe()

    def _observe(self):
        """Noisy moisture reading, weather, time of day, budget left per remaining day."""
        reading = self.moisture + self.rng.normal(0.0, self.sensor_noise)
        days_left = max((self.max_steps - self.t) / self.steps_per_day, 0.25)
        return {
            "moisture": float(np.clip(reading, 0.0, 100.0)),
            "weather": int(self.weather),
            "time_of_day": self.t % self.steps_per_day,
            "budget_per_day": self.budget_left / days_left,
        }

    def step(self, action):
        tod = self.t % self.steps_per_day

        # 1. Irrigation, limited by what is left in the budget.
        water = min(WATER_PER_ACTION[action], self.budget_left)
        self.budget_left -= water
        absorbed = water * IRRIGATION_EFFICIENCY[tod]
        lost_on_application = water - absorbed
        self.moisture += absorbed

        # 2. Rain.
        rain = 0.0
        if WEATHER[self.weather] == "rainy" and self.rng.random() < RAIN_PROB_PER_STEP:
            rain = self.rng.uniform(*RAIN_AMOUNT)
            self.moisture += rain

        # 3. Evaporation and plant uptake.
        evap = EVAP_BASE[self.weather] * EVAP_TOD_FACTOR[tod] * self.rng.uniform(0.8, 1.2)
        self.moisture -= evap + PLANT_UPTAKE

        # 4. Drainage above field capacity.
        drained = 0.0
        if self.moisture > FIELD_CAPACITY:
            drained = (self.moisture - FIELD_CAPACITY) * DRAIN_RATE
            self.moisture -= drained
        self.moisture = float(np.clip(self.moisture, 0.0, 100.0))

        reward = self._moisture_reward(self.moisture) - self.water_cost * water

        self.t += 1
        if self.t % self.steps_per_day == 0:
            self.weather = self.rng.choice(3, p=WEATHER_TRANSITION[self.weather])
        done = self.t >= self.max_steps

        info = {
            "true_moisture": self.moisture,
            "water_used": water,
            "lost_on_application": lost_on_application,
            "drained": drained,
            "rain": rain,
        }
        return self._observe(), reward, done, info

    def _moisture_reward(self, m):
        low, high = self.target
        if m < low:
            return -(low - m) / 10.0
        if m > high:
            return -(m - high) / 10.0
        return 1.0
