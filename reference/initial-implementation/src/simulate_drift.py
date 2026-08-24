"""
Week 4 / M5 - Drift simulation for Flavor A (ETA prediction).

Simulates a "festival/rush-hour surge" scenario: traffic skews heavier,
and weather skews toward rain/storm more than the training distribution.
This represents the kind of distribution shift the brief names explicitly.
"""

import numpy as np
import pandas as pd

np.random.seed(99)
N = 3000

hour_of_day = np.random.choice(range(24), N)
weekday = np.random.choice(range(7), N)
is_weekend = [1 if d >= 5 else 0 for d in weekday]

distance_km = np.round(np.random.uniform(0.5, 15, N), 2)

# Drift: traffic skews much heavier than training (surge conditions)
traffic_level = np.random.choice([0, 1, 2], size=N, p=[0.1, 0.25, 0.65])

# Drift: weather skews toward rain/storm more than training's 65/20/10/5 split
weather = np.random.choice(
    ["clear", "rain", "fog", "storm"],
    size=N,
    p=[0.30, 0.35, 0.15, 0.20]
)

weather_penalty = {"clear": 1.0, "rain": 1.15, "fog": 1.25, "storm": 1.4}
traffic_penalty = {0: 1.0, 1: 1.3, 2: 1.7}
base_speed_kmph = 25

eta = []
for d, w, tr in zip(distance_km, weather, traffic_level):
    speed = base_speed_kmph / (weather_penalty[w] * traffic_penalty[tr])
    minutes = (d / speed) * 60
    noise = np.random.normal(0, 2)
    eta.append(max(2, round(minutes + noise, 1)))

df = pd.DataFrame({
    "trip_id": [f"D{i:06d}" for i in range(N)],
    "hour_of_day": hour_of_day,
    "weekday": weekday,
    "is_weekend": is_weekend,
    "distance_km": distance_km,
    "traffic_level": traffic_level,
    "weather": weather,
    "actual_eta_minutes": eta,
})

df = pd.get_dummies(df, columns=["weather"], prefix="weather")

for col in ["weather_clear", "weather_fog", "weather_rain", "weather_storm"]:
    if col not in df.columns:
        df[col] = False

import os
os.makedirs("data/drift", exist_ok=True)
df.to_csv("data/drift/trips_drifted.csv", index=False)
print(f"Drifted dataset written: {len(df)} rows -> data/drift/trips_drifted.csv")
print(df.head())