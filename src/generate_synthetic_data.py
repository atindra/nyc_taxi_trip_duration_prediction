"""
Generates a synthetic ride/delivery ETA dataset.
Fields: trip distance, time-of-day, weekday, weather, traffic proxy,
pickup/drop-off coordinates, and the target (actual_eta_minutes).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 20000  # number of trips

# --- Base features ---
start_date = datetime(2024, 1, 1)
timestamps = [start_date + timedelta(minutes=int(x)) for x in np.random.randint(0, 525600, N)]

hour_of_day = [t.hour for t in timestamps]
weekday = [t.weekday() for t in timestamps]  # 0=Mon, 6=Sun
is_weekend = [1 if d >= 5 else 0 for d in weekday]

# Pickup/drop-off coordinates (bounding box roughly resembling a city)
pickup_lat = np.random.uniform(12.90, 13.10, N)
pickup_lon = np.random.uniform(80.10, 80.30, N)
drop_lat = np.random.uniform(12.90, 13.10, N)
drop_lon = np.random.uniform(80.10, 80.30, N)

# Trip distance (km) - derived from coordinate delta + noise
distance_km = np.sqrt((drop_lat - pickup_lat)**2 + (drop_lon - pickup_lon)**2) * 111
distance_km = np.round(distance_km + np.random.normal(0, 0.5, N).clip(min=0), 2)
distance_km = distance_km.clip(min=0.3)

# Weather (categorical)
weather = np.random.choice(
    ["clear", "rain", "fog", "storm"],
    size=N,
    p=[0.65, 0.20, 0.10, 0.05]
)

# Traffic proxy (0=light, 1=moderate, 2=heavy) - correlated with hour + weekend
def traffic_level(hour, weekend):
    if weekend:
        return np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
    if hour in [8, 9, 18, 19, 20]:  # rush hours
        return np.random.choice([0, 1, 2], p=[0.1, 0.3, 0.6])
    elif hour in range(0, 6):
        return np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
    else:
        return np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])

traffic = [traffic_level(h, w) for h, w in zip(hour_of_day, is_weekend)]

# --- Target: ETA (minutes) ---
base_speed_kmph = 25  # baseline city speed
weather_penalty = {"clear": 1.0, "rain": 1.15, "fog": 1.25, "storm": 1.4}
traffic_penalty = {0: 1.0, 1: 1.3, 2: 1.7}

eta = []
for d, w, tr in zip(distance_km, weather, traffic):
    speed = base_speed_kmph / (weather_penalty[w] * traffic_penalty[tr])
    minutes = (d / speed) * 60
    noise = np.random.normal(0, 2)
    eta.append(max(2, round(minutes + noise, 1)))

# --- Assemble dataframe ---
df = pd.DataFrame({
    "trip_id": [f"T{i:06d}" for i in range(N)],
    "timestamp": timestamps,
    "hour_of_day": hour_of_day,
    "weekday": weekday,
    "is_weekend": is_weekend,
    "pickup_lat": pickup_lat,
    "pickup_lon": pickup_lon,
    "drop_lat": drop_lat,
    "drop_lon": drop_lon,
    "distance_km": distance_km,
    "weather": weather,
    "traffic_level": traffic,
    "actual_eta_minutes": eta,
})

# --- Inject realistic data quality issues (for the validation step in M2) ---
# Missing GPS pings
missing_idx = np.random.choice(df.index, size=int(0.01 * N), replace=False)
df.loc[missing_idx, ["pickup_lat", "pickup_lon"]] = np.nan

# Invalid timestamps (a few nulls)
bad_ts_idx = np.random.choice(df.index, size=int(0.005 * N), replace=False)
df.loc[bad_ts_idx, "timestamp"] = pd.NaT

# Negative/zero distance outliers
bad_dist_idx = np.random.choice(df.index, size=int(0.003 * N), replace=False)
df.loc[bad_dist_idx, "distance_km"] = -1

df.to_csv("data/raw/trips_raw.csv", index=False)
print(f"Generated {len(df)} rows -> data/raw/trips_raw.csv")
print(df.head())