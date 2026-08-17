"""Deterministically generate the synthetic trips + weather dataset.

This IS the project's dataset (not a fallback): the brief explicitly permits a
synthetic delivery dataset. A fixed seed makes every byte of the output
reproducible; the generated CSVs are committed to git so a fresh clone needs
no downloads. Rerun this script only if the dataset itself should change,
then rerun `dvc repro --force`.

Design goals:
- Plausible NYC trip generation: rush hours, weekday/weekend, distance-based
  durations, weather effects, seasonal temperature.
- Deliberately messy: a small fraction (~1.5%) of rows violate schema rules
  (out-of-range GPS, impossible durations, missing timestamps) so the
  validation/quarantine stage has real work to do.
- Small enough to git-commit: ~50k trips ≈ 4 MB CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SEED = 7
N_TRIPS = 50_000
BAD_FRACTION = 0.015
START, END = "2016-01-01", "2016-06-30"

# NYC bounding box used by the validation stage (keep pickups inside it).
PICKUP_LAT_MIN, PICKUP_LAT_MAX = 40.55, 40.90
PICKUP_LON_MIN, PICKUP_LON_MAX = -74.05, -73.70


def _weather_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Daily weather: seasonal temperature curve + noisy precipitation."""
    dates = pd.date_range(START, END, freq="D")
    day_index = np.arange(len(dates))
    seasonal = 2 + 13 * np.sin(2 * np.pi * (day_index - 105) / 365)  # winter->summer 2016
    temp_c = seasonal + rng.normal(0, 2.5, size=len(dates))
    rainy = rng.random(len(dates)) < 0.28
    snowy = (temp_c < 0.5) & (rng.random(len(dates)) < 0.4)
    weather = np.where(snowy, "snow", np.where(rainy, "rain", np.where(rng.random(len(dates)) < 0.4, "cloudy", "clear")))
    precipitation = np.where(rainy | snowy, rng.gamma(2.0, 3.0, len(dates)), 0.0)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temp_c": temp_c.round(1),
        "precipitation_mm": precipitation.round(1),
        "weather": weather,
    })


def _clean_trips(rng: np.random.Generator, weather: pd.DataFrame) -> pd.DataFrame:
    """Plausible trips joined conceptually to the weather frame via pickup date."""
    n = N_TRIPS
    day_offsets = rng.integers(0, len(weather), size=n)
    pickup_dates = pd.to_datetime(weather["date"].to_numpy()[day_offsets])
    # Hour distribution: morning + evening rush peaks, quieter midday/night.
    weights = np.array([
        2, 1, 1, 1, 1, 2,      # 0-5 night
        3, 6, 7, 6,            # 6-9 morning rush
        5, 5, 5, 5, 5, 5,      # 10-15 midday
        6, 7, 8, 8,            # 16-19 evening rush
        6, 5, 3, 2,            # 20-23 evening
    ], dtype=float)
    hour = rng.choice(np.arange(24), size=n, p=weights / weights.sum())
    minute = rng.integers(0, 60, size=n)
    second = rng.integers(0, 60, size=n)
    pickup = pd.Series(pickup_dates + pd.to_timedelta(hour * 3600 + minute * 60 + second, unit="s"))

    weekday = pickup.dt.dayofweek.to_numpy()
    is_rush = np.isin(hour, [7, 8, 9, 16, 17, 18, 19])

    pickup_lat = rng.uniform(PICKUP_LAT_MIN, PICKUP_LAT_MAX, size=n)
    pickup_lon = rng.uniform(PICKUP_LON_MIN, PICKUP_LON_MAX, size=n)
    # Trip distance ~ lognormal-ish (many short, few long), then convert to a
    # coordinate offset with random bearing.
    distance_km = np.clip(rng.lognormal(mean=0.9, sigma=0.7, size=n), 0.3, 30.0)
    bearing = rng.uniform(0, 2 * np.pi, size=n)
    dropoff_lat = pickup_lat + (distance_km / 111.0) * np.cos(bearing)
    dropoff_lon = pickup_lon + (distance_km / 84.0) * np.sin(bearing)

    weather_lookup = weather.set_index("date")
    day_str = pickup.dt.strftime("%Y-%m-%d")
    temp_c = weather_lookup.loc[day_str, "temp_c"].to_numpy()
    precipitation = weather_lookup.loc[day_str, "precipitation_mm"].to_numpy()

    # Duration model: base + per-km with *non-linear* interactions (rush-hour
    # multiplies congestion by distance; precipitation slows long trips more),
    # plus moderate noise — so gradient boosting has real structure to exploit.
    base = 240.0
    per_km = 95.0
    duration = (
        base
        + per_km * distance_km * (1.0 + 0.35 * is_rush)                    # rush: congestion scales with trip length
        + np.where(weekday >= 5, 30.0, 0.0)
        + 0.02 * distance_km * precipitation                               # rain hurts long trips disproportionately
        + 120.0 * np.clip(distance_km - 8.0, 0, None)                      # very long trips hit highway/express traffic
        + rng.normal(0, 90.0, size=n)
    )
    duration = np.clip(duration, 60.0, 5400.0)
    dropoff = pickup + pd.to_timedelta(duration, unit="s")

    return pd.DataFrame({
        "id": [f"trip-{i:06d}" for i in range(n)],
        "pickup_datetime": pickup.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "dropoff_datetime": dropoff.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "passenger_count": rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.6, 0.2, 0.08, 0.06, 0.04, 0.02]),
        "pickup_longitude": pickup_lon.round(6),
        "pickup_latitude": pickup_lat.round(6),
        "dropoff_longitude": dropoff_lon.round(6),
        "dropoff_latitude": dropoff_lat.round(6),
        "trip_duration": duration.round(0).astype(int),
    })


def _inject_bad_rows(rng: np.random.Generator, trips: pd.DataFrame) -> pd.DataFrame:
    """Corrupt ~1.5% of rows so validation has real quarantine work."""
    n_bad = int(len(trips) * BAD_FRACTION)
    bad_index = rng.choice(trips.index, size=n_bad, replace=False)
    kinds = np.array_split(bad_index, 5)

    trips.loc[kinds[0], "pickup_latitude"] += rng.uniform(2.0, 5.0, size=len(kinds[0]))  # coords out of range
    trips.loc[kinds[1], "trip_duration"] = rng.integers(-900, 10, size=len(kinds[1]))   # impossible duration
    trips.loc[kinds[2], "pickup_datetime"] = "not-a-timestamp"                          # invalid timestamp
    trips.loc[kinds[3], "passenger_count"] = 0                                          # invalid passengers
    trips.loc[kinds[4], "dropoff_datetime"] = trips.loc[kinds[4], "pickup_datetime"]    # dropoff <= pickup
    return trips


def main() -> None:
    rng = np.random.default_rng(SEED)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    weather = _weather_frame(rng)
    trips = _inject_bad_rows(rng, _clean_trips(rng, weather))

    trips.to_csv(raw / "trips.csv", index=False)
    weather.to_csv(raw / "weather.csv", index=False)
    print(f"wrote {len(trips):,} trips + {len(weather)} weather days to {raw}")


if __name__ == "__main__":
    main()
