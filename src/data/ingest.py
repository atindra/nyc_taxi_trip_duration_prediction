"""Deterministic ingestion for trip and date-keyed weather records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

TRIP_COLUMNS = {
    "id": "trip_id",
    "pickup_datetime": "pickup_datetime",
    "dropoff_datetime": "dropoff_datetime",
    "passenger_count": "passenger_count",
    "pickup_longitude": "pickup_longitude",
    "pickup_latitude": "pickup_latitude",
    "dropoff_longitude": "dropoff_longitude",
    "dropoff_latitude": "dropoff_latitude",
    "trip_duration": "trip_duration",
}


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame]
    return frame


def ingest(trips_path: str | Path, weather_path: str | Path, output_path: str | Path, report_path: str | Path) -> pd.DataFrame:
    """Load, normalize, and join the raw inputs without dropping rows."""

    trips = _normalise_columns(pd.read_csv(trips_path))
    weather = _normalise_columns(pd.read_csv(weather_path))
    missing = sorted(set(TRIP_COLUMNS) - set(trips.columns))
    if missing:
        raise ValueError(f"Trip input is missing required columns: {missing}")
    required_weather = {"date", "temp_c", "precipitation_mm", "weather"}
    missing_weather = sorted(required_weather - set(weather.columns))
    if missing_weather:
        raise ValueError(f"Weather input is missing required columns: {missing_weather}")

    trips = trips.rename(columns=TRIP_COLUMNS)
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"], errors="coerce")
    trips["dropoff_datetime"] = pd.to_datetime(trips["dropoff_datetime"], errors="coerce")
    # Use the pickup date as the stable key for the daily weather join.
    trips["weather_date"] = trips["pickup_datetime"].dt.strftime("%Y-%m-%d")
    weather["weather_date"] = pd.to_datetime(weather["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    weather = weather[["weather_date", "temp_c", "precipitation_mm", "weather"]].drop_duplicates("weather_date")
    result = trips.merge(weather, on="weather_date", how="left", validate="many_to_one")
    result.insert(0, "row_id", range(len(result)))
    result.to_csv(output_path, index=False)

    report = {
        "input_rows": int(len(trips)),
        "output_rows": int(len(result)),
        "columns": list(result.columns),
        "weather_unmatched_rows": int(result["temp_c"].isna().sum()),
        "output_sha256": hashlib.sha256(Path(output_path).read_bytes()).hexdigest(),
    }
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return result
