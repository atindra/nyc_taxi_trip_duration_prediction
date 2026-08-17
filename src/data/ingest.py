"""Deterministic ingestion for trip and date-keyed weather records."""

import json
from pathlib import Path

import pandas as pd


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame]
    return frame


def ingest(trips_path: str | Path, weather_path: str | Path, output_path: str | Path, report_path: str | Path) -> pd.DataFrame:
    """Load, normalize, and join the raw inputs without dropping rows."""

    trips = _normalise_columns(pd.read_csv(trips_path))
    weather = _normalise_columns(pd.read_csv(weather_path))

    if "id" not in trips.columns:
        raise ValueError("Trip input is missing required column: id")
    required_weather = {"date", "temp_c", "precipitation_mm", "weather"}
    missing_weather = sorted(required_weather - set(weather.columns))
    if missing_weather:
        raise ValueError(f"Weather input is missing required columns: {missing_weather}")

    trips = trips.rename(columns={"id": "trip_id"})
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"], errors="coerce")
    trips["dropoff_datetime"] = pd.to_datetime(trips["dropoff_datetime"], errors="coerce")
    # Use the pickup date as the stable key for the daily weather join.
    trips["weather_date"] = trips["pickup_datetime"].dt.strftime("%Y-%m-%d")
    weather["weather_date"] = pd.to_datetime(weather["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    weather = weather[["weather_date", "temp_c", "precipitation_mm", "weather"]].drop_duplicates("weather_date")
    result = trips.merge(weather, on="weather_date", how="left", validate="many_to_one")
    result.to_csv(output_path, index=False)

    report = {
        "input_rows": int(len(trips)),
        "output_rows": int(len(result)),
        "weather_unmatched_rows": int(result["temp_c"].isna().sum()),
    }
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return result
