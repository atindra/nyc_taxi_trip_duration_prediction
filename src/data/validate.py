"""Data quality checks and invalid-row quarantine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "row_id", "trip_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "trip_duration", "temp_c", "precipitation_mm", "weather",
}


def validate(frame: pd.DataFrame, quarantine_path: str | Path, report_path: str | Path) -> pd.DataFrame:
    """Return valid rows and write every rejected row with a reason."""

    data = frame.copy()
    reasons = pd.Series("", index=data.index, dtype="object")

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")

    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
    data["dropoff_datetime"] = pd.to_datetime(data["dropoff_datetime"], errors="coerce")
    numeric_columns = [
        "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
        "dropoff_latitude", "trip_duration", "temp_c", "precipitation_mm",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    def add_failure(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask] = reasons.loc[mask].where(reasons.loc[mask] == "", reasons.loc[mask] + ";") + reason

    add_failure(data["row_id"].isna() | data["row_id"].duplicated(keep=False), "row_id_missing_or_duplicate")
    add_failure(data["pickup_datetime"].isna() | data["dropoff_datetime"].isna(), "invalid_timestamp")
    add_failure(data["pickup_longitude"].isna() | data["dropoff_longitude"].isna() | data["pickup_latitude"].isna() | data["dropoff_latitude"].isna(), "missing_gps")
    add_failure((data["pickup_longitude"].notna() & ~data["pickup_longitude"].between(-74.3, -73.6)) | (data["dropoff_longitude"].notna() & ~data["dropoff_longitude"].between(-74.3, -73.6)) | (data["pickup_latitude"].notna() & ~data["pickup_latitude"].between(40.4, 41.1)) | (data["dropoff_latitude"].notna() & ~data["dropoff_latitude"].between(40.4, 41.1)), "coordinates_out_of_range")
    add_failure(data["passenger_count"].isna() | ~data["passenger_count"].between(1, 6), "invalid_passenger_count")
    add_failure(data["trip_duration"].isna() | ~data["trip_duration"].between(30, 7200), "invalid_trip_duration")
    add_failure(data["dropoff_datetime"] <= data["pickup_datetime"], "dropoff_not_after_pickup")
    add_failure(data["temp_c"].isna() | data["precipitation_mm"].isna() | data["weather"].isna(), "missing_weather")
    add_failure(data["precipitation_mm"].notna() & ~data["precipitation_mm"].between(0, 500), "invalid_precipitation")
    add_failure(data["weather"].notna() & ~data["weather"].isin(["clear", "cloudy", "rain", "snow"]), "invalid_weather_category")

    invalid = data.loc[reasons != ""].copy()
    invalid["validation_errors"] = reasons.loc[reasons != ""].values
    valid = data.loc[reasons == ""].copy()
    Path(quarantine_path).parent.mkdir(parents=True, exist_ok=True)
    invalid.to_csv(quarantine_path, index=False)
    report = {
        "status": "passed" if invalid.empty else "passed_with_quarantine",
        "input_rows": int(len(data)),
        "valid_rows": int(len(valid)),
        "quarantined_rows": int(len(invalid)),
        "failure_counts": {
            reason: int(reasons.str.contains(reason, regex=False).sum())
            for reason in sorted({item for value in reasons for item in value.split(";") if item})
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return valid.reset_index(drop=True)
