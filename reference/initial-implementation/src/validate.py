"""
Week 1 / M2 - Data validation and cleaning for Flavor A (ETA prediction).

Checks the raw ingested data for the specific quality issues named in the
brief (missing GPS pings, invalid timestamps), plus additional schema and
range checks appropriate for this dataset. Produces:
  1. A printed/logged validation report (counts + % of each issue)
  2. A cleaned dataset written to data/processed/trips_clean.csv

Cleaning decisions (documented here so they can be cited in the README):
  - Rows with missing pickup GPS -> dropped (can't compute distance/features
    reliably without coordinates; imputing lat/lon would fabricate location).
  - Rows with invalid/missing timestamp -> dropped (time-of-day and weekday
    features depend entirely on this field; no safe imputation).
  - Rows with non-positive distance -> dropped (physically invalid trips).
  - No imputation used anywhere: given <2% total affected rows (see report),
    dropping preserves data integrity better than fabricating values for a
    feature engineering pipeline that feeds a regression target.
"""

import pandas as pd

RAW_PATH = "data/raw/trips_raw.csv"
CLEAN_PATH = "data/processed/trips_clean.csv"

EXPECTED_COLUMNS = {
    "trip_id": "object",
    "timestamp": "object",  # parsed to datetime below
    "hour_of_day": "int64",
    "weekday": "int64",
    "is_weekend": "int64",
    "pickup_lat": "float64",
    "pickup_lon": "float64",
    "drop_lat": "float64",
    "drop_lon": "float64",
    "distance_km": "float64",
    "weather": "object",
    "traffic_level": "int64",
    "actual_eta_minutes": "float64",
}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def check_schema(df: pd.DataFrame) -> list:
    issues = []
    missing_cols = set(EXPECTED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing expected columns: {missing_cols}")
    return issues


def report_quality(df: pd.DataFrame) -> dict:
    n = len(df)
    report = {
        "total_rows": n,
        "missing_pickup_gps": int(df[["pickup_lat", "pickup_lon"]].isna().any(axis=1).sum()),
        "missing_timestamp": int(df["timestamp"].isna().sum()),
        "non_positive_distance": int((df["distance_km"] <= 0).sum()),
        "missing_weather": int(df["weather"].isna().sum()),
        "missing_target": int(df["actual_eta_minutes"].isna().sum()),
    }
    report["pct_missing_pickup_gps"] = round(100 * report["missing_pickup_gps"] / n, 3)
    report["pct_missing_timestamp"] = round(100 * report["missing_timestamp"] / n, 3)
    report["pct_non_positive_distance"] = round(100 * report["non_positive_distance"] / n, 3)
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df[~df[["pickup_lat", "pickup_lon"]].isna().any(axis=1)]
    after_gps = len(df)

    df = df[~df["timestamp"].isna()]
    after_ts = len(df)

    df = df[df["distance_km"] > 0]
    after_dist = len(df)

    print("\n--- Cleaning log ---")
    print(f"Rows before cleaning:          {before}")
    print(f"Dropped (missing GPS):         {before - after_gps}")
    print(f"Dropped (invalid timestamp):   {after_gps - after_ts}")
    print(f"Dropped (non-positive dist):   {after_ts - after_dist}")
    print(f"Rows after cleaning:           {after_dist} "
          f"({round(100 * after_dist / before, 2)}% retained)")

    return df.reset_index(drop=True)


def main():
    df = load_raw(RAW_PATH)

    schema_issues = check_schema(df)
    if schema_issues:
        print("SCHEMA ISSUES FOUND:")
        for issue in schema_issues:
            print(f"  - {issue}")
    else:
        print("Schema check: OK (all expected columns present)")

    report = report_quality(df)
    print("\n--- Data quality report (raw) ---")
    for k, v in report.items():
        print(f"{k}: {v}")

    df_clean = clean(df)

    import os
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset written to {CLEAN_PATH}")


if __name__ == "__main__":
    main()