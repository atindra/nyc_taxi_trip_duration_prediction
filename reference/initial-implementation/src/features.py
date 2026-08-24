"""
Week 1 / M2 - Feature engineering for Flavor A (ETA prediction).

Takes the cleaned dataset (data/processed/trips_clean.csv) and produces a
model-ready feature matrix.

Feature decisions (documented for README/justification):
  - hour_of_day, weekday, is_weekend, distance_km, traffic_level:
    kept as-is (already numeric, already meaningful on a linear scale
    for tree-based and linear models alike).
  - weather: one-hot encoded (categorical, no ordinal relationship).
  - pickup_lat/lon, drop_lat/lon, timestamp: dropped from the model matrix.
    distance_km already summarizes the spatial relationship between pickup
    and drop-off: keeping raw lat/lon as flat features would add coordinate
    dependence without a clear linear/tree-splittable relationship to ETA,
    and timestamp is fully represented by hour_of_day/weekday/is_weekend.
  - trip_id: retained separately (not fed to the model) for traceability
    back to individual trips during evaluation/debugging.
  - Cyclical (sin/cos) encoding of hour_of_day was considered but not used,
    to keep the first-pass feature set simple; noted here as a possible
    future improvement in the model comparison writeup.
"""

import os
import pandas as pd

CLEAN_PATH = "data/processed/trips_clean.csv"
FEATURES_PATH = "data/processed/trips_features.csv"

NUMERIC_FEATURES = ["hour_of_day", "weekday", "is_weekend", "distance_km", "traffic_level"]
CATEGORICAL_FEATURES = ["weather"]
TARGET = "actual_eta_minutes"
ID_COL = "trip_id"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df[[ID_COL] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()

    df_feat = pd.get_dummies(df_feat, columns=CATEGORICAL_FEATURES, prefix="weather")

    return df_feat


def main():
    df = pd.read_csv(CLEAN_PATH)
    df_feat = build_features(df)

    os.makedirs("data/processed", exist_ok=True)
    df_feat.to_csv(FEATURES_PATH, index=False)

    print(f"Feature matrix written to {FEATURES_PATH}")
    print(f"Shape: {df_feat.shape}")
    print(f"Columns: {list(df_feat.columns)}")
    print(df_feat.head())


if __name__ == "__main__":
    main()