"""
Week 4 / M5 - Retraining trigger check.

Reads the Evidently drift report's underlying data and applies the
documented retraining trigger rule (see RETRAINING_TRIGGER.md).
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/processed/trips_features.csv"
DRIFTED_PATH = "data/drift/trips_drifted.csv"

FEATURE_COLUMNS = [
    "hour_of_day", "weekday", "is_weekend", "distance_km", "traffic_level",
    "weather_clear", "weather_fog", "weather_rain", "weather_storm",
]

HIGH_IMPORTANCE_FEATURES = ["distance_km", "traffic_level"]
DATASET_DRIFT_SHARE_THRESHOLD = 0.5
FEATURE_DRIFT_SCORE_THRESHOLD = 0.3


def main():
    reference = pd.read_csv(REFERENCE_PATH)[FEATURE_COLUMNS]
    drifted = pd.read_csv(DRIFTED_PATH)[FEATURE_COLUMNS]

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=drifted)
    result_dict = result.dict()

    dataset_share = None
    feature_scores = {}

    for m in result_dict["metrics"]:
        if "DriftedColumnsCount" in m["metric_name"]:
            dataset_share = m["value"]["share"]
        elif "ValueDrift" in m["metric_name"]:
            col = m["config"]["column"]
            if col in HIGH_IMPORTANCE_FEATURES:
                feature_scores[col] = float(m["value"])

    print(f"Dataset drift share: {dataset_share:.3f} (threshold: {DATASET_DRIFT_SHARE_THRESHOLD})")
    for col, score in feature_scores.items():
        print(f"{col} drift score: {score:.3f} (threshold: {FEATURE_DRIFT_SCORE_THRESHOLD})")

    condition_1 = dataset_share is not None and dataset_share > DATASET_DRIFT_SHARE_THRESHOLD
    condition_2 = any(score > FEATURE_DRIFT_SCORE_THRESHOLD for score in feature_scores.values())

    if condition_1 or condition_2:
        print("\nRETRAIN TRIGGERED")
        if condition_1:
            print(f"  - Reason: dataset drift share ({dataset_share:.3f}) exceeds threshold")
        if condition_2:
            fired = [c for c, s in feature_scores.items() if s > FEATURE_DRIFT_SCORE_THRESHOLD]
            print(f"  - Reason: high-importance feature drift on {fired}")
    else:
        print("\nNo retraining needed")


if __name__ == "__main__":
    main()