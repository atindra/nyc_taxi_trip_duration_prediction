"""
Week 4 / M5 - Drift monitoring for Flavor A (ETA prediction).

Compares the reference dataset (training data) against the drifted
dataset (simulated surge scenario) using Evidently, and against logged
live predictions if available. Produces an HTML drift report.
"""

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/processed/trips_features.csv"
DRIFTED_PATH = "data/drift/trips_drifted.csv"
REPORT_OUT = "logs/drift_report.html"

FEATURE_COLUMNS = [
    "hour_of_day", "weekday", "is_weekend", "distance_km", "traffic_level",
    "weather_clear", "weather_fog", "weather_rain", "weather_storm",
]


def main():
    reference = pd.read_csv(REFERENCE_PATH)[FEATURE_COLUMNS]
    drifted = pd.read_csv(DRIFTED_PATH)[FEATURE_COLUMNS]

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=drifted)

    import os
    os.makedirs("logs", exist_ok=True)
    result.save_html(REPORT_OUT)

    print(f"Drift report saved to {REPORT_OUT}")

    result_dict = result.dict()
    print("\n--- Drift summary ---")
    print(result_dict)


if __name__ == "__main__":
    main()