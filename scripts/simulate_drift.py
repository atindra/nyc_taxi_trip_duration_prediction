"""Simulate a rush-hour surge window and measure drift with the trained model.

Reference window: the validated trips as-is.
Shifted window: same trips moved to evening rush hour with longer distances
and a congestion multiplier on actual durations. Predictions always come from
the deployed champion model, never from the actual durations.
"""

from pathlib import Path
import sys

import joblib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.transform import TaxiFeatureTransformer
from src.monitoring.monitor import compare_windows
from sklearn.metrics import mean_absolute_error


def main() -> None:
    data = pd.read_csv(ROOT / "data/processed/validated_trips.csv", parse_dates=["pickup_datetime"])
    model = joblib.load(ROOT / "models/champion.joblib")

    def predict_seconds(frame: pd.DataFrame) -> pd.Series:
        predictions = model.predict(frame)
        return pd.Series(predictions, index=frame.index).clip(lower=0)

    # Reference window: real distribution, real predictions, real actuals.
    reference_features = TaxiFeatureTransformer().fit_transform(data)
    reference = reference_features.assign(
        prediction_seconds=predict_seconds(data).to_numpy(),
        actual_seconds=data["trip_duration"].to_numpy(),
    )

    # Shifted window: evening rush hour (17:00), +40% trip distances, and a 30%
    # congestion multiplier on actual durations -> covariate shift in inputs and
    # concept drift in the input->duration relationship.
    shifted_raw = data.copy()
    shifted_raw["pickup_datetime"] = shifted_raw["pickup_datetime"].apply(
        lambda ts: ts.replace(hour=17, minute=30)
    )
    # +40% distance: scale the dropoff OFFSETS in the raw inputs so the model's
    # predictions and the monitoring features both derive from the same perturbed
    # coordinates (distance_km is computed from coords in TaxiFeatureTransformer).
    for axis in ("latitude", "longitude"):
        shifted_raw[f"dropoff_{axis}"] = (
            shifted_raw[f"pickup_{axis}"]
            + (shifted_raw[f"dropoff_{axis}"] - shifted_raw[f"pickup_{axis}"]) * 1.4
        )
    shifted_features = TaxiFeatureTransformer().fit_transform(shifted_raw)
    shifted_predictions = predict_seconds(shifted_raw)
    shifted_actuals = data["trip_duration"] * 1.3  # congestion: same trips now take longer
    shifted = shifted_features.assign(
        prediction_seconds=shifted_predictions.to_numpy(),
        actual_seconds=shifted_actuals.to_numpy(),
    )

    drift = compare_windows(reference, shifted)
    performance = {
        "reference_mae_seconds": float(mean_absolute_error(reference["actual_seconds"], reference["prediction_seconds"])),
        "shifted_mae_seconds": float(mean_absolute_error(shifted["actual_seconds"], shifted["prediction_seconds"])),
    }
    performance["mae_change_ratio"] = round(
        performance["shifted_mae_seconds"] / performance["reference_mae_seconds"], 3
    )

    result = {"drift_metrics": drift, "performance": performance}
    output = ROOT / "monitoring/drift_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
