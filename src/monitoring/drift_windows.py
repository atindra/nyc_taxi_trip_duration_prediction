"""Reference/shifted window construction shared by the drift tooling.

Reference window: the validated trips as-is.
Shifted window: same trips moved to evening rush hour (17:30) with trip
distances stretched 40% and a 30% congestion multiplier on actual durations ->
covariate shift in inputs and concept drift in the input->duration relationship.
Predictions always come from the deployed champion model, never from the
actual durations.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features.transform import TaxiFeatureTransformer

ROOT = Path(__file__).resolve().parents[2]


def _predict_seconds(model, frame: pd.DataFrame) -> pd.Series:
    predictions = model.predict(frame)
    return pd.Series(predictions, index=frame.index).clip(lower=0)


def build_drift_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(ROOT / "data/processed/validated_trips.csv", parse_dates=["pickup_datetime"])
    model = joblib.load(ROOT / "models/champion.joblib")

    reference_features = TaxiFeatureTransformer().fit_transform(data)
    reference = reference_features.assign(
        prediction_seconds=_predict_seconds(model, data).to_numpy(),
        actual_seconds=data["trip_duration"].to_numpy(),
    )

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
    shifted_actuals = data["trip_duration"] * 1.3
    shifted = shifted_features.assign(
        prediction_seconds=_predict_seconds(model, shifted_raw).to_numpy(),
        actual_seconds=shifted_actuals.to_numpy(),
    )
    return reference, shifted
