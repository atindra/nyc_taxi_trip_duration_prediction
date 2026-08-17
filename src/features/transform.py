"""Leakage-safe feature construction shared by training and serving."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PREDICTION_INPUTS = [
    "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "passenger_count", "temp_c", "precipitation_mm", "weather",
]


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = [np.radians(series.astype(float)) for series in (lat1, lon1, lat2, lon2)]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return earth_radius_km * 2 * np.arcsin(np.sqrt(a))


class TaxiFeatureTransformer(BaseEstimator, TransformerMixin):
    """Convert raw prediction inputs to numeric and categorical model features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TaxiFeatureTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        pickup = pd.to_datetime(frame["pickup_datetime"], errors="coerce")
        hour = pickup.dt.hour.fillna(0).astype(float)
        weekday = pickup.dt.dayofweek.fillna(0).astype(float)
        distance = haversine_km(
            frame["pickup_latitude"], frame["pickup_longitude"],
            frame["dropoff_latitude"], frame["dropoff_longitude"],
        )
        return pd.DataFrame({
            "pickup_hour": hour,
            "weekday": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "is_rush_hour": hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "distance_km": distance,
            "pickup_longitude": pd.to_numeric(frame["pickup_longitude"], errors="coerce"),
            "pickup_latitude": pd.to_numeric(frame["pickup_latitude"], errors="coerce"),
            "dropoff_longitude": pd.to_numeric(frame["dropoff_longitude"], errors="coerce"),
            "dropoff_latitude": pd.to_numeric(frame["dropoff_latitude"], errors="coerce"),
            "passenger_count": pd.to_numeric(frame["passenger_count"], errors="coerce"),
            "temp_c": pd.to_numeric(frame["temp_c"], errors="coerce"),
            "precipitation_mm": pd.to_numeric(frame["precipitation_mm"], errors="coerce"),
            "weather": frame["weather"].fillna("unknown").astype(str),
        })


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Keep future rows out of training so evaluation matches deployment timing.
    ordered = frame.sort_values("pickup_datetime").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = train_end + int(len(ordered) * validation_fraction)
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]
