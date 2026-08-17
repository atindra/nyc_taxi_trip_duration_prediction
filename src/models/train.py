"""Train the candidate models, compare them, and save the best one."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transform import TaxiFeatureTransformer, chronological_split

NUMERIC_FEATURES = [
    "pickup_hour", "weekday", "is_weekend", "is_rush_hour", "hour_sin", "hour_cos",
    "distance_km", "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "passenger_count", "temp_c", "precipitation_mm",
]
CATEGORICAL_FEATURES = ["weather"]


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("features", TaxiFeatureTransformer()), ("preprocess", preprocessor), ("model", model)])


def _metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    predictions = model.predict(X)
    return {
        "mae_seconds": float(mean_absolute_error(y, predictions)),
        "rmse_seconds": float(mean_squared_error(y, predictions) ** 0.5),
    }


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _data_version() -> str:
    """Hash the raw inputs so each run can be tied back to the exact data."""

    raw_dir = Path(__file__).resolve().parents[2] / "data/raw"
    digests = []
    for name in ("trips.csv", "weather.csv"):
        path = raw_dir / name
        if path.exists():
            digests.append(hashlib.md5(path.read_bytes()).hexdigest())
    if len(digests) == 2:
        return hashlib.md5(":".join(digests).encode()).hexdigest()
    return "unknown"


def train_and_compare(data_path: str | Path, model_path: str | Path, metadata_path: str | Path, comparison_path: str | Path) -> dict:
    data = pd.read_csv(data_path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    train, validation, test = chronological_split(data)
    # Keep only the features the model is allowed to see at prediction time.
    X_columns = [column for column in data.columns if column not in {"trip_duration", "dropoff_datetime", "trip_id", "weather_date"}]

    root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    mlflow.set_experiment("eta-prediction")
    tags = {"git_sha": _git_sha(), "data_version": _data_version()}

    results = {}
    run_ids = {}
    candidates = {
        "baseline_median": build_pipeline(DummyRegressor(strategy="median")),
        "ridge": build_pipeline(Ridge(alpha=10.0)),
        "hist_gradient_boosting": build_pipeline(HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=15, random_state=7)),
    }
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name, tags=tags):
            estimator = model.named_steps["model"]
            mlflow.log_params({f"model__{key}": value for key, value in estimator.get_params().items()})
            mlflow.log_param("model_type", type(estimator).__name__)
            mlflow.log_param("train_rows", len(train))

            started = time.perf_counter()
            model.fit(train[X_columns], train["trip_duration"])
            fit_seconds = time.perf_counter() - started

            results[name] = {
                "validation": _metrics(model, validation[X_columns], validation["trip_duration"]),
                "test": _metrics(model, test[X_columns], test["trip_duration"]),
                "fit_seconds": fit_seconds,
            }
            mlflow.log_metrics({
                "val_mae_seconds": results[name]["validation"]["mae_seconds"],
                "val_rmse_seconds": results[name]["validation"]["rmse_seconds"],
                "test_mae_seconds": results[name]["test"]["mae_seconds"],
                "test_rmse_seconds": results[name]["test"]["rmse_seconds"],
                "fit_seconds": fit_seconds,
            })
            # MLflow 3.x defaults to skops and needs our custom transformer
            # explicitly trusted to reload the model safely.
            model_info = mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["numpy.dtype", "src.features.transform.TaxiFeatureTransformer"],
            )
            run_ids[name] = model_info.run_id

    champion_name = min(("ridge", "hist_gradient_boosting"), key=lambda name: results[name]["validation"]["mae_seconds"])
    champion = candidates[champion_name]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, model_path)
    metadata = {
        "model_version": "1.0.0",
        "champion": champion_name,
        "mlflow_run_id": run_ids[champion_name],
        "target": "trip_duration_seconds",
        "feature_inputs": X_columns,
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "split_boundaries": {
            "train_end": str(train["pickup_datetime"].max()),
            "validation_end": str(validation["pickup_datetime"].max()),
            "test_end": str(test["pickup_datetime"].max()),
        },
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n")
    Path(comparison_path).parent.mkdir(parents=True, exist_ok=True)
    Path(comparison_path).write_text(json.dumps({"run_ids": run_ids, "results": results}, indent=2) + "\n")
    return {"metadata": metadata, "results": results}
