"""
Week 2 / M3 - Model training and comparison for Flavor A (ETA prediction).

Trains two models on the engineered feature set and logs both as tracked
MLflow experiments, per the brief's requirement to "track experiments and
hyperparameters" and compare model runs.

Models compared (per brief's suggested pairing):
  1. Linear Regression - simple baseline, no hyperparameters to tune.
  2. XGBoost Regressor - gradient boosting comparison model.

Metrics logged: MAE, RMSE, R^2 (standard regression metrics for an ETA
prediction task - MAE is the most directly interpretable in "minutes").
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

FEATURES_PATH = "data/processed/trips_features.csv"
TARGET = "actual_eta_minutes"
ID_COL = "trip_id"

mlflow.set_experiment("eta-prediction-flavor-a")


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[ID_COL, TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def train_linear_regression(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="linear_regression_baseline"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"[Linear Regression] MAE={metrics['mae']:.3f}  "
              f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}")
        return metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    with mlflow.start_run(run_name="xgboost_gradient_boosting"):
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"[XGBoost] MAE={metrics['mae']:.3f}  "
              f"RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.3f}")
        return metrics


def main():
    X_train, X_test, y_train, y_test = load_data()

    lr_metrics = train_linear_regression(X_train, X_test, y_train, y_test)
    xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)

    print("\n--- Comparison ---")
    print(f"Linear Regression : MAE={lr_metrics['mae']:.3f}  R2={lr_metrics['r2']:.3f}")
    print(f"XGBoost            : MAE={xgb_metrics['mae']:.3f}  R2={xgb_metrics['r2']:.3f}")

    best = "XGBoost" if xgb_metrics["mae"] < lr_metrics["mae"] else "Linear Regression"
    print(f"\nBest model by MAE: {best}")


if __name__ == "__main__":
    main()