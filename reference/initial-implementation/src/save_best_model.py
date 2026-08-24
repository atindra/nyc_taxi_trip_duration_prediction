"""
Week 3 / M4 - Save the best model (XGBoost, selected in Week 2 based on
lowest MAE) as a standalone artifact for packaging/serving.
"""

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import json

FEATURES_PATH = "data/processed/trips_features.csv"
TARGET = "actual_eta_minutes"
ID_COL = "trip_id"
MODEL_OUT = "models/xgboost_eta_model.joblib"
FEATURE_COLUMNS_OUT = "models/feature_columns.json"

params = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": 42,
}


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[ID_COL, TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    # Save the exact feature column order/names the API must reconstruct
    with open(FEATURE_COLUMNS_OUT, "w") as f:
        json.dump(list(X.columns), f)

    print(f"Model saved to {MODEL_OUT}")
    print(f"Feature columns saved to {FEATURE_COLUMNS_OUT}")
    print(f"Feature columns: {list(X.columns)}")


if __name__ == "__main__":
    main()