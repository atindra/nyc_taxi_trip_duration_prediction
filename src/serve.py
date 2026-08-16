"""
Week 3 / M4 - REST API serving the trained ETA prediction model.

Accepts trip details and returns predicted ETA in minutes.
Includes input validation and error handling per the brief's requirement.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import json

MODEL_PATH = "models/xgboost_eta_model.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"

app = FastAPI(title="ETA Prediction API", version="1.0")

model = joblib.load(MODEL_PATH)
with open(FEATURE_COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

VALID_WEATHER = {"clear", "fog", "rain", "storm"}


class TripRequest(BaseModel):
    hour_of_day: int = Field(..., ge=0, le=23)
    weekday: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    distance_km: float = Field(..., gt=0)
    traffic_level: int = Field(..., ge=0, le=2)
    weather: str


class ETAResponse(BaseModel):
    predicted_eta_minutes: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ETAResponse)
def predict(trip: TripRequest):
    if trip.weather not in VALID_WEATHER:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid weather value '{trip.weather}'. Must be one of {sorted(VALID_WEATHER)}",
        )

    row = {col: 0 for col in FEATURE_COLUMNS}
    row["hour_of_day"] = trip.hour_of_day
    row["weekday"] = trip.weekday
    row["is_weekend"] = trip.is_weekend
    row["distance_km"] = trip.distance_km
    row["traffic_level"] = trip.traffic_level

    weather_col = f"weather_{trip.weather}"
    if weather_col in row:
        row[weather_col] = 1

    X = pd.DataFrame([row])[FEATURE_COLUMNS]

    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return ETAResponse(predicted_eta_minutes=round(float(pred), 2))