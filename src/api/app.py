"""FastAPI prediction service for the saved full preprocessing/model pipeline."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/champion.joblib"
METADATA_PATH = ROOT / "models/model_metadata.json"
LOG_PATH = ROOT / "monitoring/predictions.sqlite"
app = FastAPI(title="NYC Taxi ETA API", version="1.0.0")
model = None
metadata = {}


class PredictionRequest(BaseModel):
    pickup_datetime: datetime
    pickup_longitude: float = Field(ge=-74.3, le=-73.6)
    pickup_latitude: float = Field(ge=40.4, le=41.1)
    dropoff_longitude: float = Field(ge=-74.3, le=-73.6)
    dropoff_latitude: float = Field(ge=40.4, le=41.1)
    passenger_count: int = Field(ge=1, le=6)
    temp_c: float = Field(ge=-80, le=60)
    precipitation_mm: float = Field(ge=0, le=500)
    weather: str = Field(pattern="^(clear|cloudy|rain|snow)$")


def _load() -> None:
    global model, metadata
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text())


def _ensure_loaded() -> None:
    if model is None:
        _load()


def _log_event(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(LOG_PATH) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS predictions (request_id TEXT PRIMARY KEY, event_json TEXT NOT NULL)")
        connection.execute("INSERT INTO predictions VALUES (?, ?)", (event["request_id"], json.dumps(event)))
        connection.commit()


@app.on_event("startup")
def startup() -> None:
    _load()


@app.get("/health")
def health() -> dict:
    _ensure_loaded()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok"}


@app.get("/v1/model/info")
def model_info() -> dict:
    _ensure_loaded()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return metadata


@app.post("/v1/predict")
def predict(request: PredictionRequest) -> dict:
    _ensure_loaded()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    inputs = request.model_dump(mode="json")
    try:
        prediction_seconds = max(0.0, float(model.predict(pd.DataFrame([inputs]))[0]))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed") from error
    latency_ms = (time.perf_counter() - started) * 1000
    _log_event({
        "request_id": request_id, "event_time": datetime.utcnow().isoformat(),
        "model_version": metadata.get("model_version", "unknown"), "inputs": inputs,
        "prediction_seconds": prediction_seconds, "latency_ms": latency_ms,
    })
    return {"request_id": request_id, "model_version": metadata.get("model_version"), "predicted_duration_minutes": prediction_seconds / 60, "latency_ms": latency_ms}
