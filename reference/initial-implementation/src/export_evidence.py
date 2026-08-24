"""
Export self-contained, regenerable evidence of pipeline results.

Pulls real data from:
  - MLflow tracking store (mlflow.db) — actual logged runs, not re-typed numbers
  - The saved model artifact + feature columns
  - A live call to the running FastAPI service (if reachable)
  - The committed drift report

Writes everything to evidence/ as JSON + a human-readable summary,
so a grader can verify claims without re-running the full pipeline.
"""

import json
import os
from datetime import datetime, timezone

import mlflow
import requests

EVIDENCE_DIR = "evidence"
MLFLOW_DB_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "eta-prediction-flavor-a"
API_BASE_URL = "http://127.0.0.1:8000"

SAMPLE_TRIP = {
    "hour_of_day": 18,
    "weekday": 2,
    "is_weekend": 0,
    "distance_km": 5.2,
    "traffic_level": 2,
    "weather": "rain",
}


def export_mlflow_runs():
    """Pull actual logged runs from the MLflow tracking store."""
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return {"error": f"Experiment '{EXPERIMENT_NAME}' not found in {MLFLOW_DB_URI}"}

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    result = {}
    for run in runs:
        result[run.data.tags.get("mlflow.runName", run.info.run_id)] = {
            "run_id": run.info.run_id,
            "params": dict(run.data.params),
            "metrics": dict(run.data.metrics),
            "start_time": datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc
            ).isoformat(),
            "status": run.info.status,
        }
    return result


def export_model_metadata():
    """Confirm the deployed model artifact exists and record its identity."""
    model_path = "models/xgboost_eta_model.joblib"
    columns_path = "models/feature_columns.json"

    if not os.path.isfile(model_path):
        return {"error": f"{model_path} not found"}

    with open(columns_path) as f:
        feature_columns = json.load(f)

    return {
        "model_path": model_path,
        "model_size_bytes": os.path.getsize(model_path),
        "feature_columns": feature_columns,
        "champion_model_type": "XGBRegressor",
    }


def export_live_api_test():
    """Hit the running API for a real, timestamped prediction — proof it works."""
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        predict = requests.post(f"{API_BASE_URL}/predict", json=SAMPLE_TRIP, timeout=5)
        return {
            "health_check": {
                "status_code": health.status_code,
                "response": health.json(),
            },
            "sample_prediction": {
                "request_body": SAMPLE_TRIP,
                "status_code": predict.status_code,
                "response": predict.json(),
            },
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }
    except requests.exceptions.ConnectionError:
        return {
            "error": "API not reachable at " + API_BASE_URL,
            "hint": "Run 'uvicorn src.serve:app --port 8000' or the Docker container first, then re-run this script.",
        }


def export_drift_summary():
    """Reference the committed drift report and retraining trigger result."""
    report_path = "logs/drift_report.html"
    trigger_doc_path = "RETRAINING_TRIGGER.md"

    return {
        "drift_report_path": report_path,
        "drift_report_exists": os.path.isfile(report_path),
        "retraining_trigger_doc_path": trigger_doc_path,
        "retraining_trigger_doc_exists": os.path.isfile(trigger_doc_path),
        "note": "Run 'python src/check_retrain_trigger.py' for the live retrain/no-retrain decision.",
    }


def main():
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    evidence = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_runs": export_mlflow_runs(),
        "model_metadata": export_model_metadata(),
        "live_api_test": export_live_api_test(),
        "drift_and_retraining": export_drift_summary(),
    }

    out_path = os.path.join(EVIDENCE_DIR, "evidence.json")
    with open(out_path, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Evidence written to {out_path}")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()