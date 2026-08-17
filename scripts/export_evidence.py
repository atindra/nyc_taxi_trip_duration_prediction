"""Export committed evidence snapshots for review without MLflow/DVC.

Copies the latest pipeline outputs (comparison.json, model_metadata.json) and
the MLflow run table for experiment `eta-prediction` into evidence/, so the
repository carries a reviewable record of the final runs. Run this after any
`dvc repro` that should be reflected in the committed evidence.
"""

from pathlib import Path
import shutil
import sys

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = [
    "run_id", "tags.mlflow.runName", "start_time",
    "metrics.val_mae_seconds", "metrics.val_rmse_seconds",
    "metrics.test_mae_seconds", "metrics.test_rmse_seconds",
    "metrics.fit_seconds", "tags.git_sha", "tags.data_version",
]


def main() -> None:
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy(ROOT / "experiments/comparison.json", evidence / "comparison.json")
    shutil.copy(ROOT / "models/model_metadata.json", evidence / "model_metadata.json")

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["eta-prediction"], order_by=["start_time"])
    runs = runs.groupby("tags.mlflow.runName", as_index=False).tail(1)
    runs[COLUMNS].to_csv(evidence / "mlflow-runs.csv", index=False)
    print(f"exported {len(runs)} runs -> {evidence}/mlflow-runs.csv (+ comparison.json, model_metadata.json)")


if __name__ == "__main__":
    main()
