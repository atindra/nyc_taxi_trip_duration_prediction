"""Evaluate the retraining trigger against the latest drift report."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring.retrain_trigger import should_retrain


if __name__ == "__main__":
    report = json.loads((ROOT / "monitoring/drift_report.json").read_text())
    validation = json.loads((ROOT / "data/processed/validation_report.json").read_text())
    decision = should_retrain(
        report["drift_metrics"],
        data_valid=validation["valid_rows"] > 0,
        labelled_mae=report["performance"]["shifted_mae_seconds"],
        champion_mae=report["performance"]["reference_mae_seconds"],
    )
    (ROOT / "monitoring/retraining_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))
