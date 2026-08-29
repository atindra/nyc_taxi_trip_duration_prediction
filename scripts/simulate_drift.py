"""Simulate a rush-hour surge window and measure drift with the trained model.

Window construction lives in src/monitoring/drift_windows.py so the PSI/KS
report and the Evidently HTML report always describe identical windows.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring.drift_windows import build_drift_windows
from src.monitoring.monitor import compare_windows
from sklearn.metrics import mean_absolute_error


def main() -> None:
    reference, shifted = build_drift_windows()

    drift = compare_windows(reference, shifted)
    performance = {
        "reference_mae_seconds": float(mean_absolute_error(reference["actual_seconds"], reference["prediction_seconds"])),
        "shifted_mae_seconds": float(mean_absolute_error(shifted["actual_seconds"], shifted["prediction_seconds"])),
    }
    performance["mae_change_ratio"] = round(
        performance["shifted_mae_seconds"] / performance["reference_mae_seconds"], 3
    )

    result = {"drift_metrics": drift, "performance": performance}
    output = ROOT / "monitoring/drift_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
