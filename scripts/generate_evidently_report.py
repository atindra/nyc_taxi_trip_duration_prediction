"""Generate an Evidently data-drift HTML report for the simulated surge window.

Uses the same reference/shifted windows as scripts/simulate_drift.py (via
src.monitoring.drift_windows), so the HTML report and the PSI/KS JSON report
always describe identical windows. The monitored column set matches
src/monitoring/monitor.py.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

from src.monitoring.drift_windows import build_drift_windows

MONITORED_COLUMNS = ["distance_km", "pickup_hour", "prediction_seconds", "weather"]


def main() -> None:
    reference, shifted = build_drift_windows()

    report = Report([DataDriftPreset()])
    # The shifted window forces pickup_hour to a single value, so Evidently's
    # internal correlation matrix divides by a zero stddev for that column.
    with np.errstate(divide="ignore", invalid="ignore"):
        snapshot = report.run(
            reference_data=reference[MONITORED_COLUMNS],
            current_data=shifted[MONITORED_COLUMNS],
        )

    output = ROOT / "monitoring/evidently_drift_report.html"
    snapshot.save_html(str(output))
    print(f"Evidently drift report saved to {output}")


if __name__ == "__main__":
    main()
