import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.validate import validate


if __name__ == "__main__":
    report_path = ROOT / "data/processed/validation_report.json"
    valid = validate(pd.read_csv(ROOT / "data/interim/ingested_trips.csv"), ROOT / "data/quarantine/invalid_rows.csv", report_path)
    valid.to_csv(ROOT / "data/processed/validated_trips.csv", index=False)
    report = json.loads(report_path.read_text())
    if report["valid_rows"] == 0:
        print(f"validation produced zero usable rows; see {report_path}", file=sys.stderr)
        sys.exit(1)
