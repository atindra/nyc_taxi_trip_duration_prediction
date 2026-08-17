from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import ingest


if __name__ == "__main__":
    ingest(ROOT / "data/raw/trips.csv", ROOT / "data/raw/weather.csv", ROOT / "data/interim/ingested_trips.csv", ROOT / "data/interim/ingestion_report.json")
