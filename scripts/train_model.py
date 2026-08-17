from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train import train_and_compare


if __name__ == "__main__":
    train_and_compare(ROOT / "data/processed/validated_trips.csv", ROOT / "models/champion.joblib", ROOT / "models/model_metadata.json", ROOT / "experiments/comparison.json")
