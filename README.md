\# ETA Prediction — End-to-End ML Pipeline (Flavor A)



Machine Learning Engineering (PCAM ZC412) — Mini-Project (EC-1)



An end-to-end ML pipeline that predicts delivery/ride ETA from trip

distance, time-of-day, weather, and traffic conditions — covering data

ingestion, validation, feature engineering, experiment tracking, model

packaging, REST API deployment, and drift monitoring.



\## Architecture

┌─────────────────────┐

&#x20;               │  Raw Trip Data (CSV)  │

&#x20;               │  (synthetic, 20k rows)│

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │   Validation (M2)     │

&#x20;               │  schema + quality     │

&#x20;               │  checks, cleaning     │

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │ Feature Engineering   │

&#x20;               │  (M2) one-hot weather,│

&#x20;               │  numeric features     │

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐        ┌─────────────┐

&#x20;               │  DVC Dataset Versioning│───────▶│  Git / GitHub│

&#x20;               └──────────┬───────────┘        └─────────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │  Model Training (M3)   │

&#x20;               │  Linear Regression vs  │───────▶ MLflow Tracking

&#x20;               │  XGBoost               │         (params, metrics,

&#x20;               └──────────┬───────────┘         model artifacts)

&#x20;                          │  (best: XGBoost)

&#x20;               ┌──────────▼───────────┐

&#x20;               │  Model Packaging (M4)  │

&#x20;               │  joblib artifact +     │

&#x20;               │  FastAPI REST service  │

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │   Docker Container     │

&#x20;               │   (port 8000)          │

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │  Prediction Logging    │

&#x20;               │  (M5)                  │

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │  Drift Simulation \&    │

&#x20;               │  Monitoring (M5)       │

&#x20;               │  Evidently drift report│

&#x20;               └──────────┬───────────┘

&#x20;                          │

&#x20;               ┌──────────▼───────────┐

&#x20;               │  Retraining Trigger    │

&#x20;               │  (documented rule)     │

&#x20;               └───────────────────────┘





\## Repository structure





eta-prediction-mlproject/

├── data/

│ ├── raw/trips\_raw.csv # ingested synthetic data (DVC-tracked)

│ ├── processed/

│ │ ├── trips\_clean.csv # post-validation (DVC-tracked)

│ │ └── trips\_features.csv # model-ready features (DVC-tracked)

│ └── drift/trips\_drifted.csv # simulated surge-scenario data

├── src/

│ ├── generate\_synthetic\_data.py # Week 1 (M2)

│ ├── validate.py # Week 1 (M2)

│ ├── features.py # Week 1 (M2)

│ ├── train.py # Week 2 (M3)

│ ├── save\_best\_model.py # Week 3 (M4)

│ ├── serve.py # Week 3 (M4) - FastAPI app

│ ├── simulate\_drift.py # Week 4 (M5)

│ ├── monitor.py # Week 4 (M5) - Evidently drift report

│ └── check\_retrain\_trigger.py # Week 4 (M5)

├── models/

│ ├── xgboost\_eta\_model.joblib

│ └── feature\_columns.json

├── logs/

│ ├── predictions.csv # live prediction log

│ └── drift\_report.html # Evidently drift report

├── Dockerfile

├── requirements.txt

├── RETRAINING\_TRIGGER.md # documented retraining trigger rule

└── README.md





\## Setup instructions



\### Prerequisites

\- Anaconda / Miniconda

\- Docker Desktop (with WSL2 backend, on Windows)

\- Git



\### 1. Clone and set up the environment

```bash

git clone <your-repo-url>

cd eta-prediction-mlproject

conda create -n mlproject python=3.10 -y

conda activate mlproject

conda install -c conda-forge git dvc mlflow fastapi uvicorn xgboost lightgbm -y

pip install evidently

```



\### 2. Pull versioned data (if cloning fresh)

```bash

dvc pull

```



\### 3. Run the pipeline stages

```bash

python src/generate\_synthetic\_data.py

python src/validate.py

python src/features.py

python src/train.py

python src/save\_best\_model.py

```



\### 4. View experiment tracking

```bash

mlflow ui

\# open http://127.0.0.1:5000

```



\### 5. Run the API locally

```bash

uvicorn src.serve:app --reload --port 8000

\# open http://127.0.0.1:8000/docs

```



\### 6. Build and run with Docker

```bash

docker build -t eta-prediction-api .

docker run -d -p 8000:8000 --name eta-api eta-prediction-api

```



\### 7. Run drift simulation and monitoring

```bash

python src/simulate\_drift.py

python src/monitor.py

python src/check\_retrain\_trigger.py

```



\## Model comparison summary



| Model | MAE (min) | RMSE (min) | R² |

|---|---|---|---|

| Linear Regression | 3.628 | 4.901 | 0.943 |

| \*\*XGBoost (deployed)\*\* | \*\*1.653\*\* | \*\*2.078\*\* | \*\*0.990\*\* |



XGBoost was selected as the deployed model based on lowest MAE, justified

by its ability to capture non-linear interactions between weather and

traffic conditions present in the underlying trip-time formula.



\## Drift monitoring summary



Simulating a "festival/rush-hour surge" scenario showed drift in 5 of 9

monitored features (55.6%), concentrated in `distance\_km`, `traffic\_level`,

and `weather` categories — the features most directly tied to the

surge scenario. See `RETRAINING\_TRIGGER.md` for the full documented

retraining trigger design and validation against this scenario.



\## API reference



\*\*POST /predict\*\*

```json

{

&#x20; "hour\_of\_day": 18,

&#x20; "weekday": 2,

&#x20; "is\_weekend": 0,

&#x20; "distance\_km": 5.2,

&#x20; "traffic\_level": 2,

&#x20; "weather": "rain"

}

```

Response:

```json

{ "predicted\_eta\_minutes": 12.4 }

```



\*\*GET /health\*\* — returns `{"status": "ok"}`



\## Academic references



\- T1: \*Machine Learning Production Systems\*, Robert Crowe, et al.; O'Reilly, 2024.

\- T2: \*Machine Learning Engineering\*, Andriy Burkov, 2020.

\- R1: \*Machine Learning Engineering with Python\* (2nd Edition), Andrew P. McMahon, Packt, 2023.







