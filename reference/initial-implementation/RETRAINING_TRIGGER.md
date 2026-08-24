# Retraining Trigger Design — Flavor A (ETA Prediction)

## Rule

Retrain the model when **either** of the following conditions is met:

1. **Dataset-level drift**: The overall share of drifted columns
   (Evidently `DriftedColumnsCount`, threshold = 0.5) exceeds 50% of
   monitored features.
2. **Feature-level drift on high-importance features**: `distance_km`
   or `traffic_level` individually shows a drift score above 0.3
   (Wasserstein distance for `distance_km`, Jensen-Shannon distance for
   `traffic_level`).

## Justification

- `distance_km` and `traffic_level` are the two most influential
  features in the deployed XGBoost model's prediction of ETA, since
  the underlying data-generating process (Week 1 synthetic data)
  multiplies base travel time by a traffic penalty and computes
  distance directly from trip coordinates. Drift in either feature
  directly undermines the assumptions the model was trained on.
- The dataset-level rule (>50% of columns drifted) catches broader,
  systemic shifts (e.g., a data pipeline change or a truly novel
  seasonal pattern) that might not concentrate in any single feature.
- Both conditions are independently sufficient (OR, not AND) because
  either type of shift — one dominant feature drifting sharply, or
  many features drifting moderately — can degrade model accuracy for
  different underlying reasons, and requiring both to co-occur would
  delay detection of either failure mode.

## Validation against the observed drift run

Running `src/monitor.py` on the simulated "surge" scenario
(`src/simulate_drift.py`) produced:

- Dataset-level: **5 of 9 columns (55.6%) drifted**, exceeding the 50%
  threshold — condition 1 fires.
- Feature-level: `distance_km` drift score = **0.761** (>0.3),
  `traffic_level` drift score = **0.357** (>0.3) — condition 2 fires
  on both named features.

Both trigger conditions independently would have flagged this
scenario for retraining, confirming the rule behaves as intended on a
realistic drift case.

## What retraining would involve (not automated in this project)

1. Collect newly logged predictions (`logs/predictions.csv`) alongside
   actual outcomes once available.
2. Combine with a representative slice of the drifted/new-distribution
   data.
3. Re-run `src/train.py` (or an updated version) on the combined
   dataset, re-track via MLflow, and compare new metrics against the
   currently deployed model's baseline (MAE 1.653) before promoting.
4. Only replace the deployed model artifact
   (`models/xgboost_eta_model.joblib`) if the retrained model's MAE
   improves or holds steady on a held-out validation slice.