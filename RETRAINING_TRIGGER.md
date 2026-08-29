# Retraining Trigger Design — NYC Taxi Trip Duration Prediction

Adapted from the parallel implementation's trigger design doc
(`reference/initial-implementation/RETRAINING_TRIGGER.md`), rewritten for this
pipeline's more conservative AND-based guard
(`src/monitoring/retrain_trigger.py`).

## Rule

Recommend retraining only when **all three** conditions hold (AND, not OR):

1. **Data valid** — the latest validation run produced usable rows
   (`valid_rows > 0` in `data/processed/validation_report.json`).
2. **Severe drift** — PSI > 0.25 on **at least 2** of the monitored signals
   (`distance_km`, `pickup_hour`, `prediction_seconds`; categorical `weather`
   is tracked separately via chi-square and does not count toward this gate).
3. **Performance degraded** — labelled MAE on the current window exceeds
   **1.20×** the champion's reference-window MAE.

Output (`monitoring/retraining_decision.json`) is either
`retrain_and_compare` or `do_not_retrain`. The trigger **recommends**; it
never retrains or redeploys automatically.

## Justification

- **Why AND instead of OR.** The parallel implementation used an OR rule
  (dataset-level drift *or* single-feature drift above a threshold). Drift
  alone is not proof of failure: this simulation's `pickup_hour` PSI of 21.33
  is produced trivially by *any* time-of-day skew between windows, yet a model
  can remain accurate on shifted-but-learnable traffic. An OR rule therefore
  retrains on correlation with harm; requiring **measured** accuracy
  degradation ties the decision to realized harm. Requiring **multi-feature**
  severe drift guards against a single noisy metric firing the trigger.
  Retraining has real cost (compute, review, regression risk), so the guard is
  deliberately conservative: it accepts late detection of some failure modes
  in exchange for far fewer false-positive retrains.
- **Why PSI > 0.25.** Standard industry interpretation of the Population
  Stability Index: < 0.1 is insignificant, 0.1–0.25 is moderate shift,
  > 0.25 is significant shift. We use the "significant" boundary.
- **Why at least 2 features.** One drifting feature can be benign or
  measurement-local; two independent signals drifting severely is a systemic
  shift worth acting on.
- **Why a 20% MAE tolerance.** Absorbs ordinary batch-to-batch noise. On this
  model's ~73 s reference MAE, the 1.20× gate means we only act once errors
  grow by roughly 15+ seconds — a user-visible degradation, not a rounding
  effect.
- **Why chi-square for weather.** `weather` is categorical, so PSI/KS do not
  apply. It is monitored (a chi-square test in `src/monitoring/monitor.py`)
  but deliberately not part of the `severe_drift` count — categorical weather
  mix changes are expected seasonally and should not, by themselves, force a
  retrain.

## Validation against the observed drift run

`scripts/simulate_drift.py` simulates an evening-rush surge (all pickups moved
to 17:30, trip distances stretched 40%, actual durations inflated 30% for
congestion — covariate shift *and* concept drift). Predictions always come
from the deployed champion. The committed results
(`monitoring/drift_report.json`):

| Signal | Value | Threshold | Fires? |
|---|---|---|---|
| `pickup_hour` PSI | 21.33 | > 0.25 | yes |
| `prediction_seconds` PSI | 0.459 | > 0.25 | yes |
| `distance_km` PSI | 0.222 | > 0.25 | no (below — reported as-is; 2 of 3 suffices) |
| `weather` chi-square | p = 1.0 | — | no categorical drift |
| MAE | 73.36 s → 155.72 s (2.123×) | > 1.20× | yes |
| Data valid | 49,213 valid rows | > 0 | yes |

All three conditions fire → `monitoring/retraining_decision.json` records
`retrain_and_compare`. Reproduce with:

```bash
.venv/bin/python scripts/simulate_drift.py
.venv/bin/python scripts/run_retraining_check.py
```

**Contrast case.** A shift that changes only the weather mix would produce a
low chi-square p-value but no PSI breaches and no MAE degradation — the guard
correctly returns `do_not_retrain`, demonstrating the trigger is not a hair
trigger.

## What retraining would involve (not automated in this project)

1. Collect newly logged requests from `monitoring/predictions.sqlite`
   alongside ground-truth outcomes once they arrive (labels lag predictions).
2. Combine with a representative slice of the drifted/new-distribution data.
3. Re-run the pipeline (`.venv/bin/dvc repro`) on the combined dataset; the
   new candidates land in MLflow alongside existing runs.
4. Promote only if a candidate's validation MAE beats the current champion
   (74.2 s test MAE); champion selection is already part of the `train` stage.

## Limitations / hidden assumptions

- PSI > 0.25 and the 1.20× MAE factor are heuristics, tuned for this
  synthetic scenario rather than derived from production SLAs.
- The "current window" here is a **simulation**; in deployment it would be
  built from logged requests joined with delayed outcomes.
- `performance_degraded` needs labelled outcomes, which arrive with a lag —
  the trigger is only as fresh as the labels.
- With only ~50k rows of synthetic data, PSI on `pickup_hour` saturates
  (21.33) under a full time-of-day shift; the ≥2-feature gate, not the PSI
  magnitude, is what carries the decision.
