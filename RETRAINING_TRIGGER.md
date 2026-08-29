# Retraining Trigger Design

This doc explains when we would recommend retraining the model, and why we
chose those rules. The logic itself lives in `src/monitoring/retrain_trigger.py`
and runs via `scripts/run_retraining_check.py`. It is based on the trigger
design from the parallel implementation
(`reference/initial-implementation/RETRAINING_TRIGGER.md`), but we changed the
rule from an OR to an AND for the reasons below.

## The rule

We only recommend retraining when **all three** of these are true:

1. **The new data is valid.** The latest validation run actually produced
   usable rows (`valid_rows > 0` in `data/processed/validation_report.json`).
   If the incoming data is broken, retraining on it would make things worse.
2. **Drift is severe.** PSI > 0.25 on at least 2 of the monitored numeric
   signals (`distance_km`, `pickup_hour`, `prediction_seconds`). The
   categorical `weather` column is checked too (chi-square test), but it does
   not count toward this condition. More on that below.
3. **Performance actually dropped.** The labelled MAE on the new window is
   more than 1.20x the champion's MAE on the reference window.

If all three hold, the decision file (`monitoring/retraining_decision.json`)
says `retrain_and_compare`. Otherwise it says `do_not_retrain`. The trigger
only recommends. It never retrains or redeploys anything by itself.

## Why we chose these rules

**AND instead of OR.** The parallel implementation used an OR rule: retrain if
the dataset as a whole drifted, or if one important feature drifted. We think
that fires too easily. Drift on its own is not proof the model is failing. For
example, our simulation's `pickup_hour` PSI is 21.33, which sounds alarming,
but it comes from simply moving every trip to 17:30. A model can handle a
shift like that just fine. An OR rule would retrain whenever drift shows up,
even when accuracy has not moved. Our rule waits until drift and a real,
measured accuracy drop happen together. That means we might react a little
later to some failure modes, but we avoid wasting retraining effort (and the
regression risk that comes with it) on false alarms.

**At least 2 features.** One drifting feature can be noise or something local
to how the data was collected. Two features drifting hard at the same time is
a much stronger signal that something systemic changed.

**PSI threshold of 0.25.** This is the usual industry cutoff for the
Population Stability Index: below 0.1 is nothing to worry about, 0.1 to 0.25
is moderate shift, and above 0.25 is a significant shift. We act at
"significant".

**MAE tolerance of 1.20x.** MAE bounces around a bit between batches even when
nothing is wrong, so we do not want to retrain over small wobbles. On a
reference MAE of about 73s, the 1.20x gate means we only act once errors grow
by roughly 15 seconds or more. That is a degradation a rider would notice, not
a rounding error.

**Why weather is monitored but not in the drift count.** Weather is
categorical, so PSI does not apply to it, and we use a chi-square test
instead. We also expect the weather mix to change with the seasons, so a
shifted weather distribution on its own should not force a retrain. We still
track it so a human reviewer can see it.

## How the rule behaved on our drift run

`scripts/simulate_drift.py` builds a stressed window that looks like an
evening rush-hour surge: every pickup moves to 17:30, trip distances grow by
40%, and actual durations get a 30% congestion penalty. So the inputs shift
and the relationship between inputs and duration shifts too. All predictions
come from the deployed champion model. Here is what the committed results in
`monitoring/drift_report.json` show:

| Signal | Value | Threshold | Fires? |
|---|---|---|---|
| `pickup_hour` PSI | 21.33 | > 0.25 | yes |
| `prediction_seconds` PSI | 0.459 | > 0.25 | yes |
| `distance_km` PSI | 0.222 | > 0.25 | no (close, but below; 2 of 3 is enough) |
| `weather` chi-square | p = 1.0 | n/a | no categorical drift |
| MAE | 73.36s to 155.72s (2.123x) | > 1.20x | yes |
| Data valid | 49,213 valid rows | > 0 | yes |

All three conditions fire, so the committed decision in
`monitoring/retraining_decision.json` is `retrain_and_compare`. You can
reproduce it with:

```bash
.venv/bin/python scripts/simulate_drift.py
.venv/bin/python scripts/run_retraining_check.py
```

It is also worth looking at the opposite case. If only the weather mix
changed, the chi-square would flag it but no PSI threshold would break and MAE
would stay flat, so the trigger returns `do_not_retrain`. That is the point of
the AND rule: it does not jump at every single signal.

## What retraining would actually involve

This part is not automated in the project, but the steps would be:

1. Collect the requests logged in `monitoring/predictions.sqlite` and join
   them with real outcomes once those arrive (labels always lag predictions).
2. Mix that with a representative slice of the drifted data.
3. Run the pipeline again (`.venv/bin/dvc repro`) on the combined dataset.
   The new candidates get logged to MLflow next to the existing runs.
4. Only promote a new model if its validation MAE beats the current champion
   (74.2s test MAE). Champion selection is already built into the `train`
   stage, so nothing new is needed there.

## Honest limitations

- The PSI cutoff of 0.25 and the 1.20x MAE factor are heuristics. They are
  sensible defaults for this scenario, not numbers derived from a production
  SLA.
- The "current window" here is a simulation. In a real deployment it would be
  built from logged requests joined with delayed outcomes.
- The performance check needs labelled outcomes, and those arrive late, so the
  trigger is only as fresh as the labels.
- With ~50k rows of synthetic data, `pickup_hour` PSI saturates (21.33) as
  soon as the time of day shifts fully. That is why the decision leans on the
  "at least 2 features" rule rather than on how big any single PSI number is.
