"""Guarded retraining decision: drift alone is not sufficient."""


def should_retrain(monitoring_result: dict, data_valid: bool, labelled_mae: float | None, champion_mae: float) -> dict:
    severe_drift = sum(item.get("psi", 0) > 0.25 for item in monitoring_result.values()) >= 2
    performance_degraded = labelled_mae is not None and labelled_mae > champion_mae * 1.20
    promote = bool(data_valid and severe_drift and performance_degraded)
    return {
        "data_valid": data_valid,
        "severe_drift": severe_drift,
        "performance_degraded": performance_degraded,
        "action": "retrain_and_compare" if promote else "do_not_retrain",
    }
