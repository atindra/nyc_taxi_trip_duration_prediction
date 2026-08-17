"""Small, auditable drift metrics used by the local monitoring job."""

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    quantile_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantile_edges) < 2:
        return 0.0
    edges = np.concatenate(([-np.inf], quantile_edges[1:-1], [np.inf]))
    reference_counts = np.histogram(reference, bins=edges)[0] + 1e-6
    current_counts = np.histogram(current, bins=edges)[0] + 1e-6
    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def compare_windows(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    results = {}
    for column in ["distance_km", "pickup_hour", "prediction_seconds"]:
        if column in reference and column in current:
            statistic, p_value = ks_2samp(reference[column].dropna(), current[column].dropna())
            results[column] = {"psi": psi(reference[column], current[column]), "ks_statistic": float(statistic), "ks_p_value": float(p_value)}
    if "weather" in reference and "weather" in current:
        categories = sorted(set(reference["weather"].dropna()) | set(current["weather"].dropna()))
        expected = reference["weather"].value_counts().reindex(categories, fill_value=0).astype(float) + 1e-6
        observed = current["weather"].value_counts().reindex(categories, fill_value=0).astype(float)
        statistic, p_value = chisquare(observed, f_exp=expected * observed.sum() / expected.sum())
        results["weather"] = {"chi_squared": float(statistic), "chi_squared_p_value": float(p_value)}
    return results
