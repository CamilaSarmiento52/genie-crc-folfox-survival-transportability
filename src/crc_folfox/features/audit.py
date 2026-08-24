"""Aggregate diagnostics for the independent-truncation assumption."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr


def _standardized_difference(values: pd.Series, eligible: pd.Series) -> float:
    a = pd.to_numeric(values[eligible], errors="coerce").dropna()
    b = pd.to_numeric(values[~eligible], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _cramers_v(values: pd.Series, eligible: pd.Series) -> float:
    table = pd.crosstab(values.astype("string").fillna("<MISSING>"), eligible)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True)
    expected /= observed.sum()
    chi2 = np.divide(
        (observed - expected) ** 2,
        expected,
        out=np.zeros_like(observed),
        where=expected > 0,
    ).sum()
    denominator = observed.sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator > 0 else float("nan")


def truncation_assumption_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe measured associations with sequencing lag and OS entry eligibility."""
    continuous = ("age_at_folfox", "number_metastatic_sites")
    categorical = (
        "institution",
        "metastatic_presentation",
        "ca_d_site",
        "ca_grade",
        "prior_oxaliplatin",
        "biologic_with_folfox",
    )
    eligible = frame["entry"].lt(frame["exit"])
    rows = []
    for variable in continuous:
        values = pd.to_numeric(frame[variable], errors="coerce")
        valid = values.notna() & frame["sequencing_lag_days"].notna()
        x = values[valid].to_numpy(dtype=float)
        lag = frame.loc[valid, "sequencing_lag_days"].to_numpy(dtype=float)
        if len(x) >= 3 and np.unique(x).size > 1 and np.unique(lag).size > 1:
            rho, p_value = spearmanr(x, lag)
        else:
            rho = p_value = float("nan")
        rows.append(
            {
                "variable": variable,
                "kind": "continuous",
                "lag_association": "spearman_rho",
                "lag_effect": float(rho),
                "lag_p_value": float(p_value),
                "eligibility_association": "standardized_mean_difference",
                "eligibility_effect": _standardized_difference(values, eligible),
                "missing_n": int(values.isna().sum()),
            }
        )
    for variable in categorical:
        values = frame[variable].astype("string").fillna("<MISSING>")
        groups = [
            group["sequencing_lag_days"].to_numpy(dtype=float)
            for _, group in frame.assign(_level=values).groupby("_level")
            if len(group) > 0
        ]
        if len(frame) >= 3 and len(groups) > 1 and np.unique(np.concatenate(groups)).size > 1:
            statistic, p_value = kruskal(*groups)
            epsilon_squared = max(
                0.0, float((statistic - len(groups) + 1) / (len(frame) - len(groups)))
            )
        else:
            p_value = epsilon_squared = float("nan")
        rows.append(
            {
                "variable": variable,
                "kind": "categorical",
                "lag_association": "kruskal_epsilon_squared",
                "lag_effect": epsilon_squared,
                "lag_p_value": float(p_value),
                "eligibility_association": "cramers_v",
                "eligibility_effect": _cramers_v(values, eligible),
                "missing_n": int(frame[variable].isna().sum()),
            }
        )
    return pd.DataFrame(rows)
