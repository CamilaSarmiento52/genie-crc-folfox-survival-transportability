"""E6: patient-level prediction stability across bootstrap refits, published in aggregate only.

`explainability-spec.md`, "E6 -- prediction stability": per `(endpoint, centre, model, horizon)`,
the median and quartiles of the interval width across held-out patients, the proportion of
patients whose interval width exceeds 0.10 and 0.20, and the cross-family spread. No per-patient
row, key, time, prediction, or interval reaches this module's output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH_THRESHOLDS = (0.10, 0.20)


@dataclass(frozen=True)
class PredictionStabilitySummary:
    endpoint: str
    centre: str
    model_name: str
    horizon: float
    n_patients: int
    n_refits: int
    median_interval_width: float | None
    q1_interval_width: float | None
    q3_interval_width: float | None
    proportion_exceeding_010: float | None
    proportion_exceeding_020: float | None


def summarize_prediction_stability(
    endpoint: str, centre: str, model_name: str, horizon: float, refit_predictions: list[np.ndarray]
) -> PredictionStabilitySummary:
    """`refit_predictions` is a list of `(n_patients,)` risk arrays, one per successful refit."""
    n_refits = len(refit_predictions)
    if n_refits < 2:
        return PredictionStabilitySummary(
            endpoint,
            centre,
            model_name,
            horizon,
            n_patients=len(refit_predictions[0]) if refit_predictions else 0,
            n_refits=n_refits,
            median_interval_width=None,
            q1_interval_width=None,
            q3_interval_width=None,
            proportion_exceeding_010=None,
            proportion_exceeding_020=None,
        )
    stacked = np.stack(refit_predictions, axis=0)  # (n_refits, n_patients)
    lower, upper = np.percentile(stacked, [2.5, 97.5], axis=0)
    widths = upper - lower
    q1, median, q3 = np.percentile(widths, [25, 50, 75])
    return PredictionStabilitySummary(
        endpoint=endpoint,
        centre=centre,
        model_name=model_name,
        horizon=horizon,
        n_patients=stacked.shape[1],
        n_refits=n_refits,
        median_interval_width=float(median),
        q1_interval_width=float(q1),
        q3_interval_width=float(q3),
        proportion_exceeding_010=float(np.mean(widths > 0.10)),
        proportion_exceeding_020=float(np.mean(widths > 0.20)),
    )


def cross_family_spread(
    model_risk_by_patient: dict[str, np.ndarray],
) -> tuple[float | None, float | None]:
    """Median and 90th percentile of the maximum pairwise risk difference across explained models
    for the same patient, at one `(endpoint, centre, horizon)`."""
    models = sorted(model_risk_by_patient)
    if len(models) < 2:
        return None, None
    stacked = np.stack([model_risk_by_patient[m] for m in models], axis=0)  # (n_models, n_patients)
    spread = stacked.max(axis=0) - stacked.min(axis=0)
    return float(np.median(spread)), float(np.percentile(spread, 90))


__all__ = [
    "WIDTH_THRESHOLDS",
    "PredictionStabilitySummary",
    "cross_family_spread",
    "summarize_prediction_stability",
]
