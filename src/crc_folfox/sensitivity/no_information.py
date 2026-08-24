"""No-information (Kaplan-Meier) prediction-error reference and scaled Brier score.

A clearly labelled **descriptive aid**, per `analysis/the evaluation-diagnosis note`'s
"Improve interpretation and reporting": raw Brier/IBS values remain valid for comparing models on
the same patients, and the scaled version is reported only alongside them, never in place of them,
and is never retrofitted into the frozen primary-evaluation metric hierarchy or reporting-status
rule (this
module's rows live in their own artifact, `scaled_brier.csv`).

The no-information reference is the delayed-entry Kaplan-Meier survival curve
(`lifelines.KaplanMeierFitter(entry=...)`) applied identically to every patient (a model that uses
no covariate information at all), scored through the imported primary-evaluation Brier/IBS core --
the same
`horizon_brier_point`/`integrated_brier_score` used by every other evaluation or sensitivity
analysis Brier number, never a
reimplementation.
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError

from crc_folfox.evaluate.contracts import IBS_GRID
from crc_folfox.evaluate.prediction_error import horizon_brier_point, integrated_brier_score

_IDX_365, _IDX_730 = 0, len(IBS_GRID) - 1


def no_information_risk_grid(
    entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, *, grid: tuple[float, ...] = IBS_GRID
) -> tuple[np.ndarray | None, str]:
    """The population-level `1 - S_KM(t)` risk, broadcast to every patient -- a "model" that uses
    no covariate information, scored on the same estimand-eligible scale as every real candidate."""
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    if len(entry) < 2:
        return None, "insufficient_support"
    try:
        km = KaplanMeierFitter().fit(exit_, event_observed=event, entry=entry)
    except StatError as exc:
        return None, str(exc)
    predicted_survival = np.atleast_1d(np.asarray(km.predict(np.array(grid)), dtype=float))
    risk_row = 1.0 - predicted_survival
    return np.tile(risk_row, (len(entry), 1)), ""


def no_information_brier_rows(
    endpoint: str,
    centre: str,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    *,
    horizons: tuple[float, ...] = (365.0, 730.0),
) -> dict[str, dict]:
    """Keyed by `f"brier_{horizon:g}"`/`"ibs"`, matching `subset_metrics`'s metric-key convention
    so `scaled_brier_rows` can pair them directly against a model's own scored metrics."""
    risk_grid, reason = no_information_risk_grid(entry, exit_, event)
    rows: dict[str, dict] = {}
    if risk_grid is None:
        for horizon in horizons:
            rows[f"brier_{horizon:g}"] = {"value": None, "reason": reason}
        rows["ibs"] = {"value": None, "reason": reason}
        return rows

    for horizon in horizons:
        col = _IDX_365 if horizon == 365.0 else _IDX_730
        point = horizon_brier_point(entry, exit_, event, risk_grid[:, col], horizon)
        rows[f"brier_{horizon:g}"] = {
            "value": point.score,
            "reason": point.reason if point.score is None else "",
            "n_eligible": point.n_eligible,
            "n_events": point.n_events,
        }
    ibs = integrated_brier_score(entry, exit_, event, risk_grid)
    rows["ibs"] = {"value": ibs.value, "reason": ibs.reason if ibs.value is None else ""}
    return rows


def scaled_brier_rows(
    *,
    endpoint: str,
    design: str,
    centre: str,
    model_name: str,
    model_brier_by_key: dict[str, float | None],
    no_information_brier_by_key: dict[str, dict],
) -> list[dict]:
    """`scaled_brier = 1 - Brier_model / Brier_KM` for every shared `brier_*`/`ibs` key -- the sign
    convention is positive favours the model. Never computed when the no-information reference
    itself is undefined or exactly zero."""
    rows: list[dict] = []
    for key, model_value in model_brier_by_key.items():
        reference = no_information_brier_by_key.get(key)
        reference_value = reference["value"] if reference else None
        scaled: float | None = None
        reason = ""
        if model_value is None:
            reason = "model_brier_not_computable"
        elif reference_value is None:
            reason = "no_information_reference_not_computable"
        elif reference_value == 0:
            reason = "no_information_reference_is_zero"
        else:
            scaled = 1.0 - (model_value / reference_value)
        rows.append(
            {
                "endpoint": endpoint,
                "design": design,
                "centre": centre,
                "model": model_name,
                "metric": key,
                "model_brier": model_value,
                "no_information_brier": reference_value,
                "scaled_brier": scaled,
                "reason": reason,
                "descriptive_only": True,
            }
        )
    return rows


__all__ = ["no_information_brier_rows", "no_information_risk_grid", "scaled_brier_rows"]
