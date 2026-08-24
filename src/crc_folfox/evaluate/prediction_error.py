"""Delayed-entry Brier score and integrated Brier score (IBS) over the locked [365, 730] window.

Per `evaluation-spec.md` ("Frozen metric set"), the horizon-level Brier core is reused by import
from `crc_folfox.model_training.tuning.delayed_entry_ipcw_brier` -- model-training workflow's own
Gate-3-validated
implementation -- rather than reimplemented. The primary-evaluation workflow calls it with `training
== validation ==` the
assessment rows of the reporting cell, matching the locked estimand ("Horizon eligibility and the
locked estimand": the censoring distribution is estimated on the assessment rows themselves).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from crc_folfox.evaluate.censoring import HorizonWeights, horizon_weights
from crc_folfox.evaluate.contracts import IBS_GRID, IBS_WINDOW_WIDTH
from crc_folfox.model_training.contracts import ModelTrainingContractError
from crc_folfox.model_training.tuning import delayed_entry_ipcw_brier


@dataclass(frozen=True)
class BrierPoint:
    horizon: float
    score: float | None
    reason: str
    n_eligible: int
    n_events: int


def horizon_brier_point(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_at_horizon: np.ndarray,
    horizon: float,
) -> BrierPoint:
    """Delayed-entry IPCW Brier at one horizon, with the censoring distribution estimated on the
    same rows being scored (`training == validation`), per the locked evaluation estimand."""
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    eligible = entry <= horizon
    n_eligible = int(eligible.sum())
    n_events = int(((exit_ <= horizon) & (event == 1) & eligible).sum())
    if n_eligible == 0:
        return BrierPoint(horizon, None, "no_eligible_rows", 0, 0)
    survival = 1.0 - np.asarray(risk_at_horizon, dtype=float)
    frame = pd.DataFrame({"entry": entry, "exit": exit_, "event": event})
    try:
        score = delayed_entry_ipcw_brier(
            training=frame, validation=frame, survival=survival.reshape(-1, 1), horizons=(horizon,)
        )
    except ModelTrainingContractError as exc:
        return BrierPoint(horizon, None, str(exc), n_eligible, n_events)
    return BrierPoint(horizon, float(score), "", n_eligible, n_events)


@dataclass(frozen=True)
class IbsResult:
    value: float | None
    points: tuple[BrierPoint, ...]
    reason: str


def integrated_brier_score(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_grid: np.ndarray,
    grid: tuple[float, ...] = IBS_GRID,
) -> IbsResult:
    """Trapezoidal integral of the per-grid-point delayed-entry Brier score over `[365, 730]`,
    divided by the 365-day window width. `risk_grid` has one column per `grid` horizon."""
    risk_grid = np.asarray(risk_grid, dtype=float)
    if risk_grid.shape[1] != len(grid):
        raise ValueError("risk_grid must have one column per IBS grid horizon")
    points = tuple(
        horizon_brier_point(entry, exit_, event, risk_grid[:, index], horizon)
        for index, horizon in enumerate(grid)
    )
    failing = next((point for point in points if point.score is None), None)
    if failing is not None:
        return IbsResult(None, points, f"grid_point_{failing.horizon:g}_not_computable")
    scores = np.array([point.score for point in points], dtype=float)
    value = float(np.trapezoid(scores, grid) / IBS_WINDOW_WIDTH)
    if not np.isfinite(value):
        return IbsResult(None, points, "ibs_nonfinite")
    return IbsResult(value, points, "")


def fast_brier_from_weights(weights: HorizonWeights, risk_at_horizon: np.ndarray) -> float | None:
    """Vectorized equivalent of `horizon_brier_point`'s score, given precomputed `HorizonWeights`.

    `delayed_entry_ipcw_brier` (the reused model-training core) computes this exact quantity but
    loops
    over individual rows calling `KaplanMeierFitter.predict` on one scalar at a time -- cheap at
    model-training workflow's inner-fold-tuning call volume, but the dominant cost at
    primary-evaluation workflow's bootstrap volume
    (2000 resamples x up to 14 IBS grid points x 48 scopes). This function computes the identical
    formula using the already-vectorized `horizon_weights` (shared with every other horizon-specific
    metric) instead, and exists *only* for the bootstrap resampling path -- `horizon_brier_point`
    (calling the reused function directly) remains the source of truth for every one-time point
    estimate. `test_prediction_error.py` cross-validates the two against each other and against
    model-training workflow's own locked fixture on many random cases, not merely by shared
    authorship.
    """
    n_eligible = int(weights.eligible.sum())
    if n_eligible == 0:
        return None
    predicted_survival = 1.0 - np.asarray(risk_at_horizon, dtype=float)
    target = weights.survived.astype(float)
    return float(np.sum(weights.weight * (target - predicted_survival) ** 2) / n_eligible)


def fast_horizon_brier(
    km: KaplanMeierFitter,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_at_horizon: np.ndarray,
    horizon: float,
) -> float | None:
    """`fast_brier_from_weights` from raw inputs, fitting its own `HorizonWeights` -- convenience
    wrapper for standalone (non-shared-`km`) callers and cross-validation tests."""
    weights = horizon_weights(km, entry, exit_, event, horizon)
    return fast_brier_from_weights(weights, risk_at_horizon)


def fast_integrated_brier_score(
    km: KaplanMeierFitter,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_grid: np.ndarray,
    grid: tuple[float, ...] = IBS_GRID,
) -> float | None:
    """Vectorized equivalent of `integrated_brier_score`'s value, sharing one `km` fit across every
    grid point (`integrated_brier_score` itself refits the censoring KM redundantly inside each of
    its 14 `horizon_brier_point` calls, since each is independently callable). Bootstrap-path use
    only; see `fast_brier_from_weights`."""
    risk_grid = np.asarray(risk_grid, dtype=float)
    scores = []
    for index, horizon in enumerate(grid):
        weights = horizon_weights(km, entry, exit_, event, horizon)
        score = fast_brier_from_weights(weights, risk_grid[:, index])
        if score is None:
            return None
        scores.append(score)
    value = float(np.trapezoid(np.array(scores), grid) / IBS_WINDOW_WIDTH)
    return value if np.isfinite(value) else None


__all__ = [
    "BrierPoint",
    "IbsResult",
    "fast_brier_from_weights",
    "fast_horizon_brier",
    "fast_integrated_brier_score",
    "horizon_brier_point",
    "integrated_brier_score",
]
