"""Delayed-entry (left-truncation-aware) censoring estimation and horizon IPCW weights.

The censoring distribution is the reverse Kaplan-Meier -- `KaplanMeierFitter(entry=...)` fitted on
`1 - event` -- evaluated on the assessment rows of the reporting cell (see evaluation-spec.md,
"Horizon eligibility and the locked estimand"). This module is the shared core every horizon-
specific metric (concordance, AUC, Brier, calibration) builds its weights from, so the delayed-entry
handling has exactly one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError

from crc_folfox.evaluate.contracts import EvaluationContractError


def fit_censoring_km(entry: np.ndarray, exit_: np.ndarray, event: np.ndarray) -> KaplanMeierFitter:
    """Fit the delayed-entry reverse Kaplan-Meier censoring distribution.

    A degenerate risk set (lifelines' own `StatError` for "too few early truncation times and too
    many events", i.e. `S(t) == 0` for all `t` past some point) is a legitimate zero-support failure
    under `evaluation-spec.md`'s "fails safely" rule, so it is reraised as the project's own
    privacy-safe contract error rather than left to propagate as a third-party exception type.
    """
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    if not (len(entry) == len(exit_) == len(event)):
        raise EvaluationContractError("Censoring inputs are not aligned")
    if not len(exit_):
        raise EvaluationContractError("Censoring estimation requires at least one row")
    try:
        return KaplanMeierFitter().fit(exit_, event_observed=1 - event, entry=entry)
    except StatError as exc:
        raise EvaluationContractError("Censoring support degenerates to zero") from exc


def predict_km(km: KaplanMeierFitter, times: np.ndarray) -> np.ndarray:
    """Evaluate a fitted KM survival curve at arbitrary times, always returning a float array."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    predicted = km.predict(times)
    # `KaplanMeierFitter.predict` returns a bare (0-dimensional) `np.float64` for a single-element
    # input and a `pd.Series` for a multi-element one; `atleast_1d` normalizes both to a 1-D array.
    values = predicted.to_numpy(dtype=float) if hasattr(predicted, "to_numpy") else predicted
    return np.atleast_1d(np.asarray(values, dtype=float))


def _safe_inverse(probability: np.ndarray) -> np.ndarray:
    if not np.isfinite(probability).all() or np.any(probability <= 0):
        raise EvaluationContractError("Censoring support is zero or nonfinite at a required time")
    return 1.0 / probability


@dataclass(frozen=True)
class HorizonWeights:
    """Per-patient horizon-`t` eligibility, outcome, and IPCW weight.

    `eligible` is `entry <= t`; `occurred` is an observed event by `t`; `survived` is known
    event-free at `t` (i.e. `exit > t`). A row that is eligible but censored before `t` gets weight
    zero and contributes to no numerator, matching the standard Graf et al. IPCW convention already
    validated for tuning in `crc_folfox.model_training.tuning.delayed_entry_ipcw_brier`.
    """

    horizon: float
    eligible: np.ndarray
    occurred: np.ndarray
    survived: np.ndarray
    weight: np.ndarray


def horizon_weights(
    km: KaplanMeierFitter,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    horizon: float,
) -> HorizonWeights:
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    eligible = entry <= horizon
    occurred = eligible & (exit_ <= horizon) & (event == 1)
    survived = eligible & (exit_ > horizon)
    weight = np.zeros(len(exit_), dtype=float)
    if occurred.any():
        points = np.nextafter(exit_[occurred], -np.inf)
        weight[occurred] = _safe_inverse(predict_km(km, points))
    if survived.any():
        points = np.full(int(survived.sum()), horizon, dtype=float)
        weight[survived] = _safe_inverse(predict_km(km, points))
    return HorizonWeights(
        horizon=horizon, eligible=eligible, occurred=occurred, survived=survived, weight=weight
    )


__all__ = ["HorizonWeights", "fit_censoring_km", "horizon_weights", "predict_km"]
