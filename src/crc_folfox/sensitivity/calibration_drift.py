"""Calibration drift and local recalibration, decomposed in the fixed sequence `tech-stack.md`
locks: untouched -> intercept-only -> intercept-plus-slope -> exploratory local-adaptation learning
curve. Every recalibrated row is labelled `local_adaptation`, never external validation, and is
reported before-and-after against the frozen untouched prediction. Applied only to `minimal_cox`
and `spline_cox`, per `SENSITIVITY_MODEL_NAMES`.

Intercept-only and intercept-plus-slope recalibration reuse `evaluate.calibration.
horizon_calibration`'s own weighted cloglog-link GLM fit for the "before"/"after" report. That
function's own slope-fit already estimates the joint `(intercept, slope)` model but returns only
the slope; `_fit_joint_recalibration` below mirrors its exact family/link/weight setup (never a new
statistical method) purely to recover the intercept needed to reconstruct a recalibrated risk
vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from crc_folfox.evaluate.calibration import CalibrationResult, horizon_calibration
from crc_folfox.evaluate.censoring import fit_censoring_km, horizon_weights
from crc_folfox.evaluate.contracts import (
    MIN_ELIGIBLE_EVENTS as EVAL_MIN_ELIGIBLE_EVENTS,
)
from crc_folfox.evaluate.contracts import (
    MIN_EVENT_FREE,
    RISK_CLIP,
    EvaluationContractError,
)
from crc_folfox.preprocessing.splits import derive_seed
from crc_folfox.sensitivity.contracts import ANALYSIS_STATUS_LOCAL_ADAPTATION

STEP_UNTOUCHED = "untouched"
STEP_INTERCEPT_ONLY = "intercept_only"
STEP_INTERCEPT_SLOPE = "intercept_slope"
STEP_LOCAL_ADAPTATION = "local_adaptation_learning_curve"

LOCAL_ADAPTATION_FRACTIONS = (0.10, 0.20, 0.30)


def _apply_recalibration(risk: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    clipped = np.clip(np.asarray(risk, dtype=float), RISK_CLIP, 1.0 - RISK_CLIP)
    cloglog = np.log(-np.log(1.0 - clipped))
    adjusted = intercept + slope * cloglog
    return 1.0 - np.exp(-np.exp(adjusted))


def _fit_joint_recalibration(
    cloglog: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> tuple[float, float] | None:
    if not len(cloglog) or len(np.unique(outcome[weight > 0])) < 2:
        return None
    cloglog_binomial = sm.families.Binomial(link=sm.families.links.CLogLog())
    try:
        fit = sm.GLM(
            outcome, sm.add_constant(cloglog), family=cloglog_binomial, freq_weights=weight
        ).fit()
        intercept, slope = float(fit.params[0]), float(fit.params[1])
    except Exception:
        return None
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        return None
    return intercept, slope


@dataclass(frozen=True)
class CalibrationDriftStep:
    step: str
    horizon: float
    slope_before: float | None
    citl_before: float | None
    slope_after: float | None
    citl_after: float | None
    n_eligible: int
    n_events: int
    reason: str
    analysis_status: str


def _not_computable_step(
    step: str, horizon: float, before: CalibrationResult
) -> CalibrationDriftStep:
    return CalibrationDriftStep(
        step,
        horizon,
        before.slope,
        before.citl,
        None,
        None,
        before.n_eligible,
        before.n_events,
        before.reason or "recalibration_fit_failed",
        "" if step == STEP_UNTOUCHED else ANALYSIS_STATUS_LOCAL_ADAPTATION,
    )


def _horizon_column(horizon: float) -> int:
    return 0 if horizon == 365.0 else 13


def calibration_drift_steps(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_unconditional: np.ndarray,
    *,
    horizons: tuple[float, ...] = (365.0, 730.0),
) -> list[CalibrationDriftStep]:
    """Steps 1-3 of the fixed sequence: untouched, intercept-only, intercept-plus-slope."""
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    steps: list[CalibrationDriftStep] = []
    try:
        km = fit_censoring_km(entry, exit_, event)
    except EvaluationContractError:
        km = None

    for horizon in horizons:
        col = _horizon_column(horizon)
        risk = np.asarray(risk_unconditional, dtype=float)[:, col]
        if km is None:
            failed = CalibrationResult(None, None, "censoring_fit_failed", 0, 0, 0)
            steps.append(_not_computable_step(STEP_UNTOUCHED, horizon, failed))
            steps.append(_not_computable_step(STEP_INTERCEPT_ONLY, horizon, failed))
            steps.append(_not_computable_step(STEP_INTERCEPT_SLOPE, horizon, failed))
            continue

        untouched = horizon_calibration(km, entry, exit_, event, risk, horizon)
        steps.append(
            CalibrationDriftStep(
                STEP_UNTOUCHED,
                horizon,
                untouched.slope,
                untouched.citl,
                untouched.slope,
                untouched.citl,
                untouched.n_eligible,
                untouched.n_events,
                untouched.reason,
                "",
            )
        )

        if untouched.citl is None:
            steps.append(_not_computable_step(STEP_INTERCEPT_ONLY, horizon, untouched))
        else:
            recalibrated = _apply_recalibration(risk, untouched.citl, 1.0)
            after = horizon_calibration(km, entry, exit_, event, recalibrated, horizon)
            steps.append(
                CalibrationDriftStep(
                    STEP_INTERCEPT_ONLY,
                    horizon,
                    untouched.slope,
                    untouched.citl,
                    after.slope,
                    after.citl,
                    untouched.n_eligible,
                    untouched.n_events,
                    after.reason,
                    ANALYSIS_STATUS_LOCAL_ADAPTATION,
                )
            )

        weights = horizon_weights(km, entry, exit_, event, horizon)
        eligible_idx = np.flatnonzero(weights.eligible)
        clipped = np.clip(risk[eligible_idx], RISK_CLIP, 1.0 - RISK_CLIP)
        cloglog = np.log(-np.log(1.0 - clipped))
        outcome = weights.occurred.astype(float)[eligible_idx]
        weight_values = weights.weight[eligible_idx]
        joint = _fit_joint_recalibration(cloglog, outcome, weight_values)
        if joint is None:
            steps.append(_not_computable_step(STEP_INTERCEPT_SLOPE, horizon, untouched))
        else:
            intercept, slope = joint
            recalibrated = _apply_recalibration(risk, intercept, slope)
            after = horizon_calibration(km, entry, exit_, event, recalibrated, horizon)
            steps.append(
                CalibrationDriftStep(
                    STEP_INTERCEPT_SLOPE,
                    horizon,
                    untouched.slope,
                    untouched.citl,
                    after.slope,
                    after.citl,
                    untouched.n_eligible,
                    untouched.n_events,
                    after.reason,
                    ANALYSIS_STATUS_LOCAL_ADAPTATION,
                )
            )
    return steps


def _stratified_adaptation_split(
    event: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    event_idx = np.flatnonzero(event == 1)
    nonevent_idx = np.flatnonzero(event == 0)
    n_event = min(len(event_idx), max(1, round(len(event_idx) * fraction)))
    n_nonevent = min(len(nonevent_idx), max(1, round(len(nonevent_idx) * fraction)))
    adaptation = np.concatenate(
        [
            rng.choice(event_idx, size=n_event, replace=False) if len(event_idx) else event_idx,
            rng.choice(nonevent_idx, size=n_nonevent, replace=False)
            if len(nonevent_idx)
            else nonevent_idx,
        ]
    )
    evaluation = np.setdiff1d(np.arange(len(event)), adaptation)
    return adaptation, evaluation


def local_adaptation_rows(
    *,
    endpoint: str,
    centre: str,
    model_name: str,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_unconditional: np.ndarray,
    horizons: tuple[float, ...] = (365.0, 730.0),
    fractions: tuple[float, ...] = LOCAL_ADAPTATION_FRACTIONS,
) -> list[dict]:
    """Step 4: the exploratory 10/20/30% stratified local-adaptation learning curve. Both the
    adaptation and unused-evaluation subsets must satisfy the imported primary-evaluation workflow
    support gate before
    anything is fit; an infeasible fraction is reported explicitly, never manufactured."""
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    rows: list[dict] = []
    for fraction in fractions:
        seed = derive_seed(
            endpoint=endpoint,
            design="",
            held_out_institution=centre,
            purpose=f"local_adaptation|model={model_name}|fraction={fraction:g}",
        )
        adaptation_idx, evaluation_idx = _stratified_adaptation_split(event, fraction, seed)
        adaptation_events = int(event[adaptation_idx].sum())
        evaluation_events = int(event[evaluation_idx].sum())
        evaluation_event_free = len(evaluation_idx) - evaluation_events

        for horizon in horizons:
            row: dict = {
                "endpoint": endpoint,
                "centre": centre,
                "model": model_name,
                "fraction": fraction,
                "horizon": horizon,
                "n_adaptation": len(adaptation_idx),
                "n_adaptation_events": adaptation_events,
                "n_evaluation": len(evaluation_idx),
                "n_evaluation_events": evaluation_events,
                "feasible": False,
                "reason": "",
                "adaptation_intercept": "",
                "slope_before": "",
                "citl_before": "",
                "slope_after": "",
                "citl_after": "",
                "analysis_status": ANALYSIS_STATUS_LOCAL_ADAPTATION,
            }
            if (
                adaptation_events < EVAL_MIN_ELIGIBLE_EVENTS
                or evaluation_events < EVAL_MIN_ELIGIBLE_EVENTS
                or evaluation_event_free < MIN_EVENT_FREE
            ):
                row["reason"] = "insufficient_adaptation_or_evaluation_support"
                rows.append(row)
                continue

            col = _horizon_column(horizon)
            risk = np.asarray(risk_unconditional, dtype=float)[:, col]
            try:
                km_adapt = fit_censoring_km(
                    entry[adaptation_idx], exit_[adaptation_idx], event[adaptation_idx]
                )
            except EvaluationContractError:
                row["reason"] = "adaptation_censoring_fit_failed"
                rows.append(row)
                continue
            adapt_calibration = horizon_calibration(
                km_adapt,
                entry[adaptation_idx],
                exit_[adaptation_idx],
                event[adaptation_idx],
                risk[adaptation_idx],
                horizon,
            )
            if adapt_calibration.citl is None:
                row["reason"] = "adaptation_intercept_fit_failed"
                rows.append(row)
                continue

            try:
                km_eval = fit_censoring_km(
                    entry[evaluation_idx], exit_[evaluation_idx], event[evaluation_idx]
                )
            except EvaluationContractError:
                row["reason"] = "evaluation_censoring_fit_failed"
                rows.append(row)
                continue
            before = horizon_calibration(
                km_eval,
                entry[evaluation_idx],
                exit_[evaluation_idx],
                event[evaluation_idx],
                risk[evaluation_idx],
                horizon,
            )
            recalibrated_eval_risk = _apply_recalibration(
                risk[evaluation_idx], adapt_calibration.citl, 1.0
            )
            after = horizon_calibration(
                km_eval,
                entry[evaluation_idx],
                exit_[evaluation_idx],
                event[evaluation_idx],
                recalibrated_eval_risk,
                horizon,
            )
            row.update(
                {
                    "feasible": True,
                    "adaptation_intercept": adapt_calibration.citl,
                    "slope_before": before.slope,
                    "citl_before": before.citl,
                    "slope_after": after.slope,
                    "citl_after": after.citl,
                }
            )
            rows.append(row)
    return rows


__all__ = [
    "LOCAL_ADAPTATION_FRACTIONS",
    "STEP_INTERCEPT_ONLY",
    "STEP_INTERCEPT_SLOPE",
    "STEP_LOCAL_ADAPTATION",
    "STEP_UNTOUCHED",
    "CalibrationDriftStep",
    "calibration_drift_steps",
    "local_adaptation_rows",
]
