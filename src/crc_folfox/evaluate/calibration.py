"""Weighted-logistic calibration slope/CITL, the prognostic-index slope, and quintile curves.

`calibration_slope`/`citl` follow evaluation-spec.md's "Calibration": a weighted binomial GLM
regression of the horizon event indicator `D(t)` on `cloglog(risk)`, IPCW-weighted by the
delayed-entry censoring KM (shared with discrimination/prediction-error via
`crc_folfox.evaluate.censoring.horizon_weights`), with CITL estimated by constraining the slope to
1 (`cloglog(risk)` entered as a fixed GLM offset).

The GLM link is deliberately the complementary log-log link, not the default logit link: a Cox
model's absolute risk is `r(t) = 1 - S0(t) ** exp(lp)`, so `cloglog(r(t)) = lp + log(H0(t))` is
exactly linear in the prognostic index, and a model with `cloglog(P(D=1)) = a + b * cloglog(r(t))`
recovers `a=0, b=1` *algebraically* at perfect calibration (`cloglog` is invertible, so `b=1, a=0`
implies `P(D=1) = r(t)` exactly). The default logit link does not have this property -- fitting
`logit(P(D=1)) = a + b * cloglog(r(t))` does not reduce to `b=1` even when `P(D=1) = r(t)` exactly,
since `logit(p) != cloglog(p)` in general. This was caught by
`test_calibration_slope_and_citl_on_a_perfectly_calibrated_fixture` failing at `slope != 1` on an
exact fixture before the link was corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter

from crc_folfox.evaluate.censoring import horizon_weights
from crc_folfox.evaluate.contracts import (
    MIN_CALIBRATION_CELL_SUPPORT,
    MIN_CALIBRATION_GROUP_SUPPORT,
    N_QUANTILE_GROUPS,
    RISK_CLIP,
)

_NORMAL_95 = 1.959963984540054


@dataclass(frozen=True)
class CalibrationResult:
    slope: float | None
    citl: float | None
    reason: str
    n_eligible: int
    n_events: int
    clipped_count: int


def horizon_calibration(
    km: KaplanMeierFitter,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_at_horizon: np.ndarray,
    horizon: float,
) -> CalibrationResult:
    weights = horizon_weights(km, entry, exit_, event, horizon)
    eligible_idx = np.flatnonzero(weights.eligible)
    n_eligible = len(eligible_idx)
    n_events = int(weights.occurred.sum())
    if n_eligible == 0:
        return CalibrationResult(None, None, "no_eligible_rows", 0, 0, 0)

    risk = np.asarray(risk_at_horizon, dtype=float)[eligible_idx]
    if not np.isfinite(risk).all():
        return CalibrationResult(None, None, "nonfinite_risk", n_eligible, n_events, 0)
    clipped = np.clip(risk, RISK_CLIP, 1.0 - RISK_CLIP)
    clipped_count = int(np.sum(~np.isclose(clipped, risk)))
    cloglog = np.log(-np.log(1.0 - clipped))

    outcome = weights.occurred.astype(float)[eligible_idx]
    weight = weights.weight[eligible_idx]
    if not np.isfinite(weight).all() or weight.sum() <= 0:
        return CalibrationResult(
            None, None, "degenerate_weights", n_eligible, n_events, clipped_count
        )
    if len(np.unique(outcome[weight > 0])) < 2:
        return CalibrationResult(
            None, None, "no_outcome_variation", n_eligible, n_events, clipped_count
        )

    cloglog_binomial = sm.families.Binomial(link=sm.families.links.CLogLog())
    try:
        slope_fit = sm.GLM(
            outcome, sm.add_constant(cloglog), family=cloglog_binomial, freq_weights=weight
        ).fit()
        slope = float(slope_fit.params[1])
    except Exception as exc:  # statsmodels can raise a variety of numerical-failure types
        return CalibrationResult(
            None,
            None,
            f"slope_fit_failed:{type(exc).__name__}",
            n_eligible,
            n_events,
            clipped_count,
        )

    try:
        citl_fit = sm.GLM(
            outcome,
            np.ones_like(outcome),
            family=cloglog_binomial,
            freq_weights=weight,
            offset=cloglog,
        ).fit()
        citl = float(citl_fit.params[0])
    except Exception as exc:
        return CalibrationResult(
            slope,
            None,
            f"citl_fit_failed:{type(exc).__name__}",
            n_eligible,
            n_events,
            clipped_count,
        )

    if not np.isfinite(slope) or not np.isfinite(citl):
        return CalibrationResult(
            slope if np.isfinite(slope) else None,
            citl if np.isfinite(citl) else None,
            "nonfinite_fit",
            n_eligible,
            n_events,
            clipped_count,
        )
    return CalibrationResult(slope, citl, "", n_eligible, n_events, clipped_count)


def prognostic_index_slope(
    entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, linear_predictor: np.ndarray
) -> tuple[float | None, str]:
    """Delayed-entry Cox fit with the linear predictor as sole covariate; ideal coefficient is 1."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "entry": np.asarray(entry, dtype=float),
            "exit": np.asarray(exit_, dtype=float),
            "event": np.asarray(event, dtype=int),
            "lp": np.asarray(linear_predictor, dtype=float),
        }
    )
    if not np.isfinite(frame["lp"]).all() or frame["lp"].nunique() < 2:
        return None, "degenerate_linear_predictor"
    try:
        fitter = CoxPHFitter()
        fitter.fit(frame, duration_col="exit", event_col="event", entry_col="entry")
    except Exception as exc:
        return None, f"fit_failed:{type(exc).__name__}"
    coefficient = float(fitter.params_["lp"])
    if not np.isfinite(coefficient):
        return None, "nonfinite_fit"
    return coefficient, ""


@dataclass(frozen=True)
class CalibrationGroup:
    group: int
    n: int
    n_events: int
    mean_predicted_risk: float
    observed_risk: float | None
    observed_risk_se: float | None
    low_support: bool


@dataclass(frozen=True)
class CalibrationCurveResult:
    status: str
    groups: tuple[CalibrationGroup, ...]


def _survival_ci_at(km: KaplanMeierFitter, horizon: float) -> tuple[float, float | None]:
    survival = float(np.atleast_1d(np.asarray(km.predict(np.array([horizon])), dtype=float))[0])
    try:
        ci = km.confidence_interval_survival_function_
        times = ci.index.to_numpy(dtype=float)
        index = max(int(np.searchsorted(times, horizon, side="right")) - 1, 0)
        lower = float(ci.iloc[index, 0])
        upper = float(ci.iloc[index, 1])
        se = (upper - lower) / (2 * _NORMAL_95)
    except Exception:
        se = None
    return survival, se


def calibration_curve_groups(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_at_horizon: np.ndarray,
    horizon: float,
    *,
    n_groups: int = N_QUANTILE_GROUPS,
    min_group_support: int = MIN_CALIBRATION_GROUP_SUPPORT,
    min_cell_support: int = MIN_CALIBRATION_CELL_SUPPORT,
) -> CalibrationCurveResult:
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_at_horizon, dtype=float)
    eligible = (entry <= horizon) & np.isfinite(risk)
    if int(eligible.sum()) < min_cell_support:
        return CalibrationCurveResult("insufficient_support_for_curve", ())

    order = np.flatnonzero(eligible)
    order = order[np.argsort(risk[order], kind="stable")]
    groups: list[CalibrationGroup] = []
    for index, member_idx in enumerate(np.array_split(order, n_groups)):
        if not len(member_idx):
            continue
        group_entry = entry[member_idx]
        group_exit = exit_[member_idx]
        group_event = event[member_idx]
        try:
            km = KaplanMeierFitter().fit(group_exit, event_observed=group_event, entry=group_entry)
            survival, se = _survival_ci_at(km, horizon)
            observed_risk: float | None = 1.0 - survival
        except Exception:
            observed_risk, se = None, None
        groups.append(
            CalibrationGroup(
                group=index,
                n=len(member_idx),
                n_events=int(group_event.sum()),
                mean_predicted_risk=float(np.mean(risk[member_idx])),
                observed_risk=observed_risk,
                observed_risk_se=se,
                low_support=len(member_idx) < min_group_support,
            )
        )
    return CalibrationCurveResult("ok", tuple(groups))


__all__ = [
    "CalibrationCurveResult",
    "CalibrationGroup",
    "CalibrationResult",
    "calibration_curve_groups",
    "horizon_calibration",
    "prognostic_index_slope",
]
