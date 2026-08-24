"""Score the frozen primary-evaluation workflow delayed-entry metric set on an arbitrary patient
subset, with no refit.

Shared by every sensitivity analysis that restricts an already-predicted cell to a boolean subset of
its own patients (complete-case assessment-subset, sequencing-lag threshold populations, supportive
endpoint robustness): the censoring/IPCW nuisance model is re-estimated *within* the subset (never
reused from the full population), and every metric uses the same imported primary-evaluation core
uses, never a reimplementation.

Bootstrap uncertainty reuses the *full* cell's `bootstrap_index_matrix(endpoint, centre, ...)` --
the same deterministic call primary-evaluation workflow's own evaluation makes -- and restricts each
resample row to the
positions that fall inside the subset, rather than drawing a newly seeded resampler. This is what
`sensitivity-analyses-spec.md` means by "restrictions of the existing primary-evaluation index
matrices to
complete-case patients, not a newly seeded resampler": every resample is a filtration of the primary
draw, so sensitivity-analysis workflow's uncertainty stays paired with primary-evaluation workflow's
own randomness for the same
`(endpoint, centre)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crc_folfox.evaluate.calibration import horizon_calibration
from crc_folfox.evaluate.censoring import fit_censoring_km, horizon_weights
from crc_folfox.evaluate.contracts import (
    METRIC_REGISTRY,
    EvaluationContractError,
)
from crc_folfox.evaluate.discrimination import time_dependent_auc, truncation_aware_concordance
from crc_folfox.evaluate.prediction_error import (
    fast_brier_from_weights,
    fast_integrated_brier_score,
    horizon_brier_point,
    integrated_brier_score,
)
from crc_folfox.evaluate.reporting import ReportingStatus, classify_reporting_status
from crc_folfox.sensitivity.contracts import SUBSET_METRIC_SPEC

_IDX_365, _IDX_730 = 0, 13  # positions of 365d/730d inside evaluate.contracts.IBS_GRID


@dataclass(frozen=True)
class SubsetMetricRaw:
    value: float | None
    resamples: np.ndarray
    n_eligible: int
    n_events: int
    n_event_free: int
    reason: str


@dataclass(frozen=True)
class SubsetScoredMetric:
    raw: SubsetMetricRaw
    status: ReportingStatus
    ci_lower: float | None
    ci_upper: float | None


def _not_computable(n_resamples: int, reason: str) -> SubsetMetricRaw:
    return SubsetMetricRaw(None, np.full(n_resamples, np.nan), 0, 0, 0, reason)


def _horizon_column(horizon: float) -> int:
    return _IDX_365 if horizon == 365.0 else _IDX_730


def _point_metrics(
    entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, risk_u: np.ndarray, lp: np.ndarray
) -> dict[str, tuple[float | None, int, int, int, str]]:
    """One-time point estimate for every `SUBSET_METRIC_SPEC` entry, using the "source of truth"
    (non-vectorized) primary-evaluation functions, exactly as `evaluate.workflow._score_cell`
    does."""
    out: dict[str, tuple[float | None, int, int, int, str]] = {}
    n_events_overall = int(event.sum())
    n_event_free_overall = int((event == 0).sum())

    conc = truncation_aware_concordance(entry, exit_, event, lp)
    out["concordance"] = (
        conc.value,
        len(entry),
        n_events_overall,
        n_event_free_overall,
        "" if conc.value is not None else "no_comparable_pairs",
    )

    ibs = integrated_brier_score(entry, exit_, event, risk_u)
    try:
        km = fit_censoring_km(entry, exit_, event)
        ibs_weights = horizon_weights(km, entry, exit_, event, 730.0)
        ibs_support = (
            int(ibs_weights.eligible.sum()),
            int(ibs_weights.occurred.sum()),
            int(ibs_weights.survived.sum()),
        )
    except EvaluationContractError:
        km = None
        ibs_support = (0, 0, 0)
    out["ibs"] = (ibs.value, *ibs_support, ibs.reason if ibs.value is None else "")

    for horizon in (365.0, 730.0):
        col = _horizon_column(horizon)
        if km is not None:
            weights = horizon_weights(km, entry, exit_, event, horizon)
            support = (
                int(weights.eligible.sum()),
                int(weights.occurred.sum()),
                int(weights.survived.sum()),
            )
        else:
            weights, support = None, (0, 0, 0)

        auc = time_dependent_auc(km, entry, exit_, event, risk_u[:, col], horizon) if km else None
        out[f"auc_{horizon:g}"] = (
            auc.value if auc else None,
            *support,
            "" if auc and auc.value is not None else "insufficient_case_or_control",
        )

        brier = horizon_brier_point(entry, exit_, event, risk_u[:, col], horizon)
        out[f"brier_{horizon:g}"] = (
            brier.score,
            *support,
            brier.reason if brier.score is None else "",
        )

        calibration = (
            horizon_calibration(km, entry, exit_, event, risk_u[:, col], horizon) if km else None
        )
        for metric_name, value in (
            ("slope", calibration.slope if calibration else None),
            ("citl", calibration.citl if calibration else None),
        ):
            out[f"{metric_name}_{horizon:g}"] = (
                value,
                calibration.n_eligible if calibration else support[0],
                calibration.n_events if calibration else support[1],
                (calibration.n_eligible - calibration.n_events) if calibration else support[2],
                (calibration.reason if calibration and value is None else ""),
            )
    return out


def _resample_metrics(
    entry: np.ndarray, exit_: np.ndarray, event: np.ndarray, risk_u: np.ndarray, lp: np.ndarray
) -> dict[str, float] | None:
    """Vectorized-core equivalent of `_point_metrics`, for one bootstrap resample row. Returns
    `None` for the whole row (every metric key becomes NaN) only when censoring itself cannot be
    fit at this resample's subset restriction -- otherwise every metric is scored independently, as
    in `evaluate.workflow._bootstrap_cell`."""
    if len(entry) < 2:
        return None
    try:
        km = fit_censoring_km(entry, exit_, event)
    except EvaluationContractError:
        return None
    out: dict[str, float] = {}
    conc = truncation_aware_concordance(entry, exit_, event, lp)
    out["concordance"] = _safe(conc.value)
    out["ibs"] = _safe(fast_integrated_brier_score(km, entry, exit_, event, risk_u))
    for horizon in (365.0, 730.0):
        col = _horizon_column(horizon)
        weights = horizon_weights(km, entry, exit_, event, horizon)
        out[f"auc_{horizon:g}"] = _safe(
            time_dependent_auc(km, entry, exit_, event, risk_u[:, col], horizon).value
        )
        out[f"brier_{horizon:g}"] = _safe(fast_brier_from_weights(weights, risk_u[:, col]))
        calibration = horizon_calibration(km, entry, exit_, event, risk_u[:, col], horizon)
        out[f"slope_{horizon:g}"] = _safe(calibration.slope)
        out[f"citl_{horizon:g}"] = _safe(calibration.citl)
    return out


def _safe(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _metric_key(metric: str, horizon: float | None) -> str:
    return metric if horizon is None else f"{metric}_{horizon:g}"


def score_patient_subset(
    *,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_unconditional: np.ndarray,
    linear_predictor: np.ndarray,
    subset_mask: np.ndarray,
    index_matrix: np.ndarray,
) -> dict[str, SubsetScoredMetric]:
    """Score every `SUBSET_METRIC_SPEC` entry on `subset_mask`, with subset-restricted bootstrap CI.

    `entry`/`exit_`/`event`/`risk_unconditional`/`linear_predictor` are the *full* cell's arrays
    (every patient the primary evaluation predicted for that endpoint/design/centre/model);
    `subset_mask` is a boolean array of the same length selecting the sensitivity's subset (e.g.
    complete-case, or an entry-lag threshold); `index_matrix` is the *full* cell's own
    `bootstrap_index_matrix(endpoint, centre, len(entry))` -- the identical call primary-evaluation
    workflow's own
    evaluation makes for that `(endpoint, centre)`, never a newly seeded one.
    """
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    risk_unconditional = np.asarray(risk_unconditional, dtype=float)
    linear_predictor = np.asarray(linear_predictor, dtype=float)
    subset_mask = np.asarray(subset_mask, dtype=bool)
    n = len(entry)
    if not (
        n
        == len(exit_)
        == len(event)
        == len(linear_predictor)
        == risk_unconditional.shape[0]
        == len(subset_mask)
    ):
        raise EvaluationContractError("Subset scoring inputs are not aligned")

    n_resamples = index_matrix.shape[0]
    subset_idx = np.flatnonzero(subset_mask)

    if len(subset_idx) < 2:
        reason = "insufficient_subset_support"
        raw_by_key = {
            _metric_key(metric, horizon): _not_computable(n_resamples, reason)
            for metric, horizon in SUBSET_METRIC_SPEC
        }
    else:
        point = _point_metrics(
            entry[subset_idx],
            exit_[subset_idx],
            event[subset_idx],
            risk_unconditional[subset_idx],
            linear_predictor[subset_idx],
        )
        resample_arrays: dict[str, np.ndarray] = {
            _metric_key(metric, horizon): np.full(n_resamples, np.nan)
            for metric, horizon in SUBSET_METRIC_SPEC
        }
        for row in range(n_resamples):
            sampled = index_matrix[row]
            restricted = sampled[subset_mask[sampled]]
            row_result = _resample_metrics(
                entry[restricted],
                exit_[restricted],
                event[restricted],
                risk_unconditional[restricted],
                linear_predictor[restricted],
            )
            if row_result is None:
                continue
            for key, value in row_result.items():
                resample_arrays[key][row] = value

        raw_by_key = {}
        for metric, horizon in SUBSET_METRIC_SPEC:
            key = _metric_key(metric, horizon)
            value, n_eligible, n_events, n_event_free, reason = point[key]
            raw_by_key[key] = SubsetMetricRaw(
                value, resample_arrays[key], n_eligible, n_events, n_event_free, reason
            )

    scored: dict[str, SubsetScoredMetric] = {}
    for metric, horizon in SUBSET_METRIC_SPEC:
        key = _metric_key(metric, horizon)
        raw = raw_by_key[key]
        finite = raw.resamples[np.isfinite(raw.resamples)]
        finite_fraction = float(len(finite)) / n_resamples if n_resamples else 0.0
        ci_lower = ci_upper = None
        if raw.value is not None and len(finite):
            ci_lower, ci_upper = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
        status = classify_reporting_status(
            value=raw.value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_events=raw.n_events,
            n_event_free=raw.n_event_free,
            bootstrap_finite_fraction=finite_fraction,
            precision_target=METRIC_REGISTRY[metric].precision_target,
            upstream_reason=raw.reason,
        )
        scored[key] = SubsetScoredMetric(raw, status, ci_lower, ci_upper)
    return scored


__all__ = ["SubsetMetricRaw", "SubsetScoredMetric", "score_patient_subset"]
