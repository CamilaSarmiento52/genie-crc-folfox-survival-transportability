"""The prespecified reporting-status rule, applied uniformly to every metric cell.

Per `evaluation-spec.md`, "Reporting-status rule": estimable requires a finite point estimate, at
least 20 eligible events, at least 20 event-free patients, at least 90% finite bootstrap resamples,
and a 95% interval no wider than the metric's locked precision target. Any support/bootstrap/
precision shortfall with a finite point estimate is "estimable -- imprecise", never "not
computable". A `None` point estimate (undefined estimator, upstream `fit_failed`, insufficient
admissible pooled repeats) is always "not computable", regardless of support.
"""

from __future__ import annotations

from dataclasses import dataclass

from crc_folfox.evaluate.contracts import (
    MIN_BOOTSTRAP_FINITE_FRACTION,
    MIN_ELIGIBLE_EVENTS,
    MIN_EVENT_FREE,
    REPORTING_ESTIMABLE,
    REPORTING_IMPRECISE,
    REPORTING_NOT_COMPUTABLE,
)


@dataclass(frozen=True)
class ReportingStatus:
    status: str
    reasons: tuple[str, ...]


def classify_reporting_status(
    *,
    value: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    n_events: int,
    n_event_free: int,
    bootstrap_finite_fraction: float,
    precision_target: float | None,
    upstream_reason: str = "",
) -> ReportingStatus:
    if value is None:
        reason = upstream_reason or "undefined_or_upstream_failure"
        return ReportingStatus(REPORTING_NOT_COMPUTABLE, (reason,))

    reasons: list[str] = []
    if n_events < MIN_ELIGIBLE_EVENTS:
        reasons.append("insufficient_events")
    if n_event_free < MIN_EVENT_FREE:
        reasons.append("insufficient_event_free")
    if bootstrap_finite_fraction < MIN_BOOTSTRAP_FINITE_FRACTION:
        reasons.append("insufficient_bootstrap_success")
    if precision_target is not None:
        if ci_lower is None or ci_upper is None:
            reasons.append("no_bootstrap_interval")
        elif (ci_upper - ci_lower) > precision_target:
            reasons.append("precision_target_missed")

    if reasons:
        return ReportingStatus(REPORTING_IMPRECISE, tuple(reasons))
    return ReportingStatus(REPORTING_ESTIMABLE, ())


__all__ = ["ReportingStatus", "classify_reporting_status"]
