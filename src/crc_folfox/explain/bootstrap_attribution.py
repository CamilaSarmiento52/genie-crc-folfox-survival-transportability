"""E5: bootstrap-derived importance, rank, and sign-stability --
`bootstrap_attribution_stability.csv` (`explainability-spec.md`'s 2026-08-11 amendment).

A separate artifact from the E2 point-estimate `importance_summary.csv`, cleaner than adding many
bootstrap columns onto it. Every gate below is enforced explicitly and separably: a group's
`n_successful/n_attempted` refit-success rate is a model-refit diagnostic only, never itself the
publication gate for importance (a successful refit can still produce a nonfinite Brier importance
for one metric); the rank gate is a stricter, refit-level completeness requirement than the
per-group importance gate; and the sign-stability gate is independent of both, using its own
`n_group_estimable` denominator (successful, `group_active`, and dominant-coefficient-nonzero).
"""

from __future__ import annotations

import numpy as np

from crc_folfox.explain.feature_groups import ZERO_COEFFICIENT_THRESHOLD, dominant_group_coefficient
from crc_folfox.explain.refits import RefitCollection

METRICS = ("365", "730", "ibs")

REPORTING_ESTIMABLE = "estimable"
REPORTING_NOT_COMPUTABLE = "not_computable"

SIGN_STABILITY_ESTIMABLE = "estimable"
SIGN_STABILITY_NOT_APPLICABLE_ZERO = "not_applicable_point_dominant_sign_zero"
SIGN_STABILITY_NOT_COMPUTABLE = "not_computable_insufficient_group_estimable_refits"

# The spec's existing "at least 90% of the 200 refits succeeded" convention
# (`explainability-spec.md`, "Support gates and reporting status"), reused explicitly here for the
# importance, rank, and sign-stability gates alike.
MIN_FINITE_FRACTION = 0.90


def _percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lower, upper = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(lower), float(upper)


def _sign_stability_for_group(
    group_name: str,
    columns: tuple[str, ...],
    successful_records,
    n_successful: int,
    point_dominant: float | None,
) -> tuple[float | None, str, int, int]:
    """Returns `(bootstrap_sign_stability, sign_stability_status, n_group_active,
    n_group_estimable)`."""
    n_group_active = sum(
        1
        for r in successful_records
        if r.group_active_by_group and r.group_active_by_group.get(group_name)
    )
    if point_dominant is None or abs(point_dominant) <= ZERO_COEFFICIENT_THRESHOLD:
        return None, SIGN_STABILITY_NOT_APPLICABLE_ZERO, n_group_active, 0

    point_sign = np.sign(point_dominant)
    n_group_estimable = 0
    n_sign_matches = 0
    for record in successful_records:
        if not (record.group_active_by_group and record.group_active_by_group.get(group_name)):
            continue
        dominant = dominant_group_coefficient(columns, record.coefficients_by_column or {})
        if dominant is None or abs(dominant[1]) <= ZERO_COEFFICIENT_THRESHOLD:
            continue
        n_group_estimable += 1
        if np.sign(dominant[1]) == point_sign:
            n_sign_matches += 1

    if n_successful and n_group_estimable / n_successful >= MIN_FINITE_FRACTION:
        return (
            n_sign_matches / n_group_estimable if n_group_estimable else None,
            SIGN_STABILITY_ESTIMABLE,
            n_group_active,
            n_group_estimable,
        )
    return None, SIGN_STABILITY_NOT_COMPUTABLE, n_group_active, n_group_estimable


def build_bootstrap_attribution_rows(
    collection: RefitCollection | None,
    canonical_groups: dict[str, tuple[str, ...]],
    point_coefficients_by_group: dict[str, float],
) -> list[dict]:
    """One row per `(feature_group, metric)`, `metric` in `{365, 730, ibs}` -- the caller attaches
    `endpoint`/`centre`/`model` (this module has no scope context of its own)."""
    if collection is None or collection.n_attempted == 0:
        return []

    n_attempted = collection.n_attempted
    n_successful = collection.n_succeeded
    successful_records = collection.successful_records
    rank_estimable_by_metric = {
        metric: collection.complete_rank_refit_count(metric, canonical_groups) for metric in METRICS
    }

    rows: list[dict] = []
    for group_name, columns in canonical_groups.items():
        bootstrap_sign_stability, sign_stability_status, n_group_active, n_group_estimable = (
            _sign_stability_for_group(
                group_name,
                columns,
                successful_records,
                n_successful,
                point_coefficients_by_group.get(group_name),
            )
        )

        for metric in METRICS:
            importance_values = collection.importance_series(group_name, metric)
            n_importance_estimable = len(importance_values)
            n_rank_estimable = rank_estimable_by_metric[metric]

            if n_attempted and n_importance_estimable / n_attempted >= MIN_FINITE_FRACTION:
                bootstrap_median_importance = float(np.median(importance_values))
                bootstrap_ci_lower, bootstrap_ci_upper = _percentile_interval(importance_values)
                reporting_status = REPORTING_ESTIMABLE
            else:
                bootstrap_median_importance = None
                bootstrap_ci_lower = None
                bootstrap_ci_upper = None
                reporting_status = REPORTING_NOT_COMPUTABLE

            if n_attempted and n_rank_estimable / n_attempted >= MIN_FINITE_FRACTION:
                rank_values = collection.rank_series(group_name, metric)
                median_rank = float(np.median(rank_values)) if rank_values else None
                rank_ci_lower, rank_ci_upper = _percentile_interval(rank_values)
            else:
                median_rank = None
                rank_ci_lower = None
                rank_ci_upper = None

            rows.append(
                {
                    "metric": metric,
                    "feature_group": group_name,
                    "bootstrap_median_importance": bootstrap_median_importance,
                    "bootstrap_ci_lower": bootstrap_ci_lower,
                    "bootstrap_ci_upper": bootstrap_ci_upper,
                    "median_rank": median_rank,
                    "rank_ci_lower": rank_ci_lower,
                    "rank_ci_upper": rank_ci_upper,
                    "bootstrap_sign_stability": bootstrap_sign_stability,
                    "sign_stability_status": sign_stability_status,
                    "n_attempted": n_attempted,
                    "n_successful": n_successful,
                    "n_importance_estimable": n_importance_estimable,
                    "n_rank_estimable": n_rank_estimable,
                    "n_group_active": n_group_active,
                    "n_group_estimable": n_group_estimable,
                    "reporting_status": reporting_status,
                }
            )
    return rows


__all__ = [
    "METRICS",
    "MIN_FINITE_FRACTION",
    "REPORTING_ESTIMABLE",
    "REPORTING_NOT_COMPUTABLE",
    "SIGN_STABILITY_ESTIMABLE",
    "SIGN_STABILITY_NOT_APPLICABLE_ZERO",
    "SIGN_STABILITY_NOT_COMPUTABLE",
    "build_bootstrap_attribution_rows",
]
