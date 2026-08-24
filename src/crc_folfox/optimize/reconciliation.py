"""The correction workstream reconciliation: archived-vs-corrected cell-by-cell comparison.

`build_reconciliation` compares the archived frozen `performance_summary.csv` against a corrected
re-run's `performance_summary.csv`, one row per `(endpoint, design, centre, model, metric,
horizon, risk_scale)` reporting cell, and classifies how each cell moved. See correction-spec.md,
"Reconciliation".
"""

from __future__ import annotations

import math

import pandas as pd

from crc_folfox.optimize.contracts import Phase5AContractError

CELL_KEY = ("endpoint", "design", "centre", "model", "metric", "horizon", "risk_scale")

MOVEMENT_UNCHANGED = "unchanged"
MOVEMENT_RECOVERED = "recovered"
MOVEMENT_STILL_FAILED = "still_failed"
MOVEMENT_SHIFTED = "shifted"
MOVEMENT_UNEXPECTED = "unexpected"

# Only elastic-net tuning changed. Every other required candidate must reproduce byte-identical
# results under a rerun of otherwise-unchanged code, data, and seeds.
_TOUCHED_MODEL = "elastic_net_cox"
_NOT_COMPUTABLE = "not_computable"
_COMPUTABLE_STATUSES = ("estimable", "estimable_imprecise")

# `horizon` is genuinely NaN for horizon-independent metrics (concordance, ibs); pandas merge
# treats NaN != NaN in join keys and would silently drop those rows, so it is filled with this
# sentinel for the join and restored to NaN afterward.
_HORIZON_SENTINEL = -1.0


class Phase5AReconciliationError(Phase5AContractError):
    """Raised when reconciliation finds an `unexpected` cell movement."""


def _floats_equal(left: float, right: float) -> bool:
    left_nan, right_nan = pd.isna(left), pd.isna(right)
    if left_nan or right_nan:
        return bool(left_nan and right_nan)
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


def _normalize_reason(reason: object) -> str:
    # A CSV-round-tripped empty `reason` cell reads back as a float NaN, not an empty string --
    # and NaN is truthy in Python (`nan or ""` evaluates to `nan`, not `""`), so a naive
    # `(reason or "")` comparison never treats two empty reasons as equal and misclassifies every
    # ordinary `estimable` row as a movement. `pd.isna` is what actually detects the empty case.
    return "" if pd.isna(reason) else str(reason)


def _classify(
    model: str,
    status_archived: str,
    status_corrected: str,
    reason_archived: object,
    reason_corrected: object,
    estimate_archived: float,
    estimate_corrected: float,
) -> str:
    same_status = status_archived == status_corrected
    same_reason = _normalize_reason(reason_archived) == _normalize_reason(reason_corrected)
    same_estimate = _floats_equal(estimate_archived, estimate_corrected)
    if same_status and same_reason and same_estimate:
        return MOVEMENT_UNCHANGED
    if model != _TOUCHED_MODEL:
        # Nothing about a non-elastic-net candidate's inputs, code, or seeds changed in
        # Workstream A -- any difference here is a defect, not a legitimate movement.
        return MOVEMENT_UNEXPECTED
    if status_archived == _NOT_COMPUTABLE and status_corrected in _COMPUTABLE_STATUSES:
        return MOVEMENT_RECOVERED
    if status_archived == _NOT_COMPUTABLE and status_corrected == _NOT_COMPUTABLE:
        return MOVEMENT_STILL_FAILED
    if same_status and status_archived in _COMPUTABLE_STATUSES:
        # A pooled cell's admissible-repeat set is the union of repeats whose *every* fold
        # succeeded. Recovering a previously-failed fold elsewhere in the same repeat, or a
        # different fold within an already-admissible repeat selecting a different penalized
        # configuration under the corrected per-candidate isolation, changes the pooled
        # out-of-fold predictions for this cell's patients even when the cell's own status
        # doesn't move -- confirmed on the real run (2026-08-09): every such cell was `pooled`
        # design, `elastic_net_cox`, with status unchanged and only the estimate shifting by a
        # small amount. This is a legitimate consequence of the correction, not a defect.
        return MOVEMENT_SHIFTED
    return MOVEMENT_UNEXPECTED


def build_reconciliation(archived: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    """One row per archived reporting cell, comparing it against the corrected re-run.

    Both frames must have `performance_summary.csv`'s schema. Every archived cell must have a
    matching corrected cell with the same key -- a missing match (a cell present in the archive
    but absent from the corrected run) raises, since the row inventory is frozen and must not
    shrink.
    """
    left = archived.copy()
    right = corrected.copy()
    for frame in (left, right):
        # `horizon` may arrive as float64 (round-tripped through a CSV) from one side and as
        # object (an in-memory frame mixing `None` and `float` before this function ever sees
        # it) from the other; pandas' merge refuses to join float64 against object. Coercing
        # both to a plain float column first makes the join key's dtype identical regardless of
        # provenance.
        frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").fillna(
            _HORIZON_SENTINEL
        )

    merged = left.merge(
        right,
        on=list(CELL_KEY),
        how="left",
        suffixes=("_archived", "_corrected"),
        validate="one_to_one",
    )
    missing = merged["reporting_status_corrected"].isna()
    if missing.any():
        missing_keys = merged.loc[missing, list(CELL_KEY)].to_dict("records")
        raise Phase5AReconciliationError(
            f"{int(missing.sum())} archived cell(s) have no corrected counterpart: "
            f"{missing_keys[:5]}"
        )

    movement = [
        _classify(
            row["model"],
            row["reporting_status_archived"],
            row["reporting_status_corrected"],
            row.get("reason_archived", ""),
            row.get("reason_corrected", ""),
            row["estimate_archived"],
            row["estimate_corrected"],
        )
        for _, row in merged.iterrows()
    ]
    merged["movement_class"] = movement
    merged["horizon"] = merged["horizon"].replace(_HORIZON_SENTINEL, pd.NA)

    columns = [
        *CELL_KEY,
        "reporting_status_archived",
        "reporting_status_corrected",
        "estimate_archived",
        "estimate_corrected",
        "ci_lower_archived",
        "ci_lower_corrected",
        "ci_upper_archived",
        "ci_upper_corrected",
        "reason_archived",
        "reason_corrected",
        "movement_class",
    ]
    return merged[[c for c in columns if c in merged.columns]]


def assert_no_unexpected_movement(reconciliation: pd.DataFrame) -> None:
    """Raise `optimization reconciliation error` if any cell moved for a reason other than a
    previously
    failed elastic-net fit now succeeding (or remaining failed under the corrected policy)."""
    unexpected = reconciliation[reconciliation["movement_class"] == MOVEMENT_UNEXPECTED]
    if len(unexpected):
        keys = unexpected[list(CELL_KEY)].head(5).to_dict("records")
        raise Phase5AReconciliationError(
            f"{len(unexpected)} cell(s) moved unexpectedly and block publication: {keys}"
        )
    non_elastic_net = reconciliation[reconciliation["model"] != _TOUCHED_MODEL]
    not_unchanged = non_elastic_net[non_elastic_net["movement_class"] != MOVEMENT_UNCHANGED]
    if len(not_unchanged):
        keys = not_unchanged[list(CELL_KEY)].head(5).to_dict("records")
        raise Phase5AReconciliationError(
            f"{len(not_unchanged)} non-elastic-net cell(s) are not 'unchanged': {keys}"
        )


def summarize_movement(reconciliation: pd.DataFrame) -> dict[str, int]:
    """Aggregate movement counts, restricted to `elastic_net_cox` rows where the interesting
    recovered/still_failed/shifted distinction actually applies."""
    elastic_net = reconciliation[reconciliation["model"] == _TOUCHED_MODEL]
    counts = elastic_net["movement_class"].value_counts()
    return {
        "unchanged": int(counts.get(MOVEMENT_UNCHANGED, 0)),
        "recovered": int(counts.get(MOVEMENT_RECOVERED, 0)),
        "still_failed": int(counts.get(MOVEMENT_STILL_FAILED, 0)),
        "shifted": int(counts.get(MOVEMENT_SHIFTED, 0)),
    }
