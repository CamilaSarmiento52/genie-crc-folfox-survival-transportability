"""E3: penalized selection stability for sparsity-capable models.

`explainability-spec.md`, "E3": selection frequency per feature group across the `B_explain`
bootstrap refits -- the proportion of refits in which at least one column of the group has
`|coefficient| > 1e-10`. Pure-ridge candidates never produce exact zeros and report
`not_applicable_no_sparsity` instead of a spurious 100% selection rate.
"""

from __future__ import annotations

import numpy as np

from crc_folfox.explain.contracts import SPARSITY_CAPABLE_MODEL_NAMES

SELECTION_THRESHOLD = 1e-10
NOT_APPLICABLE_NO_SPARSITY = "not_applicable_no_sparsity"


def is_sparsity_capable(model_name: str) -> bool:
    return model_name in SPARSITY_CAPABLE_MODEL_NAMES


def selection_frequency(
    groups: dict[str, tuple[str, ...]],
    coefficients_by_refit: list[dict[str, float] | None],
) -> dict[str, float | None]:
    """`{feature_group: proportion_of_refits_selected}`. A group is "selected" in one refit when
    any of its columns exceeds the threshold *within that same refit's own coefficient dict* -- a
    failed refit (`None`) or a column absent from that refit's design is treated as unselected in
    that refit, not as missing.

    `coefficients_by_refit` (`explain.refits.RefitCollection.coefficients_by_refit`) is one dict
    per attempted refit, in bootstrap-ID order -- the 2026-08-11 fix for a real risk in the prior
    per-column-compacted-list shape: comparing "index 3 of column A's list" against "index 3 of
    column B's list" silently compared *different* bootstrap draws whenever A and B did not fail
    (or get pooled away) in exactly the same refits. Checking a group's columns jointly within one
    refit's own dict makes that comparison always genuinely within a single resample.
    """
    n_refits = len(coefficients_by_refit)
    if n_refits <= 0:
        raise ValueError("selection_frequency requires at least one refit")
    result: dict[str, float | None] = {}
    for group_name, columns in groups.items():
        selected_count = 0
        for refit_coefficients in coefficients_by_refit:
            if refit_coefficients is None:
                continue
            selected = any(
                abs(refit_coefficients.get(column, 0.0)) > SELECTION_THRESHOLD for column in columns
            )
            selected_count += int(selected)
        result[group_name] = selected_count / n_refits
    return result


def bootstrap_coefficient_values(
    refit_coefficients: dict[str, list[float]], column: str
) -> np.ndarray:
    """One design column's observed coefficient values across every successful bootstrap refit at
    the frozen, already-selected configuration -- the bootstrap-refit coefficient *sample*, used
    for this artifact's own percentile/quantile summaries. This is a different quantity from
    `coefficient_paths.csv`'s per-model tuning-grid trajectory (`explainability-spec.md`'s
    2026-08-11 amendment); the two are never conflated, hence this function's name -- it was
    previously (and misleadingly) named `coefficient_paths`."""
    return np.asarray(refit_coefficients.get(column, []), dtype=float)


__all__ = [
    "NOT_APPLICABLE_NO_SPARSITY",
    "SELECTION_THRESHOLD",
    "bootstrap_coefficient_values",
    "is_sparsity_capable",
    "selection_frequency",
]
