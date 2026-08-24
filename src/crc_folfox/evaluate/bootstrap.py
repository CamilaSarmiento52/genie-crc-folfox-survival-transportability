"""Paired patient-level bootstrap: one shared index matrix per `(endpoint, centre)`.

Per `evaluation-spec.md`, "Uncertainty": the seed tuple is deliberately `root_seed | endpoint |
centre | evaluation_bootstrap` -- design, model, horizon, risk scale, and metric are absent so the
same 2000 resample index sets are reused for every comparison of that `(endpoint, centre)`, which is
what makes the pooled-vs-IECV and fuller-vs-minimal differences genuinely paired.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from crc_folfox.evaluate.contracts import BOOTSTRAP_RESAMPLES, EvaluationContractError
from crc_folfox.preprocessing.splits import derive_seed


def bootstrap_seed(endpoint: str, centre: str) -> int:
    """The preprocessing workflow SHA-256 child-seed mechanic, keyed only on endpoint, centre,
    and purpose."""
    return derive_seed(
        endpoint=endpoint,
        design="",
        held_out_institution=centre,
        repeat=None,
        fold=None,
        purpose="evaluation_bootstrap",
    )


def bootstrap_index_matrix(
    endpoint: str, centre: str, n_patients: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> np.ndarray:
    """`(resamples, n_patients)` array of with-replacement resample indices into a fixed patient
    ordering. Deterministic in `(endpoint, centre)` alone."""
    if n_patients <= 0:
        raise EvaluationContractError("Bootstrap requires at least one patient")
    rng = np.random.default_rng(bootstrap_seed(endpoint, centre))
    return rng.integers(0, n_patients, size=(resamples, n_patients))


def run_bootstrap(
    index_matrix: np.ndarray, metric_fn: Callable[[np.ndarray], float | None]
) -> np.ndarray:
    """Apply `metric_fn` (recomputing eligibility/censoring/weights inside each resample) to every
    row of `index_matrix`. A raised `EvaluationContractError` or a `None` return both count as a
    nonfinite resample rather than aborting the run."""
    values = np.empty(index_matrix.shape[0], dtype=float)
    for row_index, indices in enumerate(index_matrix):
        try:
            value = metric_fn(indices)
        except EvaluationContractError:
            value = None
        values[row_index] = float("nan") if value is None else value
    return values


@dataclass(frozen=True)
class BootstrapSummary:
    value: float | None
    ci_lower: float | None
    ci_upper: float | None
    finite_fraction: float
    n_resamples: int


def summarize_bootstrap(
    point_estimate: float | None, resample_values: np.ndarray
) -> BootstrapSummary:
    """95% percentile interval over finite resample values; nonfinite resamples still count toward
    `n_resamples` and reduce `finite_fraction`, per the reporting-status rule's bootstrap-success
    gate."""
    n_resamples = int(len(resample_values))
    finite = resample_values[np.isfinite(resample_values)]
    finite_fraction = float(len(finite)) / n_resamples if n_resamples else 0.0
    if point_estimate is None or not len(finite):
        return BootstrapSummary(point_estimate, None, None, finite_fraction, n_resamples)
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return BootstrapSummary(
        point_estimate, float(lower), float(upper), finite_fraction, n_resamples
    )


__all__ = [
    "BootstrapSummary",
    "bootstrap_index_matrix",
    "bootstrap_seed",
    "run_bootstrap",
    "summarize_bootstrap",
]
