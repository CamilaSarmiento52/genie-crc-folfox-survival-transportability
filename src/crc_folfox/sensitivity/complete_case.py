"""Complete-case assessment-subset sensitivity.

Deliberately named "complete-case assessment-subset sensitivity," not a complete-case modelling
analysis: every frozen model and its frozen predictions are used unchanged (no refit, no new
registry entry). "Complete case" is recomputed directly from the raw data-cleaning workflow columns
for `crc_type`
and `ca_grade` -- the only two preprocessing workflow mode-imputed features (`tech-stack.md`,
"Preprocessing
rules") -- rather than assumed from the data-cleaning workflow audit constant (575/678 patients),
because that
figure predates endpoint eligibility and must not be assumed to carry over unchanged.

The full-population and complete-case comparisons reuse the identical `score_patient_subset`
primitive with only the subset mask differing, so the "full" side is byte-identical to what
primary-evaluation workflow's own evaluation already computed for that cell (both re-estimate
censoring on the rows
being scored, per the locked evaluation estimand) and the "complete-case" side re-estimates
censoring *within* the complete-case subset only -- never reusing the full population's nuisance
fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crc_folfox.sensitivity.contracts import COMPLETE_CASE_COLUMNS, COMPLETE_CASE_LIMITATION
from crc_folfox.sensitivity.subset_metrics import score_patient_subset


def complete_case_flag_frame(frame: pd.DataFrame) -> pd.Series:
    """A `patient_key`-indexed boolean flag: True where every complete-case column is observed
    (not imputed) in the raw cleaned analysis frame."""
    missing_any = frame.set_index("patient_key")[list(COMPLETE_CASE_COLUMNS)].isna().any(axis=1)
    return ~missing_any


def complete_case_mask_for(flag_frame: pd.Series, patient_key: tuple[str, ...]) -> np.ndarray:
    """Align the complete-case flag to one cell's own patient order."""
    return flag_frame.loc[list(patient_key)].to_numpy(dtype=bool)


def complete_case_comparison_rows(
    *,
    endpoint: str,
    design: str,
    centre: str,
    model_name: str,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_unconditional: np.ndarray,
    linear_predictor: np.ndarray,
    complete_case_mask: np.ndarray,
    index_matrix: np.ndarray,
) -> list[dict]:
    """Full-population-versus-complete-case comparison rows for one endpoint/design/centre/model
    cell, one row per metric."""
    full_mask = np.ones(len(entry), dtype=bool)
    full_scored = score_patient_subset(
        entry=entry,
        exit_=exit_,
        event=event,
        risk_unconditional=risk_unconditional,
        linear_predictor=linear_predictor,
        subset_mask=full_mask,
        index_matrix=index_matrix,
    )
    complete_case_scored = score_patient_subset(
        entry=entry,
        exit_=exit_,
        event=event,
        risk_unconditional=risk_unconditional,
        linear_predictor=linear_predictor,
        subset_mask=complete_case_mask,
        index_matrix=index_matrix,
    )

    rows: list[dict] = []
    for metric_key in full_scored:
        full = full_scored[metric_key]
        complete_case = complete_case_scored[metric_key]
        diff_estimate, diff_ci_lower, diff_ci_upper, diff_n_finite = _paired_difference(
            full.raw.resamples, complete_case.raw.resamples, full.raw.value, complete_case.raw.value
        )
        rows.append(
            {
                "endpoint": endpoint,
                "design": design,
                "centre": centre,
                "model": model_name,
                "metric": metric_key,
                "n_patients_full": int(full_mask.sum()),
                "n_events_full": full.raw.n_events,
                "full_estimate": full.raw.value,
                "full_ci_lower": full.ci_lower,
                "full_ci_upper": full.ci_upper,
                "full_reporting_status": full.status.status,
                "n_patients_complete_case": int(complete_case_mask.sum()),
                "n_events_complete_case": complete_case.raw.n_events,
                "complete_case_estimate": complete_case.raw.value,
                "complete_case_ci_lower": complete_case.ci_lower,
                "complete_case_ci_upper": complete_case.ci_upper,
                "complete_case_reporting_status": complete_case.status.status,
                "diff_estimate": diff_estimate,
                "diff_ci_lower": diff_ci_lower,
                "diff_ci_upper": diff_ci_upper,
                "diff_n_finite_resamples": diff_n_finite,
                "limitation": COMPLETE_CASE_LIMITATION,
            }
        )
    return rows


def _paired_difference(
    full_resamples: np.ndarray,
    complete_case_resamples: np.ndarray,
    full_value: float | None,
    complete_case_value: float | None,
) -> tuple[float | None, float | None, float | None, int]:
    """`complete-case - full`, from the *same* bootstrap row of the *same* index matrix for both
    sides (per `score_patient_subset`'s shared `index_matrix` contract), so this is a genuine paired
    bootstrap difference, not an independent-CI propagation. Requires >=2 rows where both sides
    resolved to a finite value on that row; a row where either side failed contributes nothing."""
    diff = np.asarray(complete_case_resamples, dtype=float) - np.asarray(
        full_resamples, dtype=float
    )
    finite = diff[np.isfinite(diff)]
    diff_estimate = (
        float(complete_case_value) - float(full_value)
        if full_value is not None and complete_case_value is not None
        else None
    )
    if len(finite) < 2:
        return diff_estimate, None, None, len(finite)
    ci_lower, ci_upper = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
    return diff_estimate, ci_lower, ci_upper, len(finite)


__all__ = [
    "complete_case_comparison_rows",
    "complete_case_flag_frame",
    "complete_case_mask_for",
]
