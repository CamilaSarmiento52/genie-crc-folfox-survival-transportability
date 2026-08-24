"""Sequencing-lag sensitivity: the headline figure named in `mission.md` success criterion 6.

Locked thresholds, from `tech-stack.md`'s "Sequencing-lag sensitivity (headline figure)": by
FOLFOX initiation -> <=90d -> <=180d -> <=365d -> unrestricted with delayed-entry adjustment. Every
point displays retained patients, eligible events, and entry distribution; delayed-entry survival
summaries or a pooled minimal-Cox sensitivity estimate are added only where mathematically
computable, and a performance cell is added only when the imported primary-evaluation
reporting-status gate is
met. No threshold is ever chosen or relabelled after inspecting a performance value --
`SEQUENCING_LAG_THRESHOLDS` is a fixed, non-configurable tuple.
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError

from crc_folfox.evaluate.contracts import CENTRES, IBS_GRID
from crc_folfox.preprocessing.contracts import EndpointView
from crc_folfox.sensitivity.contracts import SEQUENCING_LAG_THRESHOLDS
from crc_folfox.sensitivity.subset_metrics import score_patient_subset


def _threshold_mask(entry: np.ndarray, cutoff: float) -> np.ndarray:
    return entry <= cutoff


def lag_support_rows(view: EndpointView, endpoint: str) -> list[dict]:
    """Retained patients, eligible events, and entry-distribution summary at each threshold,
    overall (`group == "All"`) and by centre."""
    rows: list[dict] = []
    for group in (*CENTRES, "All"):
        data = view.data if group == "All" else view.data.loc[view.data["institution"].eq(group)]
        entry = data["entry"].to_numpy(dtype=float)
        event = data["event"].to_numpy(dtype=int)
        for label, cutoff in SEQUENCING_LAG_THRESHOLDS:
            mask = _threshold_mask(entry, cutoff)
            selected_entry = entry[mask]
            rows.append(
                {
                    "endpoint": endpoint,
                    "threshold": label,
                    "group": group,
                    "n_patients": int(mask.sum()),
                    "n_events": int(event[mask].sum()),
                    "entry_mean": float(selected_entry.mean()) if len(selected_entry) else "",
                    "entry_median": float(np.median(selected_entry)) if len(selected_entry) else "",
                    "entry_min": float(selected_entry.min()) if len(selected_entry) else "",
                    "entry_max": float(selected_entry.max()) if len(selected_entry) else "",
                }
            )
    return rows


def lag_km_rows(view: EndpointView, endpoint: str) -> list[dict]:
    """Delayed-entry Kaplan-Meier survival at 365/730 days for each threshold, overall and by
    centre, using `lifelines.KaplanMeierFitter(entry=...)` -- the actual survival curve, not the
    reverse-KM censoring nuisance model."""
    rows: list[dict] = []
    for group in (*CENTRES, "All"):
        data = view.data if group == "All" else view.data.loc[view.data["institution"].eq(group)]
        entry = data["entry"].to_numpy(dtype=float)
        exit_ = data["exit"].to_numpy(dtype=float)
        event = data["event"].to_numpy(dtype=int)
        for label, cutoff in SEQUENCING_LAG_THRESHOLDS:
            mask = _threshold_mask(entry, cutoff)
            n = int(mask.sum())
            row = {
                "endpoint": endpoint,
                "threshold": label,
                "group": group,
                "n_patients": n,
                "survival_365": "",
                "survival_730": "",
                "status": "not_computable",
                "reason": "",
            }
            if n < 2:
                row["reason"] = "insufficient_support"
                rows.append(row)
                continue
            try:
                km = KaplanMeierFitter().fit(
                    exit_[mask], event_observed=event[mask], entry=entry[mask]
                )
                predicted = km.predict(np.array([365.0, 730.0]))
                row["survival_365"] = float(predicted.iloc[0])
                row["survival_730"] = float(predicted.iloc[1])
                row["status"] = "ok"
            except StatError as exc:
                row["reason"] = str(exc)
            rows.append(row)
    return rows


def lag_km_curve_rows(view: EndpointView, endpoint: str) -> list[dict]:
    """The full delayed-entry Kaplan-Meier survival curve, overall (`group='All'`) for each locked
    threshold, at the frozen `IBS_GRID` time resolution (365..730 days, 30-day steps) -- the
    plotting companion to `lag_km_rows`'s two-point 365/730 summary. Reuses the exact grid
    primary-evaluation workflow
    already fixes for Brier/IBS scoring rather than inventing a new time resolution; still
    aggregate-only (survival probabilities, no patient-level rows)."""
    rows: list[dict] = []
    entry = view.data["entry"].to_numpy(dtype=float)
    exit_ = view.data["exit"].to_numpy(dtype=float)
    event = view.data["event"].to_numpy(dtype=int)
    for label, cutoff in SEQUENCING_LAG_THRESHOLDS:
        mask = _threshold_mask(entry, cutoff)
        n = int(mask.sum())
        if n < 2:
            continue
        try:
            km = KaplanMeierFitter().fit(exit_[mask], event_observed=event[mask], entry=entry[mask])
            predicted = km.predict(np.array(IBS_GRID))
        except StatError:
            continue
        for time_days, survival in zip(IBS_GRID, predicted, strict=False):
            rows.append(
                {
                    "endpoint": endpoint,
                    "threshold": label,
                    "n_patients": n,
                    "time_days": float(time_days),
                    "survival": float(survival),
                }
            )
    return rows


def lag_pooled_minimal_cox_rows(
    *,
    endpoint: str,
    centre: str,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk_unconditional: np.ndarray,
    linear_predictor: np.ndarray,
    index_matrix: np.ndarray,
    model_name: str = "minimal_cox",
) -> list[dict]:
    """Pooled sensitivity estimate at each threshold for one centre, reusing the already-frozen
    pooled prediction (no refit) and gated by the imported primary-evaluation reporting-status rule
    via
    `score_patient_subset`.

    The function name is historical: sensitivity-analysis workflow ran this for `minimal_cox` alone.
    The computation is
    model-agnostic — it scores whatever predictions it is handed against the locked threshold
    masks — so `model_name` was added (2026-08-16) to label the rows correctly when
    forest-model-retry workflow reuses
    it for `ltrc_cif`. The default preserves sensitivity-analysis workflow's behaviour exactly, and
    a test asserts it.
    """
    rows: list[dict] = []
    for label, cutoff in SEQUENCING_LAG_THRESHOLDS:
        mask = _threshold_mask(entry, cutoff)
        scored = score_patient_subset(
            entry=entry,
            exit_=exit_,
            event=event,
            risk_unconditional=risk_unconditional,
            linear_predictor=linear_predictor,
            subset_mask=mask,
            index_matrix=index_matrix,
        )
        for metric_key, metric in scored.items():
            rows.append(
                {
                    "endpoint": endpoint,
                    "threshold": label,
                    "centre": centre,
                    "model": model_name,
                    "metric": metric_key,
                    "estimate": metric.raw.value,
                    "ci_lower": metric.ci_lower,
                    "ci_upper": metric.ci_upper,
                    "n_eligible": metric.raw.n_eligible,
                    "n_events": metric.raw.n_events,
                    "reporting_status": metric.status.status,
                    "reason": ",".join(metric.status.reasons),
                }
            )
    return rows


__all__ = ["lag_km_rows", "lag_pooled_minimal_cox_rows", "lag_support_rows"]
