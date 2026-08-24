"""Delayed-entry / independent-truncation assumption checks.

The primary diagnostic is a left-truncation-aware **conditional Kendall's tau** testing
quasi-independence of entry (truncation) and event time (Martin & Betensky, 2005,
doi:10.1198/016214504000001538) -- not ordinary Kendall correlation, which ignores both the
truncation-induced sampling restriction and right censoring.

This implementation operationalizes the statistic by reusing the codebase's own already-tested
truncation-aware pair-comparability rule (`crc_folfox.evaluate.discrimination.
truncation_aware_concordance`): a pair is comparable only when both entries are `<= min(exit_i,
exit_j)` and the usual censoring-comparability rule holds (the earlier-exiting record is an observed
event). Passing `entry` itself as the "risk score" gives the concordance probability between entry
time and survival time among truncation/censoring-comparable pairs; Kendall's tau is the standard
`tau = 2 * C - 1` transform of that concordance probability (no ties are double-counted because the
comparable-pairs rule already resolves them, matching `truncation_aware_concordance`'s own tie
convention). This is the same estimator every other evaluation or sensitivity-analysis metric in
this codebase is built on,
reused rather than reimplemented, and is conceptually identical to what Martin-Betensky's own
risk-set-based comparability criterion measures: association between truncation time and failure
time among pairs where such a comparison is valid.

Uncertainty (CI and p-value) comes from patient-level bootstrap, consistent with every other
evaluation or sensitivity-analysis uncertainty estimate in this project, rather than the published
asymptotic variance
formula -- documented here as an implementation decision (see `implement.md`, "Truncation
diagnostic"), not a claim of bit-for-bit reproduction of Martin & Betensky's own test statistic.

This is diagnostic evidence only. Failing to reject quasi-independence does **not** prove
independent truncation -- `TRUNCATION_CONCLUSION_REJECTED`/`_NOT_REJECTED` are the two, and only
two, permitted conclusion sentences, and both explicitly say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.exceptions import StatError

from crc_folfox.evaluate.contracts import CENTRES, MIN_BOOTSTRAP_FINITE_FRACTION
from crc_folfox.evaluate.discrimination import truncation_aware_concordance
from crc_folfox.preprocessing.contracts import EndpointView
from crc_folfox.preprocessing.splits import derive_seed
from crc_folfox.sensitivity.contracts import (
    TRUNCATION_CONCLUSION_NOT_REJECTED,
    TRUNCATION_CONCLUSION_REJECTED,
)

# Below this many comparable pairs, a stratum's CI is treated as too unstable to call "estimable" --
# analogous in spirit to `evaluate.contracts.MIN_ELIGIBLE_EVENTS` (20), specific to this diagnostic.
MIN_COMPARABLE_PAIRS = 20
TAU_BOOTSTRAP_RESAMPLES = 2000
ALPHA_TWO_SIDED = 0.05


@dataclass(frozen=True)
class ConditionalKendallTau:
    stratum: str
    tau: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    n_comparable_pairs: int
    n_patients: int
    n_events: int
    reporting_status: str
    reason: str


def _tau_and_pairs(
    entry: np.ndarray, exit_: np.ndarray, event: np.ndarray
) -> tuple[float | None, int]:
    # `truncation_aware_concordance`'s convention rewards a *higher* "risk" score pairing with an
    # *earlier* exit (the proportional-hazards convention). Passing `-entry` as that score means a
    # *smaller* entry time is treated as "higher risk" of early failure, so a genuine positive
    # association between entry and event time (later truncation, later failure) yields a positive
    # tau -- the intuitive Kendall's-tau sign convention, not the risk-score one.
    result = truncation_aware_concordance(entry, exit_, event, -np.asarray(entry, dtype=float))
    if result.value is None:
        return None, 0
    return 2.0 * result.value - 1.0, result.comparable_pairs


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _summarize_bootstrap(
    tau: float | None, boot_values: np.ndarray
) -> tuple[float | None, float | None, float | None, float]:
    finite = boot_values[np.isfinite(boot_values)]
    n_resamples = len(boot_values)
    finite_fraction = float(len(finite)) / n_resamples if n_resamples else 0.0
    if tau is None or len(finite) < 2:
        return None, None, None, finite_fraction
    ci_lower, ci_upper = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
    se = float(np.std(finite, ddof=1))
    p_value = None
    if se > 0:
        p_value = float(2.0 * (1.0 - _standard_normal_cdf(abs(tau) / se)))
    return ci_lower, ci_upper, p_value, finite_fraction


def _reporting_status(
    tau: float | None, ci_lower: float | None, n_comparable_pairs: int, finite_fraction: float
) -> tuple[str, str]:
    if tau is None:
        return "not_computable", "no_comparable_pairs"
    reasons = []
    if n_comparable_pairs < MIN_COMPARABLE_PAIRS:
        reasons.append("insufficient_comparable_pairs")
    if finite_fraction < MIN_BOOTSTRAP_FINITE_FRACTION:
        reasons.append("insufficient_bootstrap_success")
    if ci_lower is None:
        reasons.append("no_bootstrap_interval")
    if reasons:
        return "estimable_imprecise", ",".join(reasons)
    return "estimable", ""


def institution_conditional_kendall_tau(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    *,
    stratum: str,
    seed: int,
    resamples: int = TAU_BOOTSTRAP_RESAMPLES,
) -> ConditionalKendallTau:
    """One institution's conditional Kendall's tau, with patient-level bootstrap uncertainty."""
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(entry)
    tau, n_pairs = _tau_and_pairs(entry, exit_, event)
    n_events = int(event.sum())
    if tau is None:
        return ConditionalKendallTau(
            stratum, None, None, None, None, 0, n, n_events, "not_computable", "no_comparable_pairs"
        )

    rng = np.random.default_rng(seed)
    boot_values = np.full(resamples, np.nan)
    for row in range(resamples):
        idx = rng.integers(0, n, size=n)
        row_tau, _ = _tau_and_pairs(entry[idx], exit_[idx], event[idx])
        if row_tau is not None:
            boot_values[row] = row_tau
    ci_lower, ci_upper, p_value, finite_fraction = _summarize_bootstrap(tau, boot_values)
    status, reason = _reporting_status(tau, ci_lower, n_pairs, finite_fraction)
    return ConditionalKendallTau(
        stratum, tau, ci_lower, ci_upper, p_value, n_pairs, n, n_events, status, reason
    )


def pooled_conditional_kendall_tau(
    per_institution: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    seed: int,
    resamples: int = TAU_BOOTSTRAP_RESAMPLES,
) -> ConditionalKendallTau:
    """The centre-stratified pooled result: a pair-count-weighted average of the per-institution
    taus, with a *stratified* bootstrap (each institution resampled independently, from its own
    patients only) -- never an unstratified pool of all patients together."""
    point_taus: dict[str, float] = {}
    point_pairs: dict[str, int] = {}
    for institution, (entry, exit_, event) in per_institution.items():
        tau, pairs = _tau_and_pairs(
            np.asarray(entry, dtype=float),
            np.asarray(exit_, dtype=float),
            np.asarray(event, dtype=int),
        )
        if tau is not None:
            point_taus[institution] = tau
            point_pairs[institution] = pairs

    n_patients = sum(len(arrays[0]) for arrays in per_institution.values())
    n_events = sum(int(np.asarray(arrays[2]).sum()) for arrays in per_institution.values())
    total_pairs = sum(point_pairs.values())
    if not point_taus or total_pairs == 0:
        return ConditionalKendallTau(
            "pooled",
            None,
            None,
            None,
            None,
            0,
            n_patients,
            n_events,
            "not_computable",
            "no_comparable_pairs",
        )
    pooled_tau = sum(point_taus[k] * point_pairs[k] for k in point_taus) / total_pairs

    rng = np.random.default_rng(seed)
    boot_values = np.full(resamples, np.nan)
    for row in range(resamples):
        row_taus: dict[str, float] = {}
        row_pairs: dict[str, int] = {}
        for institution, (entry, exit_, event) in per_institution.items():
            entry = np.asarray(entry, dtype=float)
            exit_ = np.asarray(exit_, dtype=float)
            event = np.asarray(event, dtype=int)
            idx = rng.integers(0, len(entry), size=len(entry))
            row_tau, row_pair_count = _tau_and_pairs(entry[idx], exit_[idx], event[idx])
            if row_tau is not None:
                row_taus[institution] = row_tau
                row_pairs[institution] = row_pair_count
        row_total_pairs = sum(row_pairs.values())
        if row_total_pairs > 0:
            boot_values[row] = sum(row_taus[k] * row_pairs[k] for k in row_taus) / row_total_pairs

    ci_lower, ci_upper, p_value, finite_fraction = _summarize_bootstrap(pooled_tau, boot_values)
    status, reason = _reporting_status(pooled_tau, ci_lower, total_pairs, finite_fraction)
    return ConditionalKendallTau(
        "pooled",
        pooled_tau,
        ci_lower,
        ci_upper,
        p_value,
        total_pairs,
        n_patients,
        n_events,
        status,
        reason,
    )


def truncation_kendall_tau_rows(
    view: EndpointView, endpoint: str, *, resamples: int = TAU_BOOTSTRAP_RESAMPLES
) -> tuple[list[dict], str]:
    """Per-institution and centre-stratified-pooled conditional Kendall's tau rows, plus the
    required descriptive-only conclusion sentence chosen from the pooled result."""
    per_institution: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    results: list[ConditionalKendallTau] = []
    for institution in CENTRES:
        data = view.data.loc[view.data["institution"].eq(institution)]
        entry = data["entry"].to_numpy(dtype=float)
        exit_ = data["exit"].to_numpy(dtype=float)
        event = data["event"].to_numpy(dtype=int)
        per_institution[institution] = (entry, exit_, event)
        seed = derive_seed(
            endpoint=endpoint,
            design="",
            held_out_institution=institution,
            purpose="truncation_kendall_tau",
        )
        results.append(
            institution_conditional_kendall_tau(
                entry, exit_, event, stratum=institution, seed=seed, resamples=resamples
            )
        )

    pooled_seed = derive_seed(
        endpoint=endpoint,
        design="",
        held_out_institution=None,
        purpose="truncation_kendall_tau_pooled",
    )
    pooled = pooled_conditional_kendall_tau(per_institution, seed=pooled_seed, resamples=resamples)
    results.append(pooled)

    rows = [
        {
            "endpoint": endpoint,
            "stratum": result.stratum,
            "tau": result.tau,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "p_value": result.p_value,
            "n_comparable_pairs": result.n_comparable_pairs,
            "n_patients": result.n_patients,
            "n_events": result.n_events,
            "reporting_status": result.reporting_status,
            "reason": result.reason,
        }
        for result in results
    ]

    if pooled.tau is None or pooled.p_value is None:
        conclusion = TRUNCATION_CONCLUSION_NOT_REJECTED
    elif pooled.p_value < ALPHA_TWO_SIDED:
        conclusion = TRUNCATION_CONCLUSION_REJECTED
    else:
        conclusion = TRUNCATION_CONCLUSION_NOT_REJECTED
    return rows, conclusion


def entry_time_distribution_rows(view: EndpointView, endpoint: str) -> list[dict]:
    """Aggregate-only entry-time distribution, overall and by centre (the accompanying plot's
    source data)."""
    rows = []
    for group in (*CENTRES, "All"):
        data = view.data if group == "All" else view.data.loc[view.data["institution"].eq(group)]
        entry = data["entry"].to_numpy(dtype=float)
        quantile_labels = (0, 10, 25, 50, 75, 90, 100)
        quantile_values = np.percentile(entry, quantile_labels) if len(entry) else None
        rows.append(
            {
                "endpoint": endpoint,
                "group": group,
                "n_patients": len(entry),
                **{
                    f"entry_p{label}": (float(value) if quantile_values is not None else "")
                    for label, value in zip(
                        quantile_labels,
                        quantile_values if quantile_values is not None else [],
                        strict=False,
                    )
                },
            }
        )
    return rows


def km_by_entry_quartile_rows(
    view: EndpointView, endpoint: str, *, n_quartiles: int = 4
) -> list[dict]:
    """Delayed-entry Kaplan-Meier survival at 365/730 days by entry-time quartile -- the
    accompanying graphical check for the conditional-Kendall's-tau diagnostic."""
    data = view.data.sort_values("entry", kind="stable")
    entry = data["entry"].to_numpy(dtype=float)
    exit_ = data["exit"].to_numpy(dtype=float)
    event = data["event"].to_numpy(dtype=int)
    order = np.arange(len(entry))
    rows: list[dict] = []
    for quartile_index, member_idx in enumerate(np.array_split(order, n_quartiles)):
        if not len(member_idx):
            continue
        group_entry, group_exit, group_event = (
            entry[member_idx],
            exit_[member_idx],
            event[member_idx],
        )
        row = {
            "endpoint": endpoint,
            "entry_quartile": quartile_index,
            "n_patients": len(member_idx),
            "n_events": int(group_event.sum()),
            "entry_min": float(group_entry.min()),
            "entry_max": float(group_entry.max()),
            "survival_365": "",
            "survival_730": "",
            "status": "not_computable",
        }
        try:
            km = KaplanMeierFitter().fit(group_exit, event_observed=group_event, entry=group_entry)
            predicted = km.predict(np.array([365.0, 730.0]))
            row["survival_365"] = float(predicted.iloc[0])
            row["survival_730"] = float(predicted.iloc[1])
            row["status"] = "ok"
        except StatError:
            pass
        rows.append(row)
    return rows


def entry_at_or_after_exit_summary(frame, endpoint_exit_column: str = "os_time") -> dict:
    """Descriptive-only count of patients whose sequencing entry falls at or after their exit time
    (already excluded from the endpoint-eligible view), plus the median entry ("first-report") lag
    across the full data-cleaning workflow population -- reproduced from the raw cleaned analysis
    frame, matching
    `tech-stack.md`'s "Delayed-entry feasibility audit" figures (34 excluded; 215-day median lag)
    rather than restated as unverified magic numbers."""
    entry = frame["entry"].to_numpy(dtype=float)
    exit_ = frame[endpoint_exit_column].to_numpy(dtype=float)
    n_excluded = int((entry >= exit_).sum())
    return {
        "n_total_patients": len(frame),
        "n_entry_at_or_after_exit": n_excluded,
        "median_entry_lag_days": float(np.median(entry)),
    }


__all__ = [
    "ConditionalKendallTau",
    "entry_at_or_after_exit_summary",
    "entry_time_distribution_rows",
    "institution_conditional_kendall_tau",
    "km_by_entry_quartile_rows",
    "pooled_conditional_kendall_tau",
    "truncation_kendall_tau_rows",
]
