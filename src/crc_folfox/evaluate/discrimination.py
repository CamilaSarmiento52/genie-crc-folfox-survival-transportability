"""Truncation-aware concordance and time-dependent (cumulative/dynamic) AUC.

Both estimators are project-owned because no named library implementation carries entry time (see
`evaluation-spec.md`, "Explicitly ineligible software"): `lifelines.utils.concordance_index` has no
truncation argument, and `sksurv.metrics.concordance_index_censored/_ipcw/cumulative_dynamic_auc`
take a `Surv.from_arrays(event, time)` object with no entry slot at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lifelines import KaplanMeierFitter

from crc_folfox.evaluate.censoring import horizon_weights
from crc_folfox.evaluate.contracts import EvaluationContractError


@dataclass(frozen=True)
class ConcordanceResult:
    value: float | None
    comparable_pairs: int


def truncation_aware_concordance(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    pair_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> ConcordanceResult:
    """Harrell-style concordance restricted to pairs simultaneously at risk.

    A pair `(i, j)` with the earlier exit time `t_min = min(exit_i, exit_j)` is comparable only
    when `entry_i <= t_min` and `entry_j <= t_min` (both were under observation at `t_min`) and the
    usual censoring-comparability rule holds: the record at the earlier exit time must be an
    observed event (or, on an exact time tie, at least one of the two must be an observed event).
    Ties in predicted risk contribute one half.

    On an exact time tie with both records observed events, direction is undefined by the data
    (there is no true order to recover, only a tied event time); such pairs are comparable but
    contribute a fixed 0.5 regardless of predicted risk, documented here as this estimator's own
    tie-time convention -- it is not a library import to be bit-matched.

    `pair_indices` lets a caller precompute `np.triu_indices(n, k=1)` once and reuse it across many
    calls with the same `n` (e.g. every one of a bootstrap's 2000 with-replacement resamples), since
    the index pairs depend only on `n`, not on the data.
    """
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = len(exit_)
    if not (len(entry) == n == len(event) == len(risk)):
        raise EvaluationContractError("Concordance inputs are not aligned")
    if n < 2:
        return ConcordanceResult(None, 0)

    i_idx, j_idx = pair_indices if pair_indices is not None else np.triu_indices(n, k=1)
    ei, ej = entry[i_idx], entry[j_idx]
    ti, tj = exit_[i_idx], exit_[j_idx]
    eventi, eventj = event[i_idx], event[j_idx]
    ri, rj = risk[i_idx], risk[j_idx]

    t_min = np.minimum(ti, tj)
    truncation_ok = (ei <= t_min) & (ej <= t_min)

    direction = np.zeros(len(i_idx), dtype=int)
    comparable = np.zeros(len(i_idx), dtype=bool)

    i_earlier = (ti < tj) & (eventi == 1) & truncation_ok
    direction[i_earlier] = 1
    comparable[i_earlier] = True

    j_earlier = (tj < ti) & (eventj == 1) & truncation_ok
    direction[j_earlier] = -1
    comparable[j_earlier] = True

    tied_time = ti == tj
    tie_both_events = tied_time & (eventi == 1) & (eventj == 1) & truncation_ok
    comparable[tie_both_events] = True  # direction stays 0: undefined order, fixed 0.5 credit

    tie_one_event = tied_time & truncation_ok & ((eventi + eventj) == 1)
    comparable[tie_one_event] = True
    direction[tie_one_event & (eventi == 1)] = 1
    direction[tie_one_event & (eventj == 1)] = -1

    # A pair with a non-finite (e.g. NaN) predictor on either side has no defined winner/loser and
    # must not fall through to the tied-risk 0.5 branch below -- e.g. a candidate with no
    # closed-form linear predictor reports every risk as NaN, and every pair would otherwise
    # silently
    # score as a tie, producing a spurious concordance of exactly 0.5 instead of not_computable.
    comparable &= np.isfinite(ri) & np.isfinite(rj)

    n_comparable = int(comparable.sum())
    if n_comparable == 0:
        return ConcordanceResult(None, 0)

    winner_risk = np.where(direction == 1, ri, rj)
    loser_risk = np.where(direction == 1, rj, ri)
    score = np.where(winner_risk > loser_risk, 1.0, np.where(winner_risk < loser_risk, 0.0, 0.5))
    score = np.where(direction == 0, 0.5, score)
    value = float(score[comparable].mean())
    return ConcordanceResult(value, n_comparable)


@dataclass(frozen=True)
class AucResult:
    value: float | None
    n_cases: int
    n_controls: int


def time_dependent_auc(
    km: KaplanMeierFitter,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
) -> AucResult:
    """Uno-style IPCW cumulative/dynamic AUC at `horizon`.

    Cases (`entry <= horizon`, observed event by `horizon`) are weighted by the delayed-entry
    censoring KM evaluated just before their exit time; controls (`entry <= horizon`, known
    event-free at `horizon`, i.e. `exit > horizon`) are unweighted, since their control status is
    directly observed and needs no censoring correction. This is the same asymmetric-weighting
    convention as Uno et al. (2007), extended here to a left-truncation-aware `eligible` gate.
    """
    entry = np.asarray(entry, dtype=float)
    exit_ = np.asarray(exit_, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    weights = horizon_weights(km, entry, exit_, event, horizon)
    case_idx = np.flatnonzero(weights.occurred)
    control_idx = np.flatnonzero(weights.survived)
    if not len(case_idx) or not len(control_idx):
        return AucResult(None, len(case_idx), len(control_idx))
    case_weight = weights.weight[case_idx]
    ri = risk[case_idx][:, np.newaxis]
    rj = risk[control_idx][np.newaxis, :]
    indicator = np.where(ri > rj, 1.0, np.where(ri < rj, 0.0, 0.5))
    numerator = float(np.sum(case_weight[:, np.newaxis] * indicator))
    denominator = float(case_weight.sum()) * len(control_idx)
    if denominator <= 0 or not np.isfinite(denominator):
        return AucResult(None, len(case_idx), len(control_idx))
    return AucResult(numerator / denominator, len(case_idx), len(control_idx))


__all__ = [
    "AucResult",
    "ConcordanceResult",
    "time_dependent_auc",
    "truncation_aware_concordance",
]
