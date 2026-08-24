"""E2: grouped delayed-entry permutation importance on the common prediction-loss scale.

`explainability-spec.md`, "E2 -- grouped permutation importance (the common scale)". The scale is
the increase in the delayed-entry IPCW Brier score at 365 and 730 days and the integrated Brier
score over `[365, 730]`, all **imported** from `crc_folfox.evaluate.prediction_error` --
primary-evaluation workflow's
own Gate-3-validated estimator -- never reimplemented. Permutation touches predictor columns of an
already-transformed design matrix only; `entry`/`exit`/`event` and the outcome data used to score a
permuted prediction are always the *unpermuted* originals, so risk-set membership, the censoring
model, and IPCW weights are recomputed unchanged on every repeat.

For an R-backed candidate, every permutation repeat of one feature group is sent to the R adapter
in a single stacked call (`explainability-spec.md`: "all `R_perm` permuted replicates ... are
stacked into a single batched adapter call") rather than one subprocess per repeat.

**2026-08-12 performance fix.** Scoring used `prediction_error.integrated_brier_score`/
`horizon_brier_point` -- the *unbatched* Brier core, which refits the censoring
`KaplanMeierFitter` from scratch on every one of its 14 `IBS_GRID` points and predicts row-by-row.
Since `entry`/`exit`/`event` never change across permutation repeats or across the baseline score
within one `compute_group_importance` call, that refit was pure waste: a real timing probe against
production data measured ~0.5s per single permutation evaluation, projecting to multi-day runtimes
for the bootstrap-importance artifact at any usable `R_boot`. The module already has a validated,
vectorized equivalent built for exactly this reason during primary-evaluation workflow's own
bootstrap resampling
(`fast_integrated_brier_score`/`fast_horizon_brier`, sharing one KM fit and cross-validated against
the unbatched core in `test_prediction_error.py`) -- this module now fits the censoring KM **once**
per call and reuses it across the baseline and every repeat, computing the identical formula.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from crc_folfox.evaluate.censoring import fit_censoring_km
from crc_folfox.evaluate.contracts import IBS_GRID, EvaluationContractError
from crc_folfox.evaluate.prediction_error import fast_horizon_brier, fast_integrated_brier_score
from crc_folfox.explain.contracts import R_PERM, ROOT_SEED, RUNTIME_R
from crc_folfox.explain.loading import LoadedModel, predict_from_matrix

IDX_365, IDX_730 = 0, len(IBS_GRID) - 1


def permutation_seed(
    *, endpoint: str, centre: str, model_name: str, feature_group: str, repeat: int
) -> int:
    """Deterministic seed for one `(endpoint, centre, model, feature_group, repeat)` permutation,
    following the project's SHA-256 child-seed convention (`model_training.contracts.child_seed`).
    """
    fields = (
        str(ROOT_SEED),
        endpoint,
        centre,
        model_name,
        feature_group,
        str(repeat),
        "permutation_importance",
    )
    return int(hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:8], 16)


def permute_group(
    matrix: pd.DataFrame, group_columns: tuple[str, ...], perm_index: np.ndarray
) -> pd.DataFrame:
    """Row-permute every column of one feature group together, with one shared index -- a one-hot
    or spline block moves as a unit and can never become internally inconsistent."""
    permuted = matrix.copy()
    block = matrix.loc[:, list(group_columns)].to_numpy()[perm_index]
    permuted.loc[:, list(group_columns)] = block
    return permuted


@dataclass(frozen=True)
class BrierTriple:
    ibs: float | None
    brier_365: float | None
    brier_730: float | None


def _fit_km_safe(
    entry: np.ndarray, exit_: np.ndarray, event: np.ndarray
) -> KaplanMeierFitter | None:
    """`None` on a degenerate censoring fit (zero support) -- every score derived from it then
    becomes `None` too, matching the finite-fraction/reporting-status contract exactly as the
    prior per-call slow path did (each of its independent calls failed identically, since they all
    fit the same censoring distribution from the same unpermuted `entry`/`exit`/`event`)."""
    try:
        return fit_censoring_km(entry, exit_, event)
    except EvaluationContractError:
        return None


def _score_risk_grid(
    risk_grid: np.ndarray,
    km: KaplanMeierFitter | None,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
) -> BrierTriple:
    """Scores one risk grid against a **shared, already-fitted** censoring KM -- the 2026-08-12
    performance fix (see module docstring). A degenerate censoring weight at one specific horizon
    (e.g. zero support right at 365 days) fails only that horizon's score, matching the prior
    per-horizon-call isolation."""
    if km is None:
        return BrierTriple(None, None, None)
    try:
        ibs = fast_integrated_brier_score(km, entry, exit_, event, risk_grid)
    except EvaluationContractError:
        ibs = None
    try:
        b365 = fast_horizon_brier(km, entry, exit_, event, risk_grid[:, IDX_365], 365.0)
    except EvaluationContractError:
        b365 = None
    try:
        b730 = fast_horizon_brier(km, entry, exit_, event, risk_grid[:, IDX_730], 730.0)
    except EvaluationContractError:
        b730 = None
    return BrierTriple(ibs, b365, b730)


def _repeat_scores_python(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    km: KaplanMeierFitter | None,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    group_columns: tuple[str, ...],
    *,
    endpoint: str,
    centre: str,
    group_name: str,
    r_perm: int,
) -> list[BrierTriple]:
    n = len(matrix)
    scores = []
    for repeat in range(r_perm):
        seed = permutation_seed(
            endpoint=endpoint,
            centre=centre,
            model_name=loaded.model_name,
            feature_group=group_name,
            repeat=repeat,
        )
        perm_index = np.random.default_rng(seed).permutation(n)
        permuted = permute_group(matrix, group_columns, perm_index)
        prediction = predict_from_matrix(loaded, permuted, entry, IBS_GRID)
        scores.append(_score_risk_grid(prediction.unconditional, km, entry, exit_, event))
    return scores


def _repeat_scores_r_batched(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    km: KaplanMeierFitter | None,
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    group_columns: tuple[str, ...],
    *,
    endpoint: str,
    centre: str,
    group_name: str,
    r_perm: int,
) -> list[BrierTriple]:
    n = len(matrix)
    blocks = []
    for repeat in range(r_perm):
        seed = permutation_seed(
            endpoint=endpoint,
            centre=centre,
            model_name=loaded.model_name,
            feature_group=group_name,
            repeat=repeat,
        )
        perm_index = np.random.default_rng(seed).permutation(n)
        blocks.append(permute_group(matrix, group_columns, perm_index))
    stacked = pd.concat(blocks, ignore_index=True)
    stacked_entry = np.tile(entry, r_perm)
    prediction = predict_from_matrix(loaded, stacked, stacked_entry, IBS_GRID)
    risk = prediction.unconditional.reshape(r_perm, n, len(IBS_GRID))
    return [_score_risk_grid(risk[repeat], km, entry, exit_, event) for repeat in range(r_perm)]


@dataclass(frozen=True)
class GroupImportanceResult:
    feature_group: str
    n_columns: int
    baseline: BrierTriple
    repeat_scores: tuple[BrierTriple, ...]
    finite_fraction_ibs: float
    finite_fraction_365: float
    finite_fraction_730: float

    def importance_ibs(self) -> tuple[float | None, float]:
        return _mean_increase([s.ibs for s in self.repeat_scores], self.baseline.ibs)

    def importance_365(self) -> tuple[float | None, float]:
        return _mean_increase([s.brier_365 for s in self.repeat_scores], self.baseline.brier_365)

    def importance_730(self) -> tuple[float | None, float]:
        return _mean_increase([s.brier_730 for s in self.repeat_scores], self.baseline.brier_730)


def _mean_increase(
    permuted_values: list[float | None], baseline: float | None
) -> tuple[float | None, float]:
    n = len(permuted_values)
    finite = [v for v in permuted_values if v is not None]
    finite_fraction = len(finite) / n if n else 0.0
    if baseline is None or not finite:
        return None, finite_fraction
    increases = [v - baseline for v in finite]
    return float(np.mean(increases)), finite_fraction


def compute_group_importance(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    group_columns: tuple[str, ...],
    *,
    endpoint: str,
    centre: str,
    group_name: str,
    r_perm: int = R_PERM,
    km: KaplanMeierFitter | None = None,
) -> GroupImportanceResult:
    """`importance(g) = mean over r of [Brier(permuted_r) - Brier(observed)]` on the common scale,
    for one feature group. Higher is more important; negative values are real and reported as-is.

    `km` is the censoring Kaplan-Meier already fitted on this scope's `entry`/`exit`/`event` --
    pass it in when the caller already has one (e.g. `compute_all_groups_importance_batched`,
    sharing it across every group) to avoid refitting it redundantly; a standalone caller may omit
    it and one is fitted here from `metadata`.
    """
    entry = metadata["entry"].to_numpy(dtype=float)
    exit_ = metadata["exit"].to_numpy(dtype=float)
    event = metadata["event"].to_numpy(dtype=int)
    if km is None:
        km = _fit_km_safe(entry, exit_, event)
    baseline_prediction = predict_from_matrix(loaded, matrix, entry, IBS_GRID)
    baseline = _score_risk_grid(baseline_prediction.unconditional, km, entry, exit_, event)

    repeat_fn = _repeat_scores_r_batched if loaded.runtime == RUNTIME_R else _repeat_scores_python
    repeat_scores = repeat_fn(
        loaded,
        matrix,
        km,
        entry,
        exit_,
        event,
        group_columns,
        endpoint=endpoint,
        centre=centre,
        group_name=group_name,
        r_perm=r_perm,
    )
    return _build_result(group_name, group_columns, baseline, repeat_scores)


def _build_result(
    group_name: str,
    group_columns: tuple[str, ...],
    baseline: BrierTriple,
    repeat_scores: list[BrierTriple],
) -> GroupImportanceResult:
    def _finite_fraction(key: str) -> float:
        values = [getattr(s, key) for s in repeat_scores]
        return sum(v is not None for v in values) / len(values) if values else 0.0

    return GroupImportanceResult(
        feature_group=group_name,
        n_columns=len(group_columns),
        baseline=baseline,
        repeat_scores=tuple(repeat_scores),
        finite_fraction_ibs=_finite_fraction("ibs"),
        finite_fraction_365=_finite_fraction("brier_365"),
        finite_fraction_730=_finite_fraction("brier_730"),
    )


def compute_all_groups_importance_batched(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    groups: dict[str, tuple[str, ...]],
    *,
    endpoint: str,
    centre: str,
    r_perm: int = R_PERM,
) -> dict[str, GroupImportanceResult]:
    """Every group's importance from one R round trip (R-backed models) or one loop (Python
    models) -- `explainability-spec.md`'s 2026-08-11 bootstrap-importance amendment: batching only
    one group's `r_perm` repeats per call, done naively per group per bootstrap refit, would mean
    up to `n_groups (~15) x B_explain (200)` R subprocess calls per scope. This stacks *every*
    group's *every* repeat's permuted block into a single adapter call instead, so one refit costs
    one R round trip regardless of how many groups it has.

    `groups` here is expected to already be filtered to each group's *present* columns for this
    particular matrix (e.g. a bootstrap refit's own transformed design, which can drop columns a
    canonical group has elsewhere) -- a group with zero columns has no importance to compute here
    and is the caller's responsibility to handle
    (`explain.feature_groups.align_groups_to_present_columns`, `group_active=False` -> importance
    assigned exactly `0.0`, never routed through this function).

    The censoring KM is fitted **once** here and shared across the baseline and every group's every
    repeat -- `entry`/`exit`/`event` never change across permutations or across groups within one
    call, so refitting it per group or per repeat (the pre-2026-08-12 behavior) was pure redundant
    cost, the dominant one at real bootstrap-importance volume (see module docstring).
    """
    entry = metadata["entry"].to_numpy(dtype=float)
    exit_ = metadata["exit"].to_numpy(dtype=float)
    event = metadata["event"].to_numpy(dtype=int)
    km = _fit_km_safe(entry, exit_, event)
    baseline_prediction = predict_from_matrix(loaded, matrix, entry, IBS_GRID)
    baseline = _score_risk_grid(baseline_prediction.unconditional, km, entry, exit_, event)

    if not groups:
        return {}

    if loaded.runtime != RUNTIME_R:
        return {
            group_name: compute_group_importance(
                loaded,
                matrix,
                metadata,
                columns,
                endpoint=endpoint,
                centre=centre,
                group_name=group_name,
                r_perm=r_perm,
                km=km,
            )
            for group_name, columns in groups.items()
        }

    n = len(matrix)
    group_names = list(groups.keys())
    blocks = []
    for group_name in group_names:
        columns = groups[group_name]
        for repeat in range(r_perm):
            seed = permutation_seed(
                endpoint=endpoint,
                centre=centre,
                model_name=loaded.model_name,
                feature_group=group_name,
                repeat=repeat,
            )
            perm_index = np.random.default_rng(seed).permutation(n)
            blocks.append(permute_group(matrix, columns, perm_index))
    stacked = pd.concat(blocks, ignore_index=True)
    stacked_entry = np.tile(entry, len(group_names) * r_perm)
    prediction = predict_from_matrix(loaded, stacked, stacked_entry, IBS_GRID)
    risk = prediction.unconditional.reshape(len(group_names), r_perm, n, len(IBS_GRID))

    results: dict[str, GroupImportanceResult] = {}
    for group_index, group_name in enumerate(group_names):
        repeat_scores = [
            _score_risk_grid(risk[group_index, repeat], km, entry, exit_, event)
            for repeat in range(r_perm)
        ]
        results[group_name] = _build_result(group_name, groups[group_name], baseline, repeat_scores)
    return results


__all__ = [
    "IDX_365",
    "IDX_730",
    "BrierTriple",
    "GroupImportanceResult",
    "compute_all_groups_importance_batched",
    "compute_group_importance",
    "permutation_seed",
    "permute_group",
]
