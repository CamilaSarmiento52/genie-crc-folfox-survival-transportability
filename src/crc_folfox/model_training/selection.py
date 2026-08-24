"""Nested pooled model-selection splits and the elastic-net Cox tuning loop.

Reusing an outer-development preprocessing state inside model selection is leakage and is
prohibited by the frozen spec. Every function here that touches inner-training rows refits
`FittedPreprocessor` fresh from those rows alone -- never from a scope-level preprocessor fit
beforehand.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from numpy.linalg import LinAlgError
from sklearn.model_selection import StratifiedKFold

from crc_folfox.features import FeatureManifest
from crc_folfox.model_training.contracts import ELASTIC_NET_GRID, ModelTrainingContractError
from crc_folfox.model_training.cox import FittedCoxModel, fit_cox, predict_survival
from crc_folfox.model_training.tuning import delayed_entry_ipcw_brier
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition
from crc_folfox.preprocessing.splits import ROOT_SEED, build_inner_splits
from crc_folfox.preprocessing.transform import FittedPreprocessor

# the correction workstream correction (correction-spec.md, "Expected numerical exception set").
# A failure of exactly one of these types, raised while fitting one `(penalizer, l1_ratio)`
# candidate in one inner fold, disqualifies that combination only -- it must never abort the whole
# tuning scope. `scipy.linalg.LinAlgError` is the identical class object as `numpy.linalg
# .LinAlgError` on this platform, so it is not listed separately. `ConvergenceWarning` is included
# for the case where it is escalated to an exception by the ambient warning policy; under this
# project's current policy (`warnings.simplefilter("always")` in the caller) it is not escalated,
# so it is inert here but still recorded for the set to be complete and explicit. Any exception
# outside this tuple is a genuine, unhandled pipeline failure and is deliberately left to propagate
# out of `tune_elastic_net_cox` uncaught -- a bare `except Exception` here would convert a real
# defect into a silent candidate disqualification.
EXPECTED_TUNING_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ModelTrainingContractError,
    ConvergenceError,
    ConvergenceWarning,
    LinAlgError,
)


def nested_pooled_seed(
    *,
    endpoint: str,
    parent_repeat: int,
    parent_fold: int,
    inner_repeat: int = 0,
    root_seed: int = ROOT_SEED,
) -> int:
    """Derive the one seed that drives an entire pooled parent's nested five-fold partition.

    `inner_fold` is intentionally absent from the hashed tuple (see model-training-spec.md's
    2026-08-07 amendment): a single stratified split call must produce all five folds together to
    guarantee they are disjoint and cover the parent development cohort exactly once, so there is
    no per-fold seed to derive. All five resulting `SplitDefinition` records share this one value.
    """
    fields = (
        str(root_seed),
        endpoint,
        "pooled_inner",
        str(parent_repeat),
        str(parent_fold),
        str(inner_repeat),
        "model_selection",
    )
    return int(hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:8], 16)


def build_nested_pooled_splits(
    view: EndpointView, parent_split: SplitDefinition, *, folds: int = 5
) -> tuple[SplitDefinition, ...]:
    """Create the parent pooled scope's five-fold model-selection partition.

    This nested split is model-selection state: it is derived only from the parent's own
    development frame, is never reused as a primary-evaluation workflow assessment set, and is not
    persisted at
    patient level. The preprocessing workflow's `_build_repeated_splits` is private and
    preprocessing is frozen, so this
    reimplements the same stratified-split shape as a model-training-owned function rather than
    reaching
    into or modifying preprocessing workflow internals.
    """
    development, _ = parent_split.frames(view)
    ordered = development.sort_values("patient_key", kind="stable").reset_index(drop=True)
    strata = ordered["institution"].astype(str) + "|event=" + ordered["event"].astype(str)
    counts = strata.value_counts()
    if counts.empty or int(counts.min()) < folds:
        raise ModelTrainingContractError(
            "Institution-by-event stratum has fewer rows than the required nested folds"
        )
    if parent_split.repeat is None or parent_split.fold is None:
        raise ModelTrainingContractError("Pooled parent split must carry a repeat and fold")

    seed = nested_pooled_seed(
        endpoint=view.endpoint, parent_repeat=parent_split.repeat, parent_fold=parent_split.fold
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for inner_fold, (train_index, assessment_index) in enumerate(splitter.split(ordered, strata)):
        result.append(
            SplitDefinition(
                split_id=(
                    f"pooled_inner|parent_repeat={parent_split.repeat}"
                    f"|parent_fold={parent_split.fold}|inner_fold={inner_fold}"
                ),
                endpoint=view.endpoint,
                design="pooled_inner",
                development_keys=tuple(ordered.iloc[train_index]["patient_key"].astype(str)),
                assessment_keys=tuple(ordered.iloc[assessment_index]["patient_key"].astype(str)),
                held_out_institution=None,
                repeat=0,
                fold=inner_fold,
                seed=seed,
            )
        )
    _validate_nested_coverage(ordered, result, folds=folds)
    return tuple(result)


def _validate_nested_coverage(
    frame: pd.DataFrame, splits: list[SplitDefinition], *, folds: int
) -> None:
    expected = set(frame["patient_key"].astype(str))
    if len(splits) != folds:
        raise ModelTrainingContractError("Nested pooled split count is incorrect")
    seen: list[str] = []
    for split in splits:
        development = set(split.development_keys)
        assessment = set(split.assessment_keys)
        if development & assessment or development | assessment != expected:
            raise ModelTrainingContractError("Nested pooled split coverage or disjointness failed")
        seen.extend(split.assessment_keys)
    if len(seen) != len(expected) or set(seen) != expected:
        raise ModelTrainingContractError("Nested pooled assessment coverage failed")


def select_elastic_net(scores: Mapping[tuple[float, float], float]) -> tuple[float, float]:
    """Pick the smallest-mean-score elastic-net candidate; break `1e-12` ties toward the larger
    `penalizer`, then the larger `l1_ratio` (the more regularized solution)."""
    if not scores:
        raise ModelTrainingContractError("No eligible elastic-net candidate")
    best = min(scores.values())
    tied = [candidate for candidate, score in scores.items() if abs(score - best) <= 1e-12]
    return max(tied)


TUNING_STAGE_CLEAN = "inner_tuning_clean"
TUNING_STAGE_PARTIAL = "inner_tuning_partial"
TUNING_STAGE_NO_ELIGIBLE = "no_eligible_candidate"
TUNING_STAGE_FINAL_REFIT_FAILED = "final_refit_failed"


@dataclass(frozen=True)
class TuningDiagnostics:
    """Aggregate-safe failure diagnostics for one parent scope's elastic-net tuning attempt.

    Every field is a count, a class name, a frozen grid value, or a fold index -- never a patient
    value, key, time, prediction, or coefficient. See correction-spec.md, "Failure diagnostics".
    """

    tuning_stage: str
    candidates_attempted: int
    candidates_failed: int
    candidates_nonfinite: int
    candidates_complete: int
    failure_exception_classes: tuple[str, ...]
    failure_penalizers: tuple[float, ...]
    failure_l1_ratios: tuple[float, ...]
    failure_inner_folds: tuple[int, ...]
    transformed_column_count: int
    zero_variance_column_count: int
    design_rank_deficit: int
    final_refit_failed: bool


class ElasticNetTuningError(ModelTrainingContractError):
    """Raised when a parent scope has no eligible elastic-net candidate, or its selected
    configuration fails its final development-scope refit. Carries `diagnostics` so the caller can
    populate `tuning_summary.csv` even on a scope-level failure."""

    def __init__(self, message: str, diagnostics: TuningDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def design_diagnostics(matrix: pd.DataFrame) -> tuple[int, int, int]:
    """`(transformed_column_count, zero_variance_column_count, design_rank_deficit)` for one
    transformed training matrix. An aggregate-safe diagnostic only -- no patient-level value is
    read or retained."""
    values = matrix.to_numpy(dtype=float)
    column_count = values.shape[1]
    if column_count == 0:
        return 0, 0, 0
    zero_variance = int(np.sum(np.nanstd(values, axis=0) <= 1e-12))
    rank = int(np.linalg.matrix_rank(values)) if values.size else 0
    return column_count, zero_variance, max(column_count - rank, 0)


@dataclass(frozen=True)
class ElasticNetSelection:
    penalizer: float
    l1_ratio: float
    mean_scores: Mapping[tuple[float, float], float]
    model: FittedCoxModel
    preprocessor: FittedPreprocessor
    inner_fold_count: int
    candidate_success_fold_counts: Mapping[tuple[float, float], int]
    diagnostics: TuningDiagnostics

    @property
    def tie_break_used(self) -> bool:
        best = min(self.mean_scores.values())
        return sum(abs(score - best) <= 1e-12 for score in self.mean_scores.values()) > 1


def tune_elastic_net_cox(
    view: EndpointView,
    parent_split: SplitDefinition,
    *,
    manifest: FeatureManifest | None = None,
) -> ElasticNetSelection:
    """Tune and refit `elastic_net_cox` for one development scope.

    `FittedPreprocessor` is refit exactly once per inner fold and reused across all 12 grid
    candidates within that fold -- `penalizer`/`l1_ratio` never affect imputation, encoding, or
    standardization, so a per-candidate refit would be wasted work without changing correctness.

    A candidate's fit failing with one of `EXPECTED_TUNING_EXCEPTIONS`, or returning a nonfinite
    score, disqualifies only that `(penalizer, l1_ratio, inner fold)` combination -- the remaining
    eleven grid candidates and the remaining folds still run (correction-spec.md, "The corrected
    failure policy"). An exception outside that set propagates uncaught, exactly as before.
    """
    if parent_split.design == "iecv_outer":
        inner_splits = build_inner_splits(view, parent_split)
    elif parent_split.design == "pooled":
        inner_splits = build_nested_pooled_splits(view, parent_split)
    else:
        raise ModelTrainingContractError(
            "Elastic-net tuning requires an IECV outer or pooled parent scope"
        )

    per_candidate_fold_scores: dict[tuple[float, float], list[float]] = {
        candidate: [] for candidate in ELASTIC_NET_GRID
    }
    failed_exception_classes: set[str] = set()
    failed_penalizers: set[float] = set()
    failed_l1_ratios: set[float] = set()
    failed_inner_folds: set[int] = set()
    candidates_attempted = 0
    candidates_failed = 0
    candidates_nonfinite = 0
    first_fold_design: tuple[int, int, int] | None = None

    for inner_fold, inner_split in enumerate(inner_splits):
        inner_train_frame, inner_val_frame = inner_split.frames(view)
        preprocessor = FittedPreprocessor.fit(
            inner_train_frame,
            endpoint=view.endpoint,
            feature_set="core",
            fit_scope=inner_split.split_id,
            manifest=manifest,
        )
        train_result = preprocessor.transform(inner_train_frame)
        val_result = preprocessor.transform(inner_val_frame)
        if first_fold_design is None:
            first_fold_design = design_diagnostics(train_result.matrix)
        for penalizer, l1_ratio in ELASTIC_NET_GRID:
            candidates_attempted += 1
            try:
                fitter = fit_cox(
                    train_result.matrix,
                    train_result.metadata,
                    penalizer=penalizer,
                    l1_ratio=l1_ratio,
                )
                survival = predict_survival(fitter, val_result.matrix)
                score = delayed_entry_ipcw_brier(
                    train_result.metadata, val_result.metadata, survival
                )
            except EXPECTED_TUNING_EXCEPTIONS as exc:
                candidates_failed += 1
                failed_exception_classes.add(type(exc).__name__)
                failed_penalizers.add(penalizer)
                failed_l1_ratios.add(l1_ratio)
                failed_inner_folds.add(inner_fold)
                continue
            if not math.isfinite(score):
                candidates_nonfinite += 1
                failed_penalizers.add(penalizer)
                failed_l1_ratios.add(l1_ratio)
                failed_inner_folds.add(inner_fold)
                continue
            per_candidate_fold_scores[(penalizer, l1_ratio)].append(score)

    mean_scores = {
        candidate: sum(fold_scores) / len(fold_scores)
        for candidate, fold_scores in per_candidate_fold_scores.items()
        if len(fold_scores) == len(inner_splits)
    }
    candidates_complete = len(mean_scores)
    column_count, zero_variance, rank_deficit = first_fold_design or (0, 0, 0)

    def _diagnostics(stage: str, *, final_refit_failed: bool = False) -> TuningDiagnostics:
        return TuningDiagnostics(
            tuning_stage=stage,
            candidates_attempted=candidates_attempted,
            candidates_failed=candidates_failed,
            candidates_nonfinite=candidates_nonfinite,
            candidates_complete=candidates_complete,
            failure_exception_classes=tuple(sorted(failed_exception_classes)),
            failure_penalizers=tuple(sorted(failed_penalizers)),
            failure_l1_ratios=tuple(sorted(failed_l1_ratios)),
            failure_inner_folds=tuple(sorted(failed_inner_folds)),
            transformed_column_count=column_count,
            zero_variance_column_count=zero_variance,
            design_rank_deficit=rank_deficit,
            final_refit_failed=final_refit_failed,
        )

    if not mean_scores:
        raise ElasticNetTuningError(
            "No eligible elastic-net candidate for this parent scope",
            _diagnostics(TUNING_STAGE_NO_ELIGIBLE),
        )

    selected_penalizer, selected_l1_ratio = select_elastic_net(mean_scores)

    development_frame, _ = parent_split.frames(view)
    final_preprocessor = FittedPreprocessor.fit(
        development_frame,
        endpoint=view.endpoint,
        feature_set="core",
        fit_scope=parent_split.split_id,
        manifest=manifest,
    )
    final_result = final_preprocessor.transform(development_frame)
    try:
        final_fitter = fit_cox(
            final_result.matrix,
            final_result.metadata,
            penalizer=selected_penalizer,
            l1_ratio=selected_l1_ratio,
        )
    except EXPECTED_TUNING_EXCEPTIONS as exc:
        failed_exception_classes.add(type(exc).__name__)
        raise ElasticNetTuningError(
            "Selected elastic-net configuration failed its final development-scope refit",
            _diagnostics(TUNING_STAGE_FINAL_REFIT_FAILED, final_refit_failed=True),
        ) from exc

    stage = (
        TUNING_STAGE_CLEAN
        if candidates_failed == 0 and candidates_nonfinite == 0
        else TUNING_STAGE_PARTIAL
    )
    return ElasticNetSelection(
        selected_penalizer,
        selected_l1_ratio,
        mean_scores,
        FittedCoxModel(final_fitter),
        final_preprocessor,
        len(inner_splits),
        {candidate: len(scores) for candidate, scores in per_candidate_fold_scores.items()},
        _diagnostics(stage),
    )
