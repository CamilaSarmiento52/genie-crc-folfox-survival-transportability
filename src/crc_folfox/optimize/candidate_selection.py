"""The exploratory-optimization workstream tuning/selection for B4 `penalized_nonlinear_cox`, B5
`reduced_core_penalized_cox`, and B6 `flexsurv_rp_spline` (optimization-spec.md).

Each mirrors `crc_folfox.model_training.selection.tune_elastic_net_cox`'s shape: nested inner
splits, the same delayed-entry IPCW Brier tuning objective, the corrected per-candidate failure
isolation from Workstream A, and a deterministic tie-break favouring stronger regularization (B4,
B5) or fewer knots (B6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from crc_folfox.features import FeatureManifest
from crc_folfox.model_training.contracts import HORIZONS, child_seed
from crc_folfox.model_training.cox import predict_survival
from crc_folfox.model_training.selection import (
    EXPECTED_TUNING_EXCEPTIONS,
    TUNING_STAGE_CLEAN,
    TUNING_STAGE_FINAL_REFIT_FAILED,
    TUNING_STAGE_NO_ELIGIBLE,
    TUNING_STAGE_PARTIAL,
    TuningDiagnostics,
    build_nested_pooled_splits,
)
from crc_folfox.model_training.tuning import delayed_entry_ipcw_brier
from crc_folfox.optimize.candidates import (
    PENALIZER_GRID,
    fit_penalized_nonlinear_cox,
    fit_reduced_core_cox,
    fit_reduced_core_preprocessor,
    transform_for_penalized_nonlinear_cox,
)
from crc_folfox.optimize.contracts import Phase5AContractError
from crc_folfox.optimize.r_adapter import FlexsurvAdapter, FlexsurvResult
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition
from crc_folfox.preprocessing.splits import build_inner_splits
from crc_folfox.preprocessing.transform import FittedPreprocessor

FLEXSURV_K_GRID = (1, 2)


class CandidateTuningError(Phase5AContractError):
    """Raised when a parent scope has no eligible B4/B5/B6 candidate configuration, or its
    selected configuration fails its final development-scope refit. Carries `diagnostics`."""

    def __init__(self, message: str, diagnostics: TuningDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _diagnostics(
    *,
    stage: str,
    attempted: int,
    failed: int,
    nonfinite: int,
    complete: int,
    failed_exceptions: set[str],
    failed_folds: set[int],
    failed_configs: tuple[str, ...],
    column_count: int,
    final_refit_failed: bool = False,
) -> TuningDiagnostics:
    return TuningDiagnostics(
        tuning_stage=stage,
        candidates_attempted=attempted,
        candidates_failed=failed,
        candidates_nonfinite=nonfinite,
        candidates_complete=complete,
        failure_exception_classes=tuple(sorted(failed_exceptions)),
        failure_penalizers=tuple(sorted(failed_configs)),
        failure_l1_ratios=(),
        failure_inner_folds=tuple(sorted(failed_folds)),
        transformed_column_count=column_count,
        zero_variance_column_count=0,
        design_rank_deficit=0,
        final_refit_failed=final_refit_failed,
    )


def _build_inner_splits(view: EndpointView, parent_split: SplitDefinition, label: str):
    if parent_split.design == "iecv_outer":
        return build_inner_splits(view, parent_split)
    if parent_split.design == "pooled":
        return build_nested_pooled_splits(view, parent_split)
    raise Phase5AContractError(f"{label} tuning requires an IECV outer or pooled parent scope")


# --- B4: penalized_nonlinear_cox ---------------------------------------------------------------


@dataclass(frozen=True)
class PenalizedNonlinearSelection:
    penalizer: float
    mean_scores: dict[float, float]
    fitter: object
    spline_state: object
    preprocessor: FittedPreprocessor
    inner_fold_count: int
    diagnostics: TuningDiagnostics


def tune_penalized_nonlinear_cox(
    view: EndpointView,
    parent_split: SplitDefinition,
    *,
    manifest: FeatureManifest | None = None,
) -> PenalizedNonlinearSelection:
    inner_splits = _build_inner_splits(view, parent_split, "penalized_nonlinear_cox")

    per_candidate_scores: dict[float, list[float]] = {p: [] for p in PENALIZER_GRID}
    failed_exceptions: set[str] = set()
    failed_folds: set[int] = set()
    failed_penalizers: set[str] = set()
    attempted = failed = nonfinite = 0
    first_fold_columns = 0

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
        if inner_fold == 0:
            first_fold_columns = train_result.matrix.shape[1]
        for penalizer in PENALIZER_GRID:
            attempted += 1
            try:
                fitter, spline_state = fit_penalized_nonlinear_cox(
                    train_result.matrix, train_result.metadata, penalizer=penalizer
                )
                val_design = transform_for_penalized_nonlinear_cox(val_result.matrix, spline_state)
                survival = predict_survival(fitter, val_design, horizons=HORIZONS)
                score = delayed_entry_ipcw_brier(
                    train_result.metadata, val_result.metadata, survival
                )
            except EXPECTED_TUNING_EXCEPTIONS as exc:
                failed += 1
                failed_exceptions.add(type(exc).__name__)
                failed_folds.add(inner_fold)
                failed_penalizers.add(str(penalizer))
                continue
            if not math.isfinite(score):
                nonfinite += 1
                failed_folds.add(inner_fold)
                failed_penalizers.add(str(penalizer))
                continue
            per_candidate_scores[penalizer].append(score)

    mean_scores = {
        p: sum(scores) / len(scores)
        for p, scores in per_candidate_scores.items()
        if len(scores) == len(inner_splits)
    }
    complete = len(mean_scores)

    def diag(stage: str, *, final_refit_failed: bool = False) -> TuningDiagnostics:
        return _diagnostics(
            stage=stage,
            attempted=attempted,
            failed=failed,
            nonfinite=nonfinite,
            complete=complete,
            failed_exceptions=failed_exceptions,
            failed_folds=failed_folds,
            failed_configs=tuple(failed_penalizers),
            column_count=first_fold_columns,
            final_refit_failed=final_refit_failed,
        )

    if not mean_scores:
        raise CandidateTuningError(
            "No eligible penalized_nonlinear_cox candidate for this parent scope",
            diag(TUNING_STAGE_NO_ELIGIBLE),
        )
    best = min(mean_scores.values())
    selected_penalizer = max(p for p, s in mean_scores.items() if abs(s - best) <= 1e-12)

    development_frame, _ = parent_split.frames(view)
    final_preprocessor = FittedPreprocessor.fit(
        development_frame,
        endpoint=view.endpoint,
        feature_set="core",
        fit_scope=parent_split.split_id,
        manifest=manifest,
    )
    final_transformed = final_preprocessor.transform(development_frame)
    try:
        final_fitter, final_spline_state = fit_penalized_nonlinear_cox(
            final_transformed.matrix, final_transformed.metadata, penalizer=selected_penalizer
        )
    except EXPECTED_TUNING_EXCEPTIONS as exc:
        raise CandidateTuningError(
            "Selected penalized_nonlinear_cox penalizer failed its final development-scope refit",
            diag(TUNING_STAGE_FINAL_REFIT_FAILED, final_refit_failed=True),
        ) from exc

    stage = TUNING_STAGE_CLEAN if failed == 0 and nonfinite == 0 else TUNING_STAGE_PARTIAL
    return PenalizedNonlinearSelection(
        selected_penalizer,
        mean_scores,
        final_fitter,
        final_spline_state,
        final_preprocessor,
        len(inner_splits),
        diag(stage),
    )


# --- B5: reduced_core_penalized_cox ------------------------------------------------------------


@dataclass(frozen=True)
class ReducedCoreSelection:
    penalizer: float
    mean_scores: dict[float, float]
    fitter: object
    preprocessor: FittedPreprocessor
    inner_fold_count: int
    diagnostics: TuningDiagnostics


def tune_reduced_core_cox(
    view: EndpointView,
    parent_split: SplitDefinition,
    *,
    manifest: FeatureManifest | None = None,
    min_observations: int = 25,
    min_events: int = 10,
) -> ReducedCoreSelection:
    inner_splits = _build_inner_splits(view, parent_split, "reduced_core_penalized_cox")

    per_candidate_scores: dict[float, list[float]] = {p: [] for p in PENALIZER_GRID}
    failed_exceptions: set[str] = set()
    failed_folds: set[int] = set()
    failed_penalizers: set[str] = set()
    attempted = failed = nonfinite = 0
    first_fold_columns = 0

    for inner_fold, inner_split in enumerate(inner_splits):
        inner_train_frame, inner_val_frame = inner_split.frames(view)
        preprocessor, _reduction = fit_reduced_core_preprocessor(
            inner_train_frame,
            endpoint=view.endpoint,
            fit_scope=inner_split.split_id,
            manifest=manifest,
            min_observations=min_observations,
            min_events=min_events,
        )
        train_result = preprocessor.transform(inner_train_frame)
        val_result = preprocessor.transform(inner_val_frame)
        if inner_fold == 0:
            first_fold_columns = train_result.matrix.shape[1]
        for penalizer in PENALIZER_GRID:
            attempted += 1
            try:
                fitter = fit_reduced_core_cox(
                    train_result.matrix, train_result.metadata, penalizer=penalizer
                )
                survival = predict_survival(fitter, val_result.matrix, horizons=HORIZONS)
                score = delayed_entry_ipcw_brier(
                    train_result.metadata, val_result.metadata, survival
                )
            except EXPECTED_TUNING_EXCEPTIONS as exc:
                failed += 1
                failed_exceptions.add(type(exc).__name__)
                failed_folds.add(inner_fold)
                failed_penalizers.add(str(penalizer))
                continue
            if not math.isfinite(score):
                nonfinite += 1
                failed_folds.add(inner_fold)
                failed_penalizers.add(str(penalizer))
                continue
            per_candidate_scores[penalizer].append(score)

    mean_scores = {
        p: sum(scores) / len(scores)
        for p, scores in per_candidate_scores.items()
        if len(scores) == len(inner_splits)
    }
    complete = len(mean_scores)

    def diag(stage: str, *, final_refit_failed: bool = False) -> TuningDiagnostics:
        return _diagnostics(
            stage=stage,
            attempted=attempted,
            failed=failed,
            nonfinite=nonfinite,
            complete=complete,
            failed_exceptions=failed_exceptions,
            failed_folds=failed_folds,
            failed_configs=tuple(failed_penalizers),
            column_count=first_fold_columns,
            final_refit_failed=final_refit_failed,
        )

    if not mean_scores:
        raise CandidateTuningError(
            "No eligible reduced_core_penalized_cox candidate for this parent scope",
            diag(TUNING_STAGE_NO_ELIGIBLE),
        )
    best = min(mean_scores.values())
    selected_penalizer = max(p for p, s in mean_scores.items() if abs(s - best) <= 1e-12)

    development_frame, _ = parent_split.frames(view)
    final_preprocessor, _final_reduction = fit_reduced_core_preprocessor(
        development_frame,
        endpoint=view.endpoint,
        fit_scope=parent_split.split_id,
        manifest=manifest,
        min_observations=min_observations,
        min_events=min_events,
    )
    final_transformed = final_preprocessor.transform(development_frame)
    try:
        final_fitter = fit_reduced_core_cox(
            final_transformed.matrix, final_transformed.metadata, penalizer=selected_penalizer
        )
    except EXPECTED_TUNING_EXCEPTIONS as exc:
        raise CandidateTuningError(
            "Selected reduced_core_penalized_cox penalizer failed its final refit",
            diag(TUNING_STAGE_FINAL_REFIT_FAILED, final_refit_failed=True),
        ) from exc

    stage = TUNING_STAGE_CLEAN if failed == 0 and nonfinite == 0 else TUNING_STAGE_PARTIAL
    return ReducedCoreSelection(
        selected_penalizer,
        mean_scores,
        final_fitter,
        final_preprocessor,
        len(inner_splits),
        diag(stage),
    )


# --- B6: flexsurv_rp_spline ----------------------------------------------------------------------


@dataclass(frozen=True)
class FlexsurvSelection:
    k: int
    mean_scores: dict[int, float]
    result: FlexsurvResult  # the final development-scope refit
    inner_fold_count: int
    diagnostics: TuningDiagnostics
    preprocessor: FittedPreprocessor  # needed to transform assessment rows at evaluation time


def tune_flexsurv_cox(
    view: EndpointView,
    parent_split: SplitDefinition,
    *,
    bundles_dir: Path,
    manifest: FeatureManifest | None = None,
    adapter: FlexsurvAdapter | None = None,
) -> FlexsurvSelection:
    """B6 uses the four minimal-Cox baseline variables (optimization-spec.md, "B6"), not the
    core feature set -- holding the covariate set at the benchmark's isolates the question this
    candidate exists to answer (whether relaxing the proportional-hazards baseline helps)."""
    adapter = adapter or FlexsurvAdapter()
    inner_splits = _build_inner_splits(view, parent_split, "flexsurv_rp_spline")
    bundles_dir.mkdir(parents=True, exist_ok=True)

    per_candidate_scores: dict[int, list[float]] = {k: [] for k in FLEXSURV_K_GRID}
    failed_exceptions: set[str] = set()
    failed_folds: set[int] = set()
    failed_configs: set[str] = set()
    attempted = failed = nonfinite = 0
    first_fold_columns = 0

    for inner_fold, inner_split in enumerate(inner_splits):
        inner_train_frame, inner_val_frame = inner_split.frames(view)
        preprocessor = FittedPreprocessor.fit(
            inner_train_frame,
            endpoint=view.endpoint,
            feature_set="baseline",
            fit_scope=inner_split.split_id,
            manifest=manifest,
        )
        train_result = preprocessor.transform(inner_train_frame)
        val_result = preprocessor.transform(inner_val_frame)
        if inner_fold == 0:
            first_fold_columns = train_result.matrix.shape[1]
        for k in FLEXSURV_K_GRID:
            attempted += 1
            fold_seed = child_seed(
                endpoint=parent_split.endpoint,
                design=parent_split.design,
                held_out_institution=parent_split.held_out_institution,
                repeat=parent_split.repeat,
                fold=parent_split.fold,
                model_name="flexsurv_rp_spline",
                candidate=str(k),
                purpose=f"flexsurv_inner_fold_{inner_fold}",
            )
            try:
                fold_result = adapter.fit(
                    train_result.matrix,
                    train_result.metadata,
                    val_result.matrix,
                    k=k,
                    seed=fold_seed,
                    horizons=HORIZONS,
                    destination=bundles_dir / f"_inner_fold_{inner_fold}_k{k}.rds",
                )
                score = delayed_entry_ipcw_brier(
                    train_result.metadata, val_result.metadata, fold_result.survival
                )
            except (Phase5AContractError, *EXPECTED_TUNING_EXCEPTIONS) as exc:
                failed += 1
                failed_exceptions.add(type(exc).__name__)
                failed_folds.add(inner_fold)
                failed_configs.add(str(k))
                continue
            if not math.isfinite(score):
                nonfinite += 1
                failed_folds.add(inner_fold)
                failed_configs.add(str(k))
                continue
            per_candidate_scores[k].append(score)

    mean_scores = {
        k: sum(scores) / len(scores)
        for k, scores in per_candidate_scores.items()
        if len(scores) == len(inner_splits)
    }
    complete = len(mean_scores)

    def diag(stage: str, *, final_refit_failed: bool = False) -> TuningDiagnostics:
        return _diagnostics(
            stage=stage,
            attempted=attempted,
            failed=failed,
            nonfinite=nonfinite,
            complete=complete,
            failed_exceptions=failed_exceptions,
            failed_folds=failed_folds,
            failed_configs=tuple(failed_configs),
            column_count=first_fold_columns,
            final_refit_failed=final_refit_failed,
        )

    if not mean_scores:
        raise CandidateTuningError(
            "No eligible flexsurv_rp_spline candidate for this parent scope",
            diag(TUNING_STAGE_NO_ELIGIBLE),
        )
    best = min(mean_scores.values())
    # Tie-break toward the smaller k (fewer knots, the simpler model).
    selected_k = min(k for k, s in mean_scores.items() if abs(s - best) <= 1e-12)

    development_frame, _ = parent_split.frames(view)
    final_preprocessor = FittedPreprocessor.fit(
        development_frame,
        endpoint=view.endpoint,
        feature_set="baseline",
        fit_scope=parent_split.split_id,
        manifest=manifest,
    )
    final_transformed = final_preprocessor.transform(development_frame)
    final_seed = child_seed(
        endpoint=parent_split.endpoint,
        design=parent_split.design,
        held_out_institution=parent_split.held_out_institution,
        repeat=parent_split.repeat,
        fold=parent_split.fold,
        model_name="flexsurv_rp_spline",
        candidate=str(selected_k),
        purpose="flexsurv_final_refit",
    )
    try:
        final_result = adapter.fit(
            final_transformed.matrix,
            final_transformed.metadata,
            final_transformed.matrix.iloc[:1],
            k=selected_k,
            seed=final_seed,
            horizons=HORIZONS,
            destination=bundles_dir / "final.rds",
        )
    except Phase5AContractError as exc:
        raise CandidateTuningError(
            "Selected flexsurv k failed its final development-scope refit",
            diag(TUNING_STAGE_FINAL_REFIT_FAILED, final_refit_failed=True),
        ) from exc

    stage = TUNING_STAGE_CLEAN if failed == 0 and nonfinite == 0 else TUNING_STAGE_PARTIAL
    return FlexsurvSelection(
        selected_k, mean_scores, final_result, len(inner_splits), diag(stage), final_preprocessor
    )
