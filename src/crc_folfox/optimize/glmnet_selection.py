"""The exploratory-optimization workstream tuning/selection for the glmnet ridge/elastic-net Cox
candidates
(B2 `glmnet_ridge_core`, B3 `glmnet_enet_core`; optimization-spec.md).

Mirrors `crc_folfox.model_training.selection.tune_elastic_net_cox`'s shape -- nested inner splits,
the same delayed-entry IPCW Brier tuning objective, the corrected per-attempt failure isolation
from Workstream A -- adapted to glmnet's own constraint: a `glmnet()` call fits its whole lambda
path in one shot, so a failure disqualifies that fold's whole path, not one lambda in isolation.
What makes cross-fold score averaging well-defined despite glmnet's data-dependent path is fixing
one reference lambda sequence per **parent** scope (never transferred across parent scopes,
centres, or endpoints) and forcing every inner fold onto exactly that sequence via
`GlmnetAdapter.fit`'s explicit-lambda-tuple mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from crc_folfox.features import FeatureManifest
from crc_folfox.model_training.contracts import HORIZONS, child_seed
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
from crc_folfox.optimize.contracts import Phase5AContractError
from crc_folfox.optimize.r_adapter import GlmnetAdapter, GlmnetPathResult
from crc_folfox.preprocessing.contracts import EndpointView, SplitDefinition
from crc_folfox.preprocessing.splits import build_inner_splits
from crc_folfox.preprocessing.transform import FittedPreprocessor

_MODEL_NAMES = {0.0: "glmnet_ridge_core", 0.5: "glmnet_enet_core", 1.0: "glmnet_enet_core"}


class GlmnetTuningError(Phase5AContractError):
    """Raised when a parent scope has no eligible glmnet lambda, or its selected value fails its
    final development-scope refit. Carries `diagnostics`, mirroring
    `crc_folfox.model_training.selection.ElasticNetTuningError`."""

    def __init__(self, message: str, diagnostics: TuningDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class GlmnetSelection:
    alpha: float
    lambda_value: float
    mean_scores: dict[float, float]
    result: GlmnetPathResult  # the final development-scope refit, index 0 == the selected lambda
    inner_fold_count: int
    diagnostics: TuningDiagnostics
    preprocessor: FittedPreprocessor  # fit on the full development scope; needed at evaluation
    # time to transform assessment rows into the matrix `result`'s bundle expects -- the R bundle
    # itself has no encoding/scaling state, unlike a Python `CoxBundleState`.


def _select_glmnet_lambda(scores: dict[float, float]) -> float:
    """Smallest-mean-score lambda; `1e-12` ties broken toward the larger lambda (the more
    regularized solution), matching `select_elastic_net`'s tie-break philosophy."""
    if not scores:
        raise Phase5AContractError("No eligible glmnet lambda candidate")
    best = min(scores.values())
    tied = [lam for lam, score in scores.items() if abs(score - best) <= 1e-12]
    return max(tied)


def tune_glmnet_cox(
    view: EndpointView,
    parent_split: SplitDefinition,
    *,
    alpha: float,
    bundles_dir: Path,
    manifest: FeatureManifest | None = None,
    adapter: GlmnetAdapter | None = None,
) -> GlmnetSelection:
    """Tune and refit a glmnet Cox candidate for one development scope.

    `alpha=0.0` is B2 (`glmnet_ridge_core`); `alpha in (0.5, 1.0)` is B3 (`glmnet_enet_core`).
    """
    adapter = adapter or GlmnetAdapter()
    model_name = _MODEL_NAMES.get(alpha, "glmnet_enet_core")
    if parent_split.design == "iecv_outer":
        inner_splits = build_inner_splits(view, parent_split)
    elif parent_split.design == "pooled":
        inner_splits = build_nested_pooled_splits(view, parent_split)
    else:
        raise Phase5AContractError("glmnet tuning requires an IECV outer or pooled parent scope")

    development_frame, _ = parent_split.frames(view)
    scope_preprocessor = FittedPreprocessor.fit(
        development_frame,
        endpoint=view.endpoint,
        feature_set="core",
        fit_scope=parent_split.split_id,
        manifest=manifest,
    )
    scope_transform = scope_preprocessor.transform(development_frame)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    reference_seed = child_seed(
        endpoint=parent_split.endpoint,
        design=parent_split.design,
        held_out_institution=parent_split.held_out_institution,
        repeat=parent_split.repeat,
        fold=parent_split.fold,
        model_name=model_name,
        candidate=str(alpha),
        purpose="glmnet_reference_path",
    )
    reference = adapter.fit(
        scope_transform.matrix,
        scope_transform.metadata,
        scope_transform.matrix.iloc[:1],
        alpha=alpha,
        lambda_value=None,
        seed=reference_seed,
        horizons=HORIZONS,
        destination=bundles_dir / "_reference_path.rds",
    )
    reference_lambdas = tuple(float(v) for v in reference.lambda_values)

    inner_scores: dict[float, list[float]] = {lam: [] for lam in reference_lambdas}
    failed_lambdas: set[float] = set()
    failed_inner_folds: set[int] = set()
    candidates_attempted = 0
    candidates_failed = 0
    candidates_nonfinite = 0

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
        candidates_attempted += len(reference_lambdas)
        fold_seed = child_seed(
            endpoint=parent_split.endpoint,
            design=parent_split.design,
            held_out_institution=parent_split.held_out_institution,
            repeat=parent_split.repeat,
            fold=parent_split.fold,
            model_name=model_name,
            candidate=str(alpha),
            purpose=f"glmnet_inner_fold_{inner_fold}",
        )
        try:
            fold_result = adapter.fit(
                train_result.matrix,
                train_result.metadata,
                val_result.matrix,
                alpha=alpha,
                lambda_value=reference_lambdas,
                seed=fold_seed,
                horizons=HORIZONS,
                destination=bundles_dir / f"_inner_fold_{inner_fold}.rds",
            )
        except Phase5AContractError:
            # glmnet fits its whole lambda path in one call; a failure disqualifies this fold's
            # entire path, not one lambda in isolation (see module docstring).
            candidates_failed += len(reference_lambdas)
            failed_lambdas.update(reference_lambdas)
            failed_inner_folds.add(inner_fold)
            continue

        for lambda_index, lam in enumerate(reference_lambdas):
            survival = fold_result.survival[lambda_index]
            try:
                score = delayed_entry_ipcw_brier(
                    train_result.metadata, val_result.metadata, survival
                )
            except EXPECTED_TUNING_EXCEPTIONS:
                # A degenerate cell (for example zero censoring support at a tuning horizon) is
                # an expected per-lambda scoring failure, not a fit failure -- it disqualifies
                # only this lambda in this fold, exactly like a nonfinite score below.
                candidates_failed += 1
                failed_lambdas.add(lam)
                failed_inner_folds.add(inner_fold)
                continue
            if not math.isfinite(score):
                candidates_nonfinite += 1
                failed_lambdas.add(lam)
                continue
            inner_scores[lam].append(score)

    mean_scores = {
        lam: sum(scores) / len(scores)
        for lam, scores in inner_scores.items()
        if len(scores) == len(inner_splits)
    }
    candidates_complete = len(mean_scores)

    def _diagnostics(stage: str, *, final_refit_failed: bool = False) -> TuningDiagnostics:
        return TuningDiagnostics(
            tuning_stage=stage,
            candidates_attempted=candidates_attempted,
            candidates_failed=candidates_failed,
            candidates_nonfinite=candidates_nonfinite,
            candidates_complete=candidates_complete,
            failure_exception_classes=(("Phase5AContractError",) if candidates_failed else ()),
            failure_penalizers=(),
            failure_l1_ratios=(),
            failure_inner_folds=tuple(sorted(failed_inner_folds)),
            transformed_column_count=scope_transform.matrix.shape[1],
            zero_variance_column_count=0,
            design_rank_deficit=0,
            final_refit_failed=final_refit_failed,
        )

    if not mean_scores:
        raise GlmnetTuningError(
            f"No eligible glmnet lambda candidate for this parent scope ({model_name})",
            _diagnostics(TUNING_STAGE_NO_ELIGIBLE),
        )

    selected_lambda = _select_glmnet_lambda(mean_scores)
    final_seed = child_seed(
        endpoint=parent_split.endpoint,
        design=parent_split.design,
        held_out_institution=parent_split.held_out_institution,
        repeat=parent_split.repeat,
        fold=parent_split.fold,
        model_name=model_name,
        candidate=str(alpha),
        purpose="glmnet_final_refit",
    )
    try:
        final_result = adapter.fit(
            scope_transform.matrix,
            scope_transform.metadata,
            scope_transform.matrix.iloc[:1],
            alpha=alpha,
            lambda_value=selected_lambda,
            seed=final_seed,
            horizons=HORIZONS,
            destination=bundles_dir / "final.rds",
        )
    except Phase5AContractError as exc:
        raise GlmnetTuningError(
            f"Selected glmnet lambda failed its final development-scope refit ({model_name})",
            _diagnostics(TUNING_STAGE_FINAL_REFIT_FAILED, final_refit_failed=True),
        ) from exc

    stage = (
        TUNING_STAGE_CLEAN
        if candidates_failed == 0 and candidates_nonfinite == 0
        else TUNING_STAGE_PARTIAL
    )
    return GlmnetSelection(
        alpha,
        selected_lambda,
        mean_scores,
        final_result,
        len(inner_splits),
        _diagnostics(stage),
        scope_preprocessor,
    )


# B3's alpha itself is tuned, unlike B2 (alpha fixed at 0.0). Pure ridge is excluded from this
# grid because B2 already covers it (optimization-spec.md, "B3").
ENET_ALPHA_GRID = (0.5, 1.0)


def tune_glmnet_enet_cox(
    view: EndpointView,
    parent_split: SplitDefinition,
    *,
    bundles_dir: Path,
    manifest: FeatureManifest | None = None,
    adapter: GlmnetAdapter | None = None,
) -> GlmnetSelection:
    """B3: tune both `alpha` (the frozen grid `ENET_ALPHA_GRID`) and `lambda`.

    Runs `tune_glmnet_cox` once per alpha value -- each with its own independent reference lambda
    path and inner-fold scoring -- then selects the overall best `(alpha, lambda)` pair. Tie-break:
    larger lambda, then smaller alpha (optimization-spec.md, "Tuning and selection").
    """
    selections: dict[float, GlmnetSelection] = {}
    errors: dict[float, GlmnetTuningError] = {}
    for alpha in ENET_ALPHA_GRID:
        try:
            selections[alpha] = tune_glmnet_cox(
                view,
                parent_split,
                alpha=alpha,
                bundles_dir=bundles_dir / f"alpha_{alpha}",
                manifest=manifest,
                adapter=adapter,
            )
        except GlmnetTuningError as exc:
            errors[alpha] = exc

    if not selections:
        # Every alpha failed; the smaller alpha's diagnostics are reported deterministically.
        raise errors[min(errors)]

    best_score = min(sel.mean_scores[sel.lambda_value] for sel in selections.values())
    tied_alphas = [
        alpha
        for alpha, sel in selections.items()
        if abs(sel.mean_scores[sel.lambda_value] - best_score) <= 1e-12
    ]
    best_alpha = min(tied_alphas, key=lambda a: (-selections[a].lambda_value, a))
    return selections[best_alpha]
