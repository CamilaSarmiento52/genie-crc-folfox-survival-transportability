"""Atomic explainability workflow: the one supported orchestration boundary.

Ties together preflight verification, the model-availability hierarchy, grouped permutation
importance, coefficients, partial effects, selection stability, bootstrap-refit-derived
uncertainty and prediction stability, the comparability rules, and atomic publication -- following
the same shape as `crc_folfox.evaluate.workflow.run_evaluation_workflow` and
`crc_folfox.optimize.workflow.run_workstream_b_development`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crc_folfox.evaluate.contracts import CENTRES, IBS_GRID, EvaluationContractError
from crc_folfox.explain.bootstrap_attribution import build_bootstrap_attribution_rows
from crc_folfox.explain.coefficient_paths import (
    GridPathRow,
    compute_elastic_net_grid_path,
    compute_glmnet_enet_grid_path,
)
from crc_folfox.explain.coefficients import (
    attach_bootstrap_intervals,
    attach_model_based_uncertainty,
    compute_minimal_cox_model_based_uncertainty,
    point_coefficients,
)
from crc_folfox.explain.comparability import (
    SCOPE_BASIS_IDENTICAL,
    build_cross_centre_row,
    build_cross_endpoint_row,
    build_cross_model_row,
)
from crc_folfox.explain.contracts import (
    ANALYSIS_STATUS_POST_HOC_EXPLORATORY,
    B_EXPLAIN,
    EXPLAINED_MODEL_NAMES,
    MIN_BOOTSTRAP_FINITE_FRACTION,
    MIN_ELIGIBLE_EVENTS,
    MIN_EVENT_FREE,
    MODEL_REGISTER,
    NOT_EXPLAINED_MODEL_NAMES,
    PARTIAL_EFFECT_MODEL_NAMES,
    PRIMARY_EXPLANATION_HORIZON,
    R_BOOT,
    R_PERM,
    REPORTING_ESTIMABLE,
    REPORTING_IMPRECISE,
    REPORTING_NOT_COMPUTABLE,
    SPARSITY_CAPABLE_MODEL_NAMES,
    ExplainContractError,
    registration_for,
)
from crc_folfox.explain.feature_groups import (
    assert_group_inventory,
    build_feature_groups,
    dominant_group_coefficient,
)
from crc_folfox.explain.importance import IDX_365, IDX_730, compute_group_importance
from crc_folfox.explain.loading import (
    ModelNotFitted,
    load_model,
    loaded_feature_columns,
    predict_assessment,
)
from crc_folfox.explain.partial_effects import age_grid_from_fitting_rows, compute_partial_effect
from crc_folfox.explain.prediction_stability import (
    cross_family_spread,
    summarize_prediction_stability,
)
from crc_folfox.explain.preflight import reverify_frozen_primary_unchanged, verify_all_inputs
from crc_folfox.explain.refits import run_bootstrap_refits
from crc_folfox.explain.selection_stability import (
    NOT_APPLICABLE_NO_SPARSITY,
    selection_frequency,
)
from crc_folfox.model_training.contracts import ModelTrainingContractError
from crc_folfox.optimize.contracts import OptimizationContractError
from crc_folfox.preprocessing.contracts import PreprocessingContractError, build_endpoint_view
from crc_folfox.preprocessing.splits import build_iecv_splits

REPORT = "explainability_report.md"
MANIFEST = "explainability_manifest.json"
SCOPE_STATUS = "explanation_scope_status.csv"
MODEL_AVAILABILITY = "model_availability.csv"
COEFFICIENTS = "coefficient_summary.csv"
IMPORTANCE = "importance_summary.csv"
SELECTION_STABILITY = "selection_stability.csv"
PARTIAL_EFFECTS = "partial_effects.csv"
ATTRIBUTION_STABILITY = "attribution_stability.csv"
PREDICTION_STABILITY = "prediction_stability.csv"
CROSS_FAMILY_PREDICTION_SPREAD = "cross_family_prediction_spread.csv"
COEFFICIENT_PATHS = "coefficient_paths.csv"
BOOTSTRAP_ATTRIBUTION_STABILITY = "bootstrap_attribution_stability.csv"
COMPARABILITY_MATRIX = "comparability_matrix.csv"
FIGURES = "figures"

ARTIFACT_NAMES = (
    REPORT,
    MANIFEST,
    SCOPE_STATUS,
    MODEL_AVAILABILITY,
    COEFFICIENTS,
    IMPORTANCE,
    SELECTION_STABILITY,
    PARTIAL_EFFECTS,
    ATTRIBUTION_STABILITY,
    PREDICTION_STABILITY,
    CROSS_FAMILY_PREDICTION_SPREAD,
    COEFFICIENT_PATHS,
    BOOTSTRAP_ATTRIBUTION_STABILITY,
    COMPARABILITY_MATRIX,
    FIGURES,
)

# Per-model split (2026-08-13): `run_model_explainability` persists one model's aggregate-only
# results under `<intermediate>/<model_name>/`; `aggregate_explainability_results` combines
# however many models finished into the final published bundle.
MODEL_EXPLANATION_SCOPE_SUMMARY = "scope_summary.csv"
MODEL_EXPLANATION_SUCCESS_MARKER = "_SUCCESS"
MODEL_EXPLANATION_FRAME_NAMES = (
    SCOPE_STATUS,
    COEFFICIENTS,
    IMPORTANCE,
    SELECTION_STABILITY,
    PARTIAL_EFFECTS,
    PREDICTION_STABILITY,
    COEFFICIENT_PATHS,
    BOOTSTRAP_ATTRIBUTION_STABILITY,
    MODEL_EXPLANATION_SCOPE_SUMMARY,
)


class ExplainabilityWorkflowError(RuntimeError):
    """Aggregate-safe failure of the explainability workflow."""


@dataclass(frozen=True)
class ExplainabilityArtifacts:
    directory: Path
    paths: dict[str, Path]


@dataclass(frozen=True)
class ExplainabilityResult:
    scope_status: pd.DataFrame
    model_availability: pd.DataFrame
    coefficient_summary: pd.DataFrame
    importance_summary: pd.DataFrame
    selection_stability: pd.DataFrame
    partial_effects: pd.DataFrame
    attribution_stability: pd.DataFrame
    prediction_stability: pd.DataFrame
    cross_family_prediction_spread: pd.DataFrame
    coefficient_paths: pd.DataFrame
    bootstrap_attribution_stability: pd.DataFrame
    comparability_matrix: pd.DataFrame
    artifacts: ExplainabilityArtifacts | None = None


@dataclass(frozen=True)
class ModelExplanationArtifacts:
    directory: Path
    paths: dict[str, Path]


@dataclass
class ModelExplanationResult:
    """One model's aggregate-only results across every (endpoint, centre) scope -- never the
    per-patient risk predictions, which stay in-memory-only per the project's aggregate-safety rule
    and are recomputed fresh (never persisted) by `aggregate_explainability_results`."""

    model_name: str
    scope_status: pd.DataFrame
    coefficient_summary: pd.DataFrame
    importance_summary: pd.DataFrame
    selection_stability: pd.DataFrame
    partial_effects: pd.DataFrame
    prediction_stability: pd.DataFrame
    coefficient_paths: pd.DataFrame
    bootstrap_attribution_stability: pd.DataFrame
    scope_summary: pd.DataFrame
    artifacts: ModelExplanationArtifacts | None = None


def _reporting_status(n_events: int, n_event_free: int, finite_fraction: float) -> str:
    if n_events == 0 and n_event_free == 0:
        return REPORTING_NOT_COMPUTABLE
    if (
        n_events >= MIN_ELIGIBLE_EVENTS
        and n_event_free >= MIN_EVENT_FREE
        and finite_fraction >= MIN_BOOTSTRAP_FINITE_FRACTION
    ):
        return REPORTING_ESTIMABLE
    return REPORTING_IMPRECISE


def _refit_config(loaded) -> dict:
    if loaded.cox_state is not None:
        return {"penalizer": loaded.cox_state.penalizer, "l1_ratio": loaded.cox_state.l1_ratio}
    return dict(loaded.r_record.config)


@dataclass
class _ScopeArtifacts:
    """In-memory per-scope results, kept only long enough to build cross-centre/endpoint/model
    comparability rows within one workflow run -- never persisted beyond the published summaries."""

    endpoint: str
    centre: str
    model_name: str
    tier: str
    analysis_status: str
    importance_by_group_730: dict[str, float]
    coefficients_by_group: dict[str, float]
    risk_365: np.ndarray | None
    risk_730: np.ndarray | None


_EXPECTED_SCOPE_EXCEPTIONS = (
    ExplainContractError,
    ModelTrainingContractError,
    PreprocessingContractError,
    EvaluationContractError,
    OptimizationContractError,
)


def _not_computable_scope_row(
    registration, *, endpoint: str, centre: str, model_name: str, reason: str
) -> dict:
    return {
        "endpoint": endpoint,
        "centre": centre,
        "model": model_name,
        "explanation_tier": registration.tier,
        "source_registry": registration.source_registry,
        "analysis_status": registration.analysis_status,
        "explanation_status": "not_computable",
        "reason": reason,
    }


def _explain_one_scope(
    paths,
    view,
    split,
    model_name: str,
    *,
    endpoint: str,
    centre: str,
    r_perm: int,
    b_explain: int,
    r_boot: int,
    tmp_root: Path,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    _ScopeArtifacts | None,
]:
    """Explain one `(endpoint, centre, model)` scope. A degenerate scope -- for example a
    categorical predictor that happens to carry only one level in a particular development
    partition, which can occur in a small fixture though not in the real ~400-patient IECV folds
    -- is isolated to a single `not_computable` scope row rather than aborting the whole run,
    matching this project's established per-scope failure-isolation convention (Workstream A's own
    per-candidate tuning isolation)."""
    registration = registration_for(model_name)

    try:
        loaded = load_model(paths, split, model_name)
    except ModelNotFitted as exc:
        return (
            [
                _not_computable_scope_row(
                    registration,
                    endpoint=endpoint,
                    centre=centre,
                    model_name=model_name,
                    reason=str(exc),
                )
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            None,
        )

    try:
        return _explain_loaded_scope(
            loaded,
            view,
            split,
            model_name,
            registration,
            endpoint=endpoint,
            centre=centre,
            r_perm=r_perm,
            b_explain=b_explain,
            r_boot=r_boot,
            tmp_root=tmp_root,
        )
    except _EXPECTED_SCOPE_EXCEPTIONS as exc:
        return (
            [
                _not_computable_scope_row(
                    registration,
                    endpoint=endpoint,
                    centre=centre,
                    model_name=model_name,
                    reason=str(exc),
                )
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            None,
        )


def _explain_loaded_scope(
    loaded,
    view,
    split,
    model_name: str,
    registration,
    *,
    endpoint: str,
    centre: str,
    r_perm: int,
    b_explain: int,
    r_boot: int,
    tmp_root: Path,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    _ScopeArtifacts | None,
]:
    scope_rows, coefficient_rows, importance_rows = [], [], []
    selection_rows, partial_effect_rows, prediction_stability_rows = [], [], []
    coefficient_path_rows: list[dict] = []
    bootstrap_attribution_rows: list[dict] = []

    development_frame, assessment_frame = split.frames(view)
    matrix, metadata, prediction = predict_assessment(loaded, assessment_frame, IBS_GRID)
    entry = metadata["entry"].to_numpy(dtype=float)
    event = metadata["event"].to_numpy(dtype=int)
    n_events, n_event_free = int(event.sum()), int((event == 0).sum())

    # Canonical group inventory from the point model's own frozen feature schema (Python
    # `cox_state.feature_columns` or R `r_preprocessor.output_columns`), not the assessment
    # matrix -- `explainability-spec.md`'s 2026-08-11 amendment makes explicit that no
    # assessment-side property may determine the inventory used for bootstrap ranking. The
    # assertion below still catches any drift immediately rather than silently.
    canonical_columns = loaded_feature_columns(loaded)
    if tuple(matrix.columns.astype(str)) != tuple(canonical_columns):
        raise ExplainContractError(
            f"'{model_name}' assessment matrix columns do not match the loaded model's own schema"
        )
    groups = build_feature_groups(tuple(canonical_columns))
    assert_group_inventory(model_name, groups)

    config = _refit_config(loaded)
    tmp_dir = tmp_root / f"{endpoint}_{centre}_{model_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    age_grid = None
    reference_age = None
    presentation_levels: tuple[str | None, ...] = (None,)
    if model_name in PARTIAL_EFFECT_MODEL_NAMES:
        fitting_ages = development_frame["age_at_folfox"].to_numpy(float)
        age_grid = age_grid_from_fitting_rows(fitting_ages)
        reference_age = float(np.median(fitting_ages))
        if model_name == "penalized_nonlinear_cox":
            state = loaded.cox_state
            reference = state.preprocessor.references.get("metastatic_presentation")
            encoded = state.preprocessor.encoded_levels.get("metastatic_presentation", ())
            # `encoded_levels` is populated from every distinct pooled level *including* the
            # reference (`FittedPreprocessor.fit` only excludes it from the one-hot design
            # columns, not from this tuple) -- excluding it here before prepending `reference`
            # back is what keeps every presentation level exactly once (2026-08-11 fix; the prior
            # duplicate produced real duplicate `partial_effects.csv` rows for the reference level
            # and a silently-overwritten redundant bootstrap refit for it).
            non_reference = tuple(level for level in encoded if level != reference)
            presentation_levels = tuple(
                level for level in (reference, *non_reference) if level is not None
            )

    refit_collection = None
    if b_explain > 0:
        refit_collection = run_bootstrap_refits(
            model_name,
            development_frame,
            assessment_frame,
            endpoint=endpoint,
            centre=centre,
            config=config,
            canonical_groups=groups,
            b_explain=b_explain,
            r_boot=r_boot,
            age_grid=age_grid,
            reference_age=reference_age,
            presentation_levels=presentation_levels,
            tmp_dir=tmp_dir,
        )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    primary_refit = refit_collection

    importance_by_group_730: dict[str, float] = {}
    for group_name, columns in groups.items():
        result = compute_group_importance(
            loaded,
            matrix,
            metadata,
            columns,
            endpoint=endpoint,
            centre=centre,
            group_name=group_name,
            r_perm=r_perm,
        )
        value_730, finite_730 = result.importance_730()
        value_365, finite_365 = result.importance_365()
        value_ibs, finite_ibs = result.importance_ibs()
        if value_730 is not None:
            importance_by_group_730[group_name] = value_730
        for horizon_label, value, finite in (
            ("365", value_365, finite_365),
            ("730", value_730, finite_730),
            ("ibs", value_ibs, finite_ibs),
        ):
            importance_rows.append(
                {
                    "endpoint": endpoint,
                    "centre": centre,
                    "model": model_name,
                    "explanation_tier": registration.tier,
                    "analysis_status": registration.analysis_status,
                    "feature_group": group_name,
                    "n_columns": result.n_columns,
                    "metric": horizon_label,
                    "importance": value,
                    "finite_fraction": finite,
                    "reporting_status": _reporting_status(n_events, n_event_free, finite),
                }
            )

    coefficients_by_group: dict[str, float] = {}
    if prediction.coefficients is not None:
        rows = point_coefficients(loaded, prediction)
        if model_name == "minimal_cox":
            uncertainty = compute_minimal_cox_model_based_uncertainty(
                loaded, development_frame, fit_scope=f"{endpoint}|{centre}|explain_se"
            )
            rows = attach_model_based_uncertainty(rows, uncertainty)
        elif primary_refit is not None:
            rows = attach_bootstrap_intervals(rows, primary_refit.refit_coefficients)
        columns_by_group: dict[str, list[str]] = {}
        coefficients_lookup: dict[str, float] = {}
        for row in rows:
            columns_by_group.setdefault(row.feature_group, []).append(row.design_column)
            coefficients_lookup[row.design_column] = row.coefficient
            coefficient_rows.append(
                {
                    "endpoint": endpoint,
                    "centre": centre,
                    "model": model_name,
                    "explanation_tier": registration.tier,
                    "analysis_status": registration.analysis_status,
                    "design_column": row.design_column,
                    "feature_group": row.feature_group,
                    "coefficient": row.coefficient,
                    "hazard_ratio": row.hazard_ratio,
                    "interval_provenance": row.interval_provenance,
                    "standard_error": row.standard_error,
                    "ci_lower": row.ci_lower,
                    "ci_upper": row.ci_upper,
                }
            )
        # Largest-|coefficient| column per group, not "first encountered" (the prior defect --
        # wrong for every multi-column group, e.g. the real TTNT-D/DFCI `ca_d_site` case where the
        # first column was negative but the largest-magnitude one was positive).
        for group_name, columns in columns_by_group.items():
            dominant = dominant_group_coefficient(tuple(columns), coefficients_lookup)
            if dominant is not None:
                coefficients_by_group[group_name] = dominant[1]

        if model_name in SPARSITY_CAPABLE_MODEL_NAMES and primary_refit is not None:
            frequencies = selection_frequency(groups, primary_refit.coefficients_by_refit)
            for group_name, frequency in frequencies.items():
                selection_rows.append(
                    {
                        "endpoint": endpoint,
                        "centre": centre,
                        "model": model_name,
                        "feature_group": group_name,
                        "selection_frequency": frequency,
                    }
                )
        elif model_name in SPARSITY_CAPABLE_MODEL_NAMES:
            for group_name in groups:
                selection_rows.append(
                    {
                        "endpoint": endpoint,
                        "centre": centre,
                        "model": model_name,
                        "feature_group": group_name,
                        "selection_frequency": NOT_APPLICABLE_NO_SPARSITY,
                    }
                )

        # E3, `coefficient_paths.csv` (2026-08-11 amendment): the model's own frozen tuning-grid
        # trajectory, at the development-scope fit -- a different computation from the bootstrap
        # refits above, and restricted to the two grid-capable models.
        grid_path_rows: list[GridPathRow] = []
        if model_name == "elastic_net_cox":
            development_transform = loaded.cox_state.preprocessor.transform(development_frame)
            grid_path_rows = compute_elastic_net_grid_path(
                development_transform.matrix,
                development_transform.metadata,
                selected_penalizer=config["penalizer"],
                selected_l1_ratio=config["l1_ratio"],
            )
        elif model_name == "glmnet_enet_core":
            coef_path_tmp_dir = tmp_root / f"{endpoint}_{centre}_{model_name}_coefficient_path"
            coef_path_tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                grid_path_rows = compute_glmnet_enet_grid_path(
                    development_frame,
                    loaded.r_preprocessor,
                    split,
                    model_name=model_name,
                    alpha=config["alpha"],
                    selected_lambda=config["lambda"],
                    frozen_coefficients=prediction.coefficients or {},
                    tmp_dir=coef_path_tmp_dir,
                )
            finally:
                shutil.rmtree(coef_path_tmp_dir, ignore_errors=True)
        for row in grid_path_rows:
            coefficient_path_rows.append(
                {
                    "endpoint": endpoint,
                    "centre": centre,
                    "model": model_name,
                    "tier": registration.tier,
                    "configuration_id": row.configuration_id,
                    "configuration_label": row.configuration_label,
                    "penalizer": row.penalizer,
                    "l1_ratio": row.l1_ratio,
                    "alpha": row.alpha,
                    "lambda_value": row.lambda_value,
                    "design_column": row.design_column,
                    "feature_group": row.feature_group,
                    "coefficient": row.coefficient,
                    "is_selected_configuration": row.is_selected_configuration,
                    "fit_status": row.fit_status,
                    "path_provenance": row.path_provenance,
                }
            )

        # E5, `bootstrap_attribution_stability.csv` (2026-08-11 amendment): bootstrap-refit
        # importance/rank/sign-stability, published separately from the E2 point estimate.
        for row in build_bootstrap_attribution_rows(
            refit_collection, groups, coefficients_by_group
        ):
            bootstrap_attribution_rows.append(
                {"endpoint": endpoint, "centre": centre, "model": model_name, **row}
            )

    if age_grid is not None:
        for level in presentation_levels:
            curve = compute_partial_effect(
                loaded, age_grid, reference_age=reference_age, presentation_level=level
            )
            bands = {}
            if refit_collection is not None:
                level_curves = refit_collection.partial_effect_curves_for(level)
                if level_curves:
                    stacked = np.asarray(level_curves)  # (n_refits, len(age_grid))
                    lower = np.percentile(stacked, 2.5, axis=0)
                    upper = np.percentile(stacked, 97.5, axis=0)
                    bands = {i: (float(lower[i]), float(upper[i])) for i in range(len(age_grid))}
            for index, row in enumerate(curve):
                lower_upper = bands.get(index)
                partial_effect_rows.append(
                    {
                        "endpoint": endpoint,
                        "centre": centre,
                        "model": model_name,
                        "presentation_level": row.presentation_level or "",
                        "age_value": row.age_value,
                        "effect": row.effect,
                        "ci_lower": lower_upper[0] if lower_upper else None,
                        "ci_upper": lower_upper[1] if lower_upper else None,
                    }
                )

    scope_rows.append(
        {
            "endpoint": endpoint,
            "centre": centre,
            "model": model_name,
            "explanation_tier": registration.tier,
            "source_registry": registration.source_registry,
            "analysis_status": registration.analysis_status,
            "explanation_status": "explained",
            "reason": "",
            "n_eligible": len(entry),
            "n_events": n_events,
            "n_event_free": n_event_free,
        }
    )
    artifacts = _ScopeArtifacts(
        endpoint=endpoint,
        centre=centre,
        model_name=model_name,
        tier=registration.tier,
        analysis_status=registration.analysis_status,
        importance_by_group_730=importance_by_group_730,
        coefficients_by_group=coefficients_by_group,
        risk_365=prediction.unconditional[:, IDX_365],
        risk_730=prediction.unconditional[:, IDX_730],
    )
    for horizon, values in (
        (365.0, primary_refit.unconditional_365 if primary_refit is not None else []),
        (730.0, primary_refit.unconditional_730 if primary_refit is not None else []),
    ):
        summary = summarize_prediction_stability(endpoint, centre, model_name, horizon, values)
        prediction_stability_rows.append(
            {
                "endpoint": summary.endpoint,
                "centre": summary.centre,
                "model": summary.model_name,
                "horizon": summary.horizon,
                "n_patients": summary.n_patients,
                "n_refits": summary.n_refits,
                "median_interval_width": summary.median_interval_width,
                "q1_interval_width": summary.q1_interval_width,
                "q3_interval_width": summary.q3_interval_width,
                "proportion_exceeding_010": summary.proportion_exceeding_010,
                "proportion_exceeding_020": summary.proportion_exceeding_020,
            }
        )

    return (
        scope_rows,
        coefficient_rows,
        importance_rows,
        selection_rows,
        partial_effect_rows,
        prediction_stability_rows,
        coefficient_path_rows,
        bootstrap_attribution_rows,
        artifacts,
    )


def _build_comparability_attribution_rows(
    artifacts_by_endpoint_model: dict[tuple[str, str], dict[str, _ScopeArtifacts]],
    *,
    endpoints: tuple[str, ...],
    centres: tuple[str, ...],
    model_names: tuple[str, ...],
) -> list[dict]:
    """Cross-centre (R1-R4), cross-endpoint, and cross-model attribution-stability rows -- built
    only from each scope's aggregate `importance_by_group_730`/`coefficients_by_group`, never from
    per-patient predictions, so this is safe to call from either the single-process workflow or the
    per-model-split aggregation workflow."""
    attribution_rows: list[dict] = []

    for (endpoint, model_name), scope_map in artifacts_by_endpoint_model.items():
        for i, centre_a in enumerate(centres):
            for centre_b in centres[i + 1 :]:
                row = build_cross_centre_row(
                    model_name,
                    centre_a,
                    centre_b,
                    available_a=centre_a in scope_map,
                    available_b=centre_b in scope_map,
                    importance_a=scope_map[centre_a].importance_by_group_730
                    if centre_a in scope_map
                    else None,
                    importance_b=scope_map[centre_b].importance_by_group_730
                    if centre_b in scope_map
                    else None,
                    coefficients_a=scope_map[centre_a].coefficients_by_group
                    if centre_a in scope_map
                    else None,
                    coefficients_b=scope_map[centre_b].coefficients_by_group
                    if centre_b in scope_map
                    else None,
                )
                attribution_rows.append({**vars(row), "endpoint": endpoint})

    if "OS" in endpoints and "TTNT-D" in endpoints:
        for model_name in model_names:
            os_map = artifacts_by_endpoint_model.get(("OS", model_name), {})
            ttnt_map = artifacts_by_endpoint_model.get(("TTNT-D", model_name), {})
            for centre in centres:
                if centre in os_map and centre in ttnt_map:
                    row = build_cross_endpoint_row(
                        model_name,
                        centre,
                        os_map[centre].importance_by_group_730,
                        ttnt_map[centre].importance_by_group_730,
                        os_map[centre].coefficients_by_group,
                        ttnt_map[centre].coefficients_by_group,
                    )
                    attribution_rows.append({**vars(row), "endpoint": "OS_vs_TTNT-D"})

    for endpoint in endpoints:
        minimal_map = artifacts_by_endpoint_model.get((endpoint, "minimal_cox"), {})
        for model_name in model_names:
            if model_name == "minimal_cox":
                continue
            model_map = artifacts_by_endpoint_model.get((endpoint, model_name), {})
            for centre in centres:
                if centre in model_map and centre in minimal_map:
                    row = build_cross_model_row(
                        endpoint,
                        centre,
                        model_name,
                        model_map[centre].importance_by_group_730,
                        minimal_map[centre].importance_by_group_730,
                    )
                    attribution_rows.append({**vars(row), "endpoint": endpoint})

    return attribution_rows


def _build_cross_family_rows(
    artifacts_by_endpoint_model: dict[tuple[str, str], dict[str, _ScopeArtifacts]],
    scope_rows: list[dict],
    *,
    endpoints: tuple[str, ...],
    centres: tuple[str, ...],
) -> list[dict]:
    """`cross_family_prediction_spread.csv` (E6): a property of the (endpoint, centre, horizon)
    model *set*, computed from each scope's `risk_365`/`risk_730` -- callers must populate those
    in-memory only (never read back from disk), matching the project's no-persisted-per-patient-data
    rule."""
    reason_by_scope: dict[tuple[str, str, str], str] = {
        (row["endpoint"], row["centre"], row["model"]): row.get("reason", "") for row in scope_rows
    }
    by_endpoint_centre: dict[tuple[str, str], dict[str, _ScopeArtifacts]] = {}
    for (endpoint, model_name), scope_map in artifacts_by_endpoint_model.items():
        for centre, artifacts_obj in scope_map.items():
            by_endpoint_centre.setdefault((endpoint, centre), {})[model_name] = artifacts_obj

    def _modal_model_set(sets: list[frozenset[str]]) -> frozenset[str]:
        counts: dict[frozenset[str], int] = {}
        for s in sets:
            counts[s] = counts.get(s, 0) + 1
        best_count = max(counts.values())
        candidates = [s for s, c in counts.items() if c == best_count]
        # Deterministic tie-break: prefer the larger (more "complete") set, then the
        # alphabetically-latest member tuple -- never chosen based on which result looked
        # more favourable.
        return max(candidates, key=lambda s: (len(s), sorted(s)))

    modal_set_by_endpoint: dict[str, frozenset[str]] = {}
    for endpoint in endpoints:
        sets = [frozenset(by_endpoint_centre.get((endpoint, centre), {})) for centre in centres]
        modal_set_by_endpoint[endpoint] = _modal_model_set(sets)

    cross_family_rows: list[dict] = []
    for endpoint in endpoints:
        modal_set = modal_set_by_endpoint[endpoint]
        for centre in centres:
            model_map = by_endpoint_centre.get((endpoint, centre), {})
            included = sorted(model_map)
            excluded = sorted(modal_set - set(model_map))
            model_family_status = "complete" if not excluded else "restricted"
            exclusion_reasons = [reason_by_scope.get((endpoint, centre, m), "") for m in excluded]
            for horizon, risk_attr in ((365.0, "risk_365"), (730.0, "risk_730")):
                model_risk_by_patient = {
                    m: getattr(a, risk_attr)
                    for m, a in model_map.items()
                    if getattr(a, risk_attr) is not None
                }
                median_spread, p90_spread = cross_family_spread(model_risk_by_patient)
                cross_family_rows.append(
                    {
                        "endpoint": endpoint,
                        "centre": centre,
                        "horizon": horizon,
                        "patient_scope_basis": SCOPE_BASIS_IDENTICAL,
                        "model_family_status": model_family_status,
                        "availability_confounded": model_family_status == "restricted",
                        "included_models": ",".join(included),
                        "excluded_models": ",".join(excluded),
                        "exclusion_reasons": ";".join(exclusion_reasons),
                        "median_spread": median_spread,
                        "p90_spread": p90_spread,
                    }
                )
    return cross_family_rows


def _not_explained_scope_rows(endpoints: tuple[str, ...], centres: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for registration in MODEL_REGISTER:
        if registration.model_name in NOT_EXPLAINED_MODEL_NAMES:
            for endpoint in endpoints:
                for centre in centres:
                    rows.append(
                        {
                            "endpoint": endpoint,
                            "centre": centre,
                            "model": registration.model_name,
                            "explanation_tier": registration.tier,
                            "source_registry": registration.source_registry,
                            "analysis_status": "",
                            "explanation_status": "not_explained",
                            "reason": registration.not_explained_reason,
                            "fit_success_count": registration.not_explained_fit_success_count,
                            "fit_total_count": registration.not_explained_fit_total_count,
                        }
                    )
    return rows


def _model_availability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": r.model_name,
                "tier": r.tier,
                "source_registry": r.source_registry,
                "feature_set": r.feature_set,
                "runtime": r.runtime,
                "not_explained_reason": r.not_explained_reason or "",
                "fit_success_count": r.not_explained_fit_success_count,
                "fit_total_count": r.not_explained_fit_total_count,
            }
            for r in MODEL_REGISTER
        ]
    )


def _comparability_matrix_frame(attribution_stability: pd.DataFrame) -> pd.DataFrame:
    if len(attribution_stability):
        return attribution_stability[
            [
                c
                for c in attribution_stability.columns
                if c
                not in (
                    "kendall_tau",
                    "top_k",
                    "topk_overlap",
                    "jaccard",
                    "top_k_boundary_tied",
                    "sign_agreement",
                    "n_groups",
                )
            ]
        ]
    return pd.DataFrame(
        columns=[
            "comparison_type",
            "model_name",
            "key_a",
            "key_b",
            "scope_basis",
            "comparison_status",
        ]
    )


def _read_model_explanation_csv(path: Path) -> pd.DataFrame:
    """`pd.DataFrame([]).to_csv()` round-trips as a genuinely empty file (no header row), which
    `pd.read_csv` rejects with `EmptyDataError` -- a real case here, since several per-model frames
    are legitimately empty for a given model (e.g. `selection_stability` for non-sparsity-capable
    models, `coefficient_paths` for non-grid models)."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _publish_model_explanation(
    model_directory: Path, result: ModelExplanationResult
) -> ModelExplanationArtifacts:
    """Atomic per-model publish: build in a temp dir, write a `_SUCCESS` marker last, then rename
    into place. If the process dies mid-write, `model_directory` is simply never created/updated, so
    `aggregate_explainability_results` correctly treats that model as not-yet-available."""
    model_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".phase6-model-{result.model_name}-", dir=model_directory.parent)
    )
    try:
        frames = {
            SCOPE_STATUS: result.scope_status,
            COEFFICIENTS: result.coefficient_summary,
            IMPORTANCE: result.importance_summary,
            SELECTION_STABILITY: result.selection_stability,
            PARTIAL_EFFECTS: result.partial_effects,
            PREDICTION_STABILITY: result.prediction_stability,
            COEFFICIENT_PATHS: result.coefficient_paths,
            BOOTSTRAP_ATTRIBUTION_STABILITY: result.bootstrap_attribution_stability,
            MODEL_EXPLANATION_SCOPE_SUMMARY: result.scope_summary,
        }
        for name, frame in frames.items():
            frame.to_csv(temporary / name, index=False)
        (temporary / MODEL_EXPLANATION_SUCCESS_MARKER).write_text(
            json.dumps(
                {
                    "model": result.model_name,
                    "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if model_directory.exists():
            shutil.rmtree(model_directory)
        temporary.rename(model_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ModelExplanationArtifacts(
        model_directory,
        {name: model_directory / name for name in frames}
        | {MODEL_EXPLANATION_SUCCESS_MARKER: model_directory / MODEL_EXPLANATION_SUCCESS_MARKER},
    )


def run_model_explainability(
    analysis_directory: str | Path,
    feature_manifest_path: str | Path,
    model_name: str,
    intermediate_directory: str | Path,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    centres: tuple[str, ...] = CENTRES,
    r_perm: int = R_PERM,
    b_explain: int = B_EXPLAIN,
    r_boot: int = R_BOOT,
    progress=None,
) -> ModelExplanationResult:
    """Model-level step of the per-model split: explain one registered model across every (endpoint,
    centre)
    scope and persist only aggregate-level results -- never per-patient predictions, which stay
    in-memory-only -- to `intermediate_directory/<model_name>/`. Running one model per OS process
    means a crash mid-run only loses that model's share of the full inventory, and
    `aggregate_explainability_results` can combine however many models actually finished.
    `progress`, if given, is called as `progress(endpoint, centre, model_name, explanation_status)`
    after each scope."""
    analysis_root = Path(analysis_directory).resolve()
    feature_manifest = Path(feature_manifest_path).resolve()
    paths = verify_all_inputs(analysis_root, feature_manifest)

    scope_rows, coefficient_rows, importance_rows = [], [], []
    selection_rows, partial_effect_rows, prediction_stability_rows = [], [], []
    coefficient_path_rows, bootstrap_attribution_rows = [], []
    scope_summary_rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix=f"crc-phase6-explain-{model_name}-") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        for endpoint in endpoints:
            view = build_endpoint_view(paths.frame, endpoint)
            iecv_by_centre = {s.held_out_institution: s for s in build_iecv_splits(view)}
            for centre in centres:
                split = iecv_by_centre[centre]
                (
                    new_scope,
                    new_coef,
                    new_importance,
                    new_selection,
                    new_partial,
                    new_prediction_stability,
                    new_coefficient_path,
                    new_bootstrap_attribution,
                    artifacts_obj,
                ) = _explain_one_scope(
                    paths,
                    view,
                    split,
                    model_name,
                    endpoint=endpoint,
                    centre=centre,
                    r_perm=r_perm,
                    b_explain=b_explain,
                    r_boot=r_boot,
                    tmp_root=tmp_root,
                )
                scope_rows.extend(new_scope)
                coefficient_rows.extend(new_coef)
                importance_rows.extend(new_importance)
                selection_rows.extend(new_selection)
                partial_effect_rows.extend(new_partial)
                prediction_stability_rows.extend(new_prediction_stability)
                coefficient_path_rows.extend(new_coefficient_path)
                bootstrap_attribution_rows.extend(new_bootstrap_attribution)
                if artifacts_obj is not None:
                    for group_name, value in artifacts_obj.importance_by_group_730.items():
                        scope_summary_rows.append(
                            {
                                "endpoint": endpoint,
                                "centre": centre,
                                "feature_group": group_name,
                                "kind": "importance_730",
                                "value": value,
                            }
                        )
                    for group_name, value in artifacts_obj.coefficients_by_group.items():
                        scope_summary_rows.append(
                            {
                                "endpoint": endpoint,
                                "centre": centre,
                                "feature_group": group_name,
                                "kind": "coefficient",
                                "value": value,
                            }
                        )
                if progress is not None:
                    status = new_scope[0]["explanation_status"] if new_scope else "unknown"
                    progress(endpoint, centre, model_name, status)

    result = ModelExplanationResult(
        model_name=model_name,
        scope_status=pd.DataFrame(scope_rows),
        coefficient_summary=pd.DataFrame(coefficient_rows),
        importance_summary=pd.DataFrame(importance_rows),
        selection_stability=pd.DataFrame(selection_rows),
        partial_effects=pd.DataFrame(partial_effect_rows),
        prediction_stability=pd.DataFrame(prediction_stability_rows),
        coefficient_paths=pd.DataFrame(coefficient_path_rows),
        bootstrap_attribution_stability=pd.DataFrame(bootstrap_attribution_rows),
        scope_summary=pd.DataFrame(scope_summary_rows),
    )
    result.artifacts = _publish_model_explanation(
        Path(intermediate_directory).resolve() / model_name, result
    )
    return result


def aggregate_explainability_results(
    analysis_directory: str | Path,
    feature_manifest_path: str | Path,
    intermediate_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    centres: tuple[str, ...] = CENTRES,
    model_names: tuple[str, ...] = EXPLAINED_MODEL_NAMES,
) -> ExplainabilityResult:
    """Aggregation step of the per-model split: combine however many of `model_names` have a
    completed
    `run_model_explainability` result under `intermediate_directory` into the cross-model
    comparability artifacts (which need more than one model's results at once), then atomically
    publish the same final bundle `run_explainability_workflow` would have produced -- restricted to
    whichever models actually finished. Per-patient risk predictions are recomputed fresh in memory
    for this step (a cheap point-model reload, not a refit) rather than ever being persisted between
    workflows."""
    analysis_root = Path(analysis_directory).resolve()
    feature_manifest = Path(feature_manifest_path).resolve()
    try:
        paths = verify_all_inputs(analysis_root, feature_manifest)
        intermediate = Path(intermediate_directory).resolve()

        available_models = [
            m for m in model_names if (intermediate / m / MODEL_EXPLANATION_SUCCESS_MARKER).exists()
        ]
        if not available_models:
            raise ExplainabilityWorkflowError(
                f"No completed model-phase results found under {intermediate}"
            )
        missing_models = sorted(set(model_names) - set(available_models))
        if missing_models:
            print(
                f"WARNING: aggregating only {len(available_models)}/{len(model_names)} models -- "
                f"missing (not yet computed): {missing_models}",
                flush=True,
            )

        scope_rows: list[dict] = _not_explained_scope_rows(endpoints, centres)
        coefficient_rows: list[dict] = []
        importance_rows: list[dict] = []
        selection_rows: list[dict] = []
        partial_effect_rows: list[dict] = []
        prediction_stability_rows: list[dict] = []
        coefficient_path_rows: list[dict] = []
        bootstrap_attribution_stability_rows: list[dict] = []

        aggregate_by_model: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
        for model_name in available_models:
            model_dir = intermediate / model_name
            for row in _read_model_explanation_csv(model_dir / SCOPE_STATUS).to_dict("records"):
                # `reason` is always a string (possibly "") in the single-process path; CSV
                # round-tripping an empty string reads back as NaN, which would otherwise diverge
                # from what a direct `run_explainability_workflow` call publishes for this column.
                if pd.isna(row.get("reason")):
                    row["reason"] = ""
                scope_rows.append(row)
            coefficient_rows.extend(
                _read_model_explanation_csv(model_dir / COEFFICIENTS).to_dict("records")
            )
            importance_rows.extend(
                _read_model_explanation_csv(model_dir / IMPORTANCE).to_dict("records")
            )
            selection_rows.extend(
                _read_model_explanation_csv(model_dir / SELECTION_STABILITY).to_dict("records")
            )
            partial_effect_rows.extend(
                _read_model_explanation_csv(model_dir / PARTIAL_EFFECTS).to_dict("records")
            )
            prediction_stability_rows.extend(
                _read_model_explanation_csv(model_dir / PREDICTION_STABILITY).to_dict("records")
            )
            coefficient_path_rows.extend(
                _read_model_explanation_csv(model_dir / COEFFICIENT_PATHS).to_dict("records")
            )
            bootstrap_attribution_stability_rows.extend(
                _read_model_explanation_csv(model_dir / BOOTSTRAP_ATTRIBUTION_STABILITY).to_dict(
                    "records"
                )
            )
            summary = _read_model_explanation_csv(model_dir / MODEL_EXPLANATION_SCOPE_SUMMARY)
            by_scope: dict[tuple[str, str], dict] = {}
            if len(summary):
                for (endpoint, centre), group_df in summary.groupby(["endpoint", "centre"]):
                    importance_slice = group_df[group_df["kind"] == "importance_730"]
                    coefficient_slice = group_df[group_df["kind"] == "coefficient"]
                    by_scope[(endpoint, centre)] = {
                        "importance_by_group_730": dict(
                            zip(
                                importance_slice["feature_group"],
                                importance_slice["value"],
                                strict=True,
                            )
                        ),
                        "coefficients_by_group": dict(
                            zip(
                                coefficient_slice["feature_group"],
                                coefficient_slice["value"],
                                strict=True,
                            )
                        ),
                    }
            aggregate_by_model[model_name] = by_scope

        # Fresh, in-memory-only reload of each explained scope's point prediction, purely to get
        # per-patient risk_365/risk_730 for cross_family_prediction_spread.csv -- never persisted
        # between the model workflow and this aggregation workflow (aggregate-safety rule, J).
        explained = {
            (row["endpoint"], row["centre"], row["model"])
            for row in scope_rows
            if row.get("explanation_status") == "explained"
        }
        artifacts_by_endpoint_model: dict[tuple[str, str], dict[str, _ScopeArtifacts]] = {}
        for endpoint in endpoints:
            view = build_endpoint_view(paths.frame, endpoint)
            iecv_by_centre = {s.held_out_institution: s for s in build_iecv_splits(view)}
            for model_name in available_models:
                scope_map: dict[str, _ScopeArtifacts] = {}
                agg_by_scope = aggregate_by_model.get(model_name, {})
                for centre in centres:
                    if (endpoint, centre, model_name) not in explained:
                        continue
                    split = iecv_by_centre[centre]
                    try:
                        loaded = load_model(paths, split, model_name)
                    except ModelNotFitted:
                        continue
                    _, assessment_frame = split.frames(view)
                    _, _, prediction = predict_assessment(loaded, assessment_frame, IBS_GRID)
                    agg = agg_by_scope.get(
                        (endpoint, centre),
                        {"importance_by_group_730": {}, "coefficients_by_group": {}},
                    )
                    scope_map[centre] = _ScopeArtifacts(
                        endpoint=endpoint,
                        centre=centre,
                        model_name=model_name,
                        tier="",
                        analysis_status="",
                        importance_by_group_730=agg["importance_by_group_730"],
                        coefficients_by_group=agg["coefficients_by_group"],
                        risk_365=prediction.unconditional[:, IDX_365],
                        risk_730=prediction.unconditional[:, IDX_730],
                    )
                artifacts_by_endpoint_model[(endpoint, model_name)] = scope_map

        attribution_rows = _build_comparability_attribution_rows(
            artifacts_by_endpoint_model,
            endpoints=endpoints,
            centres=centres,
            model_names=tuple(available_models),
        )
        cross_family_rows = _build_cross_family_rows(
            artifacts_by_endpoint_model, scope_rows, endpoints=endpoints, centres=centres
        )

        scope_status = pd.DataFrame(scope_rows)
        model_availability = _model_availability_frame()
        coefficient_summary = pd.DataFrame(coefficient_rows)
        importance_summary = pd.DataFrame(importance_rows)
        selection_stability = pd.DataFrame(selection_rows)
        partial_effects = pd.DataFrame(partial_effect_rows)
        attribution_stability = pd.DataFrame(attribution_rows)
        prediction_stability = pd.DataFrame(prediction_stability_rows)
        cross_family_prediction_spread = pd.DataFrame(cross_family_rows)
        coefficient_paths = pd.DataFrame(coefficient_path_rows)
        bootstrap_attribution_stability = pd.DataFrame(bootstrap_attribution_stability_rows)
        comparability_matrix = _comparability_matrix_frame(attribution_stability)

        output = (
            Path(output_directory).resolve()
            if output_directory
            else analysis_root / "explainability"
        )
        artifacts = _publish(
            output,
            scope_status,
            model_availability,
            coefficient_summary,
            importance_summary,
            selection_stability,
            partial_effects,
            attribution_stability,
            prediction_stability,
            cross_family_prediction_spread,
            coefficient_paths,
            bootstrap_attribution_stability,
            comparability_matrix,
        )
        reverify_frozen_primary_unchanged(analysis_root, feature_manifest)
        return ExplainabilityResult(
            scope_status,
            model_availability,
            coefficient_summary,
            importance_summary,
            selection_stability,
            partial_effects,
            attribution_stability,
            prediction_stability,
            cross_family_prediction_spread,
            coefficient_paths,
            bootstrap_attribution_stability,
            comparability_matrix,
            artifacts,
        )
    except ExplainabilityWorkflowError:
        raise
    except Exception as exc:
        raise ExplainabilityWorkflowError(
            "Phase 6 explainability aggregation phase failed"
        ) from exc


def run_explainability_workflow(
    analysis_directory: str | Path,
    feature_manifest_path: str | Path,
    output_directory: str | Path | None = None,
    *,
    endpoints: tuple[str, ...] = ("OS", "TTNT-D"),
    model_names: tuple[str, ...] = EXPLAINED_MODEL_NAMES,
    centres: tuple[str, ...] = CENTRES,
    r_perm: int = R_PERM,
    b_explain: int = B_EXPLAIN,
    r_boot: int = R_BOOT,
) -> ExplainabilityResult:
    """Explain every registered model on the untouched IECV assessment rows of its scope.

    `model_names`/`centres`/`r_perm`/`b_explain`/`r_boot` narrow the workload for fast synthetic
    exercises of this boundary; the notebook must call this with every default left untouched.
    """
    analysis_root = Path(analysis_directory).resolve()
    feature_manifest = Path(feature_manifest_path).resolve()
    try:
        paths = verify_all_inputs(analysis_root, feature_manifest)

        scope_rows: list[dict] = _not_explained_scope_rows(endpoints, centres)
        coefficient_rows: list[dict] = []
        importance_rows: list[dict] = []
        selection_rows: list[dict] = []
        partial_effect_rows: list[dict] = []
        prediction_stability_rows: list[dict] = []
        coefficient_path_rows: list[dict] = []
        bootstrap_attribution_stability_rows: list[dict] = []

        artifacts_by_endpoint_model: dict[tuple[str, str], dict[str, _ScopeArtifacts]] = {}

        with tempfile.TemporaryDirectory(prefix="crc-phase6-explain-") as tmp_root_str:
            tmp_root = Path(tmp_root_str)
            for endpoint in endpoints:
                view = build_endpoint_view(paths.frame, endpoint)
                iecv_by_centre = {
                    split.held_out_institution: split for split in build_iecv_splits(view)
                }
                for model_name in model_names:
                    scope_map: dict[str, _ScopeArtifacts] = {}
                    for centre in centres:
                        split = iecv_by_centre[centre]
                        (
                            new_scope,
                            new_coef,
                            new_importance,
                            new_selection,
                            new_partial,
                            new_prediction_stability,
                            new_coefficient_path,
                            new_bootstrap_attribution,
                            artifacts_obj,
                        ) = _explain_one_scope(
                            paths,
                            view,
                            split,
                            model_name,
                            endpoint=endpoint,
                            centre=centre,
                            r_perm=r_perm,
                            b_explain=b_explain,
                            r_boot=r_boot,
                            tmp_root=tmp_root,
                        )
                        scope_rows.extend(new_scope)
                        coefficient_rows.extend(new_coef)
                        importance_rows.extend(new_importance)
                        selection_rows.extend(new_selection)
                        partial_effect_rows.extend(new_partial)
                        prediction_stability_rows.extend(new_prediction_stability)
                        coefficient_path_rows.extend(new_coefficient_path)
                        bootstrap_attribution_stability_rows.extend(new_bootstrap_attribution)
                        if artifacts_obj is not None:
                            scope_map[centre] = artifacts_obj
                    artifacts_by_endpoint_model[(endpoint, model_name)] = scope_map

            attribution_rows = _build_comparability_attribution_rows(
                artifacts_by_endpoint_model,
                endpoints=endpoints,
                centres=centres,
                model_names=model_names,
            )
            cross_family_rows = _build_cross_family_rows(
                artifacts_by_endpoint_model, scope_rows, endpoints=endpoints, centres=centres
            )

        scope_status = pd.DataFrame(scope_rows)
        model_availability = _model_availability_frame()
        coefficient_summary = pd.DataFrame(coefficient_rows)
        importance_summary = pd.DataFrame(importance_rows)
        selection_stability = pd.DataFrame(selection_rows)
        partial_effects = pd.DataFrame(partial_effect_rows)
        attribution_stability = pd.DataFrame(attribution_rows)
        prediction_stability = pd.DataFrame(prediction_stability_rows)
        cross_family_prediction_spread = pd.DataFrame(cross_family_rows)
        coefficient_paths = pd.DataFrame(coefficient_path_rows)
        bootstrap_attribution_stability = pd.DataFrame(bootstrap_attribution_stability_rows)
        comparability_matrix = _comparability_matrix_frame(attribution_stability)

        output = (
            Path(output_directory).resolve()
            if output_directory
            else analysis_root / "explainability"
        )
        artifacts = _publish(
            output,
            scope_status,
            model_availability,
            coefficient_summary,
            importance_summary,
            selection_stability,
            partial_effects,
            attribution_stability,
            prediction_stability,
            cross_family_prediction_spread,
            coefficient_paths,
            bootstrap_attribution_stability,
            comparability_matrix,
        )
        reverify_frozen_primary_unchanged(analysis_root, feature_manifest)
        return ExplainabilityResult(
            scope_status,
            model_availability,
            coefficient_summary,
            importance_summary,
            selection_stability,
            partial_effects,
            attribution_stability,
            prediction_stability,
            cross_family_prediction_spread,
            coefficient_paths,
            bootstrap_attribution_stability,
            comparability_matrix,
            artifacts,
        )
    except ExplainabilityWorkflowError:
        raise
    except Exception as exc:
        raise ExplainabilityWorkflowError("Phase 6 explainability workflow failed") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish(
    output: Path,
    scope_status,
    model_availability,
    coefficient_summary,
    importance_summary,
    selection_stability,
    partial_effects,
    attribution_stability,
    prediction_stability,
    cross_family_prediction_spread,
    coefficient_paths,
    bootstrap_attribution_stability,
    comparability_matrix,
) -> ExplainabilityArtifacts:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".phase6-explainability-", dir=output.parent))
    backup = output.parent / f".{output.name}-previous"
    try:
        frames = {
            SCOPE_STATUS: scope_status,
            MODEL_AVAILABILITY: model_availability,
            COEFFICIENTS: coefficient_summary,
            IMPORTANCE: importance_summary,
            SELECTION_STABILITY: selection_stability,
            PARTIAL_EFFECTS: partial_effects,
            ATTRIBUTION_STABILITY: attribution_stability,
            PREDICTION_STABILITY: prediction_stability,
            CROSS_FAMILY_PREDICTION_SPREAD: cross_family_prediction_spread,
            COEFFICIENT_PATHS: coefficient_paths,
            BOOTSTRAP_ATTRIBUTION_STABILITY: bootstrap_attribution_stability,
            COMPARABILITY_MATRIX: comparability_matrix,
        }
        for name, frame in frames.items():
            frame.to_csv(temporary / name, index=False)
        (temporary / FIGURES).mkdir()

        report = _render_report(scope_status, model_availability)
        (temporary / REPORT).write_text(report, encoding="utf-8")

        hashable_names = tuple(frames) + (REPORT,)
        hashes = {name: _sha256(temporary / name) for name in hashable_names}
        payload: dict[str, object] = {
            "phase": "phase-6-explainability",
            "analysis_status_note": ANALYSIS_STATUS_POST_HOC_EXPLORATORY,
            "artifact_hashes": hashes,
            "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["manifest_payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        (temporary / MANIFEST).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            output.rename(backup)
        temporary.rename(output)
    except Exception as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        if not output.exists() and backup.exists():
            backup.rename(output)
        raise ExplainabilityWorkflowError("Phase 6 atomic publication failed") from exc

    try:
        published_names = {path.name for path in output.iterdir()}
        if published_names != set(ARTIFACT_NAMES):
            raise ExplainabilityWorkflowError("Published Phase 6 bundle has unexpected contents")
    except Exception as exc:
        rolled_back = output.parent / f".{output.name}-rolledback"
        if rolled_back.exists():
            shutil.rmtree(rolled_back)
        output.rename(rolled_back)
        if backup.exists():
            backup.rename(output)
        shutil.rmtree(rolled_back, ignore_errors=True)
        raise ExplainabilityWorkflowError("Phase 6 post-publication verification failed") from exc

    if backup.exists():
        shutil.rmtree(backup)
    return ExplainabilityArtifacts(output, {name: output / name for name in ARTIFACT_NAMES})


def _render_report(scope_status: pd.DataFrame, model_availability: pd.DataFrame) -> str:
    status_counts = (
        scope_status["explanation_status"].value_counts().to_dict() if len(scope_status) else {}
    )
    return (
        "# Phase 6 explainability report\n\n"
        "Explanations describe model behaviour and prognostic association only -- never "
        "biological mechanism, causation, or treatment effect.\n\n"
        "## Model availability hierarchy\n\n"
        f"{model_availability.to_string(index=False) if len(model_availability) else '(empty)'}\n\n"
        "## Explanation scope status counts\n\n"
        f"{status_counts}\n\n"
        f"Primary explanation horizon: {PRIMARY_EXPLANATION_HORIZON:g} days.\n\n"
        "## Phase 7 handoff\n\nPhase 7 owns recalibration, calibration drift, local adaptation, "
        "the sequencing-lag curve, sensitivity analyses, and manuscript artifacts. No such result "
        "was produced here.\n"
    )


__all__ = [
    "ExplainabilityArtifacts",
    "ExplainabilityResult",
    "ExplainabilityWorkflowError",
    "ModelExplanationArtifacts",
    "ModelExplanationResult",
    "aggregate_explainability_results",
    "run_model_explainability",
    "run_explainability_workflow",
]
