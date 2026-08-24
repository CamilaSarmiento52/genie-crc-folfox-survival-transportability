"""Bootstrap refits (`explainability-spec.md`, "Bootstrap refits"): `B_EXPLAIN = 200` patient-level
resamples of the outer **development** set, refit at the already-selected configuration -- no
re-tuning inside a refit. Feeds bootstrap-derived coefficient/importance/partial-effect/
prediction-stability uncertainty for every explained model.

The refit seed family is `(root_seed, endpoint, centre, model, bootstrap_id, purpose=
"explain_bootstrap_refit")` -- deliberately distinct from primary-evaluation workflow's evaluation
bootstrap
(`crc_folfox.evaluate.bootstrap.bootstrap_seed`, which has no `model`/`bootstrap_id` and a
different purpose string), so a refit interval can never be described as paired with a
primary-evaluation workflow
evaluation interval. Each per-candidate refit function reuses the exact fit primitive its
model-training or optimization counterpart uses (`fit_cox`, `GlmnetAdapter.fit`) at the frozen
configuration, then wraps the
result in an ad-hoc in-memory `LoadedModel` purely to reuse `explain.loading`'s prediction path --
no registry, hash verification, or disk persistence is involved for a refit's own fitted state
beyond the R adapter's required temporary bundle file, deleted immediately after use.

**2026-08-11 restructuring** (`explainability-spec.md` amendment): each bootstrap draw now fits
once, transforms the fixed assessment frame once, predicts once, computes importance once (for
every canonical feature group, in one batched call for an R-backed candidate), and derives *every*
presentation-level partial-effect curve from that one fit -- the prior code refit the whole model
once per presentation level, redundantly. `RefitCollection` is now one immutable record per
bootstrap ID rather than several independent same-length lists appended to in lockstep; a failed
refit is a real, identifiable record (`fit_status="failed"`), not silently absent, so no consumer
can misalign one field's list against another's by index. Per-bootstrap-draw predictions are held
only long enough to fold into the aggregate statistics this module and its callers compute --
`explainability-spec.md`'s aggregate-safety rule prohibits ever writing a per-patient prediction to
a checkpoint file or published artifact, and none of this module's own state is ever serialized.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from crc_folfox.evaluate.contracts import IBS_GRID
from crc_folfox.explain.contracts import (
    B_EXPLAIN,
    R_BOOT,
    ROOT_SEED,
    RUNTIME_PYTHON,
    RUNTIME_R,
    ExplainContractError,
)
from crc_folfox.explain.feature_groups import align_groups_to_present_columns
from crc_folfox.explain.importance import IDX_365, IDX_730, compute_all_groups_importance_batched
from crc_folfox.explain.loading import (
    LoadedModel,
    loaded_feature_columns,
    predict_from_matrix,
    transform_assessment,
)
from crc_folfox.explain.partial_effects import compute_partial_effect
from crc_folfox.model_training.bundles import extract_cox_bundle_state
from crc_folfox.model_training.contracts import ModelTrainingContractError
from crc_folfox.model_training.cox import apply_spline_state, fit_cox, fit_spline_state

# Imported by reference, not redefined: this is the exact defect optimization workflow's Workstream
# A was
# built to fix for elastic-net tuning (`correction-spec.md`) -- a raw `lifelines.exceptions.
# ConvergenceError`/`LinAlgError` escaping an unguarded fit call aborts the whole caller instead
# of disqualifying one candidate attempt. A bootstrap refit is exactly such an attempt: on real
# data, a resampled stratum can produce a near-perfectly-separated rare category by chance
# (confirmed by a real Gate 7 canary run against `analysis/out/current/`), and that one refit
# must be recorded as failed, not allowed to abort the other `B_EXPLAIN - 1` refits.
from crc_folfox.model_training.selection import EXPECTED_TUNING_EXCEPTIONS
from crc_folfox.optimize.candidates import (
    add_presentation_age_spline_interaction,
    fit_reduced_core_preprocessor,
)
from crc_folfox.optimize.contracts import OptimizationContractError
from crc_folfox.optimize.r_adapter import GlmnetAdapter
from crc_folfox.preprocessing.contracts import PreprocessingContractError
from crc_folfox.preprocessing.transform import FittedPreprocessor

REFIT_PURPOSE = "explain_bootstrap_refit"
FIT_SUCCEEDED = "succeeded"
FIT_FAILED = "failed"
_METRICS = ("365", "730", "ibs")

_EXPECTED_REFIT_EXCEPTIONS = (
    *EXPECTED_TUNING_EXCEPTIONS,
    ModelTrainingContractError,
    ExplainContractError,
    OptimizationContractError,
    PreprocessingContractError,
)


def refit_seed(*, endpoint: str, centre: str, model_name: str, bootstrap_id: int) -> int:
    fields = (str(ROOT_SEED), endpoint, centre, model_name, str(bootstrap_id), REFIT_PURPOSE)
    return int(hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:8], 16)


def bootstrap_resample_frame(development_frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Patient-level resample stratified by institution and event status.

    Sampling with replacement structurally produces repeated `patient_key` values (the same
    patient drawn more than once), but `FittedPreprocessor.fit` requires patient-unique rows --
    a defensive check written for preprocessing workflow's own never-resampled callers, which this
    is not. Each
    resampled row is given a fresh, resample-local key (`{original_key}__boot{row_index}`) so it
    is refit as the independent draw a bootstrap resample actually is; the mapping back to a real
    patient is discarded immediately after refit (no bundle ever retains it -- see
    `_clear_patient_keys`), so this never weakens the patient-identifier privacy rule.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _key, group in development_frame.groupby(["institution", "event"], sort=True):
        idx = rng.integers(0, len(group), size=len(group))
        parts.append(group.iloc[idx])
    resampled = pd.concat(parts, ignore_index=True)
    resampled["patient_key"] = [
        f"{key}__boot{index}" for index, key in enumerate(resampled["patient_key"])
    ]
    return resampled


def _clear_patient_keys(preprocessor: FittedPreprocessor) -> FittedPreprocessor:
    return replace(preprocessor, fit_patient_keys=frozenset())


def _bundle_state_kwargs(resampled: pd.DataFrame, model_name: str) -> dict:
    return {
        "model_id": f"explain_refit|{model_name}",
        "design": "explain_refit",
        "held_out_institution": None,
        "repeat": None,
        "fold": None,
        "development_patient_count": len(resampled),
        "development_event_count": int(resampled["event"].sum()),
    }


def _refit_minimal_cox(resampled, *, endpoint, config, seed, tmp_dir):
    del config, seed, tmp_dir
    preprocessor = FittedPreprocessor.fit(
        resampled, endpoint=endpoint, feature_set="baseline", fit_scope="explain_refit"
    )
    transformed = preprocessor.transform(resampled)
    fitter = fit_cox(transformed.matrix, transformed.metadata, penalizer=0.0, l1_ratio=0.0)
    state = extract_cox_bundle_state(
        fitter,
        endpoint=endpoint,
        model_name="minimal_cox",
        feature_set="baseline",
        penalizer=0.0,
        l1_ratio=0.0,
        preprocessor=_clear_patient_keys(preprocessor),
        spline_state=None,
        **_bundle_state_kwargs(resampled, "minimal_cox"),
    )
    loaded = LoadedModel(
        "minimal_cox",
        RUNTIME_PYTHON,
        len(resampled),
        int(resampled["event"].sum()),
        cox_state=state,
    )
    return loaded, dict(zip(state.feature_columns, state.coefficients, strict=True))


def _refit_spline_cox(resampled, *, endpoint, config, seed, tmp_dir):
    del seed, tmp_dir
    preprocessor = FittedPreprocessor.fit(
        resampled, endpoint=endpoint, feature_set="core", fit_scope="explain_refit"
    )
    transformed = preprocessor.transform(resampled)
    spline_state = fit_spline_state(transformed.matrix)
    design = apply_spline_state(transformed.matrix, spline_state)
    fitter = fit_cox(
        design, transformed.metadata, penalizer=config["penalizer"], l1_ratio=config["l1_ratio"]
    )
    state = extract_cox_bundle_state(
        fitter,
        endpoint=endpoint,
        model_name="spline_cox",
        feature_set="core",
        penalizer=config["penalizer"],
        l1_ratio=config["l1_ratio"],
        preprocessor=_clear_patient_keys(preprocessor),
        spline_state=spline_state,
        **_bundle_state_kwargs(resampled, "spline_cox"),
    )
    loaded = LoadedModel(
        "spline_cox", RUNTIME_PYTHON, len(resampled), int(resampled["event"].sum()), cox_state=state
    )
    return loaded, dict(zip(state.feature_columns, state.coefficients, strict=True))


def _refit_elastic_net_cox(resampled, *, endpoint, config, seed, tmp_dir):
    del seed, tmp_dir
    preprocessor = FittedPreprocessor.fit(
        resampled, endpoint=endpoint, feature_set="core", fit_scope="explain_refit"
    )
    transformed = preprocessor.transform(resampled)
    fitter = fit_cox(
        transformed.matrix,
        transformed.metadata,
        penalizer=config["penalizer"],
        l1_ratio=config["l1_ratio"],
    )
    state = extract_cox_bundle_state(
        fitter,
        endpoint=endpoint,
        model_name="elastic_net_cox",
        feature_set="core",
        penalizer=config["penalizer"],
        l1_ratio=config["l1_ratio"],
        preprocessor=_clear_patient_keys(preprocessor),
        spline_state=None,
        **_bundle_state_kwargs(resampled, "elastic_net_cox"),
    )
    loaded = LoadedModel(
        "elastic_net_cox",
        RUNTIME_PYTHON,
        len(resampled),
        int(resampled["event"].sum()),
        cox_state=state,
    )
    return loaded, dict(zip(state.feature_columns, state.coefficients, strict=True))


def _refit_penalized_nonlinear_cox(resampled, *, endpoint, config, seed, tmp_dir):
    del seed, tmp_dir
    preprocessor = FittedPreprocessor.fit(
        resampled, endpoint=endpoint, feature_set="core", fit_scope="explain_refit"
    )
    transformed = preprocessor.transform(resampled)
    spline_state = fit_spline_state(transformed.matrix)
    splined = apply_spline_state(transformed.matrix, spline_state)
    design = add_presentation_age_spline_interaction(splined, spline_state)
    fitter = fit_cox(design, transformed.metadata, penalizer=config["penalizer"], l1_ratio=0.0)
    state = extract_cox_bundle_state(
        fitter,
        endpoint=endpoint,
        model_name="penalized_nonlinear_cox",
        feature_set="core",
        penalizer=config["penalizer"],
        l1_ratio=0.0,
        preprocessor=_clear_patient_keys(preprocessor),
        spline_state=spline_state,
        **_bundle_state_kwargs(resampled, "penalized_nonlinear_cox"),
    )
    loaded = LoadedModel(
        "penalized_nonlinear_cox",
        RUNTIME_PYTHON,
        len(resampled),
        int(resampled["event"].sum()),
        cox_state=state,
    )
    return loaded, dict(zip(state.feature_columns, state.coefficients, strict=True))


def _refit_reduced_core_penalized_cox(resampled, *, endpoint, config, seed, tmp_dir):
    del seed, tmp_dir
    preprocessor, reduction = fit_reduced_core_preprocessor(
        resampled, endpoint=endpoint, fit_scope="explain_refit"
    )
    transformed = preprocessor.transform(reduction.reduced_frame)
    fitter = fit_cox(
        transformed.matrix, transformed.metadata, penalizer=config["penalizer"], l1_ratio=0.0
    )
    state = extract_cox_bundle_state(
        fitter,
        endpoint=endpoint,
        model_name="reduced_core_penalized_cox",
        feature_set="core",
        penalizer=config["penalizer"],
        l1_ratio=0.0,
        preprocessor=_clear_patient_keys(preprocessor),
        spline_state=None,
        **_bundle_state_kwargs(resampled, "reduced_core_penalized_cox"),
    )
    loaded = LoadedModel(
        "reduced_core_penalized_cox",
        RUNTIME_PYTHON,
        len(resampled),
        int(resampled["event"].sum()),
        cox_state=state,
    )
    return loaded, dict(zip(state.feature_columns, state.coefficients, strict=True))


def _refit_glmnet(model_name: str):
    def _refit(resampled, *, endpoint, config, seed, tmp_dir):
        preprocessor = FittedPreprocessor.fit(
            resampled, endpoint=endpoint, feature_set="core", fit_scope="explain_refit"
        )
        transformed = preprocessor.transform(resampled)
        destination = Path(tmp_dir) / f"{model_name}_{seed}.rds"
        result = GlmnetAdapter().fit(
            transformed.matrix,
            transformed.metadata,
            transformed.matrix,
            alpha=config["alpha"],
            lambda_value=config["lambda"],
            seed=seed,
            horizons=IBS_GRID,
            destination=destination,
        )
        loaded = LoadedModel(
            model_name,
            RUNTIME_R,
            len(resampled),
            int(resampled["event"].sum()),
            r_bundle_path=destination,
            r_preprocessor=preprocessor,
        )
        return loaded, result.coefficients

    return _refit


_REFIT_DISPATCH: dict[str, Callable] = {
    "minimal_cox": _refit_minimal_cox,
    "spline_cox": _refit_spline_cox,
    "elastic_net_cox": _refit_elastic_net_cox,
    "penalized_nonlinear_cox": _refit_penalized_nonlinear_cox,
    "reduced_core_penalized_cox": _refit_reduced_core_penalized_cox,
    "glmnet_ridge_core": _refit_glmnet("glmnet_ridge_core"),
    "glmnet_enet_core": _refit_glmnet("glmnet_enet_core"),
}


@dataclass(frozen=True)
class BootstrapRefitRecord:
    """One bootstrap draw's complete result -- every field for one `bootstrap_id` lives on the
    same record, so no consumer can compare two fields at "the same index" that actually came from
    different resamples (the exact risk of the prior parallel-list `RefitCollection` shape).

    `predictions_365`/`predictions_730` are patient-level risk arrays held only in memory for the
    lifetime of this collection, purely to feed aggregate statistics
    (`prediction_stability.summarize_prediction_stability`, `cross_family_spread`) -- never
    serialized, checkpointed, or published (`explainability-spec.md`'s aggregate-safety rule).
    """

    bootstrap_id: int
    fit_status: str  # FIT_SUCCEEDED | FIT_FAILED
    coefficients_by_column: dict[str, float] | None
    importance_by_group: dict[str, dict[str, float | None]] | None  # group -> {365, 730, ibs}
    rank_by_group: dict[str, dict[str, float]] | None  # group -> {365, 730, ibs}, average ranks
    group_active_by_group: dict[str, bool] | None
    active_column_count_by_group: dict[str, int] | None
    predictions_365: np.ndarray | None
    predictions_730: np.ndarray | None
    partial_effect_curves: dict[str | None, list[float]] | None  # presentation_level -> curve


def _rank_by_metric(
    importance_by_group: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float]]:
    """Average (fractional) ranks per metric, descending by importance (rank 1 = most important),
    over every canonical group in this refit -- including `group_active=False` groups at their
    structural `0.0`, so ranks are always computed over the full canonical inventory, never a
    refit-specific subset (`explainability-spec.md`'s 2026-08-11 amendment)."""
    groups = list(importance_by_group.keys())
    result: dict[str, dict[str, float]] = {g: {} for g in groups}
    for metric in _METRICS:
        values = np.array([importance_by_group[g].get(metric) for g in groups], dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        # rankdata over -values gives rank 1 to the largest value; NaNs are excluded from ranking
        # (a non-finite importance in this refit contributes no rank information for that metric).
        ranks = np.full(len(groups), np.nan)
        ranks[finite] = rankdata(-values[finite], method="average")
        for group_name, rank in zip(groups, ranks, strict=True):
            if np.isfinite(rank):
                result[group_name][metric] = float(rank)
    return result


def run_bootstrap_refits(
    model_name: str,
    development_frame: pd.DataFrame,
    assessment_frame: pd.DataFrame,
    *,
    endpoint: str,
    centre: str,
    config: dict,
    canonical_groups: dict[str, tuple[str, ...]],
    b_explain: int = B_EXPLAIN,
    r_boot: int = R_BOOT,
    age_grid: np.ndarray | None = None,
    reference_age: float | None = None,
    presentation_levels: tuple[str | None, ...] = (None,),
    tmp_dir: Path | None = None,
) -> RefitCollection:
    """`b_explain` bootstrap draws of `development_frame`; each draw fits once, transforms
    `assessment_frame` once through that fit, predicts once, computes importance once for every
    `canonical_groups` entry, and -- when `age_grid` is given -- derives one partial-effect curve
    per entry of `presentation_levels` from that same fit (2026-08-11 restructuring, see module
    docstring).
    """
    if model_name not in _REFIT_DISPATCH:
        raise ExplainContractError(f"'{model_name}' has no refit path")
    refit_fn = _REFIT_DISPATCH[model_name]

    records: list[BootstrapRefitRecord] = []

    for bootstrap_id in range(b_explain):
        seed = refit_seed(
            endpoint=endpoint, centre=centre, model_name=model_name, bootstrap_id=bootstrap_id
        )
        resampled = bootstrap_resample_frame(development_frame, seed)
        try:
            loaded, coefficients = refit_fn(
                resampled, endpoint=endpoint, config=config, seed=seed, tmp_dir=tmp_dir
            )
            matrix, metadata = transform_assessment(loaded, assessment_frame)
            entry = metadata["entry"].to_numpy(dtype=float)
            prediction = predict_from_matrix(loaded, matrix, entry, IBS_GRID)

            present_columns = loaded_feature_columns(loaded)
            alignment = align_groups_to_present_columns(canonical_groups, present_columns)
            groups_for_importance = {
                name: a.active_columns for name, a in alignment.items() if a.group_active
            }
            computed_importance = compute_all_groups_importance_batched(
                loaded,
                matrix,
                metadata,
                groups_for_importance,
                endpoint=endpoint,
                centre=centre,
                r_perm=r_boot,
            )
            importance_by_group: dict[str, dict[str, float | None]] = {}
            for group_name, a in alignment.items():
                if not a.group_active:
                    importance_by_group[group_name] = {metric: 0.0 for metric in _METRICS}
                    continue
                result = computed_importance[group_name]
                value_365, _ = result.importance_365()
                value_730, _ = result.importance_730()
                value_ibs, _ = result.importance_ibs()
                importance_by_group[group_name] = {
                    "365": value_365,
                    "730": value_730,
                    "ibs": value_ibs,
                }

            curves: dict[str | None, list[float]] | None = None
            if age_grid is not None:
                if reference_age is None:
                    raise ExplainContractError("age_grid requires an explicit reference_age")
                curves = {}
                for level in presentation_levels:
                    curve = compute_partial_effect(
                        loaded, age_grid, reference_age=reference_age, presentation_level=level
                    )
                    curves[level] = [row.effect for row in curve]
        except _EXPECTED_REFIT_EXCEPTIONS:
            records.append(
                BootstrapRefitRecord(
                    bootstrap_id=bootstrap_id,
                    fit_status=FIT_FAILED,
                    coefficients_by_column=None,
                    importance_by_group=None,
                    rank_by_group=None,
                    group_active_by_group=None,
                    active_column_count_by_group=None,
                    predictions_365=None,
                    predictions_730=None,
                    partial_effect_curves=None,
                )
            )
            continue
        finally:
            if model_name in ("glmnet_ridge_core", "glmnet_enet_core"):
                bundle_path = Path(tmp_dir) / f"{model_name}_{seed}.rds"
                bundle_path.unlink(missing_ok=True)

        records.append(
            BootstrapRefitRecord(
                bootstrap_id=bootstrap_id,
                fit_status=FIT_SUCCEEDED,
                coefficients_by_column=dict(coefficients or {}),
                importance_by_group=importance_by_group,
                rank_by_group=_rank_by_metric(importance_by_group),
                group_active_by_group={name: a.group_active for name, a in alignment.items()},
                active_column_count_by_group={
                    name: len(a.active_columns) for name, a in alignment.items()
                },
                predictions_365=prediction.unconditional[:, IDX_365],
                predictions_730=prediction.unconditional[:, IDX_730],
                partial_effect_curves=curves,
            )
        )

    return RefitCollection(model_name=model_name, n_attempted=b_explain, records=tuple(records))


@dataclass(frozen=True)
class RefitCollection:
    model_name: str
    n_attempted: int
    records: tuple[BootstrapRefitRecord, ...]

    @property
    def n_succeeded(self) -> int:
        return sum(1 for r in self.records if r.fit_status == FIT_SUCCEEDED)

    @property
    def finite_fraction(self) -> float:
        return self.n_succeeded / self.n_attempted if self.n_attempted else 0.0

    @property
    def successful_records(self) -> tuple[BootstrapRefitRecord, ...]:
        return tuple(r for r in self.records if r.fit_status == FIT_SUCCEEDED)

    @property
    def coefficients_by_refit(self) -> list[dict[str, float] | None]:
        """One entry per attempted refit, in bootstrap-ID order -- `None` for a failed refit.
        The structurally-correct input for `selection_stability.selection_frequency`: checking a
        group's columns *within one refit's own dict* (not across independently-compacted
        per-column lists) is what keeps a multi-column group's selection check joint over a single
        real resample, rather than comparing unrelated resamples that merely landed at the same
        list position."""
        return [r.coefficients_by_column for r in self.records]

    @property
    def refit_coefficients(self) -> dict[str, list[float]]:
        """Per design column, the observed coefficient values across every successful refit that
        produced one (order not meaningful, used only for percentile/distribution statistics) --
        the backward-compatible shape `coefficients.attach_bootstrap_intervals` and
        `selection_stability.bootstrap_coefficient_values` already consume."""
        result: dict[str, list[float]] = {}
        for record in self.successful_records:
            for column, value in (record.coefficients_by_column or {}).items():
                result.setdefault(column, []).append(value)
        return result

    @property
    def unconditional_365(self) -> list[np.ndarray]:
        return [r.predictions_365 for r in self.successful_records if r.predictions_365 is not None]

    @property
    def unconditional_730(self) -> list[np.ndarray]:
        return [r.predictions_730 for r in self.successful_records if r.predictions_730 is not None]

    def partial_effect_curves_for(self, presentation_level: str | None) -> list[list[float]]:
        return [
            r.partial_effect_curves[presentation_level]
            for r in self.successful_records
            if r.partial_effect_curves is not None and presentation_level in r.partial_effect_curves
        ]

    def importance_series(self, group_name: str, metric: str) -> list[float]:
        """Per-refit importance values for one `(group, metric)`, successful refits only, in
        bootstrap-ID order -- the raw material for `bootstrap_attribution_stability.csv`'s
        median/interval/rank aggregation."""
        values = []
        for record in self.successful_records:
            if record.importance_by_group is None:
                continue
            value = record.importance_by_group.get(group_name, {}).get(metric)
            if value is not None:
                values.append(value)
        return values

    def rank_series(self, group_name: str, metric: str) -> list[float]:
        values = []
        for record in self.successful_records:
            if record.rank_by_group is None:
                continue
            value = record.rank_by_group.get(group_name, {}).get(metric)
            if value is not None:
                values.append(value)
        return values

    def complete_rank_refit_count(
        self, metric: str, canonical_groups: dict[str, tuple[str, ...]]
    ) -> int:
        """Count of successful refits whose *entire* canonical rank vector for `metric` is
        complete -- every canonical group had a finite importance value in that refit, for that
        metric (`_rank_by_metric` only assigns a rank to groups whose value was finite). This is
        the `n_rank_estimable` gate for `bootstrap_attribution_stability.csv`'s rank fields, a
        stricter, refit-level completeness requirement than a single group's own
        `n_importance_estimable`."""
        count = 0
        for record in self.successful_records:
            if record.rank_by_group is None:
                continue
            if all(metric in record.rank_by_group.get(g, {}) for g in canonical_groups):
                count += 1
        return count


__all__ = [
    "FIT_FAILED",
    "FIT_SUCCEEDED",
    "REFIT_PURPOSE",
    "BootstrapRefitRecord",
    "RefitCollection",
    "bootstrap_resample_frame",
    "refit_seed",
    "run_bootstrap_refits",
]
