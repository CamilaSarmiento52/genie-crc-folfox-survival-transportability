"""The exploratory-optimization workstream pure-Python candidates: B4 `penalized_nonlinear_cox` and
B5
`reduced_core_penalized_cox` (optimization-spec.md).

Both reuse the model-training-validated spline and Cox-fitting machinery by import
(`crc_folfox.model_training.cox`) rather than reimplementing it, and both reuse preprocessing
workflow's
`FittedPreprocessor` for imputation, encoding, and scaling -- neither candidate modifies
preprocessing workflow or
model-training code. B5's stricter category-reduction rule is applied to the *raw* development frame
before preprocessing workflow ever sees it, so preprocessing workflow's own machinery is exercised
unchanged; the effect is that
preprocessing workflow observes fewer distinct raw levels, not that its own `RARE_MIN_COUNT = 10`
rule is patched.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crc_folfox.features import FeatureManifest
from crc_folfox.model_training.cox import SplineState, apply_spline_state, fit_cox, fit_spline_state
from crc_folfox.optimize.contracts import Phase5AContractError
from crc_folfox.preprocessing.transform import FittedPreprocessor

PENALIZER_GRID = (0.01, 0.1, 1.0)

# The one admitted interaction (optimization-spec.md, "B4"): every prior_oxaliplatin interaction
# is prospectively excluded because only 19/678 cohort patients are exposed.
_PRESENTATION_COLUMN_PREFIX = "metastatic_presentation__"


def _training_only_one_hot_columns(manifest: FeatureManifest | None = None) -> tuple[str, ...]:
    """The core feature set's categorical columns preprocessing workflow pools and one-hot encodes
    -- the
    candidate set for B5's stricter development-only reduction rule."""
    loaded = manifest or FeatureManifest.load()
    definitions = loaded.by_name()
    return tuple(
        name for name in loaded.core if definitions[name].encoding == "training_only_one_hot"
    )


@dataclass(frozen=True)
class RareCategoryReduction:
    reduced_frame: pd.DataFrame
    merged_levels: dict[str, tuple[str, ...]]  # column -> levels merged into __RARE__
    reference_merges: dict[str, tuple[str, ...]]  # column -> levels merged into the reference


RARE_LEVEL = "__RARE__"


def _reference_level(frame: pd.DataFrame, column: str, reference_rule: str) -> str:
    """Reimplements preprocessing workflow's mode-based reference resolution (`transform.py`'s
    `_deterministic_mode`) as optimization-owned code, rather than importing preprocessing
    workflow's private
    helper. The preprocessing workflow is frozen; this project's established convention is to
    reimplement the small
    piece of shape a downstream workflow needs (see `build_nested_pooled_splits`'s docstring) rather
    than
    reach into private internals."""
    if reference_rule == "training_mode_then_lexical":
        counts = frame[column].astype(str).value_counts()
        tied_for_mode = sorted(counts[counts == counts.max()].index)
        return tied_for_mode[0]
    return reference_rule


def reduce_rare_categories(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] | None = None,
    min_observations: int = 25,
    min_events: int = 10,
    manifest: FeatureManifest | None = None,
) -> RareCategoryReduction:
    """B5's stricter, development-only pooling rule (optimization-spec.md, "B5").

    Pool a non-reference level into `__RARE__` when it has fewer than `min_observations` fitting-
    partition rows **or** fewer than `min_events` fitting-partition events -- stricter than (a
    superset of) preprocessing workflow's own floor of `min_observations=10` alone. The declared
    reference level
    of a manifest-known column is never pooled, matching preprocessing workflow's own rule that only
    a
    *non-reference* level may be pooled; if the reference itself is too thin, this function leaves
    it alone and `FittedPreprocessor.fit` will legitimately reject it downstream, exactly as it
    would for the unreduced frame. If the pooled `__RARE__` bucket itself remains below
    `min_observations`, it merges into the reference level (or, for a column with no declared
    reference, the modal remaining level) instead of being left as an unsupported category. Must
    be called on fitting/training rows only -- calling it on assessment rows is a leakage error
    the caller is responsible for avoiding, exactly as preprocessing workflow's own
    `FittedPreprocessor.fit`
    requires.
    """
    if "event" not in frame.columns:
        raise Phase5AContractError("Frame must carry an 'event' column to reduce by event count")
    loaded_manifest = manifest or FeatureManifest.load()
    definitions = loaded_manifest.by_name()
    target_columns = columns or _training_only_one_hot_columns(loaded_manifest)
    reduced = frame.copy()
    merged_levels: dict[str, tuple[str, ...]] = {}
    reference_merges: dict[str, tuple[str, ...]] = {}

    for column in target_columns:
        if column not in reduced.columns:
            continue
        reference_level = (
            _reference_level(reduced, column, definitions[column].reference_rule)
            if column in definitions
            else None
        )
        counts = reduced.groupby(column, dropna=False)["event"].agg(
            observations="size", events="sum"
        )
        thin_mask = (counts["observations"] < min_observations) | (counts["events"] < min_events)
        thin_levels = tuple(
            level
            for level in counts.index[thin_mask]
            # A missing raw level (real data has some; the synthetic fixtures never happened to)
            # groups under `dropna=False` as a `NA`-like index entry, and `NA != reference_level`
            # returns `pd.NA` rather than a bool -- `pd.isna` short-circuits before that comparison
            # ever runs. A missing level can never equal a real (non-null) reference level, so it is
            # always eligible for pooling like any other thin level.
            if pd.isna(level) or level != reference_level
        )
        if not thin_levels:
            continue
        reduced[column] = reduced[column].where(~reduced[column].isin(thin_levels), RARE_LEVEL)
        merged_levels[column] = thin_levels

        pooled_observations = int((reduced[column] == RARE_LEVEL).sum())
        if 0 < pooled_observations < min_observations:
            merge_target = reference_level
            if merge_target is None:
                remaining_counts = reduced.loc[reduced[column] != RARE_LEVEL, column].value_counts()
                merge_target = str(remaining_counts.idxmax()) if len(remaining_counts) else None
            if merge_target and merge_target != RARE_LEVEL:
                reduced[column] = reduced[column].replace(RARE_LEVEL, merge_target)
                reference_merges[column] = (RARE_LEVEL,)

    return RareCategoryReduction(reduced, merged_levels, reference_merges)


def fit_reduced_core_preprocessor(
    development_frame: pd.DataFrame,
    *,
    endpoint: str,
    fit_scope: str,
    manifest: FeatureManifest | None = None,
    min_observations: int = 25,
    min_events: int = 10,
) -> tuple[FittedPreprocessor, RareCategoryReduction]:
    """B5: reduce rare categories on the raw development frame, then fit preprocessing workflow's
    ordinary core
    preprocessor on the reduced frame unchanged."""
    reduction = reduce_rare_categories(
        development_frame,
        min_observations=min_observations,
        min_events=min_events,
        manifest=manifest,
    )
    preprocessor = FittedPreprocessor.fit(
        reduction.reduced_frame,
        endpoint=endpoint,
        feature_set="core",
        fit_scope=fit_scope,
        manifest=manifest,
    )
    return preprocessor, reduction


def add_presentation_age_spline_interaction(
    matrix: pd.DataFrame, spline_state: SplineState
) -> pd.DataFrame:
    """B4's one admitted interaction: `metastatic_presentation` x each age spline basis column.

    Must be called after `apply_spline_state` has already replaced `age_at_folfox` with its
    spline basis. Adds one new column per spline basis column; nothing else in the matrix is
    touched.
    """
    presentation_columns = [c for c in matrix.columns if c.startswith(_PRESENTATION_COLUMN_PREFIX)]
    if len(presentation_columns) != 1:
        raise Phase5AContractError(
            "Expected exactly one encoded metastatic_presentation column for the B4 interaction"
        )
    presentation = matrix[presentation_columns[0]]
    missing_spline_columns = set(spline_state.columns) - set(matrix.columns)
    if missing_spline_columns:
        raise Phase5AContractError("Spline basis columns are missing from the design matrix")
    interaction = pd.DataFrame(
        {
            f"{presentation_columns[0]}__x__{column}": presentation * matrix[column]
            for column in spline_state.columns
        },
        index=matrix.index,
    )
    return pd.concat([matrix, interaction], axis=1)


def fit_penalized_nonlinear_cox(
    train_matrix: pd.DataFrame,
    train_metadata: pd.DataFrame,
    *,
    penalizer: float,
):
    """B4: fit ridge Cox (`l1_ratio=0.0`) on the spline-plus-interaction design.

    `train_matrix` must already be a preprocessed core-feature-set transform (age untouched); this
    function fits the spline basis, applies it, adds the interaction, and fits Cox -- returning
    `(fitter, spline_state)` so the caller can apply the identical transform to assessment rows.
    """
    spline_state = fit_spline_state(train_matrix)
    splined = apply_spline_state(train_matrix, spline_state)
    design = add_presentation_age_spline_interaction(splined, spline_state)
    fitter = fit_cox(design, train_metadata, penalizer=penalizer, l1_ratio=0.0)
    return fitter, spline_state


def transform_for_penalized_nonlinear_cox(
    matrix: pd.DataFrame, spline_state: SplineState
) -> pd.DataFrame:
    """Apply a training-fitted B4 spline state and interaction to any other rows (assessment or a
    different partition), without inspecting those rows' outcomes."""
    splined = apply_spline_state(matrix, spline_state)
    return add_presentation_age_spline_interaction(splined, spline_state)


def fit_reduced_core_cox(
    train_matrix: pd.DataFrame, train_metadata: pd.DataFrame, *, penalizer: float
):
    """B5: fit ridge Cox (`l1_ratio=0.0`) on an already-reduced-and-transformed core matrix."""
    return fit_cox(train_matrix, train_metadata, penalizer=penalizer, l1_ratio=0.0)
