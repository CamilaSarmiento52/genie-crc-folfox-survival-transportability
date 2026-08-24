"""E4: grouped spline and partial-effect curves for `spline_cox` and `penalized_nonlinear_cox`.

`explainability-spec.md`, "E4": the partial effect of `age_at_folfox` on the log relative hazard,
on a frozen 25-point grid over the 1st-99th percentile of the development-scope fitting rows,
holding every other predictor at a frozen reference profile, centred at the fitting-partition
median age.

**Age standardization (2026-08-11 fix).** `SplineTransformer` is fit in model-training workflow
(`model_training.cox.fit_spline_state`) on the **already-standardized** `age_at_folfox` column --
preprocessing workflow z-scores it with the training mean and population SD before model-training
workflow ever sees it. A raw
age value (e.g. 55) must therefore be standardized with that same mean/SD -- read from the loaded
bundle's own `state.preprocessor.scaling["age_at_folfox"]` -- before it is passed into
`spline_state.transformer.transform()`; passing a raw age directly (the prior defect) lands far
outside the fitted knot range and produces extreme extrapolated effects.

The reference profile's non-age columns are read directly from the loaded bundle's own frozen
`FittedPreprocessor` state -- a one-hot block at its stored reference level is exactly the zero
vector in the transformed design, and `age_at_folfox` is confirmed the only standardized continuous
predictor in the `core` feature set, so no other reference-row column needs special handling. The
age curve itself is centred at the **fitting-partition median** raw age, computed once by the
caller from the development-scope fitting rows and passed in as `reference_age` -- never derived
from the age grid, and never recomputed per bootstrap resample (`workflow.py` computes it once at
the point estimate and threads it unchanged into every bootstrap refit's partial-effect call).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crc_folfox.explain.contracts import ExplainContractError
from crc_folfox.explain.feature_groups import INTERACTION_MARKER
from crc_folfox.explain.loading import LoadedModel
from crc_folfox.optimize.candidates import add_presentation_age_spline_interaction

N_AGE_GRID_POINTS = 25
AGE_PERCENTILE_LOW, AGE_PERCENTILE_HIGH = 1.0, 99.0


def age_grid_from_fitting_rows(
    fitting_ages: np.ndarray, n_points: int = N_AGE_GRID_POINTS
) -> np.ndarray:
    lower = np.percentile(fitting_ages, AGE_PERCENTILE_LOW)
    upper = np.percentile(fitting_ages, AGE_PERCENTILE_HIGH)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ExplainContractError("Age grid bounds are degenerate")
    return np.linspace(lower, upper, n_points)


def _reference_row(state) -> pd.DataFrame:
    """One row, every transformed design column at the loaded bundle's own frozen reference
    (zero): categorical one-hot blocks at their stored reference level, scaled continuous columns
    at their stored fitting-partition mean, binary columns at zero.

    Excludes both the age spline columns (rebuilt fresh per age value by `_age_spline_row`) and
    any admitted-interaction columns (rebuilt fresh by `add_presentation_age_spline_interaction`
    once the presentation one-hot and spline columns are in place) -- an interaction column such
    as `metastatic_presentation__De novo__x__age_at_folfox__spline_00` also starts with the
    `metastatic_presentation__` prefix, so leaving it in the reference row would double the count
    of columns `add_presentation_age_spline_interaction` sees with that prefix and made it refuse
    every real `penalized_nonlinear_cox` scope with "Expected exactly one encoded
    metastatic_presentation column" -- found by the Gate 8 real full-inventory run, not by any
    synthetic test (no synthetic fixture in this workflow's test suite exercises a bundle whose
    `feature_columns` already contain interaction columns going into `_reference_row`).
    """
    columns = [
        c
        for c in state.feature_columns
        if not c.startswith("age_at_folfox__") and INTERACTION_MARKER not in c
    ]
    return pd.DataFrame({column: [0.0] for column in columns})


@dataclass(frozen=True)
class PartialEffectRow:
    model_name: str
    presentation_level: str | None  # None except for penalized_nonlinear_cox's per-level curves
    age_value: float
    effect: float
    ci_lower: float | None = None
    ci_upper: float | None = None


def _standardized_age(state, age_value: float) -> float:
    """Raw age -> the same standardization preprocessing and model-training applied before ever
    fitting the spline
    transformer (see module docstring's "Age standardization" note)."""
    scaling = state.preprocessor.scaling["age_at_folfox"]
    std = scaling["population_std"]
    if std == 0:
        raise ExplainContractError("age_at_folfox has zero fitting-partition standard deviation")
    return (age_value - scaling["mean"]) / std


def _age_spline_row(state, age_value: float) -> pd.DataFrame:
    spline_state = state.spline_state
    standardized_age = _standardized_age(state, age_value)
    raw = spline_state.transformer.transform(np.array([[standardized_age]]))
    normalized = (raw - np.asarray(spline_state.means)) / np.asarray(
        spline_state.standard_deviations
    )
    return pd.DataFrame(normalized[:, list(spline_state.retained)], columns=spline_state.columns)


def compute_partial_effect(
    loaded: LoadedModel,
    age_grid: np.ndarray,
    *,
    reference_age: float,
    presentation_level: str | None = None,
) -> list[PartialEffectRow]:
    """One model's partial-effect curve, optionally at one fixed `metastatic_presentation` level
    for the admitted-interaction candidate (`penalized_nonlinear_cox`).

    `reference_age` is the frozen fitting-partition median raw age (development-scope, never a
    bootstrap resample's own median -- see module docstring) at which the curve is centred.
    """
    state = loaded.cox_state
    if state is None or state.spline_state is None:
        raise ExplainContractError(f"'{loaded.model_name}' has no spline state to plot")

    reference = _reference_row(state)
    has_interaction = any("__x__" in column for column in state.feature_columns)
    if has_interaction and presentation_level is None:
        raise ExplainContractError(
            f"'{loaded.model_name}' requires a presentation_level for its admitted interaction"
        )
    if not has_interaction and presentation_level is not None:
        raise ExplainContractError(
            f"'{loaded.model_name}' has no admitted interaction to hold fixed"
        )

    # Excludes interaction columns for the same reason `_reference_row` does: an interaction
    # column such as `metastatic_presentation__De novo__x__age_at_folfox__spline_00` also starts
    # with the `metastatic_presentation__` prefix, and `row[presentation_columns] = 0.0` below
    # would silently *recreate* it as a fresh zero column (pandas creates missing list-indexed
    # columns on assignment) even though `_reference_row` already excluded it -- the second half
    # of the same real bug the Gate 8 full-inventory run found (see `_reference_row`'s docstring).
    presentation_columns = [
        c
        for c in state.feature_columns
        if c.startswith("metastatic_presentation__") and INTERACTION_MARKER not in c
    ]

    def _row_at(age_value: float) -> pd.DataFrame:
        row = reference.copy()
        if presentation_level is not None:
            row[presentation_columns] = 0.0
            level_column = f"metastatic_presentation__{presentation_level}"
            if level_column in row.columns:
                row[level_column] = 1.0
        spline_columns = _age_spline_row(state, age_value)
        merged = pd.concat(
            [row.reset_index(drop=True), spline_columns.reset_index(drop=True)], axis=1
        )
        if has_interaction:
            merged = add_presentation_age_spline_interaction(merged, state.spline_state)
        return merged.loc[:, list(state.feature_columns)]

    reference_matrix = _row_at(reference_age)
    reference_lp = float(
        (
            (reference_matrix.to_numpy(float) - np.asarray(state.norm_mean))
            @ np.asarray(state.coefficients)
        )[0]
    )

    rows = []
    for age_value in age_grid:
        matrix = _row_at(float(age_value))
        lp = float(
            (
                (matrix.to_numpy(float) - np.asarray(state.norm_mean))
                @ np.asarray(state.coefficients)
            )[0]
        )
        rows.append(
            PartialEffectRow(
                model_name=loaded.model_name,
                presentation_level=presentation_level,
                age_value=float(age_value),
                effect=lp - reference_lp,
            )
        )
    return rows


def attach_bootstrap_bands(
    rows: list[PartialEffectRow], refit_effects: dict[float, list[float]]
) -> list[PartialEffectRow]:
    """Percentile 95% band from refit-derived curves at the same age grid points."""
    updated = []
    for row in rows:
        values = refit_effects.get(row.age_value, [])
        if not values:
            updated.append(row)
            continue
        lower, upper = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
        updated.append(
            PartialEffectRow(
                model_name=row.model_name,
                presentation_level=row.presentation_level,
                age_value=row.age_value,
                effect=row.effect,
                ci_lower=float(lower),
                ci_upper=float(upper),
            )
        )
    return updated


__all__ = [
    "AGE_PERCENTILE_HIGH",
    "AGE_PERCENTILE_LOW",
    "N_AGE_GRID_POINTS",
    "PartialEffectRow",
    "age_grid_from_fitting_rows",
    "attach_bootstrap_bands",
    "compute_partial_effect",
]
