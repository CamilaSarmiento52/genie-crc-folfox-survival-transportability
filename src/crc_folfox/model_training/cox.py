"""Delayed-entry Cox fits and spline preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import SplineTransformer

from crc_folfox.model_training.contracts import (
    HORIZONS,
    DelayedEntryAdapter,
    FittedDelayedEntryModel,
    ModelTrainingContractError,
    validate_survival_inputs,
)


@dataclass(frozen=True)
class SplineState:
    transformer: SplineTransformer
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]
    retained: tuple[int, ...]
    columns: tuple[str, ...]


def fit_spline_state(matrix: pd.DataFrame) -> SplineState:
    """Replace only the standardized linear age column with the frozen spline basis."""
    if "age_at_folfox" not in matrix:
        raise ModelTrainingContractError("Core schema does not include age_at_folfox")
    if not np.isfinite(matrix[["age_at_folfox"]].to_numpy(float)).all():
        raise ModelTrainingContractError("age_at_folfox contains missing or nonfinite values")
    transformer = SplineTransformer(
        degree=3, n_knots=4, knots="quantile", include_bias=False, extrapolation="linear", order="C"
    )
    raw = transformer.fit_transform(matrix[["age_at_folfox"]])
    means = raw.mean(axis=0)
    stds = raw.std(axis=0, ddof=0)
    retained = tuple(int(index) for index, std in enumerate(stds) if std > 0)
    if not retained:
        raise ModelTrainingContractError("Spline basis contains no nonconstant columns")
    columns = tuple(f"age_at_folfox__spline_{index:02d}" for index in retained)
    return SplineState(transformer, tuple(means), tuple(stds), retained, columns)


def apply_spline_state(matrix: pd.DataFrame, state: SplineState) -> pd.DataFrame:
    """Apply a training-fitted spline state without inspecting the target rows."""
    if "age_at_folfox" not in matrix:
        raise ModelTrainingContractError("Core schema does not include age_at_folfox")
    if not np.isfinite(matrix[["age_at_folfox"]].to_numpy(float)).all():
        raise ModelTrainingContractError("age_at_folfox contains missing or nonfinite values")
    colliding = set(state.columns) & set(matrix.columns.drop("age_at_folfox"))
    if colliding:
        raise ModelTrainingContractError("Spline basis names collide with an existing column")
    raw = state.transformer.transform(matrix[["age_at_folfox"]])
    normalized = (raw - np.asarray(state.means)) / np.asarray(state.standard_deviations)
    spline = pd.DataFrame(
        normalized[:, list(state.retained)], columns=state.columns, index=matrix.index
    )
    result = pd.concat([matrix.drop(columns="age_at_folfox"), spline], axis=1)
    if not np.isfinite(result.to_numpy(float)).all():
        raise ModelTrainingContractError("Spline transform contains nonfinite values")
    return result


def fit_cox(
    matrix: pd.DataFrame, metadata: pd.DataFrame, *, penalizer: float, l1_ratio: float
) -> CoxPHFitter:
    """Fit the prescribed counting-process Cox model."""
    frame = pd.concat(
        [
            metadata.loc[:, ["entry", "exit", "event"]].reset_index(drop=True),
            matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    validate_survival_inputs(
        matrix.to_numpy(float),
        frame["entry"].to_numpy(float),
        frame["exit"].to_numpy(float),
        frame["event"].to_numpy(int),
    )
    fitter = CoxPHFitter(
        penalizer=penalizer, l1_ratio=l1_ratio, baseline_estimation_method="breslow"
    )
    fitter.fit(
        frame,
        duration_col="exit",
        event_col="event",
        entry_col="entry",
        fit_options={"max_steps": 500},
    )
    if not np.isfinite(fitter.params_.to_numpy(float)).all():
        raise ModelTrainingContractError("Cox fitted state is nonfinite")
    return fitter


def predict_survival(
    fitter: CoxPHFitter, matrix: pd.DataFrame, horizons: tuple[float, ...] = HORIZONS
) -> np.ndarray:
    """Return bounded, nonincreasing survival at the requested horizons."""
    values = fitter.predict_survival_function(matrix, times=list(horizons)).to_numpy(float).T
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ModelTrainingContractError("Cox survival prediction is invalid")
    if np.any(np.diff(values, axis=1) > 1e-12):
        raise ModelTrainingContractError("Cox survival prediction is not nonincreasing")
    return values


@dataclass(frozen=True)
class FittedCoxModel(FittedDelayedEntryModel):
    """A fitted, unpenalized-or-locked-penalty Cox candidate."""

    fitter: CoxPHFitter

    def predict_survival(
        self, matrix: pd.DataFrame, horizons: tuple[float, ...] = HORIZONS
    ) -> np.ndarray:
        return predict_survival(self.fitter, matrix, horizons)


@dataclass(frozen=True)
class CoxAdapter(DelayedEntryAdapter):
    """`minimal_cox`/`core_cox`/`spline_cox` share this adapter with a locked penalty.

    Feature-set selection (baseline vs. core) and any spline transform happen upstream, in the
    preprocessing workflow preprocessor and `apply_spline_state`; this adapter only fits the
    already-selected
    predictor matrix, so the same class conforms to all three unpenalized/locked-ridge candidates.
    """

    penalizer: float
    l1_ratio: float

    def fit(
        self,
        matrix: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        seed: int,
        config: Mapping[str, object],
    ) -> FittedCoxModel:
        del seed, config  # Cox fitting is deterministic given data and a locked penalty.
        fitter = fit_cox(matrix, metadata, penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        return FittedCoxModel(fitter)
