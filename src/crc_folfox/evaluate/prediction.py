"""Bundle-verified, training-only-transform prediction on both locked risk scales.

Every prediction in primary-evaluation workflow comes from a registry-verified model-training
workflow `CoxBundleState` with no
refitting. The unconditional scale reuses `crc_folfox.model_training.bundles.
predict_from_bundle_state` (already validated by model-training workflow's own tests) by import; the
entry-conditional scale is primary-evaluation workflow's own addition and shares the same
baseline-lookup and
partial-hazard math so the two scales cannot silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crc_folfox.evaluate.contracts import EvaluationContractError
from crc_folfox.model_training.bundles import CoxBundleState, predict_from_bundle_state
from crc_folfox.model_training.contracts import HORIZONS, ModelTrainingContractError
from crc_folfox.model_training.cox import apply_spline_state


def transform_for_bundle(
    state: CoxBundleState, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the bundle's saved training-only preprocessing (and spline state, if any)."""
    result = state.preprocessor.transform(frame)
    matrix = result.matrix
    if state.spline_state is not None:
        matrix = apply_spline_state(matrix, state.spline_state)
    return matrix, result.metadata


def _partial_hazard(state: CoxBundleState, matrix: pd.DataFrame) -> np.ndarray:
    if tuple(matrix.columns.astype(str)) != state.feature_columns:
        raise EvaluationContractError("Predictor schema does not match the bundle's design")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise EvaluationContractError("Predictor matrix contains nonfinite values")
    centered = values - np.asarray(state.norm_mean)
    return np.exp(centered @ np.asarray(state.coefficients))


def linear_predictor(state: CoxBundleState, matrix: pd.DataFrame) -> np.ndarray:
    """`lp = coefficients . (x - norm_mean)`, i.e. `log(partial_hazard)`.

    Used for the horizon-independent prognostic-index slope: under a Cox proportional-hazards
    model, ranking by `lp` is identical to ranking by unconditional absolute risk at any fixed
    horizon (both are monotone transforms of the same `exp(lp)` factor), so this is also the
    natural single risk score for the horizon-independent concordance statistic.
    """
    return np.log(_partial_hazard(state, matrix))


def _baseline_survival_at(state: CoxBundleState, query: np.ndarray) -> np.ndarray:
    """Evaluate the bundle's baseline survival step function at arbitrary (per-row) times."""
    times = np.asarray(state.baseline_survival_times, dtype=float)
    values = np.asarray(state.baseline_survival_values, dtype=float)
    query = np.asarray(query, dtype=float)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    index = np.searchsorted(times, query, side="right") - 1
    return np.where(index < 0, 1.0, values[np.clip(index, 0, len(values) - 1)])


def unconditional_risk(
    state: CoxBundleState, matrix: pd.DataFrame, horizons: tuple[float, ...] = HORIZONS
) -> np.ndarray:
    """`1 - S0(t) ** exp(linear_predictor)`, the SAP-locked primary absolute-risk scale."""
    try:
        survival = predict_from_bundle_state(state, matrix, horizons)
    except ModelTrainingContractError as exc:
        # Normalize the reused model-training workflow function's exception type to
        # primary-evaluation workflow's own, so every
        # prediction-path failure in this module is uniformly an `EvaluationContractError`.
        raise EvaluationContractError(str(exc)) from exc
    risk = 1.0 - survival
    _assert_valid_risk(risk)
    return risk


def entry_conditional_risk(
    state: CoxBundleState,
    matrix: pd.DataFrame,
    entry: np.ndarray,
    horizons: tuple[float, ...] = HORIZONS,
) -> np.ndarray:
    """`1 - (S0(t) / S0(entry)) ** exp(linear_predictor)`, defined for `entry <= t`.

    `entry <= t` (not strict `<`) deliberately matches the overall evaluation eligibility
    convention (`entry_i <= t`) used everywhere else -- see `horizon_weights`. At `entry == t`
    exactly, `_baseline_survival_at` looks up the identical step-function value for both the
    numerator and denominator queries, so the ratio is exactly 1 and the risk is exactly 0: a
    well-defined limiting case (zero elapsed time since entry), not an undefined one. Excluding it
    would only produce spurious not-computable cells at a boundary that day-scale entry times can
    plausibly land on exactly. Undefined cells (`entry > horizon`) are `NaN`; callers must treat
    those as ineligible rather than as a computed value of zero risk.
    """
    partial_hazard = _partial_hazard(state, matrix)
    entry = np.asarray(entry, dtype=float)
    s0_entry = _baseline_survival_at(state, entry)
    baseline_at_horizon = _baseline_survival_at(state, np.asarray(horizons, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = baseline_at_horizon[np.newaxis, :] / s0_entry[:, np.newaxis]
        conditional_survival = ratio ** partial_hazard[:, np.newaxis]
    risk = 1.0 - conditional_survival
    eligible = entry[:, np.newaxis] <= np.asarray(horizons, dtype=float)[np.newaxis, :]
    risk = np.where(eligible, risk, np.nan)
    finite_where_eligible = np.isfinite(risk) | ~eligible
    if not finite_where_eligible.all():
        raise EvaluationContractError("Entry-conditional risk is nonfinite where entry <= horizon")
    computed = risk[eligible]
    if len(computed) and (np.any(computed < -1e-9) or np.any(computed > 1 + 1e-9)):
        raise EvaluationContractError("Entry-conditional risk is outside [0, 1]")
    return risk


def _assert_valid_risk(risk: np.ndarray) -> None:
    if not np.isfinite(risk).all() or np.any(risk < -1e-9) or np.any(risk > 1 + 1e-9):
        raise EvaluationContractError("Predicted risk is nonfinite or outside [0, 1]")
    if np.any(np.diff(risk, axis=1) < -1e-12):
        raise EvaluationContractError("Predicted risk is not nondecreasing across horizons")


@dataclass(frozen=True)
class PredictionSet:
    """In-memory-only per-patient predictions for one evaluation cell. Never serialized."""

    patient_key: tuple[str, ...]
    institution: tuple[str, ...]
    entry: np.ndarray
    exit: np.ndarray
    event: np.ndarray
    unconditional: np.ndarray
    entry_conditional: np.ndarray
    linear_predictor: np.ndarray

    def __len__(self) -> int:
        return len(self.patient_key)


def predict_for_frame(
    state: CoxBundleState, frame: pd.DataFrame, horizons: tuple[float, ...] = HORIZONS
) -> PredictionSet:
    """Reconstruct predictions for one assessment frame from a registry-verified bundle."""
    required = {"patient_key", "institution", "entry", "exit", "event"}
    if missing := required - set(frame.columns):
        raise EvaluationContractError(
            f"Assessment frame is missing {len(missing)} required columns"
        )
    matrix, metadata = transform_for_bundle(state, frame)
    entry = metadata["entry"].to_numpy(dtype=float)
    unconditional = unconditional_risk(state, matrix, horizons)
    conditional = entry_conditional_risk(state, matrix, entry, horizons)
    lp = linear_predictor(state, matrix)
    return PredictionSet(
        patient_key=tuple(metadata["patient_key"].astype(str)),
        institution=tuple(metadata["institution"].astype(str)),
        entry=entry,
        exit=metadata["exit"].to_numpy(dtype=float),
        event=metadata["event"].to_numpy(dtype=int),
        unconditional=unconditional,
        entry_conditional=conditional,
        linear_predictor=lp,
    )


__all__ = [
    "PredictionSet",
    "entry_conditional_risk",
    "linear_predictor",
    "predict_for_frame",
    "transform_for_bundle",
    "unconditional_risk",
]
