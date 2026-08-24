"""Load one registered, fitted model for one `(endpoint, centre, model)` explanation scope, and
predict from an arbitrary (possibly permuted) already-transformed design matrix.

Reuses the model-training, primary-evaluation, and optimization prediction primitives directly
rather than reimplementing any risk-scale
math: `crc_folfox.evaluate.prediction` for the Python Cox-bundle path (shared by `minimal_cox`,
`spline_cox`, `elastic_net_cox`, `reduced_core_penalized_cox`), `crc_folfox.optimize.candidates`'s
B4 transform for `penalized_nonlinear_cox`, and the R exchange convention already validated by
`crc_folfox.optimize.evaluation.predict_r_candidate_cell` for the two explained R candidates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from crc_folfox.evaluate.contracts import IBS_GRID
from crc_folfox.evaluate.prediction import (
    entry_conditional_risk,
    linear_predictor,
    transform_for_bundle,
    unconditional_risk,
)
from crc_folfox.explain.contracts import (
    MODEL_REGISTER_BY_NAME,
    RUNTIME_PYTHON,
    RUNTIME_R,
    ExplainContractError,
    registration_for,
)
from crc_folfox.explain.preflight import VerifiedInputs
from crc_folfox.model_training.bundles import CoxBundleState, verify_registry_load
from crc_folfox.model_training.bundles import RegistryRecord as CoxRegistryRecord
from crc_folfox.model_training.contracts import ModelScope
from crc_folfox.optimize.candidates import transform_for_penalized_nonlinear_cox
from crc_folfox.optimize.r_adapter import GlmnetAdapter
from crc_folfox.optimize.r_registry import (
    RRegistryRecord,
    verify_r_preprocessor_path,
    verify_r_registry_load,
)
from crc_folfox.optimize.workflow import _model_id
from crc_folfox.preprocessing.contracts import SplitDefinition


class ModelNotFitted(ExplainContractError):
    """Raised when a registered model has no `fit_succeeded` record for a requested scope."""


_PYTHON_COX_REGISTRY_MODELS = (
    "minimal_cox",
    "spline_cox",
    "elastic_net_cox",
    "reduced_core_penalized_cox",
)
_R_GLMNET_MODELS = ("glmnet_ridge_core", "glmnet_enet_core")

#: `(loaded, matrix, entry, horizons) -> RiskPrediction`, the same signature `predict_from_matrix`
#: presents. Note it carries **no** `exit`/`event`: a permutation repeat only ever varies predictor
#: columns, and the outcome used to score it is always the caller's unpermuted original.
MatrixPredictor = Callable[
    ["LoadedModel", pd.DataFrame, np.ndarray, tuple[float, ...]], "RiskPrediction"
]

#: Matrix-level prediction paths contributed by models **outside** explainability workflow's frozen
#: nine-model
#: register (`forest-retry-spec.md` §9). Keyed by model name and consulted only after every
#: explainability workflow
#: name has already been excluded, so registering an entry cannot change how any of the nine
#: resolves. forest-model-retry workflow's `ltrc_cif` uses this to borrow `explain.importance`
#: without joining explainability workflow's
#: register -- whose inventory assertion is a hard `== 9` -- or forking its permutation machinery.
_MATRIX_PREDICTORS: dict[str, MatrixPredictor] = {}


def register_matrix_predictor(model_name: str, predictor: MatrixPredictor) -> None:
    """Contribute a matrix-level prediction path for a non-workflow-6 model.

    Idempotent for the same callable so repeated module imports are harmless; re-registering a
    *different* callable under a live name is refused, because two prediction paths for one model
    name is a defect that would otherwise resolve by import order. A explainability model name can
    never be
    registered -- its dispatch is frozen.
    """
    if model_name in MODEL_REGISTER_BY_NAME:
        raise ExplainContractError(
            f"'{model_name}' is a registered Phase 6 model; its prediction path is frozen"
        )
    existing = _MATRIX_PREDICTORS.get(model_name)
    if existing is not None and existing is not predictor:
        raise ExplainContractError(f"'{model_name}' already has a different matrix predictor")
    _MATRIX_PREDICTORS[model_name] = predictor


@dataclass(frozen=True)
class LoadedModel:
    model_name: str
    runtime: str
    development_patient_count: int
    development_event_count: int
    cox_state: CoxBundleState | None = None
    r_record: RRegistryRecord | None = None
    r_bundle_path: Path | None = None
    r_preprocessor: object | None = None


def _model_id_for(split: SplitDefinition, model_name: str) -> str:
    if model_name in ("minimal_cox", "spline_cox", "elastic_net_cox"):
        return ModelScope(split).model_id(model_name)
    return _model_id(split, model_name)


def load_model(paths: VerifiedInputs, split: SplitDefinition, model_name: str) -> LoadedModel:
    """Load a registered, fitted model by registry verification. Never fits anything."""
    registration = registration_for(model_name)
    if not registration.is_explained:
        raise ExplainContractError(f"'{model_name}' is Tier 4 and cannot be loaded for explanation")

    if model_name in ("minimal_cox", "spline_cox"):
        models_dir, registry = (
            paths.frozen_model_training_dir / "models",
            paths.frozen_model_training_registry,
        )
    elif model_name == "elastic_net_cox":
        models_dir, registry = (
            paths.corrected_model_training_dir / "models",
            paths.corrected_model_training_registry,
        )
    elif model_name in ("penalized_nonlinear_cox", "reduced_core_penalized_cox"):
        models_dir, registry = paths.optimization_dir / "models", paths.optimization_python_registry
    elif model_name in _R_GLMNET_MODELS:
        models_dir, registry = paths.optimization_dir / "models", paths.optimization_r_registry
    else:
        raise ExplainContractError(f"Unhandled explained model '{model_name}'")

    model_id = _model_id_for(split, model_name)
    record_json = registry.get(model_id)
    if record_json is None or record_json.get("fit_status") != "fit_succeeded":
        raise ModelNotFitted(f"'{model_id}' has no fit_succeeded record")

    if registration.runtime == RUNTIME_PYTHON:
        state = verify_registry_load(models_dir, CoxRegistryRecord(**record_json))
        return LoadedModel(
            model_name,
            RUNTIME_PYTHON,
            state.development_patient_count,
            state.development_event_count,
            cox_state=state,
        )

    record = RRegistryRecord.from_json(record_json)
    bundle_path = verify_r_registry_load(models_dir, record)
    preprocessor_path = verify_r_preprocessor_path(models_dir, record)
    preprocessor = joblib.load(preprocessor_path)
    return LoadedModel(
        model_name,
        RUNTIME_R,
        record.development_patient_count,
        record.development_event_count,
        r_record=record,
        r_bundle_path=bundle_path,
        r_preprocessor=preprocessor,
    )


def loaded_feature_columns(loaded: LoadedModel) -> tuple[str, ...]:
    """The loaded model's own frozen transformed-design column schema -- `cox_state.feature_columns`
    for a Python-fit model, `r_preprocessor.output_columns` for an R-fit model (`r_preprocessor` is
    the same `FittedPreprocessor` class as the Python path, loaded via `joblib.load` above; it has
    the identical `.transform()`/`.output_columns` interface). Used as the canonical feature-group
    inventory for bootstrap importance/ranking (`explainability-spec.md`'s 2026-08-11 amendment) so
    that inventory is derived from frozen model state for both runtimes, never from an
    assessment-side or refit-side matrix.
    """
    if loaded.cox_state is not None:
        return loaded.cox_state.feature_columns
    return loaded.r_preprocessor.output_columns


def transform_assessment(
    loaded: LoadedModel, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the loaded model's saved training-only transform to a frame (assessment or a
    row-permuted copy of it). No fitting occurs."""
    if loaded.runtime == RUNTIME_PYTHON:
        state = loaded.cox_state
        if loaded.model_name == "penalized_nonlinear_cox":
            transformed = state.preprocessor.transform(frame)
            design = transform_for_penalized_nonlinear_cox(transformed.matrix, state.spline_state)
            return design, transformed.metadata
        matrix, metadata = transform_for_bundle(state, frame)
        return matrix, metadata
    transformed = loaded.r_preprocessor.transform(frame)
    return transformed.matrix, transformed.metadata


@dataclass(frozen=True)
class RiskPrediction:
    unconditional: np.ndarray  # (n, len(horizons))
    entry_conditional: np.ndarray
    linear_predictor: np.ndarray  # NaN for every R-backed model
    # `{design_column: coefficient}` -- always available (a side channel of the same call, not a
    # second one) so a caller never pays an extra R round trip just to read coefficients.
    coefficients: dict[str, float] | None = None


def predict_from_matrix(
    loaded: LoadedModel,
    matrix: pd.DataFrame,
    entry: np.ndarray,
    horizons: tuple[float, ...] = IBS_GRID,
) -> RiskPrediction:
    """Predict directly from an already-transformed design matrix -- the shared primitive behind
    both baseline scoring and permutation importance, so permuting a design column never touches
    the transform step itself."""
    if loaded.runtime == RUNTIME_PYTHON:
        state = loaded.cox_state
        unconditional = unconditional_risk(state, matrix, horizons)
        conditional = entry_conditional_risk(state, matrix, entry, horizons)
        lp = linear_predictor(state, matrix)
        coefficients = dict(zip(state.feature_columns, state.coefficients, strict=True))
        return RiskPrediction(unconditional, conditional, lp, coefficients=coefficients)

    if loaded.model_name not in _R_GLMNET_MODELS:
        # Consulted only here, after every explainability model name has already been resolved
        # above, so a
        # contributed predictor cannot shadow a frozen explainability workflow dispatch
        # (`register_matrix_predictor`
        # refuses the nine registered names outright).
        contributed = _MATRIX_PREDICTORS.get(loaded.model_name)
        if contributed is not None:
            return contributed(loaded, matrix, np.asarray(entry, dtype=float), tuple(horizons))
        raise ExplainContractError(f"'{loaded.model_name}' has no matrix-level R prediction path")
    entry = np.asarray(entry, dtype=float)
    unique_entries = sorted({float(value) for value in entry})
    combined_grid = tuple(sorted(set(horizons) | set(unique_entries)))
    index_of = {time_value: index for index, time_value in enumerate(combined_grid)}
    result = GlmnetAdapter().predict(loaded.r_bundle_path, matrix, horizons=combined_grid)
    survival = result.survival[0]

    horizon_index = [index_of[h] for h in horizons]
    unconditional = 1.0 - survival[:, horizon_index]
    entry_index = np.array([index_of[float(value)] for value in entry])
    s_entry = survival[np.arange(len(entry)), entry_index]
    s_horizon = survival[:, horizon_index]
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = 1.0 - (s_horizon / s_entry[:, None])
    eligible = entry[:, None] <= np.asarray(horizons, dtype=float)[None, :]
    conditional = np.where(eligible, conditional, np.nan)
    if (
        not np.isfinite(unconditional).all()
        or np.any(unconditional < -1e-9)
        or np.any(unconditional > 1 + 1e-9)
    ):
        raise ExplainContractError("R candidate unconditional risk is nonfinite or out of [0, 1]")
    finite_where_eligible = np.isfinite(conditional) | ~eligible
    if not finite_where_eligible.all():
        raise ExplainContractError("R candidate entry-conditional risk is nonfinite where eligible")
    return RiskPrediction(
        unconditional, conditional, np.full(len(entry), np.nan), coefficients=result.coefficients
    )


def predict_assessment(
    loaded: LoadedModel, frame: pd.DataFrame, horizons: tuple[float, ...] = IBS_GRID
) -> tuple[pd.DataFrame, pd.DataFrame, RiskPrediction]:
    """Convenience: transform then predict, returning the matrix/metadata alongside the risk."""
    matrix, metadata = transform_assessment(loaded, frame)
    entry = metadata["entry"].to_numpy(dtype=float)
    prediction = predict_from_matrix(loaded, matrix, entry, horizons)
    return matrix, metadata, prediction


__all__ = [
    "LoadedModel",
    "MatrixPredictor",
    "ModelNotFitted",
    "RiskPrediction",
    "load_model",
    "loaded_feature_columns",
    "predict_assessment",
    "predict_from_matrix",
    "register_matrix_predictor",
    "transform_assessment",
]
