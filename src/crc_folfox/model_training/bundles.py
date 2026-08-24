"""Allowlisted minimal-state Cox bundle serialization and the model registry.

Cox bundles never carry the live `lifelines.CoxPHFitter` object. A fitted `CoxPHFitter` retains
undocumented internal surface (training design references, variance/SE machinery) far beyond what
prediction needs, so this module extracts exactly the coefficients and the baseline survival step
function -- the minimal state the spec's absolute-risk mapping `1 - S0(t) ** exp(linear_predictor)`
requires -- and discards the fitter immediately after extraction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from crc_folfox.model_training.contracts import HORIZONS, ModelTrainingContractError
from crc_folfox.model_training.cox import SplineState
from crc_folfox.preprocessing.transform import FittedPreprocessor

MODEL_STATE_SPEC_VERSION = "2026-08-07"

# Every type an extracted bundle's object graph may contain. A `CoxPHFitter` (or any third-party
# fitted-model object) failing this allowlist is exactly the defect this module exists to prevent.
_ALLOWLISTED_LEAF_TYPES = (str, float, int, bool, type(None))


@dataclass(frozen=True)
class CoxBundleState:
    """The complete allowlisted, reloadable prediction state for one Cox candidate fit."""

    model_id: str
    endpoint: str
    design: str
    held_out_institution: str | None
    repeat: int | None
    fold: int | None
    model_name: str
    feature_set: str
    spec_version: str
    lifelines_version: str
    feature_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    norm_mean: tuple[float, ...]
    baseline_survival_times: tuple[float, ...]
    baseline_survival_values: tuple[float, ...]
    penalizer: float
    l1_ratio: float
    preprocessor: FittedPreprocessor
    spline_state: SplineState | None
    development_patient_count: int
    development_event_count: int
    input_hashes: Mapping[str, str] = field(default_factory=dict)


def extract_cox_bundle_state(
    fitter: CoxPHFitter,
    *,
    model_id: str,
    endpoint: str,
    design: str,
    held_out_institution: str | None,
    repeat: int | None,
    fold: int | None,
    model_name: str,
    feature_set: str,
    penalizer: float,
    l1_ratio: float,
    preprocessor: FittedPreprocessor,
    spline_state: SplineState | None,
    development_patient_count: int,
    development_event_count: int,
    input_hashes: Mapping[str, str] | None = None,
) -> CoxBundleState:
    """Pull the minimal reloadable state off a fitted `CoxPHFitter` and discard the fitter.

    Uses `fitter._norm_mean`: lifelines centers covariates at their training mean when computing
    the partial hazard that `baseline_survival_` is defined against (confirmed empirically against
    `predict_survival_function` on lifelines 0.30.3, the exact pinned version -- the absolute-risk
    mapping below reproduces `predict_survival_function`'s output to float precision only when the
    partial hazard is centered this way). This is a private lifelines attribute; the exact pin
    guards against a silent convention change in a future release.
    """
    if preprocessor.fit_patient_keys:
        raise ModelTrainingContractError("Bundle preprocessor state still carries patient keys")
    coefficients = fitter.params_
    baseline = fitter.baseline_survival_.iloc[:, 0]
    if not np.isfinite(coefficients.to_numpy(float)).all():
        raise ModelTrainingContractError("Cox coefficients are nonfinite")
    baseline_values = baseline.to_numpy(float)
    if not np.isfinite(baseline_values).all() or baseline.lt(0).any() or baseline.gt(1).any():
        raise ModelTrainingContractError("Baseline survival state is invalid")
    return CoxBundleState(
        model_id=model_id,
        endpoint=endpoint,
        design=design,
        held_out_institution=held_out_institution,
        repeat=repeat,
        fold=fold,
        model_name=model_name,
        feature_set=feature_set,
        spec_version=MODEL_STATE_SPEC_VERSION,
        lifelines_version=version("lifelines"),
        feature_columns=tuple(coefficients.index.astype(str)),
        coefficients=tuple(float(value) for value in coefficients.to_numpy(float)),
        norm_mean=tuple(float(value) for value in fitter._norm_mean.to_numpy(float)),
        baseline_survival_times=tuple(float(value) for value in baseline.index.to_numpy(float)),
        baseline_survival_values=tuple(float(value) for value in baseline.to_numpy(float)),
        penalizer=float(penalizer),
        l1_ratio=float(l1_ratio),
        preprocessor=preprocessor,
        spline_state=spline_state,
        development_patient_count=int(development_patient_count),
        development_event_count=int(development_event_count),
        input_hashes=dict(input_hashes or {}),
    )


def predict_from_bundle_state(
    state: CoxBundleState, matrix: pd.DataFrame, horizons: tuple[float, ...] = HORIZONS
) -> np.ndarray:
    """Reconstruct `1 - S0(t) ** exp(linear_predictor)` from stored state alone (no fitter)."""
    if tuple(matrix.columns.astype(str)) != state.feature_columns:
        raise ModelTrainingContractError("Predictor schema does not match the bundle's design")
    values = matrix.to_numpy(float)
    if not np.isfinite(values).all():
        raise ModelTrainingContractError("Predictor matrix contains nonfinite values")
    centered = values - np.asarray(state.norm_mean)
    partial_hazard = np.exp(centered @ np.asarray(state.coefficients))
    times = np.asarray(state.baseline_survival_times)
    survival_curve = np.asarray(state.baseline_survival_values)
    baseline_at_horizon = np.array(
        [
            1.0 if not len(times[times <= horizon]) else survival_curve[times <= horizon][-1]
            for horizon in horizons
        ]
    )
    result = baseline_at_horizon[np.newaxis, :] ** partial_hazard[:, np.newaxis]
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ModelTrainingContractError("Reconstructed survival prediction is invalid")
    if np.any(np.diff(result, axis=1) > 1e-12):
        raise ModelTrainingContractError("Reconstructed survival prediction is not nonincreasing")
    return result


def canonical_state_hash(state: CoxBundleState) -> str:
    """SHA-256 of the fitted numeric state's canonical JSON -- stable across clean reruns even
    though raw joblib bytes may carry timestamps or platform metadata."""
    payload = {
        "model_id": state.model_id,
        "endpoint": state.endpoint,
        "design": state.design,
        "held_out_institution": state.held_out_institution,
        "repeat": state.repeat,
        "fold": state.fold,
        "model_name": state.model_name,
        "feature_set": state.feature_set,
        "spec_version": state.spec_version,
        "lifelines_version": state.lifelines_version,
        "feature_columns": list(state.feature_columns),
        "coefficients": list(state.coefficients),
        "norm_mean": list(state.norm_mean),
        "baseline_survival_times": list(state.baseline_survival_times),
        "baseline_survival_values": list(state.baseline_survival_values),
        "penalizer": state.penalizer,
        "l1_ratio": state.l1_ratio,
        "preprocessor_state": state.preprocessor.aggregate_state(),
        "development_patient_count": state.development_patient_count,
        "development_event_count": state.development_event_count,
    }
    if state.spline_state is not None:
        payload["spline_state"] = {
            "means": list(state.spline_state.means),
            "standard_deviations": list(state.spline_state.standard_deviations),
            "retained": list(state.spline_state.retained),
            "columns": list(state.spline_state.columns),
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def feature_schema_hash(state: CoxBundleState) -> str:
    """SHA-256 of the ordered feature schema alone, independent of fitted coefficient values."""
    canonical = json.dumps(list(state.feature_columns), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_bundle(state: CoxBundleState, path: Path) -> None:
    _assert_allowlisted_payload(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, path)


def load_bundle(path: Path) -> CoxBundleState:
    state = joblib.load(path)
    if not isinstance(state, CoxBundleState):
        raise ModelTrainingContractError("Bundle does not contain a CoxBundleState")
    _assert_allowlisted_payload(state)
    return state


def _assert_allowlisted_payload(state: CoxBundleState) -> None:
    """Walk the bundle's object graph and reject any type outside the allowlist.

    This is the automated form of "payload inspection": a `CoxPHFitter`, a live sklearn estimator
    other than the locked `SplineTransformer`, or any other undeclared object cannot silently ride
    along inside a published bundle.
    """
    if isinstance(state, CoxPHFitter):
        raise ModelTrainingContractError("Bundle contains a raw CoxPHFitter")
    _walk_allowlisted(state, path="state")


def _walk_allowlisted(value: object, *, path: str) -> None:
    from sklearn.preprocessing import SplineTransformer

    if isinstance(value, _ALLOWLISTED_LEAF_TYPES):
        return
    if isinstance(value, CoxPHFitter):
        raise ModelTrainingContractError(f"Disallowed CoxPHFitter found at {path}")
    if isinstance(value, SplineTransformer):
        # Locked, deterministic, patient-row-free (Step 3 leakage sentinel) -- the spec's explicit
        # allowance for "model-specific transform state" covers this, unlike the Cox fitter.
        return
    if isinstance(value, (tuple, list, frozenset, set)):
        for index, item in enumerate(value):
            _walk_allowlisted(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk_allowlisted(item, path=f"{path}[{key!r}]")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for declared in fields(value):
            _walk_allowlisted(getattr(value, declared.name), path=f"{path}.{declared.name}")
        return
    raise ModelTrainingContractError(f"Disallowed object type {type(value).__name__} at {path}")


@dataclass(frozen=True)
class RegistryRecord:
    """One `model_registry.json` entry. A failed fit has no bundle path or hash."""

    model_id: str
    fit_status: str
    bundle_relative_path: str | None
    per_file_sha256: str | None
    canonical_state_hash: str | None
    feature_schema_hash: str | None
    format: str
    runtime: str
    endpoint: str
    development_scope: str
    model_family: str

    def to_json(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "fit_status": self.fit_status,
            "bundle_relative_path": self.bundle_relative_path,
            "per_file_sha256": self.per_file_sha256,
            "canonical_state_hash": self.canonical_state_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "format": self.format,
            "runtime": self.runtime,
            "endpoint": self.endpoint,
            "development_scope": self.development_scope,
            "model_family": self.model_family,
        }


def registry_record_from_state(
    state: CoxBundleState, *, bundle_relative_path: str, per_file_sha256: str
) -> RegistryRecord:
    return RegistryRecord(
        model_id=state.model_id,
        fit_status="fit_succeeded",
        bundle_relative_path=bundle_relative_path,
        per_file_sha256=per_file_sha256,
        canonical_state_hash=canonical_state_hash(state),
        feature_schema_hash=feature_schema_hash(state),
        format="joblib",
        runtime="python",
        endpoint=state.endpoint,
        development_scope=state.design,
        model_family=state.model_name,
    )


def verify_registry_load(models_directory: Path, record: RegistryRecord) -> CoxBundleState:
    """Load a registered bundle only after path-traversal-safety and hash validation."""
    if record.bundle_relative_path is None or record.per_file_sha256 is None:
        raise ModelTrainingContractError("Registry record has no bundle to load")
    relative = Path(record.bundle_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelTrainingContractError("Registry bundle path is not traversal-safe")
    resolved_models = models_directory.resolve()
    bundle_path = (models_directory / relative).resolve()
    if bundle_path.parent != resolved_models:
        raise ModelTrainingContractError("Registry bundle path escapes the models directory")
    if not bundle_path.is_file():
        raise ModelTrainingContractError("Registered bundle file is missing")
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    if digest != record.per_file_sha256:
        raise ModelTrainingContractError("Registered bundle hash does not match its file")
    state = load_bundle(bundle_path)
    if canonical_state_hash(state) != record.canonical_state_hash:
        raise ModelTrainingContractError("Reloaded canonical state hash does not match registry")
    return state


__all__ = [
    "CoxBundleState",
    "RegistryRecord",
    "canonical_state_hash",
    "extract_cox_bundle_state",
    "feature_schema_hash",
    "load_bundle",
    "predict_from_bundle_state",
    "registry_record_from_state",
    "save_bundle",
    "verify_registry_load",
]
