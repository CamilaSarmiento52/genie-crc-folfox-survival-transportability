"""Separate registry for the R-backed optimization workflow candidates (B2 `glmnet_ridge_core`, B3
`glmnet_enet_core`, B6 `flexsurv_rp_spline`).

model-training workflow's `model_registry.json`/`CoxBundleState` format is Cox-coefficient specific
-- it stores
fitted coefficients, feature-centering means, and a baseline survival curve, and predicts with one
fixed formula. B4 and B5 fit plain `lifelines.CoxPHFitter` objects and reuse that format and its
machinery (`extract_cox_bundle_state`/`save_bundle`/`registry_record_from_state`/
`verify_registry_load`) completely unchanged. B2, B3, and B6 are fit in R with different
statistical machinery (penalized regression, spline-based survival curves); their fitted state is
an opaque `.rds` file with no coefficients-plus-baseline-curve shape to translate into
`CoxBundleState`, and predicting from a saved bundle means calling back into the R adapter's
`--predict` path rather than doing arithmetic on a deserialized Python object. This module is that
second, R-specific registry, kept deliberately separate rather than folded into one polymorphic
registry -- see the optimization workflow architecture note in `implement.md` once written.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from crc_folfox.optimize.contracts import Phase5AContractError

R_CANDIDATES = ("glmnet_ridge_core", "glmnet_enet_core", "flexsurv_rp_spline")


@dataclass(frozen=True)
class RRegistryRecord:
    """One R-registry entry. A failed fit has no bundle path or hash, mirroring
    `crc_folfox.model_training.bundles.RegistryRecord`."""

    model_id: str
    candidate: str
    fit_status: str
    bundle_relative_path: str | None
    per_file_sha256: str | None
    # The R bundle has no imputation/encoding/scaling state of its own -- unlike a Python
    # `CoxBundleState`, which embeds its `FittedPreprocessor`. This sidecar is what lets
    # evaluation transform new assessment rows into the matrix the R model expects.
    preprocessor_relative_path: str | None
    preprocessor_sha256: str | None
    endpoint: str
    design: str
    held_out_institution: str | None
    repeat: int | None
    fold: int | None
    feature_set: str
    config: dict[str, float]
    development_patient_count: int
    development_event_count: int

    def __post_init__(self) -> None:
        if self.candidate not in R_CANDIDATES:
            raise Phase5AContractError(f"Unknown R candidate '{self.candidate}'")
        if self.feature_set not in {"core", "baseline"}:
            raise Phase5AContractError(f"Unknown feature set '{self.feature_set}'")

    def to_json(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "candidate": self.candidate,
            "fit_status": self.fit_status,
            "bundle_relative_path": self.bundle_relative_path,
            "per_file_sha256": self.per_file_sha256,
            "preprocessor_relative_path": self.preprocessor_relative_path,
            "preprocessor_sha256": self.preprocessor_sha256,
            "endpoint": self.endpoint,
            "design": self.design,
            "held_out_institution": self.held_out_institution,
            "repeat": self.repeat,
            "fold": self.fold,
            "feature_set": self.feature_set,
            "config": dict(self.config),
            "development_patient_count": self.development_patient_count,
            "development_event_count": self.development_event_count,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> RRegistryRecord:
        return cls(
            model_id=str(payload["model_id"]),
            candidate=str(payload["candidate"]),
            fit_status=str(payload["fit_status"]),
            bundle_relative_path=payload.get("bundle_relative_path"),
            per_file_sha256=payload.get("per_file_sha256"),
            preprocessor_relative_path=payload.get("preprocessor_relative_path"),
            preprocessor_sha256=payload.get("preprocessor_sha256"),
            endpoint=str(payload["endpoint"]),
            design=str(payload["design"]),
            held_out_institution=payload.get("held_out_institution"),
            repeat=payload.get("repeat"),
            fold=payload.get("fold"),
            feature_set=str(payload["feature_set"]),
            config=dict(payload.get("config") or {}),
            development_patient_count=int(payload["development_patient_count"]),
            development_event_count=int(payload["development_event_count"]),
        )


def r_registry_record_succeeded(
    *,
    model_id: str,
    candidate: str,
    bundle_relative_path: str,
    per_file_sha256: str,
    preprocessor_relative_path: str,
    preprocessor_sha256: str,
    endpoint: str,
    design: str,
    held_out_institution: str | None,
    repeat: int | None,
    fold: int | None,
    feature_set: str,
    config: dict[str, float],
    development_patient_count: int,
    development_event_count: int,
) -> RRegistryRecord:
    return RRegistryRecord(
        model_id=model_id,
        candidate=candidate,
        fit_status="fit_succeeded",
        bundle_relative_path=bundle_relative_path,
        per_file_sha256=per_file_sha256,
        preprocessor_relative_path=preprocessor_relative_path,
        preprocessor_sha256=preprocessor_sha256,
        endpoint=endpoint,
        design=design,
        held_out_institution=held_out_institution,
        repeat=repeat,
        fold=fold,
        feature_set=feature_set,
        config=config,
        development_patient_count=development_patient_count,
        development_event_count=development_event_count,
    )


def r_registry_record_failed(
    *,
    model_id: str,
    candidate: str,
    endpoint: str,
    design: str,
    held_out_institution: str | None,
    repeat: int | None,
    fold: int | None,
    feature_set: str,
    development_patient_count: int,
    development_event_count: int,
) -> RRegistryRecord:
    return RRegistryRecord(
        model_id=model_id,
        candidate=candidate,
        fit_status="fit_failed",
        bundle_relative_path=None,
        per_file_sha256=None,
        preprocessor_relative_path=None,
        preprocessor_sha256=None,
        endpoint=endpoint,
        design=design,
        held_out_institution=held_out_institution,
        repeat=repeat,
        fold=fold,
        feature_set=feature_set,
        config={},
        development_patient_count=development_patient_count,
        development_event_count=development_event_count,
    )


def verify_r_registry_load(models_directory: Path, record: RRegistryRecord) -> Path:
    """Path-traversal-safe, hash-verified resolution of a registered R bundle.

    Returns the verified absolute bundle path; the caller then predicts from it via
    `GlmnetAdapter.predict`/`FlexsurvAdapter.predict` depending on `record.candidate` -- this
    function does not load the bundle itself, since only the R adapter knows how.
    """
    if record.bundle_relative_path is None or record.per_file_sha256 is None:
        raise Phase5AContractError("R registry record has no bundle to load")
    return _verify_safe_hashed_path(
        models_directory, record.bundle_relative_path, record.per_file_sha256, label="R bundle"
    )


def verify_r_preprocessor_path(models_directory: Path, record: RRegistryRecord) -> Path:
    """Path-traversal-safe, hash-verified resolution of a registered R candidate's preprocessor
    sidecar. Returns the verified path; the caller deserializes it (`joblib.load`)."""
    if record.preprocessor_relative_path is None or record.preprocessor_sha256 is None:
        raise Phase5AContractError("R registry record has no preprocessor sidecar to load")
    return _verify_safe_hashed_path(
        models_directory,
        record.preprocessor_relative_path,
        record.preprocessor_sha256,
        label="R preprocessor sidecar",
    )


def _verify_safe_hashed_path(
    models_directory: Path, relative_path: str, expected_sha256: str, *, label: str
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase5AContractError(f"{label} path is not traversal-safe")
    resolved_models = models_directory.resolve()
    resolved_path = (models_directory / relative).resolve()
    if resolved_path.parent != resolved_models:
        raise Phase5AContractError(f"{label} path escapes the models directory")
    if not resolved_path.is_file():
        raise Phase5AContractError(f"Registered {label} file is missing")
    digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise Phase5AContractError(f"Registered {label} hash does not match its file")
    return resolved_path
